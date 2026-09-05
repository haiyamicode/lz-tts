"""Phonemization utilities for Piper TTS inference."""

import ctypes
import json
import logging
import os
import re
from collections import Counter
from ctypes import CFUNCTYPE, POINTER, Structure, Union, c_char, c_int, c_short, c_uint, c_void_p
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from piper_phonemize import (
    phonemize_espeak as _phonemize_espeak_raw,
    phonemize_espeak_with_mapping as _phonemize_espeak_with_mapping_raw,
    phoneme_ids_espeak,
    tashkeel_run,
)

from ..multilingual_splitter import MultilingualSplitter
from ..text_norm import normalize_text as _normalize_written_text
from .heteronym import get_resolver as _get_heteronym_resolver
from .language_frontends import get_language_frontend, has_language_frontend
from .word_segmentation import icu_word_spans as _icu_word_spans


def _load_model_config(config_path):
    if isinstance(config_path, dict):
        return config_path

    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# espeak-ng 1.52+ ctypes interface for word-to-phoneme alignment
#
# Uses espeak's synth callback to get WORD and PHONEME events with exact
# character positions and stress markers. Requires espeak-ng 1.52+.
# -----------------------------------------------------------------------------

_ESPEAK_LIB = None
_ESPEAK_INITIALIZED = False
_ESPEAK_EVENTS: List[Tuple] = []
_KOREAN_G2P = None
_ESPEAK_TOKEN_CACHE: Dict[Tuple[str, str, str], List[str]] = {}


class _EspeakID(Union):
    _fields_ = [("number", c_int), ("name", ctypes.c_char_p), ("string", c_char * 8)]


class _EspeakEvent(Structure):
    _fields_ = [
        ("type", c_int), ("unique_identifier", c_uint), ("text_position", c_int),
        ("length", c_int), ("audio_position", c_int), ("sample", c_int),
        ("user_data", c_void_p), ("id", _EspeakID),
    ]


def _espeak_callback(wav, numsamples, events):
    """Collect WORD and PHONEME events from espeak synthesis."""
    i = 0
    while events[i].type != 0:  # LIST_TERMINATED
        ev = events[i]
        if ev.type == 1:  # WORD
            _ESPEAK_EVENTS.append(("WORD", ev.text_position, ev.length))
        elif ev.type == 7:  # PHONEME
            phoneme = ev.id.string.decode("utf-8", errors="replace").rstrip("\x00")
            if phoneme:
                _ESPEAK_EVENTS.append(("PHONEME", phoneme))
        i += 1
    return 0


_ESPEAK_CB_TYPE = CFUNCTYPE(c_int, POINTER(c_short), c_int, POINTER(_EspeakEvent))
_ESPEAK_CB_REF = _ESPEAK_CB_TYPE(_espeak_callback)


def _espeak_get_word_phonemes(text: str, voice: str = "en-us") -> Dict[Tuple[int, int], str]:
    """Get word-to-phoneme mapping from espeak.

    Returns dict mapping (start, end) character positions to phoneme strings.
    espeak sees the full sentence, preserving context for prosody/stress.
    Requires espeak-ng 1.52+ for stress markers in phoneme events.
    """
    result = _espeak_get_aligned_phonemes(text, voice)
    return result["words"]


def _espeak_get_aligned_phonemes(text: str, voice: str = "en-us") -> Dict:
    """Get phonemes with word alignment from a single espeak run.

    Returns:
        {
            "phonemes": list of all phonemes (preserving full output),
            "words": dict mapping (start, end) -> phoneme string for each word,
            "word_spans": dict mapping (start, end) -> (phoneme_start_idx, phoneme_end_idx),
        }
    """
    global _ESPEAK_LIB, _ESPEAK_INITIALIZED

    if _ESPEAK_LIB is None:
        for name in ["libespeak-ng.so.1", "libespeak-ng.so"]:
            try:
                _ESPEAK_LIB = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if _ESPEAK_LIB is None:
            raise RuntimeError("Could not load espeak-ng library (requires 1.52+)")

    if not _ESPEAK_INITIALIZED:
        # 0x0003 = PHONEME_EVENTS | PHONEME_IPA
        _ESPEAK_LIB.espeak_Initialize(1, 0, None, 0x0003)
        _ESPEAK_LIB.espeak_SetSynthCallback(_ESPEAK_CB_REF)
        _ESPEAK_INITIALIZED = True

    _ESPEAK_LIB.espeak_SetVoiceByName(voice.encode("utf-8"))

    _ESPEAK_EVENTS.clear()
    text_bytes = text.encode("utf-8")
    _ESPEAK_LIB.espeak_Synth(text_bytes, len(text_bytes) + 1, 0, 0, 0, 0, None, None)
    _ESPEAK_LIB.espeak_Synchronize()

    # Build full phoneme list AND per-word mapping with indices
    all_phonemes: List[str] = []
    words: Dict[Tuple[int, int], str] = {}
    word_spans: Dict[Tuple[int, int], Tuple[int, int]] = {}

    current_word: Optional[Tuple[int, int]] = None
    current_word_phonemes: List[str] = []
    current_word_start_idx: int = 0

    for ev in _ESPEAK_EVENTS:
        if ev[0] == "WORD":
            # Finish previous word
            if current_word is not None:
                words[current_word] = "".join(current_word_phonemes)
                word_spans[current_word] = (current_word_start_idx, len(all_phonemes))

            # Start new word (text_position is 1-indexed byte offset)
            start = ev[1] - 1
            end = start + ev[2]
            current_word = (start, end)
            current_word_phonemes = []
            current_word_start_idx = len(all_phonemes)

        elif ev[0] == "PHONEME":
            phoneme = ev[1]
            all_phonemes.append(phoneme)
            if current_word is not None:
                current_word_phonemes.append(phoneme)

    # Finish last word
    if current_word is not None:
        words[current_word] = "".join(current_word_phonemes)
        word_spans[current_word] = (current_word_start_idx, len(all_phonemes))

    return {
        "phonemes": all_phonemes,
        "words": words,
        "word_spans": word_spans,
    }


def _phonemize_espeak_with_reset(text: str, voice: str, data_path) -> list:
    """Wrapper around phonemize_espeak.

    The C++ extension owns eSpeak lifecycle reset for each call.
    """
    return _phonemize_espeak_raw(text, voice, data_path)


def _phonemize_espeak_with_mapping(text: str, voice: str, data_path) -> Tuple[list, List[Tuple[int, int, int, int, int]]]:
    """Phonemize text and return word-to-phoneme mapping.

    Returns:
        Tuple of (phonemes, word_mapping) where:
        - phonemes: List of sentences, each a list of phoneme strings
        - word_mapping: List of (textStart, textLength, phonemeStart, phonemeEnd, punctuationLength)
          tuples. Note: textStart is 1-indexed character position.
          punctuationLength is 0 or 1, indicating trailing punctuation (.?!,;:).
    """
    if (voice or "").lower().startswith("ja"):
        return _phonemize_japanese_with_mapping(text, voice, data_path)

    if (voice or "").lower().startswith("ko"):
        return _phonemize_korean_with_mapping(text, voice, data_path)

    return _phonemize_espeak_with_mapping_dp(text, voice, data_path)


_DEBUG = os.environ.get("PREPROCESS_DEBUG", "").lower() in ("1", "true", "yes")
if _DEBUG:
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

_LOGGER = logging.getLogger("preprocess")
_LOGGER.setLevel(logging.DEBUG if _DEBUG else logging.INFO)


def _short_text(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _short_list(xs, n: int = 24):
    return list(xs[:n]) + (["…"] if len(xs) > n else [])


# -----------------------------------------------------------------------------
# Text casing
# -----------------------------------------------------------------------------


def get_text_casing(casing: str):
    if casing == "lower":
        return str.lower
    if casing == "upper":
        return str.upper
    if casing == "casefold":
        return str.casefold
    return lambda s: s


# -----------------------------------------------------------------------------
# Punctuation normalization
# -----------------------------------------------------------------------------

_PUNCT_MAP = {
    "。": ". ",
    "，": ", ",
    "、": ", ",
    "：": ": ",
    "；": "; ",
    "？": "? ",
    "！": "! ",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
    "［": "[",
    "］": "]",
    "｛": "{",
    "｝": "}",
    "「": '"',
    "」": '"',
    "『": '"',
    "』": '"',
    "《": '"',
    "》": '"',
    "〈": '"',
    "〉": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u2018": "'",
    "\u2019": "'",
    "…": "...",
    "‥": "..",
    "—": "-",
    "–": "-",
    "―": "-",
    "〜": "~",
    "～": "~",
    "・": "-",
    "·": "-",
    "．": ". ",
    "،": ",",
    "؛": ";",
    "؟": "?",
    "।": ".",
    "॥": "..",
    "׳": "'",
    "״": '"',
    "／": "/",
    "＼": "\\",
    "％": "%",
    "＋": "+",
    "－": "-",
    "＝": "=",
    "＆": "&",
    "＃": "#",
    "＠": "@",
    "｜": "|",
}

_PUNCT_PATTERN = re.compile(
    "(" + "|".join(map(re.escape, sorted(_PUNCT_MAP.keys(), key=len, reverse=True))) + ")"
)


def _normalize_punct_and_space(text: str) -> str:
    """Normalize CJK and full-width punctuation to ASCII equivalents."""
    return _PUNCT_PATTERN.sub(lambda m: _PUNCT_MAP.get(m.group(0), m.group(0)), text)


def _katakana_to_hiragana(text: str) -> str:
    chars = []
    for ch in text:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:
            chars.append(chr(code - 0x60))
        else:
            chars.append(ch)
    return "".join(chars)


_KANA_LONG_VOWELS = {
    **dict.fromkeys("あかさたなはまやらわがざだばぱぁゃ", "あ"),
    **dict.fromkeys("いきしちにひみりぎじぢびぴぃ", "い"),
    **dict.fromkeys("うくすつぬふむゆるぐずづぶぷぅゅ", "う"),
    **dict.fromkeys("えけせてねへめれげぜでべぺぇ", "え"),
    **dict.fromkeys("おこそとのほもよろをごぞどぼぽぉょ", "お"),
}


def _japanese_long_vowel_suffix(reading: str) -> str:
    for ch in reversed(_katakana_to_hiragana(reading)):
        suffix = _KANA_LONG_VOWELS.get(ch)
        if suffix is not None:
            return suffix
    return "う"


def _append_to_last_japanese_reading(
    tokens: List[Tuple[int, int, str]], end: int, suffix: str
) -> bool:
    if not tokens:
        return False
    start, _, reading = tokens[-1]
    tokens[-1] = (start, end, reading + suffix)
    return True


_JAPANESE_READING_OVERRIDES = {
    # These occur in malformed/truncated dataset rows where ipadic has no
    # reading. Keeping them as raw CJK makes eSpeak say "Chinese letter".
    "詳": "しょう",
    "也": "や",
    "遤": "おそ",
    "衔": "しゅう",
    "衔国": "しゅうこく",
}

_JAPANESE_READING_REPLACEMENTS = (
    ("・", " "),
    ("ゔぁ", "ば"),
    ("ゔぃ", "び"),
    ("ゔぇ", "べ"),
    ("ゔぉ", "ぼ"),
    ("ゔゅ", "びゅ"),
    ("ゔ", "ぶ"),
    ("うぃ", "うい"),
    ("うぇ", "うえ"),
    ("うぉ", "うお"),
)


def _normalize_japanese_reading_for_espeak(reading: str) -> str:
    normalized = _katakana_to_hiragana(reading)
    for old, new in _JAPANESE_READING_REPLACEMENTS:
        normalized = normalized.replace(old, new)
    return normalized


def _fallback_japanese_reading(surface: str) -> str:
    override = _JAPANESE_READING_OVERRIDES.get(surface)
    if override is not None:
        return override

    try:
        import pykakasi

        converted = pykakasi.kakasi().convert(surface)
    except Exception:
        converted = []

    reading = "".join(str(item.get("hira") or "") for item in converted)
    if reading:
        return reading

    if any("\u4e00" <= ch <= "\u9fff" for ch in surface):
        return ""
    return surface


def _token_feature_value(token, index: int) -> str:
    feature = getattr(token, "feature", None)
    if feature is None:
        return ""
    try:
        value = feature[index]
    except (IndexError, TypeError):
        return ""
    return "" if value in (None, "*") else str(value)


def _locate_japanese_surface(text: str, surface: str, cursor: int) -> Tuple[int, int]:
    start = text.find(surface, cursor)
    if start < 0:
        start = cursor
    return start, start + len(surface)


def _japanese_reading_tokens(text: str) -> List[Tuple[int, int, str]]:
    import ipadic
    from fugashi import GenericTagger

    norm_text = text
    tagger = GenericTagger(ipadic.MECAB_ARGS)
    parsed_tokens = list(tagger(norm_text))
    tokens: List[Tuple[int, int, str]] = []
    cursor = 0
    token_index = 0

    while token_index < len(parsed_tokens):
        token = parsed_tokens[token_index]
        surface = token.surface
        if not surface:
            token_index += 1
            continue

        start, end = _locate_japanese_surface(norm_text, surface, cursor)

        next_surfaces = [
            parsed_tokens[token_index + offset].surface
            for offset in range(3)
            if token_index + offset < len(parsed_tokens)
        ]
        if len(next_surfaces) >= 3 and next_surfaces[:3] == ["N", "700", "系"]:
            _, end_700 = _locate_japanese_surface(norm_text, "700", end)
            _, end_series = _locate_japanese_surface(norm_text, "系", end_700)
            tokens.append((start, end_series, "えぬななひゃくけい"))
            cursor = end_series
            token_index += 3
            continue
        if len(next_surfaces) >= 2 and next_surfaces[:2] == ["700", "系"]:
            _, end_series = _locate_japanese_surface(norm_text, "系", end)
            tokens.append((start, end_series, "ななひゃくけい"))
            cursor = end_series
            token_index += 2
            continue

        cursor = end

        if surface == "ー":
            _append_to_last_japanese_reading(tokens, end, _japanese_long_vowel_suffix(tokens[-1][2] if tokens else ""))
            token_index += 1
            continue

        if surface == "々":
            if tokens:
                _append_to_last_japanese_reading(tokens, end, tokens[-1][2])
            token_index += 1
            continue

        if surface in _PUNCT_MAP:
            tokens.append((start, end, _PUNCT_MAP[surface].strip() or surface))
            token_index += 1
            continue

        pos = _token_feature_value(token, 0)
        pronunciation = _token_feature_value(token, 8)
        reading = pronunciation or _token_feature_value(token, 7) or _fallback_japanese_reading(surface)

        if pos == "助詞":
            if surface == "は":
                reading = "ワ"
            elif surface == "へ":
                reading = "エ"
            elif surface == "を":
                reading = "オ"

        if reading:
            tokens.append((start, end, _normalize_japanese_reading_for_espeak(reading)))
        token_index += 1

    return tokens


def _normalize_japanese_text(text: str) -> str:
    return " ".join(reading for _, _, reading in _japanese_reading_tokens(text))


def _has_lexical_text(text: str) -> bool:
    return any(ch.isalpha() or ch.isnumeric() for ch in text)


def _extend_anchor_text_end(text: str, end: int) -> int:
    while end < len(text) and text[end] in {".", ",", ";", ":", "!", "?"}:
        end += 1
    return end


def _flatten_sentences(sentences: list) -> List[str]:
    return [p for sentence in sentences for p in sentence]


def _trim_phoneme_edges(phonemes: List[str]) -> List[str]:
    start = 0
    end = len(phonemes)
    while start < end and phonemes[start] == " ":
        start += 1
    while end > start and phonemes[end - 1] == " ":
        end -= 1
    return phonemes[start:end]


def _phonemize_espeak_token_cached(text: str, voice: str, data_path) -> List[str]:
    key = (text, voice, str(data_path or ""))
    cached = _ESPEAK_TOKEN_CACHE.get(key)
    if cached is not None:
        return list(cached)

    phonemes = _trim_phoneme_edges(_flatten_sentences(_phonemize_espeak_with_reset(text, voice, data_path)))
    if len(_ESPEAK_TOKEN_CACHE) > 20000:
        _ESPEAK_TOKEN_CACHE.clear()
    _ESPEAK_TOKEN_CACHE[key] = list(phonemes)
    return phonemes


def _is_low_value_phoneme(phoneme: str) -> bool:
    return phoneme == " " or phoneme in {".", ",", ";", ":", "!", "?"}


def _is_phoneme_punctuation(phoneme: str) -> bool:
    return phoneme in {".", ",", ";", ":", "!", "?"}


def _phoneme_insert_cost(phoneme: str) -> float:
    return 0.02 if _is_low_value_phoneme(phoneme) else 0.2


def _phoneme_delete_cost(phoneme: str) -> float:
    return 1.0


def _phoneme_substitution_cost(left: str, right: str) -> float:
    if left == right:
        return 0.0
    if _is_low_value_phoneme(left) and _is_low_value_phoneme(right):
        return 0.05
    if left in {"ˈ", "ˌ"} or right in {"ˈ", "ˌ"}:
        return 0.35
    return 1.0


def _align_phoneme_sequences(anchor: List[str], full: List[str]) -> Tuple[float, List[Optional[int]]]:
    n = len(anchor)
    m = len(full)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + _phoneme_delete_cost(anchor[i - 1])
        back[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + _phoneme_insert_cost(full[j - 1])
        back[0][j] = "I"

    for i in range(1, n + 1):
        left = anchor[i - 1]
        for j in range(1, m + 1):
            right = full[j - 1]
            subst_cost = dp[i - 1][j - 1] + _phoneme_substitution_cost(left, right)
            delete_cost = dp[i - 1][j] + _phoneme_delete_cost(left)
            insert_cost = dp[i][j - 1] + _phoneme_insert_cost(right)

            if subst_cost <= delete_cost and subst_cost <= insert_cost:
                dp[i][j] = subst_cost
                back[i][j] = "M"
            elif insert_cost <= delete_cost:
                dp[i][j] = insert_cost
                back[i][j] = "I"
            else:
                dp[i][j] = delete_cost
                back[i][j] = "D"

    anchor_to_full: List[Optional[int]] = [None] * n
    i = n
    j = m
    while i > 0 or j > 0:
        op = back[i][j]
        if op == "M":
            anchor_to_full[i - 1] = j - 1
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            j -= 1
        else:
            break

    return dp[n][m], anchor_to_full


def _phonemize_espeak_with_mapping_dp(
    text: str,
    voice: str,
    data_path,
) -> Tuple[list, List[Tuple[int, int, int, int, int]]]:
    spans = _icu_word_spans(text, voice)
    full_phonemes = _flatten_sentences(_phonemize_espeak_with_reset(text, voice, data_path))
    if not spans:
        return [full_phonemes], [[]]
    if not full_phonemes:
        return [full_phonemes], [[]]

    anchor: List[str] = []
    anchor_tokens: List[int] = []
    active_spans: List[Tuple[int, int, str]] = []

    for start, end, piece in spans:
        anchor_end = _extend_anchor_text_end(text, end)
        token_phonemes = _phonemize_espeak_token_cached(text[start:anchor_end], voice, data_path)
        if not token_phonemes:
            continue

        token_index = len(active_spans)
        active_spans.append((start, end, piece))
        anchor.extend(token_phonemes)
        anchor_tokens.extend([token_index] * len(token_phonemes))

    if not anchor:
        return _phonemize_espeak_with_mapping_raw(text, voice, data_path)

    cost, anchor_to_full = _align_phoneme_sequences(anchor, full_phonemes)
    average_cost = cost / max(len(anchor), 1)
    if average_cost > 1.8:
        raise ValueError(f"high DP alignment cost {average_cost:.3f}")

    matched_by_token: List[List[int]] = [[] for _ in active_spans]
    for anchor_index, full_index in enumerate(anchor_to_full):
        if full_index is None:
            continue
        matched_by_token[anchor_tokens[anchor_index]].append(full_index)

    mappings: List[Tuple[int, int, int, int, int]] = []
    previous_end = 0
    pending_start: Optional[int] = None
    for token_index, (start, end, _) in enumerate(active_spans):
        if pending_start is not None:
            start = pending_start
        pending_start = None
        matched = matched_by_token[token_index]
        if not matched:
            if mappings:
                text_start, text_len, ph_start, ph_end, punct_len = mappings[-1]
                mappings[-1] = (
                    text_start,
                    max(text_len, end - (text_start - 1)),
                    ph_start,
                    ph_end,
                    punct_len,
                )
                continue
            pending_start = start
            continue

        ph_start = min(matched)
        ph_end = max(matched) + 1
        if ph_start < previous_end:
            ph_start = previous_end
        if ph_end <= ph_start:
            raise ValueError(f"non-monotonic token {text[start:end]!r}")

        while ph_start < ph_end and full_phonemes[ph_start] == " ":
            ph_start += 1
        while ph_end > ph_start and full_phonemes[ph_end - 1] == " ":
            ph_end -= 1
        if ph_end <= ph_start:
            if mappings:
                text_start, text_len, prev_ph_start, prev_ph_end, punct_len = mappings[-1]
                mappings[-1] = (
                    text_start,
                    max(text_len, end - (text_start - 1)),
                    prev_ph_start,
                    prev_ph_end,
                    punct_len,
                )
                continue
            pending_start = start
            continue

        base_ph_end = ph_end
        next_start = len(full_phonemes)
        for later in matched_by_token[token_index + 1 :]:
            if later:
                next_start = min(later)
                break
        while ph_end < next_start and ph_end < len(full_phonemes) and _is_phoneme_punctuation(full_phonemes[ph_end]):
            ph_end += 1

        punct_len = ph_end - base_ph_end
        punct_scan = ph_end
        while punct_scan > ph_start and _is_phoneme_punctuation(full_phonemes[punct_scan - 1]):
            punct_scan -= 1
        punct_len = max(punct_len, ph_end - punct_scan)
        mappings.append((start + 1, end - start, ph_start, ph_end, punct_len))
        previous_end = ph_end

    if not mappings and active_spans:
        return _phonemize_espeak_with_mapping_raw(text, voice, data_path)

    return [full_phonemes], [mappings]


def _korean_g2p():
    global _KOREAN_G2P
    if _KOREAN_G2P is None:
        from g2pk import G2p

        _KOREAN_G2P = G2p()
    return _KOREAN_G2P


def _nonspace_spans(text: str) -> List[Tuple[int, int, str]]:
    return [(match.start(), match.end(), match.group(0)) for match in re.finditer(r"\S+", text)]


def _korean_reading_tokens(text: str) -> List[Tuple[int, int, str]]:
    original_spans = _nonspace_spans(text)
    if not original_spans:
        return []

    pronounced_text = _korean_g2p()(text, descriptive=True)
    pronounced_spans = _nonspace_spans(pronounced_text)

    if len(original_spans) != len(pronounced_spans):
        return [
            (start, end, _korean_g2p()(surface, descriptive=True))
            for start, end, surface in original_spans
        ]

    return [
        (start, end, pronounced)
        for (start, end, _), (_, _, pronounced) in zip(original_spans, pronounced_spans)
    ]


def _phonemize_korean_with_mapping(
    text: str,
    voice: str,
    data_path,
) -> Tuple[list, List[Tuple[int, int, int, int, int]]]:
    sentence: List[str] = []
    mappings: List[Tuple[int, int, int, int, int]] = []

    for start, end, reading in _korean_reading_tokens(text):
        if not reading or reading.isspace():
            continue

        if sentence and sentence[-1] != " ":
            sentence.append(" ")

        ph_start = len(sentence)
        token_sents = _phonemize_espeak_with_reset(reading, voice, data_path)
        token_phonemes = [p for sent in token_sents for p in sent]
        while token_phonemes and token_phonemes[0] == " ":
            token_phonemes.pop(0)
        while token_phonemes and token_phonemes[-1] == " ":
            token_phonemes.pop()
        if not token_phonemes:
            if mappings:
                text_start, text_len, prev_ph_start, prev_ph_end, _ = mappings[-1]
                mappings[-1] = (
                    text_start,
                    max(text_len, end - (text_start - 1)),
                    prev_ph_start,
                    prev_ph_end,
                    0,
                )
            continue
        sentence.extend(token_phonemes)
        ph_end = len(sentence)

        mappings.append((start + 1, end - start, ph_start, ph_end, 0))

    return [sentence], [mappings]


def _phonemize_registered_frontend(
    text: str,
    voice: str,
    casing_fn,
) -> Optional[Tuple[list, Optional[List[List[int]]], str]]:
    frontend = get_language_frontend(voice)
    if frontend is None:
        return None

    processed_text = casing_fn(_normalize_text_for_mapping(text, voice))
    phonemes, word_spans = frontend(processed_text)
    _validate_word_spans(
        phonemes,
        word_spans,
        processed_text,
        f"voice={voice}",
    )
    return phonemes, word_spans, processed_text


def _phonemize_espeak_for_voice(text: str, voice: str, casing_fn, espeak_data) -> list:
    frontend_result = _phonemize_registered_frontend(text, voice, casing_fn)
    if frontend_result is not None:
        return frontend_result[0]

    norm_text = _normalize_text_for_voice(text, voice)
    sent_ph = _phonemize_espeak_with_reset(casing_fn(norm_text), voice, espeak_data)
    return [p for sent in sent_ph for p in sent]


def _word_spans_from_mapping(
    sentences: list,
    sent_word_mapping: list,
) -> List[List[int]]:
    spans: List[List[int]] = []
    ph_offset = 0
    for sentence, mappings in zip(sentences, sent_word_mapping):
        for text_start, text_len, ph_start, ph_end, _punct_len in mappings:
            start = int(text_start) - 1
            end = start + int(text_len)
            spans.append(
                [
                    start,
                    end,
                    ph_offset + int(ph_start),
                    ph_offset + int(ph_end),
                ]
            )
        ph_offset += len(sentence)
    return spans


def _validate_word_spans(
    phonemes: List[str],
    word_spans: Optional[List[List[int]]],
    text: str,
    context: str,
) -> None:
    if not phonemes:
        if _has_lexical_text(text):
            raise ValueError(f"{context}: lexical text produced no phonemes: {text!r}")
        return

    if not word_spans:
        if _has_lexical_text(text):
            raise ValueError(f"{context}: lexical text produced phonemes without word spans: {text!r}")
        return

    previous_ph_end = -1
    previous_text_end = -1
    for raw in word_spans:
        if raw is None or len(raw) < 4:
            raise ValueError(f"{context}: malformed word span: {raw!r}")
        text_start, text_end, ph_start, ph_end = [int(value) for value in raw[:4]]
        if not (0 <= text_start < text_end <= len(text)):
            raise ValueError(
                f"{context}: word span text bounds out of range: "
                f"span={raw!r}, text_len={len(text)}, text={text!r}"
            )
        if not (0 <= ph_start < ph_end <= len(phonemes)):
            raise ValueError(
                f"{context}: word span phoneme bounds out of range: "
                f"span={raw!r}, phoneme_len={len(phonemes)}, text={text!r}"
            )
        if ph_start < previous_ph_end:
            raise ValueError(f"{context}: non-monotonic phoneme span: span={raw!r}")
        if text_start < previous_text_end:
            raise ValueError(f"{context}: non-monotonic text span: span={raw!r}")
        previous_ph_end = ph_end
        previous_text_end = text_end


def _phonemize_espeak_for_voice_with_spans(
    text: str,
    voice: str,
    casing_fn,
    espeak_data,
    *,
    include_word_spans: bool = True,
) -> Tuple[list, Optional[List[List[int]]], str]:
    frontend_result = _phonemize_registered_frontend(text, voice, casing_fn)
    if frontend_result is not None:
        if not include_word_spans:
            return frontend_result[0], None, frontend_result[2]
        return frontend_result

    norm_text = _normalize_text_for_mapping(text, voice)
    processed_text = casing_fn(norm_text)
    if not include_word_spans:
        phonemes = _flatten_sentences(
            _phonemize_espeak_with_reset(processed_text, voice, espeak_data)
        )
        return phonemes, None, processed_text

    sent_ph, sent_word_mapping = _phonemize_espeak_with_mapping(
        processed_text, voice, espeak_data
    )
    phonemes = _flatten_sentences(sent_ph)
    word_spans = _word_spans_from_mapping(sent_ph, sent_word_mapping)
    _validate_word_spans(
        phonemes,
        word_spans,
        processed_text,
        f"voice={voice}",
    )
    return phonemes, word_spans or None, processed_text


def _phonemize_japanese_with_mapping(
    text: str,
    voice: str,
    data_path,
) -> Tuple[list, List[Tuple[int, int, int, int, int]]]:
    sentence: List[str] = []
    mappings: List[Tuple[int, int, int, int, int]] = []

    for start, end, reading in _japanese_reading_tokens(text):
        if not reading or reading.isspace():
            continue

        if reading in {".", "?", "!", ",", ":", ";"}:
            if mappings:
                punct_start = len(sentence)
                sentence.append(reading)
                if reading in {",", ":", ";"}:
                    sentence.append(" ")

                text_start, text_len, ph_start, _, _ = mappings[-1]
                mappings[-1] = (
                    text_start,
                    max(text_len, end - (text_start - 1)),
                    ph_start,
                    punct_start + 1,
                    0,
                )
            else:
                sentence.append(reading)
                if reading in {",", ":", ";"}:
                    sentence.append(" ")
            continue

        if sentence and sentence[-1] != " ":
            sentence.append(" ")

        ph_start = len(sentence)
        token_sents = _phonemize_espeak_with_reset(reading, voice, data_path)
        token_phonemes = [p for sent in token_sents for p in sent]
        while token_phonemes and token_phonemes[0] == " ":
            token_phonemes.pop(0)
        while token_phonemes and token_phonemes[-1] == " ":
            token_phonemes.pop()
        if not token_phonemes:
            if mappings:
                text_start, text_len, prev_ph_start, prev_ph_end, _ = mappings[-1]
                mappings[-1] = (
                    text_start,
                    max(text_len, end - (text_start - 1)),
                    prev_ph_start,
                    prev_ph_end,
                    0,
                )
            continue
        sentence.extend(token_phonemes)
        ph_end = len(sentence)

        mappings.append((start + 1, end - start, ph_start, ph_end, 0))

    return [sentence], [mappings]


# -----------------------------------------------------------------------------
# Language/voice mapping
# -----------------------------------------------------------------------------


def _map_cld2_to_espeak(lang_code: str, primary_voice: str = "en-us") -> str:
    """Map language code to an espeak-ng voice."""
    if not lang_code:
        return "en-us"

    code = lang_code.strip().lower().replace("_", "-")
    if not code or code in ("und", "undetermined"):
        return primary_voice
    base = code.split("-", 1)[0] if code else "en"

    if code in ("en-us", "en-us+f3", "en-us+f4"):
        return code

    if code in ("en-gb", "en-uk"):
        return "en-gb-x-rp"

    if code.startswith("en-gb-"):
        return code

    if code == "en":
        return primary_voice

    if code == "pt-br":
        return "pt-BR"

    if base == "yue" or code in ("zh-hk", "zh-yue"):
        return "yue"

    if base in ("zh", "cmn"):
        return "cmn-latn-pinyin"

    if base in ("fil", "jv", "su"):
        return "id"

    if base == "so":
        return "om"

    if has_language_frontend(base):
        return base

    if base == "zu":
        return "tn"

    if base == "gl":
        return "es"

    return base


def _supports_neural_heteronyms(voice: str) -> bool:
    return (voice or "").lower().startswith("en")


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------


def _normalize_text_for_voice(text: str, voice: str) -> str:
    """Apply language-specific normalization before espeak phonemization."""
    norm_text = _normalize_punct_and_space(text)
    v = (voice or "").lower()
    norm_text = _normalize_written_text(norm_text, v)

    if v.startswith("ja"):
        norm_text = _normalize_japanese_text(norm_text)

    elif v.startswith("ar"):
        norm_text = tashkeel_run(norm_text)

    return norm_text


def _normalize_text_for_mapping(text: str, voice: str) -> str:
    """Normalize text before phonemization paths that build word mappings.

    This is text-only normalization. It intentionally does not call G2P,
    IPA conversion, or phone/tone generation.
    """
    v = (voice or "").lower()
    norm_text = _normalize_punct_and_space(text)
    norm_text = _normalize_written_text(norm_text, v)

    if v.startswith("ar"):
        norm_text = tashkeel_run(norm_text)

    return norm_text


# -----------------------------------------------------------------------------
# Phonemization
# -----------------------------------------------------------------------------


def _phonemize_multilingual(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    primary_voice: str = "en-us",
) -> list:
    """Phonemize mixed-language text using MultilingualSplitter."""
    phonemes, _word_spans, _semantic_text = _phonemize_multilingual_with_spans(
        text,
        casing_fn,
        espeak_data,
        primary_voice,
        neural=False,
    )
    return phonemes


def _phonemize_multilingual_with_spans(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    primary_voice: str = "en-us",
    neural: bool = False,
    include_word_spans: bool = True,
) -> Tuple[list, Optional[List[List[int]]], str]:
    """Phonemize mixed-language text and preserve word-to-phoneme spans."""
    splitter = MultilingualSplitter()
    result = splitter.split(text)
    segments = result.segments
    main_lang = result.effective_main_language()

    _LOGGER.debug(
        "multilingual_splitter segments: %s (main=%s)",
        [(seg.language, _short_text(seg.text, 40)) for seg in segments],
        main_lang,
    )

    phonemes: list = []
    word_spans: List[List[int]] = []
    semantic_parts: List[str] = []
    for idx, seg in enumerate(segments):
        span_text = seg.text
        lang = seg.language if seg.language and seg.language != "und" else main_lang

        if not span_text.strip():
            continue

        voice = _map_cld2_to_espeak(lang or "en", primary_voice)
        norm_span_text = _normalize_text_for_voice(span_text, voice)

        _LOGGER.debug("span[%s]: lang=%s voice=%s text='%s'", idx, lang, voice, _short_text(norm_span_text, 80))

        if neural:
            span_phonemes, span_word_spans, semantic_text = _phonemize_neural_with_spans(
                span_text,
                casing_fn,
                espeak_data,
                voice,
            )
            if not include_word_spans:
                span_word_spans = None
        else:
            span_phonemes, span_word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
                span_text,
                voice,
                casing_fn,
                espeak_data,
                include_word_spans=include_word_spans,
            )

        _LOGGER.debug("span[%s] phonemes=%s", idx, _short_list(span_phonemes, 32))

        if not span_phonemes:
            continue
        if include_word_spans:
            _validate_word_spans(
                span_phonemes,
                span_word_spans,
                semantic_text,
                f"multilingual span={idx} voice={voice}",
            )

        if phonemes and span_phonemes:
            phonemes.append(" ")
        if semantic_parts and semantic_text:
            semantic_parts.append(" ")

        phoneme_offset = len(phonemes)
        semantic_offset = sum(len(part) for part in semantic_parts)

        if span_word_spans:
            for text_start, text_end, ph_start, ph_end in span_word_spans:
                word_spans.append(
                    [
                        semantic_offset + int(text_start),
                        semantic_offset + int(text_end),
                        phoneme_offset + int(ph_start),
                        phoneme_offset + int(ph_end),
                    ]
                )
        semantic_parts.append(semantic_text)
        phonemes.extend(span_phonemes)

    semantic_text = "".join(semantic_parts) if semantic_parts else text
    if include_word_spans:
        _validate_word_spans(
            phonemes,
            word_spans,
            semantic_text,
            "multilingual",
        )
    return phonemes, word_spans or None, semantic_text


def _phonemize_neural_with_spans(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    voice: str = "en-us",
    resolved_heteronyms: Optional[List[Tuple[str, int, int, str]]] = None,
) -> Tuple[list, Optional[List[List[int]]], str]:
    """Phonemize text using neural heteronym disambiguation with word spans.

    Strategy:
    1. Get phoneme output with word-to-phoneme mapping from C++ (using espeak_Synth callback)
    2. Match heteronym text positions to the mapping
    3. Replace the corresponding phoneme segments with BERT pronunciations
    4. Rebuild word spans against the rewritten phoneme list

    Uses reliable C++ word mapping instead of space-based segmentation.
    """
    norm_text = _normalize_text_for_voice(text, voice)
    processed_text = casing_fn(norm_text)

    # Only apply neural disambiguation for English voices
    # The heteronym model is trained on English
    if not voice.startswith("en"):
        return _phonemize_espeak_for_voice_with_spans(text, voice, casing_fn, espeak_data)

    # Find all heteronyms with their positions and correct pronunciations
    if resolved_heteronyms is None:
        resolver = _get_heteronym_resolver()
        heteronyms = resolver.resolve_all(processed_text)
    else:
        heteronyms = resolved_heteronyms

    if not heteronyms:
        return _phonemize_espeak_for_voice_with_spans(text, voice, casing_fn, espeak_data)

    _LOGGER.debug(
        "neural: found %d heteronyms: %s",
        len(heteronyms),
        [(h[0], h[1], h[2], h[3]) for h in heteronyms],
    )

    # Get phonemes WITH word-to-phoneme mapping from C++ (per sentence).
    sent_ph, sent_word_mapping = _phonemize_espeak_with_mapping(processed_text, voice, espeak_data)

    _LOGGER.debug("neural: %d sentences", len(sent_ph))
    for sent_idx, (sentence, mappings) in enumerate(zip(sent_ph, sent_word_mapping)):
        _LOGGER.debug("neural: sentence %d phonemes=%s", sent_idx, _short_list(sentence, 32))
        _LOGGER.debug(
            "neural: sentence %d word_mapping: %s",
            sent_idx,
            [(ts, tl, ps, pe, pl, processed_text[ts-1:ts-1+tl] if ts > 0 else "") for ts, tl, ps, pe, pl in mappings],
        )

    # Build replacement map: (sent_idx, word_idx) -> new_phonemes
    # Match heteronym positions to word mapping entries
    replacements: Dict[Tuple[int, int], List[str]] = {}

    for word, h_start, h_end, correct_ipa in heteronyms:
        # Find which word mapping entry this heteronym corresponds to
        # Note: word_mapping textStart is 1-indexed character position
        # heteronym positions are 0-indexed character positions
        # For ASCII (English), byte position == character position
        matched = None

        for sent_idx, mappings in enumerate(sent_word_mapping):
            for word_idx, (text_start, text_len, ph_start, ph_end, punct_len) in enumerate(mappings):
                # Convert 1-indexed character position to 0-indexed
                map_start = text_start - 1
                map_end = map_start + text_len

                # Check if heteronym overlaps with this word
                if map_start <= h_start < map_end or map_start < h_end <= map_end:
                    matched = (sent_idx, word_idx)
                    break
                # Exact match
                if map_start == h_start and map_end == h_end:
                    matched = (sent_idx, word_idx)
                    break
            if matched:
                break

        if matched is not None:
            replacements[matched] = list(correct_ipa)
            _LOGGER.debug(
                "neural: heteronym '%s' at text[%d:%d] -> sent=%d, word=%d, replacement=%s",
                word, h_start, h_end, matched[0], matched[1], correct_ipa,
            )
        else:
            _LOGGER.warning(
                "neural: couldn't map heteronym '%s' at [%d:%d] to word mapping",
                word, h_start, h_end,
            )

    # Build result by processing each sentence separately using LOCAL indices.
    # Word spans are rebuilt in the same pass so later spans naturally shift when
    # a heteronym replacement has a different phoneme length.
    result: List[str] = []
    word_spans: List[List[int]] = []

    for sent_idx, (sentence, mappings) in enumerate(zip(sent_ph, sent_word_mapping)):
        last_end = 0

        for word_idx, (text_start, text_len, ph_start, ph_end, punct_len) in enumerate(mappings):
            # Add any phonemes between the last word and this word (spaces, etc.)
            if ph_start > last_end:
                result.extend(sentence[last_end:ph_start])

            new_ph_start = len(result)
            if (sent_idx, word_idx) in replacements:
                # Use BERT pronunciation, but preserve trailing punctuation
                # punct_len tells us exactly how many trailing phonemes are punctuation
                word_ph_end = ph_end - punct_len
                trailing_punct = sentence[word_ph_end:ph_end]

                result.extend(replacements[(sent_idx, word_idx)])
                result.extend(trailing_punct)

                _LOGGER.debug(
                    "neural: sent %d word %d replaced: %s -> %s + %s",
                    sent_idx, word_idx, sentence[ph_start:ph_end],
                    replacements[(sent_idx, word_idx)], trailing_punct,
                )
            else:
                # Keep original
                result.extend(sentence[ph_start:ph_end])

            new_ph_end = len(result)
            if new_ph_end > new_ph_start:
                start = int(text_start) - 1
                word_spans.append([start, start + int(text_len), new_ph_start, new_ph_end])

            last_end = ph_end

        # Add any remaining phonemes after the last word in this sentence
        if last_end < len(sentence):
            result.extend(sentence[last_end:])

    _validate_word_spans(
        result,
        word_spans,
        processed_text,
        f"neural voice={voice}",
    )
    return result, word_spans or None, processed_text


def _phonemize_neural(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    voice: str = "en-us",
) -> list:
    """Phonemize text using neural heteronym disambiguation."""
    phonemes, _word_spans, _processed_text = _phonemize_neural_with_spans(
        text,
        casing_fn,
        espeak_data,
        voice,
    )
    return phonemes


def _segment_phonemes_by_space(phonemes: List[str]) -> List[Tuple[int, int]]:
    """Segment a phoneme list by spaces, returning (start, end) indices for each word."""
    segments: List[Tuple[int, int]] = []
    current_start = 0

    for i, p in enumerate(phonemes):
        if p == " ":
            if i > current_start:
                segments.append((current_start, i))
            current_start = i + 1

    # Don't forget the last segment
    if current_start < len(phonemes):
        segments.append((current_start, len(phonemes)))

    return segments


def phonemize_text_for_infer(
    text: str,
    config_path: "Path | str",
    espeak_data: Optional[str] = None,
    neural: bool = False,
    include_word_spans: bool = True,
) -> Dict[str, List[str]]:
    """Phonemize text for inference.

    Args:
        text: Input text to phonemize.
        config_path: Path to model config JSON.
        espeak_data: Optional path to espeak-ng data.
        neural: If True, use neural heteronym disambiguation.
        include_word_spans: Build source-word to phoneme mappings. Disable this
            for callers that only consume phoneme IDs.

    Returns a dict with 'phonemes' and 'phoneme_ids'.
    """
    cfg = _load_model_config(config_path)

    lang_code = (cfg.get("language") or {}).get("code")
    es_conf = cfg.get("espeak") or {}
    es_voice = es_conf.get("voice")
    primary = es_conf.get("primary") or "en-us"

    casing = get_text_casing("ignore")
    is_multi = (lang_code == "multilingual") or (es_voice == "multilingual")

    _LOGGER.debug("infer: is_multi=%s voice=%s primary=%s neural=%s", is_multi, es_voice or lang_code, primary, neural)

    if neural and not is_multi and _supports_neural_heteronyms(es_voice or lang_code or primary):
        # Neural heteronym disambiguation mode
        voice = es_voice or lang_code or primary
        voice = _map_cld2_to_espeak(voice, primary)
        phonemes, word_spans, semantic_text = _phonemize_neural_with_spans(
            text, casing, espeak_data, voice
        )
        if not include_word_spans:
            word_spans = None
    elif is_multi:
        phonemes, word_spans, semantic_text = _phonemize_multilingual_with_spans(
            text,
            casing,
            espeak_data,
            primary,
            neural=neural,
            include_word_spans=include_word_spans,
        )
    else:
        voice = es_voice or lang_code or primary
        voice = _map_cld2_to_espeak(voice, primary)
        phonemes, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
            text,
            voice,
            casing,
            espeak_data,
            include_word_spans=include_word_spans,
        )
        _LOGGER.debug("infer: voice=%s text='%s'", voice, _short_text(semantic_text, 120))

    _LOGGER.debug("infer: phonemes=%s", _short_list(phonemes, 48))
    ids = phoneme_ids_espeak(phonemes)
    return {
        "phonemes": phonemes,
        "phoneme_ids": ids,
        "text": semantic_text,
        "word_spans": word_spans,
    }


def phonemize_spans_with_speakers(
    text: str,
    config_path: "Path | str",
    espeak_data: Optional[str] = None,
    neural: bool = False,
) -> List[Dict[str, object]]:
    """Phonemize text with language-based speaker assignment.

    Args:
        text: Input text to phonemize.
        config_path: Path to model config JSON.
        espeak_data: Optional path to espeak-ng data.
        neural: If True, use neural heteronym disambiguation.

    Returns a list of spans:
        [{"phonemes": [...], "phoneme_ids": [...], "speaker_id": int, "text": str}, ...]
    """
    cfg = _load_model_config(config_path)

    es_conf = cfg.get("espeak") or {}
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}
    supported_langs = {
        (label or "").lower().split("-")[0]
        for label in set(spk_id_map.keys()) | set(lang_spk_map.keys()) | set(lang_spk_map.values())
        if label
    }
    splitter = MultilingualSplitter(languages=sorted(supported_langs) if supported_langs else None)

    casing = get_text_casing("ignore")
    split_result = splitter.split(text)
    segments = split_result.segments
    main_lang = split_result.effective_main_language()
    if main_lang not in supported_langs:
        primary_lang = primary.split("-")[0]
        main_lang = primary_lang if primary_lang in supported_langs else next(iter(supported_langs), primary_lang)

    _LOGGER.debug(
        "infer-multispan: segments=%s (main=%s)",
        [(seg.language, _short_text(seg.text, 60)) for seg in segments],
        main_lang,
    )

    def _find_speaker_for_lang(lang_code: str) -> Optional[tuple]:
        voice = _map_cld2_to_espeak(lang_code or "en", primary)
        base = "en" if voice.startswith("en") else voice

        spk_label = lang_spk_map.get(base)
        if spk_label and spk_label in spk_id_map:
            return (spk_label, spk_id_map[spk_label], voice)

        candidates = [base]
        if base == "cmn" or base.startswith("cmn-latn"):
            candidates = ["zh", "cmn-latn-pinyin"]
        elif base.startswith("en"):
            candidates = ["en", "en-us", "en-gb"]

        for cand in candidates:
            if cand in spk_id_map:
                return (cand, spk_id_map[cand], voice)

        return None

    results: List[Dict[str, object]] = []
    for seg in segments:
        span_text = seg.text
        if not span_text.strip():
            continue

        lang = (seg.language if seg.language and seg.language != "und" else main_lang).lower()
        if supported_langs and lang.split("-")[0] not in supported_langs:
            lang = main_lang or primary.split("-")[0]

        speaker_info = _find_speaker_for_lang(lang)
        if speaker_info is None and main_lang and main_lang != lang:
            _LOGGER.debug("infer-multispan: lang=%s not available, trying main_lang=%s", lang, main_lang)
            speaker_info = _find_speaker_for_lang(main_lang)
        if speaker_info is None:
            _LOGGER.debug("infer-multispan: main_lang=%s not available, trying primary=%s", main_lang, primary)
            speaker_info = _find_speaker_for_lang(primary.split("-")[0])
        if speaker_info is None:
            _LOGGER.debug("infer-multispan: no speaker found for lang=%s, using speaker 0", lang)
            spk_label = list(spk_id_map.keys())[0] if spk_id_map else "en"
            spk_id = 0
            voice = _map_cld2_to_espeak(spk_label or "en", primary)
        else:
            spk_label, spk_id, voice = speaker_info

        _LOGGER.debug("infer-multispan: lang=%s voice=%s spk_label=%s spk_id=%s neural=%s", lang, voice, spk_label, spk_id, neural)

        if neural and _supports_neural_heteronyms(voice):
            ph, word_spans, semantic_text = _phonemize_neural_with_spans(
                span_text, casing, espeak_data, voice
            )
        else:
            ph, word_spans, semantic_text = _phonemize_espeak_for_voice_with_spans(
                span_text, voice, casing, espeak_data
            )

        _LOGGER.debug("infer-multispan: phonemes=%s", _short_list(ph, 40))

        ids = phoneme_ids_espeak(ph)
        results.append(
            {
                "phonemes": ph,
                "phoneme_ids": ids,
                "speaker_id": int(spk_id),
                "text": semantic_text,
                "word_spans": word_spans,
                "language": lang,
                "voice": voice,
                "source_text": span_text,
                "source_start": seg.start,
                "source_end": seg.end,
            }
        )

    return results


def _forced_speaker_context(config_path: "Path | str", speaker_label: str):
    cfg = _load_model_config(config_path)

    es_conf = cfg.get("espeak") or {}
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}

    label = speaker_label
    rev = {v: k for k, v in lang_spk_map.items()} if lang_spk_map else {}
    base = rev.get(label, label)
    voice = _map_cld2_to_espeak(base, primary)
    spk_id = int(spk_id_map.get(label, 0))
    return label, base, voice, primary, spk_id, get_text_casing("ignore")


def _phonemize_forced_speaker_text(
    text: str,
    casing_fn,
    espeak_data: Optional[str],
    voice: str,
    neural: bool,
    resolved_heteronyms: Optional[List[Tuple[str, int, int, str]]] = None,
) -> Tuple[list, Optional[List[List[int]]], str]:
    if neural and _supports_neural_heteronyms(voice):
        return _phonemize_neural_with_spans(
            text,
            casing_fn,
            espeak_data,
            voice,
            resolved_heteronyms=resolved_heteronyms,
        )
    return _phonemize_espeak_for_voice_with_spans(text, voice, casing_fn, espeak_data)


def phonemize_text_for_speaker(
    text: str,
    config_path: "Path | str",
    speaker_label: str,
    espeak_data: Optional[str] = None,
    neural: bool = False,
) -> Dict[str, object]:
    """Phonemize text for a specific speaker (skip language detection).

    Args:
        text: Input text to phonemize.
        config_path: Path to model config JSON.
        speaker_label: Speaker label to use.
        espeak_data: Optional path to espeak-ng data.
        neural: If True, use neural heteronym disambiguation.

    Returns: {"phonemes": [...], "phoneme_ids": [...], "speaker_id": int, "text": str}
    """
    label, base, voice, primary, spk_id, casing = _forced_speaker_context(config_path, speaker_label)
    _LOGGER.debug("infer-forced: label=%s base=%s -> voice=%s primary=%s neural=%s", label, base, voice, primary, neural)

    use_neural = neural and _supports_neural_heteronyms(voice)
    phonemes, word_spans, semantic_text = _phonemize_forced_speaker_text(
        text,
        casing,
        espeak_data,
        voice,
        use_neural,
    )
    if not use_neural:
        _LOGGER.debug("infer-forced: text='%s'", _short_text(semantic_text, 120))

    _LOGGER.debug("infer-forced: phonemes=%s", _short_list(phonemes, 48))

    ids = phoneme_ids_espeak(phonemes)
    return {
        "phonemes": phonemes,
        "phoneme_ids": ids,
        "speaker_id": int(spk_id),
        "text": semantic_text,
        "word_spans": word_spans,
        "voice": voice,
        "source_text": text,
    }


def phonemize_texts_for_speaker(
    texts: List[str],
    config_path: "Path | str",
    speaker_label: str,
    espeak_data: Optional[str] = None,
    neural: bool = False,
) -> List[Dict[str, object]]:
    """Phonemize a batch of texts for one forced speaker.

    This keeps the output contract identical to ``phonemize_text_for_speaker``
    while batching neural heteronym resolution for English voices.
    """
    _label, _base, voice, _primary, spk_id, casing = _forced_speaker_context(config_path, speaker_label)
    use_neural = neural and _supports_neural_heteronyms(voice)

    resolved_by_text: List[Optional[List[Tuple[str, int, int, str]]]] = [None for _ in texts]
    if use_neural:
        resolver = _get_heteronym_resolver()
        processed_texts = [
            casing(_normalize_text_for_voice(text, voice))
            for text in texts
        ]
        resolved_by_text = resolver.resolve_all_many(processed_texts)

    outputs: List[Dict[str, object]] = []
    for text, resolved_heteronyms in zip(texts, resolved_by_text):
        phonemes, word_spans, semantic_text = _phonemize_forced_speaker_text(
            text,
            casing,
            espeak_data,
            voice,
            use_neural,
            resolved_heteronyms=resolved_heteronyms,
        )
        outputs.append(
            {
                "phonemes": phonemes,
                "phoneme_ids": phoneme_ids_espeak(phonemes),
                "speaker_id": spk_id,
                "text": semantic_text,
                "word_spans": word_spans,
                "voice": voice,
                "source_text": text,
            }
        )
    return outputs


def _normalized_override_bounds(
    source_text: str,
    source_start: int,
    source_end: int,
    voice: str,
    normalized_text: str,
) -> Tuple[int, int]:
    """Map source offsets through the same text normalization as phonemization.

    The normalizers can expand text before the frontend creates word mappings
    (for example ``Dr.`` to ``doctor``).  Sentinels let us carry an exact source
    boundary through that transformation without guessing from string
    similarity or assuming normalization preserves string length.  Most
    normalizers preserve Unicode private-use characters; restrictive
    normalizers such as Chinese require markers from their accepted script.
    """
    private_use: list[str] = []
    for codepoint in range(0xE000, 0xF900):
        candidate = chr(codepoint)
        if candidate not in source_text:
            private_use.append(candidate)
            if len(private_use) == 2:
                break

    candidates: list[Tuple[str, str]] = []
    if len(private_use) == 2:
        candidates.append((private_use[0], private_use[1]))
    candidates.extend(
        pair
        for pair in (("龘", "龖"), ("齉", "爨"))
        if pair[0] not in source_text and pair[1] not in source_text
    )

    for opening, closing in candidates:
        marked = (
            source_text[:source_start]
            + opening
            + source_text[source_start:source_end]
            + closing
            + source_text[source_end:]
        )
        normalized_marked = _normalize_text_for_mapping(marked, voice)
        if normalized_marked.count(opening) != 1 or normalized_marked.count(closing) != 1:
            continue
        normalized_start = normalized_marked.index(opening)
        normalized_end_with_marker = normalized_marked.index(closing)
        # At least one character must survive between the sentinels; a
        # collapsed span would map to a degenerate (n, n) override.
        if normalized_start + 1 >= normalized_end_with_marker:
            continue
        unmarked = normalized_marked.replace(opening, "").replace(closing, "")
        if unmarked == normalized_text:
            return normalized_start, normalized_end_with_marker - 1

    raise ValueError("Text normalization did not preserve IPA override boundaries")


def _phoneme_edit_distance(left: List[str], right: List[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _align_partial_override_phonemes(
    mapped_phonemes: List[str],
    selected_phonemes: List[str],
    *,
    selected_text_start: int,
    selected_text_end: int,
    mapped_text_length: int,
) -> Tuple[int, int]:
    """Locate a wrapped text fragment inside its frontend phoneme unit."""
    phoneme_count = len(mapped_phonemes)
    if not phoneme_count:
        return 0, 0

    text_length = max(1, mapped_text_length)
    expected_start = phoneme_count * selected_text_start / text_length
    expected_end = phoneme_count * selected_text_end / text_length
    best: Optional[Tuple[float, int, int]] = None
    for start in range(phoneme_count + 1):
        for end in range(start, phoneme_count + 1):
            candidate = mapped_phonemes[start:end]
            edit_scale = max(1, len(selected_phonemes), len(candidate))
            edit_score = _phoneme_edit_distance(selected_phonemes, candidate) / edit_scale
            position_score = (
                abs(start - expected_start) + abs(end - expected_end)
            ) / phoneme_count
            score = edit_score + (0.15 * position_score)
            ranked = (score, start, end)
            if best is None or ranked < best:
                best = ranked
    assert best is not None
    return best[1], best[2]


def apply_ipa_overrides(
    spans: List[Dict[str, object]],
    overrides: List[Tuple[int, int, str]],
    partial_word_phonemizer=None,
) -> List[Dict[str, object]]:
    """Replace mapped phonemes for exact source-text spans with IPA.

    ``phonemize_spans_with_speakers`` records each language span's source
    offsets, while a forced-speaker result represents the complete source
    string. Partial frontend words are rebuilt from their untouched fragments
    when ``partial_word_phonemizer`` is provided.
    """
    prepared = [dict(span) for span in spans]
    ordered = sorted(overrides, key=lambda item: (item[0], item[1]))
    for index, (start, end, ipa) in enumerate(ordered):
        if not 0 <= start < end or not ipa:
            raise ValueError(f"Invalid IPA override [{start}, {end})")
        if index and start < ordered[index - 1][1]:
            raise ValueError("IPA pronunciation spans must not overlap")

    assigned: set[int] = set()
    for span_index, span in enumerate(prepared):
        source_start = int(span.get("source_start", 0))
        source_end = int(span.get("source_end", source_start + len(str(span.get("text", "")))))
        source_text = str(span.get("source_text", span.get("text", "")))
        normalized_text = str(span.get("text", ""))
        voice = str(span.get("voice", span.get("language", "en-us")))
        local_overrides = [
            (
                override_index,
                *_normalized_override_bounds(
                    source_text,
                    start - source_start,
                    end - source_start,
                    voice,
                    normalized_text,
                ),
                ipa,
            )
            for override_index, (start, end, ipa) in enumerate(ordered)
            if source_start <= start and end <= source_end
        ]
        if not local_overrides:
            continue

        phonemes = list(span.get("phonemes") or [])
        mappings = [list(map(int, mapping)) for mapping in (span.get("word_spans") or [])]
        if not mappings:
            raise ValueError("Sparrow frontend returned no text-to-phoneme mapping for an IPA override")

        for override_index, local_start, local_end, ipa in reversed(local_overrides):
            selected_indices = [
                mapping_index
                for mapping_index, mapping in enumerate(mappings)
                if mapping[1] > local_start and mapping[0] < local_end
            ]
            if not selected_indices:
                raise ValueError(
                    f"IPA override [{local_start}, {local_end}) does not cover a pronounced text unit"
                )
            first_index, last_index = selected_indices[0], selected_indices[-1]
            first, last = mappings[first_index], mappings[last_index]
            replacement_start = local_start
            replacement_end = local_end
            replacement = list(ipa)
            if first[0] < local_start or last[1] > local_end:
                if partial_word_phonemizer is None:
                    raise ValueError(
                        "SSML <phoneme> boundary cuts through a frontend word and "
                        "no partial-word phonemizer was provided"
                    )
                mapped_text = normalized_text[first[0] : last[1]]
                selected_text = normalized_text[local_start:local_end]
                mapped_phonemes = phonemes[first[2] : last[3]]
                selected_phonemes = list(partial_word_phonemizer(selected_text, voice))
                selected_start, selected_end = _align_partial_override_phonemes(
                    mapped_phonemes,
                    selected_phonemes,
                    selected_text_start=local_start - first[0],
                    selected_text_end=local_end - first[0],
                    mapped_text_length=len(mapped_text),
                )
                replacement = (
                    mapped_phonemes[:selected_start]
                    + replacement
                    + mapped_phonemes[selected_end:]
                )
                replacement_start = first[0]
                replacement_end = last[1]

            if not replacement:
                raise ValueError(
                    "IPA override and its preserved frontend fragments produced no phonemes"
                )

            phoneme_start, phoneme_end = first[2], last[3]
            missing: Counter[str] = Counter()
            phoneme_ids_espeak(replacement, missing_phonemes=missing)
            if missing:
                unsupported = ", ".join(repr(value) for value in sorted(missing))
                raise ValueError(f"IPA override contains unsupported Sparrow phonemes: {unsupported}")

            phonemes[phoneme_start:phoneme_end] = replacement
            shift = len(replacement) - (phoneme_end - phoneme_start)
            replacement_mapping = [
                replacement_start,
                replacement_end,
                phoneme_start,
                phoneme_start + len(replacement),
            ]
            mappings[first_index : last_index + 1] = [replacement_mapping]
            for mapping in mappings[first_index + 1 :]:
                mapping[2] += shift
                mapping[3] += shift
            assigned.add(override_index)

        span["phonemes"] = phonemes
        span["phoneme_ids"] = phoneme_ids_espeak(phonemes)
        span["word_spans"] = mappings
        prepared[span_index] = span

    missing_overrides = sorted(set(range(len(ordered))) - assigned)
    if missing_overrides:
        start, end, _ipa = ordered[missing_overrides[0]]
        raise ValueError(f"IPA override [{start}, {end}) crosses a Sparrow language span")
    return prepared
