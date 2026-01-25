"""Phonemization utilities for Piper TTS inference."""

import ctypes
import json
import logging
import os
import re
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
from .heteronym import get_resolver as _get_heteronym_resolver


# -----------------------------------------------------------------------------
# espeak-ng 1.52+ ctypes interface for word-to-phoneme alignment
#
# Uses espeak's synth callback to get WORD and PHONEME events with exact
# character positions and stress markers. Requires espeak-ng 1.52+.
# -----------------------------------------------------------------------------

_ESPEAK_LIB = None
_ESPEAK_INITIALIZED = False
_ESPEAK_EVENTS: List[Tuple] = []


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
    """Wrapper around phonemize_espeak that resets espeak state after each call.

    espeak-ng has a bug where processing text with consecutive periods (..)
    corrupts internal state, causing "dot" to be prepended to the NEXT call's output.
    Switching voices resets this state.
    """
    result = _phonemize_espeak_raw(text, voice, data_path)
    # Reset espeak state by switching to a different voice
    # This prevents the ".." bug from affecting subsequent calls
    _phonemize_espeak_raw("", "de", data_path)
    return result


def _phonemize_espeak_with_mapping(text: str, voice: str, data_path) -> Tuple[list, List[Tuple[int, int, int, int]]]:
    """Phonemize text and return word-to-phoneme mapping.

    Returns:
        Tuple of (phonemes, word_mapping) where:
        - phonemes: List of sentences, each a list of phoneme strings
        - word_mapping: List of (textStart, textLength, phonemeStart, phonemeEnd)
          tuples. Note: textStart is 1-indexed byte position from espeak.
    """
    phonemes, mapping = _phonemize_espeak_with_mapping_raw(text, voice, data_path)
    # Reset espeak state
    _phonemize_espeak_raw("", "de", data_path)
    return phonemes, mapping

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
    """: '"',
    """: '"',
    "'": "'",
    "'": "'",
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


# -----------------------------------------------------------------------------
# Language/voice mapping
# -----------------------------------------------------------------------------


def _map_cld2_to_espeak(lang_code: str, primary_voice: str = "en-us") -> str:
    """Map language code to an espeak-ng voice."""
    if not lang_code:
        return "en-us"

    code = lang_code.strip().lower().replace("_", "-")
    base = code.split("-", 1)[0] if code else "en"

    if base == "yue" or code in ("zh-hk", "zh-yue"):
        return "yue"

    if base in ("zh", "cmn"):
        return "cmn-latn-pinyin"

    if base in ("jv", "su"):
        return "id"

    if base == "gl":
        return "es"

    if base == "mn":
        return "ru"

    if lang_code == "en":
        return primary_voice

    return base


# -----------------------------------------------------------------------------
# Text normalization
# -----------------------------------------------------------------------------


def _normalize_text_for_voice(text: str, voice: str) -> str:
    """Apply language-specific normalization before espeak phonemization."""
    norm_text = _normalize_punct_and_space(text)
    v = (voice or "").lower()

    if v.startswith("ja"):
        import ipadic
        from fugashi import GenericTagger
        from pykakasi import kakasi

        tagger = GenericTagger(ipadic.MECAB_ARGS)
        parts = []
        for token in tagger(norm_text):
            surf = token.surface
            pos = str(token.part_of_speech) if hasattr(token, "part_of_speech") else ""
            primary_pos = pos.split(",")[0] if pos else ""
            if primary_pos == "助詞":
                if surf == "は":
                    surf = "わ"
                elif surf == "へ":
                    surf = "え"
                elif surf == "を":
                    surf = "お"
            parts.append(surf)
        norm_text = "".join(parts)

        kks = kakasi()
        kks.setMode("J", "H")
        kks.setMode("K", "H")
        conv = kks.getConverter()
        norm_text = conv.do(norm_text)

    elif v.startswith("ar"):
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
    splitter = MultilingualSplitter()
    result = splitter.split(text)
    segments = result.segments
    main_lang = result.main_language

    _LOGGER.debug(
        "multilingual_splitter segments: %s (main=%s)",
        [(seg.language, _short_text(seg.text, 40)) for seg in segments],
        main_lang,
    )

    phonemes: list = []
    for idx, seg in enumerate(segments):
        span_text = seg.text
        lang = seg.language if seg.language and seg.language != "und" else main_lang or "en"

        if not span_text.strip():
            continue

        voice = _map_cld2_to_espeak(lang or "en", primary_voice)
        span_text = _normalize_text_for_voice(span_text, voice)

        _LOGGER.debug("span[%s]: lang=%s voice=%s text='%s'", idx, lang, voice, _short_text(span_text, 80))

        span_phoneme_sents = _phonemize_espeak_with_reset(casing_fn(span_text), voice, espeak_data)
        span_phonemes = [p for sp in span_phoneme_sents for p in sp]

        _LOGGER.debug("span[%s] phonemes=%s", idx, _short_list(span_phonemes, 32))

        if phonemes and span_phonemes:
            phonemes.append(" ")
        phonemes.extend(span_phonemes)

    return phonemes


def _phonemize_neural(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    voice: str = "en-us",
) -> list:
    """Phonemize text using neural heteronym disambiguation.

    Strategy:
    1. Get phoneme output with word-to-phoneme mapping from C++ (using espeak_Synth callback)
    2. Match heteronym text positions to the mapping
    3. Replace the corresponding phoneme segments with BERT pronunciations

    Uses reliable C++ word mapping instead of space-based segmentation.
    """
    # Only apply neural disambiguation for English voices
    # The heteronym model is trained on English
    if not voice.startswith("en"):
        norm_text = _normalize_text_for_voice(text, voice)
        sent_ph = _phonemize_espeak_with_reset(casing_fn(norm_text), voice, espeak_data)
        return [p for sent in sent_ph for p in sent]

    resolver = _get_heteronym_resolver()

    # Find all heteronyms with their positions and correct pronunciations
    heteronyms = resolver.resolve_all(text)

    if not heteronyms:
        # No heteronyms - just phonemize normally
        norm_text = _normalize_text_for_voice(text, voice)
        sent_ph = _phonemize_espeak_with_reset(casing_fn(norm_text), voice, espeak_data)
        return [p for sent in sent_ph for p in sent]

    _LOGGER.debug(
        "neural: found %d heteronyms: %s",
        len(heteronyms),
        [(h[0], h[1], h[2], h[3]) for h in heteronyms],
    )

    # Get phonemes WITH word-to-phoneme mapping from C++
    norm_text = _normalize_text_for_voice(text, voice)
    processed_text = casing_fn(norm_text)
    sent_ph, word_mapping = _phonemize_espeak_with_mapping(processed_text, voice, espeak_data)
    all_phonemes = [p for sent in sent_ph for p in sent]

    _LOGGER.debug("neural: full phonemes=%s", _short_list(all_phonemes, 48))
    _LOGGER.debug(
        "neural: word mapping from C++: %s",
        [(ts, tl, ps, pe, processed_text[ts-1:ts-1+tl] if ts > 0 else "") for ts, tl, ps, pe in word_mapping],
    )

    # Build replacement map: mapping_index -> new_phonemes
    # Match heteronym positions to word mapping entries
    replacements: Dict[int, List[str]] = {}

    for word, h_start, h_end, correct_ipa in heteronyms:
        # Find which word mapping entry this heteronym corresponds to
        # Note: word_mapping textStart is 1-indexed byte position
        # heteronym positions are 0-indexed character positions
        # For ASCII (English), byte position == character position
        matched_idx = None

        for idx, (text_start, text_len, ph_start, ph_end) in enumerate(word_mapping):
            # Convert 1-indexed byte position to 0-indexed
            map_start = text_start - 1
            map_end = map_start + text_len

            # Check if heteronym overlaps with this word
            if map_start <= h_start < map_end or map_start < h_end <= map_end:
                matched_idx = idx
                break
            # Exact match
            if map_start == h_start and map_end == h_end:
                matched_idx = idx
                break

        if matched_idx is not None:
            replacements[matched_idx] = list(correct_ipa)
            _LOGGER.debug(
                "neural: heteronym '%s' at text[%d:%d] -> mapping_idx=%d, replacement=%s",
                word, h_start, h_end, matched_idx, correct_ipa,
            )
        else:
            _LOGGER.warning(
                "neural: couldn't map heteronym '%s' at [%d:%d] to word mapping",
                word, h_start, h_end,
            )

    # Build result by replacing segments using the mapping
    result: List[str] = []
    last_end = 0

    for idx, (text_start, text_len, ph_start, ph_end) in enumerate(word_mapping):
        # Add any phonemes between the last word and this word (spaces, etc.)
        if ph_start > last_end:
            result.extend(all_phonemes[last_end:ph_start])

        if idx in replacements:
            # Use BERT pronunciation, but preserve trailing punctuation
            segment = list(all_phonemes[ph_start:ph_end])
            # Check for trailing punctuation (., !, ?, etc.)
            trailing_punct = []
            while segment and segment[-1] in ".!?;:,":
                trailing_punct.insert(0, segment.pop())

            result.extend(replacements[idx])
            result.extend(trailing_punct)

            _LOGGER.debug(
                "neural: segment %d replaced: %s -> %s%s",
                idx, all_phonemes[ph_start:ph_end], replacements[idx], trailing_punct,
            )
        else:
            # Keep original
            result.extend(all_phonemes[ph_start:ph_end])

        last_end = ph_end

    # Add any remaining phonemes after the last word
    if last_end < len(all_phonemes):
        result.extend(all_phonemes[last_end:])

    return result


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
) -> Dict[str, List[str]]:
    """Phonemize text for inference.

    Args:
        text: Input text to phonemize.
        config_path: Path to model config JSON.
        espeak_data: Optional path to espeak-ng data.
        neural: If True, use neural heteronym disambiguation.

    Returns a dict with 'phonemes' and 'phoneme_ids'.
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    lang_code = (cfg.get("language") or {}).get("code")
    es_conf = cfg.get("espeak") or {}
    es_voice = es_conf.get("voice")
    primary = es_conf.get("primary") or "en-us"

    casing = get_text_casing("ignore")
    is_multi = (lang_code == "multilingual") or (es_voice == "multilingual")

    _LOGGER.debug("infer: is_multi=%s voice=%s primary=%s neural=%s", is_multi, es_voice or lang_code, primary, neural)

    if neural and not is_multi:
        # Neural heteronym disambiguation mode
        voice = es_voice or lang_code or primary
        voice = _map_cld2_to_espeak(voice, primary)
        phonemes = _phonemize_neural(text, casing, espeak_data, voice)
    elif is_multi:
        phonemes = _phonemize_multilingual(text, casing, espeak_data, primary)
    else:
        voice = es_voice or lang_code or primary
        voice = _map_cld2_to_espeak(voice, primary)
        norm_text = _normalize_text_for_voice(text, voice)
        _LOGGER.debug("infer: voice=%s text='%s'", voice, _short_text(norm_text, 120))
        sent_ph = _phonemize_espeak_with_reset(casing(norm_text), voice, espeak_data)
        phonemes = [p for sent in sent_ph for p in sent]

    _LOGGER.debug("infer: phonemes=%s", _short_list(phonemes, 48))
    ids = phoneme_ids_espeak(phonemes)
    return {"phonemes": phonemes, "phoneme_ids": ids}


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

    Returns a list of spans: [{"phoneme_ids": [...], "speaker_id": int, "text": str}, ...]
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    es_conf = cfg.get("espeak") or {}
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}

    casing = get_text_casing("ignore")
    splitter = MultilingualSplitter()
    split_result = splitter.split(text)
    segments = split_result.segments
    main_lang = split_result.main_language

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

        lang = (seg.language if seg.language and seg.language != "und" else main_lang or "en").lower()

        speaker_info = _find_speaker_for_lang(lang)
        if speaker_info is None and main_lang and main_lang != lang:
            _LOGGER.debug("infer-multispan: lang=%s not available, trying main_lang=%s", lang, main_lang)
            speaker_info = _find_speaker_for_lang(main_lang)
        if speaker_info is None:
            _LOGGER.debug("infer-multispan: main_lang=%s not available, trying primary=%s", main_lang, primary)
            speaker_info = _find_speaker_for_lang(primary.split("-")[0])
        if speaker_info is None:
            _LOGGER.warning("infer-multispan: no speaker found for lang=%s, using speaker 0", lang)
            spk_label = list(spk_id_map.keys())[0] if spk_id_map else "en"
            spk_id = 0
            voice = _map_cld2_to_espeak(spk_label or "en", primary)
        else:
            spk_label, spk_id, voice = speaker_info

        _LOGGER.debug("infer-multispan: lang=%s voice=%s spk_label=%s spk_id=%s neural=%s", lang, voice, spk_label, spk_id, neural)

        if neural:
            ph = _phonemize_neural(span_text, casing, espeak_data, voice)
        else:
            ph_sent = _phonemize_espeak_with_reset(casing(span_text), voice, espeak_data)
            ph = [p for s in ph_sent for p in s]

        _LOGGER.debug("infer-multispan: phonemes=%s", _short_list(ph, 40))

        ids = phoneme_ids_espeak(ph)
        results.append({"phoneme_ids": ids, "speaker_id": int(spk_id), "text": span_text})

    return results


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

    Returns: {"phoneme_ids": [...], "speaker_id": int, "text": str}
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    es_conf = cfg.get("espeak") or {}
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}

    label = speaker_label
    rev = {v: k for k, v in lang_spk_map.items()} if lang_spk_map else {}
    base = rev.get(label, label)
    voice = _map_cld2_to_espeak(base, primary)

    _LOGGER.debug("infer-forced: label=%s base=%s -> voice=%s primary=%s neural=%s", label, base, voice, primary, neural)

    casing = get_text_casing("ignore")

    if neural:
        phonemes = _phonemize_neural(text, casing, espeak_data, voice)
    else:
        norm_text = _normalize_text_for_voice(text, voice)
        _LOGGER.debug("infer-forced: text='%s'", _short_text(norm_text, 120))
        ph_sents = _phonemize_espeak_with_reset(casing(norm_text), voice, espeak_data)
        phonemes = [p for s in ph_sents for p in s]

    _LOGGER.debug("infer-forced: phonemes=%s", _short_list(phonemes, 48))

    ids = phoneme_ids_espeak(phonemes)
    spk_id = spk_id_map.get(label, 0)
    return {"phoneme_ids": ids, "speaker_id": int(spk_id), "text": text}
