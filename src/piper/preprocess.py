"""Phonemization utilities for Piper TTS inference."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from piper_phonemize import phonemize_espeak, phoneme_ids_espeak, tashkeel_run

from ..multilingual_splitter import MultilingualSplitter

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

        span_phoneme_sents = phonemize_espeak(casing_fn(span_text), voice, espeak_data)
        span_phonemes = [p for sp in span_phoneme_sents for p in sp]

        _LOGGER.debug("span[%s] phonemes=%s", idx, _short_list(span_phonemes, 32))

        if phonemes and span_phonemes:
            phonemes.append(" ")
        phonemes.extend(span_phonemes)

    return phonemes


def phonemize_text_for_infer(
    text: str,
    config_path: "Path | str",
    espeak_data: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Phonemize text for inference.

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

    _LOGGER.debug("infer: is_multi=%s voice=%s primary=%s", is_multi, es_voice or lang_code, primary)

    if is_multi:
        phonemes = _phonemize_multilingual(text, casing, espeak_data, primary)
    else:
        voice = es_voice or lang_code or primary
        voice = _map_cld2_to_espeak(voice, primary)
        norm_text = _normalize_text_for_voice(text, voice)
        _LOGGER.debug("infer: voice=%s text='%s'", voice, _short_text(norm_text, 120))
        sent_ph = phonemize_espeak(casing(norm_text), voice, espeak_data)
        phonemes = [p for sent in sent_ph for p in sent]

    _LOGGER.debug("infer: phonemes=%s", _short_list(phonemes, 48))
    ids = phoneme_ids_espeak(phonemes)
    return {"phonemes": phonemes, "phoneme_ids": ids}


def phonemize_spans_with_speakers(
    text: str,
    config_path: "Path | str",
    espeak_data: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Phonemize text with language-based speaker assignment.

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

        _LOGGER.debug("infer-multispan: lang=%s voice=%s spk_label=%s spk_id=%s", lang, voice, spk_label, spk_id)

        ph_sent = phonemize_espeak(casing(span_text), voice, espeak_data)
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
) -> Dict[str, object]:
    """Phonemize text for a specific speaker (skip language detection).

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

    _LOGGER.debug("infer-forced: label=%s base=%s -> voice=%s primary=%s", label, base, voice, primary)

    casing = get_text_casing("ignore")
    norm_text = _normalize_text_for_voice(text, voice)

    _LOGGER.debug("infer-forced: text='%s'", _short_text(norm_text, 120))

    ph_sents = phonemize_espeak(casing(norm_text), voice, espeak_data)
    phonemes = [p for s in ph_sents for p in s]

    _LOGGER.debug("infer-forced: phonemes=%s", _short_list(phonemes, 48))

    ids = phoneme_ids_espeak(phonemes)
    spk_id = spk_id_map.get(label, 0)
    return {"phoneme_ids": ids, "speaker_id": int(spk_id), "text": text}
