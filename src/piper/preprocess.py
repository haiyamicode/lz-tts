#!/usr/bin/env python3
import argparse
import csv
import dataclasses
import itertools
import json
import logging
import os
import unicodedata
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing import JoinableQueue, Process, Queue
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from piper_phonemize import (
    phonemize_espeak,
    phonemize_codepoints,
    phoneme_ids_espeak,
    phoneme_ids_codepoints,
    get_codepoints_map,
    get_espeak_map,
    get_max_phonemes,
    tashkeel_run,
)
from ..multilingual_splitter import MultilingualSplitter, split_text

# Training-only imports (optional for inference)
try:
    from .norm_audio import cache_norm_audio, make_silence_detector
except ImportError:
    cache_norm_audio = None
    make_silence_detector = None

_DIR = Path(__file__).parent
_VERSION = (_DIR / "VERSION").read_text(encoding="utf-8").strip()

# Configure logging immediately so it actually works
_DEBUG = os.environ.get("PREPROCESS_DEBUG", "").lower() in ("1", "true", "yes")
if _DEBUG:
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] %(name)s: %(message)s',
        force=True
    )

_LOGGER = logging.getLogger("preprocess")
_LOGGER.setLevel(logging.DEBUG if _DEBUG else logging.INFO)


def _short_text(s: str, n: int = 80) -> str:
    try:
        return s if len(s) <= n else s[: n - 1] + "…"
    except Exception:
        return s


def _short_list(xs, n: int = 24):
    try:
        return list(xs[:n]) + (["…"] if len(xs) > n else [])
    except Exception:
        return xs


class PhonemeType(str, Enum):
    ESPEAK = "espeak"
    """Phonemes come from espeak-ng"""

    TEXT = "text"
    """Phonemes come from text itself"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Directory with audio dataset"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write output files for training",
    )
    parser.add_argument("--language", required=True, help="eSpeak-ng voice or 'multilingual'")
    parser.add_argument(
        "--primary-voice",
        default="en-us",
        help="Primary espeak-ng voice to use for generic 'en' segments when --language=multilingual (e.g., en-gb)",
    )
    # Normalizations are always enabled automatically for multilingual spans
    parser.add_argument(
        "--sample-rate",
        type=int,
        required=True,
        help="Target sample rate for voice (hertz)",
    )
    parser.add_argument(
        "--dataset-format", choices=("ljspeech", "mycroft"), required=True
    )
    parser.add_argument("--cache-dir", help="Directory to cache processed audio files")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument(
        "--single-speaker", action="store_true", help="Force single speaker dataset"
    )
    parser.add_argument(
        "--speaker-id", type=int, help="Add speaker id to single speaker dataset"
    )
    parser.add_argument(
        "--espeak-data",
        help="Path to espeak-ng-data directory (overrides packaged data)",
    )
    #
    parser.add_argument(
        "--phoneme-type",
        choices=list(PhonemeType),
        default=PhonemeType.ESPEAK,
        help="Type of phonemes to use (default: espeak)",
    )
    parser.add_argument(
        "--text-casing",
        choices=("ignore", "lower", "upper", "casefold"),
        default="ignore",
        help="Casing applied to utterance text",
    )
    #
    parser.add_argument(
        "--dataset-name",
        help="Name of dataset to put in config (default: name of <ouput_dir>/../)",
    )
    parser.add_argument(
        "--audio-quality",
        help="Audio quality to put in config (default: name of <output_dir>)",
    )
    #
    parser.add_argument(
        "--tashkeel",
        action="store_true",
        help="Diacritize Arabic text with libtashkeel",
    )
    #
    parser.add_argument(
        "--skip-audio", action="store_true", help="Don't preprocess audio"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Print DEBUG messages to the console"
    )
    args = parser.parse_args()

    if args.single_speaker and (args.speaker_id is not None):
        _LOGGER.fatal("--single-speaker and --speaker-id cannot both be provided")
        return

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)

    # Prevent log spam
    logging.getLogger("numba").setLevel(logging.WARNING)

    # Ensure enum
    args.phoneme_type = PhonemeType(args.phoneme_type)

    # Convert to paths and create output directories
    args.input_dir = Path(args.input_dir)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.cache_dir = (
        Path(args.cache_dir)
        if args.cache_dir
        else args.output_dir / "cache" / str(args.sample_rate)
    )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_format == "mycroft":
        make_dataset = mycroft_dataset
    else:
        make_dataset = ljspeech_dataset

    # Count speakers (from dataset only)
    _LOGGER.debug("Counting number of speakers/utterances in the dataset")
    speaker_counts: "Counter[str]" = Counter()
    num_utterances = 0
    for utt in make_dataset(args):
        speaker = utt.speaker or ""
        speaker_counts[speaker] += 1
        num_utterances += 1

    assert num_utterances > 0, "No utterances found"

    is_multispeaker = len(speaker_counts) > 1
    speaker_ids: Dict[str, int] = {}

    if is_multispeaker:
        _LOGGER.info("%s speakers detected", len(speaker_counts))
        # Assign speaker ids by most number of utterances first
        for speaker_id, (speaker, _speaker_count) in enumerate(
            speaker_counts.most_common()
        ):
            speaker_ids[speaker] = speaker_id
    else:
        _LOGGER.info("Single speaker dataset")

    # Write config
    audio_quality = args.audio_quality or args.output_dir.name
    dataset_name = args.dataset_name or args.output_dir.parent.name

    # Build language/espeak config
    espeak_config = {"voice": args.language}
    language_config = {"code": args.language}
    if args.language.lower() == "multilingual":
        # Expose multilingual plus available voices list and primary fallback
        # Normalize primary voice for espeak-ng common codes
        primary_norm = args.primary_voice.lower()
        if primary_norm in ("en-gb", "en_gb", "gb", "en"):
            primary_norm = "en"
        elif primary_norm in ("en-us", "en_us", "us"):
            primary_norm = "en-us"

        espeak_config = {
            "voice": "multilingual",
            "voices": [primary_norm, "cmn-latn-pinyin"],
            "primary": primary_norm,
        }
        language_config = {"code": "multilingual"}

    # Build language->speaker label mapping for inference routing
    language_speakers: Dict[str, str] = {}
    if speaker_counts:
        # Default: identity map for known labels
        for lbl in speaker_counts.keys():
            language_speakers[lbl] = lbl
        # Special-case: espeak 'cmn-latn-pinyin' voice corresponds to dataset label 'zh' when present
        if "zh" in speaker_counts:
            language_speakers.setdefault("cmn-latn-pinyin", "zh")

    with open(args.output_dir / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {
                "dataset": dataset_name,
                "audio": {
                    "sample_rate": args.sample_rate,
                    "quality": audio_quality,
                },
                "espeak": espeak_config,
                "language": language_config,
                "inference": {"noise_scale": 0.667, "length_scale": 1, "noise_w": 0.8},
                "phoneme_type": args.phoneme_type.value,
                "phoneme_map": {},
                "phoneme_id_map": get_codepoints_map()[args.language]
                if args.phoneme_type == PhonemeType.TEXT
                else get_espeak_map(),
                "num_symbols": get_max_phonemes(),
                "num_speakers": len(speaker_counts),
                "speaker_id_map": speaker_ids,
                "language_speakers": language_speakers,
                "piper_version": _VERSION,
            },
            config_file,
            ensure_ascii=False,
            indent=4,
        )
    _LOGGER.info("Wrote dataset config")

    if (args.max_workers is None) or (args.max_workers < 1):
        args.max_workers = os.cpu_count()

    assert args.max_workers is not None

    batch_size = int(num_utterances / (args.max_workers * 2))
    queue_in: "Queue[Iterable[Utterance]]" = JoinableQueue()
    queue_out: "Queue[Optional[Utterance]]" = Queue()

    # Start workers
    if args.phoneme_type == PhonemeType.TEXT:
        target = phonemize_batch_text
    else:
        target = phonemize_batch_espeak

    processes = [
        Process(target=target, args=(args, queue_in, queue_out))
        for _ in range(args.max_workers)
    ]
    for proc in processes:
        proc.start()

    _LOGGER.info(
        "Processing %s utterance(s) with %s worker(s)", num_utterances, args.max_workers
    )

    # Add index to each utterance for order preservation
    def add_indices(dataset_iter):
        for i, utt in enumerate(dataset_iter):
            utt.index = i
            yield utt

    for utt_batch in batched(
        add_indices(make_dataset(args)),
        batch_size,
    ):
        queue_in.put(utt_batch)

    _LOGGER.debug("Waiting for jobs to finish")
    missing_phonemes: "Counter[str]" = Counter()
    results = []
    for _ in range(num_utterances):
        utt = queue_out.get()
        if utt is not None:
            if utt.speaker is not None:
                utt.speaker_id = speaker_ids[utt.speaker]
            results.append(utt)
            missing_phonemes.update(utt.missing_phonemes)

    # Sort by original index to preserve order
    results.sort(key=lambda u: u.index)

    # Write sorted results to jsonl
    with open(args.output_dir / "dataset.jsonl", "w", encoding="utf-8") as dataset_file:
        for utt in results:
            utt_dict = dataclasses.asdict(utt)
            utt_dict.pop("missing_phonemes")
            utt_dict.pop("index")  # Don't write index to output

            # JSONL
            json.dump(
                utt_dict,
                dataset_file,
                ensure_ascii=False,
                cls=PathEncoder,
            )
            print("", file=dataset_file)

    if missing_phonemes:
        for phoneme, count in missing_phonemes.most_common():
            _LOGGER.warning("Missing %s (%s)", phoneme, count)

        _LOGGER.warning("Missing %s phoneme(s)", len(missing_phonemes))

    # Signal workers to stop
    for proc in processes:
        queue_in.put(None)

    # Wait for workers to stop
    for proc in processes:
        proc.join(timeout=1)


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
# Punctuation/space normalization (language-agnostic)

# Map common CJK and full-width punctuation to ASCII equivalents so eSpeak-ng
# treats them consistently for sentence/phrase boundaries.
_PUNCT_MAP = {
    # Chinese/Japanese punctuation
    # CJK punctuation → ASCII plus a trailing space
    # (CJK doesn't require spaces; we insert one to match ASCII usage)
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
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    # Ellipsis/dashes/various separators
    "…": "...",
    "‥": "..",
    "—": "-",
    "–": "-",
    "―": "-",
    "〜": "~",
    "～": "~",
    # Middle dots are typically intra-word separators; normalize to hyphen
    # to preserve token connectivity without forcing sentence breaks.
    "・": "-",
    "·": "-",
    "．": ". ",
    # Arabic punctuation to ASCII
    "،": ",",
    "؛": ";",
    "؟": "?",
    # Devanagari danda(s) to ASCII periods
    "।": ".",
    "॥": "..",
    # Hebrew punctuation marks to ASCII quotes
    "׳": "'",
    "״": '"',
    # Slashes and misc symbols (full-width variants)
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

_IDEOGRAPHIC_SPACE = "\u3000"

def _normalize_punct_and_space(text: str) -> str:
    """Punctuation-only normalization.

    - No whitespace collapsing or insertion beyond what the mapping encodes.
    - No global NFKC; only explicit substitutions in _PUNCT_MAP.
    """
    s = text
    keys = sorted(_PUNCT_MAP.keys(), key=len, reverse=True)
    if keys:
        pattern = re.compile("(" + "|".join(map(re.escape, keys)) + ")")
        s = pattern.sub(lambda m: _PUNCT_MAP.get(m.group(0), m.group(0)), s)
    return s


def _map_cld2_to_espeak(lang_code: str, primary_voice: str = "en-us") -> str:
    """Map language code to an espeak-ng voice in a simple, predictable way.

    - Normalize to lowercase, replace '_' with '-'
    - Take the base language (split on '-')
    - Special-cases: Chinese -> 'cmn-latn-pinyin'; 'en' -> 'en-us'
    - Otherwise return the base code directly (most espeak voices match base code)
    """
    if not lang_code:
        return "en-us"

    code = lang_code.strip().lower().replace("_", "-")
    base = code.split("-", 1)[0] if code else "en"

    # Cantonese: zh-HK, zh-yue, yue -> espeak "yue" voice
    if base == "yue" or code in ("zh-hk", "zh-yue"):
        return "yue"

    if base in ("zh", "cmn"):
        # Use eSpeak's Mandarin Pinyin voice code
        return "cmn-latn-pinyin"

    # Languages not directly supported by packaged eSpeak-ng data.
    # Fall back to the closest available voice instead of erroring.
    if base in ("jv", "su"):
        # Javanese / Sundanese -> approximate with Indonesian
        return "id"

    if base == "gl":
        # Galician -> approximate with Spanish
        return "es"

    if base == "mn":
        # Mongolian -> approximate with Russian
        return "ru"

    if lang_code == "en":
        return primary_voice

    return base


def _select_voice_for_span(lang_code: str, primary_voice: str) -> str:
    """Map language code to espeak voice."""
    code = (lang_code or "en").lower()
    return _map_cld2_to_espeak(code, primary_voice)


def _phonemize_multilingual(
    text: str,
    casing_fn,
    espeak_data: Optional[str] = None,
    primary_voice: str = "en-us",
) -> list:
    """Phonemize mixed-language text using MultilingualSplitter for segmentation.
    """
    splitter = MultilingualSplitter()
    result = splitter.split(text)
    segments = result.segments
    main_lang = result.main_language
    _LOGGER.debug("multilingual_splitter segments: %s (main=%s)", [(seg.language, _short_text(seg.text, 40)) for seg in segments], main_lang)
    phonemes: list = []

    for idx, seg in enumerate(segments):
        span_text = seg.text
        # Use main language as fallback for undetermined/empty segments
        lang = seg.language if seg.language and seg.language != "und" else main_lang or "en"
        # Skip empty segments
        if not span_text.strip():
            continue
        voice = _select_voice_for_span(lang, primary_voice)
        span_text = _normalize_text_for_voice(span_text, voice)
        _LOGGER.debug(
            "span[%s]: lang=%s voice=%s text='%s'",
            idx,
            lang,
            voice,
            _short_text(span_text, 80),
        )
        # No explicit language tokens; language is conveyed via speaker (multi-speaker setup)
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
    """Public helper for inference scripts.

    - Reads training config.json
    - Applies multilingual span handling if configured
    - Returns a dict with phonemes and phoneme_ids suitable for piper_train.infer
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    lang_code = (cfg.get("language") or {}).get("code")
    es_conf = (cfg.get("espeak") or {})
    es_voice = es_conf.get("voice")
    primary = es_conf.get("primary") or "en-us"

    casing = get_text_casing("ignore")

    is_multi = (lang_code == "multilingual") or (es_voice == "multilingual")
    _LOGGER.debug(
        "infer: is_multi=%s voice=%s primary=%s", is_multi, es_voice or lang_code, primary
    )
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
    """Return a list of spans with phoneme_ids and mapped speaker_id for multi-speaker-by-language infer.

    Each item: {"phoneme_ids": [...], "speaker_id": int, "text": <span text>}
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    lang_code = (cfg.get("language") or {}).get("code")
    es_conf = (cfg.get("espeak") or {})
    es_voice = es_conf.get("voice")
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}

    casing = get_text_casing("ignore")
    splitter = MultilingualSplitter()
    split_result = splitter.split(text)
    segments = split_result.segments
    main_lang = split_result.main_language
    _LOGGER.debug("infer-multispan: segments=%s (main=%s)", [(seg.language, _short_text(seg.text, 60)) for seg in segments], main_lang)

    # Helper to check if a language has an available speaker
    def _find_speaker_for_lang(lang_code: str) -> Optional[tuple]:
        """Returns (speaker_label, speaker_id) if found, else None."""
        voice = _select_voice_for_span(lang_code, primary)
        base = "en" if voice.startswith("en") else voice
        # Try explicit mapping first
        spk_label = lang_spk_map.get(base)
        if spk_label and spk_label in spk_id_map:
            return (spk_label, spk_id_map[spk_label], voice)
        # Try common aliases
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
        # Skip empty segments
        if not span_text.strip():
            continue
        # Use main language as fallback for undetermined segments
        lang = (seg.language if seg.language and seg.language != "und" else main_lang or "en").lower()

        # Try to find a speaker for the detected language
        speaker_info = _find_speaker_for_lang(lang)
        if speaker_info is None:
            # Language not available - try main language
            _LOGGER.debug("infer-multispan: lang=%s not available, trying main_lang=%s", lang, main_lang)
            if main_lang and main_lang != lang:
                speaker_info = _find_speaker_for_lang(main_lang)
        if speaker_info is None:
            # Still not found - try primary voice
            _LOGGER.debug("infer-multispan: main_lang=%s not available, trying primary=%s", main_lang, primary)
            speaker_info = _find_speaker_for_lang(primary.split("-")[0])
        if speaker_info is None:
            # Last resort - use speaker 0 with primary voice
            _LOGGER.warning("infer-multispan: no speaker found for lang=%s, using speaker 0", lang)
            spk_label = list(spk_id_map.keys())[0] if spk_id_map else "en"
            spk_id = 0
            voice = _select_voice_for_span(spk_label, primary)
        else:
            spk_label, spk_id, voice = speaker_info

        _LOGGER.debug("infer-multispan: lang=%s voice=%s spk_label=%s spk_id=%s", lang, voice, spk_label, spk_id)
        ph_sent = phonemize_espeak(casing(span_text), voice, espeak_data)
        ph = [p for s in ph_sent for p in s]
        _LOGGER.debug("infer-multispan: phonemes=%s", _short_list(ph, 40))
        ids = phoneme_ids_espeak(ph)
        results.append(
            {
                "phoneme_ids": ids,
                "speaker_id": int(spk_id),
                "text": span_text,
            }
        )

    return results


def phonemize_text_for_speaker(
    text: str,
    config_path: "Path | str",
    speaker_label: str,
    espeak_data: Optional[str] = None,
) -> Dict[str, object]:
    """Phonemize entire text for a specific speaker label (skip language split).

    Returns a single dict: {"phoneme_ids": [...], "speaker_id": int, "text": <text>}
    """
    cfg_path = Path(config_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    es_conf = (cfg.get("espeak") or {})
    primary = es_conf.get("primary") or "en-us"
    spk_id_map: Dict[str, int] = cfg.get("speaker_id_map") or {}
    lang_spk_map: Dict[str, str] = cfg.get("language_speakers") or {}

    # Determine voice from speaker label by reverse mapping
    # If label equals a language code (our default), use that
    label = speaker_label
    # Map label back to language code when language_speakers provided
    rev = {v: k for k, v in lang_spk_map.items()} if lang_spk_map else {}
    base = rev.get(label, label)
    voice = _map_cld2_to_espeak(base, primary)
    _LOGGER.debug(
        "infer-forced: label=%s base=%s -> voice=%s primary=%s", label, base, voice, primary
    )

    casing = get_text_casing("ignore")
    norm_text = _normalize_text_for_voice(text, voice)
    _LOGGER.debug("infer-forced: text='%s'", _short_text(norm_text, 120))
    ph_sents = phonemize_espeak(casing(norm_text), voice, espeak_data)
    phonemes = [p for s in ph_sents for p in s]
    _LOGGER.debug("infer-forced: phonemes=%s", _short_list(phonemes, 48))
    ids = phoneme_ids_espeak(phonemes)
    spk_id = spk_id_map.get(label, 0)
    return {"phoneme_ids": ids, "speaker_id": int(spk_id), "text": text}


def phonemize_batch_espeak(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug("utt: speaker=%s text='%s'", getattr(utt, "speaker", None), _short_text(utt.text, 120))
                    if args.language.lower() == "multilingual":
                        # If speaker label is present, skip language split and phonemize with that voice.
                        if utt.speaker:
                            spk = str(utt.speaker).lower()
                            voice = _map_cld2_to_espeak(
                                spk, getattr(args, "primary_voice", "en-us")
                            )
                            _LOGGER.debug("train-ms: skip split, speaker=%s -> voice=%s", spk, voice)
                            norm_text = _normalize_text_for_voice(utt.text, voice)
                            all_phonemes = phonemize_espeak(
                                casing(norm_text), voice, args.espeak_data
                            )
                            utt.phonemes = [p for sent in all_phonemes for p in sent]
                            _LOGGER.debug("train-ms: phonemes=%s", _short_list(utt.phonemes, 48))
                        else:
                            # No speaker provided; segment by language and normalize per-span
                            _LOGGER.debug("train-ms: no speaker -> split by language")
                            utt.phonemes = _phonemize_multilingual(
                                utt.text,
                                casing,
                                args.espeak_data,
                                getattr(args, "primary_voice", "en-us"),
                            )
                            _LOGGER.debug("train-ms: phonemes=%s", _short_list(utt.phonemes, 48))
                    else:
                        all_phonemes = phonemize_espeak(
                            casing(utt.text), args.language, args.espeak_data
                        )
                        # Flatten
                        utt.phonemes = [
                            phoneme
                            for sentence_phonemes in all_phonemes
                            for phoneme in sentence_phonemes
                        ]
                        _LOGGER.debug("train-ss: voice=%s phonemes=%s", args.language, _short_list(utt.phonemes, 48))
                    utt.phoneme_ids = phoneme_ids_espeak(
                        utt.phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_espeak")


def phonemize_batch_text(
    args: argparse.Namespace, queue_in: JoinableQueue, queue_out: Queue
):
    try:
        casing = get_text_casing(args.text_casing)
        silence_detector = make_silence_detector()

        while True:
            utt_batch = queue_in.get()
            if utt_batch is None:
                break

            for utt in utt_batch:
                try:
                    if args.tashkeel:
                        utt.text = tashkeel_run(utt.text)

                    _LOGGER.debug(utt)
                    all_phonemes = phonemize_codepoints(casing(utt.text))
                    # Flatten
                    utt.phonemes = [
                        phoneme
                        for sentence_phonemes in all_phonemes
                        for phoneme in sentence_phonemes
                    ]
                    utt.phoneme_ids = phoneme_ids_codepoints(
                        args.language,
                        utt.phonemes,
                        missing_phonemes=utt.missing_phonemes,
                    )
                    if not args.skip_audio:
                        utt.audio_norm_path, utt.audio_spec_path = cache_norm_audio(
                            utt.audio_path,
                            args.cache_dir,
                            silence_detector,
                            args.sample_rate,
                        )
                    queue_out.put(utt)
                except TimeoutError:
                    _LOGGER.error("Skipping utterance due to timeout: %s", utt)
                except Exception:
                    _LOGGER.exception("Failed to process utterance: %s", utt)
                    queue_out.put(None)

            queue_in.task_done()
    except Exception:
        _LOGGER.exception("phonemize_batch_text")


# -----------------------------------------------------------------------------


@dataclass
class Utterance:
    text: str
    audio_path: Path
    speaker: Optional[str] = None
    speaker_id: Optional[int] = None
    phonemes: Optional[List[str]] = None
    phoneme_ids: Optional[List[int]] = None
    audio_norm_path: Optional[Path] = None
    audio_spec_path: Optional[Path] = None
    missing_phonemes: "Counter[str]" = field(default_factory=Counter)
    index: Optional[int] = None  # For preserving order with multiprocessing


class PathEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def ljspeech_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    speaker_id = args.speaker_id
    skip_audio = args.skip_audio

    # filename|speaker|text
    # speaker is optional
    metadata_path = dataset_dir / "metadata.csv"
    assert metadata_path.exists(), f"Missing {metadata_path}"

    wav_dir = dataset_dir / "wav"
    if not wav_dir.is_dir():
        wav_dir = dataset_dir / "wavs"

    with open(metadata_path, "r", encoding="utf-8") as csv_file:
        reader = csv.reader(csv_file, delimiter="|")
        for row in reader:
            assert len(row) >= 2, "Not enough columns"

            speaker: Optional[str] = None
            if is_single_speaker or (len(row) == 2):
                filename, text = row[0], row[-1]
            else:
                filename, speaker, text = row[0], row[1], row[-1]

            # Try file name relative to metadata
            wav_path = metadata_path.parent / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = metadata_path.parent / f"{filename}.wav"

            if not wav_path.exists():
                # Try wav/ or wavs/
                wav_path = wav_dir / filename

            if not wav_path.exists():
                # Try with .wav
                wav_path = wav_dir / f"{filename}.wav"

            if not skip_audio:
                if not wav_path.exists():
                    _LOGGER.warning("Missing %s", filename)
                    continue

                if wav_path.stat().st_size == 0:
                    _LOGGER.warning("Empty file: %s", wav_path)
                    continue

            yield Utterance(
                text=text, audio_path=wav_path, speaker=speaker, speaker_id=speaker_id
            )


def mycroft_dataset(args: argparse.Namespace) -> Iterable[Utterance]:
    dataset_dir = args.input_dir
    is_single_speaker = args.single_speaker
    skip_audio = args.skip_audio

    speaker_id = 0
    for metadata_path in dataset_dir.glob("**/*-metadata.txt"):
        speaker = metadata_path.parent.name if not is_single_speaker else None
        with open(metadata_path, "r", encoding="utf-8") as csv_file:
            # filename|text|length
            reader = csv.reader(csv_file, delimiter="|")
            for row in reader:
                filename, text = row[0], row[1]
                wav_path = metadata_path.parent / filename
                if skip_audio or (wav_path.exists() and (wav_path.stat().st_size > 0)):
                    yield Utterance(
                        text=text,
                        audio_path=wav_path,
                        speaker=speaker,
                        speaker_id=speaker_id if not is_single_speaker else None,
                    )
        speaker_id += 1


# -----------------------------------------------------------------------------


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError("n must be at least one")
    it = iter(iterable)
    batch = list(itertools.islice(it, n))
    while batch:
        yield batch
        batch = list(itertools.islice(it, n))


# -----------------------------------------------------------------------------

def _normalize_text_for_voice(text: str, voice: str) -> str:
    """Apply language-specific normalization before espeak phonemization.

    - ja: particle normalization (は→わ, へ→え, を→お) + Kanji/Katakana→Hiragana
    - ar: diacritize with libtashkeel
    - others: unchanged
    """
    # Always normalize punctuation/space first
    norm_text = _normalize_punct_and_space(text)
    v = (voice or "").lower()
    if v.startswith("ja"):
        from fugashi import Tagger  # type: ignore
        tagger = Tagger()
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
        from pykakasi import kakasi  # type: ignore
        kks = kakasi()
        kks.setMode("J", "H")
        kks.setMode("K", "H")
        conv = kks.getConverter()
        norm_text = conv.do(norm_text)
    elif v.startswith("ar"):
        norm_text = tashkeel_run(norm_text)
    return norm_text

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
