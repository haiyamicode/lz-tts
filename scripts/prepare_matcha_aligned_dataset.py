#!/usr/bin/env python3
"""Prepare clean aligned Matcha filelists from the source multilingual dataset."""

import argparse
import json
import math
import random
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from src.piper.preprocess import _map_cld2_to_espeak, _normalize_punct_and_space, _phonemize_espeak_with_mapping


DEFAULT_INPUT_DIR = Path.home() / "Projects/gen-tts/data/lzspeech-multilingual-plus"
DEFAULT_OUTPUT_DIR = Path("local/Matcha-TTS/data/lzspeech_multilingual_plus_22050")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_VALID_RATIO = 0.025
DEFAULT_SEED = 1234
ASTERISK_STAGE_DIRECTION = re.compile(r"\*\([^)]*\)\*")


@dataclass(frozen=True)
class SourceRow:
    utt_id: str
    lang: str
    text: str
    audio_path: Path


def read_metadata(input_dir: Path):
    metadata_path = input_dir / "metadata.csv"
    wav_dir = input_dir / "wav"
    rows = []
    seen_ids = set()
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_no, line in enumerate(metadata_file, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 3:
                raise ValueError(f"Expected utt_id|lang|text at {metadata_path}:{line_no}: {line[:200]!r}")
            utt_id, lang, text = parts
            if utt_id in seen_ids:
                raise ValueError(f"Duplicate metadata row id at {metadata_path}:{line_no}: {utt_id}")
            seen_ids.add(utt_id)
            audio_path = wav_dir / f"{utt_id}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio for {utt_id} at {audio_path}")
            rows.append(SourceRow(utt_id, lang, text, audio_path))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    return rows


def split_rows(rows, valid_ratio: float, seed: int):
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for row in rows:
        by_lang[row.lang].append(row)

    train_rows = []
    valid_rows = []
    for lang, lang_rows in sorted(by_lang.items()):
        shuffled = list(lang_rows)
        rng.shuffle(shuffled)
        valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
        valid_rows.extend(shuffled[:valid_count])
        train_rows.extend(shuffled[valid_count:])

    rng.shuffle(train_rows)
    rng.shuffle(valid_rows)
    return train_rows, valid_rows


def _resample_audio(input_path: Path, output_path: Path, sample_rate: int):
    audio, source_rate = sf.read(input_path, always_2d=True, dtype="float32")
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    if source_rate != sample_rate:
        gcd = math.gcd(source_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, source_rate // gcd).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")


def _flatten_phonemes(sentence):
    return "".join(sentence)


def _text_units_by_space(text: str):
    units = []
    start = None
    for index, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                units.append(text[start:index])
                start = None
        elif start is None:
            start = index
    if start is not None:
        units.append(text[start:])
    return units


def _phoneme_groups_by_space(sentence):
    groups = []
    current = []
    for phoneme in sentence:
        if phoneme == " ":
            if current:
                groups.append("".join(current))
                current = []
        else:
            current.append(phoneme)
    if current:
        groups.append("".join(current))
    return groups


def build_korean_units(text: str, sentences):
    text_units = _text_units_by_space(text)
    phoneme_groups = []
    for sentence in sentences:
        phoneme_groups.extend(_phoneme_groups_by_space(sentence))

    if not text_units or not phoneme_groups:
        return []

    if len(text_units) != len(phoneme_groups):
        return [{"text": unicodedata.normalize("NFC", text), "phonemes": unicodedata.normalize("NFC", "".join(phoneme_groups))}]

    return [
        {"text": unicodedata.normalize("NFC", unit_text), "phonemes": unicodedata.normalize("NFC", phonemes)}
        for unit_text, phonemes in zip(text_units, phoneme_groups)
    ]


def normalize_for_alignment(text: str, lang: str):
    text = ASTERISK_STAGE_DIRECTION.sub(" ", text)
    voice = _map_cld2_to_espeak(lang, "en-us")
    if voice.lower().startswith("ja"):
        return text
    return " ".join(_normalize_punct_and_space(text).split())


def build_units(text: str, lang: str):
    voice = _map_cld2_to_espeak(lang, "en-us")
    sentences, mappings = _phonemize_espeak_with_mapping(text, voice, None)
    if lang == "ko" and not any(sentence_mappings for sentence_mappings in mappings):
        units = build_korean_units(text, sentences)
        if units:
            return units

    units = []

    for sentence, sentence_mappings in zip(sentences, mappings):
        for text_start, text_len, ph_start, ph_end, punct_len in sentence_mappings:
            if text_start <= 0:
                raise ValueError(f"Invalid text_start={text_start} for lang={lang!r}, text={text!r}")
            start = text_start - 1
            end = start + text_len
            if punct_len and end < len(text) and text[end : end + punct_len].strip():
                end += punct_len

            unit_text = text[start:end]
            phonemes = _flatten_phonemes(sentence[ph_start:ph_end])
            if "|" in phonemes:
                raise ValueError(f"Pipe character is not supported in phonemes: lang={lang!r} text={text!r}")
            if not unit_text:
                raise ValueError(
                    f"Empty aligned unit for lang={lang!r}, text={text!r}, unit_text={unit_text!r}, phonemes={phonemes!r}"
                )
            units.append({"text": unicodedata.normalize("NFC", unit_text), "phonemes": unicodedata.normalize("NFC", phonemes)})

    if not units:
        raise ValueError(f"No alignment units produced for lang={lang!r}, text={text!r}")
    return units


def has_lexical_text(text: str):
    return any(unicodedata.category(ch)[0] in {"L", "N"} for ch in text)


def process_row(args):
    row, output_dir, sample_rate = args
    wav_path = output_dir / "wavs" / f"{row.utt_id}.wav"
    _resample_audio(row.audio_path, wav_path, sample_rate)
    raw_text = unicodedata.normalize("NFC", normalize_for_alignment(row.text, row.lang))
    if not has_lexical_text(raw_text):
        return None
    units = build_units(raw_text, row.lang)
    phoneme_text = unicodedata.normalize("NFC", " ".join(unit["phonemes"] for unit in units))
    if not phoneme_text:
        raise ValueError(f"Empty phoneme text for {row.utt_id}")
    return {
        "split": None,
        "utt_id": row.utt_id,
        "lang": row.lang,
        "filepath": str(wav_path.resolve()),
        "phoneme_text": phoneme_text,
        "raw_text": raw_text,
        "units": units,
        "symbols": sorted(set("".join(unit["phonemes"] for unit in units))),
    }


def write_filelist(path: Path, items):
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                "|".join(
                    [
                        item["filepath"],
                        item["lang"],
                        item["phoneme_text"],
                        item["raw_text"],
                        json.dumps(item["units"], ensure_ascii=False, separators=(",", ":")),
                    ]
                )
                + "\n"
            )


def write_vocab(path: Path, symbols):
    items = [{"symbol": symbol, "id": index} for index, symbol in enumerate(sorted(symbols), start=1)]
    payload = {"size": len(items) + 1, "symbols": items}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload["size"]


def process_split(name: str, rows, output_dir: Path, sample_rate: int, workers: int):
    results = []
    skipped = 0
    by_lang = defaultdict(list)
    for row in rows:
        by_lang[row.lang].append(row)

    for lang, lang_rows in sorted(by_lang.items()):
        tasks = [(row, output_dir, sample_rate) for row in lang_rows]
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for item in tqdm(executor.map(process_row, tasks, chunksize=8), total=len(tasks), desc=f"{name}:{lang}"):
                if item is None:
                    skipped += 1
                    continue
                item["split"] = name
                results.append(item)

    if skipped:
        print(f"{name}: skipped {skipped} rows with no lexical text")
    return results


def write_summary(path: Path, train_items, valid_items, sample_rate: int, vocab_size: int):
    counts = Counter(item["lang"] for item in train_items + valid_items)
    payload = {
        "total": len(train_items) + len(valid_items),
        "train": len(train_items),
        "valid": len(valid_items),
        "sample_rate": sample_rate,
        "phoneme_vocab_size": vocab_size,
        "langs": dict(sorted(counts.items())),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    rows = read_metadata(input_dir)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)

    if args.purge and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = process_split("train", train_rows, output_dir, args.sample_rate, args.workers)
    valid_items = process_split("valid", valid_rows, output_dir, args.sample_rate, args.workers)

    symbols = set()
    for item in train_items + valid_items:
        symbols.update(item["symbols"])

    write_filelist(output_dir / "aligned_fused_train.txt", train_items)
    write_filelist(output_dir / "aligned_fused_test.txt", valid_items)
    vocab_size = write_vocab(output_dir / "fused_phoneme_vocab.json", symbols)
    write_summary(output_dir / "summary.json", train_items, valid_items, args.sample_rate, vocab_size)

    print(f"wrote {len(train_items)} train rows")
    print(f"wrote {len(valid_items)} valid rows")
    print(f"wrote phoneme vocab size {vocab_size}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
