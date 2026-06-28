#!/usr/bin/env python3
"""Prepare lzspeech-vctk for Matcha aligned text plus per-utterance prompt embeddings."""

import argparse
import json
import math
import random
import shutil
import unicodedata
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from prepare_matcha_aligned_dataset import build_units, has_lexical_text, normalize_for_alignment


DEFAULT_INPUT_DIR = Path("local/exp/lzspeech-vctk-full")
DEFAULT_OUTPUT_DIR = Path("local/exp/starling/data/lzspeech_vctk_prompt_embeddings_22050")
DEFAULT_PHONEME_VOCAB = Path("local/exp/starling/data/lzspeech_multilingual_plus_22050/fused_phoneme_vocab.json")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_VALID_RATIO = 0.025
DEFAULT_SEED = 1234


@dataclass(frozen=True)
class SourceRow:
    utt_id: str
    speaker: str
    text: str
    audio_path: Path
    embedding_path: Path


def init_worker():
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def read_rows(input_dir: Path):
    rows = []
    metadata_path = input_dir / "metadata.jsonl"
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            utt_id = item["id"]
            speaker = item["speaker"]
            audio_path = input_dir / item["audio_path"]
            embedding_path = input_dir / "spk_emb" / f"{utt_id}.npy"
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio at {metadata_path}:{line_no}: {audio_path}")
            if not embedding_path.exists():
                raise FileNotFoundError(f"Missing prompt embedding at {metadata_path}:{line_no}: {embedding_path}")
            rows.append(SourceRow(utt_id, speaker, item["text"], audio_path, embedding_path))
    if not rows:
        raise ValueError(f"No rows found in {metadata_path}")
    return rows


def split_rows(rows, valid_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    valid_count = max(1, int(round(len(shuffled) * valid_ratio)))
    return shuffled[valid_count:], shuffled[:valid_count]


def load_allowed_symbols(path: Path):
    vocab = json.loads(path.read_text(encoding="utf-8"))
    return {item["symbol"] for item in vocab["symbols"]}


def resample_audio(input_path: Path, output_path: Path, sample_rate: int):
    audio, source_rate = sf.read(input_path, always_2d=True, dtype="float32")
    audio = audio.mean(axis=1)
    if source_rate != sample_rate:
        gcd = math.gcd(source_rate, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, source_rate // gcd).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate, subtype="PCM_16")


def process_row(args):
    row, output_dir, sample_rate, allowed_symbols = args
    raw_text = unicodedata.normalize("NFC", normalize_for_alignment(row.text, "en"))
    if not has_lexical_text(raw_text):
        return None
    if "|" in raw_text:
        raw_text = raw_text.replace("|", " ")

    units = build_units(raw_text, "en")
    phoneme_text = unicodedata.normalize("NFC", " ".join(unit["phonemes"] for unit in units))
    unknown = sorted({symbol for symbol in phoneme_text if symbol != " " and symbol not in allowed_symbols})
    if unknown:
        raise ValueError(f"Unknown phoneme symbols for {row.utt_id}: {unknown}")

    wav_path = output_dir / "wavs" / f"{row.utt_id}.wav"
    resample_audio(row.audio_path, wav_path, sample_rate)
    return {
        "utt_id": row.utt_id,
        "speaker": row.speaker,
        "filepath": str(wav_path.resolve()),
        "embedding_path": str(row.embedding_path.resolve()),
        "raw_text": raw_text,
        "phoneme_text": phoneme_text,
        "units": units,
    }


def process_split(name, rows, output_dir: Path, sample_rate: int, workers: int, allowed_symbols):
    tasks = [(row, output_dir, sample_rate, allowed_symbols) for row in rows]
    results = []
    skipped = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as executor:
        for item in tqdm(executor.map(process_row, tasks, chunksize=16), total=len(tasks), desc=name):
            if item is None:
                skipped += 1
                continue
            results.append(item)
    if skipped:
        print(f"{name}: skipped {skipped} rows with no lexical text")
    return results


def write_filelist(path: Path, items):
    with path.open("w", encoding="utf-8") as file:
        for item in items:
            file.write(
                "|".join(
                    [
                        item["filepath"],
                        "en",
                        item["phoneme_text"],
                        item["raw_text"],
                        json.dumps(item["units"], ensure_ascii=False, separators=(",", ":")),
                        item["embedding_path"],
                    ]
                )
                + "\n"
            )


def write_summary(path: Path, train_items, valid_items, sample_rate: int):
    speakers = Counter(item["speaker"] for item in train_items + valid_items)
    payload = {
        "total": len(train_items) + len(valid_items),
        "train": len(train_items),
        "valid": len(valid_items),
        "sample_rate": sample_rate,
        "langs": {"en": len(train_items) + len(valid_items)},
        "speakers": len(speakers),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phoneme-vocab", type=Path, default=DEFAULT_PHONEME_VOCAB)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    allowed_symbols = load_allowed_symbols(args.phoneme_vocab.resolve())
    rows = read_rows(input_dir)
    if args.limit is not None:
        rows = rows[: args.limit]
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)

    if args.purge and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = process_split("train", train_rows, output_dir, args.sample_rate, args.workers, allowed_symbols)
    valid_items = process_split("valid", valid_rows, output_dir, args.sample_rate, args.workers, allowed_symbols)
    write_filelist(output_dir / "aligned_fused_train.txt", train_items)
    write_filelist(output_dir / "aligned_fused_test.txt", valid_items)
    shutil.copy2(args.phoneme_vocab.resolve(), output_dir / "fused_phoneme_vocab.json")
    write_summary(output_dir / "summary.json", train_items, valid_items, args.sample_rate)
    print(f"wrote {len(train_items)} train rows")
    print(f"wrote {len(valid_items)} valid rows")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
