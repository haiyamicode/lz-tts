#!/usr/bin/env python3
"""Prepare the original English lzspeech dataset for fresh Matcha/Vocos training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from scripts.prepare_matcha_aligned_dataset import (
    SourceRow,
    process_split,
    write_filelist,
    write_vocab,
)


def read_split(input_dir: Path, split: str) -> list[SourceRow]:
    metadata_path = input_dir / split / "metadata.csv"
    wav_dir = input_dir / split / "wavs"
    rows: list[SourceRow] = []
    with metadata_path.open("r", encoding="utf-8") as metadata:
        for line_no, line in enumerate(metadata, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                raise ValueError(f"Expected id|text at {metadata_path}:{line_no}: {line[:200]!r}")
            utt_id, text = parts
            audio_path = wav_dir / f"{utt_id}.wav"
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing audio for {split}/{utt_id}: {audio_path}")
            rows.append(SourceRow(f"{split}_{utt_id}", "en", text, audio_path))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Projects/gen-tts/data/lzspeech")
    parser.add_argument("--output-dir", type=Path, default=Path("local/Matcha-TTS/data/lzspeech_en_vocos24k_24000"))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    if args.purge and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_split(input_dir, "train")
    valid_rows = read_split(input_dir, "test")
    train_items = process_split("train", train_rows, output_dir, args.sample_rate, args.workers)
    valid_items = process_split("valid", valid_rows, output_dir, args.sample_rate, args.workers)

    write_filelist(output_dir / "aligned_fused_train.txt", train_items)
    write_filelist(output_dir / "aligned_fused_test.txt", valid_items)

    symbols = set()
    for item in train_items + valid_items:
        symbols.update(item["symbols"])
    vocab_size = write_vocab(output_dir / "fused_phoneme_vocab.json", symbols)

    summary = {
        "total": len(train_items) + len(valid_items),
        "train": len(train_items),
        "valid": len(valid_items),
        "sample_rate": args.sample_rate,
        "phoneme_vocab_size": vocab_size,
        "langs": {"en": len(train_items) + len(valid_items)},
        "source_dir": str(input_dir),
        "icbpe_vocab_path": "data/flores200/tokenizers/unicode_codepoint_byte_fallback_10000/vocab.json",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {len(train_items)} train rows")
    print(f"wrote {len(valid_items)} valid rows")
    print(f"wrote phoneme vocab size {vocab_size}")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
