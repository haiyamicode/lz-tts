#!/usr/bin/env python3
"""Create a 24 kHz Matcha filelist dataset from an existing aligned dataset."""

import argparse
import json
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


def read_filelist(path: Path) -> list[list[str]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if line:
                rows.append(line.split("|"))
    return rows


def resample_audio(input_path: Path, output_path: Path, sample_rate: int) -> None:
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


def convert_row(args):
    row, output_dir, sample_rate = args
    source = Path(row[0])
    target = output_dir / "wavs" / source.name
    resample_audio(source, target, sample_rate)
    converted = [str(target.resolve()), *row[1:]]
    return converted


def write_filelist(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write("|".join(row) + "\n")


def count_langs(rows: list[list[str]]) -> dict[str, int]:
    counts = {}
    for row in rows:
        if len(row) > 1:
            counts[row[1]] = counts.get(row[1], 0) + 1
    return dict(sorted(counts.items()))


def convert_split(name: str, rows: list[list[str]], output_dir: Path, sample_rate: int, workers: int) -> list[list[str]]:
    tasks = [(row, output_dir, sample_rate) for row in rows]
    converted = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row in tqdm(executor.map(convert_row, tasks, chunksize=16), total=len(tasks), desc=name):
            converted.append(row)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("local/Matcha-TTS/data/lzspeech_multilingual_plus_22050"))
    parser.add_argument("--output-dir", type=Path, default=Path("local/Matcha-TTS/data/lzspeech_multilingual_plus_bigvgan24k_24000"))
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.purge and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_filelist(input_dir / "aligned_fused_train.txt")
    valid_rows = read_filelist(input_dir / "aligned_fused_test.txt")
    converted_train = convert_split("train", train_rows, output_dir, args.sample_rate, args.workers)
    converted_valid = convert_split("valid", valid_rows, output_dir, args.sample_rate, args.workers)

    write_filelist(output_dir / "aligned_fused_train.txt", converted_train)
    write_filelist(output_dir / "aligned_fused_test.txt", converted_valid)
    shutil.copy2(input_dir / "fused_phoneme_vocab.json", output_dir / "fused_phoneme_vocab.json")
    summary = {
        "total": len(converted_train) + len(converted_valid),
        "train": len(converted_train),
        "valid": len(converted_valid),
        "sample_rate": args.sample_rate,
        "langs": count_langs(converted_train + converted_valid),
        "source_dir": str(input_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(converted_train)} train rows")
    print(f"wrote {len(converted_valid)} valid rows")
    print(f"output: {output_dir}")


if __name__ == "__main__":
    main()
