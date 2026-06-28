#!/usr/bin/env python3
"""Build a Matcha codepoint byte-fallback vocab from FLORES-200 text."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def iter_flores_texts(flores_dir: Path):
    for split in ("dev", "devtest"):
        split_dir = flores_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing FLORES split directory: {split_dir}")
        for path in sorted(split_dir.iterdir()):
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    text = line.rstrip("\n")
                    if text:
                        yield text


def build_vocab(flores_dir: Path, output_path: Path, max_size: int) -> dict:
    if max_size <= 257:
        raise ValueError("max_size must leave room for byte fallback IDs 1..256 plus codepoints")

    counts: Counter[str] = Counter()
    lines = 0
    for text in iter_flores_texts(flores_dir):
        counts.update(text)
        lines += 1

    max_codepoints = max_size - 257
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_codepoints]
    codepoints = [
        {"char": char, "id": index, "count": count}
        for index, (char, count) in enumerate(ranked, start=257)
    ]
    payload = {
        "source": str(flores_dir),
        "lines": lines,
        "unique_codepoints": len(counts),
        "max_size": max_size,
        "size": (codepoints[-1]["id"] + 1) if codepoints else 257,
        "byte_fallback_offset": 1,
        "reserved_byte_fallback_ids": [1, 256],
        "codepoints": codepoints,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--flores-dir",
        type=Path,
        default=Path("local/exp/starling/data/flores200/flores200_dataset"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("local/exp/starling/data/flores200/tokenizers/unicode_codepoint_byte_fallback_10000/vocab.json"),
    )
    parser.add_argument("--max-size", type=int, default=10000)
    args = parser.parse_args()

    payload = build_vocab(args.flores_dir.resolve(), args.output.resolve(), args.max_size)
    print(f"wrote {args.output}")
    print(f"lines: {payload['lines']}")
    print(f"unique_codepoints: {payload['unique_codepoints']}")
    print(f"vocab_size: {payload['size']}")
    print(f"codepoints: {len(payload['codepoints'])}")


if __name__ == "__main__":
    main()
