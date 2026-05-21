#!/usr/bin/env python3
"""Regenerate Matcha fused filelists with the Piper phonemization pipeline."""

import argparse
import json
import shutil
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from src.piper.preprocess import phonemize_text_for_speaker


DEFAULT_MATCHA_DATA_DIR = Path("local/Matcha-TTS/data/lzspeech_multilingual_plus_22050")
DEFAULT_PIPER_CONFIG = Path("data/lzspeech-multilingual-bert/config.json")
DEFAULT_SPLITS = ("fused_train.txt", "fused_test.txt")


def parse_row(line: str):
    parts = line.rstrip("\n").split("|", 3)
    if len(parts) != 4:
        raise ValueError(f"Expected 4 pipe-delimited fields, got {len(parts)}: {line[:120]}")
    filepath, lang, _old_phoneme_text, raw_text = parts
    return filepath, lang, raw_text


def phonemize_text(text: str, lang: str, piper_config: Path):
    result = phonemize_text_for_speaker(text, piper_config, lang)
    phoneme_text = unicodedata.normalize("NFC", "".join(result["phonemes"]))
    if not phoneme_text:
        return None
    if "|" in phoneme_text or "|" in text:
        raise ValueError(f"Pipe character is not supported in fused filelists: lang={lang!r} text={text!r}")
    return phoneme_text


def build_split(split_path: Path, piper_config: Path, on_empty: str):
    rows = []
    skipped = []
    symbols = set()
    counts = Counter()
    identical_raw = Counter()
    examples = defaultdict(list)

    lines = split_path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(tqdm(lines, desc=split_path.name), start=1):
        filepath, lang, raw_text = parse_row(line)
        phoneme_text = phonemize_text(raw_text, lang, piper_config)
        if phoneme_text is None:
            if on_empty == "error":
                raise ValueError(f"Piper produced empty phonemes at {split_path}:{line_number}: {raw_text!r}")
            if on_empty == "skip":
                skipped.append((line_number, lang, filepath, raw_text))
                continue
            phoneme_text = raw_text.strip()
            if not phoneme_text:
                skipped.append((line_number, lang, filepath, raw_text))
                continue

        rows.append((filepath, lang, phoneme_text, raw_text))
        symbols.update(phoneme_text)
        counts[lang] += 1
        if phoneme_text == raw_text:
            identical_raw[lang] += 1
        if len(examples[lang]) < 3:
            examples[lang].append((raw_text, phoneme_text))

    return rows, skipped, symbols, counts, identical_raw, examples


def write_split(path: Path, rows):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for filepath, lang, phoneme_text, raw_text in rows:
            f.write(f"{filepath}|{lang}|{phoneme_text}|{raw_text}\n")
    tmp_path.replace(path)


def write_vocab(path: Path, symbols):
    items = [{"symbol": symbol, "id": index} for index, symbol in enumerate(sorted(symbols), start=1)]
    payload = {"size": len(items) + 1, "symbols": items}
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return payload["size"]


def backup_once(path: Path):
    backup_path = path.with_suffix(path.suffix + ".pre_piper.bak")
    if not backup_path.exists():
        shutil.copy2(path, backup_path)
    return backup_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcha-data-dir", type=Path, default=DEFAULT_MATCHA_DATA_DIR)
    parser.add_argument("--piper-config", type=Path, default=DEFAULT_PIPER_CONFIG)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--on-empty", choices=("error", "skip", "keep-raw"), default="skip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    split_paths = [args.matcha_data_dir / split for split in args.splits]
    vocab_path = args.matcha_data_dir / "fused_phoneme_vocab.json"
    for path in [*split_paths, args.piper_config]:
        if not path.exists():
            raise FileNotFoundError(path)

    all_rows = {}
    all_skipped = {}
    all_symbols = set()
    total_counts = Counter()
    total_identical_raw = Counter()
    all_examples = defaultdict(list)

    for split_path in split_paths:
        rows, skipped, symbols, counts, identical_raw, examples = build_split(split_path, args.piper_config, args.on_empty)
        all_rows[split_path] = rows
        all_skipped[split_path] = skipped
        all_symbols.update(symbols)
        total_counts.update(counts)
        total_identical_raw.update(identical_raw)
        for lang, lang_examples in examples.items():
            all_examples[lang].extend(lang_examples)

    print("\nKept rows:")
    for lang, count in sorted(total_counts.items()):
        print(f"  {lang}: {count}")

    skipped_count = sum(len(items) for items in all_skipped.values())
    print(f"\nSkipped empty-phoneme rows: {skipped_count}")
    for split_path, skipped in all_skipped.items():
        for line_number, lang, filepath, raw_text in skipped[:20]:
            print(f"  {split_path.name}:{line_number} {lang} {Path(filepath).name}: {raw_text!r}")

    if total_identical_raw:
        print("\nIdentical phoneme/raw rows:")
        for lang, count in sorted(total_identical_raw.items()):
            print(f"  {lang}: {count}/{total_counts[lang]}")
    else:
        print("\nIdentical phoneme/raw rows: none")

    print("\nExamples:")
    for lang in sorted(all_examples):
        raw_text, phoneme_text = all_examples[lang][0]
        print(f"  {lang}: {raw_text}")
        print(f"      -> {phoneme_text}")

    vocab_size = len(all_symbols) + 1
    print(f"\nPiper phoneme vocab size including blank: {vocab_size}")

    if args.dry_run:
        print("Dry run: no files written.")
        return

    for split_path in split_paths:
        backup_path = backup_once(split_path)
        print(f"Backup: {backup_path}")
        write_split(split_path, all_rows[split_path])
        print(f"Wrote: {split_path}")

    backup_path = backup_once(vocab_path)
    print(f"Backup: {backup_path}")
    written_vocab_size = write_vocab(vocab_path, all_symbols)
    print(f"Wrote: {vocab_path} (size={written_vocab_size})")


if __name__ == "__main__":
    main()
