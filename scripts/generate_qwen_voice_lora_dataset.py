#!/usr/bin/env python3
"""Generate a small voice-clone dataset for Matcha voice-LoRA experiments."""

import argparse
import base64
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import requests
import soundfile as sf


DEFAULT_INPUT_DIR = Path("local/datasets/lzspeech-multilingual-plus")
DEFAULT_OUTPUT_DIR = Path("local/datasets/andrew-qwen-lora-v1")
DEFAULT_SERVER_URL = "http://127.0.0.1:7860/generate"
DEFAULT_REF_AUDIO = Path("local/samples/andrew.mp3")
DEFAULT_REF_TEXT = (
    "After decades of serving big businesses with large budgets, we realized that smaller businesses "
    "were underserved and in desperate need of marketing services that perform."
)
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "jv": "Javanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}
QWEN_SUPPORTED_LANGS = ("de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh")


@dataclass(frozen=True)
class Row:
    utt_id: str
    lang: str
    text: str
    wav_path: Path
    duration: float


def read_rows(input_dir: Path, min_seconds: float, max_seconds: float, max_chars: int, langs: set[str] | None):
    rows = []
    metadata_path = input_dir / "metadata.csv"
    wav_dir = input_dir / "wav"
    for line_no, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError(f"Bad metadata row at {metadata_path}:{line_no}: {line[:200]!r}")
        utt_id, lang, text = parts
        if langs is not None and lang not in langs:
            continue
        if len(text) > max_chars:
            continue
        wav_path = wav_dir / f"{utt_id}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio for {utt_id}: {wav_path}")
        info = sf.info(wav_path)
        duration = info.frames / info.samplerate
        if min_seconds <= duration <= max_seconds:
            rows.append(Row(utt_id, lang, text, wav_path, duration))
    return rows


def select_balanced(rows, target_seconds: float, seed: int):
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for row in rows:
        by_lang[row.lang].append(row)
    for lang_rows in by_lang.values():
        rng.shuffle(lang_rows)

    langs = sorted(by_lang)
    per_lang_target = target_seconds / max(len(langs), 1)
    selected = []
    durations = defaultdict(float)
    made_progress = True
    while made_progress and sum(row.duration for row in selected) < target_seconds:
        made_progress = False
        for lang in langs:
            if durations[lang] >= per_lang_target and sum(row.duration for row in selected) < target_seconds:
                continue
            while by_lang[lang]:
                row = by_lang[lang].pop()
                if row not in selected:
                    selected.append(row)
                    durations[lang] += row.duration
                    made_progress = True
                    break
    return selected


def write_metadata(output_dir: Path, records):
    metadata_path = output_dir / "metadata.csv"
    tmp_path = metadata_path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(f"{record['utt_id']}|{record['lang']}|{record['text']}\n")
    tmp_path.replace(metadata_path)


def write_json(path: Path, payload):
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--ref-audio", type=Path, default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--target-minutes", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--max-seconds", type=float, default=9.0)
    parser.add_argument("--max-chars", type=int, default=260)
    parser.add_argument("--langs", default=",".join(QWEN_SUPPORTED_LANGS))
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Keep existing generated manifest rows and append newly selected rows.",
    )
    parser.add_argument(
        "--exclude-done-from-selection",
        action="store_true",
        help="Do not count existing generated ids toward the requested target duration.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    selected_path = output_dir / "selected_rows.json"
    done_records = []
    done_ids = set()
    if manifest_path.exists() and not args.force:
        done_records = json.loads(manifest_path.read_text(encoding="utf-8"))
        done_ids = {record["utt_id"] for record in done_records if record.get("status") == "generated"}

    langs = {lang.strip() for lang in args.langs.split(",") if lang.strip()}
    rows = read_rows(input_dir, args.min_seconds, args.max_seconds, args.max_chars, langs)
    if not rows:
        raise ValueError(f"No candidate rows found for langs={sorted(langs)}")
    if args.exclude_done_from_selection:
        rows = [row for row in rows if row.utt_id not in done_ids]
        if not rows:
            raise ValueError(f"No new candidate rows remain after excluding {len(done_ids)} done ids")
    selected = select_balanced(rows, args.target_minutes * 60.0, args.seed)
    selected_records = [
        {
            "utt_id": row.utt_id,
            "lang": row.lang,
            "text": row.text,
            "source_wav": str(row.wav_path),
            "source_duration_seconds": round(row.duration, 3),
        }
        for row in selected
    ]
    write_json(selected_path, selected_records)

    records_by_id = {record["utt_id"]: record for record in done_records}
    for index, row in enumerate(selected, 1):
        if row.utt_id in done_ids and (wav_dir / f"{row.utt_id}.wav").exists():
            print(f"[{index}/{len(selected)}] skip {row.utt_id}", flush=True)
            continue

        out_wav = wav_dir / f"{row.utt_id}.wav"
        language = LANGUAGE_NAMES.get(row.lang, "Auto")
        print(f"[{index}/{len(selected)}] generate {row.utt_id} lang={row.lang} dur={row.duration:.2f}s", flush=True)
        started = time.time()
        with args.ref_audio.open("rb") as ref_file:
            response = requests.post(
                args.server_url,
                data={
                    "text": row.text,
                    "language": language,
                    "mode": "voice_clone",
                    "ref_text": args.ref_text,
                    "temperature": str(args.temperature),
                    "top_k": str(args.top_k),
                    "repetition_penalty": str(args.repetition_penalty),
                    "non_streaming_mode": "true",
                    "use_dp_budget": "false",
                    "use_postprocess": "true" if args.use_postprocess else "false",
                },
                files={"ref_audio": (args.ref_audio.name, ref_file, "audio/mpeg")},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        out_wav.write_bytes(base64.b64decode(payload["audio_b64"]))
        info = sf.info(out_wav)
        record = {
            "utt_id": row.utt_id,
            "lang": row.lang,
            "text": row.text,
            "source_wav": str(row.wav_path),
            "generated_wav": str(out_wav),
            "source_duration_seconds": round(row.duration, 3),
            "duration_seconds": round(info.frames / info.samplerate, 3),
            "sample_rate": info.samplerate,
            "elapsed_seconds": round(time.time() - started, 3),
            "metrics": payload.get("metrics"),
            "status": "generated",
        }
        records_by_id[row.utt_id] = record
        ordered_records = []
        if args.append_existing:
            ordered_records.extend(done_records)
        ordered_records.extend(
            records_by_id[item.utt_id]
            for item in selected
            if item.utt_id in records_by_id and item.utt_id not in {record["utt_id"] for record in ordered_records}
        )
        write_json(manifest_path, ordered_records)
        write_metadata(output_dir, ordered_records)

    final_records = []
    if args.append_existing:
        final_records.extend(done_records)
    final_records.extend(
        records_by_id[item.utt_id]
        for item in selected
        if item.utt_id in records_by_id and item.utt_id not in {record["utt_id"] for record in final_records}
    )
    write_json(manifest_path, final_records)
    write_metadata(output_dir, final_records)
    total_seconds = sum(record["duration_seconds"] for record in final_records if record.get("status") == "generated")
    print(
        f"done generated={len(final_records)}/{len(selected)} audio_minutes={total_seconds / 60.0:.2f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
