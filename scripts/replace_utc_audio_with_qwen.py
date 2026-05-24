#!/usr/bin/env python3
"""Replace source dataset rows containing UTC with Qwen-generated audio."""

import argparse
import base64
import json
import re
import shutil
import time
from pathlib import Path

import requests
import soundfile as sf


DEFAULT_DATASET = Path("local/datasets/lzspeech-multilingual-plus")
DEFAULT_OUTPUT_DIR = Path("output/qwen_utc_audio_replacements")
DEFAULT_SERVER_URL = "http://127.0.0.1:7860/generate"
DEFAULT_REF_ID = "EN_000001"
DEFAULT_REF_TEXT = (
    "Good morning! Did you see the news about Dr. Martinez's conference yesterday? "
    "It was held in Madrid, Spain, at 10:30 a.m. on March 15th."
)


def find_utc_rows(metadata_path: Path):
    rows = []
    for line_no, line in enumerate(metadata_path.read_text(encoding="utf-8").splitlines(), 1):
        if not re.search(r"\bUTC\b", line, re.IGNORECASE):
            continue
        utt_id, lang, text = line.split("|", 2)
        rows.append({"line_no": line_no, "utt_id": utt_id, "lang": lang, "text": text})
    return rows


def write_manifest(path: Path, records):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=DEFAULT_SERVER_URL)
    parser.add_argument("--ref-id", default=DEFAULT_REF_ID)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--max-new-tokens", type=int, default=420)
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset = args.dataset
    metadata_path = dataset / "metadata.csv"
    wav_dir = dataset / "wav"
    backup_dir = dataset / ".backup_before_qwen_utc_audio"
    generated_dir = args.output_dir / "generated_wavs"
    manifest_path = args.output_dir / "manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    backup_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)

    rows = find_utc_rows(metadata_path)
    if args.limit:
        rows = rows[: args.limit]

    ref_audio = wav_dir / f"{args.ref_id}.wav"
    if not ref_audio.exists():
        raise FileNotFoundError(f"Missing reference audio: {ref_audio}")

    existing_records = []
    done = set()
    if manifest_path.exists() and not args.force:
        existing_records = json.loads(manifest_path.read_text(encoding="utf-8"))
        done = {record["utt_id"] for record in existing_records if record.get("status") == "replaced"}

    records = list(existing_records)
    for index, row in enumerate(rows, 1):
        utt_id = row["utt_id"]
        if utt_id in done:
            print(f"[{index}/{len(rows)}] skip {utt_id}: already replaced", flush=True)
            continue

        source_wav = wav_dir / f"{utt_id}.wav"
        backup_wav = backup_dir / f"{utt_id}.wav"
        generated_wav = generated_dir / f"{utt_id}.wav"
        if not source_wav.exists():
            raise FileNotFoundError(f"Missing source wav for {utt_id}: {source_wav}")
        if not backup_wav.exists():
            shutil.copy2(source_wav, backup_wav)

        spoken_text = row["text"]
        print(f"[{index}/{len(rows)}] generating {utt_id}", flush=True)
        started = time.time()
        with ref_audio.open("rb") as ref_file:
            response = requests.post(
                args.server_url,
                data={
                    "text": spoken_text,
                    "language": "English",
                    "mode": "voice_clone",
                    "ref_text": args.ref_text,
                    "temperature": str(args.temperature),
                    "top_k": str(args.top_k),
                    "repetition_penalty": str(args.repetition_penalty),
                    "max_new_tokens": str(args.max_new_tokens),
                    "non_streaming_mode": "true",
                    "use_dp_budget": "false",
                    "use_postprocess": "true" if args.use_postprocess else "false",
                },
                files={"ref_audio": (ref_audio.name, ref_file, "audio/wav")},
                timeout=900,
            )
        response.raise_for_status()
        payload = response.json()
        generated_wav.write_bytes(base64.b64decode(payload["audio_b64"]))

        info = sf.info(generated_wav)
        if info.samplerate != 24000 or info.channels != 1:
            raise RuntimeError(f"Unexpected generated wav format for {utt_id}: {info}")

        shutil.copy2(generated_wav, source_wav)
        record = {
            **row,
            "spoken_text": spoken_text,
            "source_wav": str(source_wav.resolve()),
            "backup_wav": str(backup_wav.resolve()),
            "generated_wav": str(generated_wav.resolve()),
            "sample_rate": info.samplerate,
            "duration_seconds": round(info.frames / info.samplerate, 3),
            "elapsed_seconds": round(time.time() - started, 3),
            "metrics": payload.get("metrics"),
            "status": "replaced",
        }
        records.append(record)
        write_manifest(manifest_path, records)

    rows_path = args.output_dir / "utc_rows.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"done rows={len(rows)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
