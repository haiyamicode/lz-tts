#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf


DEFAULT_INPUT_DIR = Path("/home/haiyami/Projects/gen-tts/data/lzspeech")
DEFAULT_OUTPUT_DIR = Path("local/datasets/andrew-edge-tts-lzspeech-full")
DEFAULT_VOICE = "en-US-AndrewNeural"


@dataclass(frozen=True)
class SourceRow:
    utt_id: str
    split: str
    source_id: str
    text: str
    source_wav: Path | None


def read_split(input_dir: Path, split: str) -> list[SourceRow]:
    metadata_path = input_dir / split / "metadata.csv"
    wav_dir = input_dir / split / "wavs"
    rows: list[SourceRow] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if "|" not in line:
                continue
            source_id, text = line.split("|", 1)
            source_id = source_id.strip()
            text = text.strip()
            if not source_id or not text:
                continue
            source_wav = wav_dir / f"{source_id}.wav"
            rows.append(
                SourceRow(
                    utt_id=f"{split}_{int(source_id):06d}" if source_id.isdigit() else f"{split}_{source_id}",
                    split=split,
                    source_id=source_id,
                    text=text,
                    source_wav=source_wav if source_wav.exists() else None,
                )
            )
    return rows


def read_source_rows(input_dir: Path) -> list[SourceRow]:
    rows = read_split(input_dir, "train") + read_split(input_dir, "test")
    if not rows:
        raise ValueError(f"No rows found under {input_dir}")
    return rows


async def generate_mp3(text: str, voice: str, output_mp3: Path):
    import edge_tts  # pylint: disable=import-outside-toplevel

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_mp3))


def convert_to_wav(input_mp3: Path, output_wav: Path, sample_rate: int):
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_mp3),
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            str(output_wav),
        ],
        check=True,
    )


def wav_duration(path: Path) -> tuple[float, int]:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


def load_existing_records(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def write_outputs(output_dir: Path, records: list[dict], settings: dict):
    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="|", lineterminator="\n")
        for record in records:
            if record.get("status") == "generated":
                writer.writerow([record["utt_id"], "en", record["text"]])

    total_duration = sum(record.get("duration_seconds", 0.0) for record in records if record.get("status") == "generated")
    payload = {
        "settings": settings,
        "summary": {
            "generated": sum(1 for record in records if record.get("status") == "generated"),
            "failed": sum(1 for record in records if record.get("status") == "failed"),
            "duration_seconds": round(total_duration, 3),
            "duration_minutes": round(total_duration / 60.0, 3),
        },
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--max-failures", type=int, default=50)
    args = parser.parse_args()

    output_dir = args.output_dir
    wav_dir = output_dir / "wav"
    mp3_dir = output_dir / "mp3"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_source_rows(args.input_dir)
    manifest_path = output_dir / "manifest.json"
    records = load_existing_records(manifest_path)
    done_ids = {record["utt_id"] for record in records if record.get("status") == "generated"}
    total_duration = sum(record.get("duration_seconds", 0.0) for record in records if record.get("status") == "generated")
    failures = sum(1 for record in records if record.get("status") == "failed")
    settings = {
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "voice": args.voice,
        "sample_rate": args.sample_rate,
        "rate_limit_seconds": args.rate_limit_seconds,
    }

    last_request_start = 0.0
    try:
        for index, row in enumerate(source_rows, start=1):
            if row.utt_id in done_ids:
                continue

            wait = args.rate_limit_seconds - (time.monotonic() - last_request_start)
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_start = time.monotonic()

            mp3_path = mp3_dir / f"{row.utt_id}.mp3"
            wav_path = wav_dir / f"{row.utt_id}.wav"
            started = time.time()
            print(
                f"[{index}/{len(source_rows)}] {row.utt_id} done={len(done_ids)} total={total_duration / 60:.2f}m text={row.text[:80]}",
                flush=True,
            )
            try:
                await generate_mp3(row.text, args.voice, mp3_path)
                convert_to_wav(mp3_path, wav_path, args.sample_rate)
                duration, sample_rate = wav_duration(wav_path)
                record = {
                    "utt_id": row.utt_id,
                    "split": row.split,
                    "source_id": row.source_id,
                    "text": row.text,
                    "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                    "generated_mp3": str(mp3_path.resolve()),
                    "generated_wav": str(wav_path.resolve()),
                    "duration_seconds": round(duration, 3),
                    "sample_rate": sample_rate,
                    "voice": args.voice,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "generated",
                }
                total_duration += duration
                done_ids.add(row.utt_id)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if wav_path.exists():
                    wav_path.unlink()
                if mp3_path.exists():
                    mp3_path.unlink()
                record = {
                    "utt_id": row.utt_id,
                    "split": row.split,
                    "source_id": row.source_id,
                    "text": row.text,
                    "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                    "voice": args.voice,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error": repr(exc),
                    "status": "failed",
                }
                print(f"FAILED {row.utt_id}: {exc!r}", flush=True)
                if failures >= args.max_failures:
                    records.append(record)
                    raise RuntimeError(f"Stopping after {failures} failures") from exc
            records.append(record)
            write_outputs(output_dir, records, settings)
    finally:
        write_outputs(output_dir, records, settings)


if __name__ == "__main__":
    asyncio.run(main())
