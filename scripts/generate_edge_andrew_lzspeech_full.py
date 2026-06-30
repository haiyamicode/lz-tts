#!/usr/bin/env python3
import argparse
import asyncio
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from dotenv import load_dotenv


DEFAULT_INPUT_DIR = Path("/home/haiyami/Projects/gen-tts/data/lzspeech")
DEFAULT_OUTPUT_DIR = Path("local/datasets/andrew-azure-andrew-multilingual-en-48k-lzspeech-full")
DEFAULT_BACKEND = "azure"
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
AZURE_48K_WAV_FORMAT = "Riff48Khz16BitMonoPcm"

METADATA_SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("speaker", pa.string()),
        ("text", pa.string()),
    ]
)


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


async def generate_wav_edge(text: str, voice: str, output_wav: Path, sample_rate: int):
    import edge_tts  # pylint: disable=import-outside-toplevel

    communicate = edge_tts.Communicate(text=text, voice=voice)
    audio = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])

    if not audio:
        raise RuntimeError("Edge synthesis returned no audio")

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            "pipe:0",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-sample_fmt",
            "s16",
            str(output_wav),
        ],
        input=bytes(audio),
        check=True,
    )


def voice_candidates(requested_voice: str) -> list[str]:
    requested = requested_voice.strip()
    if not requested:
        raise ValueError("Voice name cannot be empty")

    candidates = [requested]
    if "." in requested:
        alias = requested.replace(".", "-", 1)
        candidates.append(alias)
        if not alias.endswith("Neural"):
            candidates.append(f"{alias}Neural")
    elif not requested.endswith("Neural"):
        candidates.append(f"{requested}Neural")
    return candidates


def resolve_azure_voice_name(requested_voice: str) -> str:
    return voice_candidates(requested_voice)[-1]


def load_azure_config(token_env: str, region_env: str, region: str | None) -> tuple[str, str]:
    azure_key = os.getenv(token_env) or os.getenv("AZURE_TTS_TOKEN")
    azure_region = region or os.getenv(region_env) or os.getenv("AZURE_TTS_REGION") or os.getenv("SPEECH_REGION") or "eastus"
    if not azure_key:
        raise ValueError(f"Missing Azure TTS key. Set {token_env} or AZURE_TTS_TOKEN in .env.")
    return azure_key, azure_region


def _azure_output_format(format_name: str):
    import azure.cognitiveservices.speech as speechsdk  # pylint: disable=import-outside-toplevel

    try:
        return getattr(speechsdk.SpeechSynthesisOutputFormat, format_name)
    except AttributeError as exc:
        supported = sorted(name for name in dir(speechsdk.SpeechSynthesisOutputFormat) if "48Khz" in name or "24Khz" in name)
        raise ValueError(f"Unsupported Azure output format {format_name!r}; supported examples: {supported}") from exc


def generate_wav_azure_sync(
    text: str,
    voice: str,
    output_wav: Path,
    azure_key: str,
    azure_region: str,
    output_format: str,
):
    import azure.cognitiveservices.speech as speechsdk  # pylint: disable=import-outside-toplevel

    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(_azure_output_format(output_format))
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_wav))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure synthesis canceled: {details.reason} {details.error_details}")
    raise RuntimeError(f"Azure synthesis failed: {result.reason}")


def wav_duration(path: Path) -> tuple[float, int]:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate), int(info.samplerate)


def load_existing_records(manifest_path: Path) -> list[dict]:
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return list(payload.get("records", []))


def write_outputs(output_dir: Path, records: list[dict], settings: dict):
    metadata_rows = [
        {
            "filename": f"wav/{record['utt_id']}.wav",
            "speaker": "en",
            "text": record["text"],
        }
        for record in records
        if record.get("status") == "generated"
    ]
    pq.write_table(pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA), output_dir / "metadata.parquet")

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="|", lineterminator="\n")
        for record in records:
            if record.get("status") == "generated":
                writer.writerow([f"wav/{record['utt_id']}.wav", "en", record["text"]])

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
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("edge", "azure"), default=DEFAULT_BACKEND)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent synthesis workers.")
    parser.add_argument("--max-failures", type=int, default=50)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke limit. 0 means all rows.")
    parser.add_argument("--azure-token-env", default="AZURE_TTS_TOKEN")
    parser.add_argument("--azure-region-env", default="AZURE_TTS_REGION")
    parser.add_argument("--azure-region", default=None, help="Azure Speech region value, default from env or eastus.")
    parser.add_argument("--azure-output-format", default=AZURE_48K_WAV_FORMAT)
    args = parser.parse_args()

    output_dir = args.output_dir
    wav_dir = output_dir / "wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_source_rows(args.input_dir)
    if args.max_rows > 0:
        source_rows = source_rows[: args.max_rows]
    manifest_path = output_dir / "manifest.json"
    records = load_existing_records(manifest_path)
    done_ids = {record["utt_id"] for record in records if record.get("status") == "generated"}
    total_duration = sum(record.get("duration_seconds", 0.0) for record in records if record.get("status") == "generated")
    failures = sum(1 for record in records if record.get("status") == "failed")

    if args.backend == "azure":
        resolved_voice = resolve_azure_voice_name(args.voice)
        azure_key, azure_region = load_azure_config(args.azure_token_env, args.azure_region_env, args.azure_region)
    else:
        resolved_voice = args.voice
        azure_key = None
        azure_region = None

    if resolved_voice != args.voice:
        print(f"resolved voice: {args.voice} -> {resolved_voice}", flush=True)

    settings = {
        "backend": args.backend,
        "input_dir": str(args.input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "voice": resolved_voice,
        "azure_region": azure_region if args.backend == "azure" else None,
        "azure_output_format": args.azure_output_format if args.backend == "azure" else None,
        "sample_rate": args.sample_rate,
        "rate_limit_seconds": args.rate_limit_seconds,
        "workers": args.workers,
        "max_rows": args.max_rows,
    }

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    last_request_start = 0.0
    rate_lock = asyncio.Lock()
    records_lock = asyncio.Lock()
    queue: asyncio.Queue[tuple[int, SourceRow] | None] = asyncio.Queue()
    stop_event = asyncio.Event()
    stopped_for_failures = False

    async def wait_for_rate_slot():
        nonlocal last_request_start
        if args.rate_limit_seconds <= 0:
            return
        async with rate_lock:
            wait = args.rate_limit_seconds - (time.monotonic() - last_request_start)
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_start = time.monotonic()

    async def generate_row(worker_id: int, index: int, row: SourceRow) -> tuple[dict, float, bool]:
        await wait_for_rate_slot()

        wav_path = wav_dir / f"{row.utt_id}.wav"
        started = time.time()
        async with records_lock:
            print(
                f"[worker {worker_id} {index}/{len(source_rows)}] {row.utt_id} "
                f"done={len(done_ids)} total={total_duration / 60:.2f}m text={row.text[:80]}",
                flush=True,
            )
        try:
            if args.backend == "azure":
                await asyncio.to_thread(
                    generate_wav_azure_sync,
                    row.text,
                    resolved_voice,
                    wav_path,
                    azure_key,
                    azure_region,
                    args.azure_output_format,
                )
            else:
                await generate_wav_edge(row.text, resolved_voice, wav_path, args.sample_rate)
            duration, sample_rate = wav_duration(wav_path)
            if sample_rate != args.sample_rate:
                raise RuntimeError(f"Expected {args.sample_rate} Hz audio, got {sample_rate} Hz from {wav_path}")
            record = {
                "utt_id": row.utt_id,
                "split": row.split,
                "source_id": row.source_id,
                "text": row.text,
                "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                "generated_wav": str(wav_path.resolve()),
                "duration_seconds": round(duration, 3),
                "sample_rate": sample_rate,
                "voice": resolved_voice,
                "elapsed_seconds": round(time.time() - started, 3),
                "worker": worker_id,
                "status": "generated",
            }
            return record, duration, True
        except Exception as exc:  # noqa: BLE001
            if wav_path.exists():
                wav_path.unlink()
            record = {
                "utt_id": row.utt_id,
                "split": row.split,
                "source_id": row.source_id,
                "text": row.text,
                "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                "voice": resolved_voice,
                "elapsed_seconds": round(time.time() - started, 3),
                "worker": worker_id,
                "error": repr(exc),
                "status": "failed",
            }
            print(f"FAILED {row.utt_id}: {exc!r}", flush=True)
            return record, 0.0, False

    async def worker_loop(worker_id: int):
        nonlocal failures, stopped_for_failures, total_duration
        while True:
            item = await queue.get()
            try:
                if item is None:
                    return
                if stop_event.is_set():
                    continue
                index, row = item
                record, duration, success = await generate_row(worker_id, index, row)
                async with records_lock:
                    records.append(record)
                    if success:
                        total_duration += duration
                        done_ids.add(row.utt_id)
                    else:
                        failures += 1
                    write_outputs(output_dir, records, settings)
                    if failures >= args.max_failures:
                        stopped_for_failures = True
                        stop_event.set()
                        print(f"Stopping after {failures} failures", flush=True)
            finally:
                queue.task_done()

    try:
        pending_count = 0
        for index, row in enumerate(source_rows, start=1):
            if row.utt_id in done_ids:
                continue
            queue.put_nowait((index, row))
            pending_count += 1

        worker_count = min(args.workers, pending_count) if pending_count else 0
        print(
            f"starting {worker_count} worker(s); pending={pending_count} generated={len(done_ids)} failed={failures}",
            flush=True,
        )
        workers = [asyncio.create_task(worker_loop(worker_id)) for worker_id in range(1, worker_count + 1)]
        for _ in workers:
            queue.put_nowait(None)
        await queue.join()
        for task in workers:
            await task
        if stopped_for_failures:
            raise RuntimeError(f"Stopping after {failures} failures")
    finally:
        write_outputs(output_dir, records, settings)


if __name__ == "__main__":
    asyncio.run(main())
