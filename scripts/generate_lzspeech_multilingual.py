#!/usr/bin/env python3
"""Generate multilingual LZSpeech rows with Edge TTS or Azure Speech."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from scripts.generate_edge_andrew_en_dataset import convert_to_wav, load_existing_records, wav_duration


DEFAULT_INPUT_METADATA = Path("local/datasets/lzspeech-multilingual-plus/metadata.csv")
DEFAULT_OUTPUT_DIR = Path("local/datasets/lzspeech-multilingual-tts-full")
DEFAULT_BACKEND = "edge"
DEFAULT_VOICE = "en-US-AndrewMultilingualNeural"
DEFAULT_SKIP_LANGS = "en"
METADATA_SCHEMA = pa.schema(
    [
        ("utt_id", pa.string()),
        ("lang", pa.string()),
        ("text", pa.string()),
        ("wav_path", pa.string()),
        ("source_wav", pa.string()),
        ("duration_seconds", pa.float64()),
        ("sample_rate", pa.int64()),
        ("voice", pa.string()),
    ]
)


@dataclass(frozen=True)
class SourceRow:
    utt_id: str
    lang: str
    text: str
    source_wav: Path | None


def clean_text(text: str) -> str:
    return " ".join(text.replace("|", " ").split())


def is_usable_text(text: str) -> bool:
    stripped = clean_text(text)
    if len(stripped) < 2 or len(stripped) > 300:
        return False
    return any(ch.isalpha() for ch in stripped)


def parse_langs(value: str) -> set[str]:
    return {item.strip().lower() for item in value.split(",") if item.strip()}


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


async def resolve_edge_voice_name(requested_voice: str) -> str:
    import edge_tts  # pylint: disable=import-outside-toplevel

    voices = await edge_tts.list_voices()
    by_short = {str(voice["ShortName"]).casefold(): str(voice["ShortName"]) for voice in voices}
    by_name = {str(voice["Name"]).casefold(): str(voice["ShortName"]) for voice in voices}
    by_friendly = {str(voice["FriendlyName"]).casefold(): str(voice["ShortName"]) for voice in voices}

    for candidate in voice_candidates(requested_voice):
        resolved = by_short.get(candidate.casefold())
        if resolved:
            return resolved

    requested = requested_voice.strip()
    resolved = by_name.get(requested.casefold()) or by_friendly.get(requested.casefold())
    if resolved:
        return resolved

    raise ValueError(f"Invalid Edge voice {requested_voice!r}")


def resolve_azure_voice_name(requested_voice: str) -> str:
    return voice_candidates(requested_voice)[-1]


def read_source_rows(metadata_path: Path, source_wav_dir: Path | None, include_langs: set[str], skip_langs: set[str]) -> list[SourceRow]:
    rows: list[SourceRow] = []
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, 1):
            parts = line.rstrip("\n").split("|", 2)
            if len(parts) != 3:
                print(f"Skipping malformed metadata line {line_no}: {line[:80]!r}", flush=True)
                continue
            utt_id = parts[0].strip()
            lang = parts[1].strip().lower()
            text = clean_text(parts[2])
            if not utt_id or not lang or not is_usable_text(text):
                continue
            if include_langs and lang not in include_langs:
                continue
            if lang in skip_langs:
                continue
            source_wav = None
            if source_wav_dir:
                candidate = source_wav_dir / f"{utt_id}.wav"
                if candidate.exists():
                    source_wav = candidate
            rows.append(SourceRow(utt_id=utt_id, lang=lang, text=text, source_wav=source_wav))
    if not rows:
        raise ValueError(f"No usable rows found in {metadata_path}")
    return rows


def round_robin_by_language(rows: list[SourceRow]) -> list[SourceRow]:
    grouped: dict[str, deque[SourceRow]] = defaultdict(deque)
    for row in rows:
        grouped[row.lang].append(row)

    ordered: list[SourceRow] = []
    languages = sorted(grouped)
    while any(grouped.values()):
        for lang in languages:
            if grouped[lang]:
                ordered.append(grouped[lang].popleft())
    return ordered


def record_lang(record: dict) -> str:
    return str(record.get("lang") or "en").lower()


def metadata_record(record: dict) -> dict:
    generated_wav = record.get("generated_wav")
    return {
        "utt_id": str(record["utt_id"]).strip(),
        "lang": record_lang(record),
        "text": clean_text(str(record["text"])),
        "wav_path": str(generated_wav) if generated_wav else None,
        "source_wav": record.get("source_wav"),
        "duration_seconds": record.get("duration_seconds"),
        "sample_rate": record.get("sample_rate"),
        "voice": record.get("voice"),
    }


def write_outputs(output_dir: Path, records: list[dict], settings: dict):
    metadata_rows = [metadata_record(record) for record in records if record.get("status") == "generated"]
    metadata_path = output_dir / "metadata.parquet"
    pq.write_table(pa.Table.from_pylist(metadata_rows, schema=METADATA_SCHEMA), metadata_path)

    stale_csv_path = output_dir / "metadata.csv"
    if stale_csv_path.exists():
        stale_csv_path.unlink()

    total_duration = sum(record.get("duration_seconds", 0.0) for record in records if record.get("status") == "generated")
    payload = {
        "settings": settings,
        "summary": {
            "generated": sum(1 for record in records if record.get("status") == "generated"),
            "failed": sum(1 for record in records if record.get("status") == "failed"),
            "duration_seconds": round(total_duration, 3),
            "duration_minutes": round(total_duration / 60.0, 3),
            "languages": {
                lang: sum(1 for record in records if record.get("status") == "generated" and record_lang(record) == lang)
                for lang in sorted({record_lang(record) for record in records if record.get("status") == "generated"})
            },
        },
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


async def generate_mp3_edge(text: str, voice: str, output_mp3: Path):
    import edge_tts  # pylint: disable=import-outside-toplevel

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output_mp3))


def generate_mp3_azure_sync(text: str, voice: str, output_mp3: Path, azure_key: str, azure_region: str):
    import azure.cognitiveservices.speech as speechsdk  # pylint: disable=import-outside-toplevel

    speech_config = speechsdk.SpeechConfig(subscription=azure_key, region=azure_region)
    speech_config.speech_synthesis_voice_name = voice
    speech_config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Audio24Khz160KBitRateMonoMp3
    )
    audio_config = speechsdk.audio.AudioOutputConfig(filename=str(output_mp3))
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
    result = synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return
    if result.reason == speechsdk.ResultReason.Canceled:
        details = result.cancellation_details
        raise RuntimeError(f"Azure synthesis canceled: {details.reason} {details.error_details}")
    raise RuntimeError(f"Azure synthesis failed: {result.reason}")


async def generate_mp3(backend: str, text: str, voice: str, output_mp3: Path, azure_key: str | None = None, azure_region: str | None = None):
    if backend == "edge":
        await generate_mp3_edge(text, voice, output_mp3)
        return
    if backend == "azure":
        if not azure_key or not azure_region:
            raise ValueError("Azure backend requires AZURE_TTS_TOKEN and AZURE_TTS_REGION (or SPEECH_REGION)")
        await asyncio.to_thread(generate_mp3_azure_sync, text, voice, output_mp3, azure_key, azure_region)
        return
    raise ValueError(f"Unsupported backend {backend!r}")


def load_azure_config(token_env: str, region_env: str, region: str | None) -> tuple[str, str]:
    azure_key = os.getenv(token_env) or os.getenv("AZURE_TTS_TOKEN")
    azure_region = region or os.getenv(region_env) or os.getenv("AZURE_TTS_REGION") or os.getenv("SPEECH_REGION") or "eastus"
    if not azure_key:
        raise ValueError(f"Missing Azure TTS key. Set {token_env} or AZURE_TTS_TOKEN in .env.")
    return azure_key, azure_region


async def main(argv: list[str] | None = None):
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("edge", "azure"), default=DEFAULT_BACKEND)
    parser.add_argument("--input-metadata", type=Path, default=DEFAULT_INPUT_METADATA)
    parser.add_argument("--source-wav-dir", type=Path, default=Path("local/datasets/lzspeech-multilingual-plus/wav"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--voice",
        default=DEFAULT_VOICE,
        help="Voice short name or alias. Example: it-IT.IsabellaMultilingual or it-IT-IsabellaMultilingualNeural.",
    )
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--rate-limit-seconds", type=float, default=1.0)
    parser.add_argument("--max-failures", type=int, default=100)
    parser.add_argument(
        "--include-langs",
        default="",
        help="Comma-separated language allow-list. Empty means all. Example: de for the German slice.",
    )
    parser.add_argument("--skip-langs", default=DEFAULT_SKIP_LANGS, help="Comma-separated languages to skip.")
    parser.add_argument("--order", choices=("metadata", "balanced"), default="balanced")
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke limit. 0 means all rows.")
    parser.add_argument("--azure-token-env", default="AZURE_TTS_TOKEN")
    parser.add_argument("--azure-region-env", default="AZURE_TTS_REGION")
    parser.add_argument("--azure-region", default=None, help="Azure Speech region value, for example southeastasia or eastasia.")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    wav_dir = output_dir / "wav"
    mp3_dir = output_dir / "mp3"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    mp3_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_source_rows(
        args.input_metadata,
        args.source_wav_dir,
        include_langs=parse_langs(args.include_langs),
        skip_langs=parse_langs(args.skip_langs),
    )
    if args.order == "balanced":
        source_rows = round_robin_by_language(source_rows)
    if args.max_rows > 0:
        source_rows = source_rows[: args.max_rows]

    manifest_path = output_dir / "manifest.json"
    records = load_existing_records(manifest_path)
    generated_ids = {record["utt_id"] for record in records if record.get("status") == "generated"}
    failed_ids = {record["utt_id"] for record in records if record.get("status") == "failed"}
    failures = sum(1 for record in records if record.get("status") == "failed")
    total_duration = sum(record.get("duration_seconds", 0.0) for record in records if record.get("status") == "generated")

    if args.backend == "edge":
        resolved_voice = await resolve_edge_voice_name(args.voice)
        azure_key = None
        azure_region = None
    else:
        resolved_voice = resolve_azure_voice_name(args.voice)
        azure_key, azure_region = load_azure_config(args.azure_token_env, args.azure_region_env, args.azure_region)

    if resolved_voice != args.voice:
        print(f"resolved voice: {args.voice} -> {resolved_voice}", flush=True)

    settings = {
        "backend": args.backend,
        "input_metadata": str(args.input_metadata.resolve()),
        "source_wav_dir": str(args.source_wav_dir.resolve()) if args.source_wav_dir else None,
        "output_dir": str(output_dir.resolve()),
        "voice": resolved_voice,
        "azure_region": azure_region if args.backend == "azure" else None,
        "sample_rate": args.sample_rate,
        "rate_limit_seconds": args.rate_limit_seconds,
        "include_langs": sorted(parse_langs(args.include_langs)),
        "skip_langs": sorted(parse_langs(args.skip_langs)),
        "order": args.order,
        "max_rows": args.max_rows,
    }

    pending_rows = [row for row in source_rows if row.utt_id not in generated_ids and row.utt_id not in failed_ids]
    print(
        f"backend={args.backend} loaded={len(source_rows)} pending={len(pending_rows)} generated={len(generated_ids)} "
        f"failed={len(failed_ids)} duration={total_duration / 60.0:.2f}m voice={resolved_voice}",
        flush=True,
    )

    last_request_start = 0.0
    try:
        for index, row in enumerate(pending_rows, start=1):
            wait = args.rate_limit_seconds - (time.monotonic() - last_request_start)
            if wait > 0:
                await asyncio.sleep(wait)
            last_request_start = time.monotonic()

            mp3_path = mp3_dir / f"{row.utt_id}.mp3"
            wav_path = wav_dir / f"{row.utt_id}.wav"
            started = time.time()
            print(
                f"[{index}/{len(pending_rows)}] {row.utt_id} lang={row.lang} "
                f"done={len(generated_ids)} total={total_duration / 60:.2f}m text={row.text[:80]}",
                flush=True,
            )
            try:
                await generate_mp3(args.backend, row.text, resolved_voice, mp3_path, azure_key=azure_key, azure_region=azure_region)
                convert_to_wav(mp3_path, wav_path, args.sample_rate)
                duration, sample_rate = wav_duration(wav_path)
                record = {
                    "utt_id": row.utt_id,
                    "lang": row.lang,
                    "text": row.text,
                    "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                    "generated_mp3": str(mp3_path.resolve()),
                    "generated_wav": str(wav_path.resolve()),
                    "duration_seconds": round(duration, 3),
                    "sample_rate": sample_rate,
                    "voice": resolved_voice,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "status": "generated",
                }
                total_duration += duration
                generated_ids.add(row.utt_id)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if wav_path.exists():
                    wav_path.unlink()
                if mp3_path.exists():
                    mp3_path.unlink()
                record = {
                    "utt_id": row.utt_id,
                    "lang": row.lang,
                    "text": row.text,
                    "source_wav": str(row.source_wav.resolve()) if row.source_wav else None,
                    "voice": resolved_voice,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "error": repr(exc),
                    "status": "failed",
                }
                failed_ids.add(row.utt_id)
                print(f"FAILED {row.utt_id}: {exc!r}", flush=True)
                if failures >= args.max_failures:
                    records.append(record)
                    raise RuntimeError(f"Stopping after {failures} failures") from exc
            records.append(record)
            write_outputs(output_dir, records, settings)
    finally:
        write_outputs(output_dir, records, settings)

    print(f"done: {total_duration / 60:.2f} minutes -> {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
