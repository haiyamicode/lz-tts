#!/usr/bin/env python3
"""Replay an existing Qwen voice-clone manifest with a new reference voice."""

import argparse
import base64
import json
import time
from pathlib import Path

import requests
import soundfile as sf


LANGUAGE_NAMES = {
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
}


def write_json(path: Path, payload) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_metadata(path: Path, records: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for record in records:
            if record.get("status") == "generated":
                file.write(f"{record['utt_id']}|{record['lang']}|{record['text']}\n")
    tmp.replace(path)


def load_done(manifest_path: Path) -> dict[str, dict]:
    if not manifest_path.exists():
        return {}
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["utt_id"]: record for record in records if record.get("status") == "generated"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--server-url", default="http://127.0.0.1:7860/generate")
    parser.add_argument("--status-url", default="http://127.0.0.1:7860/status")
    parser.add_argument("--ref-audio", type=Path)
    parser.add_argument("--ref-text-file", type=Path)
    parser.add_argument(
        "--ref-map-json",
        type=Path,
        help="JSON map keyed by language code with wav/text or audio/text_file entries.",
    )
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.03)
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    args = parser.parse_args()

    source_manifest = args.source_manifest.resolve()
    output_dir = args.output_dir.resolve()
    wav_dir = output_dir / "wav"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    source_records = json.loads(source_manifest.read_text(encoding="utf-8"))
    rows = [
        {
            "utt_id": record["utt_id"],
            "lang": record["lang"],
            "text": record["text"],
            "source_wav": record.get("source_wav"),
            "source_duration_seconds": record.get("source_duration_seconds"),
        }
        for record in source_records
        if record.get("status", "generated") == "generated"
    ]

    if args.ref_map_json:
        raw_ref_map = json.loads(args.ref_map_json.read_text(encoding="utf-8"))
        ref_map = {}
        for lang, item in raw_ref_map.items():
            audio = Path(item.get("wav") or item.get("audio")).resolve()
            text_file = Path(item.get("text_file")).resolve() if item.get("text_file") else None
            text = item.get("text") or (text_file.read_text(encoding="utf-8").strip() if text_file else "")
            if not audio.exists():
                raise FileNotFoundError(f"Missing ref audio for {lang}: {audio}")
            if not text:
                raise ValueError(f"Missing ref text for {lang}")
            ref_map[lang] = {**item, "audio": audio, "text": text}
    else:
        if not args.ref_audio or not args.ref_text_file:
            raise ValueError("--ref-audio and --ref-text-file are required when --ref-map-json is not used")
        ref_text = args.ref_text_file.read_text(encoding="utf-8").strip()
        ref_map = {
            "*": {
                "audio": args.ref_audio.resolve(),
                "text": ref_text,
                "text_file": str(args.ref_text_file.resolve()),
            }
        }
    manifest_path = output_dir / "manifest.json"
    metadata_path = output_dir / "metadata.csv"
    done = {} if args.force else load_done(manifest_path)
    if args.force:
        for wav in wav_dir.glob("*.wav"):
            wav.unlink()

    try:
        server_status = requests.get(args.status_url, timeout=10).json()
    except Exception as exc:
        server_status = {"error": str(exc)}

    ordered_records = []
    total = len(rows)
    for index, row in enumerate(rows, 1):
        utt_id = row["utt_id"]
        out_wav = wav_dir / f"{utt_id}.wav"
        if utt_id in done and out_wav.exists():
            print(f"[{index}/{total}] skip {utt_id}", flush=True)
            ordered_records.append(done[utt_id])
            continue

        language = LANGUAGE_NAMES.get(row["lang"], "Auto")
        print(f"[{index}/{total}] generate {utt_id} lang={row['lang']} language={language}", flush=True)
        started = time.time()
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                ref = ref_map.get(row["lang"]) or ref_map.get("*")
                if not ref:
                    raise ValueError(f"No reference configured for lang={row['lang']}")
                with ref["audio"].open("rb") as ref_file:
                    response = requests.post(
                        args.server_url,
                        data={
                            "text": row["text"],
                            "language": language,
                            "mode": "voice_clone",
                            "ref_text": ref["text"],
                            "xvec_only": "false",
                            "temperature": str(args.temperature),
                            "top_k": str(args.top_k),
                            "repetition_penalty": str(args.repetition_penalty),
                            "non_streaming_mode": "true",
                            "use_dp_budget": "false",
                            "use_postprocess": "true" if args.use_postprocess else "false",
                        },
                        files={"ref_audio": (ref["audio"].name, ref_file, "audio/wav")},
                        timeout=900,
                    )
                response.raise_for_status()
                payload = response.json()
                out_wav.write_bytes(base64.b64decode(payload["audio_b64"]))
                info = sf.info(out_wav)
                record = {
                    **row,
                    "reference_audio": str(ref["audio"]),
                    "reference_voice": ref.get("voice"),
                    "generated_wav": str(out_wav),
                    "duration_seconds": round(info.frames / info.samplerate, 3),
                    "sample_rate": info.samplerate,
                    "elapsed_seconds": round(time.time() - started, 3),
                    "metrics": payload.get("metrics"),
                    "status": "generated",
                }
                done[utt_id] = record
                ordered_records.append(record)
                break
            except Exception as exc:
                last_error = exc
                print(f"  attempt {attempt}/{args.max_retries} failed: {exc}", flush=True)
                if attempt < args.max_retries:
                    time.sleep(3 * attempt)
        else:
            record = {**row, "generated_wav": str(out_wav), "status": "failed", "error": str(last_error)}
            ordered_records.append(record)

        write_json(
            manifest_path,
            {
                "source_manifest": str(source_manifest),
                "server_status": server_status,
                "reference": {
                    "map_json": str(args.ref_map_json.resolve()) if args.ref_map_json else None,
                    "items": {
                        lang: {
                            **{k: v for k, v in item.items() if k not in {"audio"}},
                            "audio": str(item["audio"]),
                        }
                        for lang, item in ref_map.items()
                    },
                },
                "settings": {
                    "mode": "voice_clone",
                    "xvec_only": False,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "repetition_penalty": args.repetition_penalty,
                    "non_streaming_mode": True,
                    "use_dp_budget": False,
                    "use_postprocess": args.use_postprocess,
                },
                "records": ordered_records,
            },
        )
        write_metadata(metadata_path, ordered_records)

    generated = [record for record in ordered_records if record.get("status") == "generated"]
    total_seconds = sum(record.get("duration_seconds", 0.0) for record in generated)
    print(
        f"done generated={len(generated)}/{total} audio_minutes={total_seconds / 60.0:.2f} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
