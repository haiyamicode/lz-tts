#!/usr/bin/env python3
"""Generate comparable multilingual Sparrow samples from a checkpoint."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from src.piper.inference import PiperInference
from src.piper.vits.wavfile import write as write_wav


def _safe_name(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return value[:64] or "sample"


def _duration(path: str | Path) -> float:
    return float(sf.info(str(path)).duration)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline-prompts", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--extra-probes", type=int, default=2)
    parser.add_argument(
        "--priority-speakers",
        default="th-TH,lo-LA,km-KH,my-MM,mn-MN,ps-AF,as-IN,or-IN,he-IL",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).resolve()
    config = Path(args.config).resolve()
    dataset_path = Path(args.dataset).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = pq.read_table(dataset_path).to_pylist()
    by_source = {Path(str(row["audio_path"])).stem: row for row in dataset_rows}
    by_speaker: dict[str, list[dict]] = {}
    for row in dataset_rows:
        by_speaker.setdefault(str(row["speaker"]), []).append(row)

    baseline = json.loads(Path(args.baseline_prompts).read_text(encoding="utf-8"))
    requests: list[dict] = []
    used_sources: set[str] = set()
    for item in baseline:
        source = str(item["source"])
        row = by_source[source]
        requests.append(
            {
                "group": "all_languages",
                "language": str(row["speaker"]),
                "speaker": str(row["speaker"]),
                "source": source,
                "text": str(row["text"]),
                "reference": str(row["audio_path"]),
            }
        )
        used_sources.add(source)

    priorities = [value.strip() for value in args.priority_speakers.split(",") if value.strip()]
    target_durations = [4.0 + index for index in range(max(0, args.extra_probes))]
    for speaker in priorities:
        candidates = []
        for row in by_speaker.get(speaker, []):
            source = Path(str(row["audio_path"])).stem
            if source in used_sources:
                continue
            try:
                reference_duration = _duration(row["audio_path"])
            except Exception:
                continue
            if 2.5 <= reference_duration <= 8.0:
                candidates.append((row, reference_duration))

        for probe_index, target in enumerate(target_durations):
            if not candidates:
                break
            chosen_index = min(
                range(len(candidates)),
                key=lambda index: abs(candidates[index][1] - target),
            )
            row, reference_duration = candidates.pop(chosen_index)
            source = Path(str(row["audio_path"])).stem
            used_sources.add(source)
            requests.append(
                {
                    "group": "priority_probe",
                    "probe_index": probe_index,
                    "language": speaker,
                    "speaker": speaker,
                    "source": source,
                    "text": str(row["text"]),
                    "reference": str(row["audio_path"]),
                    "reference_duration": reference_duration,
                }
            )

    engine = PiperInference(checkpoint, config, device=args.device)
    texts = [item["text"] for item in requests]
    speakers = [item["speaker"] for item in requests]
    started = time.monotonic()
    generated = engine.synthesize_batch(
        texts,
        speaker=speakers,
        batch_size=args.batch_size,
        neural=True,
        sdp_ratio=0.2,
    )
    elapsed = time.monotonic() - started

    manifest = []
    for index, (item, audio) in enumerate(zip(requests, generated)):
        filename = (
            f"{index:03d}_{item['language']}_{item['source']}_"
            f"{_safe_name(item['text'])}.wav"
        )
        output_path = output_dir / filename
        write_wav(str(output_path), engine.sample_rate, audio)
        reference_duration = item.get("reference_duration") or _duration(item["reference"])
        generated_duration = len(audio) / engine.sample_rate
        spans = engine.phonemize(item["text"], speaker=item["speaker"], neural=True)
        record = {
            **item,
            "path": str(output_path.resolve()),
            "generated_duration": generated_duration,
            "reference_duration": reference_duration,
            "duration_ratio": generated_duration / reference_duration,
            "phonemes": ["".join(span["phonemes"]) for span in spans],
            "checkpoint": str(checkpoint),
            "config": str(config),
        }
        manifest.append(record)
        print(
            f"{index + 1}/{len(requests)} {item['language']} "
            f"duration={generated_duration:.2f}s ratio={record['duration_ratio']:.2f}",
            flush=True,
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "run": args.run_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "dataset": str(dataset_path),
        "num_samples": len(manifest),
        "num_all_languages": sum(item["group"] == "all_languages" for item in manifest),
        "num_priority_probes": sum(item["group"] == "priority_probe" for item in manifest),
        "elapsed_sec": elapsed,
        "sample_rate": engine.sample_rate,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
