#!/usr/bin/env python3
"""Generate and score Piper checkpoint samples.

This is the reusable script for the repeated "test latest checkpoint" workflow:

  uv run python scripts/evaluate_piper_checkpoint.py \
    --experiment-root local/data/exp/lzspeech-bert-87m-24k-enus-dpblend \
    --config local/data/exp/lzspeech-piper-24k/config.json \
    --output-dir local/data/exp/piper_87m_latest_eval \
    --speaker en --device cuda

It writes:
  - prompts.json: full per-sample metadata
  - utmos_scores.csv: score table
  - summary.json: aggregate metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.piper import PiperInference
from src.piper.vits.wavfile import write as write_wav


DEFAULT_ENGLISH_PROMPTS: list[dict[str, str]] = [
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Anyway, I need to finalize the report by 5:00 p.m., but I still have to check the budget numbers.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Did Sarah confirm the train tickets for Thursday, or should I call the station again?",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "The NASA briefing starts at 9:30 tomorrow morning, so please send me the notes before breakfast.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Oh, I almost forgot: the package from London arrived yesterday, and the label says fragile.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Can you believe Dr. Chen finished the prototype in just three weeks? That's honestly impressive.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "I got twenty-seven messages this morning, but only five of them were actually urgent.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Please remind Alex that the meeting moved from room 204 to room 318 after lunch.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "The old library on Main Street closed last year, but the new branch opens next Monday.",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Wait, did you say the flight lands at 7:45, or does boarding start at 7:45?",
    },
    {
        "language": "en",
        "speaker": "en",
        "source": "custom_english_lzspeech_style",
        "text": "Thanks for checking the spreadsheet; I know the formatting was a mess after the export.",
    },
]


def find_latest_checkpoint(experiment_root: Path) -> Path:
    checkpoints = sorted(
        experiment_root.glob("lightning_logs/**/checkpoints/*.ckpt"),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found under {experiment_root}")
    return checkpoints[0]


def load_prompts(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return list(DEFAULT_ENGLISH_PROMPTS)

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        if "prompts" in payload:
            payload = payload["prompts"]
        elif "items" in payload:
            payload = payload["items"]
        else:
            raise ValueError(f"Unsupported prompt object keys in {path}")

    if not isinstance(payload, list):
        raise ValueError(f"Prompt file must contain a JSON list: {path}")

    prompts: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in payload:
        if isinstance(item, str):
            row = {
                "language": "en",
                "speaker": "en",
                "source": "prompt_file",
                "text": item,
            }
        elif isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            row = {
                "language": str(item.get("language") or item.get("lang") or "en"),
                "speaker": str(item.get("speaker") or item.get("language") or item.get("lang") or "en"),
                "source": str(item.get("source") or "prompt_file"),
                "text": text,
            }
        else:
            raise ValueError(f"Unsupported prompt item in {path}: {item!r}")

        key = (row["language"], row["speaker"], row["text"])
        if key in seen:
            continue
        seen.add(key)
        prompts.append(row)

    if not prompts:
        raise ValueError(f"No prompts loaded from {path}")
    return prompts


def score_utmos(
    wav_paths: list[Path],
    python_bin: Path,
    worker_path: Path,
) -> dict[str, float]:
    if not python_bin.exists():
        raise FileNotFoundError(f"UTMOS python not found: {python_bin}")
    if not worker_path.exists():
        raise FileNotFoundError(f"UTMOS worker not found: {worker_path}")

    env = os.environ.copy()
    proc = subprocess.Popen(
        [str(python_bin), str(worker_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None

    scores: dict[str, float] = {}
    try:
        ready = proc.stdout.readline()
        if not ready:
            raise RuntimeError("UTMOS worker did not start")

        for idx, wav_path in enumerate(wav_paths):
            proc.stdin.write(json.dumps({"id": idx, "path": str(wav_path)}) + "\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
            if not line:
                raise RuntimeError(f"UTMOS worker stopped while scoring {wav_path}")
            response = json.loads(line)
            if "mos_score" not in response:
                raise RuntimeError(f"UTMOS scoring failed for {wav_path}: {response}")
            scores[str(wav_path)] = float(response["mos_score"])
    finally:
        if proc.stdin:
            proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr else ""
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if stderr.strip():
            (Path.cwd() / "local/data/exp/utmos_worker_stderr.log").write_text(
                stderr,
                encoding="utf-8",
            )

    return scores


def safe_name(text: str, max_len: int = 48) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len].strip("_") or "sample"


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    values = sorted(values)
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint to evaluate")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        help="Experiment root; latest checkpoint is used when --checkpoint is omitted",
    )
    parser.add_argument("--config", type=Path, required=True, help="Piper config.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-name", help="Subdirectory/run label for this evaluation")
    parser.add_argument("--prompts", type=Path, help="JSON prompt list; defaults to English custom set")
    parser.add_argument("--speaker", help="Force speaker label for every prompt")
    parser.add_argument("--device", default="cuda", help="Torch device for synthesis")
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--length-scale", type=float)
    parser.add_argument("--noise-w", type=float)
    parser.add_argument(
        "--sdp-ratio",
        type=float,
        default=0.2,
        help="Duration predictor blend ratio; defaults to production Sparrow DP/SDP blend",
    )
    parser.add_argument(
        "--neural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use neural heteronym frontend; enabled by default to match production",
    )
    parser.add_argument("--no-utmos", action="store_true", help="Skip UTMOS scoring")
    parser.add_argument("--utmos-python", type=Path, default=Path("local/utmos_probe/.venv/bin/python"))
    parser.add_argument("--utmos-worker", type=Path, default=Path("local/utmos_probe/utmos_stdin_worker.py"))
    args = parser.parse_args()

    if args.checkpoint is None and args.experiment_root is None:
        parser.error("provide --checkpoint or --experiment-root")
    return args


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint(args.experiment_root)
    checkpoint = checkpoint.resolve()
    config = args.config.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not config.exists():
        raise FileNotFoundError(config)

    run_name = args.run_name or checkpoint.stem
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.prompts)
    inference = PiperInference(checkpoint, config, device=args.device)

    synth_kwargs: dict[str, Any] = {"neural": args.neural}
    for attr, key in (
        ("noise_scale", "noise_scale"),
        ("length_scale", "length_scale"),
        ("noise_w", "noise_w"),
        ("sdp_ratio", "sdp_ratio"),
    ):
        value = getattr(args, attr)
        if value is not None:
            synth_kwargs[key] = value

    rows: list[dict[str, Any]] = []
    wav_paths: list[Path] = []
    for index, prompt in enumerate(prompts):
        text = prompt["text"]
        speaker = args.speaker if args.speaker is not None else prompt.get("speaker")
        language = prompt.get("language") or speaker or ""
        name = f"{index:03d}_{language}_{safe_name(text)}.wav"
        wav_path = run_dir / name

        start = time.perf_counter()
        audio = inference.synthesize(text, speaker=speaker, **synth_kwargs)
        infer_sec = time.perf_counter() - start
        write_wav(str(wav_path), inference.sample_rate, audio)

        duration_sec = float(len(audio) / inference.sample_rate)
        rtf = float(infer_sec / duration_sec) if duration_sec > 0 else math.nan
        row = {
            "run": run_name,
            "index": index,
            "language": language,
            "speaker": speaker,
            "source": prompt.get("source", ""),
            "text": text,
            "path": str(wav_path),
            "duration_sec": duration_sec,
            "infer_sec": float(infer_sec),
            "rtf": rtf,
            "utmos": "",
            "checkpoint": str(checkpoint),
            "config": str(config),
            "synth_kwargs": json.dumps(synth_kwargs, sort_keys=True),
        }
        rows.append(row)
        wav_paths.append(wav_path)
        print(f"{index + 1}/{len(prompts)} wrote {wav_path} rtf={rtf:.3f}", flush=True)

    if not args.no_utmos:
        scores = score_utmos(wav_paths, args.utmos_python, args.utmos_worker)
        for row in rows:
            row["utmos"] = scores.get(row["path"], "")

    with (run_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "run",
        "index",
        "language",
        "speaker",
        "source",
        "text",
        "path",
        "duration_sec",
        "infer_sec",
        "rtf",
        "utmos",
        "checkpoint",
        "config",
        "synth_kwargs",
    ]
    with (run_dir / "utmos_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    utmos_values = [float(row["utmos"]) for row in rows if row["utmos"] != ""]
    rtf_values = [float(row["rtf"]) for row in rows if not math.isnan(float(row["rtf"]))]
    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "output_dir": str(run_dir),
        "sample_rate": inference.sample_rate,
        "num_prompts": len(rows),
        "synth_kwargs": synth_kwargs,
        "utmos": numeric_summary(utmos_values),
        "rtf": numeric_summary(rtf_values),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
