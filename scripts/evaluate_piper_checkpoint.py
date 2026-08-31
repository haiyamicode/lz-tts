#!/usr/bin/env python3
"""Generate and score Piper checkpoint samples with windowed SCOREQ.

This is the reusable script for the repeated "test latest checkpoint" workflow:

  uv run python scripts/evaluate_piper_checkpoint.py \
    --experiment-root local/data/exp/lzspeech-bert-87m-24k-enus-dpblend \
    --config local/data/exp/lzspeech-piper-24k/config.json \
    --output-dir local/data/exp/piper_87m_latest_eval \
    --speaker en --device cuda

It writes:
  - prompts.json: full per-sample metadata
  - scoreq_scores.csv: score table
  - summary.json: aggregate metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from src.piper import PiperInference
from src.piper.vits.wavfile import write as write_wav


SCOREQ_FIELDS = (
    "window_min",
    "window_mean",
    "window_median",
    "window_p10",
)


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
        payload = (
            yaml.safe_load(f)
            if path.suffix.lower() in {".yaml", ".yml"}
            else json.load(f)
        )

    if isinstance(payload, dict):
        if isinstance(payload.get("evaluation"), dict):
            payload = payload["evaluation"]
        if "prompts" in payload:
            payload = payload["prompts"]
        elif "items" in payload:
            payload = payload["items"]
        else:
            raise ValueError(f"Unsupported prompt object keys in {path}")

    if not isinstance(payload, list):
        raise ValueError(f"Prompt configuration must contain a prompt list: {path}")

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
                "speaker": str(
                    item.get("speaker")
                    or item.get("language")
                    or item.get("lang")
                    or "en"
                ),
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


def load_scoreq_config(path: Path | None) -> dict[str, Any]:
    if path is None or path.suffix.lower() not in {".yaml", ".yml"}:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        return {}
    monitor = payload.get("quality_monitor")
    return monitor if isinstance(monitor, dict) else {}


def score_scoreq(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    python_bin: Path,
    script_path: Path,
    model_path: Path,
    window_seconds: float,
    hop_ratio: float,
    min_threshold: float,
    mean_threshold: float,
    cpu_threads: int,
) -> dict[str, dict[str, Any]]:
    for dependency in (python_bin, script_path, model_path):
        if not dependency.is_file():
            raise FileNotFoundError(dependency)

    manifest_path = output_dir / "scoreq_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "item_id": f"sample_{row['index']:04d}",
                        "text": row["text"],
                        "audio_path": str(Path(row["path"]).resolve()),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    score_dir = output_dir / "scoreq"
    # Audio generation is stochastic and may overwrite an existing run name.
    # Never reuse scores that were computed for the previous contents.
    (score_dir / "scores.jsonl").unlink(missing_ok=True)
    (score_dir / "summary.json").unlink(missing_ok=True)
    command = [
        str(python_bin),
        str(script_path),
        "--manifest",
        str(manifest_path),
        "--audio-field",
        "audio_path",
        "--id-field",
        "item_id",
        "--output-dir",
        str(score_dir),
        "--model",
        str(model_path),
        "--window-seconds",
        str(window_seconds),
        "--hop-ratio",
        str(hop_ratio),
        "--min-threshold",
        str(min_threshold),
        "--mean-threshold",
        str(mean_threshold),
        "--listening-count",
        "0",
        "--cpu-threads",
        str(cpu_threads),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "SCOREQ scoring failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )

    scores: dict[str, dict[str, Any]] = {}
    with (score_dir / "scores.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            result = json.loads(line)
            scores[result["audio_path"]] = result
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
    parser.add_argument(
        "--prompts",
        type=Path,
        help="JSON prompt list or experiment YAML with evaluation.prompts; defaults to English custom set",
    )
    parser.add_argument("--speaker", help="Force speaker label for every prompt")
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=1,
        help="Number of independent stochastic generations for each prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Base sampling seed; each generated item uses seed + item index",
    )
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
    parser.add_argument(
        "--no-scoreq", action="store_true", help="Skip windowed SCOREQ scoring"
    )
    parser.add_argument("--scoreq-python", type=Path)
    parser.add_argument("--scoreq-script", type=Path)
    parser.add_argument("--scoreq-model", type=Path)
    parser.add_argument("--scoreq-window-seconds", type=float)
    parser.add_argument("--scoreq-hop-ratio", type=float)
    parser.add_argument("--scoreq-min-threshold", type=float)
    parser.add_argument("--scoreq-mean-threshold", type=float)
    parser.add_argument("--scoreq-cpu-threads", type=int)
    args = parser.parse_args()

    if args.checkpoint is None and args.experiment_root is None:
        parser.error("provide --checkpoint or --experiment-root")
    if args.samples_per_prompt < 1:
        parser.error("--samples-per-prompt must be at least 1")
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
    total_samples = len(prompts) * args.samples_per_prompt
    for prompt_index, prompt in enumerate(prompts):
        text = prompt["text"]
        speaker = args.speaker if args.speaker is not None else prompt.get("speaker")
        language = prompt.get("language") or speaker or ""
        for sample_index in range(args.samples_per_prompt):
            index = prompt_index * args.samples_per_prompt + sample_index
            variant = (
                f"_sample{sample_index + 1:02d}" if args.samples_per_prompt > 1 else ""
            )
            name = f"{prompt_index:03d}_{language}_{safe_name(text)}{variant}.wav"
            wav_path = run_dir / name

            sample_seed = args.seed + index
            random.seed(sample_seed)
            np.random.seed(sample_seed)
            torch.manual_seed(sample_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sample_seed)

            start = time.perf_counter()
            audio = inference.synthesize(text, speaker=speaker, **synth_kwargs)
            infer_sec = time.perf_counter() - start
            write_wav(str(wav_path), inference.sample_rate, audio)

            duration_sec = float(len(audio) / inference.sample_rate)
            rtf = float(infer_sec / duration_sec) if duration_sec > 0 else math.nan
            row = {
                "run": run_name,
                "index": index,
                "prompt_index": prompt_index,
                "sample_index": sample_index,
                "seed": sample_seed,
                "language": language,
                "speaker": speaker,
                "source": prompt.get("source", ""),
                "text": text,
                "path": str(wav_path),
                "duration_sec": duration_sec,
                "infer_sec": float(infer_sec),
                "rtf": rtf,
                **{field: "" for field in SCOREQ_FIELDS},
                "checkpoint": str(checkpoint),
                "config": str(config),
                "synth_kwargs": json.dumps(synth_kwargs, sort_keys=True),
            }
            rows.append(row)
            print(
                f"{index + 1}/{total_samples} wrote {wav_path} rtf={rtf:.3f}",
                flush=True,
            )

    scoreq_config = load_scoreq_config(args.prompts)
    if not args.no_scoreq:

        def option(name: str, default: Any) -> Any:
            cli_value = getattr(args, name)
            return (
                cli_value if cli_value is not None else scoreq_config.get(name, default)
            )

        scores = score_scoreq(
            rows,
            run_dir,
            python_bin=Path(option("scoreq_python", ".venv/bin/python")),
            script_path=Path(
                option("scoreq_script", "scripts/score_dpo_scoreq_windows.py")
            ),
            model_path=Path(
                option(
                    "scoreq_model",
                    Path.home() / ".cache/scoreq/onnx-models/adapt_nr_synthetic.onnx",
                )
            ),
            window_seconds=float(option("scoreq_window_seconds", 0.75)),
            hop_ratio=float(option("scoreq_hop_ratio", 0.25)),
            min_threshold=float(option("scoreq_min_threshold", 3.5)),
            mean_threshold=float(option("scoreq_mean_threshold", 3.7)),
            cpu_threads=int(option("scoreq_cpu_threads", 16)),
        )
        for row in rows:
            result = scores.get(str(Path(row["path"]).resolve()))
            if result is not None:
                for field in SCOREQ_FIELDS:
                    row[field] = result[field]

    with (run_dir / "prompts.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "run",
        "index",
        "prompt_index",
        "sample_index",
        "seed",
        "language",
        "speaker",
        "source",
        "text",
        "path",
        "duration_sec",
        "infer_sec",
        "rtf",
        *SCOREQ_FIELDS,
        "checkpoint",
        "config",
        "synth_kwargs",
    ]
    with (run_dir / "scoreq_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rtf_values = [
        float(row["rtf"]) for row in rows if not math.isnan(float(row["rtf"]))
    ]
    summary = {
        "run": run_name,
        "checkpoint": str(checkpoint),
        "config": str(config),
        "output_dir": str(run_dir),
        "sample_rate": inference.sample_rate,
        "num_prompts": len(rows),
        "source_prompt_count": len(prompts),
        "samples_per_prompt": args.samples_per_prompt,
        "seed": args.seed,
        "synth_kwargs": synth_kwargs,
        "scoreq": {
            field: numeric_summary(
                [float(row[field]) for row in rows if row[field] != ""]
            )
            for field in SCOREQ_FIELDS
        },
        "rtf": numeric_summary(rtf_values),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
