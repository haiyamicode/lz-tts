#!/usr/bin/env python3
"""Build strict VoxCPM DPO and SFT manifests from windowed SCOREQ scores."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo-train", required=True)
    parser.add_argument("--dpo-validation", required=True)
    parser.add_argument("--dpo-scores", required=True)
    parser.add_argument("--sft-manifest", required=True)
    parser.add_argument("--sft-scores", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-prompts")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _score_index(path: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        audio_path = str(Path(row["audio_path"]).resolve())
        if audio_path in index:
            raise ValueError(f"Duplicate SCOREQ audio path in {path}: {audio_path}")
        index[audio_path] = row
    return index


def _quality(
    row: dict[str, Any],
    score: dict[str, Any],
    thresholds: dict[str, dict[str, Any]],
    language: str,
) -> tuple[dict[str, Any], bool]:
    threshold = thresholds[language]
    window_min = float(score["window_min"])
    window_mean = float(score["window_mean"])
    min_threshold = float(threshold["window_min"])
    mean_threshold = float(threshold["window_mean"])
    failed = []
    if window_min < min_threshold:
        failed.append("window_min")
    if window_mean < mean_threshold:
        failed.append("window_mean")
    quality = {
        "window_min": window_min,
        "window_mean": window_mean,
        "window_median": float(score["window_median"]),
        "window_p10": float(score["window_p10"]),
        "window_seconds": float(score["window_seconds"]),
        "hop_seconds": float(score["hop_seconds"]),
        "window_count": int(score["window_count"]),
        "min_threshold": min_threshold,
        "mean_threshold": mean_threshold,
        "failed_gates": failed,
    }
    return {**row, "scoreq_windows": quality}, not failed


def _filter_rows(
    rows: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    thresholds: dict[str, dict[str, Any]],
    *,
    audio_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    missing: list[str] = []

    for row in rows:
        language = str(row.get("language") or str(row["pair_id"]).split("_", 1)[0])
        source_counts[language] += 1
        audio_path = str(Path(row[audio_field]).resolve())
        score = scores.get(audio_path)
        if score is None:
            missing.append(audio_path)
            continue
        annotated, is_accepted = _quality(row, score, thresholds, language)
        if is_accepted:
            accepted.append(annotated)
            accepted_counts[language] += 1
        else:
            rejected.append(annotated)
            failed = annotated["scoreq_windows"]["failed_gates"]
            reason_counts["+".join(failed)] += 1

    if missing:
        preview = "\n".join(missing[:10])
        raise ValueError(f"Missing SCOREQ scores for {len(missing)} rows:\n{preview}")

    languages = sorted(source_counts)
    summary = {
        "source_count": len(rows),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted_rate": len(accepted) / max(1, len(rows)),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "by_language": {
            language: {
                "source": source_counts[language],
                "accepted": accepted_counts[language],
                "rejected": source_counts[language] - accepted_counts[language],
                "accepted_rate": accepted_counts[language]
                / source_counts[language],
            }
            for language in languages
        },
    }
    return accepted, rejected, summary


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    dpo_dir = output_dir / "dpo"
    sft_dir = output_dir / "sft"
    dpo_scores = _score_index(Path(args.dpo_scores))
    sft_scores = _score_index(Path(args.sft_scores))
    threshold_payload = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    thresholds = threshold_payload["languages"]

    split_summaries = {}
    for split, manifest in (
        ("train", Path(args.dpo_train)),
        ("validation", Path(args.dpo_validation)),
    ):
        accepted, rejected, summary = _filter_rows(
            _read_jsonl(manifest),
            dpo_scores,
            thresholds,
            audio_field="chosen_audio",
        )
        _write_jsonl(dpo_dir / f"{split}.jsonl", accepted)
        _write_jsonl(dpo_dir / f"{split}_rejected_scoreq.jsonl", rejected)
        split_summaries[split] = summary

    sft_accepted, sft_rejected, sft_summary = _filter_rows(
        _read_jsonl(Path(args.sft_manifest)),
        sft_scores,
        thresholds,
        audio_field="audio",
    )
    _write_jsonl(sft_dir / "accepted.jsonl", sft_accepted)
    _write_jsonl(sft_dir / "rejected_scoreq.jsonl", sft_rejected)

    if args.heldout_prompts is not None:
        shutil.copy2(args.heldout_prompts, dpo_dir / "heldout_prompts.jsonl")

    summary = {
        "method": threshold_payload["method"],
        "thresholds": thresholds,
        "dpo": split_summaries,
        "sft": sft_summary,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
