#!/usr/bin/env python3
"""Score manifest audio with overlapping SCOREQ windows."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from src.duration_alignment import trim_boundary_silence


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--audio-field", default="chosen_audio")
    parser.add_argument("--id-field", default="pair_id")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--language-field")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--window-seconds", type=float, default=0.75)
    parser.add_argument("--hop-ratio", type=float, default=0.25)
    parser.add_argument("--min-threshold", type=float, default=3.5)
    parser.add_argument("--mean-threshold", type=float, default=4.5)
    parser.add_argument("--listening-count", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--audio-batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--language", action="append")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def _load_rows(
    paths: list[str],
    *,
    audio_field: str,
    id_field: str,
    text_field: str,
    language_field: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        split = path.stem
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                audio_path = str(Path(row[audio_field]).resolve())
                if audio_path in seen:
                    continue
                seen.add(audio_path)
                item_id = str(row[id_field])
                language = (
                    str(row[language_field])
                    if language_field is not None
                    else item_id.split("_", 1)[0]
                )
                rows.append(
                    {
                        "split": split,
                        "item_id": item_id,
                        "language": language,
                        "text": str(row[text_field]),
                        "audio_path": audio_path,
                    }
                )
    return rows


def _window_starts(length: int, window: int, hop: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    final_start = length - window
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _prepare_audio(
    path: str,
    window_seconds: float,
    hop_ratio: float,
) -> tuple[np.ndarray, int, int, list[int], list[np.ndarray]]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1, dtype=np.float32)
    audio = np.asarray(audio, dtype=np.float32)
    trimmed, trim_start, _ = trim_boundary_silence(audio, sample_rate)
    if trimmed.size == 0:
        raise ValueError(f"Audio is silent after trimming: {path}")

    window_samples = max(1, round(window_seconds * sample_rate))
    hop_samples = max(1, round(window_samples * hop_ratio))
    starts = _window_starts(trimmed.size, window_samples, hop_samples)
    windows = [trimmed[start : start + window_samples] for start in starts]
    return trimmed, sample_rate, trim_start, starts, windows


def _resample_and_pad(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate != 16_000:
        ratio = Fraction(16_000, sample_rate)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator).astype(
            np.float32
        )
    padded_length = math.ceil(len(audio) / 320) * 320
    if padded_length != len(audio):
        audio = np.pad(audio, (0, padded_length - len(audio)))
    return audio.astype(np.float32, copy=False)


def _score_window_groups(
    session: ort.InferenceSession,
    window_groups: list[tuple[list[np.ndarray], int]],
    batch_size: int,
) -> list[np.ndarray]:
    by_length: dict[int, list[tuple[int, int, np.ndarray]]] = defaultdict(list)
    scores = [np.empty(len(windows), dtype=np.float32) for windows, _ in window_groups]
    for group_index, (windows, sample_rate) in enumerate(window_groups):
        for window_index, window in enumerate(windows):
            prepared = _resample_and_pad(window, sample_rate)
            by_length[len(prepared)].append((group_index, window_index, prepared))

    input_name = session.get_inputs()[0].name
    for entries in by_length.values():
        for offset in range(0, len(entries), batch_size):
            chunk = entries[offset : offset + batch_size]
            batch = np.stack([entry[2] for entry in chunk])
            outputs = session.run(None, {input_name: batch})[0].reshape(-1)
            for (group_index, window_index, _), score in zip(chunk, outputs):
                scores[group_index][window_index] = score
    return scores


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_name(row: dict[str, Any], rank: int) -> str:
    pair_id = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in row["item_id"]
    )
    return f"{rank:02d}_{row['language']}_{pair_id}"


def _threshold_name(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def _export_listening_set(
    output_dir: Path,
    name: str,
    selected: list[dict[str, Any]],
    window_seconds: float,
    hop_ratio: float,
) -> None:
    target = output_dir / "listening_samples" / name
    if target.exists():
        shutil.rmtree(target)
    (target / "full_audio").mkdir(parents=True)
    (target / "worst_window").mkdir(parents=True)
    manifest_rows = []
    for rank, row in enumerate(selected, start=1):
        audio, sample_rate, _, starts, _ = _prepare_audio(
            row["audio_path"], window_seconds, hop_ratio
        )
        window_samples = max(1, round(window_seconds * sample_rate))
        stem = _safe_name(row, rank)
        full_path = target / "full_audio" / f"{stem}.wav"
        window_path = target / "worst_window" / f"{stem}.wav"
        sf.write(full_path, audio, sample_rate, subtype="PCM_16")
        worst_index = int(row["worst_window_index"])
        start = starts[worst_index]
        sf.write(
            window_path,
            audio[start : start + window_samples],
            sample_rate,
            subtype="PCM_16",
        )
        manifest_rows.append(
            {
                **row,
                "listening_full_audio": str(full_path.resolve()),
                "listening_worst_window": str(window_path.resolve()),
            }
        )
    _write_jsonl(target / "manifest.jsonl", manifest_rows)


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_rows(
        args.manifest,
        audio_field=args.audio_field,
        id_field=args.id_field,
        text_field=args.text_field,
        language_field=args.language_field,
    )
    if args.language:
        languages = set(args.language)
        rows = [row for row in rows if row["language"] in languages]
    if args.limit is not None:
        rows = rows[: args.limit]

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.cpu_threads
    options.inter_op_num_threads = 1
    providers = ["CPUExecutionProvider"]
    if args.provider == "cuda":
        ort.preload_dlls(directory="")
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(
        str(Path(args.model).resolve()),
        sess_options=options,
        providers=providers,
    )
    if args.provider == "cuda" and session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError("SCOREQ failed to initialize CUDAExecutionProvider")

    scores_path = output_dir / "scores.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if scores_path.exists():
        with scores_path.open(encoding="utf-8") as handle:
            for line in handle:
                result = json.loads(line)
                completed[result["audio_path"]] = result

    pending = [row for row in rows if row["audio_path"] not in completed]
    with scores_path.open("a", encoding="utf-8") as score_handle:
        progress = tqdm(total=len(pending), desc="SCOREQ windows", unit="audio")
        for batch_offset in range(0, len(pending), args.audio_batch_size):
            row_batch = pending[batch_offset : batch_offset + args.audio_batch_size]
            prepared_batch = [
                _prepare_audio(row["audio_path"], args.window_seconds, args.hop_ratio)
                for row in row_batch
            ]
            score_batch = _score_window_groups(
                session,
                [(prepared[4], prepared[1]) for prepared in prepared_batch],
                args.batch_size,
            )
            for row, prepared, scores in zip(row_batch, prepared_batch, score_batch):
                trimmed, sample_rate, trim_start, starts, _ = prepared
                worst_index = int(np.argmin(scores))
                window_samples = max(1, round(args.window_seconds * sample_rate))
                result = {
                    **row,
                    "raw_trim_start_seconds": trim_start / sample_rate,
                    "trimmed_seconds": len(trimmed) / sample_rate,
                    "window_seconds": args.window_seconds,
                    "hop_seconds": args.window_seconds * args.hop_ratio,
                    "window_count": len(scores),
                    "window_min": float(scores[worst_index]),
                    "window_mean": float(np.mean(scores)),
                    "window_median": float(np.median(scores)),
                    "window_p10": float(np.quantile(scores, 0.10)),
                    "worst_window_index": worst_index,
                    "worst_window_start_seconds": starts[worst_index] / sample_rate,
                    "worst_window_end_seconds": min(
                        len(trimmed), starts[worst_index] + window_samples
                    )
                    / sample_rate,
                    "below_min_threshold": bool(
                        scores[worst_index] < args.min_threshold
                    ),
                    "below_mean_threshold": bool(
                        np.mean(scores) < args.mean_threshold
                    ),
                }
                completed[row["audio_path"]] = result
                score_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.update(len(row_batch))
            if progress.n % 100 == 0:
                score_handle.flush()
                os.fsync(score_handle.fileno())
        progress.close()
        score_handle.flush()
        os.fsync(score_handle.fileno())

    results = [completed[row["audio_path"]] for row in rows]

    min_below = sorted(
        (row for row in results if row["below_min_threshold"]),
        key=lambda row: args.min_threshold - row["window_min"],
    )
    mean_below = sorted(
        (row for row in results if row["below_mean_threshold"]),
        key=lambda row: args.mean_threshold - row["window_mean"],
    )
    count = args.listening_count
    _export_listening_set(
        output_dir,
        f"min_just_below_{_threshold_name(args.min_threshold)}",
        min_below[:count],
        args.window_seconds,
        args.hop_ratio,
    )
    _export_listening_set(
        output_dir,
        f"mean_just_below_{_threshold_name(args.mean_threshold)}",
        mean_below[:count],
        args.window_seconds,
        args.hop_ratio,
    )

    language_counts = Counter(row["language"] for row in results)
    min_counts = Counter(
        row["language"] for row in results if row["below_min_threshold"]
    )
    mean_counts = Counter(
        row["language"] for row in results if row["below_mean_threshold"]
    )
    summary = {
        "audio_count": len(results),
        "window_seconds": args.window_seconds,
        "hop_ratio": args.hop_ratio,
        "min_threshold": args.min_threshold,
        "mean_threshold": args.mean_threshold,
        "below_min_count": len(min_below),
        "below_mean_count": len(mean_below),
        "below_either_count": sum(
            row["below_min_threshold"] or row["below_mean_threshold"]
            for row in results
        ),
        "by_language": {
            language: {
                "count": language_counts[language],
                "below_min": min_counts[language],
                "below_mean": mean_counts[language],
            }
            for language in sorted(language_counts)
        },
    }
    _write_jsonl(scores_path, results)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
