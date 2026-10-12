"""Batched GPU language identification for the voice language backfill.

Scores many voice samples with the same models as the ``detect-language``
task (src/api/language_id.py), but in one process: the models load once on
the GPU, a thread pool downloads and decodes clips, and clips are scored in
length-sorted padded batches. Thousands of samples take minutes instead of
one Taskflow round trip each.

Input is JSONL, one ``{"id": ..., "url": ...}`` per line (Lazybird:
``tools backfillVoiceLanguages --input '{"export": "<file>"}'``). Output is
JSONL, one line per input id: ``{"id", "url", "result"}`` with the
``identify`` result, or ``{"id", "url", "error"}``. The output is appended to
and ids already in it are skipped, so an interrupted run resumes. Feed it
back with ``tools backfillVoiceLanguages --input '{"file": "<file>"}'``.

    uv run python scripts/backfill_language_id.py targets.jsonl results.jsonl
    uv run python scripts/backfill_language_id.py targets.jsonl results.jsonl --max-batch-seconds 960 --workers 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.language_id import (  # noqa: E402
    SAMPLE_RATE,
    TOP_K,
    LanguageInputError,
    SpokenLanguageIdentifier,
    decode_audio_16k_mono,
)

_LOGGER = logging.getLogger("backfill_language_id")


def _read_targets(path: Path, done: set[str]) -> list[dict[str, str]]:
    targets = []
    seen = set(done)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        target = json.loads(line)
        if target["id"] in seen:
            continue
        seen.add(target["id"])
        targets.append({"id": target["id"], "url": target["url"]})
    return targets


def _read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            done.add(json.loads(line)["id"])
    return done


def _fetch(client: httpx.Client, target: dict[str, str]) -> np.ndarray:
    response = client.get(target["url"])
    response.raise_for_status()
    return decode_audio_16k_mono(response.content)


def _decoded(
    targets: list[dict[str, str]], workers: int, chunk: int
) -> Iterator[list[tuple[dict[str, str], np.ndarray | Exception]]]:
    """Yield chunks of (target, clip or error), decoding the next chunk while
    the caller scores the current one."""
    with (
        httpx.Client(follow_redirects=True, timeout=60.0, limits=httpx.Limits(max_connections=workers)) as client,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):

        def submit(start: int) -> list[tuple[dict[str, str], Future[np.ndarray]]]:
            return [(t, pool.submit(_fetch, client, t)) for t in targets[start : start + chunk]]

        pending = submit(0)
        for start in range(chunk, len(targets) + chunk, chunk):
            upcoming = submit(start) if start < len(targets) else []
            results: list[tuple[dict[str, str], np.ndarray | Exception]] = []
            for target, future in pending:
                try:
                    results.append((target, future.result()))
                except Exception as exc:  # noqa: BLE001 — recorded per id
                    results.append((target, exc))
            yield results
            pending = upcoming


def _batches(
    clips: list[tuple[dict[str, str], np.ndarray]], max_clips: int, max_seconds: float
) -> Iterator[list[tuple[dict[str, str], np.ndarray]]]:
    """Length-sorted batches bounded by clip count and padded audio seconds
    (memory grows with batch size x longest clip)."""
    batch: list[tuple[dict[str, str], np.ndarray]] = []
    for item in sorted(clips, key=lambda item: item[1].size):
        # Sorted ascending: the newest clip is the longest, so padded size is
        # (len(batch) + 1) * its length.
        padded_seconds = (len(batch) + 1) * item[1].size / SAMPLE_RATE
        if batch and (len(batch) >= max_clips or padded_seconds > max_seconds):
            yield batch
            batch = []
        batch.append(item)
    if batch:
        yield batch


def _identify(
    identifier: SpokenLanguageIdentifier, batch: list[tuple[dict[str, str], np.ndarray]], top_k: int
) -> list[dict[str, Any]]:
    """Score a batch, halving it on CUDA OOM."""
    try:
        return identifier.identify_samples([clip for _, clip in batch], top_k)
    except torch.OutOfMemoryError:
        if len(batch) == 1:
            raise
        torch.cuda.empty_cache()
        half = len(batch) // 2
        _LOGGER.warning("CUDA OOM on %d clips; splitting", len(batch))
        return _identify(identifier, batch[:half], top_k) + _identify(identifier, batch[half:], top_k)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", type=Path, help="JSONL of {id, url}")
    parser.add_argument("output", type=Path, help="JSONL results (appended; resumes)")
    parser.add_argument("--batch-size", type=int, default=64, help="max clips per GPU batch")
    parser.add_argument(
        "--max-batch-seconds", type=float, default=480.0, help="max padded audio seconds per GPU batch"
    )
    parser.add_argument("--workers", type=int, default=16, help="concurrent downloads/decodes")
    parser.add_argument("--chunk", type=int, default=256, help="clips decoded ahead per step")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    done = _read_done(args.output)
    targets = _read_targets(args.targets, done)
    _LOGGER.info("targets=%d already-done=%d device=%s", len(targets), len(done), args.device)
    if not targets:
        return

    identifier = SpokenLanguageIdentifier(device=args.device)
    started = time.monotonic()
    processed = failed = 0
    with args.output.open("a") as out:
        for chunk in _decoded(targets, args.workers, args.chunk):
            ok: list[tuple[dict[str, str], np.ndarray]] = []
            for target, clip in chunk:
                if isinstance(clip, Exception):
                    kind = "input" if isinstance(clip, LanguageInputError) else type(clip).__name__
                    out.write(json.dumps({**target, "error": f"{kind}: {clip}"[:500]}) + "\n")
                    failed += 1
                else:
                    ok.append((target, clip))
            # Similar lengths per batch keep padding (wasted compute) low.
            for batch in _batches(ok, args.batch_size, args.max_batch_seconds):
                results = _identify(identifier, batch, args.top_k)
                for (target, _), result in zip(batch, results, strict=True):
                    out.write(json.dumps({**target, "result": result}) + "\n")
            out.flush()
            processed += len(chunk)
            elapsed = time.monotonic() - started
            _LOGGER.info(
                "progress %d/%d failed=%d %.1f clips/s",
                processed,
                len(targets),
                failed,
                processed / max(elapsed, 1e-6),
            )
    _LOGGER.info("done processed=%d failed=%d in %.1fs", processed, failed, time.monotonic() - started)


if __name__ == "__main__":
    main()
