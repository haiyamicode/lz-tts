#!/usr/bin/env python3
"""Build the explicit production voice preset catalog."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _language_from_voice_id(voice_id: str) -> str:
    parts = voice_id.split(".")
    if len(parts) < 3 or not parts[1]:
        raise ValueError(f"Cannot derive language from voice id {voice_id!r}")
    return parts[1]


def _reference_source(value: str) -> dict[str, str]:
    if value.startswith(("https://", "http://")):
        return {"url": value}

    path = Path(value)
    if path.is_absolute():
        try:
            path = path.relative_to(PROJECT_ROOT)
        except ValueError as exc:
            raise ValueError(f"Reference path is outside the project: {value}") from exc
    return {"path": path.as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("local/premade_voice_clone_samples/manifest.jsonl"),
    )
    parser.add_argument(
        "--fallback-embeddings",
        type=Path,
        default=Path("data/seed-vc/embeddings/vtts_embeddings_sparrow_fallback.h5"),
    )
    parser.add_argument(
        "--celebrity-voice-samples",
        type=Path,
        default=Path("data/voice-samples/celebrities"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/voice-presets.json"))
    args = parser.parse_args()

    with h5py.File(args.fallback_embeddings, "r") as embeddings:
        fallback_embedding_keys = sorted(embeddings.keys())
    fallback_voice_ids: set[str] = set()
    for embedding_key in fallback_embedding_keys:
        if not embedding_key.endswith(".general"):
            raise ValueError(f"Unsupported fallback embedding key {embedding_key!r}")
        fallback_voice_ids.add(embedding_key.removesuffix(".general"))

    references: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    engines: dict[str, str] = {}
    with args.manifest.open(encoding="utf-8") as manifest_file:
        for line_number, line in enumerate(manifest_file, start=1):
            if not line.strip():
                continue
            entry: dict[str, Any] = json.loads(line)
            if entry.get("status") != "available":
                continue
            voice_id = entry.get("voice_id")
            style = entry.get("style")
            source = entry.get("source")
            if not all(isinstance(value, str) and value for value in (voice_id, style, source)):
                raise ValueError(f"Malformed available entry at {args.manifest}:{line_number}")
            # Unsupported VoxCPM languages use Sparrow for speech generation and
            # Seed-VC for non-root timbres. Their cached embedding takes priority
            # over any reference sample that could otherwise create a VoxCPM preset.
            if voice_id in fallback_voice_ids:
                continue
            local_celebrity_sample = args.celebrity_voice_samples / f"{voice_id}.wav"
            if style == "general" and local_celebrity_sample.is_file():
                source = local_celebrity_sample.as_posix()
            references[voice_id][style] = _reference_source(source)
            engine = entry.get("engine")
            if isinstance(engine, str) and engine:
                engines[voice_id] = engine

    voices: list[dict[str, Any]] = []
    for voice_id, style_references in sorted(references.items()):
        if "general" not in style_references:
            continue
        voice = {
            "id": voice_id,
            "language": _language_from_voice_id(voice_id),
            "pipeline": "voxcpm",
            "references": dict(sorted(style_references.items())),
        }
        if engines.get(voice_id) == "lzrv":
            voice["style_fallback"] = "general"
        voices.append(voice)

    for voice_id in sorted(fallback_voice_ids):
        voices.append(
            {
                "id": voice_id,
                "language": _language_from_voice_id(voice_id),
                "pipeline": "sparrow_seed_vc",
                "embedding_style": "general",
            }
        )

    voice_ids = [entry["id"] for entry in voices]
    duplicates = sorted({voice_id for voice_id in voice_ids if voice_ids.count(voice_id) > 1})
    if duplicates:
        raise ValueError(f"Voices occur in multiple pipelines: {duplicates}")

    payload = {"version": 1, "voices": sorted(voices, key=lambda entry: entry["id"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(args.output)

    pipeline_counts = {
        pipeline: sum(entry["pipeline"] == pipeline for entry in voices)
        for pipeline in ("voxcpm", "sparrow_seed_vc")
    }
    print(f"Created {args.output}: {pipeline_counts}")


if __name__ == "__main__":
    main()
