#!/usr/bin/env python3
"""Create Seed-VC embeddings for Sparrow languages unsupported by VoxCPM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py


# VoxCPM2's advertised language set, normalized to the base codes used by the
# serving configuration. Chinese dialect codes are included because they are
# routed through VoxCPM's documented Chinese dialect support.
VOXCPM_SUPPORTED_BASE_LANGUAGES = {
    "ar", "da", "de", "el", "en", "es", "fi", "fil", "fr", "he", "hi",
    "id", "it", "ja", "km", "ko", "lo", "ms", "my", "nb", "nl", "pl",
    "pt", "ru", "sv", "sw", "th", "tr", "vi", "wuu", "yue", "zh",
}


def _base_language(locale: str) -> str:
    return locale.strip().lower().split("-", maxsplit=1)[0]


def _fallback_languages(server_config_path: Path) -> set[str]:
    config = json.loads(server_config_path.read_text(encoding="utf-8"))
    language_map = config.get("pipertts", {}).get("lang_speaker_map")
    if not isinstance(language_map, dict):
        raise ValueError(f"{server_config_path} has no pipertts.lang_speaker_map")
    sparrow_languages = {
        _base_language(language)
        for language in (*language_map.keys(), *language_map.values())
        if isinstance(language, str) and language
    }
    return sparrow_languages - VOXCPM_SUPPORTED_BASE_LANGUAGES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--supplemental-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("local/voices.json"))
    parser.add_argument("--server-config", type=Path, default=Path("local/server.json"))
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Override the derived Sparrow-minus-VoxCPM language set",
    )
    args = parser.parse_args()

    languages = (
        {_base_language(language) for language in args.languages}
        if args.languages
        else _fallback_languages(args.server_config)
    )
    payload = json.loads(args.catalog.read_text(encoding="utf-8"))
    voices = payload.get("voices")
    if not isinstance(voices, list):
        raise ValueError(f"{args.catalog} must contain a voices array")

    voice_ids = {
        entry["id"]
        for entry in voices
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and _base_language(str(entry.get("language") or "")) in languages
    }
    if not voice_ids:
        raise ValueError(f"No voices in {args.catalog} match languages {sorted(languages)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.unlink(missing_ok=True)

    copied_keys: list[str] = []
    matched_voice_ids: set[str] = set()
    try:
        source_paths = [args.source, *args.supplemental_source]
        with h5py.File(temporary_output, "w") as output:
            for source_path in source_paths:
                with h5py.File(source_path, "r") as source:
                    for voice_id in sorted(voice_ids - matched_voice_ids):
                        keys = [
                            key
                            for key in source.keys()
                            if key == voice_id or key.startswith(f"{voice_id}.")
                        ]
                        if not keys:
                            continue
                        for key in sorted(keys):
                            if key in output:
                                raise RuntimeError(f"Duplicate embedding key {key!r} in {source_path}")
                            source.copy(key, output)
                            copied_keys.append(key)
                        matched_voice_ids.add(voice_id)

        missing = sorted(voice_ids - matched_voice_ids)
        if missing:
            raise RuntimeError(f"Source embeddings are missing catalog voices: {missing}")
        if not copied_keys:
            raise RuntimeError("No embeddings were copied")
        temporary_output.replace(args.output)
    except Exception:
        temporary_output.unlink(missing_ok=True)
        raise

    size_mib = args.output.stat().st_size / (1024**2)
    print(
        f"Created {args.output}: voices={len(matched_voice_ids)} "
        f"languages={len(languages)} embeddings={len(copied_keys)} size={size_mib:.1f} MiB"
    )
    print(f"Fallback languages: {', '.join(sorted(languages))}")


if __name__ == "__main__":
    main()
