#!/usr/bin/env python3
"""Collect reference audio for every voice and expression in voices.json."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOICES_JSON = PROJECT_ROOT / "local" / "voices.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local" / "premade_voice_clone_samples"
DEFAULT_CELEBRITY_DIR = PROJECT_ROOT / "data" / "seed-vc" / "voice-samples" / "celebrities"
STYLE_SAMPLE_BASE_URL = "https://cdn.lazybird.app/global/voice-style-samples"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--voices-json", type=Path, default=DEFAULT_VOICES_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--celebrity-dir", type=Path, default=DEFAULT_CELEBRITY_DIR)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=30.0)
    return parser.parse_args()


def _safe_component(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def _derived_style_url(voice_id: str, style: str) -> str:
    voice_component = urllib.parse.quote(voice_id, safe=".-")
    style_component = urllib.parse.quote(style, safe=".-")
    return f"{STYLE_SAMPLE_BASE_URL}/{voice_component}/{style_component}.mp3"


def _style_declarations(voice: dict[str, Any]) -> dict[str, set[str]]:
    declarations: dict[str, set[str]] = {"general": {"general"}}
    for field in ("styleList", "rolePlayList"):
        for style in voice.get(field) or []:
            declarations.setdefault(str(style), set()).add(field)
    for style, enabled in (voice.get("vttsStyles") or {}).items():
        if enabled:
            declarations.setdefault(str(style), set()).add("vttsStyles")
    for style in (voice.get("styleSamples") or {}):
        declarations.setdefault(str(style), set()).add("styleSamples")
    return declarations


def _catalog_tasks(
    voices: list[dict[str, Any]],
    output_dir: Path,
    celebrity_dir: Path,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for voice in voices:
        voice_id = str(voice["id"])
        explicit_styles = voice.get("styleSamples") or {}
        sample_url = voice.get("sampleUrl")
        local_general = celebrity_dir / f"{voice_id}.wav"
        clone_enabled = bool(voice.get("useVTTS") or voice.get("engine") == "lzrv")
        declarations = _style_declarations(voice)

        for style in sorted(declarations, key=lambda value: (value != "general", value)):
            source_type = "derived_url"
            source: str | Path = _derived_style_url(voice_id, style)
            extension = ".mp3"

            if style in explicit_styles:
                source_type = "explicit_style_url"
                source = str(explicit_styles[style])
                extension = Path(urllib.parse.urlparse(source).path).suffix or ".mp3"
            elif style == "general" and sample_url:
                source_type = "explicit_sample_url"
                source = str(sample_url)
                extension = Path(urllib.parse.urlparse(source).path).suffix or ".wav"
            elif style == "general" and local_general.is_file():
                source_type = "existing_local"
                source = local_general
                extension = local_general.suffix

            vtts_styles = voice.get("vttsStyles") or {}
            required = (
                style in explicit_styles
                or bool(vtts_styles.get(style))
                or (
                    style == "general"
                    and clone_enabled
                    and vtts_styles.get("general", True) is not False
                )
            )
            destination = (
                output_dir
                / "audio"
                / _safe_component(voice_id)
                / f"{_safe_component(style)}{extension.lower()}"
            )
            tasks.append(
                {
                    "voice_id": voice_id,
                    "display_name": voice.get("displayName") or voice.get("name"),
                    "language": voice.get("language"),
                    "engine": voice.get("engine"),
                    "clone_enabled": clone_enabled,
                    "style": style,
                    "declared_by": sorted(declarations[style]),
                    "required": required,
                    "source_type": source_type,
                    "source": str(source),
                    "destination": str(destination),
                }
            )
    return tasks


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_download(url: str, destination: Path, timeout: float) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "lz-tts-voice-sample-collector/1.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get_content_type()
        data = response.read()
    if not data:
        raise RuntimeError("empty response")
    if not (content_type.startswith("audio/") or data[:4] in {b"RIFF", b"ID3\x03", b"ID3\x04"}):
        raise RuntimeError(f"unexpected content type {content_type!r}")

    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.name}.",
        suffix=".part",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return content_type, len(data)


def _collect_one(task: dict[str, Any], timeout: float) -> dict[str, Any]:
    destination = Path(task["destination"])
    result = dict(task)
    if destination.is_file() and destination.stat().st_size > 0:
        result.update(
            status="available",
            bytes=destination.stat().st_size,
            sha256=_sha256(destination),
            cached=True,
        )
        return result

    try:
        if task["source_type"] == "existing_local":
            _atomic_copy(Path(task["source"]), destination)
            content_type = f"audio/{destination.suffix.lstrip('.')}"
            byte_count = destination.stat().st_size
        else:
            content_type, byte_count = _atomic_download(
                task["source"],
                destination,
                timeout,
            )
        result.update(
            status="available",
            content_type=content_type,
            bytes=byte_count,
            sha256=_sha256(destination),
            cached=False,
        )
    except urllib.error.HTTPError as exc:
        result.update(status="missing", reason=f"http_{exc.code}")
    except urllib.error.URLError as exc:
        result.update(status="error", reason=f"url_error:{exc.reason}")
    except Exception as exc:
        result.update(status="error", reason=f"{type(exc).__name__}:{exc}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_report(
    path: Path,
    results: list[dict[str, Any]],
    voices_without_samples: list[dict[str, Any]],
) -> None:
    available_by_voice: dict[str, list[str]] = {}
    missing_by_voice: dict[str, list[str]] = {}
    metadata_by_voice: dict[str, dict[str, Any]] = {}
    for row in results:
        metadata_by_voice[row["voice_id"]] = row
        if row["status"] != "available" and not row["required"]:
            continue
        destination = available_by_voice if row["status"] == "available" else missing_by_voice
        destination.setdefault(row["voice_id"], []).append(row["style"])

    clone_enabled_without_samples = [
        row for row in voices_without_samples if row["clone_enabled"]
    ]
    clone_enabled_partial = sorted(
        voice_id
        for voice_id in set(available_by_voice) & set(missing_by_voice)
        if metadata_by_voice[voice_id]["clone_enabled"]
    )
    speculative_without_samples = [
        row for row in voices_without_samples if not row["clone_enabled"]
    ]

    lines = [
        "# Premade Voice Clone Sample Audit",
        "",
        "## Actionable Missing Voices",
        "",
        "These voices are clone-enabled but have no available reference sample:",
        "",
    ]
    if clone_enabled_without_samples:
        lines.extend(
            f"- `{row['voice_id']}` ({row.get('language') or 'unknown language'}): "
            f"{', '.join(missing_by_voice[row['voice_id']])}"
            for row in clone_enabled_without_samples
        )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Partial Expression Coverage",
            "",
            "These clone-enabled voices have a usable sample, but one or more declared "
            "expression or role-play variants could not be found:",
            "",
        ]
    )
    if clone_enabled_partial:
        lines.extend(
            f"- `{voice_id}`: available [{', '.join(available_by_voice[voice_id])}]; "
            f"missing [{', '.join(missing_by_voice[voice_id])}]"
            for voice_id in clone_enabled_partial
        )
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Non-Clone-Enabled Catalog Entries",
            "",
            f"{len(speculative_without_samples)} Azure catalog voices have no inferred "
            "CDN sample, but they have `useVTTS: false`; these are not treated as "
            "production clone-sample gaps. Their complete list is in "
            "`voices_without_samples.jsonl`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    payload = json.loads(args.voices_json.read_text(encoding="utf-8"))
    voices = payload["voices"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = _catalog_tasks(voices, args.output_dir, args.celebrity_dir)

    results: list[dict[str, Any]] = []
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_collect_one, task, args.timeout): task
            for task in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                counts = Counter(row["status"] for row in results)
                print(
                    f"\rCollected {completed}/{len(tasks)} "
                    f"available={counts['available']} missing={counts['missing']} "
                    f"errors={counts['error']}",
                    end="",
                    flush=True,
                )
    print()

    results.sort(key=lambda row: (row["voice_id"], row["style"]))
    missing = [row for row in results if row["status"] != "available"]
    available_voice_ids = {
        row["voice_id"] for row in results if row["status"] == "available"
    }
    voices_without_samples = [
        {
            "voice_id": voice["id"],
            "display_name": voice.get("displayName") or voice.get("name"),
            "language": voice.get("language"),
            "engine": voice.get("engine"),
            "clone_enabled": bool(
                voice.get("useVTTS") or voice.get("engine") == "lzrv"
            ),
            "reason": "no_available_clone_sample",
        }
        for voice in voices
        if voice["id"] not in available_voice_ids
    ]
    status_counts = Counter(row["status"] for row in results)
    source_counts = Counter(
        row["source_type"] for row in results if row["status"] == "available"
    )
    clone_enabled_voice_ids = {
        voice["id"]
        for voice in voices
        if voice.get("useVTTS") or voice.get("engine") == "lzrv"
    }
    clone_enabled_with_samples = clone_enabled_voice_ids & available_voice_ids
    clone_enabled_missing = [
        row
        for row in missing
        if row["voice_id"] in clone_enabled_voice_ids and row["required"]
    ]
    missing_required = [row for row in missing if row["required"]]
    summary = {
        "voices_in_catalog": len(voices),
        "clone_enabled_voices": len(clone_enabled_voice_ids),
        "clone_enabled_voices_with_any_sample": len(clone_enabled_with_samples),
        "clone_enabled_voices_without_any_sample": len(
            clone_enabled_voice_ids - available_voice_ids
        ),
        "clone_enabled_missing_voice_style_samples": len(clone_enabled_missing),
        "probed_voice_style_samples": len(tasks),
        "available_voice_style_samples": status_counts["available"],
        "missing_voice_style_samples": status_counts["missing"],
        "missing_required_voice_style_samples": len(missing_required),
        "errored_voice_style_samples": status_counts["error"],
        "voices_with_any_sample": len(available_voice_ids),
        "voices_without_any_sample": len(voices_without_samples),
        "available_by_source_type": dict(sorted(source_counts.items())),
    }

    _write_jsonl(args.output_dir / "manifest.jsonl", results)
    _write_jsonl(args.output_dir / "missing.jsonl", missing)
    _write_jsonl(args.output_dir / "voices_without_samples.jsonl", voices_without_samples)
    _write_report(
        args.output_dir / "REPORT.md",
        results,
        voices_without_samples,
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
