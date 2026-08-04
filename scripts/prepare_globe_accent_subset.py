#!/usr/bin/env python3
"""Download a speaker-diverse GLOBE V2 accent subset."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import hydra
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm


DATASET_SERVER = "https://datasets-server.huggingface.co"


def _path(value: str) -> Path:
    return Path(to_absolute_path(os.path.expanduser(value))).resolve()


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _request_headers() -> dict[str, str]:
    headers = {"User-Agent": "lz-tts-globe-subset/1.0"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.is_file():
            token = token_path.read_text(encoding="utf-8").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str) -> dict[str, Any]:
    for attempt in range(10):
        try:
            request = urllib.request.Request(url, headers=_request_headers())
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = json.load(response)
            if payload.get("error"):
                raise RuntimeError(str(payload["error"]))
            return payload
        except Exception:
            if attempt == 9:
                raise
            time.sleep(min(30, 2**attempt))
    raise AssertionError("unreachable")


def _filter_url(cfg: DictConfig, accent: str, offset: int, length: int) -> str:
    params = {
        "dataset": str(cfg.source.dataset),
        "config": str(cfg.source.config),
        "split": str(cfg.source.split),
        "where": f'"accent"=\'{accent}\'',
        "offset": str(offset),
        "length": str(length),
    }
    return f"{DATASET_SERVER}/filter?{urllib.parse.urlencode(params)}"


def _rows_url(cfg: DictConfig, offset: int, length: int) -> str:
    params = {
        "dataset": str(cfg.source.dataset),
        "config": str(cfg.source.config),
        "split": str(cfg.source.split),
        "offset": str(offset),
        "length": str(length),
    }
    return f"{DATASET_SERVER}/rows?{urllib.parse.urlencode(params)}"


def _fetch_accent_rows(cfg: DictConfig, accent: str, cache_dir: Path) -> list[dict[str, Any]]:
    page_size = int(cfg.download.metadata_page_size)
    accent_cache = cache_dir / hashlib.sha256(accent.encode("utf-8")).hexdigest()[:16]
    accent_cache.mkdir(parents=True, exist_ok=True)

    def fetch_page(offset: int) -> dict[str, Any]:
        path = accent_cache / f"{offset:09d}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        payload = _request_json(_filter_url(cfg, accent, offset, page_size))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        return payload

    first = fetch_page(0)
    total = int(first["num_rows_total"])
    pages: dict[int, dict[str, Any]] = {0: first}
    offsets = list(range(page_size, total, page_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.download.metadata_workers)) as pool:
        futures = {
            pool.submit(fetch_page, offset): offset
            for offset in offsets
        }
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"metadata {accent}",
            unit="page",
            dynamic_ncols=True,
        ):
            pages[futures[future]] = future.result()

    rows = []
    for offset in sorted(pages):
        for wrapped in pages[offset]["rows"]:
            row = dict(wrapped["row"])
            audio = row.get("audio") or []
            if not audio or not audio[0].get("src"):
                continue
            row["source_row_index"] = int(wrapped["row_idx"])
            row["audio_url"] = str(audio[0]["src"])
            row.pop("audio", None)
            rows.append(row)
    if len(rows) != total:
        raise RuntimeError(f"Expected {total} {accent} rows, received {len(rows)}")
    return rows


def _fetch_all_rows(cfg: DictConfig, cache_dir: Path) -> list[dict[str, Any]]:
    """Read every metadata row; the dataset server's filtered view is incomplete."""
    page_size = int(cfg.download.metadata_page_size)
    all_cache = cache_dir / "all-rows"
    all_cache.mkdir(parents=True, exist_ok=True)

    def fetch_page(offset: int) -> Path:
        path = all_cache / f"{offset:09d}.json"
        if path.is_file():
            return path
        payload = _request_json(_rows_url(cfg, offset, page_size))
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(temporary, path)
        return path

    first_path = fetch_page(0)
    first = json.loads(first_path.read_text(encoding="utf-8"))
    total = int(first["num_rows_total"])
    offsets = list(range(page_size, total, page_size))
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.download.metadata_workers)) as pool:
        futures = {pool.submit(fetch_page, offset): offset for offset in offsets}
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="metadata full split",
            unit="page",
            dynamic_ncols=True,
        ):
            future.result()

    rows = []
    for offset in range(0, total, page_size):
        page = json.loads((all_cache / f"{offset:09d}.json").read_text(encoding="utf-8"))
        for wrapped in page["rows"]:
            row = dict(wrapped["row"])
            audio = row.get("audio") or []
            if not audio or not audio[0].get("src"):
                continue
            row["source_row_index"] = int(wrapped["row_idx"])
            row["audio_url"] = str(audio[0]["src"])
            row.pop("audio", None)
            rows.append(row)
    if len(rows) != total:
        raise RuntimeError(f"Expected {total} rows, received {len(rows)}")
    return rows


def _speaker_round_robin(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    for speaker_id, speaker_rows in by_speaker.items():
        speaker_rows.sort(
            key=lambda row: _stable_int(
                f"clip:{seed}:{speaker_id}:{row['source_row_index']}:{row.get('transcript', row.get('text', ''))}"
            )
        )
    speaker_order = sorted(
        by_speaker,
        key=lambda speaker_id: _stable_int(f"speaker:{seed}:{speaker_id}"),
    )
    queues = {speaker_id: deque(by_speaker[speaker_id]) for speaker_id in speaker_order}
    selected = []
    while len(selected) < count:
        made_progress = False
        for speaker_id in speaker_order:
            if queues[speaker_id]:
                selected.append(queues[speaker_id].popleft())
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            break
    if len(selected) != count:
        raise ValueError(f"Requested {count} rows but only {len(selected)} are available")
    return selected


def _audio_extension(path: str | None) -> str:
    suffix = Path(path or "").suffix.lower()
    return suffix if suffix in {".flac", ".mp3", ".ogg", ".wav"} else ".wav"


def _write_candidate_audio(
    output_dir: Path,
    locale: str,
    speaker_id: str,
    utterance_id: str,
    audio: dict[str, Any],
) -> Path:
    audio_bytes = audio.get("bytes")
    if not audio_bytes:
        raise ValueError(f"Missing embedded audio for {utterance_id}")
    path = (
        output_dir
        / ".candidate-audio"
        / locale
        / speaker_id
        / f"{utterance_id}{_audio_extension(audio.get('path'))}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(audio_bytes)
    os.replace(temporary, path)
    return path


def _load_candidates(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    candidates: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    if not path.is_file():
        return candidates
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                candidates[str(row["locale"])][str(row["speaker_id"])].append(row)
    return candidates


def _save_candidates(path: Path, candidates: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    rows = [
        row
        for locale in sorted(candidates)
        for speaker_id in sorted(candidates[locale])
        for row in sorted(candidates[locale][speaker_id], key=lambda item: int(item["selection_key"]))
    ]
    _write_jsonl(path, rows)


def _extract_parquet_candidates(cfg: DictConfig, output_dir: Path) -> list[dict[str, Any]]:
    """Scan full shards once and retain a bounded deterministic pool per speaker."""
    state_dir = output_dir / ".parquet-state"
    shard_dir = state_dir / "shards"
    state_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = state_dir / "candidates.jsonl"
    processed_path = state_dir / "processed_shards.json"
    candidates = _load_candidates(candidates_path)
    processed = set(json.loads(processed_path.read_text()) if processed_path.is_file() else [])

    labels_to_locale = {
        str(label): str(locale)
        for locale, locale_cfg in cfg.accents.items()
        for label in locale_cfg.labels
    }
    clips_per_speaker = int(cfg.download.candidate_clips_per_speaker)
    seed = int(cfg.seed)
    split_prefix = f"data/{cfg.source.split}-"
    api = HfApi(token=_request_headers().get("Authorization", "").removeprefix("Bearer ") or None)
    shard_names = sorted(
        name
        for name in api.list_repo_files(str(cfg.source.dataset), repo_type="dataset")
        if name.startswith(split_prefix) and name.endswith(".parquet")
    )
    max_shards = int(cfg.download.get("max_shards", 0))
    if max_shards > 0:
        shard_names = shard_names[:max_shards]
    progress = tqdm(shard_names, desc="GLOBE parquet shards", unit="shard", dynamic_ncols=True)
    for shard_index, shard_name in enumerate(progress):
        if shard_name in processed:
            continue
        local_path = Path(
            hf_hub_download(
                repo_id=str(cfg.source.dataset),
                filename=shard_name,
                repo_type="dataset",
                local_dir=shard_dir,
            )
        )
        table = pq.read_table(local_path)
        columns = {name: table[name] for name in table.column_names}
        accepted_in_shard = 0
        for row_index in range(table.num_rows):
            source_accent = str(columns["accent"][row_index].as_py())
            locale = labels_to_locale.get(source_accent)
            if locale is None:
                continue
            duration = float(columns["duration"][row_index].as_py())
            transcript = " ".join(str(columns["transcript"][row_index].as_py()).split())
            if not transcript or not (
                float(cfg.selection.min_seconds) <= duration <= float(cfg.selection.max_seconds)
            ):
                continue
            speaker_id = str(columns["speaker_id"][row_index].as_py())
            utterance_id = f"globe-v2-{shard_index:05d}-{row_index:05d}"
            selection_key = _stable_int(f"clip:{seed}:{speaker_id}:{utterance_id}:{transcript}")
            speaker_candidates = candidates[locale][speaker_id]
            if len(speaker_candidates) >= clips_per_speaker:
                worst = max(speaker_candidates, key=lambda item: int(item["selection_key"]))
                if selection_key >= int(worst["selection_key"]):
                    continue
                Path(str(worst["candidate_audio"])).unlink(missing_ok=True)
                speaker_candidates.remove(worst)
            audio = columns["audio"][row_index].as_py()
            candidate_path = _write_candidate_audio(
                output_dir, locale, speaker_id, utterance_id, audio
            )
            speaker_candidates.append(
                {
                    "candidate_audio": str(candidate_path),
                    "text": transcript,
                    "duration": duration,
                    "speaker_id": speaker_id,
                    "locale": locale,
                    "source_accent": source_accent,
                    "utterance_id": utterance_id,
                    "source_dataset": str(cfg.source.dataset),
                    "source_split": str(cfg.source.split),
                    "source_shard": shard_name,
                    "source_row_index": row_index,
                    "selection_key": selection_key,
                    "age": columns["age"][row_index].as_py(),
                    "gender": columns["gender"][row_index].as_py(),
                }
            )
            accepted_in_shard += 1
        del columns, table
        local_path.unlink(missing_ok=True)
        processed.add(shard_name)
        _save_candidates(candidates_path, candidates)
        temporary = processed_path.with_name(f".{processed_path.name}.tmp")
        temporary.write_text(json.dumps(sorted(processed)), encoding="utf-8")
        os.replace(temporary, processed_path)
        progress.set_postfix(candidates=sum(len(rows) for locale in candidates.values() for rows in locale.values()), added=accepted_in_shard)

    selected = []
    for locale, locale_cfg in cfg.accents.items():
        locale_rows = [row for rows in candidates[str(locale)].values() for row in rows]
        requested_count = int(locale_cfg.count)
        if len(locale_rows) < requested_count and bool(
            cfg.download.get("allow_fewer_than_target", False)
        ):
            print(
                f"{locale}: requested {requested_count} clips, but only "
                f"{len(locale_rows)} qualifying clips are available; using all available clips",
                flush=True,
            )
            requested_count = len(locale_rows)
        chosen = _speaker_round_robin(locale_rows, requested_count, seed)
        for row in chosen:
            source = Path(str(row["candidate_audio"]))
            destination = output_dir / "audio" / str(locale) / str(row["speaker_id"]) / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.is_file():
                os.replace(source, destination)
            elif not destination.is_file():
                raise FileNotFoundError(source)
            item = dict(row)
            item["audio"] = str(destination)
            item["accent"] = str(locale)
            item.pop("candidate_audio", None)
            item.pop("locale", None)
            item.pop("selection_key", None)
            selected.append(item)
    shutil.rmtree(output_dir / ".candidate-audio", ignore_errors=True)
    return selected


def _download_audio(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    accent = str(row["locale"])
    speaker_id = str(row["speaker_id"])
    utterance_id = f"globe-v2-{int(row['source_row_index']):09d}"
    path = output_dir / "audio" / accent / speaker_id / f"{utterance_id}.flac"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        for attempt in range(10):
            try:
                request = urllib.request.Request(
                    str(row["audio_url"]),
                    headers=_request_headers(),
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    temporary.write_bytes(response.read())
                os.replace(temporary, path)
                break
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 9:
                    raise
                time.sleep(min(30, 2**attempt))
    return {
        "audio": str(path),
        "text": " ".join(str(row["transcript"]).split()),
        "duration": float(row["duration"]),
        "speaker_id": speaker_id,
        "accent": accent,
        "source_accent": str(row["accent"]),
        "utterance_id": utterance_id,
        "source_dataset": "MushanW/GLOBE_V2",
        "source_split": str(row["source_split"]),
        "source_row_index": int(row["source_row_index"]),
        "age": row.get("age"),
        "gender": row.get("gender"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


@hydra.main(version_base=None, config_path="../local/configs/voxcpm", config_name="globe_accent_subset")
def main(cfg: DictConfig) -> None:
    output_dir = _path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = output_dir / "source.jsonl"
    if source_manifest.is_file() and (output_dir / "summary.json").is_file():
        print((output_dir / "summary.json").read_text(encoding="utf-8"))
        return
    seed = int(cfg.seed)
    source_split = str(cfg.source.split)
    metadata_mode = str(cfg.download.get("metadata_mode", "filtered"))
    if metadata_mode == "parquet_shards":
        downloaded = _extract_parquet_candidates(cfg, output_dir)
    else:
        selected = []
        all_rows = None
        if metadata_mode == "full_scan":
            all_rows = _fetch_all_rows(cfg, output_dir / ".metadata-cache")
        elif metadata_mode != "filtered":
            raise ValueError(f"Unsupported metadata_mode: {metadata_mode}")
        for locale, locale_cfg in cfg.accents.items():
            labels = {str(label) for label in locale_cfg.labels}
            if all_rows is None:
                candidates = []
                for label in labels:
                    candidates.extend(_fetch_accent_rows(cfg, label, output_dir / ".metadata-cache"))
            else:
                candidates = [row for row in all_rows if str(row["accent"]) in labels]
            candidates = [
                row
                for row in candidates
                if float(cfg.selection.min_seconds) <= float(row["duration"]) <= float(cfg.selection.max_seconds)
                and str(row["transcript"]).strip()
            ]
            locale_rows = _speaker_round_robin(candidates, int(locale_cfg.count), seed)
            for row in locale_rows:
                row["locale"] = str(locale)
                row["source_split"] = source_split
            selected.extend(locale_rows)

        downloaded = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(cfg.download.audio_workers)) as pool:
            futures = [pool.submit(_download_audio, row, output_dir) for row in selected]
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="GLOBE audio",
                unit="clip",
                dynamic_ncols=True,
            ):
                downloaded.append(future.result())
    downloaded.sort(key=lambda row: (row["accent"], row["speaker_id"], row["utterance_id"]))
    _write_jsonl(output_dir / "source.jsonl", downloaded)
    summary = {
        "items": len(downloaded),
        "speakers": len({row["speaker_id"] for row in downloaded}),
        "by_accent": {
            locale: {
                "items": sum(row["accent"] == locale for row in downloaded),
                "speakers": len({row["speaker_id"] for row in downloaded if row["accent"] == locale}),
                "hours": sum(row["duration"] for row in downloaded if row["accent"] == locale) / 3600,
            }
            for locale in sorted({row["accent"] for row in downloaded})
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
