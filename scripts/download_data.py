#!/usr/bin/env python3
"""
Download (sync) data files from Wasabi S3

Downloads model data from S3.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --filter lzspeech
    uv run python scripts/download_data.py --data-dir ./data
    uv run python scripts/download_data.py --dry-run
    uv run python scripts/download_data.py --parallel 8
    uv run python scripts/download_data.py --stage required   # then --stage deferred

Model sets (src/model_sets.py): LZAI_MODELS / LZAI_REQUIRED_MODELS pick what downloads.
"""
import argparse
import hashlib
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import boto3
import boto3.s3.transfer
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import model_sets  # noqa: E402  (stdlib only; runs before uv sync)

PRIORITY_DATA_DIRS = ("runtime-wheels",)
DEFAULT_PARALLELISM = 8

_thread_local = threading.local()


def get_s3_client():
    """One S3 client per thread (boto3 clients are not guaranteed thread-safe)."""
    if not hasattr(_thread_local, "client"):
        import botocore.config
        _thread_local.client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            config=botocore.config.Config(
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
    return _thread_local.client


# ---------------------------------------------------------------------------
# ETag / skip helpers
# ---------------------------------------------------------------------------

def _multipart_md5(file_path: str, part_size: int = 8 * 1024 * 1024) -> tuple[str, int]:
    file_size = os.path.getsize(file_path)
    part_count = max(1, (file_size + part_size - 1) // part_size)
    part_md5s = []
    with open(file_path, "rb") as f:
        for _ in range(part_count):
            part_md5s.append(hashlib.md5(f.read(part_size)).digest())
    if part_count == 1:
        return part_md5s[0].hex(), 1
    return hashlib.md5(b"".join(part_md5s)).hexdigest(), part_count


def _local_etag(file_path: str) -> str:
    md5_hex, part_count = _multipart_md5(file_path)
    if part_count == 1:
        return md5_hex
    return f"{md5_hex}-{part_count}"


def _should_skip(local_path: Path, s3_obj: dict, force: bool) -> bool:
    if force or not local_path.exists():
        return False
    s3_etag = s3_obj.get("ETag", "").strip('"')
    if not s3_etag:
        return local_path.stat().st_size == s3_obj["Size"]
    return _local_etag(str(local_path)) == s3_etag


def download_file(bucket: str, s3_key: str, local_path: Path, size_mb: float = 0.0) -> bool:
    """Download one file from S3 to a temp file, then atomically move into place."""
    try:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        s3 = get_s3_client()

        cfg = boto3.s3.transfer.TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=4,
        )

        tmp_path = local_path.with_suffix(local_path.suffix + ".part")
        # botocore writes multipart downloads to <tmp_path>.<uuid8>; clean stale leftovers
        for stale in local_path.parent.glob(local_path.name + ".part.*"):
            stale.unlink(missing_ok=True)
        if tmp_path.exists():
            tmp_path.unlink()
        s3.download_file(bucket, s3_key, str(tmp_path), Config=cfg)
        os.replace(tmp_path, local_path)
        print(f"  done: {s3_key} ({size_mb:.1f} MB)", flush=True)
        return True
    except (ClientError, OSError) as e:
        print(f"  Error downloading {s3_key}: {e}", flush=True)
        return False


# ---------------------------------------------------------------------------
# Sync from lz-tts data prefix
# ---------------------------------------------------------------------------

_EXCLUDE_PREFIXES = (
    "seed-vc/checkpoints/",
)


def _is_excluded(rel: str) -> bool:
    # Retired model; older uploads may still sit in the bucket.
    if rel.startswith("acestep-xl-turbo/"):
        return True
    # HF cache blobs/ duplicate snapshots/ (see upload_data.py).
    return "/hf_cache/" in f"/{rel}" and "/blobs/" in f"/{rel}"



def _dir_filter(stage: str) -> tuple[Callable[[str], bool], list[str]]:
    """Which top-level data dirs a stage downloads, and the deferred sets (for markers).

    ``all``: core + LZAI_MODELS. ``required``: core + LZAI_REQUIRED_MODELS.
    Dirs claimed only by sets outside LZAI_MODELS are never downloaded.
    """
    models = model_sets.selected_sets()
    required = model_sets.required_sets()
    deferred = [name for name in models if name not in required]
    stage_sets = set(required if stage == "required" else models)
    print(f"Model sets: {', '.join(models) or 'none'} (required: {', '.join(required) or 'none'})")

    def wanted(dir_name: str) -> bool:
        owners = model_sets.set_of_dir(dir_name)
        return not owners or bool(owners & stage_sets)

    return wanted, deferred


def _list_objects(bucket: str, s3_data_path: str) -> list[tuple[dict, str]]:
    s3 = get_s3_client()
    print(f"Listing s3://{bucket}/{s3_data_path}/")
    paginator = s3.get_paginator('list_objects_v2')
    objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{s3_data_path}/"):
        for obj in page.get('Contents', []):
            rel = obj['Key'][len(s3_data_path) + 1:]
            if rel:
                objects.append((obj, rel))
    return objects


def _sync_objects(
    bucket: str,
    objects: list[tuple[dict, str]],
    local_data_dir: Path,
    force: bool,
    dry_run: bool,
    parallel: int,
) -> int:
    priority_index = {name: idx for idx, name in enumerate(PRIORITY_DATA_DIRS)}
    objects = sorted(
        objects,
        key=lambda item: (
            priority_index.get(item[1].split("/")[0], len(priority_index)),
            item[1].split("/")[0],
            item[1],
        ),
    )
    print(f"Found {len(objects)} files → {local_data_dir}/\n")

    # Pass 1: skip checks (local etag compare) — single-threaded, fast
    to_download = []
    skipped = 0
    for obj, rel in objects:
        if _is_excluded(rel) or any(rel.startswith(p) for p in _EXCLUDE_PREFIXES):
            print(f"Skipping {rel} (handled separately)")
            skipped += 1
            continue

        local_path = local_data_dir / rel
        size_mb = obj['Size'] / (1024 * 1024)

        if _should_skip(local_path, obj, force):
            print(f"Skipping {rel} (same content)")
            skipped += 1
            continue

        if dry_run:
            print(f"  WOULD DOWNLOAD {rel} ({size_mb:.1f} MB)")
        to_download.append((obj, rel, size_mb))

    if dry_run:
        total_mb = sum(m for _, _, m in to_download)
        print(f"\nDone: {len(to_download)} files ({total_mb:.0f} MB) to download, {skipped} skipped (dry run)")
        return 0

    if not to_download:
        print(f"\nDone: 0 downloaded, {skipped} skipped")
        return 0

    total_mb = sum(m for _, _, m in to_download)
    print(f"Downloading {len(to_download)} files ({total_mb:.1f} MB) with {parallel} parallel workers:\n")

    # Pass 2: parallel downloads
    downloaded = failed = 0
    done_mb = 0.0
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(download_file, bucket, obj['Key'], local_data_dir / rel, size_mb): (rel, size_mb)
            for obj, rel, size_mb in to_download
        }
        for fut in as_completed(futures):
            rel, size_mb = futures[fut]
            if fut.result():
                downloaded += 1
                done_mb += size_mb
                print(f"  [{downloaded + failed}/{len(futures)}] {done_mb:.0f}/{total_mb:.0f} MB total", flush=True)
            else:
                failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def sync_data_from_s3(
    data_dir: Path | None = None,
    name_filter: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    parallel: int = DEFAULT_PARALLELISM,
    stage: str = "all",
):
    """Sync data from S3.

    ``stage``: ``all`` (core + LZAI_MODELS), ``required`` (core + LZAI_REQUIRED_MODELS; marks
    the other LZAI_MODELS sets pending in data/.pending/), ``deferred`` (the pending sets one by
    one, each marker removed once its set is complete).
    """
    bucket = os.getenv('AWS_S3_BUCKET_NAME')
    s3_data_path = os.getenv('S3_DATA_PATH', 'lz-tts/data')
    local_data_dir = data_dir or Path('./data')

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    local_data_dir.mkdir(parents=True, exist_ok=True)
    try:
        wanted, deferred = _dir_filter(stage)
    except ValueError as error:
        print(f"Error: {error}")
        return 2

    try:
        print("Connecting to Wasabi S3...")
        objects = [
            (obj, rel)
            for obj, rel in _list_objects(bucket, s3_data_path)
            if not name_filter or name_filter in rel.split('/')[0]
        ]

        if stage != "deferred":
            if stage == "required" and not dry_run:
                pending = model_sets.pending_dir(local_data_dir)
                pending.mkdir(parents=True, exist_ok=True)
                for marker in pending.iterdir():
                    marker.unlink()
                for name in deferred:
                    (pending / name).touch()
            selected = [(obj, rel) for obj, rel in objects if wanted(rel.split('/')[0])]
            if not selected:
                print(f"No matching files (filter: {name_filter})")
                return 0
            return _sync_objects(bucket, selected, local_data_dir, force, dry_run, parallel)

        pending = model_sets.pending_sets(local_data_dir)
        # Dirs shared with a required set (or an earlier deferred one) are already synced:
        # don't re-hash them (comfyui is 43 GB).
        required = set(model_sets.required_sets())
        synced = {rel.split('/')[0] for _, rel in objects if model_sets.set_of_dir(rel.split('/')[0]) & required}
        for name in deferred:
            if name not in pending and not dry_run:
                continue
            print(f"\n=== Model set {name} ===")
            dirs = {rel.split('/')[0] for _, rel in objects if name in model_sets.set_of_dir(rel.split('/')[0])}
            selected = [(obj, rel) for obj, rel in objects if rel.split('/')[0] in dirs - synced]
            synced |= dirs
            rc = _sync_objects(bucket, selected, local_data_dir, force, dry_run, parallel)
            if rc != 0:
                return rc
            if not dry_run:
                (model_sets.pending_dir(local_data_dir) / name).unlink(missing_ok=True)
                print(f"Model set {name} ready", flush=True)
        return 0

    except ClientError as e:
        print(f"S3 error: {e}")
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LZ-TTS data from Wasabi S3."
    )
    parser.add_argument("--data-dir", default=None, help="Destination directory (default: ./data)")
    parser.add_argument("--filter", help="Substring filter for model names.")
    parser.add_argument("--force", action="store_true", help="Re-download even if unchanged.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without downloading.")
    parser.add_argument("--stage", choices=("all", "required", "deferred"), default="all",
                        help="all: core + LZAI_MODELS; required: core + LZAI_REQUIRED_MODELS (marks the rest "
                             "pending); deferred: the pending sets (default: all)")
    parser.add_argument("--parallel", type=int, default=DEFAULT_PARALLELISM,
                        help=f"Concurrent downloads (default: {DEFAULT_PARALLELISM})")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    rc = sync_data_from_s3(
        data_dir=data_dir,
        name_filter=args.filter,
        force=args.force,
        dry_run=args.dry_run,
        parallel=max(1, args.parallel),
        stage=args.stage,
    )
    sys.exit(rc)
