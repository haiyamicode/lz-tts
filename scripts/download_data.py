#!/usr/bin/env python3
"""
Download (sync) data files from Wasabi S3

Downloads model data and RVC assets from S3.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --filter lzspeech
    uv run python scripts/download_data.py --data-dir ./data
    uv run python scripts/download_data.py --skip-rvc
    uv run python scripts/download_data.py --dry-run
"""
import argparse
import hashlib
import os
import sys
from pathlib import Path

import boto3
import boto3.s3.transfer
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def get_s3_client():
    import botocore.config
    return boto3.client(
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


# ---------------------------------------------------------------------------
# Download with progress
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, label: str, total: int):
        self._label = label
        self._total = total
        self._seen = 0

    def __call__(self, nbytes: int):
        self._seen += nbytes
        done_mb = self._seen / (1024 * 1024)
        total_mb = self._total / (1024 * 1024)
        pct = (self._seen / self._total * 100) if self._total else 0
        print(f"\r  {self._label}  {done_mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)


def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3 with progress output."""
    try:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cfg = boto3.s3.transfer.TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            multipart_chunksize=8 * 1024 * 1024,
            max_concurrency=8,
        )

        label = local_path.name
        if len(label) > 30:
            label = label[:27] + "..."
        seen = {"n": 0}

        def _cb(nbytes):
            seen["n"] += nbytes
            mb = seen["n"] / (1024 * 1024)
            print(f"\r  {label}  {mb:.1f} MB", end="", flush=True)

        s3_client.download_file(
            bucket, s3_key, str(local_path), Config=cfg, Callback=_cb
        )
        print()
        return True
    except ClientError as e:
        print(f"\n  Error downloading {s3_key}: {e}")
        return False


# ---------------------------------------------------------------------------
# Sync from lz-tts data prefix
# ---------------------------------------------------------------------------

_EXCLUDE_PREFIXES = (
    "seed-vc/embeddings/",
    "seed-vc/checkpoints/",
)
_EXCLUDE_NAMES = {"server.json"}


def sync_data_from_s3(
    data_dir: Path | None = None,
    name_filter: str | None = None,
    force: bool = False,
    dry_run: bool = False,
):
    bucket = os.getenv('AWS_S3_BUCKET_NAME')
    s3_data_path = os.getenv('S3_DATA_PATH', 'lz-tts/data')
    local_data_dir = data_dir or Path('./data')

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    local_data_dir.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()

    try:
        print(f"Connecting to Wasabi S3...")
        print(f"Listing s3://{bucket}/{s3_data_path}/")

        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{s3_data_path}/")
        all_objects = [o for page in pages for o in page.get('Contents', [])]

        if not all_objects:
            print("No files found.")
            return 0

        filtered = []
        for obj in all_objects:
            key = obj['Key']
            if key == f"{s3_data_path}/":
                continue
            rel = key[len(s3_data_path) + 1:]
            if not rel:
                continue
            model_name = rel.split('/')[0]
            if name_filter and name_filter not in model_name:
                continue
            filtered.append((obj, rel))

        if not filtered:
            print(f"No matching files (filter: {name_filter})")
            return 0

        print(f"Found {len(filtered)} files → {local_data_dir}/\n")

        downloaded = skipped = failed = 0

        for obj, rel in filtered:
            if any(rel.startswith(p) for p in _EXCLUDE_PREFIXES) or rel in _EXCLUDE_NAMES:
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
                downloaded += 1
                continue

            print(f"Downloading {rel} ({size_mb:.1f} MB)")
            if download_file(s3, bucket, obj['Key'], local_path):
                downloaded += 1
            else:
                failed += 1

        print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")

        return 0 if failed == 0 else 1

    except ClientError as e:
        print(f"S3 error: {e}")
        return 1


# ---------------------------------------------------------------------------
# RVC assets
# ---------------------------------------------------------------------------

def sync_rvc_assets(data_dir: Path, force: bool = False, dry_run: bool = False):
    rvc_dir = data_dir / "rvc"
    failed = 0

    bucket = os.getenv('AWS_S3_BUCKET_NAME')
    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    s3 = get_s3_client()

    try:
        # Foundation models: rvc/hubert/, rvc/rmvpe/
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="rvc/")
        hubert_rmvpe = resp.get("Contents", [])

        # Model weights: weights/*.pth
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="weights/")
        weights = [o for o in resp.get("Contents", []) if o["Key"].endswith(".pth")]

        print("=== RVC: foundation models ===")
        for obj in hubert_rmvpe:
            name = Path(obj["Key"]).name
            subdir = Path(obj["Key"]).parts[1]
            local_path = rvc_dir / subdir / name
            size_mb = obj["Size"] / (1024 * 1024)

            if _should_skip(local_path, obj, force):
                print(f"  Skipping {subdir}/{name} (same content)")
                continue

            if dry_run:
                print(f"  WOULD DOWNLOAD {subdir}/{name} ({size_mb:.0f} MB)")
                continue

            print(f"  {subdir}/{name} ({size_mb:.0f} MB)")
            if download_file(s3, bucket, obj["Key"], local_path):
                pass
            else:
                failed += 1

        print(f"\n=== RVC: model weights ({len(weights)}) ===")
        weights_dir = rvc_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)

        for obj in weights:
            name = Path(obj["Key"]).name
            local_path = weights_dir / name
            size_mb = obj["Size"] / (1024 * 1024)

            if _should_skip(local_path, obj, force):
                print(f"  Skipping {name} (same content)")
                continue

            if dry_run:
                print(f"  WOULD DOWNLOAD {name} ({size_mb:.1f} MB)")
                continue

            print(f"  {name} ({size_mb:.1f} MB)")
            if download_file(s3, bucket, obj["Key"], local_path):
                pass
            else:
                failed += 1

    except ClientError as e:
        print(f"S3 error: {e}")
        failed += 1

    return 1 if failed else 0


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
    parser.add_argument("--skip-rvc", action="store_true", help="Skip RVC asset downloads.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    rc = sync_data_from_s3(
        data_dir=data_dir,
        name_filter=args.filter,
        force=args.force,
        dry_run=args.dry_run,
    )
    if not args.skip_rvc:
        rvc_rc = sync_rvc_assets(
            data_dir=data_dir or Path("./data"),
            force=args.force,
            dry_run=args.dry_run,
        )
        rc = rc or rvc_rc
    sys.exit(rc)
