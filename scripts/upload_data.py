#!/usr/bin/env python3
"""
Upload data files to Wasabi S3

Uploads model data to S3.

Usage:
    uv run python scripts/upload_data.py
    uv run python scripts/upload_data.py --model lzspeech-enzhja-1000-bert
    uv run python scripts/upload_data.py --data-dir ./data
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()


def get_s3_client():
    """Create and return S3 client with Wasabi configuration"""
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload LZ-TTS data to Wasabi S3"
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="Source directory containing model data (default: ./data)",
    )
    parser.add_argument(
        "--model",
        help="Specific model directory to upload (e.g., 'lzspeech-enzhja-1000-bert'). If not specified, uploads all models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if file exists with same size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without writing to S3.",
    )
    return parser.parse_args()


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
    """Compute the S3-equivalent ETag for a local file."""
    md5_hex, part_count = _multipart_md5(str(file_path))
    if part_count == 1:
        return md5_hex
    return f"{md5_hex}-{part_count}"


def upload_file(s3_client, local_path, bucket, s3_key, display_path, force=False, dry_run=False):
    """Upload a single file to S3"""
    file_size_mb = local_path.stat().st_size / (1024 * 1024)

    # Compare local ETag vs S3 ETag (multipart-aware)
    if not force:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            s3_etag = response.get("ETag", "").strip('"')
            if s3_etag and _local_etag(str(local_path)) == s3_etag:
                print(f"Skipping {display_path} (same content)")
                return "skipped"
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                print(f"Error checking {s3_key}: {e}")
                return "failed"

    if dry_run:
        print(f"WOULD UPLOAD {display_path} ({file_size_mb:.2f} MB)")
        return "would_upload"

    print(f"Uploading {display_path} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print("done")
        return "uploaded"
    except ClientError as e:
        print(f"FAILED: {e}")
        return "failed"


def _manifest_files(model_dir: Path) -> tuple[list[Path], list[Path]]:
    manifest_path = model_dir / "manifest.json"
    if not manifest_path.exists():
        files = [f for f in model_dir.rglob("*") if f.is_file()]
        return sorted(files), []

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("files")
    if not isinstance(entries, list) or not all(isinstance(item, str) for item in entries):
        raise ValueError(f"{manifest_path} must contain a string array field named 'files'")

    files = [manifest_path]
    missing: list[Path] = []
    for entry in entries:
        rel = Path(entry)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"{manifest_path} contains unsafe path: {entry}")
        file_path = model_dir / rel
        if file_path.is_file():
            files.append(file_path)
        else:
            missing.append(file_path)

    return sorted(dict.fromkeys(files)), missing


def upload_data_to_s3(
    data_dir: Path,
    model_name: str | None = None,
    force: bool = False,
    dry_run: bool = False,
):
    """Upload model data files from local to S3"""
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_data_path = os.getenv("S3_DATA_PATH", "lz-tts/data")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1

    print(f"Connecting to Wasabi S3...")
    s3_client = get_s3_client()

    # Determine which models to upload
    if model_name:
        model_dirs = [data_dir / model_name]
        if not model_dirs[0].exists():
            print(f"Error: Model directory not found: {model_dirs[0]}")
            return 1
    else:
        model_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not model_dirs:
            print(f"No model directories found in {data_dir}")
            return 0

    print(f"Uploading from: {data_dir}/")
    print(f"Target: s3://{bucket}/{s3_data_path}/")
    print()

    uploaded = 0
    would_upload = 0
    skipped = 0
    failed = 0

    for model_dir in sorted(model_dirs):
        print(f"\n=== Processing {model_dir.name} ===")

        try:
            files, missing = _manifest_files(model_dir)
        except (OSError, ValueError, json.JSONDecodeError) as e:
            print(f"Error reading manifest for {model_dir.name}: {e}")
            failed += 1
            continue

        if missing:
            for path in missing:
                print(f"Missing manifest file: {path}")
            failed += len(missing)
            continue

        if not files:
            print(f"No files found in {model_dir.name}")
            continue

        if (model_dir / "manifest.json").exists():
            print(f"Found manifest with {len(files) - 1} file(s)")
        else:
            print(f"Found {len(files)} files")

        for file_path in sorted(files):
            # Calculate relative path from data_dir
            relative_path = file_path.relative_to(data_dir)
            s3_key = f"{s3_data_path}/{relative_path}"

            result = upload_file(
                s3_client,
                file_path,
                bucket,
                s3_key,
                str(relative_path),
                force,
                dry_run,
            )

            if result == "uploaded":
                uploaded += 1
            elif result == "would_upload":
                would_upload += 1
            elif result == "skipped":
                skipped += 1
            elif result == "failed":
                failed += 1

    print()
    if dry_run:
        print(f"Dry run complete: {would_upload} would upload, {skipped} skipped, {failed} failed")
    else:
        print(f"Upload complete: {uploaded} uploaded, {skipped} skipped, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir)
    sys.exit(
        upload_data_to_s3(
            data_dir=data_dir,
            model_name=args.model,
            force=args.force,
            dry_run=args.dry_run,
        )
    )
