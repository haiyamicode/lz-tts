#!/usr/bin/env python3
"""Download Seed-VC model checkpoints from Wasabi S3.

Usage:
    uv run python scripts/download_seed_vc_models.py
    uv run python scripts/download_seed_vc_models.py --filter reflow
    uv run python scripts/download_seed_vc_models.py --models-dir data/seed-vc/models
"""

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path("data/seed-vc/models")


def resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Download Seed-VC models from Wasabi S3.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("SEED_VC_MODELS_DIR", DEFAULT_MODELS_DIR)),
        help=f"Destination directory for model files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--filter",
        help="Substring filter applied to S3 filenames, for example 'reflow'.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists with the same size.",
    )
    return parser.parse_args()


def sync_models_from_s3(models_dir: Path, name_filter: str | None, force: bool) -> int:
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_models_path = os.getenv("S3_MODELS_PATH", "seed-vc/models")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    models_dir.mkdir(parents=True, exist_ok=True)
    s3_client = get_s3_client()

    try:
        print(f"Listing files from s3://{bucket}/{s3_models_path}/")
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=f"{s3_models_path}/")
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return 1

    objects = response.get("Contents", [])
    model_files = [
        obj
        for obj in objects
        if obj["Key"].endswith(".pth")
        and (name_filter is None or name_filter in Path(obj["Key"]).name)
    ]

    if not model_files:
        print("No matching .pth model files found in S3")
        return 0

    print(f"Found {len(model_files)} model file(s)")
    print(f"Target: {models_dir}/")

    downloaded = skipped = failed = 0
    for obj in model_files:
        s3_key = obj["Key"]
        local_path = models_dir / Path(s3_key).name
        file_size_mb = obj["Size"] / (1024 * 1024)

        if local_path.exists() and not force and local_path.stat().st_size == obj["Size"]:
            print(f"Skipping {local_path.name} (already exists, same size)")
            skipped += 1
            continue

        print(f"Downloading {local_path.name} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
        try:
            s3_client.download_file(bucket, s3_key, str(local_path))
            print("done")
            downloaded += 1
        except ClientError as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nDownload complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    args = parse_args()
    return sync_models_from_s3(resolve_repo_path(args.models_dir), args.filter, args.force)


if __name__ == "__main__":
    sys.exit(main())
