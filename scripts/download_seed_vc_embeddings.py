#!/usr/bin/env python3
"""Download Seed-VC voice embedding files from Wasabi S3.

Usage:
    uv run python scripts/download_seed_vc_embeddings.py
    uv run python scripts/download_seed_vc_embeddings.py --embeddings-dir data/seed-vc/embeddings
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
DEFAULT_EMBEDDINGS_DIR = Path("data/seed-vc/embeddings")


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
    parser = argparse.ArgumentParser(description="Download Seed-VC embeddings from Wasabi S3.")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=Path(os.environ.get("SEED_VC_EMBEDDINGS_DIR", DEFAULT_EMBEDDINGS_DIR)),
        help=f"Destination directory for embeddings (default: {DEFAULT_EMBEDDINGS_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists with the same size.",
    )
    return parser.parse_args()


def sync_embeddings(embeddings_dir: Path, force: bool) -> int:
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_embeddings_path = os.getenv("S3_EMBEDDINGS_PATH", "seed-vc/embeddings")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    s3_client = get_s3_client()

    try:
        print(f"Listing files from s3://{bucket}/{s3_embeddings_path}/")
        response = s3_client.list_objects_v2(
            Bucket=bucket, Prefix=f"{s3_embeddings_path}/"
        )
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return 1

    files = [obj for obj in response.get("Contents", []) if obj["Key"].endswith(".h5")]
    if not files:
        print("No .h5 embedding files found in S3")
        return 0

    print(f"Found {len(files)} embedding file(s)")
    print(f"Target: {embeddings_dir}/")

    downloaded = skipped = failed = 0
    for obj in files:
        s3_key = obj["Key"]
        local_path = embeddings_dir / Path(s3_key).name
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
    return sync_embeddings(resolve_repo_path(args.embeddings_dir), args.force)


if __name__ == "__main__":
    sys.exit(main())
