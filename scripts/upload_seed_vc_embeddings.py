#!/usr/bin/env python3
"""Upload a Seed-VC voice embedding HDF5 file to Wasabi S3.

Usage:
    uv run python scripts/upload_seed_vc_embeddings.py
    uv run python scripts/upload_seed_vc_embeddings.py --file data/seed-vc/embeddings/vtts_embeddings.h5
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
DEFAULT_EMBEDDINGS_FILE = Path("data/seed-vc/embeddings/vtts_embeddings.h5")


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
    parser = argparse.ArgumentParser(description="Upload Seed-VC embeddings to Wasabi S3.")
    parser.add_argument(
        "--file",
        type=Path,
        default=Path(os.environ.get("SEED_VC_EMBEDDINGS_FILE", DEFAULT_EMBEDDINGS_FILE)),
        help=f"Embedding file to upload (default: {DEFAULT_EMBEDDINGS_FILE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if the remote file already exists with the same size.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_embeddings_path = os.getenv("S3_EMBEDDINGS_PATH", "seed-vc/embeddings")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    local_path = resolve_repo_path(args.file)
    if not local_path.exists():
        print(f"Error: File not found: {local_path}")
        return 1

    s3_key = f"{s3_embeddings_path}/{local_path.name}"
    local_size = local_path.stat().st_size
    file_size_mb = local_size / (1024 * 1024)
    s3_client = get_s3_client()

    if not args.force:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            if response["ContentLength"] == local_size:
                print(f"Skipping {local_path.name} (already exists, same size)")
                return 0
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                print(f"Error checking remote file: {e}")

    print(f"Uploading {local_path.name} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print("done")
        print(f"Uploaded to: s3://{bucket}/{s3_key}")
        return 0
    except ClientError as e:
        print(f"FAILED: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
