#!/usr/bin/env python3
"""
Upload model data files to Wasabi S3

Usage:
    uv run python scripts/upload_models.py
    uv run python scripts/upload_models.py --model lzspeech-enzhja-1000-bert
    uv run python scripts/upload_models.py --data-dir ./data
"""
import argparse
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
        description="Upload LZ-TTS model data to Wasabi S3"
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
    return parser.parse_args()


def upload_file(s3_client, local_path, bucket, s3_key, display_path, force=False):
    """Upload a single file to S3"""
    local_size = local_path.stat().st_size
    file_size_mb = local_size / (1024 * 1024)

    # Check if file already exists with same size
    if not force:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            if response["ContentLength"] == local_size:
                print(f"Skipping {display_path} (already exists, same size)")
                return "skipped"
        except ClientError as e:
            if e.response["Error"]["Code"] != "404":
                print(f"Error checking {s3_key}: {e}")
                return "failed"

    print(f"Uploading {display_path} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
    try:
        s3_client.upload_file(str(local_path), bucket, s3_key)
        print("done")
        return "uploaded"
    except ClientError as e:
        print(f"FAILED: {e}")
        return "failed"


def upload_data_to_s3(
    data_dir: Path,
    model_name: str | None = None,
    force: bool = False,
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
    skipped = 0
    failed = 0

    for model_dir in sorted(model_dirs):
        print(f"\n=== Processing {model_dir.name} ===")

        # Find all files in this model directory
        files = list(model_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        if not files:
            print(f"No files found in {model_dir.name}")
            continue

        print(f"Found {len(files)} files")

        for file_path in sorted(files):
            # Calculate relative path from data_dir
            relative_path = file_path.relative_to(data_dir)
            s3_key = f"{s3_data_path}/{relative_path}"

            result = upload_file(s3_client, file_path, bucket, s3_key, str(relative_path), force)

            if result == "uploaded":
                uploaded += 1
            elif result == "skipped":
                skipped += 1
            elif result == "failed":
                failed += 1

    print()
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
        )
    )
