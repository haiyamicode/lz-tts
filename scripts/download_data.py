#!/usr/bin/env python3
"""
Download (sync) data files from Wasabi S3

Downloads model data and server configuration (local/server.json) from S3.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --filter lzspeech
    uv run python scripts/download_data.py --data-dir ./data
"""
import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_s3_client():
    """Create and return S3 client with Wasabi configuration"""
    return boto3.client(
        's3',
        endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}",
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download LZ-TTS data (models and server config) from Wasabi S3."
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Destination directory for data files (default: ./data)",
    )
    parser.add_argument(
        "--filter",
        help="Substring filter applied to model names (e.g., 'lzspeech'). Only matching models are downloaded.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if local file exists with same size.",
    )
    return parser.parse_args()


def download_file(s3_client, bucket, s3_key, local_path):
    """Download a single file from S3"""
    try:
        s3_client.download_file(bucket, s3_key, str(local_path))
        return True
    except ClientError as e:
        print(f"Error downloading {s3_key}: {e}")
        return False


def sync_data_from_s3(
    data_dir: Path | None = None,
    name_filter: str | None = None,
    force: bool = False,
):
    """Sync model data files from S3 to local"""
    # Get configuration
    bucket = os.getenv('AWS_S3_BUCKET_NAME')
    s3_data_path = os.getenv('S3_DATA_PATH', 'lz-tts/data')
    local_data_dir = data_dir or Path('./data')

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    # Create local directory if it doesn't exist
    local_data_dir.mkdir(parents=True, exist_ok=True)

    # Initialize S3 client
    print(f"Connecting to Wasabi S3...")
    s3_client = get_s3_client()

    # List all objects in S3
    try:
        print(f"Listing files from s3://{bucket}/{s3_data_path}/")

        # Use paginator to handle large number of objects
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{s3_data_path}/")

        all_objects = []
        for page in pages:
            if 'Contents' in page:
                all_objects.extend(page['Contents'])

        if not all_objects:
            print(f"No files found in s3://{bucket}/{s3_data_path}/")
            return 0

        # Filter model directories if filter is specified
        filtered_objects = []
        for obj in all_objects:
            key = obj['Key']
            # Skip the root prefix itself
            if key == f"{s3_data_path}/":
                continue

            # Extract model name (first directory after prefix)
            relative_path = key[len(s3_data_path)+1:]
            if not relative_path:
                continue

            model_name = relative_path.split('/')[0]

            # Apply filter if specified
            if name_filter and name_filter not in model_name:
                continue

            filtered_objects.append((obj, relative_path))

        if not filtered_objects:
            print(f"No matching files found (filter: {name_filter})")
            return 0

        print(f"Found {len(filtered_objects)} files to sync")
        print(f"Target: {local_data_dir}/")
        print()

        # Download each file
        downloaded = 0
        skipped = 0
        failed = 0

        for obj, relative_path in filtered_objects:
            s3_key = obj['Key']
            local_path = local_data_dir / relative_path
            file_size = obj['Size'] / (1024 * 1024)  # MB

            # Create parent directories
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file already exists with same size
            if local_path.exists() and not force:
                local_size = local_path.stat().st_size
                if local_size == obj['Size']:
                    print(f"Skipping {relative_path} (already exists, same size)")
                    skipped += 1
                    continue

            print(f"Downloading {relative_path} ({file_size:.2f} MB)...", end=" ", flush=True)

            if download_file(s3_client, bucket, s3_key, local_path):
                print("done")
                downloaded += 1
            else:
                print("FAILED")
                failed += 1

        print()
        print(f"Download complete: {downloaded} downloaded, {skipped} skipped, {failed} failed")

        # Download server.json configuration file
        print("\n=== Downloading server configuration ===")
        server_config_s3_key = f"{s3_data_path}/server.json"
        local_server_config = Path("local/server.json")
        local_server_config.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Check if server.json exists in S3
            s3_client.head_object(Bucket=bucket, Key=server_config_s3_key)
            print(f"Downloading server.json...", end=" ", flush=True)
            if download_file(s3_client, bucket, server_config_s3_key, local_server_config):
                print("done")
            else:
                print("FAILED (non-fatal)")
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                print("server.json not found in S3 (skipping)")
            else:
                print(f"Error checking server.json: {e}")

        return 0 if failed == 0 else 1

    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return 1


if __name__ == "__main__":
    args = parse_args()
    data_dir = Path(args.data_dir) if args.data_dir else None
    sys.exit(
        sync_data_from_s3(
            data_dir=data_dir,
            name_filter=args.filter,
            force=args.force,
        )
    )
