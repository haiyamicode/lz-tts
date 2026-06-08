#!/usr/bin/env python3
"""Upload the current production serving artifacts to Wasabi S3.

This script uploads only the files that the live service actually needs:
- local/server.json
- Sparrow multilingual bundle
- Sparrow en-GB bundle
- Seed-VC runtime assets used by the embedded voice conversion service

It intentionally avoids any training-only or archival checkpoints.
The large Seed-VC embeddings HDF5 is treated as a shared external artifact and
is not uploaded here; use scripts/download_seed_vc_embeddings.py to restore it
from the existing `seed-vc/embeddings` S3 prefix when provisioning a machine.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILES = [
    (Path("local/server.json"), Path("server.json")),
    (Path("data/lzspeech-sparrow/model.ckpt"), Path("lzspeech-sparrow/model.ckpt")),
    (Path("data/lzspeech-sparrow/config.json"), Path("lzspeech-sparrow/config.json")),
    (Path("data/lzspeech-sparrow-en-GB/model.ckpt"), Path("lzspeech-sparrow-en-GB/model.ckpt")),
    (Path("data/lzspeech-sparrow-en-GB/config.json"), Path("lzspeech-sparrow-en-GB/config.json")),
    (Path("data/seed-vc/voices_final.pkl"), Path("seed-vc/voices_final.pkl")),
    (Path("data/seed-vc/voice-samples/andrew.mp3"), Path("seed-vc/voice-samples/andrew.mp3")),
]


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if the remote file already exists with the same size.",
    )
    return parser.parse_args()


def upload_file(s3_client, bucket: str, local_path: Path, s3_key: str, force: bool) -> str:
    size = local_path.stat().st_size
    size_mb = size / (1024 * 1024)

    if not force:
        try:
            response = s3_client.head_object(Bucket=bucket, Key=s3_key)
            if response["ContentLength"] == size:
                print(f"Skipping {local_path} (already exists, same size)")
                return "skipped"
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "404":
                raise

    print(f"Uploading {local_path} ({size_mb:.2f} MB)...", end=" ", flush=True)
    s3_client.upload_file(str(local_path), bucket, s3_key)
    print("done")
    return "uploaded"


def main() -> int:
    args = parse_args()
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_data_path = os.getenv("S3_DATA_PATH", "lz-tts/data")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    s3_client = get_s3_client()
    uploaded = skipped = failed = 0

    print(f"Target: s3://{bucket}/{s3_data_path}/")
    print()

    for local_rel_path, s3_rel_path in DEFAULT_FILES:
        local_path = resolve_repo_path(local_rel_path)
        if not local_path.exists():
            print(f"Missing local file, skipping: {local_rel_path}")
            skipped += 1
            continue

        s3_key = f"{s3_data_path}/{s3_rel_path.as_posix()}"
        try:
            result = upload_file(s3_client, bucket, local_path, s3_key, args.force)
            if result == "uploaded":
                uploaded += 1
            else:
                skipped += 1
        except ClientError as exc:
            print(f"FAILED: {local_rel_path}: {exc}")
            failed += 1

    print()
    print(f"Upload complete: {uploaded} uploaded, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
