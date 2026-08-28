#!/usr/bin/env python3
"""Clean up S3 bucket: delete files NOT in the serving manifest.

Compares the current data/ directory against everything under lz-tts/data/
on S3 and deletes any remote files that aren't in the local serving set.

Usage:
    uv run python scripts/clean_s3_orphans.py            # dry-run (list orphans)
    uv run python scripts/clean_s3_orphans.py --delete   # actually delete
"""

import argparse, os, sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Everything under data/ that should be on S3
SERVING_FILES = {
    "heteronyms/best.pt",
    "heteronyms/heretonyms.jsonl",
    "lzspeech-sparrow-en-GB/config.json",
    "lzspeech-sparrow-en-GB/model.ckpt",
    "lzspeech-sparrow/config.json",
    "lzspeech-sparrow/manifest.json",
    "lzspeech-sparrow/model.ckpt",
    "lzspeech-starling/model.ckpt",
    "voice-presets.json",
    "seed-vc/manifest.json",
    "seed-vc/embeddings/vtts_embeddings.h5",
    "seed-vc/embeddings/vtts_embeddings_sparrow_fallback.h5",
    "seed-vc/models/reflow_v2.pth",
    "seed-vc/voice-samples/andrew.mp3",
    "seed-vc/voices_final.pkl",
}


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=f"https://{os.getenv('AWS_S3_ENDPOINT')}",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="Actually delete orphans (default is dry-run)")
    args = parser.parse_args()

    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_data_path = os.getenv("S3_DATA_PATH", "lz-tts/data")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set", file=sys.stderr)
        return 1

    s3_client = get_s3_client()

    # List everything on S3 under lz-tts/data/
    print(f"Listing s3://{bucket}/{s3_data_path}/ ...")
    paginator = s3_client.get_paginator("list_objects_v2")
    all_objects = []
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{s3_data_path}/"):
        all_objects.extend(page.get("Contents", []))

    if not all_objects:
        print("No files found on S3")
        return 0

    orphans = []
    for obj in all_objects:
        key = obj["Key"]
        if key == f"{s3_data_path}/":
            continue
        rel = key[len(s3_data_path) + 1 :]
        if not rel:
            continue
        if rel not in SERVING_FILES:
            orphans.append((key, obj.get("Size", 0)))

    if not orphans:
        print("No orphan files found. S3 is clean.")
        return 0

    total_size = sum(s for _, s in orphans)
    print(f"\nFound {len(orphans)} orphan file(s) ({total_size / (1024**2):.1f} MB):")
    for key, size in orphans:
        print(f"  {key}  ({size / (1024**2):.1f} MB)")

    if args.delete:
        print(f"\nDeleting {len(orphans)} orphan file(s)...")
        failed = 0
        for key, _ in orphans:
            try:
                s3_client.delete_object(Bucket=bucket, Key=key)
                print(f"  DELETED {key}")
            except ClientError as e:
                print(f"  FAILED {key}: {e}", file=sys.stderr)
                failed += 1
        print(f"\nDone: {len(orphans) - failed} deleted, {failed} failed")
        return 1 if failed else 0
    else:
        print("\nDRY-RUN. Run with --delete to actually delete.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
