#!/usr/bin/env python3
"""Upload Seed-VC model checkpoints to Wasabi S3.

Usage:
    uv run python scripts/upload_seed_vc_models.py
    uv run python scripts/upload_seed_vc_models.py --models reflow_v2
    uv run python scripts/upload_seed_vc_models.py --models-dir local/seed-vc/models --filter reflow
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = Path("local/seed-vc/models")


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
    parser = argparse.ArgumentParser(description="Upload Seed-VC models to Wasabi S3.")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(os.environ.get("SEED_VC_MODELS_DIR", DEFAULT_MODELS_DIR)),
        help=f"Directory containing model files (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        help="Specific model files or names to upload. Can be full paths or names in models-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload even if the remote file already exists with the same size.",
    )
    parser.add_argument(
        "--filter",
        help="Substring filter applied to filenames before uploading.",
    )
    parser.add_argument(
        "--rename",
        help="Rename the file when uploading. Only valid when uploading one file.",
    )
    return parser.parse_args()


def resolve_model_files(
    models_dir: Path, models: Optional[list[str]], substring_filter: Optional[str]
) -> tuple[list[Path], list[Path]]:
    resolved: list[Path] = []
    missing: list[Path] = []

    if not models:
        if models_dir.exists():
            resolved = list(models_dir.glob("*.pth"))
    else:
        for model in models:
            candidate = Path(model)
            if candidate.is_file():
                resolved.append(candidate)
                continue

            model_name = model if model.endswith(".pth") else f"{model}.pth"
            candidate = models_dir / model_name
            if candidate.exists():
                resolved.append(candidate)
            else:
                missing.append(candidate)

    if substring_filter:
        resolved = [path for path in resolved if substring_filter in path.name]

    return resolved, missing


def sync_models_to_s3(
    models_dir: Path,
    models: Optional[list[str]],
    skip_existing: bool,
    substring_filter: Optional[str],
    rename: Optional[str],
) -> int:
    bucket = os.getenv("AWS_S3_BUCKET_NAME")
    s3_models_path = os.getenv("S3_MODELS_PATH", "seed-vc/models")

    if not bucket:
        print("Error: AWS_S3_BUCKET_NAME not set in .env")
        return 1

    model_files, missing = resolve_model_files(models_dir, models, substring_filter)
    for missing_file in missing:
        print(f"Warning: model not found locally: {missing_file}")

    if not model_files:
        print("No .pth files found to upload")
        return 0

    if rename and len(model_files) > 1:
        print("Error: --rename can only be used when uploading a single file")
        return 1

    s3_client = get_s3_client()
    print(f"Found {len(model_files)} model file(s)")
    print(f"Target: s3://{bucket}/{s3_models_path}/")

    uploaded = skipped = failed = 0
    for model_file in model_files:
        target_name = rename or model_file.name
        s3_key = f"{s3_models_path}/{target_name}"
        local_size = model_file.stat().st_size
        file_size_mb = local_size / (1024 * 1024)

        if skip_existing:
            try:
                response = s3_client.head_object(Bucket=bucket, Key=s3_key)
                if response["ContentLength"] == local_size:
                    print(f"Skipping {model_file.name} (already exists, same size)")
                    skipped += 1
                    continue
            except ClientError as e:
                if e.response["Error"]["Code"] != "404":
                    print(f"Error checking {model_file.name}: {e}")

        display_name = f"{model_file.name} -> {target_name}" if rename else model_file.name
        print(f"Uploading {display_name} ({file_size_mb:.2f} MB)...", end=" ", flush=True)
        try:
            s3_client.upload_file(str(model_file), bucket, s3_key)
            print("done")
            uploaded += 1
        except ClientError as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nUpload complete: {uploaded} uploaded, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


def main() -> int:
    args = parse_args()
    return sync_models_to_s3(
        models_dir=resolve_repo_path(args.models_dir),
        models=args.models or None,
        skip_existing=not args.force,
        substring_filter=args.filter,
        rename=args.rename,
    )


if __name__ == "__main__":
    sys.exit(main())
