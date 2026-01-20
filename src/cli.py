#!/usr/bin/env python3
"""Minimal Piper TTS inference entry point.

Usage:
    python main.py [OPTIONS] [TEXT]

Examples:
    python main.py "Hello, this is a test."
    python main.py -m lzspeech-enzhja-1000-bert -s en "Hello world"
    python main.py -o output.wav "Testing TTS"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default model name and paths
DEFAULT_MODEL = "lzspeech-enzhja-1000-bert"
DATA_DIR = Path("data")          # Model files (not tracked in git)
OUTPUT_DIR = Path("output")


def find_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Find a checkpoint in the given directory."""
    if not checkpoint_dir.exists():
        return None

    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if checkpoints:
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return checkpoints[0]

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Piper TTS inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "text",
        nargs="*",
        default=["Hello, this is a test."],
        help="Text to synthesize",
    )
    parser.add_argument(
        "-m", "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=Path,
        help="Path to .ckpt checkpoint (overrides model lookup)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config.json (overrides model lookup)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=OUTPUT_DIR / "output.wav",
        help="Output WAV file path",
    )
    parser.add_argument(
        "-s", "--speaker",
        help="Speaker label (e.g., 'en', 'ja', 'zh'). Uses auto-detection if not specified.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        help="Prosody randomness (default: from config)",
    )
    parser.add_argument(
        "--length-scale",
        type=float,
        help="Speech rate multiplier (default: from config, >1 = slower)",
    )
    parser.add_argument(
        "--noise-w",
        type=float,
        help="Duration predictor noise (default: from config)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto)",
    )
    args = parser.parse_args()

    # Model directory
    model_dir = DATA_DIR / args.model

    # Resolve config path
    config_path = args.config or (model_dir / "config.json")
    if not config_path.exists():
        print(f"Error: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Resolve checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = find_checkpoint(model_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoint found in {model_dir}", file=sys.stderr)
            print("Use -c to provide a checkpoint path.", file=sys.stderr)
            sys.exit(1)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}", file=sys.stderr)
        sys.exit(1)

    # Join text arguments
    text = " ".join(args.text)

    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    print(f"Output: {args.output}")
    print(f"Text: {text}")
    if args.speaker:
        print(f"Speaker: {args.speaker}")

    # Import here to avoid slow startup for --help
    from src.piper import PiperInference

    # Initialize inference pipeline
    print("\nLoading model...")
    inference = PiperInference(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device=args.device,
    )

    print(f"BERT enabled: {inference.use_bert}")
    print(f"Available speakers: {list(inference.speakers.keys())}")

    # Build synthesis kwargs
    synth_kwargs = {}
    if args.noise_scale is not None:
        synth_kwargs["noise_scale"] = args.noise_scale
    if args.length_scale is not None:
        synth_kwargs["length_scale"] = args.length_scale
    if args.noise_w is not None:
        synth_kwargs["noise_w"] = args.noise_w

    # Synthesize
    print("\nSynthesizing...")
    output_path = inference.synthesize_to_file(
        text=text,
        output_path=args.output,
        speaker=args.speaker,
        **synth_kwargs,
    )

    print(f"\nDone. Output WAV: {output_path}")


if __name__ == "__main__":
    main()
