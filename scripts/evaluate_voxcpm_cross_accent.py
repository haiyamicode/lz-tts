#!/usr/bin/env python3
"""Generate matched baseline/LoRA samples for held-out cross-accent evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig


EVAL_TEXTS = (
    "After class, Sarah parked her car near the market and walked home through the garden.",
    "The first train leaves early on Thursday, so remember to bring your passport and a bottle of water.",
    "Our new research centre will open next year after the final building inspection is complete.",
    "I asked whether the smaller parcel could arrive before dinner on Saturday evening.",
    "The weather forecast changed again, but we still planned to meet beside the old railway station.",
    "Please call the office tomorrow morning and confirm that every document has been signed.",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument(
        "--reference-field",
        choices=("audio", "ref_audio"),
        default="audio",
        help="Manifest field used as VoxCPM's reference-only conditioning audio.",
    )
    parser.add_argument("--reference-accent", choices=("en-GB", "en-US"), required=True)
    parser.add_argument("--target-accent", choices=("en-GB", "en-US"), required=True)
    parser.add_argument("--lora-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--cfg-value", type=float, default=2.0)
    parser.add_argument("--inference-timesteps", type=int, default=10)
    parser.add_argument("--max-len", type=int, default=600)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _stable_key(seed: int, row: dict) -> bytes:
    value = f"{seed}:{row['speaker_id']}:{row['utterance_id']}"
    return hashlib.sha256(value.encode()).digest()


def _select_speakers(rows: list[dict], count: int, seed: int) -> list[dict]:
    by_speaker = {}
    for row in sorted(rows, key=lambda item: _stable_key(seed, item)):
        by_speaker.setdefault(str(row["speaker_id"]), row)
    selected = list(by_speaker.values())[:count]
    if len(selected) < count:
        raise ValueError(f"Only {len(selected)} held-out speakers available; requested {count}")
    return selected


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_model(checkpoint: Path) -> VoxCPM:
    config = json.loads((checkpoint / "lora_config.json").read_text(encoding="utf-8"))
    return VoxCPM.from_pretrained(
        hf_model_id=config["base_model"],
        load_denoiser=False,
        optimize=True,
        lora_config=LoRAConfig(**config["lora_config"]),
        lora_weights_path=str(checkpoint),
    )


def main() -> None:
    args = _parse_args()
    rows = _read_jsonl(args.reference_manifest)
    selected = _select_speakers(rows, args.samples, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = _load_model(args.lora_checkpoint.resolve())
    sample_rate = int(model.tts_model.sample_rate)

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(selected):
            seed = args.seed + index
            text = EVAL_TEXTS[index % len(EVAL_TEXTS)]
            stem = f"{index:02d}_{row['speaker_id']}"
            reference_path = args.output_dir / f"{stem}_reference_{args.reference_accent}.wav"
            baseline_path = args.output_dir / f"{stem}_baseline.wav"
            lora_path = args.output_dir / f"{stem}_{args.target_accent}_lora.wav"

            source_reference = Path(row[args.reference_field]).resolve()
            reference_audio, reference_rate = sf.read(str(source_reference), dtype="float32")
            sf.write(reference_path, reference_audio, reference_rate)

            generation_args = {
                "text": text,
                "reference_wav_path": str(source_reference),
                "cfg_value": args.cfg_value,
                "inference_timesteps": args.inference_timesteps,
                "max_len": args.max_len,
                "normalize": False,
                "denoise": False,
            }

            model.set_lora_enabled(False)
            _seed_everything(seed)
            baseline = model.generate(**generation_args)
            sf.write(baseline_path, baseline, sample_rate)

            model.set_lora_enabled(True)
            _seed_everything(seed)
            lora = model.generate(**generation_args)
            sf.write(lora_path, lora, sample_rate)

            result = {
                "reference_accent": args.reference_accent,
                "target_accent": args.target_accent,
                "speaker_id": row["speaker_id"],
                "utterance_id": row["utterance_id"],
                "source_reference": str(source_reference),
                "reference_field": args.reference_field,
                "reference": str(reference_path.resolve()),
                "reference_text": row["text"],
                "text": text,
                "seed": seed,
                "baseline": str(baseline_path.resolve()),
                "baseline_duration": len(baseline) / sample_rate,
                "lora": str(lora_path.resolve()),
                "lora_duration": len(lora) / sample_rate,
                "lora_checkpoint": str(args.lora_checkpoint.resolve()),
            }
            manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
            manifest.flush()
            print(
                f"[{index + 1}/{len(selected)}] {row['speaker_id']}: "
                f"baseline={result['baseline_duration']:.2f}s, lora={result['lora_duration']:.2f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
