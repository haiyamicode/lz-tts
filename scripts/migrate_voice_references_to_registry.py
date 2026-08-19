#!/usr/bin/env python3
"""Materialize LZ-TTS voice references and add them to a Lazybird manifest.

This is a one-time data migration. Product voice routing metadata belongs in the
Lazybird voice registry; LZ-TTS receives only the selected reference URL.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEED_VC_RUNTIME = PROJECT_ROOT / "src" / "seed_vc_runtime"
sys.path.insert(0, str(SEED_VC_RUNTIME))

from modules.bigvgan import bigvgan  # noqa: E402
from modules.lazy_embedding_loader import HDF5EmbeddingLoader  # noqa: E402

INTENSITY_LEVELS = (0.25, 0.5, 1.0, 1.5, 2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "local" / "voices.json")
    parser.add_argument("--output-manifest", type=Path, default=PROJECT_ROOT / "local" / "voices.metadata-migrated.json")
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "voice-presets.json")
    parser.add_argument("--embeddings", type=Path, default=PROJECT_ROOT / "data" / "seed-vc" / "embeddings" / "vtts_embeddings.h5")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "local" / "voice-reference-migration")
    parser.add_argument("--storage-prefix", default="global/voice-references")
    parser.add_argument("--cdn-base", default="https://cdn.lazybird.app")
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_key(prefix: str, voice_id: str, name: str) -> str:
    return f"{prefix.strip('/')}/{voice_id}/{name}.mp3"


def _cdn_url(base: str, key: str) -> str:
    return f"{base.rstrip('/')}/{key}"


def _encode_mp3(audio: np.ndarray, sample_rate: int, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "96k", str(output),
        ],
        input=np.asarray(audio, dtype=np.float32).tobytes(),
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed while writing {output}")


def _convert_audio(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
            "-codec:a", "libmp3lame", "-b:a", "96k", str(output),
        ],
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"ffmpeg failed while converting {source}")


class ReferenceExporter:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.main_embeddings = HDF5EmbeddingLoader(args.embeddings, cache_size=1)
        self.vocoder: Any = None

    def close(self) -> None:
        self.main_embeddings.close()

    def _load_vocoder(self) -> Any:
        if self.vocoder is None:
            self.vocoder = bigvgan.BigVGAN.from_pretrained(
                "nvidia/bigvgan_v2_22khz_80band_256x",
                use_cuda_kernel=False,
            )
            self.vocoder.remove_weight_norm()
            self.vocoder = self.vocoder.eval().to(self.args.device)
        return self.vocoder

    def export_embedding(self, key: str, output: Path) -> None:
        if output.is_file():
            return
        embedding = self.main_embeddings.get(key)
        if embedding is None:
            raise KeyError(
                f"Missing reference embedding {key!r} in "
                f"{self.main_embeddings.hdf5_path}"
            )
        mel = embedding["mel_ref"].to(self.args.device)
        with torch.inference_mode():
            audio = self._load_vocoder()(mel).squeeze().float().cpu().numpy()
        _encode_mp3(np.clip(audio, -1.0, 1.0), 22050, output)


def main() -> int:
    args = _parse_args()
    manifest = _load_json(args.manifest)
    catalog = _load_json(args.catalog)
    voices = {voice["id"]: voice for voice in manifest["voices"]}
    assets: dict[str, Path] = {}
    exporter = ReferenceExporter(args)

    try:
        for entry in catalog["voices"]:
            voice_id = entry["id"]
            voice = voices.get(voice_id)
            if voice is None:
                raise KeyError(f"Production voice {voice_id!r} is absent from the Lazybird manifest")

            style_samples = dict(voice.get("styleSamples") or {})
            if entry["pipeline"] == "voxcpm":
                for style, reference in entry["references"].items():
                    if reference.get("url"):
                        style_samples[style] = reference["url"]
                        continue
                    key = _asset_key(args.storage_prefix, voice_id, style)
                    output = args.output_dir / key
                    source = PROJECT_ROOT / reference["path"]
                    if not output.is_file():
                        _convert_audio(source, output)
                    assets[key] = output
                    style_samples[style] = _cdn_url(args.cdn_base, key)
            elif entry["pipeline"] == "sparrow_seed_vc" and not (
                style_samples.get("general") or voice.get("sampleUrl")
            ):
                key = _asset_key(args.storage_prefix, voice_id, "general")
                output = args.output_dir / key
                exporter.export_embedding(f"{voice_id}.general", output)
                assets[key] = output
                style_samples["general"] = _cdn_url(args.cdn_base, key)

            if style_samples:
                voice["styleSamples"] = style_samples

            intensity_samples: dict[str, dict[str, str]] = {}
            for style in entry.get("references", {}):
                if style == "general" or style not in style_samples:
                    continue
                urls: dict[str, str] = {"1": style_samples[style]}
                for intensity in INTENSITY_LEVELS:
                    if intensity == 1.0:
                        continue
                    suffix = f"{intensity:g}"
                    embedding_key = f"{voice_id}.{style}.{suffix}"
                    if embedding_key not in exporter.main_embeddings:
                        continue
                    key = _asset_key(args.storage_prefix, voice_id, f"{style}-{suffix}")
                    output = args.output_dir / key
                    exporter.export_embedding(embedding_key, output)
                    assets[key] = output
                    urls[suffix] = _cdn_url(args.cdn_base, key)
                if len(urls) > 1:
                    intensity_samples[style] = urls
            if intensity_samples:
                voice["styleIntensitySamples"] = intensity_samples

        # The old LZ-TTS preset catalog is not the product voice registry.
        # Every registry voice needs a general reference so supported languages
        # can use VoxCPM without special-casing former Sparrow root voices.
        for voice_id, voice in voices.items():
            if voice.get("sampleUrl"):
                continue
            style_samples = dict(voice.get("styleSamples") or {})
            if style_samples.get("general"):
                continue
            key = _asset_key(args.storage_prefix, voice_id, "general")
            output = args.output_dir / key
            exporter.export_embedding(f"{voice_id}.general", output)
            assets[key] = output
            style_samples["general"] = _cdn_url(args.cdn_base, key)
            voice["styleSamples"] = style_samples
    finally:
        exporter.close()

    missing = [
        voice_id
        for voice_id, voice in voices.items()
        if not (voice.get("sampleUrl") or (voice.get("styleSamples") or {}).get("general"))
    ]
    if missing:
        raise RuntimeError(f"Production voices still lack references: {missing}")

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    asset_manifest = args.output_dir / "assets.json"
    asset_manifest.parent.mkdir(parents=True, exist_ok=True)
    asset_manifest.write_text(
        json.dumps({key: str(path) for key, path in sorted(assets.items())}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output_manifest}")
    print(f"Prepared {len(assets)} assets under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
