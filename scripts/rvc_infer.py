#!/usr/bin/env python3
"""Minimal RVC voice conversion CLI.

Assets live in data/rvc/ (weights, hubert, rmvpe).
Inference source is bundled at src/rvc/.

Usage:
  uv run python scripts/rvc_infer.py local/samples/source.wav -o output.wav
  uv run python scripts/rvc_infer.py input.wav --model mrbeast.pth --pitch 2
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

LZ_ROOT = Path(__file__).resolve().parents[1]
DATA_RVC = LZ_ROOT / "data" / "rvc"
RVC_SRC = LZ_ROOT / "src" / "rvc"

os.environ["weight_root"] = str(DATA_RVC / "weights")
os.environ["index_root"] = str(DATA_RVC / "weights")
os.environ["rmvpe_root"] = str(DATA_RVC / "rmvpe")
os.environ["hubert_path"] = str(DATA_RVC / "hubert" / "hubert_base.pt")

import torch
torch.serialization.add_safe_globals(
    [__import__("fairseq.data.dictionary", fromlist=["Dictionary"]).Dictionary]
)


def convert(
    input_audio: str | Path,
    output: str | Path | None = None,
    model: str = "mrbeast.pth",
    speaker_id: int = 0,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.0,
    rms_mix_rate: float = 0.25,
    protect: float = 0.33,
) -> tuple[int, np.ndarray]:
    output = Path(output or tempfile.mktemp(suffix=".wav")).resolve()
    input_path = str(Path(input_audio).resolve())

    for check, label in [
        (DATA_RVC / "weights" / model, f"model {model}"),
        (DATA_RVC / "hubert" / "hubert_base.pt", "hubert_base.pt"),
        (DATA_RVC / "rmvpe" / "rmvpe.pt", "rmvpe.pt"),
    ]:
        if not check.exists():
            raise FileNotFoundError(f"{label} not found at {check}. Run: uv run python scripts/download_data.py")

    sys.path.insert(0, str(RVC_SRC))
    saved_argv = sys.argv
    sys.argv = ["rvc-infer"]

    try:
        from configs.config import Config
        from infer.modules.vc.modules import VC

        config = Config()
        vc = VC(config)
        sys.argv = saved_argv

        vc.get_vc(model)

        result = vc.vc_single(
            sid=speaker_id,
            input_audio_path=input_path,
            f0_up_key=pitch_shift,
            f0_file=None,
            f0_method=f0_method,
            file_index="",
            file_index2="",
            index_rate=index_rate,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=rms_mix_rate,
            protect=protect,
        )

        msg, (sr, audio_data) = result
        if msg is None or "Success" not in msg:
            raise RuntimeError(f"Inference failed: {msg}")

        sf.write(str(output), audio_data, sr)
    finally:
        sys.argv = saved_argv

    return sr, audio_data


def main():
    parser = argparse.ArgumentParser(description="RVC voice conversion CLI")
    parser.add_argument("input", type=str, help="Input audio file path")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output WAV path")
    parser.add_argument("-m", "--model", type=str, default="mrbeast.pth", help="Model filename")
    parser.add_argument("-s", "--speaker-id", type=int, default=0, help="Speaker ID")
    parser.add_argument("-p", "--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("--f0-method", type=str, default="rmvpe", choices=["pm", "harvest", "crepe", "rmvpe"])
    parser.add_argument("--index-rate", type=float, default=0.0, help="FAISS index blending rate 0-1")
    parser.add_argument("--rms-mix-rate", type=float, default=0.25, help="Volume envelope mix 0-1")
    parser.add_argument("--protect", type=float, default=0.33, help="Protect voiceless consonants 0-0.5")
    args = parser.parse_args()

    sr, _ = convert(
        input_audio=args.input,
        output=args.output,
        model=args.model,
        speaker_id=args.speaker_id,
        pitch_shift=args.pitch,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
    )
    out_path = str(args.output) if args.output else os.path.join(
        tempfile.gettempdir(), f"rvc_output_{os.urandom(4).hex()}.wav"
    )
    print(f"OK  sr={sr}  output={out_path}")


if __name__ == "__main__":
    main()
