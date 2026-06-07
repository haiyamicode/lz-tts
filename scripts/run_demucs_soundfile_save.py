#!/usr/bin/env python3
"""Run Demucs while saving WAV output through soundfile instead of torchaudio."""

import sys
from pathlib import Path

import soundfile as sf

import demucs.audio
import demucs.separate


def save_audio_soundfile(
    wav,
    path,
    samplerate,
    bitrate=320,
    clip="rescale",
    bits_per_sample=16,
    as_float=False,
    preset=2,
):
    wav = demucs.audio.prevent_clip(wav, mode=clip)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != ".wav":
        raise ValueError(f"This wrapper only supports WAV output, got {path}")
    data = wav.detach().cpu().numpy().T
    subtype = "FLOAT" if as_float else {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}[bits_per_sample]
    sf.write(path, data, samplerate, subtype=subtype)


def main() -> None:
    demucs.audio.save_audio = save_audio_soundfile
    demucs.separate.save_audio = save_audio_soundfile
    sys.argv = ["demucs", *sys.argv[1:]]
    demucs.separate.main()


if __name__ == "__main__":
    main()
