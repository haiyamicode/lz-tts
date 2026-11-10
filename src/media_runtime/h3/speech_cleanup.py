"""The clean voice an lipsync render hears (``lipsync``).

Music, room tone and noise in a job's audio make H3 invent things - overlay
graphics, punch-ins, flares - and blur the pauses the pins are placed in. So the
render, Whisper and the pin placement all get only the voice:

1. Demucs ``htdemucs`` separates the vocals (44.1 kHz stereo in, the rate the
   model was trained at);
2. the vocals go to 16 kHz mono, all the rest needs;
3. ZipEnhancer (ModelScope ``iic/speech_zipenhancer_ans_multiloss_16k_base``;
   the network vendored in :mod:`ai_worker._vendor.zipenhancer`) removes the
   noise left, in the 2 s windows (1.5 s stride, each window's outer 0.25 s
   dropped) of ModelScope's own pipeline, batched on the GPU;
4. the result is set to the vocals' speech level, its peak kept under full
   scale.

The caller's audio is still what the job delivers. Both models are the
``speech_cleanup`` model artifact
(:data:`apiserver.model_artifact_prepare.SPEECH_CLEANUP_ASSET_PATHS`).
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Any

from .compat import log_event

SAMPLE_RATE = 16000
DEMUCS_RATE = 44100
#: ZipEnhancer's STFT and its pipeline's windows, samples at 16 kHz.
N_FFT, HOP, WIN = 400, 100, 400
COMPRESS = 0.3
WINDOW, STRIDE = 2 * SAMPLE_RATE, int(2 * SAMPLE_RATE * 0.75)
#: Inputs up to this long are enhanced in one pass, as in ModelScope's pipeline.
WHOLE_MAX = 3 * WINDOW
#: ZipEnhancer windows per forward pass.
BATCH = 16
#: Speech level: the 90th percentile of 10 ms RMS, in dB.
LEVEL_HOP = SAMPLE_RATE // 100
PEAK = 0.98

_LOCK = threading.Lock()
_DEMUCS: Any = None
_ZIPENHANCER: Any = None


def _paths() -> tuple[Path, Path, Path]:
    from .compat import (
        SPEECH_CLEANUP_DEMUCS_PATH,
        SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH,
        SPEECH_CLEANUP_ZIPENHANCER_PATH,
    )
    from .compat import project_path

    paths = tuple(project_path(path) for path in (
        SPEECH_CLEANUP_DEMUCS_PATH, SPEECH_CLEANUP_ZIPENHANCER_PATH, SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"speech_cleanup models are missing (expected prepare-provisioned {', '.join(missing)})")
    return paths


def _models():
    """Both models, loaded once and kept in host memory between jobs."""
    global _DEMUCS, _ZIPENHANCER
    if _DEMUCS is None or _ZIPENHANCER is None:
        import torch
        from demucs.states import load_model

        from .zipenhancer.model import AttrDict, ZipEnhancer

        demucs_path, zipenhancer_path, config_path = _paths()
        # A trusted, provisioned Demucs package: it pickles its model class, which
        # torch.load's default weights_only mode refuses.
        _DEMUCS = load_model(torch.load(demucs_path, map_location="cpu", weights_only=False)).eval()
        config = json.loads(config_path.read_text())["model"]
        zipenhancer = ZipEnhancer(AttrDict({key: config[key] for key in (
            "num_tsconformers", "dense_channel", "former_conf", "batch_first", "model_num_spks")}))
        zipenhancer.load_state_dict(torch.load(zipenhancer_path, map_location="cpu", weights_only=True)["generator"])
        _ZIPENHANCER = zipenhancer.eval()
    return _DEMUCS, _ZIPENHANCER


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _decode(path: Path, *, rate: int, channels: int):
    import numpy as np

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", str(channels), "-ar", str(rate), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32).reshape(-1, channels).T.copy()


def separate_vocals(path: Path, model) -> Any:
    """Demucs vocals of ``path`` as 16 kHz mono float32."""
    import julius
    import torch
    from demucs.apply import apply_model

    mix = torch.from_numpy(_decode(path, rate=DEMUCS_RATE, channels=2))
    # Demucs' own normalization (demucs.separate): zero mean, unit variance.
    reference = mix.mean(0)
    mean, std = reference.mean(), reference.std().clamp_min(1e-8)
    device = _device()
    try:
        model.to(device)
        with torch.no_grad():
            sources = apply_model(model, ((mix - mean) / std)[None], shifts=1, split=True, overlap=0.25, device=device)[0]
    finally:
        model.cpu()
        if device == "cuda":
            torch.cuda.empty_cache()
    vocals = (sources[model.sources.index("vocals")].cpu() * std + mean).mean(0)
    return julius.resample_frac(vocals, DEMUCS_RATE, SAMPLE_RATE).numpy().astype("float32")


def _enhance_windows(windows, model, device: str):
    """ZipEnhancer on equal-length windows ``[N, L]``, as ModelScope's model wraps
    it: each window at unit RMS in, the compressed magnitude and the phase of its
    STFT through the network, back out at its own level (a silent window stays
    silent)."""
    import torch

    x = windows.to(device)
    energy = (x ** 2).sum(dim=1, keepdim=True)
    norm = torch.sqrt(x.shape[1] / energy.clamp_min(1e-20)) * (energy > 0)
    hann = torch.hann_window(WIN, device=device)
    spec = torch.stft(x * norm, N_FFT, hop_length=HOP, win_length=WIN, window=hann, center=True,
                      pad_mode="reflect", normalized=False, return_complex=True)
    magnitude = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9) ** COMPRESS
    phase = torch.atan2(spec.imag, spec.real + 1e-5)
    amplitude, phase = model(magnitude, phase)[:2]
    amplitude = amplitude ** (1.0 / COMPRESS)
    wav = torch.istft(torch.complex(amplitude * torch.cos(phase), amplitude * torch.sin(phase)), N_FFT,
                      hop_length=HOP, win_length=WIN, window=hann, center=True, length=x.shape[1])
    return (wav / norm.clamp_min(1e-20) * (energy > 0)).cpu()


def enhance(samples, model):
    """ZipEnhancer over 16 kHz mono ``samples``: in one pass up to ``WHOLE_MAX``,
    else in ``WINDOW``-long windows every ``STRIDE``, each keeping its middle (the
    first from the start, the last to the end)."""
    import numpy as np
    import torch

    samples = np.asarray(samples, dtype=np.float32)
    length = len(samples)
    if length <= WHOLE_MAX:
        span, steps = max(length, WINDOW), 0
    else:
        span, steps = WINDOW, -(-(length - WINDOW) // STRIDE)
    padded = torch.from_numpy(np.concatenate([samples, np.zeros(span + steps * STRIDE - length, dtype=np.float32)]))
    windows = torch.stack([padded[step * STRIDE:step * STRIDE + span] for step in range(steps + 1)])
    device = _device()
    try:
        model.to(device)
        with torch.no_grad():
            enhanced = torch.cat([_enhance_windows(windows[i:i + BATCH], model, device) for i in range(0, len(windows), BATCH)])
    finally:
        model.cpu()
        if device == "cuda":
            torch.cuda.empty_cache()
    if steps == 0:
        return enhanced[0, :length].numpy()
    out = torch.zeros(len(padded))
    give_up = (WINDOW - STRIDE) // 2
    for step in range(steps + 1):
        keep_from = 0 if step == 0 else give_up
        keep_to = WINDOW if step == steps else WINDOW - give_up
        out[step * STRIDE + keep_from:step * STRIDE + keep_to] = enhanced[step, keep_from:keep_to]
    return out[:length].numpy()


def speech_level(samples) -> float:
    """The 90th percentile of 10 ms RMS, dB."""
    import numpy as np

    windows = len(samples) // LEVEL_HOP
    if windows == 0:
        return -240.0
    frames = np.asarray(samples[: windows * LEVEL_HOP], dtype=np.float64).reshape(windows, LEVEL_HOP)
    return float(np.percentile(20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-12), 90))


def matched_level(samples, reference):
    """``samples`` at ``reference``'s speech level, the peak kept under ``PEAK``."""
    import numpy as np

    out = np.asarray(samples, dtype=np.float32) * np.float32(10 ** ((speech_level(reference) - speech_level(samples)) / 20))
    peak = float(np.abs(out).max()) if len(out) else 0.0
    return out * np.float32(PEAK / peak) if peak > PEAK else out


def write_wav(path: Path, samples) -> Path:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    return path


def clean_voice(audio_path: Path, output_path: Path, *, job_id: str = "") -> Path:
    """The job's voice, separated and denoised, as a 16 kHz mono wav at ``output_path``."""
    with _LOCK:
        demucs, zipenhancer = _models()
        began = time.perf_counter()
        vocals = separate_vocals(audio_path, demucs)
        separated = time.perf_counter()
        voice = matched_level(enhance(vocals, zipenhancer), vocals)
    write_wav(output_path, voice)
    log_event("voice_cleaned", job_id=job_id, audio_seconds=round(len(voice) / SAMPLE_RATE, 3),
              demucs_seconds=round(separated - began, 3), zipenhancer_seconds=round(time.perf_counter() - separated, 3))
    return output_path
