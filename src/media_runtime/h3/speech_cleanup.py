"""The clean voice a lipsync render hears (``lipsync``).

Room tone and noise in a job's audio make H3 invent things - overlay graphics,
punch-ins, flares - and blur the pauses the pins are placed in. So the render,
the transcription and the pin placement get the denoised voice:

1. the audio goes to 16 kHz mono;
2. ZipEnhancer (ModelScope ``iic/speech_zipenhancer_ans_multiloss_16k_base``;
   the network vendored in :mod:`h3.zipenhancer`) removes the noise, in the 2 s
   windows (1.5 s stride, each window's outer 0.25 s dropped) of ModelScope's own
   pipeline, batched on the GPU;
3. the result is set to the input's speech level, its peak kept under full scale.

No music separation (Demucs was dropped): a song's backing track reaches the
render as it is. The caller's audio is still what the job delivers. ZipEnhancer
is ``data/lipsync/speech-cleanup/zipenhancer`` (``SPEECH_CLEANUP_ZIPENHANCER_*``
in :mod:`h3.compat`), loaded once and kept on the GPU.
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
_ZIPENHANCER: Any = None


def _zipenhancer():
    """ZipEnhancer, loaded once and kept on the GPU (about 8 MB of weights)."""
    global _ZIPENHANCER
    if _ZIPENHANCER is None:
        import torch

        from .compat import SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH, SPEECH_CLEANUP_ZIPENHANCER_PATH, project_path
        from .zipenhancer.model import AttrDict, ZipEnhancer

        weights, config_path = project_path(SPEECH_CLEANUP_ZIPENHANCER_PATH), project_path(SPEECH_CLEANUP_ZIPENHANCER_CONFIG_PATH)
        missing = [str(path) for path in (weights, config_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(f"ZipEnhancer is missing (expected {', '.join(missing)})")
        config = json.loads(config_path.read_text())["model"]
        model = ZipEnhancer(AttrDict({key: config[key] for key in (
            "num_tsconformers", "dense_channel", "former_conf", "batch_first", "model_num_spks")}))
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True)["generator"])
        _ZIPENHANCER = model.eval().to(_device())
    return _ZIPENHANCER


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
        with torch.no_grad():
            enhanced = torch.cat([_enhance_windows(windows[i:i + BATCH], model, device) for i in range(0, len(windows), BATCH)])
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()  # the activations; the weights stay
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
    """The job's voice, denoised, as a 16 kHz mono wav at ``output_path``."""
    with _LOCK:
        began = time.perf_counter()
        samples = _decode(audio_path, rate=SAMPLE_RATE, channels=1)[0]
        voice = matched_level(enhance(samples, _zipenhancer()), samples)
    write_wav(output_path, voice)
    log_event("voice_cleaned", job_id=job_id, audio_seconds=round(len(voice) / SAMPLE_RATE, 3),
              zipenhancer_seconds=round(time.perf_counter() - began, 3))
    return output_path
