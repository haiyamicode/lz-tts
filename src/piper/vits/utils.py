import numpy as np
import torch


def to_gpu(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().cuda(non_blocking=True)


def audio_float_to_int16(
    audio: np.ndarray, max_wav_value: float = 32767.0
) -> np.ndarray:
    """Convert float audio in [-1, 1] to int16 without per-sample gain changes."""
    audio_i16 = np.clip(audio, -1.0, 1.0) * max_wav_value
    return audio_i16.astype("int16")
