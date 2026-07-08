import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def normalize_audio_rms(audio, target_rms=0.1, peak_limit=0.99, eps=1e-6):
    if target_rms is None or float(target_rms) <= 0:
        return audio.float()

    audio = audio.float()
    rms = torch.sqrt(torch.mean(audio.pow(2))).clamp_min(float(eps))
    audio = audio * (float(target_rms) / rms)

    if peak_limit is not None and float(peak_limit) > 0:
        peak = audio.abs().amax()
        peak_value = float(peak.detach().cpu())
        if peak_value > float(peak_limit):
            audio = audio * (float(peak_limit) / peak_value)

    return audio.clamp(-1.0, 1.0)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}
vocos_mel_extractors = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window  # pylint: disable=global-statement,global-variable-not-assigned
    if f"{str(fmax)}_{str(y.device)}" not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + "_" + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect"
    )
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=hann_window[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + "_" + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def vocos_mel_spectrogram(y, sampling_rate=24000, n_fft=1024, hop_size=256, num_mels=100):
    from vocos.feature_extractors import MelSpectrogramFeatures  # pylint: disable=import-outside-toplevel

    key = (str(y.device), str(y.dtype), sampling_rate, n_fft, hop_size, num_mels)
    if key not in vocos_mel_extractors:
        vocos_mel_extractors[key] = MelSpectrogramFeatures(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_size,
            n_mels=num_mels,
        ).to(device=y.device)

    if y.dim() == 2 and y.size(0) > 1:
        y = y.mean(dim=0, keepdim=True)
    return vocos_mel_extractors[key](y)
