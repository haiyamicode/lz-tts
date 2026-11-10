"""generate-music: ACE-Step song, then YingMusic-SVC when there is a clone reference.

Without a reference the song is the ACE-Step render, untouched.
With a reference: ACE-Step renders the song, BS-RoFormer splits the lead vocal
from the rest, YingMusic-SVC re-voices the vocal toward the reference, and the
converted vocal is mixed back over the accompaniment.
"""

from __future__ import annotations

import gc
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from separation import SAMPLE_RATE, VocalSeparator
from song import AceStepSong, SongRequest
from voice_convert import YingMusicConverter

PEAK_CEILING = 0.98


@dataclass
class RenderResult:
    path: Path
    seed: int
    mode: str  # "prompt" (ACE-Step only) or "clone" (ACE-Step + YingMusic-SVC)


def mix_over_accompaniment(vocal_path: Path, accompaniment: np.ndarray, out_path: Path) -> Path:
    """Sum a mono vocal onto a (2, samples) accompaniment at 44.1 kHz and write 16-bit WAV."""
    vocal, sr = sf.read(str(vocal_path), dtype="float32", always_2d=False)
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=1)
    if sr != SAMPLE_RATE:
        vocal = librosa.resample(vocal, orig_sr=sr, target_sr=SAMPLE_RATE)

    backing = np.asarray(accompaniment, dtype=np.float32).T  # (samples, 2)
    n = backing.shape[0]
    vocal = vocal[:n]
    if vocal.shape[0] < n:
        vocal = np.pad(vocal, (0, n - vocal.shape[0]))

    mixed = backing + vocal[:, None]
    peak = float(np.abs(mixed).max()) if mixed.size else 0.0
    if peak > PEAK_CEILING:
        mixed *= PEAK_CEILING / peak

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), mixed, SAMPLE_RATE, subtype="PCM_16")
    return out_path


class MusicPipeline:
    """One model group on the GPU at a time: ACE-Step ("song"), or BS-RoFormer + YingMusic-SVC ("voice").

    ACE-Step alone takes ~22 GB, so a clone job loads it, renders, unloads it, then loads
    the two voice models. A model group stays resident until the other one is needed.
    """

    def __init__(self, device: str = "cuda:0", fp16: bool = True):
        self.song = AceStepSong(device=device)
        self.separator = VocalSeparator(device=device)
        self.converter = YingMusicConverter(device=device, fp16=fp16)
        self._resident: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load ACE-Step so /health reports ok only when it can generate (the voice models load on first clone)."""
        with self._lock:
            self._activate("song")

    def _activate(self, group: str) -> None:
        if self._resident == group:
            return
        if self._resident == "song":
            self.song.unload()
        elif self._resident == "voice":
            self.separator.unload()
            self.converter.unload()
        self._resident = None
        gc.collect()
        torch.cuda.empty_cache()
        started = time.monotonic()
        if group == "song":
            self.song.load()
        else:
            self.separator.load()
            self.converter.load()
        self._resident = group
        print(f"[timing] load {group}: {time.monotonic() - started:.1f}s", flush=True)

    def render(self, request: SongRequest, output_dir: Path, reference_path: Path | None = None) -> RenderResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with self._lock, tempfile.TemporaryDirectory(prefix="music-", dir=output_dir) as work_dir:
            work = Path(work_dir)
            self._activate("song")
            started = time.monotonic()
            song_path, seed = self.song.generate(request, work / "song.wav")
            print(f"[timing] ace-step generate: {time.monotonic() - started:.1f}s", flush=True)
            if reference_path is None:
                final = output_dir / f"song-{seed}.wav"
                shutil.move(str(song_path), final)
                return RenderResult(path=final, seed=seed, mode="prompt")

            self._activate("voice")
            started = time.monotonic()
            vocal, accompaniment = self.separator.split(song_path)
            print(f"[timing] separate: {time.monotonic() - started:.1f}s", flush=True)
            started = time.monotonic()
            vocal_path = work / "vocal.wav"
            sf.write(str(vocal_path), vocal.T, SAMPLE_RATE, subtype="FLOAT")
            converted = self.converter.convert(vocal_path, reference_path, work / "converted.wav")
            print(f"[timing] convert: {time.monotonic() - started:.1f}s", flush=True)
            final = mix_over_accompaniment(converted, accompaniment, output_dir / f"song-{seed}.wav")
            return RenderResult(path=final, seed=seed, mode="clone")
