"""YuE2-3B: style + lyrics to a full song with vocals, prompt only.

The song length is not a request parameter: YuE2 plans the structure (intro,
verse, chorus...) from the lyrics sections and writes as much as they need.
YuE2 runs bf16 on Ampere or newer.
"""

from __future__ import annotations

import dataclasses
import gc
import random
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch

from paths import YUE2_DIR

MIN_COMPUTE_CAPABILITY = (8, 0)


@dataclass
class SongRequest:
    style: str
    lyrics: str
    cot: str = "full"  # "full" plans melody and chords, "melody" melody only, "off" no plan
    cfg_scale: float | None = None
    seed: int | None = None
    abc: str | None = None  # your own ABC melody score (needs cot "melody" or "full"); skips the planning pass
    ode_steps: int | None = None  # flow-matching steps of the audio stage (model default 32)
    # Sampling overrides for the planning pass and the song-token pass: any of temperature, top_p,
    # top_k, repetition_penalty, penalty_window, min_tokens, max_tokens (about 25 song tokens per second).
    abc_sampling: dict | None = None
    semantic_sampling: dict | None = None


class YuE2Song:
    def __init__(self, model_dir: Path = YUE2_DIR, device: str = "cuda:0"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._pipe = None
        self._base_config = None

    def load(self) -> None:
        if self._pipe is not None:
            return
        capability = torch.cuda.get_device_capability(torch.device(self.device))
        if capability < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"YuE2 needs bf16 (compute capability >= {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}); "
                f"{self.device} is {capability[0]}.{capability[1]}"
            )
        from yue2 import YuE2Pipeline

        self._pipe = YuE2Pipeline.from_pretrained(
            "m-a-p/YuE2-3B",
            device=self.device,
            cache_dir=str(self.model_dir / "hf_cache"),
            local_files_only=True,
            progress=False,
        )
        self._base_config = self._pipe.generation_config

    def unload(self) -> None:
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.close()
        del pipe
        gc.collect()

    def generate(self, request: SongRequest, out_path: str | Path) -> tuple[Path, int]:
        """Render the song to ``out_path`` (48 kHz WAV). Returns (path, seed used)."""
        self.load()
        seed = request.seed if request.seed is not None else random.randrange(2**31 - 1)
        base = self._base_config
        self._pipe.generation_config = (
            base if request.ode_steps is None else dataclasses.replace(base, ode_steps=request.ode_steps)
        )
        song = self._pipe(
            style=request.style,
            lyrics=request.lyrics,
            cot=request.cot,
            cfg_scale=request.cfg_scale,
            seed=seed,
            abc=request.abc,
            abc_sampling=request.abc_sampling,
            semantic_sampling=request.semantic_sampling,
        )
        out_path = Path(out_path)
        sf.write(str(out_path), song.audio, song.sample_rate, subtype="PCM_16")
        return out_path, seed
