"""ACE-Step 1.5 XL Turbo: text (+ lyrics) to a full song, prompt only.

ACE-Step runs bf16 on Ampere and newer only. Pre-Ampere cards fall back to
fp16, which produces NaN latents (observed on a V100), so this module refuses
to load there rather than generating silence.
"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import torch

from paths import ACESTEP_DIR

MIN_COMPUTE_CAPABILITY = (8, 0)
CONFIG_PATH = "acestep-v15-xl-turbo"


@dataclass
class SongRequest:
    caption: str
    duration: float
    lyrics: str = ""
    instrumental: bool = False
    vocal_language: str = "en"
    bpm: int | None = None
    keyscale: str | None = None
    timesignature: str = "4/4"
    seed: int | None = None
    inference_steps: int = 8


class AceStepSong:
    def __init__(self, model_dir: Path = ACESTEP_DIR, device: str = "cuda:0"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._dit = None
        self._llm = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._dit is not None:
            return
        capability = torch.cuda.get_device_capability(torch.device(self.device))
        if capability < MIN_COMPUTE_CAPABILITY:
            raise RuntimeError(
                f"ACE-Step needs bf16 (compute capability >= {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}); "
                f"{self.device} is {capability[0]}.{capability[1]}"
            )
        # Checkpoints resolve as <ACESTEP_CHECKPOINTS_DIR>/<component>; the model
        # dir holds the components directly, so point the resolver at it.
        os.environ["ACESTEP_CHECKPOINTS_DIR"] = str(self.model_dir)
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler

        dit = AceStepHandler()
        status = dit.initialize_service(
            project_root=str(self.model_dir),
            config_path=CONFIG_PATH,
            device=self.device,
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=False,
            offload_dit_to_cpu=False,
        )
        message = status[0] if isinstance(status, tuple) else str(status)
        if not message.startswith("[OK]"):
            raise RuntimeError(f"ACE-Step init failed: {message}")
        self._llm = LLMHandler()
        self._dit = dit

    def unload(self) -> None:
        """Drop the handler's models; the caller then frees the CUDA cache."""
        dit, self._dit, self._llm = self._dit, None, None
        if dit is not None:
            for name in ("model", "vae", "text_encoder", "text_tokenizer", "silence_latent"):
                setattr(dit, name, None)

    def generate(self, request: SongRequest, out_path: str | Path) -> tuple[Path, int]:
        """Render the song to ``out_path`` (WAV as ACE-Step writes it). Returns (path, seed used)."""
        self.load()
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        seed = request.seed if request.seed is not None else random.randrange(2**31 - 1)
        params = GenerationParams(
            task_type="text2music",
            caption=request.caption,
            lyrics="" if request.instrumental else request.lyrics,
            instrumental=request.instrumental,
            vocal_language=request.vocal_language,
            bpm=request.bpm,
            keyscale=request.keyscale or "",
            timesignature=request.timesignature,
            duration=float(request.duration),
            inference_steps=request.inference_steps,
            seed=seed,
            shift=3.0,
            infer_method="ode",
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_lyrics=False,
            use_cot_language=False,
        )
        config = GenerationConfig(
            batch_size=1,
            use_random_seed=False,
            seeds=[seed],
            audio_format="wav",
        )
        with self._lock, tempfile.TemporaryDirectory(prefix="acestep-") as scratch:
            result = generate_music(self._dit, self._llm, params, config, save_dir=scratch)
            if not result.success or not result.audios:
                raise RuntimeError(f"ACE-Step generation failed: {result.error}")
            produced = Path(result.audios[0]["path"])
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(produced), out_path)
        return out_path, seed
