"""BS-RoFormer vocal separation for the reference path.

Splits a generated song into the lead vocal (which YingMusic converts) and the
accompaniment (mix minus vocal, so backing vocals and instruments are kept).
Vendored model code lives in vendor/yingmusic/separation; the checkpoint and
config are data/yingmusic-svc/bs_roformer.ckpt and config_bd_roformer.yaml.
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import librosa
import numpy as np
import torch

from paths import VENDOR_SEPARATION_DIR, YINGMUSIC_DIR

SAMPLE_RATE = 44100


class VocalSeparator:
    def __init__(self, model_dir: Path = YINGMUSIC_DIR, device: str = "cuda:0"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._model = None
        self._config = None
        self._lock = threading.Lock()

    def load(self) -> None:
        if self._model is not None:
            return
        if str(VENDOR_SEPARATION_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_SEPARATION_DIR))
        from utils.model_utils import load_start_checkpoint
        from utils.settings import get_model_from_config

        model, config = get_model_from_config("bs_roformer", str(self.model_dir / "config_bd_roformer.yaml"))
        checkpoint_path = str(self.model_dir / "bs_roformer.ckpt")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        load_start_checkpoint(
            argparse.Namespace(start_check_point=checkpoint_path, model_type="bs_roformer", lora_checkpoint=None),
            model,
            checkpoint,
            type_="inference",
        )
        self._model = model.to(self.device).eval()
        self._config = config

    def unload(self) -> None:
        self._model = None
        self._config = None

    def split(self, audio_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """Return (vocal, accompaniment), both float32 (2, samples) at 44.1 kHz."""
        self.load()
        from utils.audio_utils import denormalize_audio, normalize_audio
        from utils.model_utils import demix

        config = self._config
        mix, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=False)
        if mix.ndim == 1:
            mix = np.stack([mix, mix])
        mix_orig = mix.copy()

        normalized = bool(config.inference.get("normalize", False)) is True
        if normalized:
            mix, norm_params = normalize_audio(mix)

        with self._lock, torch.no_grad():
            stems = demix(config, self._model, mix, torch.device(self.device), model_type="bs_roformer", pbar=False)
        vocal = stems["vocals"]
        if normalized:
            vocal = denormalize_audio(vocal, norm_params)
        vocal = np.asarray(vocal, dtype=np.float32)
        accompaniment = (mix_orig - vocal).astype(np.float32)
        return vocal, accompaniment
