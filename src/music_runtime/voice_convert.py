"""YingMusic-SVC: convert a sung vocal toward the voice in a reference clip.

The model code is vendored under vendor/yingmusic/svc (``my_inference``,
``mm4``, ``hf_utils``, ``modules``). Its CLI and the Remix accompaniment step
are removed; this module calls ``load_models_api`` and ``run_inference``
directly. Weights (``YingMusic-SVC-full.pt``, ``YingMusic-SVC.yml``) and the
offline Hugging Face cache (RMVPE, CAMPPlus, BigVGAN, Whisper) live in
data/yingmusic-svc.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import torch

from paths import VENDOR_SVC_DIR, YINGMUSIC_DIR


class YingMusicConverter:
    """YingMusic-SVC loaded once; ``convert`` is serialized by a lock."""

    def __init__(
        self,
        model_dir: Path = YINGMUSIC_DIR,
        device: str = "cuda:0",
        fp16: bool = True,
        diffusion_steps: int = 100,
    ):
        self.model_dir = Path(model_dir)
        self.device = device
        self.fp16 = fp16
        self.diffusion_steps = diffusion_steps
        self._bundle = None
        self._mi = None
        self._lock = threading.Lock()

    def _args(self, **extra) -> argparse.Namespace:
        args = argparse.Namespace(
            checkpoint=str(self.model_dir / "YingMusic-SVC-full.pt"),
            config=str(self.model_dir / "YingMusic-SVC.yml"),
            cuda=self.device.split(":", 1)[-1] if ":" in self.device else 0,
            fp16=self.fp16,
            diffusion_steps=self.diffusion_steps,
            length_adjust=1.0,
            inference_cfg_rate=0.7,
            f0_condition=True,
            semi_tone_shift=None,
        )
        for key, value in extra.items():
            setattr(args, key, value)
        return args

    def load(self) -> None:
        if self._bundle is not None:
            return
        if str(VENDOR_SVC_DIR) not in sys.path:
            sys.path.insert(0, str(VENDOR_SVC_DIR))
        import my_inference

        args = self._args()
        self._bundle = my_inference.load_models_api(args, device=torch.device(self.device))
        self._mi = my_inference

    def unload(self) -> None:
        self._bundle = None
        self._mi = None

    def convert(self, source_path: str | Path, reference_path: str | Path, out_path: str | Path) -> Path:
        """Re-voice the vocal at ``source_path`` with the timbre of ``reference_path``; writes ``out_path`` (WAV, 44.1 kHz mono)."""
        self.load()
        out_path = Path(out_path)
        with self._lock, tempfile.TemporaryDirectory(prefix="ymsvc-") as scratch:
            args = self._args(
                source=str(source_path),
                target=str(reference_path),
                output=scratch,
                expname="vc",
                uuid=uuid.uuid4().hex,
            )
            produced = self._mi.run_inference(args, self._bundle, device=torch.device(self.device))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(produced, out_path)
        return out_path
