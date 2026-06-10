"""RVC (Retrieval-based Voice Conversion) backend for the LZ-TTS API.

Models and assets live in ``data/rvc/``.  The inference source code is bundled
at ``src/rvc/`` so the project is fully self-contained.

Directory layout::

    data/rvc/
        weights/        ← model .pth files (mrbeast.pth, etc.)
        hubert/         ← hubert_base.pt
        rmvpe/          ← rmvpe.pt

    src/rvc/            ← copied RVC inference source (configs, infer/)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

_LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RVC = PROJECT_ROOT / "data" / "rvc"
RVC_SRC = PROJECT_ROOT / "src" / "rvc"

# Env vars read by the RVC source code
os.environ.setdefault("weight_root", str(DATA_RVC / "weights"))
os.environ.setdefault("index_root", str(DATA_RVC / "weights"))
os.environ.setdefault("rmvpe_root", str(DATA_RVC / "rmvpe"))
os.environ.setdefault("hubert_path", str(DATA_RVC / "hubert" / "hubert_base.pt"))

# PyTorch 2.6+ defaults weights_only=True; fairseq's hubert.pt needs pickle
try:
    import torch
    torch.serialization.add_safe_globals(
        [__import__("fairseq.data.dictionary", fromlist=["Dictionary"]).Dictionary]
    )
except Exception:
    pass


@dataclass
class RVCSettings:
    enabled: bool = True
    preload: bool = False
    default_model: str = "mrbeast.pth"
    default_f0_method: str = "rmvpe"
    default_pitch: int = 0
    default_index_rate: float = 0.0
    default_rms_mix_rate: float = 0.25
    default_protect: float = 0.33


class RVCBackend:
    """Lazy-loading RVC voice conversion backend."""

    def __init__(self, settings: RVCSettings | None = None):
        self.settings = settings or RVCSettings()
        self._lock = threading.Lock()
        self._vc = None
        self._current_model: str | None = None
        self._weights_dir = DATA_RVC / "weights"

    def _validate_assets(self) -> None:
        missing = []
        if not (DATA_RVC / "hubert" / "hubert_base.pt").exists():
            missing.append("hubert_base.pt")
        if not (DATA_RVC / "rmvpe" / "rmvpe.pt").exists():
            missing.append("rmvpe.pt")
        if not self._weights_dir.exists():
            missing.append("weights/")
        if missing:
            raise FileNotFoundError(
                f"Missing RVC assets: {', '.join(missing)}. "
                "Run: uv run python scripts/download_data.py"
            )

    def _load_vc(self):
        self._validate_assets()

        rvc_src = str(RVC_SRC)
        if rvc_src not in sys.path:
            sys.path.insert(0, rvc_src)

        saved_argv = sys.argv
        sys.argv = ["rvc-backend"]

        try:
            from configs.config import Config
            from infer.modules.vc.cached_vc import CachedVC

            config = Config()
            self._vc = CachedVC(config, max_cache_size=5)
            _LOGGER.info(
                "RVC backend ready: device=%s is_half=%s", config.device, config.is_half
            )
        finally:
            sys.argv = saved_argv

    def _ensure_loaded(self):
        if self._vc is None:
            self._load_vc()

    def list_models(self) -> list[str]:
        if not self._weights_dir.exists():
            return []
        return sorted(p.name for p in self._weights_dir.glob("*.pth"))

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.settings.enabled,
            "loaded": self._vc is not None,
            "current_model": self._current_model,
            "available_models": self.list_models(),
            "data_dir": str(DATA_RVC),
        }

    def convert(
        self,
        audio_bytes: bytes,
        model: str = "mrbeast.pth",
        f0_method: str = "rmvpe",
        pitch: int = 0,
        index_rate: float = 0.0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        output_format: str = "wav",
    ) -> tuple[bytes, int]:
        """Convert audio bytes through RVC. Returns ``(audio_bytes, sample_rate)``."""
        with self._lock:
            self._ensure_loaded()

            model_path = self._weights_dir / model
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            tmp_in = DATA_RVC / "tmp" / f"{uuid.uuid4().hex}.input"
            tmp_in.parent.mkdir(parents=True, exist_ok=True)
            tmp_in.write_bytes(audio_bytes)

            try:
                self._vc.get_vc(model)
                self._current_model = model

                result = self._vc.vc_single(
                    sid=0,
                    input_audio_path=str(tmp_in),
                    f0_up_key=pitch,
                    f0_file=None,
                    f0_method=f0_method,
                    file_index="",
                    file_index2="",
                    index_rate=index_rate,
                    filter_radius=3,
                    resample_sr=0,
                    rms_mix_rate=rms_mix_rate,
                    protect=protect,
                )

                msg, (sr, audio_data) = result
                if msg is None or "Success" not in msg:
                    raise RuntimeError(f"RVC inference failed: {msg}")

                buf = io.BytesIO()
                sf.write(buf, audio_data, sr, format="WAV")
                wav_bytes = buf.getvalue()

                if output_format == "mp3":
                    from pydub import AudioSegment

                    wav_buf = io.BytesIO(wav_bytes)
                    seg = AudioSegment.from_wav(wav_buf)
                    mp3_buf = io.BytesIO()
                    seg.export(mp3_buf, format="mp3", bitrate="320k", parameters=["-q:a", "0"])
                    return mp3_buf.getvalue(), sr

                return wav_bytes, sr

            finally:
                tmp_in.unlink(missing_ok=True)
