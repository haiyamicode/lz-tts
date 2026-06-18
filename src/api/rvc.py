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
from dataclasses import dataclass, field
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
    cache_size: int = 5
    preload_models: list[str] = field(default_factory=list)
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
            self._vc = CachedVC(config, max_cache_size=self.settings.cache_size)
            _LOGGER.info(
                "RVC backend ready: device=%s is_half=%s cache_size=%d",
                config.device,
                config.is_half,
                self.settings.cache_size,
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

    def preload_models(self, models: list[str]) -> None:
        """Load selected RVC voice models into the LRU cache."""
        if not models:
            return
        if len(models) > self.settings.cache_size:
            raise ValueError(
                f"RVC preload model count ({len(models)}) exceeds cache_size ({self.settings.cache_size})"
            )

        with self._lock:
            self._ensure_loaded()
            for model in models:
                model_path = self._weights_dir / model
                if not model_path.exists():
                    raise FileNotFoundError(f"Model not found: {model_path}")

                _LOGGER.info("Loading RVC model model=%s", model)
                self._vc.get_vc(model)
                self._current_model = model
                _LOGGER.info("Loaded RVC model model=%s", model)

    def status(self) -> dict[str, Any]:
        cache = None
        if self._vc is not None and hasattr(self._vc, "get_cache_stats"):
            cache = self._vc.get_cache_stats()
        return {
            "enabled": self.settings.enabled,
            "loaded": self._vc is not None,
            "current_model": self._current_model,
            "cache_size": self.settings.cache_size,
            "preload_models": self.settings.preload_models,
            "cache": cache,
            "available_models": self.list_models(),
            "data_dir": str(DATA_RVC),
        }

    def convert(
        self,
        audio_bytes: bytes,
        model: str,
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

    def convert_batch(
        self,
        audio_items: list[bytes],
        model: str,
        f0_method: str = "rmvpe",
        pitch: int = 0,
        index_rate: float = 0.0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        output_format: str = "wav",
    ) -> list[tuple[bytes, int]]:
        """Convert multiple audio items through one real RVC batch call."""
        if not audio_items:
            raise ValueError("audio_items must not be empty")
        if index_rate != 0:
            raise ValueError("RVC real batch conversion does not support index_rate != 0")

        with self._lock:
            self._ensure_loaded()

            model_path = self._weights_dir / model
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            from infer.lib.audio import load_audio
            from infer.modules.vc.utils import load_hubert

            tmp_paths = []
            try:
                for audio_bytes in audio_items:
                    tmp_in = DATA_RVC / "tmp" / f"{uuid.uuid4().hex}.input"
                    tmp_in.parent.mkdir(parents=True, exist_ok=True)
                    tmp_in.write_bytes(audio_bytes)
                    tmp_paths.append(tmp_in)

                self._vc.get_vc(model)
                self._current_model = model

                if self._vc.hubert_model is None:
                    self._vc.hubert_model = load_hubert(self._vc.config)

                audios = []
                for tmp_path in tmp_paths:
                    audio = load_audio(str(tmp_path), 16000)
                    audio_max = np.abs(audio).max() / 0.95
                    if audio_max > 1:
                        audio = audio / audio_max
                    audios.append(audio)

                times = [0, 0, 0]
                outputs = self._vc.pipeline.pipeline_batch(
                    self._vc.hubert_model,
                    self._vc.net_g,
                    0,
                    audios,
                    [str(path) for path in tmp_paths],
                    times,
                    pitch,
                    f0_method,
                    index_rate,
                    self._vc.if_f0,
                    3,
                    self._vc.tgt_sr,
                    0,
                    rms_mix_rate,
                    self._vc.version,
                    protect,
                )

                encoded = []
                for sr, audio_data in outputs:
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, sr, format="WAV")
                    wav_bytes = buf.getvalue()

                    if output_format == "mp3":
                        from pydub import AudioSegment

                        wav_buf = io.BytesIO(wav_bytes)
                        seg = AudioSegment.from_wav(wav_buf)
                        mp3_buf = io.BytesIO()
                        seg.export(mp3_buf, format="mp3", bitrate="320k", parameters=["-q:a", "0"])
                        encoded.append((mp3_buf.getvalue(), sr))
                    else:
                        encoded.append((wav_bytes, sr))

                return encoded
            finally:
                for tmp_path in tmp_paths:
                    tmp_path.unlink(missing_ok=True)
