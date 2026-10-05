"""Nudity/NSFW image classification for the ``detect-nudity`` task.

Single-label classifier over images (Marqo/nsfw-image-detection-384 —
timm ``vit_tiny_patch16_384`` fine-tuned for NSFW detection, labels
``SFW`` / ``NSFW``, ~5.5M params / 22 MB weights — the only small NSFW
classifier on HF worth using; binary normal/nsfw alternates are all ViT-
base, i.e. 15x heavier). Runs on CPU inside the main worker process — no
backend child subprocess, no CUDA. Weights load lazily on the first
detection so TTS-only workers (and the model-less forwarder) never pay
the load cost.

Model directory resolution: the pre-baked ``data/nsfw-detection`` snapshot
(baked into Docker images via scripts/download_data.py) when present, else
the Hugging Face hub.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision import transforms
from torchvision.transforms import InterpolationMode

_LOGGER = logging.getLogger(__name__)

TIMM_ARCH = "vit_tiny_patch16_384"
HUB_MODEL_ID = "Marqo/nsfw-image-detection-384"
SNAPSHOT_DIR = "marqo-nsfw-image-detection-384"
# repo config.json / pretrained_cfg: fixed 384x384 input, bicubic resize,
# mean/std 0.5 (ViT augreg_in21k_ft_in1k recipe), no further normalization.
INPUT_SIZE = 384


class NudityDetectRequest(BaseModel):
    """Input for the ``detect-nudity`` operation (one image)."""

    image_url: str | None = None
    image_base64: str | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class NudityInputError(ValueError):
    """Invalid image input (undecodable bytes, empty data, neither source set)."""


class NudityDetector:
    """Lazy CPU classifier; returns label scores for one image."""

    def __init__(self, model_root: str | Path | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        self._model_root = (
            Path(model_root)
            if model_root is not None
            else repo_root / "data" / "nsfw-detection"
        )
        self._snapshot_dir = self._model_root / SNAPSHOT_DIR
        self._transform = transforms.Compose(
            [
                transforms.Resize(INPUT_SIZE, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )
        self._model: Any = None
        self._labels: tuple[str, ...] = ()
        self._lock = threading.Lock()

    @property
    def model_source(self) -> str:
        """Pre-baked snapshot when available, else the hub model id."""
        if (self._snapshot_dir / "model.safetensors").exists():
            return str(self._snapshot_dir)
        return HUB_MODEL_ID

    @property
    def model_label(self) -> str:
        """Short model name for result payloads."""
        source = self.model_source
        if source == HUB_MODEL_ID:
            return HUB_MODEL_ID
        return Path(source).name

    def _snapshot_config_labels(self) -> tuple[str, ...] | None:
        """Label names from the baked timm config.json ("label_names" field)."""
        config_path = self._snapshot_dir / "config.json"
        if not config_path.exists():
            return None
        try:
            import json

            with config_path.open(encoding="utf-8") as f:
                names = json.load(f).get("label_names")
            if isinstance(names, list) and len(names) == 2:
                return (str(names[0]), str(names[1]))
        except (OSError, ValueError, KeyError):
            pass
        return None

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                source = self.model_source
                _LOGGER.info("Nudity detector loading model=%s", self.model_label)
                if source == HUB_MODEL_ID:
                    model = timm.create_model(f"hf-hub:{HUB_MODEL_ID}", pretrained=True, num_classes=2)
                else:
                    from safetensors.torch import load_file

                    model = timm.create_model(TIMM_ARCH, pretrained=False, num_classes=2)
                    model.load_state_dict(load_file(str(self._snapshot_dir / "model.safetensors")), strict=True)
                model.eval()
                self._model = model
                labels = self._snapshot_config_labels() or ("NSFW", "SFW")
                self._labels = labels
                _LOGGER.info("Nudity detector loaded model=%s labels=%s", self.model_label, self._labels)
            except Exception as exc:
                _LOGGER.exception("Nudity detector load failed model=%s", self.model_label)
                raise RuntimeError(f"Nudity detector load failed: {exc}") from exc

    def classify(self, image_bytes: bytes, threshold: float = 0.5) -> dict[str, Any]:
        """Score one encoded image; returns label, scores, and nsfw flag."""
        if not image_bytes:
            raise NudityInputError("Nudity detection input is empty")
        self._ensure_loaded()
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise NudityInputError("Unsupported or corrupt image data") from exc

        inputs = self._transform(image).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(inputs)[0]
        probabilities = torch.softmax(logits.float(), dim=0).tolist()

        scored = sorted(
            (
                {"label": label, "score": round(float(score), 6)}
                for label, score in zip(self._labels, probabilities, strict=True)
            ),
            key=lambda entry: entry["score"],
            reverse=True,
        )
        top = scored[0]
        nsfw_score = next(
            (entry["score"] for entry in scored if entry["label"] == "NSFW"),
            0.0,
        )
        return {
            "label": top["label"],
            "score": top["score"],
            "nsfw": nsfw_score >= threshold,
            "nsfwScore": nsfw_score,
            "threshold": threshold,
            "labels": scored,
            "model": self.model_label,
        }