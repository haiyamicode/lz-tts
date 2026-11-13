"""Face detection for the ``detect-face`` task.

OpenCV YuNet (``face_detection_yunet_2023mar.onnx``, MIT, ~230 KB) through
``cv2.FaceDetectorYN``: boxes, a score and five landmarks per face, a few
milliseconds per image on CPU. Runs inside the main worker process, no CUDA;
the model loads lazily on the first detection. Weights come from the
S3-baked ``data/face-detection`` directory.

Images are EXIF-transposed and, when their long side exceeds
``MAX_DETECT_SIDE``, downscaled for detection; boxes and landmarks are
returned in pixels of the transposed full-size image.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_LABEL = "yunet-2023mar"
# YuNet is anchor-free at strides 8/16/32: faces down to ~10 px at the detection
# scale are found, so 1920 keeps ~1% of a 4K frame's width.
MAX_DETECT_SIDE = 1920
NMS_THRESHOLD = 0.3
# YuNet's landmark order.
LANDMARKS = ("rightEye", "leftEye", "nose", "mouthRight", "mouthLeft")


class FaceDetectRequest(BaseModel):
    """Input for the ``detect-face`` operation (one image)."""

    image_url: str | None = None
    image_base64: str | None = None
    min_score: float = Field(default=0.6, ge=0.0, le=1.0)
    max_faces: int = Field(default=100, ge=1, le=1000)


class FaceInputError(ValueError):
    """Invalid image input (undecodable bytes, empty data)."""


class FaceDetector:
    """Lazy CPU YuNet detector; one image per call."""

    def __init__(self, model_root: str | Path | None = None):
        repo_root = Path(__file__).resolve().parents[2]
        root = Path(model_root) if model_root is not None else repo_root / "data" / "face-detection"
        self._model_path = root / MODEL_FILE
        self._detector: Any = None
        # FaceDetectorYN keeps per-size state (setInputSize), so calls are serialized.
        self._lock = threading.Lock()

    def detect(self, image_bytes: bytes, min_score: float = 0.6, max_faces: int = 100) -> dict[str, Any]:
        """Detect faces in one encoded image; faces sorted by score, highest first."""
        if not image_bytes:
            raise FaceInputError("Face detection input is empty")
        try:
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(image_bytes))).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise FaceInputError("Unsupported or corrupt image data") from exc

        width, height = image.size
        scale = min(1.0, MAX_DETECT_SIDE / max(width, height))
        if scale < 1.0:
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))), Image.BILINEAR)
        bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])

        with self._lock:
            if self._detector is None:
                if not self._model_path.exists():
                    raise RuntimeError(f"Face detection model missing: {self._model_path}")
                self._detector = cv2.FaceDetectorYN.create(str(self._model_path), "", (320, 320), 0.0, NMS_THRESHOLD, 5000)
            self._detector.setInputSize((bgr.shape[1], bgr.shape[0]))
            self._detector.setScoreThreshold(min_score)
            _, rows = self._detector.detect(bgr)

        faces = []
        for row in [] if rows is None else rows:
            x, y, w, h = (float(v) / scale for v in row[:4])
            # Clip the box to the image; YuNet can extend boxes past the border.
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            faces.append(
                {
                    "x": round(x0, 1),
                    "y": round(y0, 1),
                    "width": round(x1 - x0, 1),
                    "height": round(y1 - y0, 1),
                    "score": round(float(row[14]), 4),
                    "landmarks": {
                        name: [round(float(row[4 + 2 * i]) / scale, 1), round(float(row[5 + 2 * i]) / scale, 1)]
                        for i, name in enumerate(LANDMARKS)
                    },
                }
            )
        faces.sort(key=lambda face: face["score"], reverse=True)
        return {
            "width": width,
            "height": height,
            "faceCount": len(faces[:max_faces]),
            "faces": faces[:max_faces],
            "minScore": min_score,
            "model": MODEL_LABEL,
        }
