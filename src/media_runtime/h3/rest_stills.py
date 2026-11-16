"""imagegen ``rest_frame``: rest stills of the person in one image, for lipsync frame pins.

Port of vigen's avatar stills (``ai_worker.avatar_jobs``, local/experiments/h3-avatar-frameset-20261116).
Still ``k`` is an H3 edit of the image (``<Picture 1>``) with :data:`REST_PROMPT` and variation
``k % 6``: the same person in the same shot a few seconds later, in a calm pause between sentences:
- hands rest still, and a hand that holds something keeps holding it;
- the mouth is closed and the eyes are on the lens;
- one small head or posture variation per still.

Still ``k`` renders on ``seed + k``, as vigen's ``seed + index``, so a caller fanning out ``k = 0..n-1``
on one seed gets vigen's set. Unlike vigen, a still whose mouth is open (inner-lip gap over face
height > :data:`MOUTH_CLOSED`, MediaPipe FaceLandmarker) is rendered again on ``seed + k + 100 * attempt``,
up to :data:`MAX_ATTEMPTS` takes, keeping the most closed: vigen's own run had 2 of 6 stills open on one
avatar, and an open-mouth rest pin holds the mouth open at a pause. A still with no face found is kept.

Pinned as a lipsync's rest frames (``pinned_frames`` ``multi_image``, ``order: random``), the stills
replace one stale image with varied rest poses, lower a hand the image catches mid-gesture, and keep the
face's look (frames taken from a videogen scene carry its tint into the lipsync).
"""

from __future__ import annotations

import threading
from pathlib import Path

REST_PROMPT = (
    "Image 1 is a frame from a video of a person talking to the camera. Show the same video a few seconds later, "
    "in a short pause between two sentences: the same person, face, hairstyle, makeup, clothing and accessories, in "
    "the same place, seen from exactly the same camera position with the same framing, lens, lighting and background "
    "as Image 1. The person rests in a calm, natural pose with the shoulders relaxed. A hand that holds something in "
    "Image 1 keeps holding it in the same place; every free hand rests naturally in front of the body - in the lap, "
    "on the table or clasped with the other hand - still, not raised and not gesturing. The mouth is closed with "
    "relaxed lips, the expression calm and neutral, the eyes open and looking into the lens. {variation}"
)
#: One small variation per still, in this order and cycled past the sixth.
REST_VARIATIONS = (
    "The head is level and faces the camera straight on.",
    "The head is turned very slightly to one side, the eyes still on the lens.",
    "The head is tilted very slightly, with a soft, attentive look.",
    "The chin is a touch lower and the shoulders a little more relaxed.",
    "The head is turned very slightly to the other side, the eyes still on the lens.",
    "The posture is a little more upright, the head level.",
)
#: Longest side a still is delivered at by default; a larger image's stills are scaled down to it.
MAX_LONG_SIDE = 2048
#: Inner-lip gap (landmarks 13-14) over face height (10-152) above which a still's mouth counts as open.
MOUTH_CLOSED = 0.012
MAX_ATTEMPTS = 3
#: Seed step between the takes of one still.
ATTEMPT_SEED_STEP = 100

_landmarker = None
_landmarker_lock = threading.Lock()


def rest_prompt(index: int) -> str:
    return REST_PROMPT.format(variation=REST_VARIATIONS[index % len(REST_VARIATIONS)])


def delivered_size(width: int, height: int) -> tuple[int, int]:
    """The image's size, scaled down to :data:`MAX_LONG_SIDE` on its long side, on even numbers."""
    ratio = min(1.0, MAX_LONG_SIDE / max(width, height))
    return max(2, int(round(width * ratio / 2)) * 2), max(2, int(round(height * ratio / 2)) * 2)


def attempt_seed(seed: int, index: int, attempt: int) -> int:
    return (seed + index + ATTEMPT_SEED_STEP * attempt) % 2**31


def mouth_opening(path: Path) -> float | None:
    """Inner-lip gap over face height in the image at ``path``; ``None`` when no face is found."""
    import mediapipe as mp  # type: ignore
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision  # type: ignore
    from PIL import Image  # type: ignore

    from .compat import FACE_LANDMARKER_PATH

    global _landmarker
    with _landmarker_lock:
        if _landmarker is None:
            _landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)), num_faces=1))
        with Image.open(path) as opened:
            rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
        found = _landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
    if not found:
        return None
    height, width = rgb.shape[:2]
    points = np.array([(found[0][i].x * width, found[0][i].y * height) for i in (10, 152, 13, 14)])
    face = float(np.linalg.norm(points[0] - points[1]))
    return float(np.linalg.norm(points[2] - points[3])) / face if face > 0 else None
