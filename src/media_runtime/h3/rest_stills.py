"""imagegen ``rest_frame``: rest stills of the person in one image, for lipsync frame pins.

Port of vigen's avatar stills (``ai_worker.avatar_jobs``, local/experiments/h3-avatar-frameset-20261116).
Still ``k`` is an H3 edit of the image (``<Picture 1>``) with :data:`REST_PROMPT` and variation
``k % 6``: the same person in the same shot a few seconds later, in a calm pause between sentences:
- hands rest still, and a hand that holds something keeps holding it;
- the mouth is closed and the eyes are on the lens;
- one small head or posture variation per still.

Still ``k`` renders on ``seed + k``, as vigen's ``seed + index``, so a caller fanning out ``k = 0..n-1``
on one seed gets vigen's set. Unlike vigen, every take is checked against the image's face (MediaPipe
FaceLandmarker, then ArcFace on the aligned face) and retaken on ``seed + k + 100 * attempt``, at most
``max_retries`` times, when it:
- lost the person: ArcFace cosine below :data:`MIN_IDENTITY`, or no face at all;
- opened the mouth: inner-lip gap over face height above :data:`MOUTH_CLOSED` (vigen's own run had 2 of 6
  stills open on one avatar, and an open-mouth rest pin holds the mouth open at a pause).
Out of retakes, the best take is delivered all the same (:func:`keep`): a failed likeness check is an expected
outcome, never a failed job; the result's ``likeness`` says which take won and whether it passed. No take is
checked when the image has no face found, or shows a cartoon or other non-realistic character (anime, 2D/3D
animation, drawing, painting, clay, game render: :func:`stylized_score` at least :data:`MAX_STYLIZED`), whose
face ArcFace and FaceLandmarker are not made for.

Pinned as a lipsync's rest frames (``pinned_frames`` ``multi_image``, ``order: random``), the stills
replace one stale image with varied rest poses, lower a hand the image catches mid-gesture, and keep the
face's look (frames taken from a videogen scene carry its tint into the lipsync).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

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
#: ArcFace cosine (still's face vs the image's) below which the still has lost the person. 2026-11-16, 8
#: avatars: decent H3 rest stills / videogen frames 0.73-0.97, but 0.55-0.64 from a 640x360 image with a
#: small face; prompted strangers -0.05 and 0.44; different real people <= 0.30. Look-alike drift (0.58)
#: overlaps the low-res stills, so only the clearly lost face is retaken.
MIN_IDENTITY = 0.45
#: CLIP ViT-L/14 zero-shot prompts: the image is a real photo, or a non-realistic picture.
REAL_PROMPTS = ("a photo", "a photograph of a real person", "a webcam video frame", "a smartphone selfie")
STYLIZED_PROMPTS = (
    "a cartoon", "a 2D cartoon drawing", "an anime drawing", "a 3D animated movie still", "a digital illustration",
    "a painting", "a sketch", "a comic", "a claymation", "a video game render", "a vector art avatar",
)
#: Softmax share of :data:`STYLIZED_PROMPTS` from which the image's character is not realistic and no take is
#: checked. 2026-11-17: 37 photos and real-person stills (B&W, filtered, dim webcam, clown makeup, cosplay, news
#: overlay, 8 avatars + their H3 stills) <= 0.12; 19 non-realistic characters (2D/anime/chibi/Disney-style/comic,
#: Pixar-style/clay/low-poly/bitmoji/VTuber 3D, oil/watercolor/pencil, pixel art) >= 0.52, but a robot 0.04 (it
#: has no face to check). More prompts ("avatar", "doll", "robot") lift real avatars to 0.56-0.83.
MAX_STYLIZED = 0.3
#: Addons (:data:`h3.minimax_h3_runtime.VIDEO_ADDON_LORAS`) a still of a realistic reference renders with: the
#: fal Realism People LoRA (natural skin, not the glossy look; local/exp/h3-realism-lora). A stylized reference
#: gets none: realism would pull a cartoon toward a photo.
REALISTIC_ADDONS = ("realism",)


def still_addons(stylized: float) -> tuple[str, ...]:
    """The addons rest stills of a reference with :func:`stylized_score` ``stylized`` render with."""
    return () if stylized >= MAX_STYLIZED else REALISTIC_ADDONS


#: Retakes after the first take (``max_retries``), by default and at most. Default: two takes in all.
DEFAULT_RETRIES = 1
MAX_RETRIES = 5
#: Seed step between the takes of one still.
ATTEMPT_SEED_STEP = 100

_models: dict[str, object] = {}
_models_lock = threading.Lock()
#: ArcFace's 112x112 alignment template: eye centres, nose tip, mouth corners (image left first).
_ARCFACE_TEMPLATE = (
    (38.2946, 51.6963), (73.5318, 51.5014), (56.0252, 71.7366), (41.5493, 92.3655), (70.7299, 92.2041)
)


def rest_prompt(index: int) -> str:
    return REST_PROMPT.format(variation=REST_VARIATIONS[index % len(REST_VARIATIONS)])


def delivered_size(width: int, height: int) -> tuple[int, int]:
    """The image's size, scaled down to :data:`MAX_LONG_SIDE` on its long side, on even numbers."""
    ratio = min(1.0, MAX_LONG_SIDE / max(width, height))
    return max(2, int(round(width * ratio / 2)) * 2), max(2, int(round(height * ratio / 2)) * 2)


def attempt_seed(seed: int, index: int, attempt: int) -> int:
    return (seed + index + ATTEMPT_SEED_STEP * attempt) % 2**31


@dataclass
class Take:
    seed: int
    path: Path
    #: None: no face found in the still.
    mouth: float | None = None
    identity: float | None = None
    #: What rejects it: "no_face", "identity" (below MIN_IDENTITY), "mouth" (open), or None.
    problem: str | None = None

    def report(self) -> dict:
        return {"seed": self.seed, "mouth": None if self.mouth is None else round(self.mouth, 4),
                "identity": None if self.identity is None else round(self.identity, 3), "problem": self.problem}


def judge(take: Take, face: "Face | None", reference: "Face | None") -> Take:
    """Fill ``take``'s measures from its face against the image's (``reference``; None = no face there,
    nothing to check)."""
    if reference is None:
        return take
    if face is None:
        take.problem = "no_face"
        return take
    take.mouth, take.identity = face.mouth, float(face.identity @ reference.identity)
    take.problem = ("identity" if take.identity < MIN_IDENTITY else "mouth" if take.mouth > MOUTH_CLOSED else None)
    return take


def keep(takes: list[Take]) -> Take:
    """The take to deliver, ranked by the gate's own measures: the first without a problem, else the least open
    mouth among those that kept the person, else the most similar face (a take with no face last; ties go to
    the earlier take)."""
    clean = [take for take in takes if take.problem is None]
    if clean:
        return clean[0]
    kept_person = [take for take in takes if take.problem == "mouth"]
    if kept_person:
        return min(kept_person, key=lambda take: take.mouth)
    return max(takes, key=lambda take: -2.0 if take.identity is None else take.identity)


@dataclass
class Still:
    """Rest still ``index``: every take rendered and the one to deliver (:func:`keep`)."""

    index: int
    takes: list[Take]
    kept: Take
    render_size: tuple[int, int]
    #: The last take's render metrics.
    metrics: dict
    #: Whether the takes were held to the image's face (False: no face there, or a non-realistic character).
    checked: bool

    def likeness(self) -> dict:
        """Which take won and how it scored: ``score`` is its ArcFace cosine to the image's face, ``passed``
        whether it cleared every check (None: nothing was checked), ``problem`` what it failed."""
        return {"checked": self.checked, "takes": len(self.takes), "chosen": self.takes.index(self.kept),
                "score": self.kept.report()["identity"], "passed": (self.kept.problem is None) if self.checked else None,
                "problem": self.kept.problem}

    def report(self) -> dict:
        kept = self.kept.report()
        return {"index": self.index, "variation": REST_VARIATIONS[self.index % len(REST_VARIATIONS)],
                "seed": kept["seed"], "identity": kept["identity"], "mouth": kept["mouth"],
                "likeness": self.likeness(), "takes": [take.report() for take in self.takes]}


def render_still(*, job_id: str, image: Path, index: int, seed: int, retries: int, out_dir: Path,
                 reference: "Face | None", preset: str | None = None, addons: tuple[str, ...] = ()) -> Still:
    """Rest still ``index`` of the person in ``image`` (an upright RGB PNG) on H3's native canvas with the
    image's shape (+ ``addons``, :func:`still_addons`): take ``attempt`` on :func:`attempt_seed`, judged against
    ``reference`` (the image's face, :func:`measure_face`), retaken until one has no problem or ``retries``
    retakes are spent."""
    from PIL import Image  # type: ignore

    from . import h3_still
    from .compat import log_event
    from .minimax_h3_runtime import run_minimax_h3_still

    with Image.open(image) as opened:
        render_size = h3_still.fit_canvas(*opened.size)
    out_dir.mkdir(parents=True, exist_ok=True)
    takes: list[Take] = []
    metrics: dict = {}
    for attempt in range(1 + retries):
        take_seed = attempt_seed(seed, index, attempt)
        path = out_dir / f"still_{index}_take_{attempt}.png"
        metrics = run_minimax_h3_still(job_id=job_id, prompt=h3_still.picture_tags(rest_prompt(index)),
                                       reference_paths=[image], output_path=path, width=render_size[0],
                                       height=render_size[1], seed=take_seed,
                                       preset=preset or h3_still.DEFAULT_PRESET, addons=addons)
        take = judge(Take(seed=take_seed, path=path), measure_face(path) if reference else None, reference)
        takes.append(take)
        log_event("rest_still_take", job_id=job_id, index=index, attempt=attempt, **take.report())
        if take.problem is None:
            break
    return Still(index=index, takes=takes, kept=keep(takes), render_size=render_size, metrics=metrics,
                 checked=reference is not None)


def save_fitted(source: Path, target: Path, size: tuple[int, int]) -> Path:
    """``source`` cover-fitted (Lanczos) to ``size`` as an RGB image at ``target``."""
    from PIL import Image, ImageOps  # type: ignore

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as rendered:
        image = rendered.convert("RGB")
        if image.size != size:
            image = ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)
        image.save(target)
    return target


@dataclass(frozen=True)
class Face:
    #: Inner-lip gap (landmarks 13-14) over face height (10-152).
    mouth: float
    #: Unit ArcFace embedding of the aligned face.
    identity: "np.ndarray"


def _model(name: str):
    from .compat import FACE_IDENTITY_PATH, FACE_LANDMARKER_PATH, REALISM_CLIP_DIR

    with _models_lock:
        if name not in _models:
            if name == "realism":
                import torch
                from transformers import CLIPModel, CLIPProcessor  # type: ignore

                model = CLIPModel.from_pretrained(REALISM_CLIP_DIR, dtype=torch.float32).eval()
                processor = CLIPProcessor.from_pretrained(REALISM_CLIP_DIR)
                with torch.no_grad():
                    text = model.get_text_features(**processor(text=list(REAL_PROMPTS + STYLIZED_PROMPTS),
                                                               return_tensors="pt", padding=True))
                text = getattr(text, "pooler_output", text)
                _models[name] = (model, processor, text / text.norm(dim=-1, keepdim=True))
            elif name == "landmarker":
                from mediapipe.tasks.python import BaseOptions, vision  # type: ignore

                _models[name] = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(model_asset_path=str(FACE_LANDMARKER_PATH)), num_faces=1))
            else:
                import onnxruntime  # type: ignore

                _models[name] = onnxruntime.InferenceSession(str(FACE_IDENTITY_PATH),
                                                             providers=["CPUExecutionProvider"])
        return _models[name]


def stylized_score(path: Path) -> float:
    """How surely the image at ``path`` is not a real photo (cartoon, anime, 3D animation, drawing, painting):
    the share of :data:`STYLIZED_PROMPTS` in CLIP's softmax over both prompt sets (CPU, ~1.2 s)."""
    import torch
    from PIL import Image  # type: ignore

    model, processor, text = _model("realism")
    with Image.open(path) as opened, _models_lock, torch.no_grad():
        image = model.get_image_features(**processor(images=opened.convert("RGB"), return_tensors="pt"))
        image = getattr(image, "pooler_output", image)
        image = image / image.norm(dim=-1, keepdim=True)
        shares = (model.logit_scale.exp() * image @ text.T).softmax(-1)[0]
    return float(shares[len(REAL_PROMPTS):].sum())


def reference_face(path: Path) -> tuple[Face | None, float]:
    """The face takes of the image at ``path`` are held to, and its :func:`stylized_score`; the face is None
    (no take checked) for a non-realistic character or an image with no face found."""
    stylized = stylized_score(path)
    return (None if stylized >= MAX_STYLIZED else measure_face(path)), stylized


def measure_face(path: Path) -> Face | None:
    """The mouth opening and identity of the (largest) face in the image at ``path``; ``None`` without one."""
    import cv2  # type: ignore
    import mediapipe as mp  # type: ignore
    import numpy as np
    from PIL import Image  # type: ignore

    with Image.open(path) as opened:
        rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
    landmarker = _model("landmarker")
    with _models_lock:
        found = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).face_landmarks
    if not found:
        return None
    height, width = rgb.shape[:2]
    points = np.array([(p.x * width, p.y * height) for p in found[0]], dtype=np.float32)
    face = float(np.linalg.norm(points[10] - points[152]))
    if face <= 0:
        return None
    # Eye centres from the corners (steady under gaze changes), the nose tip, the mouth corners.
    five = np.stack([(points[33] + points[133]) / 2, (points[362] + points[263]) / 2, points[1], points[61],
                     points[291]])
    matrix, _ = cv2.estimateAffinePartial2D(five, np.array(_ARCFACE_TEMPLATE, dtype=np.float32), method=cv2.LMEDS)
    if matrix is None:
        return None
    crop = cv2.warpAffine(rgb, matrix, (112, 112), borderValue=0.0)
    blob = ((crop.astype(np.float32) - 127.5) / 127.5).transpose(2, 0, 1)[None]
    session = _model("identity")
    embedding = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
    return Face(mouth=float(np.linalg.norm(points[13] - points[14])) / face,
                identity=embedding / np.linalg.norm(embedding))
