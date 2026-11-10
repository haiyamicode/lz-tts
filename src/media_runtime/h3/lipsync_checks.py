"""Per-chunk validators for H3 freestyle lipsync: ``detect_flicker``, ``detect_desync`` and ``detect_glitches``.

Flicker and desync judge one rendered chunk against its own first frame, which
is the job's input image pinned (decoded at the render's size): the camera is
locked and every pin returns the person to that pose, so that frame is the
ground truth for lighting, framing and where the face is. Glitches are judged
against the input image itself. A chunk that moves into a pinned chunk's own
image (``pin_image_url``) or out of it is asked to drift from its first frame,
so its flicker check judges only sudden steps, and its glitches are judged
against the image it opens on.

**Flicker** (no learning): per frame at a ~384 px grid, CIELAB (L in 0-100),

- *step*: the median over pixels of L(t) - L(t-1). A light switching off or on,
  a punch-in or a cut moves every pixel at once; talking and gestures move a
  few, so the median stays near 0 (0.05 typical);
- *light* / *colour*: the offset from the first frame of the patches whose
  structure still matches it (ZNCC >= ``ZNCC_MIN``, the background the person
  is not covering), as a half-second rolling median;
- *zoom*: the camera's scale against the first frame (``_framing``), counted
  only for a zoom about the middle of the frame. A push-in that builds over a
  second moves each pixel a little per frame, so the step sees only its snap
  back to the pin, if that is sudden enough: the loliteo gate's chunk 7 pushed
  in 3-4% and passed. A sideways shift (*pan*, reported, never flagged) cannot
  be told from the person moving: in close-ups on plain dark walls the
  trackable detail is the person, and leaning or raising an arm fits as a 2-4%
  pan (prod jobs PxsEVg9ocmmw9Rj9tixYEo, W3fYQsvnbetDWHfCem6zZP).

Calibrated on 43 renders (gates, references, prod tests): every one over a
threshold was a real lighting flicker, push-in or flash cut, and the cleanest
flagged / dirtiest clean renders sit at step 0.49 / 0.36. Zooms and cuts count
as flicker on purpose: both are rejected. The zoom read was calibrated on 1,194
chunk renders and prod outputs: it adds 12 flags, 11 of them real push-ins of
3-22% and one person leaning into a microphone (1.032); no prod output reads
over 1.02.

**Desync**: SyncNet v2 (joonson/syncnet_python's scoring) on a fixed face crop.
S3FD finds the face once on the first frame; the canonical crop around it
(``crop_scale`` 0.4, grey padding, 224x224, 25 fps) is cut by ffmpeg for the
whole chunk instead of tracking every frame - cheap, and valid because the
camera does not move. ``conf`` is SyncNet's confidence (median minus minimum
audio-video distance over +-15 frame shifts) and ``offset`` the best shift in
25 fps frames. Both are taken per window (``SYNC_WINDOW_SECONDS`` every
``SYNC_WINDOW_STEP_SECONDS``) over the windows that carry voice, and the worst
window judges the chunk: one score over a whole chunk lets a few seconds in sync
hide a second or two of lips frozen under speech (prod job
cX9t2mQr5penQ2DHMXSV9D: chunk conf 5.95, the 2 s window over its frozen
second 2.76; iJ32XG4BNSZRS7NrBZLjpB: 3.20 and 0.68). Calibrated on 361 chunks of
earlier renders and 33 of recent prod jobs: 2 s windows add no flag but those
two frozen stretches (worst window of a clean chunk 3.31); 1 s windows flag 96.

**Glitches**: a ResNet18 (``glitch_detector`` artifact, trained in
local/experiments/overlay-cnn-20261101: run14 step 5000, EMA weights with
BatchNorm statistics re-estimated) gives every kept frame the probability that
it shows something the input image does not have: an invented graphic, text,
sparkles or a transition, or a botched hand or limb. It sees the frames
``GLITCH_PRIOR`` before the judged one, the judged frame and the input image,
each fitted to ``GLITCH_LONG_SIDE`` and letterboxed square; a frame over
``GLITCH_THRESHOLD`` fails the chunk. Trained on clean TalkVid clips and
synthetic overlays / hand fumbles only. On 379 held-out clean TalkVid clips it
flags 0.26% of frames (8 clips: hands mid-gesture, objects held up) and catches
98% of held-out synthetic edits; on prod renders it flags the sparkle job
(4iZAcQva) and the ff3sla3 g90 hand fumble, and no frame of the gesture-heavy
Wdj24NArCZQ34vub6d388Z. See-through or fading limbs are not trained (their
weak cases look like a hand that just moved), so it does not catch them.

The network is built on the CPU once and copied to the GPU only for a check,
``GLITCH_BATCH`` frames per pass (about 0.25 GB at 640 px), then dropped and
the cache emptied, so the H3 render that follows gets the memory back.
"""

from __future__ import annotations

import math
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .compat import log_event

FPS = 24.0
CHECK_LONG_SIDE = 384
PATCH = 16
ZNCC_MIN = 0.8
#: Median per-pixel L* jump between consecutive frames that is a flicker, cut or zoom snap.
FLICKER_STEP = 0.45
#: Half-second median L* offset from the first frame that is a lighting change.
FLICKER_LIGHT = 1.0
#: Half-second median a*b* distance from the first frame that is a colour change.
FLICKER_COLOUR = 0.6
FLICKER_WINDOW = 12
#: Camera scale against the first frame that is a zoom (either way), counted only for a zoom about the middle of
#: the frame: its frame-centre shift (fraction of the width) at most ``FRAMING_ZOOM_CENTRE`` times its scale change.
#: A person leaning into the camera fits as a smaller scale about a point off to the side.
FRAMING_ZOOM = 1.03
FRAMING_ZOOM_CENTRE = 0.5
#: Framing is read on every this many frames, from up to this many corners of the first frame, and only where
#: at least this many of them fit one camera move.
FRAMING_EVERY = 3
FRAMING_CORNERS = 400
FRAMING_MIN_INLIERS = 30
#: SyncNet confidence under which the lips do not follow the voice.
SYNC_MIN_CONF = 3.0
#: Best audio-video shift (25 fps frames) accepted as in sync.
SYNC_OFFSETS = (-3, 3)
#: Shortest chunk SyncNet scores: 5-frame windows over at least this much speech.
SYNC_MIN_SECONDS = 1.5
SYNC_CROP_SCALE = 0.4
SYNC_FPS = 25
#: Audio shifts (25 fps frames, each way) SyncNet searches for the best match.
SYNC_VSHIFT = 15
#: SyncNet window length and spacing; the chunk's worst voiced window judges it.
SYNC_WINDOW_SECONDS = 2.0
SYNC_WINDOW_STEP_SECONDS = 0.5
#: A window is scored when this share of its frames carry voice: level within ``SYNC_VOICE_DB``
#: of the chunk's loud frames. In a pause SyncNet has nothing to match the lips against.
SYNC_VOICE_SHARE = 0.5
SYNC_VOICE_DB = 30.0
GLITCH_LONG_SIDE = 640
#: Frames before the judged one the detector also sees (167 and 83 ms at 24 fps).
GLITCH_PRIOR = (4, 2)
GLITCH_THRESHOLD = 0.5
#: Frames per forward pass: ~60 MB of activations each at 640 px.
GLITCH_BATCH = 4
#: ImageNet statistics, BGR, in 0-255.
GLITCH_MEAN = np.float32([0.406, 0.456, 0.485]) * 255
GLITCH_STD = np.float32([0.225, 0.224, 0.229]) * 255

_MODELS: dict[str, Any] = {}
_MODELS_LOCK = threading.Lock()
_GLITCH_NET: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


def _probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0:s=x", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    width, height = out.split("x")[:2]
    return int(width), int(height)


def _check_size(width: int, height: int) -> tuple[int, int]:
    """The analysis grid: long side ``CHECK_LONG_SIDE``, both sides multiples of ``PATCH``."""
    scale = CHECK_LONG_SIDE / max(width, height)
    snap = lambda value: max(PATCH, int(round(value * scale / PATCH)) * PATCH)  # noqa: E731
    return snap(width), snap(height)


def _decode(path: Path, *, frames: int, vf: str, width: int, height: int, pix_fmt: str = "rgb24") -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vf", vf, "-frames:v", str(frames),
         "-f", "rawvideo", "-pix_fmt", pix_fmt, "-"],
        check=True, capture_output=True,
    ).stdout
    count = len(raw) // (width * height * 3)
    return np.frombuffer(raw[: count * width * height * 3], np.uint8).reshape(count, height, width, 3)


def _lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8 -> CIELAB (D65), L in 0-100: OpenCV's float ``COLOR_RGB2Lab``."""
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = c @ np.array([[0.412453, 0.357580, 0.180423],
                        [0.212671, 0.715160, 0.072169],
                        [0.019334, 0.119193, 0.950227]], np.float32).T
    xyz /= np.array([0.950456, 1.0, 1.088754], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    lightness = np.where(xyz[..., 1] > 0.008856, 116.0 * f[..., 1] - 16.0, 903.3 * xyz[..., 1])
    return np.stack([lightness, 500.0 * (f[..., 0] - f[..., 1]), 200.0 * (f[..., 1] - f[..., 2])], axis=-1)


# ---------------------------------------------------------------------------
# flicker
# ---------------------------------------------------------------------------


def _patches(x: np.ndarray) -> np.ndarray:  # (H, W, C) -> (patches, P*P, C)
    h, w, c = x.shape
    return x.reshape(h // PATCH, PATCH, w // PATCH, PATCH, c).transpose(0, 2, 1, 3, 4).reshape(-1, PATCH * PATCH, c)


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.array([np.median(values)]) if len(values) else np.zeros(1)
    return np.array([np.median(values[i:i + window]) for i in range(len(values) - window + 1)])


def _framing(lightness: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Camera scale and frame-centre shift (fraction of the width) of every ``FRAMING_EVERY``-th frame against the
    first: the first frame's corners tracked into the frame (pyramidal Lucas-Kanade) and one RANSAC similarity fit,
    which the static background dominates while the person's own motion falls out as outliers. Frames where too few
    corners fit (a cut; the step sees it) read as unmoved."""
    import cv2

    frames = np.clip(lightness * 2.55, 0, 255).astype(np.uint8)
    height, width = frames.shape[1:]
    scale = np.ones(len(frames), np.float32)
    shift = np.zeros(len(frames), np.float32)
    corners = cv2.goodFeaturesToTrack(frames[0], FRAMING_CORNERS, 0.01, 6)
    if corners is None or len(corners) < FRAMING_MIN_INLIERS:
        return scale, shift
    centre = np.array([width / 2, height / 2, 1.0], np.float32)
    for t in range(FRAMING_EVERY, len(frames), FRAMING_EVERY):
        moved, status, _ = cv2.calcOpticalFlowPyrLK(frames[0], frames[t], corners, None, winSize=(21, 21), maxLevel=3)
        tracked = status.ravel() == 1
        if tracked.sum() < FRAMING_MIN_INLIERS:
            continue
        fit, inliers = cv2.estimateAffinePartial2D(corners[tracked], moved[tracked], method=cv2.RANSAC,
                                                   ransacReprojThreshold=1.5)
        if fit is None or inliers.sum() < FRAMING_MIN_INLIERS:
            continue
        scale[t] = math.hypot(fit[0, 0], fit[1, 0])
        shift[t] = float(np.hypot(*(fit @ centre - centre[:2]))) / width
    return scale, shift


def flicker_scores(render: Path, *, kept_frames: int, judge_drift: bool = True) -> dict[str, Any]:
    """Lighting / framing jumps in a chunk's kept frames, against its first frame. Without
    ``judge_drift`` (a chunk pinned to a different image at its end) the light, colour and zoom
    offsets from the first frame are the change it was asked for: they are reported, and only a
    sudden step fails the chunk."""
    width, height = _check_size(*_probe_size(render))
    frames = _decode(render, frames=kept_frames, vf=f"scale={width}:{height}:flags=area", width=width, height=height)
    lab = _lab(frames)
    ref = _patches(lab[0])
    ref_l = ref[..., 0] - ref[..., 0].mean(1, keepdims=True)
    ref_n = np.sqrt((ref_l ** 2).sum(1)) + 1e-6
    ref_mean = ref.mean(1)
    steps = np.zeros(len(lab), np.float32)
    offsets = np.zeros((len(lab), 3), np.float32)
    for t in range(1, len(lab)):
        steps[t] = np.median(lab[t, ..., 0] - lab[t - 1, ..., 0])
        cur = _patches(lab[t])
        cur_l = cur[..., 0] - cur[..., 0].mean(1, keepdims=True)
        keep = (cur_l * ref_l).sum(1) / (np.sqrt((cur_l ** 2).sum(1)) + 1e-6) / ref_n >= ZNCC_MIN
        if keep.sum() >= 10:  # else framing changed too much to read lighting; the step sees it
            offsets[t] = np.median(cur.mean(1)[keep] - ref_mean[keep], axis=0)
    light = np.abs(_rolling_median(offsets[:, 0], FLICKER_WINDOW))
    colour = _rolling_median(np.hypot(offsets[:, 1], offsets[:, 2]), FLICKER_WINDOW)
    step = np.abs(steps)
    scale, shift = _framing(lab[..., 0])
    zoom = np.maximum(scale, 1.0 / scale)
    zoom = np.where(shift <= FRAMING_ZOOM_CENTRE * (zoom - 1.0), zoom, 1.0)
    scores = {
        "frames": int(len(lab)),
        "max_step": round(float(step.max()), 3),
        "step_frame": int(step.argmax()),
        "light": round(float(light.max()), 3),
        "light_frame": int(light.argmax()),
        "colour": round(float(colour.max()), 3),
        "colour_frame": int(colour.argmax()),
        "zoom": round(float(zoom.max()), 4),
        "zoom_frame": int(zoom.argmax()),
        "pan": round(float(shift.max()), 4),
        "pan_frame": int(shift.argmax()),
    }
    ratios = {"step": scores["max_step"] / FLICKER_STEP, "light": scores["light"] / FLICKER_LIGHT,
              "colour": scores["colour"] / FLICKER_COLOUR, "zoom": (scores["zoom"] - 1.0) / (FRAMING_ZOOM - 1.0)}
    if not judge_drift:
        ratios = {"step": ratios["step"]}
        scores["drift_judged"] = False
    scores["badness"] = round(max(ratios.values()), 3)
    scores["reasons"] = [name for name, ratio in ratios.items() if ratio >= 1.0]
    scores["flagged"] = bool(scores["reasons"])
    return scores


# ---------------------------------------------------------------------------
# desync
# ---------------------------------------------------------------------------


def _sync_models() -> dict[str, Any]:
    """S3FD and SyncNet v2 from the ``syncnet_eval`` artifact, loaded once per process."""
    with _MODELS_LOCK:
        if _MODELS:
            return _MODELS
        import torch

        from .compat import SYNCNET_EVAL_S3FD_PATH, SYNCNET_EVAL_SYNCNET_PATH
        from .compat import project_path
        from .syncnet.s3fd_nets import S3FDNet
        from .syncnet.syncnet import S

        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector = S3FDNet(device=device).to(device)
        detector.load_state_dict(torch.load(project_path(SYNCNET_EVAL_S3FD_PATH), map_location=device, weights_only=True))
        syncnet = S(num_layers_in_fc_layers=1024).to(device)
        state = syncnet.state_dict()
        for name, param in torch.load(project_path(SYNCNET_EVAL_SYNCNET_PATH), map_location=device, weights_only=True).items():
            state[name].copy_(param)
        _MODELS.update(device=device, detector=detector.eval(), syncnet=syncnet.eval())
        return _MODELS


def _face_box(frame_rgb: np.ndarray, models: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """The largest S3FD face (x1, y1, x2, y2) in ``frame_rgb``, at syncnet_python's 0.25 detection scale."""
    import torch
    import torch.nn.functional as F

    from .syncnet.s3fd_box_utils import nms_

    height, width = frame_rgb.shape[:2]
    image = torch.from_numpy(frame_rgb.astype(np.float32)).permute(2, 0, 1)[None]
    image = F.interpolate(image, scale_factor=0.25, mode="bilinear", align_corners=False)
    image = image - torch.tensor([123.0, 117.0, 104.0]).view(1, 3, 1, 1)  # RGB means, as S3FD.detect_faces
    with torch.no_grad():
        detections = models["detector"](image.to(models["device"])).cpu()
    boxes = [
        (*(detections[0, i, j, 1:] * torch.tensor([width, height, width, height])).tolist(), float(detections[0, i, j, 0]))
        for i in range(detections.size(1))
        for j in range(detections.size(2))
        if detections[0, i, j, 0] > 0.9
    ]
    if not boxes:
        return None
    kept = np.array(boxes)[nms_(np.array(boxes), 0.1)]
    x1, y1, x2, y2, _ = max(kept, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    return float(x1), float(y1), float(x2), float(y2)


def _syncnet_distances(crops_bgr: np.ndarray, audio_16k: np.ndarray, models: dict[str, Any]) -> np.ndarray:
    """syncnet_python's ``evaluate`` distances: (2 * SYNC_VSHIFT + 1 audio shifts, video windows at 25 fps)."""
    import python_speech_features
    import torch

    video = torch.from_numpy(crops_bgr.astype(np.float32)).permute(3, 0, 1, 2)[None]  # (1, 3, T, 224, 224)
    mfcc = np.stack([np.array(c) for c in zip(*python_speech_features.mfcc(audio_16k, 16000))])
    audio = torch.from_numpy(mfcc.astype(np.float32))[None, None]  # (1, 1, 13, T*4)
    last = min(len(crops_bgr), math.floor(len(audio_16k) / 640)) - 5
    net, device = models["syncnet"], models["device"]
    video_feats, audio_feats = [], []
    with torch.no_grad():
        for start in range(0, last, 20):
            frames = range(start, min(last, start + 20))
            video_feats.append(net.forward_lip(torch.cat([video[:, :, f:f + 5] for f in frames]).to(device)).cpu())
            audio_feats.append(net.forward_aud(torch.cat([audio[:, :, :, f * 4:f * 4 + 20] for f in frames]).to(device)).cpu())
    video_feat, audio_feat = torch.cat(video_feats), torch.cat(audio_feats)
    vshift = SYNC_VSHIFT
    padded = torch.nn.functional.pad(audio_feat, (0, 0, vshift, vshift))
    return torch.stack([
        torch.nn.functional.pairwise_distance(video_feat[[i]].repeat(2 * vshift + 1, 1), padded[i:i + 2 * vshift + 1])
        for i in range(len(video_feat))
    ], 1).numpy()


def _sync_verdict(dists: np.ndarray) -> dict[str, Any]:
    """Best audio shift and SyncNet's confidence (median minus minimum mean distance) over ``dists``' windows."""
    mean_dists = dists.mean(1)
    best = int(mean_dists.argmin())
    offset, conf = SYNC_VSHIFT - best, float(np.median(mean_dists) - mean_dists[best])
    in_range = SYNC_OFFSETS[0] <= offset <= SYNC_OFFSETS[1]
    reasons = (["low_confidence"] if conf < SYNC_MIN_CONF else []) + ([] if in_range else ["offset"])
    return {"offset": offset, "conf": round(conf, 3), "reasons": reasons, "flagged": bool(reasons),
            "badness": round(max(SYNC_MIN_CONF / max(conf, 0.1), 1.0 if in_range else 2.0), 3)}


def _voiced(audio_16k: np.ndarray, windows: int) -> np.ndarray:
    """Whether each SyncNet video window (5 frames at 25 fps) has voice under it."""
    hop = 16000 // SYNC_FPS
    power = np.array([np.mean(audio_16k[i * hop:(i + 5) * hop].astype(np.float64) ** 2) for i in range(windows)])
    level = 10 * np.log10(power + 1e-9)
    return level > np.percentile(level, 95) - SYNC_VOICE_DB


def desync_scores(render: Path, *, audio: Path, kept_frames: int) -> dict[str, Any]:
    """SyncNet offset and confidence of a chunk's kept frames against its pinned voice."""
    seconds = kept_frames / FPS
    if seconds < SYNC_MIN_SECONDS:
        return {"skipped": f"chunk shorter than {SYNC_MIN_SECONDS} s", "flagged": False, "badness": 0.0}
    models = _sync_models()
    width, height = _probe_size(render)
    first = _decode(render, frames=1, vf="null", width=width, height=height)[0]
    box = _face_box(first, models)
    if box is None:
        return {"skipped": "no face in the pinned frame", "flagged": False, "badness": 0.0}
    x1, y1, x2, y2 = box
    half = max(x2 - x1, y2 - y1) / 2
    pad = int(half * (1 + 2 * SYNC_CROP_SCALE))
    cx, cy = (x1 + x2) / 2 + pad, (y1 + y2) / 2 + pad
    crop_w, crop_h = int(2 * half * (1 + SYNC_CROP_SCALE)), int(half * (2 + 2 * SYNC_CROP_SCALE))
    vf = (f"trim=end_frame={kept_frames},setpts=PTS-STARTPTS,fps={SYNC_FPS},"
          f"pad=iw+{2 * pad}:ih+{2 * pad}:{pad}:{pad}:color=0x6e6e6e,"
          f"crop={crop_w}:{crop_h}:{int(cx - half * (1 + SYNC_CROP_SCALE))}:{int(cy - half)},scale=224:224")
    crops = _decode(render, frames=int(seconds * SYNC_FPS) + 1, vf=vf, width=224, height=224, pix_fmt="bgr24")
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(audio), "-t", f"{seconds:.4f}", "-ac", "1", "-ar", "16000",
         "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    pcm = np.frombuffer(pcm, np.int16)
    dists = _syncnet_distances(crops, pcm, models)
    voiced = _voiced(pcm, dists.shape[1])
    span, step = round(SYNC_WINDOW_SECONDS * SYNC_FPS), round(SYNC_WINDOW_STEP_SECONDS * SYNC_FPS)
    starts = list(range(0, max(dists.shape[1] - span, 0) + 1, step))
    if starts[-1] + span < dists.shape[1]:
        starts.append(dists.shape[1] - span)
    windows = [
        {**_sync_verdict(dists[:, start:start + span]), "window": [round(start / SYNC_FPS, 2), round(min(start + span, dists.shape[1]) / SYNC_FPS, 2)]}
        for start in starts
        if voiced[start:start + span].mean() >= SYNC_VOICE_SHARE
    ]
    if not windows:
        return {"skipped": "no voiced window", "flagged": False, "badness": 0.0}
    worst = max(windows, key=lambda window: (window["badness"], -window["conf"]))
    failed = [window for window in windows if window["flagged"]]
    whole = _sync_verdict(dists)
    return {**worst, "reasons": sorted({reason for window in failed for reason in window["reasons"]}),
            "face": [round(v, 1) for v in box], "windows": len(windows), "failed_windows": len(failed),
            "chunk_offset": whole["offset"], "chunk_conf": whole["conf"]}


# ---------------------------------------------------------------------------
# glitches
# ---------------------------------------------------------------------------


def _glitch_net() -> Any:
    """The glitch detector on the CPU, built once per process; each check copies it to the GPU."""
    with _MODELS_LOCK:
        if not _GLITCH_NET:
            import torch
            import torchvision

            from .compat import GLITCH_DETECTOR_PATH
            from .compat import project_path

            state = torch.load(project_path(GLITCH_DETECTOR_PATH), map_location="cpu", weights_only=True)
            channels = 3 * (len(GLITCH_PRIOR) + 2)
            if state["conv1.weight"].shape[1] != channels:
                raise ValueError(f"glitch detector takes {state['conv1.weight'].shape[1]} channels, expected {channels}")
            net = torchvision.models.resnet18(weights=None)
            net.conv1 = torch.nn.Conv2d(channels, 64, 7, 2, 3, bias=False)
            net.fc = torch.nn.Linear(net.fc.in_features, 1)
            net.load_state_dict(state)
            _GLITCH_NET["net"] = net.eval()
        return _GLITCH_NET["net"]


def _fit_size(width: int, height: int) -> tuple[int, int]:
    scale = GLITCH_LONG_SIDE / max(width, height)
    return round(width * scale), round(height * scale)


def glitch_scores(render: Path, *, image: Path, kept_frames: int) -> dict[str, Any]:
    """The glitch detector's probability for each of a chunk's kept frames, against the job's input image."""
    import copy

    import torch

    width, height = _fit_size(*_probe_size(image))
    vf = f"scale={width}:{height}:flags=area"
    ref = _decode(image, frames=1, vf=vf, width=width, height=height, pix_fmt="bgr24")[0]
    frames = _decode(render, frames=kept_frames, vf=vf, width=width, height=height, pix_fmt="bgr24")
    if not len(frames):
        return {"skipped": "no frames decoded", "flagged": False, "badness": 0.0}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    side = GLITCH_LONG_SIDE
    top, left = (side - height) // 2, (side - width) // 2
    mean = torch.tensor(np.tile(GLITCH_MEAN, len(GLITCH_PRIOR) + 2), device=device)[None, :, None, None]
    std = torch.tensor(np.tile(GLITCH_STD, len(GLITCH_PRIOR) + 2), device=device)[None, :, None, None]
    net = copy.deepcopy(_glitch_net()).to(device, memory_format=torch.channels_last)
    probs = []
    try:
        with torch.no_grad(), torch.autocast(device, torch.bfloat16, enabled=device == "cuda"):
            for start in range(0, len(frames), GLITCH_BATCH):
                judged = range(start, min(start + GLITCH_BATCH, len(frames)))
                # per judged frame: the prior frames (earliest first, clamped at the chunk start), it, the input image
                windows = np.stack([[*(frames[max(i - k, 0)] for k in GLITCH_PRIOR), frames[i], ref] for i in judged])
                x = torch.from_numpy(windows).to(device).permute(0, 1, 4, 2, 3).reshape(len(judged), -1, height, width)
                boxed = torch.zeros((len(judged), x.shape[1], side, side), dtype=torch.uint8, device=device)
                boxed[:, :, top:top + height, left:left + width] = x
                logits = net(((boxed.float() - mean) / std).contiguous(memory_format=torch.channels_last))
                probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
    finally:
        del net
        if device == "cuda":
            torch.cuda.empty_cache()
    p = np.concatenate(probs)
    over = np.flatnonzero(p > GLITCH_THRESHOLD)
    return {
        "frames": int(len(p)),
        "max_p": round(float(p.max()), 3),
        "max_frame": int(p.argmax()),
        "frames_over": int(len(over)),
        **({"over": [int(over[0]), int(over[-1])]} if len(over) else {}),
        "badness": round(float(p.max()) / GLITCH_THRESHOLD, 3),
        "flagged": bool(len(over)),
    }


def check_chunk(render: Path, *, audio: Path, kept_frames: int, flicker: bool, desync: bool, glitches: bool = False,
                image: Path | None = None, job_id: str = "", chunk_index: int = 0,
                judge_drift: bool = True) -> dict[str, Any]:
    """The enabled validators' scores for one rendered chunk; ``flagged`` when any fails.

    ``glitches`` needs ``image``, the image the chunk opens on. ``judge_drift`` false: the chunk
    ends on a different image, so flicker judges only sudden steps (:func:`flicker_scores`).
    """
    verdict: dict[str, Any] = {}
    if flicker:
        verdict["flicker"] = flicker_scores(render, kept_frames=kept_frames, judge_drift=judge_drift)
    if desync:
        verdict["desync"] = desync_scores(render, audio=audio, kept_frames=kept_frames)
    if glitches:
        try:
            if image is None:
                raise ValueError("the glitch check needs the input image")
            verdict["glitches"] = glitch_scores(render, image=image, kept_frames=kept_frames)
        except Exception as exc:  # a missing detector or an OOM must not drop the other validators' verdicts
            verdict["glitches"] = {"error": f"{type(exc).__name__}: {exc}", "flagged": False, "badness": 0.0}
    verdict["flagged"] = any(scores["flagged"] for scores in verdict.values() if isinstance(scores, dict))
    verdict["badness"] = max((scores["badness"] for scores in verdict.values() if isinstance(scores, dict)), default=0.0)
    log_event("lipsync_chunk_checked", job_id=job_id, chunk=chunk_index, **{
        name: {key: value for key, value in scores.items() if key != "face"}
        for name, scores in verdict.items() if isinstance(scores, dict)
    }, flagged=verdict["flagged"])
    return verdict
