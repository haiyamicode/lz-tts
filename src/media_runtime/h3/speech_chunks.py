"""Speech timing and prompts for lipsync (``lipsync``).

- **Timed transcript.** The caller's ``transcript`` (items of text with their
  start and end seconds), or greedy CTC (:func:`transcribe`, romanized words)
  when there is none. It only places each chunk's words in its prompt.
- **Split points.** The audio is cut into the fewest chunks that fit the cap,
  none shorter than ``MIN_CHUNK_SECONDS``. Each cut goes to the widest gap
  between timed items inside the window where both sides still fit; when that
  window holds no gap, the cut is its middle.
- **Freestyle cuts and pins.** :func:`quiet_cuts` cuts the timeline into
  chunks and :func:`pause_pins` places the input image inside each chunk every
  ``PIN_GAP_SECONDS``, both at the quietest frames of the voice
  (:func:`frame_loudness`): the middle of a real silence where one is in reach,
  else the lowest speech energy.
- **Prompt.** :data:`PROMPT_TEMPLATE`, general for any person on camera; each
  chunk gets only its own words.
"""

from __future__ import annotations

import math
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .compat import log_event

SAMPLE_RATE = 16000
#: Longest speech / lipsync chunk, seconds (the render cap may be shorter).
MAX_CHUNK_SECONDS = 10.0
#: No cut leaves a chunk shorter than this, unless the audio itself is.
MIN_CHUNK_SECONDS = 1.0
#: Freestyle pins the input image this far apart (and no closer), at pauses.
PIN_GAP_SECONDS = (2.5, 5.0)
#: Silence: 10 ms of the voice this far under its speech level.
SILENCE_BELOW_SPEECH_DB = 35.0
#: Each pin's cost on top of its loudness, so the spacing gets no extra pins.
PIN_COST_DB = 1.0
HOP_SECONDS = 0.01
#: Gaps between timed items shorter than this are not treated as pauses.
MIN_PAUSE_SECONDS = 0.08

#: The chunk prompt, general for any person on camera: it names nothing about
#: one avatar (no glasses, jewellery, props, pose or framing) and nothing about
#: the voice, which is the pinned audio itself. The words and the order to keep
#: the mouth moving come first, the rest after it and brief: the earlier
#: 495-word prompt put the words halfway in, and on prod job
#: dPvPt2aMffuv6qiuM4wHgB the lips stayed closed through a whole chunk of speech
#: on all three seeds; this one talks there (SyncNet 6.4) with as much head and
#: body motion. The only pins are the input image, at the chunk ends and in the
#: pauses between, so the prompt moves the hands and brings the body back to it.
#: The template's hand choreography: free hands lifting into gestures on key phrases, and settling back
#: into the first frame's pose in pauses. A pinned chunk's instruction takes their place
#: (:func:`render_prompt`): left in, they win over it. "He lays his free hand flat on his chest and keeps it
#: there" stayed an explaining gesture with the line at the end of the prompt or at its top, on two seeds,
#: and was held on both once these two lines were gone (local/experiments/h3-pinned-chunks-20261106).
GESTURE_LINE = (
    "The hands start where they are in the first frame, and anything a hand holds stays held. Free hands move with "
    "the speech: on key phrases they lift into relaxed explaining gestures in front of the body, palms opening "
    "toward the camera or a small push forward on an emphasized word, then lower again between phrases. The "
    "gestures vary in size and timing like a real person explaining something, never constant or repetitive, and "
    "never cover the face."
)
SETTLE_LINE = "In pauses the body and hands settle back into the first frame's pose while the face keeps moving."
PROMPT_TEMPLATE = (
    "A real person talking straight to the camera{language}, in a live video, not a still image, saying: "
    "\"{words}\"\n"
    "Their mouth moves the whole time they speak: lips, jaw and tongue shape every syllable in time with the voice, "
    "and the lips never rest closed or freeze while words are being spoken. The face reacts naturally to the meaning "
    "of the words.\n"
    "The head moves freely and naturally the whole time, with small turns, tilts and nods that carry the meaning of "
    "the words, and the neck and shoulders move with the head as one connected body, the way a real person moves "
    "while talking. Breathing is calm and barely visible, with no heaving, swaying or rocking.\n"
    f"{GESTURE_LINE}\n"
    "The eyes keep coming back to the lens, with natural blinks between phrases.\n"
    "The camera is locked, with no pan, tilt or zoom; the background, lighting, clothing and the person stay as in "
    "the first frame, and only the person moves.\n"
    f"{SETTLE_LINE}"
)
#: Language codes named in the prompt; others (and an unknown language) leave it out.
LANGUAGE_NAMES = {
    "ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fil": "Filipino", "fr": "French",
    "hi": "Hindi", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
    "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "th": "Thai", "tl": "Tagalog",
    "tr": "Turkish", "uk": "Ukrainian", "vi": "Vietnamese", "zh": "Chinese",
}


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    #: Language code, or ``None`` when unknown (CTC and caller transcripts).
    language: str | None
    #: Timed items: the CTC words, or the caller's words or phrases.
    words: tuple[Word, ...] = ()

    @classmethod
    def from_items(cls, items: list[dict[str, Any]]) -> Transcript:
        """A caller transcript: ``[{"text", "start", "end"}, ...]`` in seconds."""
        words = tuple(Word(float(i["start"]), float(i["end"]), str(i["text"]).strip()) for i in items)
        return cls(text=" ".join(w.text for w in words), language=None, words=words)


def render_prompt(words: str, language: str | None, *, pinned_instruction: str | None = None) -> str:
    """The chunk prompt: the words first, then the mouth, body, hands and the return to the first frame.
    A pinned chunk's ``pinned_instruction`` replaces the hand choreography (:data:`GESTURE_LINE`,
    :data:`SETTLE_LINE`)."""
    name = LANGUAGE_NAMES.get(language or "")
    prompt = PROMPT_TEMPLATE.format(language=f" in {name}" if name else "", words=" ".join(words.split()))
    if pinned_instruction:
        prompt = prompt.replace(GESTURE_LINE, pinned_instruction).replace(f"\n{SETTLE_LINE}", "")
    return prompt


def load_audio_16k(path: Path):
    """Mono 16 kHz float32 samples (decoded by ffmpeg, like the rest of the job)."""
    import numpy as np

    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(pcm, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# CTC transcription
# ---------------------------------------------------------------------------

#: CTC frames a second (wav2vec2's 320-sample stride at 16 kHz).
CTC_FPS = 50
#: Audio per forward pass and the context on each side of it, as ctc-forced-aligner runs this model.
CTC_WINDOW_SECONDS, CTC_CONTEXT_SECONDS = 30, 2
#: Blank frames between two letters that start a new word: the vocabulary has no word separator.
WORD_GAP_FRAMES = 3

_CTC: dict[str, Any] = {}
_CTC_LOCK = threading.Lock()


def _ctc_model() -> dict[str, Any]:
    """MMS-300m CTC (``data/lipsync/mms-300m-ctc``: MahmoudAshraf/mms-300m-1130-forced-aligner in fp16, the
    speech worker's aligner), loaded once and kept on the GPU."""
    if not _CTC:
        import json

        import torch
        from transformers import Wav2Vec2ForCTC

        from .compat import CTC_ROOT, project_path

        model_dir = project_path(CTC_ROOT)
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"the CTC transcription model is missing (expected {model_dir})")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), dtype=torch.float16 if device == "cuda" else torch.float32)
        vocab = json.loads((model_dir / "vocab.json").read_text())
        letters = {index: token for token, index in vocab.items() if len(token) == 1}
        _CTC.update(model=model.to(device).eval(), device=device, letters=letters, blank=vocab["<blank>"])
    return _CTC


def _ctc_frames(samples, ctc: dict[str, Any]):
    """The most likely token of every CTC frame of 16 kHz ``samples``."""
    import torch

    model, device = ctc["model"], ctc["device"]
    audio = torch.from_numpy(samples).to(device, model.dtype)
    window, context = CTC_WINDOW_SECONDS * SAMPLE_RATE, CTC_CONTEXT_SECONDS * SAMPLE_RATE
    stride = SAMPLE_RATE // CTC_FPS
    try:
        with torch.inference_mode():
            if len(audio) <= window:
                return model(audio[None]).logits[0].argmax(-1).cpu().numpy()
            extension = -len(audio) % window
            padded = torch.nn.functional.pad(audio, (context, context + extension))
            pieces = padded.unfold(0, window + 2 * context, window)
            keep = slice(context // stride, (context + window) // stride)
            frames = torch.cat([model(piece[None]).logits[0, keep].argmax(-1) for piece in pieces])
            return frames[: -(-len(audio) // stride)].cpu().numpy()
    finally:
        if device == "cuda":
            torch.cuda.empty_cache()  # the activations; the weights stay


def ctc_words(frames, letters: dict[int, str], *, blank: int) -> tuple[Word, ...]:
    """Greedy CTC decoding of per-frame tokens: each run of a letter is one letter (a blank between two runs of the
    same letter keeps both); a letter starts a new word after ``WORD_GAP_FRAMES`` frames without one. A word runs
    from its first letter's first frame to its last letter's last frame."""
    words: list[Word] = []
    text, start, end, previous = "", 0, 0, blank
    for frame, token in enumerate(int(token) for token in frames):
        if token != previous and token in letters:
            if text and frame - end > WORD_GAP_FRAMES:
                words.append(Word(start / CTC_FPS, end / CTC_FPS, text))
                text = ""
            if not text:
                start = frame
            text += letters[token]
        if token in letters:
            end = frame + 1
        previous = token
    if text:
        words.append(Word(start / CTC_FPS, end / CTC_FPS, text))
    return tuple(words)


def transcribe(audio) -> Transcript:
    """The words of 16 kHz mono ``audio`` with their times, by greedy CTC (MMS-300m, 1130 languages): one forward
    pass per 30 s, so it cannot loop, run away or invent text in silence the way an autoregressive decoder can. The
    text is romanized lowercase letters (uroman, the model's vocabulary) and the language is unknown; the words only
    place each chunk's speech in its prompt."""
    import numpy as np

    from .minimax_h3_runtime import make_room_beside_h3

    with _CTC_LOCK:
        make_room_beside_h3()
        ctc = _ctc_model()
        frames = _ctc_frames(np.asarray(audio, dtype=np.float32), ctc)
    words = ctc_words(frames, ctc["letters"], blank=ctc["blank"])
    log_event("speech_transcribed", words=len(words), audio_seconds=round(len(audio) / SAMPLE_RATE, 3))
    return Transcript(text=" ".join(word.text for word in words), language=None, words=words)


# ---------------------------------------------------------------------------
# freestyle pins (pure)
# ---------------------------------------------------------------------------


def frame_loudness(samples, *, fps: float, frames: int):
    """How far each output frame's loudest 10 ms of the voice rises above
    silence, in dB; 0 where the whole frame is silent. Silence is
    ``SILENCE_BELOW_SPEECH_DB`` under the speech level (the 90th percentile of
    the 10 ms windows), so it adapts to how loud the recording is."""
    import numpy as np

    hop = int(SAMPLE_RATE * HOP_SECONDS)
    windows = len(samples) // hop
    if windows == 0:
        return np.zeros(frames)
    rms = np.sqrt((np.asarray(samples[: windows * hop], dtype=np.float64).reshape(windows, hop) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-12)
    floor = float(np.percentile(db, 90)) - SILENCE_BELOW_SPEECH_DB
    loudness = np.zeros(frames)
    for frame in range(frames):
        first = int(frame / fps / HOP_SECONDS)
        span = db[first:max(first + 1, math.ceil((frame + 1) / fps / HOP_SECONDS))]
        if len(span):
            loudness[frame] = max(0.0, float(span.max()) - floor)
    return loudness


def _quiet_costs(loudness):
    """Loudness as the cost of a pin or cut there; inside a silence a hair dearer
    away from its middle, so they land mid-pause."""
    import numpy as np

    cost = np.asarray(loudness, dtype=np.float64).copy()
    last = len(cost) - 1
    frame = 0
    while frame <= last:
        if cost[frame] > 0:
            frame += 1
            continue
        end = frame
        while end < last and cost[end + 1] == 0:
            end += 1
        cost[frame:end + 1] = 0.01 * np.abs(np.arange(frame, end + 1) - (frame + end) / 2)
        frame = end + 1
    return cost


def _backtrack(previous, last: int) -> list[int]:
    points = [last]
    while points[-1] != 0:
        points.append(int(previous[points[-1]]))
    return points[::-1]


def _cheapest_steps(cost, *, shortest: int, longest: int) -> list[int]:
    """``0 = p0 < ... < pn = last``, each step ``shortest..longest`` frames, with
    the least summed ``cost`` at the inner points (any number of them)."""
    import numpy as np

    last = len(cost) - 1
    best = np.full(last + 1, np.inf)
    best[0] = 0.0
    previous = np.zeros(last + 1, dtype=int)
    for frame in range(shortest, last + 1):
        lo, hi = max(0, frame - longest), frame - shortest
        k = int(np.argmin(best[lo:hi + 1]))
        if np.isfinite(best[lo + k]):
            best[frame] = best[lo + k] + (cost[frame] if frame < last else 0.0)
            previous[frame] = lo + k
    if not np.isfinite(best[last]):
        raise ValueError(f"steps of {shortest}..{longest} frames cannot cover {last} frames")
    return _backtrack(previous, last)


def _cheapest_n_steps(cost, steps: int, *, shortest: int, longest: int) -> list[int] | None:
    """Like :func:`_cheapest_steps` with exactly ``steps`` steps; None if impossible."""
    import numpy as np

    last = len(cost) - 1
    best = np.full(last + 1, np.inf)
    best[0] = 0.0
    trail = []
    for _ in range(steps):
        reach = np.full(last + 1, np.inf)
        previous = np.zeros(last + 1, dtype=int)
        for frame in range(shortest, last + 1):
            lo, hi = max(0, frame - longest), frame - shortest
            k = int(np.argmin(best[lo:hi + 1]))
            if np.isfinite(best[lo + k]):
                reach[frame] = best[lo + k] + (cost[frame] if frame < last else 0.0)
                previous[frame] = lo + k
        best = reach
        trail.append(previous)
    if not np.isfinite(best[last]):
        return None
    points = [last]
    for previous in reversed(trail):
        points.append(int(previous[points[-1]]))
    return points[::-1]


def quiet_cuts(loudness, *, shortest: int, longest: int) -> list[int]:
    """Chunk cuts (each the last frame of one chunk and the first of the next)
    at the quietest frames: the fewest chunks, each ``shortest..longest`` frames
    past its first, and one chunk more when that makes the loudest cut quieter."""
    last = len(loudness) - 1
    if last <= longest:
        return []
    cost = _quiet_costs(loudness)
    fewest = math.ceil(last / longest)
    best = _cheapest_n_steps(cost, fewest, shortest=shortest, longest=longest)
    if best is None:
        raise ValueError(f"{fewest} chunks of {shortest}..{longest} frames cannot cover {last} frames")
    if max(loudness[c] for c in best[1:-1]) > 0:
        more = _cheapest_n_steps(cost, fewest + 1, shortest=shortest, longest=longest)
        if more is not None and max(loudness[c] for c in more[1:-1]) < max(loudness[c] for c in best[1:-1]):
            best = more
    return best[1:-1]


def pause_pins(loudness, *, shortest: int, longest: int) -> list[int]:
    """Frames ``0 = p0 < p1 < ... < pn = last`` (``last`` = the final frame),
    each ``shortest..longest`` frames after the one before, at the quietest
    frames: the least summed loudness with ``PIN_COST_DB`` a pin, so no more pins
    than the spacing needs; inside a silence, the frame nearest its middle."""
    last = len(loudness) - 1
    if last <= 0:
        return [0]
    if last <= longest:
        return [0, last]
    return _cheapest_steps(_quiet_costs(loudness) + PIN_COST_DB, shortest=min(shortest, last // 2), longest=longest)


# ---------------------------------------------------------------------------
# split planning (pure)
# ---------------------------------------------------------------------------


def pauses_between(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Gaps between consecutive (sorted) speech spans or words."""
    ordered = sorted(spans)
    return [(a_end, b_start) for (_, a_end), (b_start, _) in zip(ordered, ordered[1:]) if b_start - a_end >= MIN_PAUSE_SECONDS]


def split_times(
    duration: float,
    *,
    max_seconds: float,
    gaps: list[tuple[float, float]],
    allowed=lambda seconds: True,
    step: float = 1 / 24,
    min_seconds: float = MIN_CHUNK_SECONDS,
) -> list[float]:
    """Cut points (seconds, on the ``step`` frame grid) splitting ``duration``
    into the fewest chunks of at most ``max_seconds``, none under ``min_seconds``.

    A span needing ``n`` chunks is cut into ``k = n // 2`` and ``n - k``; the cut
    may go anywhere both sides still fit, and goes to the middle of the widest
    gap (between timed items) reaching into that window - else to the middle of
    the span's split. Frames ``allowed`` vetoes (an anchor right after) are
    never used.
    """
    pauses = [(a, b) for a, b in gaps if b - a >= MIN_PAUSE_SECONDS]

    def frames(lo: float, hi: float) -> list[float]:
        first, last = math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9)
        return [k * step for k in range(first, last + 1)]

    cuts: list[float] = []

    def split(start: float, end: float, chunks: int) -> None:
        if chunks <= 1:
            return
        left = chunks // 2
        right = chunks - left
        lo = max(start + left * min_seconds, end - right * max_seconds)
        hi = min(start + left * max_seconds, end - right * min_seconds)
        ideal = start + (end - start) * left / chunks
        best: tuple[tuple[float, float], float] | None = None
        for a, b in pauses:
            inside = [t for t in frames(max(a, lo), min(b, hi)) if allowed(t)]
            if not inside:
                continue
            middle = (max(a, lo) + min(b, hi)) / 2
            t = min(inside, key=lambda x: abs(x - middle))
            key = (-(b - a), abs(t - ideal))
            if best is None or key < best[0]:
                best = (key, t)
        if best is not None:
            cut = best[1]
        else:
            candidates = [t for t in frames(lo, hi) if allowed(t)]
            cut = min(candidates, key=lambda t: abs(t - ideal)) if candidates else round(ideal / step) * step
        split(start, cut, left)
        cuts.append(cut)
        split(cut, end, right)

    duration = float(duration)
    if duration > max_seconds:
        split(0.0, duration, math.ceil(duration / max_seconds - 1e-9))
    return sorted(cuts)


def words_between(words: tuple[Word, ...], start: float, end: float, *, last: bool) -> str:
    """The words whose middle falls in ``[start, end)`` (``[start, end]`` for the last chunk)."""
    picked = [w.text for w in words if start <= (w.start + w.end) / 2 < end or (last and (w.start + w.end) / 2 >= start)]
    return " ".join(picked)
