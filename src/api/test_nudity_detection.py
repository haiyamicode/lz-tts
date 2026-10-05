"""Nudity detection task tests — real model, real images, no mocks."""

import io
import uuid

import pytest
from PIL import Image

from src.api.nudity_detection import NudityDetectRequest, NudityDetector, NudityInputError

_MODEL_DIR = "data/nsfw-detection/marqo-nsfw-image-detection-384"
pytestmark = pytest.mark.skipif(
    not __import__("os").path.isdir(_MODEL_DIR),
    reason="nsfw detection snapshot not present under data/nsfw-detection",
)


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (320, 240)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def detector() -> NudityDetector:
    instance = NudityDetector()
    instance.classify(_png_bytes((255, 255, 255)))  # loads the ViT-tiny once
    return instance


def test_rejects_empty_input(detector: NudityDetector) -> None:
    with pytest.raises(NudityInputError, match="empty"):
        detector.classify(b"")


def test_rejects_non_image_bytes(detector: NudityDetector) -> None:
    with pytest.raises(NudityInputError, match="corrupt"):
        detector.classify(b"not an image at all" * 16)
    with pytest.raises(NudityInputError, match="corrupt"):
        detector.classify(uuid.uuid4().bytes)


def _assert_result_shape(result: dict) -> None:
    assert result["model"] == "marqo-nsfw-image-detection-384"
    scores = {e["label"]: e["score"] for e in result["labels"]}
    assert set(scores) == {"SFW", "NSFW"}
    assert result["label"] == max(scores, key=lambda key: scores[key])
    assert result["nsfwScore"] == scores["NSFW"]
    assert result["nsfw"] is (scores["NSFW"] >= result["threshold"])
    assert 0.0 <= scores["SFW"] <= 1.0 and 0.0 <= scores["NSFW"] <= 1.0
    assert scores["SFW"] + scores["NSFW"] == pytest.approx(1.0, abs=1e-4)
    assert result["labels"] == sorted(result["labels"], key=lambda e: e["score"], reverse=True)


def test_classify_shape(detector: NudityDetector) -> None:
    result = detector.classify(_png_bytes((30, 120, 240)))
    _assert_result_shape(result)


def test_result_is_deterministic(detector: NudityDetector) -> None:
    first = detector.classify(_png_bytes((30, 120, 240)))
    second = detector.classify(_png_bytes((30, 120, 240)))
    assert first == second


def test_threshold_overrides_flag(detector: NudityDetector) -> None:
    base = detector.classify(_png_bytes((30, 120, 240)))
    strict = detector.classify(_png_bytes((30, 120, 240)), threshold=1.0)
    loose = detector.classify(_png_bytes((30, 120, 240)), threshold=0.0)
    assert base["nsfwScore"] == strict["nsfwScore"] == loose["nsfwScore"]
    assert strict["nsfw"] is (base["nsfwScore"] >= 1.0)
    assert loose["nsfw"] is (base["nsfwScore"] >= 0.0)


def test_request_validation() -> None:
    assert NudityDetectRequest(image_url="https://x/img.png").threshold == 0.5
    with pytest.raises(Exception):
        NudityDetectRequest(threshold=1.5)