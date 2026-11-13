"""Face detection task tests: real YuNet model, real images, no mocks."""

import io
from pathlib import Path

import pytest
from PIL import Image

from src.api.face_detection import MAX_DETECT_SIDE, MODEL_FILE, FaceDetector, FaceInputError

_MODEL = Path("data/face-detection") / MODEL_FILE
_FACE = Path(__file__).parent / "testdata" / "face.jpg"
pytestmark = pytest.mark.skipif(not _MODEL.exists(), reason="YuNet model not present under data/face-detection")


def _encode(image: Image.Image, **save_kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, **save_kwargs)
    return buffer.getvalue()


@pytest.fixture(scope="module")
def detector() -> FaceDetector:
    return FaceDetector()


def _center(face: dict) -> tuple[float, float]:
    return face["x"] + face["width"] / 2, face["y"] + face["height"] / 2


def test_finds_the_face_with_landmarks_inside_its_box(detector):
    result = detector.detect(_FACE.read_bytes())
    assert (result["width"], result["height"]) == (256, 256)
    assert result["faceCount"] == 1
    face = result["faces"][0]
    assert face["score"] >= 0.8
    for x, y in face["landmarks"].values():
        assert face["x"] <= x <= face["x"] + face["width"]
        assert face["y"] <= y <= face["y"] + face["height"]


def test_blank_image_has_no_faces(detector):
    result = detector.detect(_encode(Image.new("RGB", (640, 480), (200, 200, 200))))
    assert result["faceCount"] == 0 and result["faces"] == []


def test_downscaled_detection_maps_back_to_full_size_pixels(detector):
    # A 2400 px image is detected at MAX_DETECT_SIDE; its boxes must equal the boxes found
    # on the same image pre-shrunk to that size, scaled back up.
    side = 2400
    big_image = Image.open(_FACE).convert("RGB").resize((side, side), Image.BICUBIC)
    shrunk = big_image.resize((MAX_DETECT_SIDE, MAX_DETECT_SIDE), Image.BILINEAR)
    big = detector.detect(_encode(big_image))
    ref = detector.detect(_encode(shrunk))["faces"][0]
    assert (big["width"], big["height"]) == (side, side)
    face, factor = big["faces"][0], side / MAX_DETECT_SIDE
    for got, want in zip(_center(face), _center(ref), strict=True):
        assert got == pytest.approx(want * factor, abs=0.02 * ref["width"] * factor)
    assert face["width"] == pytest.approx(ref["width"] * factor, rel=0.05)
    assert face["landmarks"]["nose"][0] == pytest.approx(ref["landmarks"]["nose"][0] * factor, rel=0.02)


def test_exif_orientation_is_applied(detector):
    # Stored sideways with an EXIF "rotate 90 CW" tag: viewers (and the detector) see it upright.
    upright = Image.open(_FACE).convert("RGB")
    canvas = upright.resize((256, 384))
    exif = Image.Exif()
    exif[0x0112] = 6
    stored = canvas.transpose(Image.Transpose.ROTATE_90)
    result = detector.detect(_encode(stored, exif=exif))
    assert (result["width"], result["height"]) == (256, 384)
    assert result["faceCount"] == 1


def test_min_score_and_max_faces(detector):
    assert detector.detect(_FACE.read_bytes(), min_score=0.99)["faceCount"] == 0
    assert detector.detect(_FACE.read_bytes(), max_faces=1)["faceCount"] == 1


def test_bad_input_raises(detector):
    with pytest.raises(FaceInputError):
        detector.detect(b"")
    with pytest.raises(FaceInputError):
        detector.detect(b"not an image")
