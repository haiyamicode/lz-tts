from collections import Counter
import ctypes
from enum import Enum
from pathlib import Path
import sys
from typing import Dict, List, Optional, Union

_DIR = Path(__file__).parent

# Pre-load bundled shared libraries before importing the C++ extension
# This ensures the libraries are found regardless of system LD_LIBRARY_PATH
if sys.platform == "linux":
    _libespeak = _DIR / "libespeak-ng.so.1"
    _libonnx = _DIR / "libonnxruntime.so.1.14.1"
    if _libespeak.exists():
        ctypes.CDLL(str(_libespeak), mode=ctypes.RTLD_GLOBAL)
    if _libonnx.exists():
        ctypes.CDLL(str(_libonnx), mode=ctypes.RTLD_GLOBAL)

# Import C++ extension (relative import since it's in this package)
from .piper_phonemize_cpp import (  # noqa: E402
    phonemize_espeak as _phonemize_espeak,
    phonemize_espeak_with_mapping as _phonemize_espeak_with_mapping,
    phonemize_codepoints as _phonemize_codepoints,
    phoneme_ids_espeak as _phonemize_ids_espeak,
    phoneme_ids_codepoints as _phonemize_ids_codepoints,
    get_espeak_map,
    get_codepoints_map,
    get_max_phonemes,
    tashkeel_run as _tashkeel_run,
)

_TASHKEEL_MODEL = _DIR / "libtashkeel_model.ort"


class TextCasing(str, Enum):
    """Casing applied to text for phonemize_codepoints"""

    IGNORE = "ignore"
    LOWER = "lower"
    UPPER = "upper"
    FOLD = "fold"


def phonemize_espeak(
    text: str,
    voice: str,
    data_path: Optional[Union[str, Path]] = None,
) -> List[List[str]]:
    if data_path is None:
        data_path = _DIR / "espeak-ng-data"

    return _phonemize_espeak(text, voice, str(data_path))


# Type alias for word mapping: (textStart, textLength, phonemeStart, phonemeEnd)
WordMapping = tuple[int, int, int, int]


def phonemize_espeak_with_mapping(
    text: str,
    voice: str,
    data_path: Optional[Union[str, Path]] = None,
) -> tuple[List[List[str]], List[WordMapping]]:
    """Phonemize text and return word-to-phoneme mapping.

    Args:
        text: Input text to phonemize.
        voice: eSpeak voice to use.
        data_path: Optional path to espeak-ng data.

    Returns:
        Tuple of (phonemes, word_mapping) where:
        - phonemes: List of sentences, each a list of phoneme strings
        - word_mapping: List of (textStart, textLength, phonemeStart, phonemeEnd)
          tuples mapping input text byte positions to phoneme indices.
          Note: textStart is 1-indexed (from espeak).
    """
    if data_path is None:
        data_path = _DIR / "espeak-ng-data"

    return _phonemize_espeak_with_mapping(text, voice, str(data_path))


def phonemize_codepoints(
    text: str,
    casing: Union[str, TextCasing] = TextCasing.FOLD,
) -> List[List[str]]:
    casing = TextCasing(casing)
    return _phonemize_codepoints(text, casing.value)


def phoneme_ids_espeak(
    phonemes: List[str],
    missing_phonemes: "Optional[Counter[str]]" = None,
) -> List[int]:
    phoneme_ids, missing_counts = _phonemize_ids_espeak(phonemes)
    if missing_phonemes is not None:
        missing_phonemes.update(missing_counts)

    return phoneme_ids


def phoneme_ids_codepoints(
    language: str,
    phonemes: List[str],
    missing_phonemes: "Optional[Counter[str]]" = None,
) -> List[int]:
    phoneme_ids, missing_counts = _phonemize_ids_codepoints(language, phonemes)
    if missing_phonemes is not None:
        missing_phonemes.update(missing_counts)

    return phoneme_ids


def tashkeel_run(text: str, tashkeel_model: Union[str, Path] = _TASHKEEL_MODEL) -> str:
    tashkeel_model = str(tashkeel_model)
    return _tashkeel_run(tashkeel_model, text)
