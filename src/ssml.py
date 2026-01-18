"""SSML parser for TTS synthesis.

Supports a subset of SSML similar to Google's Text-to-Speech API.
Currently supported elements:
- <speak>: Root element (optional)
- <break>: Insert pause with time="Xms" or time="Xs"
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Union


@dataclass
class TextSegment:
    """A segment of text to synthesize."""

    text: str


@dataclass
class BreakSegment:
    """A pause/break in speech."""

    duration_ms: int


Segment = Union[TextSegment, BreakSegment]


def _parse_time(time_str: str) -> int:
    """Parse SSML time string to milliseconds.

    Supports formats: "500ms", "1s", "1.5s"
    """
    time_str = time_str.strip().lower()

    if time_str.endswith("ms"):
        return int(float(time_str[:-2]))
    elif time_str.endswith("s"):
        return int(float(time_str[:-1]) * 1000)
    else:
        # Assume milliseconds if no unit
        return int(float(time_str))


def _extract_segments(element: ET.Element) -> list[Segment]:
    """Recursively extract text and break segments from an element."""
    segments: list[Segment] = []

    # Add text before first child
    if element.text:
        text = element.text.strip()
        if text:
            segments.append(TextSegment(text=text))

    # Process children
    for child in element:
        if child.tag == "break":
            time_attr = child.get("time", "250ms")
            duration_ms = _parse_time(time_attr)
            segments.append(BreakSegment(duration_ms=duration_ms))
        else:
            # Recursively process other elements
            segments.extend(_extract_segments(child))

        # Add tail text after this child
        if child.tail:
            text = child.tail.strip()
            if text:
                segments.append(TextSegment(text=text))

    return segments


def _merge_adjacent_text(segments: list[Segment]) -> list[Segment]:
    """Merge adjacent TextSegments into single segments."""
    if not segments:
        return []

    merged: list[Segment] = []
    for seg in segments:
        if isinstance(seg, TextSegment) and merged and isinstance(merged[-1], TextSegment):
            merged[-1] = TextSegment(text=merged[-1].text + " " + seg.text)
        else:
            merged.append(seg)
    return merged


def parse_ssml(ssml: str) -> list[Segment]:
    """Parse SSML string into segments.

    Args:
        ssml: SSML string, must be wrapped in <speak> tags.

    Returns:
        List of TextSegment and BreakSegment objects.

    Raises:
        ValueError: If SSML is malformed or missing <speak> wrapper.
    """
    ssml = ssml.strip()

    if not ssml.startswith("<speak"):
        raise ValueError("SSML must be wrapped in <speak> tags")

    try:
        root = ET.fromstring(ssml)
    except ET.ParseError as e:
        raise ValueError(f"Invalid SSML: {e}") from e

    if root.tag != "speak":
        raise ValueError("SSML root element must be <speak>")

    segments = _extract_segments(root)
    return _merge_adjacent_text(segments)


def generate_silence(duration_ms: int, sample_rate: int = 22050) -> "np.ndarray":
    """Generate silence as int16 audio samples.

    Args:
        duration_ms: Duration in milliseconds.
        sample_rate: Audio sample rate.

    Returns:
        Numpy array of zeros (int16).
    """
    import numpy as np

    num_samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(num_samples, dtype=np.int16)
