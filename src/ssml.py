"""Strict parsing for the SSML subset supported by the synthesis API.

The parser deliberately handles XML as XML.  It produces the plain text sent
to the normal TTS route plus source-character spans for the two operations we
currently implement: timed breaks and IPA pronunciation overrides.
"""

from __future__ import annotations

import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TypeAlias

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException

SSML_NAMESPACE = "http://www.w3.org/2001/10/synthesis"
MAX_BREAK_SECONDS = Decimal("60")


@dataclass(frozen=True)
class BreakOperation:
    """Insert silence at ``position`` in the decoded synthesis text."""

    position: int
    duration_seconds: float


@dataclass(frozen=True)
class PronunciationOperation:
    """Pronounce one decoded text span using the supplied IPA string."""

    start: int
    end: int
    alphabet: str
    phonemes: str


SSMLOperation: TypeAlias = BreakOperation | PronunciationOperation


@dataclass(frozen=True)
class SSMLDocument:
    """Decoded synthesis text and its ordered post-processing operations."""

    text: str
    operations: tuple[SSMLOperation, ...]

    @property
    def breaks(self) -> tuple[BreakOperation, ...]:
        return tuple(op for op in self.operations if isinstance(op, BreakOperation))

    @property
    def pronunciations(self) -> tuple[PronunciationOperation, ...]:
        return tuple(op for op in self.operations if isinstance(op, PronunciationOperation))


def _expanded_name(tag: str) -> tuple[str | None, str]:
    if tag.startswith("{"):
        namespace, separator, local_name = tag[1:].partition("}")
        if not separator:
            raise ValueError(f"Invalid XML element name {tag!r}")
        return namespace, local_name
    return None, tag


def _validate_element_name(tag: str) -> str:
    namespace, local_name = _expanded_name(tag)
    if namespace not in {None, SSML_NAMESPACE}:
        raise ValueError(f"Unsupported SSML namespace {namespace!r}")
    return local_name


def _parse_break_time(value: str) -> float:
    normalized = value.strip().lower()
    if normalized.endswith("ms"):
        number, divisor = normalized[:-2], Decimal(1000)
    elif normalized.endswith("s"):
        number, divisor = normalized[:-1], Decimal(1)
    else:
        raise ValueError("SSML <break> time must use the 'ms' or 's' unit")

    integer, dot, fraction = number.partition(".")
    if (
        (dot and "." in fraction)
        or (integer and not integer.isdigit())
        or (fraction and not fraction.isdigit())
        or not (integer or fraction)
    ):
        raise ValueError(f"Invalid SSML <break> time {value!r}")

    try:
        seconds = Decimal(number) / divisor
    except (InvalidOperation, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid SSML <break> time {value!r}") from exc
    if not seconds.is_finite() or seconds <= 0 or seconds > MAX_BREAK_SECONDS:
        raise ValueError(
            f"SSML <break> time must be greater than 0 and at most {MAX_BREAK_SECONDS:g} seconds"
        )
    return float(seconds)


def _break_separates_spoken_text(text: str, position: int) -> bool:
    """Return whether removing a break would join two spoken-text runs."""
    if not 0 < position < len(text) or not text[position].isalnum():
        return False

    previous = position - 1
    while previous >= 0 and unicodedata.category(text[previous]).startswith("M"):
        previous -= 1
    return previous >= 0 and text[previous].isalnum()


class _DocumentBuilder:
    def __init__(self) -> None:
        self._text_parts: list[str] = []
        self._length = 0
        self.operations: list[SSMLOperation] = []

    @property
    def position(self) -> int:
        return self._length

    def append_text(self, value: str | None) -> None:
        if not value:
            return
        self._text_parts.append(value)
        self._length += len(value)

    def visit_contents(self, element) -> None:
        self.append_text(element.text)
        for child in element:
            self.visit_element(child)
            self.append_text(child.tail)

    def visit_element(self, element) -> None:
        name = _validate_element_name(element.tag)
        if name == "break":
            unsupported = set(element.attrib) - {"time"}
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"Unsupported SSML <break> attribute(s): {names}")
            if len(element) or (element.text and element.text.strip()):
                raise ValueError("SSML <break> must be empty")
            self.operations.append(
                BreakOperation(
                    position=self.position,
                    duration_seconds=_parse_break_time(element.attrib.get("time", "1s")),
                )
            )
            return

        if name == "phoneme":
            unsupported = set(element.attrib) - {"alphabet", "ph"}
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"Unsupported SSML <phoneme> attribute(s): {names}")
            if len(element):
                raise ValueError("SSML <phoneme> may contain text only")
            alphabet = element.attrib.get("alphabet")
            phonemes = element.attrib.get("ph")
            if alphabet is None or alphabet.strip().lower() != "ipa":
                raise ValueError("SSML <phoneme> currently requires alphabet='ipa'")
            if phonemes is None or not phonemes.strip():
                raise ValueError("SSML <phoneme> requires a non-empty 'ph' attribute")
            display_text = element.text or ""
            if not display_text.strip():
                raise ValueError("SSML <phoneme> requires non-empty display text")
            start = self.position
            self.append_text(display_text)
            self.operations.append(
                PronunciationOperation(
                    start=start,
                    end=self.position,
                    alphabet="ipa",
                    phonemes=phonemes.strip(),
                )
            )
            return

        raise ValueError(f"Unsupported SSML element <{name}>")

    def finish(self) -> SSMLDocument:
        raw_text = "".join(self._text_parts)
        break_positions = {
            operation.position
            for operation in self.operations
            if isinstance(operation, BreakOperation)
        }
        separator_positions = sorted(
            position
            for position in break_positions
            if _break_separates_spoken_text(raw_text, position)
        )
        if separator_positions:
            parts: list[str] = []
            cursor = 0
            for position in separator_positions:
                parts.extend((raw_text[cursor:position], " "))
                cursor = position
            parts.append(raw_text[cursor:])
            rendered_text = "".join(parts)
        else:
            rendered_text = raw_text

        def shift_before(position: int) -> int:
            return position + bisect_left(separator_positions, position)

        def shift_through(position: int) -> int:
            return position + bisect_right(separator_positions, position)

        leading = len(rendered_text) - len(rendered_text.lstrip())
        trailing_end = len(rendered_text.rstrip())
        text = rendered_text[leading:trailing_end]
        if not text:
            raise ValueError("SSML contains no text to synthesize")

        adjusted: list[SSMLOperation] = []
        for operation in self.operations:
            if isinstance(operation, BreakOperation):
                position = shift_before(operation.position)
                position = min(max(position, leading), trailing_end) - leading
                adjusted.append(
                    BreakOperation(position=position, duration_seconds=operation.duration_seconds)
                )
            else:
                start = shift_through(operation.start)
                end = shift_before(operation.end)
                if start < leading or end > trailing_end:
                    raise ValueError("SSML <phoneme> display text cannot be only outer whitespace")
                adjusted.append(
                    PronunciationOperation(
                        start=start - leading,
                        end=end - leading,
                        alphabet=operation.alphabet,
                        phonemes=operation.phonemes,
                    )
                )
        return SSMLDocument(text=text, operations=tuple(adjusted))


def parse_ssml(ssml: str) -> SSMLDocument:
    """Parse a complete SSML document without HTML recovery or regex stripping.

    XML character/entity references are decoded by the XML parser before text
    offsets are recorded.  DTDs and entity declarations are rejected by
    ``defusedxml`` rather than expanded.
    """
    if not isinstance(ssml, str) or not ssml.strip():
        raise ValueError("SSML input is empty")
    try:
        root = SafeElementTree.fromstring(ssml)
    except (SafeElementTree.ParseError, DefusedXmlException) as exc:
        raise ValueError(f"Invalid SSML: {exc}") from exc

    if _validate_element_name(root.tag) != "speak":
        raise ValueError("SSML root element must be <speak>")

    builder = _DocumentBuilder()
    builder.visit_contents(root)
    return builder.finish()


__all__ = [
    "BreakOperation",
    "PronunciationOperation",
    "SSMLDocument",
    "SSMLOperation",
    "SSML_NAMESPACE",
    "parse_ssml",
]
