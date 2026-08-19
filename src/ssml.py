"""Strict parsing for the SSML subset supported by the synthesis API.

The parser deliberately handles XML as XML.  It produces the plain text sent
to the normal TTS route plus source-character spans for the two operations we
currently implement: timed breaks and IPA pronunciation overrides.
"""

from __future__ import annotations

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


def _collapse_text_whitespace(text: str) -> tuple[str, list[int], list[int]]:
    """Collapse decoded SSML whitespace and retain biased boundary mappings.

    A raw boundary inside a whitespace run can mean either side of the one
    collapsed space. ``before`` maps to its left edge and ``after`` to its
    right edge. This lets breaks remain before the separator while phoneme
    spans exclude surrounding whitespace.
    """
    before: list[int | None] = [None] * (len(text) + 1)
    after: list[int | None] = [None] * (len(text) + 1)
    output: list[str] = []

    def record(position: int, left: int, right: int | None = None) -> None:
        right = left if right is None else right
        current_before = before[position]
        current_after = after[position]
        before[position] = left if current_before is None else min(current_before, left)
        after[position] = right if current_after is None else max(current_after, right)

    cursor = 0
    while cursor < len(text):
        if not text[cursor].isspace():
            output_position = len(output)
            record(cursor, output_position)
            output.append(text[cursor])
            record(cursor + 1, output_position + 1)
            cursor += 1
            continue

        run_start = cursor
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        run_end = cursor
        left = len(output)
        if output and run_end < len(text):
            output.append(" ")
        right = len(output)
        for position in range(run_start, run_end + 1):
            record(position, left, right)

    record(0, 0)
    record(len(text), len(output))
    return (
        "".join(output),
        [int(position) for position in before],
        [int(position) for position in after],
    )


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
            # A break is a boundary in the decoded text stream. Treating it as
            # whitespace lets the one normalization pass handle punctuation,
            # adjacent breaks, indentation, and languages without spaces alike.
            self.append_text(" ")
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
        text, before, after = _collapse_text_whitespace(raw_text)
        if not text:
            raise ValueError("SSML contains no text to synthesize")

        adjusted: list[SSMLOperation] = []
        for operation in self.operations:
            if isinstance(operation, BreakOperation):
                adjusted.append(
                    BreakOperation(
                        position=before[operation.position],
                        duration_seconds=operation.duration_seconds,
                    )
                )
            else:
                start = after[operation.start]
                end = before[operation.end]
                if start >= end:
                    raise ValueError("SSML <phoneme> display text cannot be only outer whitespace")
                adjusted.append(
                    PronunciationOperation(
                        start=start,
                        end=end,
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
