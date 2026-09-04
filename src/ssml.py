"""Parsing for the SSML subset supported by the synthesis API.

Documents are parsed with the standard library HTML parser, so plain text
that an XML parser would reject (bare ``&``, unescaped ``<``) passes through
unchanged.  The tag set stays strict: a single ``<speak>`` root containing
text, ``<break>`` and ``<phoneme>`` elements is accepted, and the parser
produces the plain text sent to the normal TTS route plus source-character
spans for the two operations we currently implement: timed breaks and IPA
pronunciation overrides.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from typing import TypeAlias

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

    def add_break(self, time: str | None) -> None:
        self.operations.append(
            BreakOperation(
                position=self.position,
                duration_seconds=_parse_break_time(time if time is not None else "1s"),
            )
        )
        # A break is a boundary in the decoded text stream. Treating it as
        # whitespace lets the one normalization pass handle punctuation,
        # adjacent breaks, indentation, and languages without spaces alike.
        self.append_text(" ")

    def open_phoneme(self, alphabet: str | None, phonemes: str | None) -> int:
        if alphabet is None or alphabet.strip().lower() != "ipa":
            raise ValueError("SSML <phoneme> currently requires alphabet='ipa'")
        if phonemes is None or not phonemes.strip():
            raise ValueError("SSML <phoneme> requires a non-empty 'ph' attribute")
        return self.position

    def close_phoneme(self, start: int, phonemes: str) -> None:
        if start == self.position:
            raise ValueError("SSML <phoneme> requires non-empty display text")
        self.operations.append(
            PronunciationOperation(
                start=start,
                end=self.position,
                alphabet="ipa",
                phonemes=phonemes.strip(),
            )
        )

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


class _SSMLParser(HTMLParser):
    """Lenient-text, strict-tag SSML front end built on the HTML parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.builder = _DocumentBuilder()
        self._root_open = False
        self._root_closed = False
        self._phonemes: list[tuple[int, str]] = []

    def _require_root(self, tag: str) -> None:
        if self._root_open:
            return
        if self._root_closed:
            raise ValueError("SSML must contain exactly one <speak> root element")
        if tag != "speak":
            raise ValueError("SSML root element must be <speak>")
        self._root_open = True

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._require_root(tag)
        if tag == "speak":
            return
        if tag == "break":
            unsupported = {name for name, _ in attrs} - {"time"}
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"Unsupported SSML <break> attribute(s): {names}")
            self.builder.add_break(dict(attrs).get("time"))
            return
        if tag == "phoneme":
            unsupported = {name for name, _ in attrs} - {"alphabet", "ph"}
            if unsupported:
                names = ", ".join(sorted(unsupported))
                raise ValueError(f"Unsupported SSML <phoneme> attribute(s): {names}")
            values = dict(attrs)
            start = self.builder.open_phoneme(values.get("alphabet"), values.get("ph"))
            self._phonemes.append((start, values.get("ph") or ""))
            return
        raise ValueError(f"Unsupported SSML element <{tag}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "speak":
            if not self._root_open:
                raise ValueError("SSML root element must be <speak>")
            self._root_open = False
            self._root_closed = True
            return
        if not self._root_open:
            raise ValueError("SSML root element must be <speak>")
        if tag == "break":
            return
        if tag == "phoneme":
            if not self._phonemes:
                raise ValueError("SSML </phoneme> has no matching <phoneme> start tag")
            start, phonemes = self._phonemes.pop()
            self.builder.close_phoneme(start, phonemes)
            return
        raise ValueError(f"Unsupported SSML element <{tag}>")

    def handle_data(self, data: str) -> None:
        if not self._root_open:
            if data.strip():
                raise ValueError("SSML root element must be <speak>")
            return
        self.builder.append_text(data)

    def handle_decl(self, decl: str) -> None:
        raise ValueError("Invalid SSML: DOCTYPE declarations are not allowed")


def parse_ssml(ssml: str) -> SSMLDocument:
    """Parse an SSML document into decoded text and post-processing operations.

    Character references (``&amp;``, ``&#x26;``) are decoded by the parser.
    Text that is not valid XML, such as a bare ``&``, is accepted as-is.
    """
    if not isinstance(ssml, str) or not ssml.strip():
        raise ValueError("SSML input is empty")
    parser = _SSMLParser()
    parser.feed(ssml)
    parser.close()
    if not parser._root_closed:
        raise ValueError("SSML must end with </speak>")
    return parser.builder.finish()


__all__ = [
    "BreakOperation",
    "PronunciationOperation",
    "SSMLDocument",
    "SSMLOperation",
    "parse_ssml",
]
