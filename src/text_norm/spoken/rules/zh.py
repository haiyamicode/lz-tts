"""Chinese-only spoken normalization rules."""

from __future__ import annotations

import re

from .base import RuleContext

_NUMERIC_TOKEN = r"[-−]?\d+(?:(?:[.,])\d+)*"
_CJK_SCRIPT = "\u3400-\u4dbf\u4e00-\u9fff"
_TOKEN_LEFT_BOUNDARY = rf"(?:(?<![\w/])|(?<=[{_CJK_SCRIPT}]))"
_TOKEN_RIGHT_BOUNDARY = rf"(?:(?![\w/])|(?=[{_CJK_SCRIPT}]))"
_PERCENT_RE = re.compile(
    rf"{_TOKEN_LEFT_BOUNDARY}(?P<number>{_NUMERIC_TOKEN})\s*%"
    rf"{_TOKEN_RIGHT_BOUNDARY}"
)


def apply(text: str, context: RuleContext) -> str:
    return _PERCENT_RE.sub(
        lambda match: f"百分之{context.spoken_number(match.group('number'))}",
        text,
    )
