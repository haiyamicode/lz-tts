"""Korean text normalization before the repo's Korean frontend.

This intentionally preserves written Korean. G2P and span mapping live in
src.piper.preprocess.
"""

from __future__ import annotations

import re
import unicodedata

_REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
}
_PUNCT_PATTERN = re.compile("|".join(re.escape(p) for p in _REP_MAP))


def normalize_korean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return _PUNCT_PATTERN.sub(lambda match: _REP_MAP[match.group()], text)
