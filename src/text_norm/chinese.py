"""Chinese text normalization without G2P."""

from __future__ import annotations

import re

import cn2an

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
    "...": "...",
    "$": ".",
    '"': "'",
    """: "'",
    """: "'",
    "'": "'",
    "'": "'",
    "（": "'",
    "）": "'",
    "(": "'",
    ")": "'",
    "《": "'",
    "》": "'",
    "【": "'",
    "】": "'",
    "[": "'",
    "]": "'",
    "—": "-",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}

_PUNCT_PATTERN = re.compile("|".join(re.escape(p) for p in _REP_MAP))
_NUMBER_PATTERN = re.compile(r"\d+(?:\.?\d+)?")


def _replace_punctuation(text: str) -> str:
    text = text.replace("嗯", "恩").replace("呣", "母")
    return _PUNCT_PATTERN.sub(lambda match: _REP_MAP[match.group()], text)


def normalize_chinese(text: str) -> str:
    for number in _NUMBER_PATTERN.findall(text):
        text = text.replace(number, cn2an.an2cn(number), 1)
    return _replace_punctuation(text)
