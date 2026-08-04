"""Thai written-text normalization."""

from __future__ import annotations


def normalize_thai(text: str) -> str:
    """Compose decomposed SARA AM spellings expected by the Thai frontend."""
    return text.replace("\u0e4d\u0e32", "\u0e33")
