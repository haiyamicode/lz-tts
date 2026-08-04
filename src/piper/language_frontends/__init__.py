"""Registry for language frontends that do not use eSpeak directly."""

from __future__ import annotations

from .base import LanguageFrontend
from .khmer import phonemize as phonemize_khmer
from .lao import phonemize as phonemize_lao
from .mongolian import phonemize as phonemize_mongolian
from .myanmar import phonemize as phonemize_myanmar
from .pashto import phonemize as phonemize_pashto
from .thai import phonemize as phonemize_thai

LANGUAGE_FRONTENDS: dict[str, LanguageFrontend] = {
    "km": phonemize_khmer,
    "lo": phonemize_lao,
    "mn": phonemize_mongolian,
    "my": phonemize_myanmar,
    "ps": phonemize_pashto,
    "th": phonemize_thai,
}


def get_language_frontend(voice: str) -> LanguageFrontend | None:
    return LANGUAGE_FRONTENDS.get((voice or "").lower())


def has_language_frontend(voice: str) -> bool:
    return (voice or "").lower() in LANGUAGE_FRONTENDS


__all__ = ["get_language_frontend", "has_language_frontend"]
