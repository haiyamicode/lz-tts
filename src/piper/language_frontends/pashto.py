"""Yousafzai Pashto frontend backed by Epitran."""

from __future__ import annotations

from .base import FrontendResult, phonemize_phrases

_G2P = None
_ESPEAK_TRANSLATION = str.maketrans(
    {
        "\u0361": "",
        # The upstream map leaves these graphemes unchanged.
        "ږ": "ɡ",
        "ڼ": "ɳ",
        "ذ": "z",
        # Preserve numbers using symbols in Sparrow's existing table.
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)


def _get_g2p():
    global _G2P
    if _G2P is None:
        import epitran

        _G2P = epitran.Epitran("pbu-Arab")
    return _G2P


def phonemize(text: str) -> FrontendResult:
    g2p = _get_g2p()
    return phonemize_phrases(
        text,
        lambda fragment: g2p.transliterate(fragment).translate(
            _ESPEAK_TRANSLATION
        ),
    )
