"""Thai frontend backed by Epitran."""

from __future__ import annotations

from .base import FrontendResult, phonemize_phrases

_G2P = None
_ESPEAK_TRANSLATION = str.maketrans({"\u0361": ""})


def _get_g2p():
    global _G2P
    if _G2P is None:
        import epitran

        _G2P = epitran.Epitran("tha-Thai")
    return _G2P


def phonemize(text: str) -> FrontendResult:
    g2p = _get_g2p()
    return phonemize_phrases(
        text,
        lambda fragment: g2p.transliterate(fragment).translate(
            _ESPEAK_TRANSLATION
        ),
    )
