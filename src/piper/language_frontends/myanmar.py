"""Myanmar frontend backed by Epitran."""

from __future__ import annotations

from .base import FrontendResult, phonemize_phrases

_G2P = None
_ESPEAK_TRANSLATION = str.maketrans(
    {
        # Remove orthographic marks occasionally leaked by Epitran.
        "\u1039": "",
        "\u1037": "",
        "\u1038": "",
        "\u103e": "h",
        "\u0361": "",
        "\u0325": "",
        "\u104a": ",",
        "\u104b": ".",
    }
)


def _get_g2p():
    global _G2P
    if _G2P is None:
        import epitran

        _G2P = epitran.Epitran("mya-Mymr")
    return _G2P


def _normalize_ipa(ipa: str) -> str:
    for source, target in (
        ("w\u103e", "hw"),
        ("j\u103e", "hj"),
        ("m\u0325", "hm"),
        ("n\u0325", "hn"),
        ("\u0272\u0325", "h\u0272"),
        ("\u014b\u0325", "h\u014b"),
        ("l\u0325", "hl"),
        ("r\u0325", "hr"),
        ("w\u0325", "hw"),
        ("j\u0325", "hj"),
    ):
        ipa = ipa.replace(source, target)
    return ipa.translate(_ESPEAK_TRANSLATION)


def _normalize_g2p_input(text: str) -> str:
    """Use the U+102C spelling covered by Epitran's Burmese map."""
    return text.replace("\u102b", "\u102c")


def phonemize(text: str) -> FrontendResult:
    g2p = _get_g2p()
    return phonemize_phrases(
        text,
        lambda fragment: _normalize_ipa(
            g2p.transliterate(_normalize_g2p_input(fragment))
        ),
    )
