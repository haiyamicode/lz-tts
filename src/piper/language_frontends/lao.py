"""Lao frontend backed by lao2ipa."""

from __future__ import annotations

from .base import FrontendResult, phonemize_phrases

_ESPEAK_TRANSLATION = str.maketrans(
    {
        # Sparrow's existing eSpeak symbol table has no IPA tone letters.
        "˩": "",
        "˨": "",
        "˧": "",
        "˥": "",
        # Represent labialized clusters with the supported /w/.
        "ʷ": "w",
    }
)


def phonemize(text: str) -> FrontendResult:
    from lao2ipa import transliterate

    return phonemize_phrases(
        text,
        lambda fragment: transliterate(fragment, "ipa").translate(
            _ESPEAK_TRANSLATION
        ),
    )
