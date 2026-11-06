"""IPA as the symbols Sparrow's phoneme map spells them."""

import unicodedata
from functools import lru_cache

from piper_phonemize import get_espeak_map

# Common IPA spellings with no Unicode decomposition that Sparrow writes another way.
_SPELLINGS = {
    "\u025d": "\u025c\u02de",  # r-coloured open-mid vowel
    "\u02a7": "t\u0283",  # tesh ligature
    "\u02a4": "d\u0292",  # dezh ligature
    "\u02a6": "ts",  # ts ligature
    "\u02a3": "dz",  # dz ligature
    "\u0361": "",  # tie bar above
    "\u035c": "",  # tie bar below
}


@lru_cache(maxsize=1)
def _sparrow_symbols() -> frozenset[str]:
    return frozenset(get_espeak_map())


def sparrow_ipa_symbols(ipa: str) -> list[str]:
    """Split IPA into Sparrow's symbols, in the form its map holds each one.

    The map holds ``ç`` precomposed but nasal vowels as a letter plus a combining
    tilde, so a symbol missing from the map is tried decomposed (``ã`` -> ``a`` +
    U+0303) and through ``_SPELLINGS``. What still misses is returned as is, for
    the caller to report.
    """
    symbols = _sparrow_symbols()
    out: list[str] = []
    for char in unicodedata.normalize("NFC", ipa):
        if char in symbols:
            out.append(char)
            continue
        for part in _SPELLINGS.get(char, unicodedata.normalize("NFD", char)):
            out.extend(_SPELLINGS.get(part, part))
    return out
