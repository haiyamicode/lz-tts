"""Written forms a TTS model would read wrong, rewritten as they are said.

Only the spoken text changes; callers that show or time the text keep the original.
"""

from __future__ import annotations

import functools
import re
import threading

_LETTER = r"[^\W\d_]"

# A slash between two words ("Tolstoy/Tolstoi", "and/or", "he / she"). Not part of a URL, a
# domain, a path or an e-mail ("example.com/about", "/usr/bin", "user@host/x"), not a unit
# ("km/h", "m/s": one-letter sides), and not between numbers (dates, fractions).
_WORD_SLASH_RE = re.compile(
    rf"(?<![\w/.:@\\]){_LETTER}{{2,}}(?:\s*/\s*{_LETTER}{{2,}})+(?![\w/@\\]|\.{_LETTER})"
)
_SLASH_RE = re.compile(r"\s*/\s*")
_ENDS_WITH_NUMBER_RE = re.compile(r"\d\s*$")


def speakable_word_slashes(text: str) -> str:
    """Read a slash between words as a short pause: "Tolstoy/Tolstoi" -> "Tolstoy, Tolstoi".

    A unit after a number ("120 mg/dL") keeps its slash: number reading says it as "per".
    """

    def replace(match: re.Match[str]) -> str:
        if _ENDS_WITH_NUMBER_RE.search(text, 0, match.start()):
            return match.group(0)
        return _SLASH_RE.sub(", ", match.group(0))

    return _WORD_SLASH_RE.sub(replace, text)


_LETTER_RUN_RE = re.compile(rf"{_LETTER}+")
_ESPEAK_LOCK = threading.Lock()
_ESPEAK_VOICES = {"en": "en-us", "fil": "tl", "nb": "nb", "yue": "yue", "wuu": "cmn"}


@functools.lru_cache(maxsize=4096)
def _spelled_out(word: str, language: str) -> bool:
    """Whether ``word`` (lowercase) is an acronym read letter by letter in ``language``.

    espeak's dictionaries know them: it reads such a word one stressed letter name at a
    time ("fbi" ˌɛfbˌiːˈaɪ, "sncf" ˌɛsˌɛnsˌeˈɛf), and every ordinary word with at most one
    stress per syllable ("peace" pˈiːs, "nasa" nˈæsɐ). A language espeak lacks is read as English.
    """
    from piper_phonemize import phonemize_espeak  # pylint: disable=import-outside-toplevel

    base = (language or "en").lower().replace("_", "-").split("-", 1)[0]
    phonemes = ""
    for voice in dict.fromkeys((_ESPEAK_VOICES.get(base, base), "en-us")):
        try:
            with _ESPEAK_LOCK:
                phonemes = "".join("".join(sentence) for sentence in phonemize_espeak(word, voice))
            break
        except RuntimeError:
            continue
    return sum(phonemes.count(mark) for mark in "ˈˌ") >= len(word)


def speakable_caps_runs(text: str, language: str | None) -> str:
    """Read a run of all-caps words as words: "WAR AND PEACE" -> "War And Peace".

    A run is two or more all-caps words of two or more letters next to each other (a title,
    shouting). Its words become title case, except acronyms read letter by letter (FBI, USA,
    CEO stay). A lone all-caps word (NASA, US, "I said NO") is left as written.
    """
    words = [match for match in _LETTER_RUN_RE.finditer(text)]
    runs: list[list[re.Match[str]]] = [[]]
    for word in words:
        if word.group(0).isupper():
            runs[-1].append(word)
        elif runs[-1]:
            runs.append([])
    replacements: list[tuple[int, int, str]] = []
    for run in runs:
        if sum(len(word.group(0)) >= 2 for word in run) < 2:
            continue
        for word in run:
            token = word.group(0)
            if len(token) < 2:
                # A letter after an apostrophe ("DON'T") belongs to the word before it.
                if word.start() and text[word.start() - 1] in "'’":
                    replacements.append((word.start(), word.end(), token.lower()))
                continue
            if not _spelled_out(token.lower(), language or "en"):
                replacements.append((word.start(), word.end(), token[0] + token[1:].lower()))
    for start, end, replacement in reversed(replacements):
        text = text[:start] + replacement + text[end:]
    return text
