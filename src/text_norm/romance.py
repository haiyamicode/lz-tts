"""French and Spanish text normalization without G2P."""

from __future__ import annotations

import re

_WHITESPACE_RE = re.compile(r"\s+")

_FRENCH_ABBREVIATIONS = [
    (re.compile(r"\b%s\." % key, re.IGNORECASE), value)
    for key, value in [
        ("M", "monsieur"),
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
        ("N.B", "nota bene"),
        ("M", "monsieur"),
        ("p.c.q", "parce que"),
        ("Pr", "professeur"),
        ("qqch", "quelque chose"),
        ("rdv", "rendez-vous"),
        ("max", "maximum"),
        ("min", "minimum"),
        ("no", "numéro"),
        ("adr", "adresse"),
        ("dr", "docteur"),
        ("st", "saint"),
        ("co", "companie"),
        ("jr", "junior"),
        ("sgt", "sergent"),
        ("capt", "capitain"),
        ("col", "colonel"),
        ("av", "avenue"),
        ("av. J.-C", "avant Jésus-Christ"),
        ("apr. J.-C", "après Jésus-Christ"),
        ("art", "article"),
        ("boul", "boulevard"),
        ("c.-à-d", "c'est-à-dire"),
        ("etc", "et cetera"),
        ("ex", "exemple"),
        ("excl", "exclusivement"),
        ("boul", "boulevard"),
    ]
] + [
    (re.compile(r"\b%s" % key), value)
    for key, value in [
        ("Mlle", "mademoiselle"),
        ("Mlles", "mesdemoiselles"),
        ("Mme", "Madame"),
        ("Mmes", "Mesdames"),
    ]
]

_FRENCH_REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "...": ".",
    "$": ".",
    '"': "",
    """: "",
    """: "",
    "'": "",
    "'": "",
    "（": "",
    "）": "",
    "(": "",
    ")": "",
    "《": "",
    "》": "",
    "【": "",
    "】": "",
    "[": "",
    "]": "",
    "—": "",
    "～": "-",
    "~": "-",
    "「": "",
    "」": "",
    "¿": "",
    "¡": "",
}

_SPANISH_REP_MAP = {
    "：": ",",
    "；": ",",
    "，": ",",
    "。": ".",
    "！": "!",
    "？": "?",
    "\n": ".",
    "·": ",",
    "、": ",",
    "...": ".",
    "...": ".",
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
    "—": "",
    "～": "-",
    "~": "-",
    "「": "'",
    "」": "'",
}


def _replace_punctuation(text: str, replacements: dict[str, str]) -> str:
    pattern = re.compile("|".join(re.escape(p) for p in replacements))
    return pattern.sub(lambda match: replacements[match.group()], text)


def _expand_french_abbreviations(text: str) -> str:
    for regex, replacement in _FRENCH_ABBREVIATIONS:
        text = re.sub(regex, replacement, text)
    return text


def _collapse_whitespace(text: str) -> str:
    return re.sub(_WHITESPACE_RE, " ", text).strip()


def _remove_punctuation_at_begin(text: str) -> str:
    return re.sub(r"^[,.!?]+", "", text)


def _remove_aux_symbols(text: str, keep_apostrophe: bool) -> str:
    chars = r"[\<\>\(\)\[\]\"\«\»]+"
    if not keep_apostrophe:
        chars = r"[\<\>\(\)\[\]\"\«\»']+"
    return re.sub(chars, "", text)


def _replace_symbols(text: str, lang: str) -> str:
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    if lang == "fr":
        return text.replace("&", " et ")
    if lang == "es":
        return text.replace("&", "y").replace("'", "")
    return text


def _ensure_terminal_punctuation(text: str) -> str:
    return re.sub(r"([^.,!?\-...])$", r"\1.", text)


def normalize_french(text: str) -> str:
    text = _expand_french_abbreviations(text)
    text = _replace_punctuation(text, _FRENCH_REP_MAP)
    text = _replace_symbols(text, "fr")
    text = _remove_aux_symbols(text, keep_apostrophe=True)
    text = _remove_punctuation_at_begin(text)
    text = _collapse_whitespace(text)
    return _ensure_terminal_punctuation(text)


def normalize_spanish(text: str) -> str:
    text = text.lower()
    text = _replace_symbols(text, "es")
    text = _replace_punctuation(text, _SPANISH_REP_MAP)
    text = _remove_aux_symbols(text, keep_apostrophe=False)
    text = _remove_punctuation_at_begin(text)
    text = _collapse_whitespace(text)
    return _ensure_terminal_punctuation(text)
