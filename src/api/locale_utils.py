from __future__ import annotations


def normalize_locale(locale: str) -> str:
    """Return the primary language and, when present, the BCP-47 region.

    Script and variant subtags are intentionally discarded because the TTS
    routing configuration is keyed by language/region pairs. A script subtag
    must never be mistaken for a region (for example, ``zh-Hant`` is ``zh``,
    while ``zh-Hant-HK`` is ``zh-HK``).
    """
    parts = [part for part in locale.strip().replace("_", "-").split("-") if part]
    if not parts:
        return ""

    language = parts[0].lower()
    region = next(
        (
            part.upper()
            for part in parts[1:]
            if (len(part) == 2 and part.isalpha())
            or (len(part) == 3 and part.isdigit())
        ),
        None,
    )
    return f"{language}-{region}" if region else language
