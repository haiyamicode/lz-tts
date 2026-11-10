"""The caller's lipsync prompt template (``/jobs/lipsync``, field ``prompt``).

lipsync renders the audio in chunks of up to 10 s, and each chunk's prompt
must say the words spoken in it, or the lips do not follow the speech. A
custom prompt is therefore a template rendered once per chunk:

- ``{{text}}`` (required, at least once): the words spoken in the chunk;
- ``{{language}}`` (optional): the English name of the spoken language, for
  example ``Vietnamese``, or empty when it is unknown (a caller transcript).

Whitespace inside the braces is allowed (``{{ text }}``). Any other
``{{name}}`` is an error, so a misspelt variable never reaches the model as
literal text; single braces are ordinary text.
"""

from __future__ import annotations

import re

TEXT_VARIABLE = "text"
LANGUAGE_VARIABLE = "language"
PROMPT_TEMPLATE_VARIABLES = (TEXT_VARIABLE, LANGUAGE_VARIABLE)
MAX_PROMPT_TEMPLATE_CHARS = 6000
_PLACEHOLDER = re.compile(r"\{\{\s*([^{}]*?)\s*\}\}")


def prompt_template_error(template: str) -> str | None:
    """Why ``template`` is not a valid lipsync prompt template, or ``None``."""
    if len(template) > MAX_PROMPT_TEMPLATE_CHARS:
        return f"prompt is {len(template)} characters, over the {MAX_PROMPT_TEMPLATE_CHARS}-character limit"
    names = [match.group(1) for match in _PLACEHOLDER.finditer(template)]
    unknown = sorted({name for name in names if name not in PROMPT_TEMPLATE_VARIABLES})
    if unknown:
        return (
            "prompt has unknown template variables "
            + ", ".join("{{" + name + "}}" for name in unknown)
            + "; the variables are {{text}} (required) and {{language}}"
        )
    if TEXT_VARIABLE not in names:
        return (
            "prompt must contain {{text}}, where each chunk's spoken words go: "
            "without them the lips do not follow the speech"
        )
    return None


def render_prompt_template(template: str, *, text: str, language: str | None) -> str:
    """``template`` with one chunk's words (whitespace collapsed) and the language name filled in."""
    error = prompt_template_error(template)
    if error is not None:
        raise ValueError(error)
    values = {TEXT_VARIABLE: " ".join(text.split()), LANGUAGE_VARIABLE: language or ""}
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
