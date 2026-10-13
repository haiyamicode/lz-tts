from __future__ import annotations

import pytest
from fastapi import HTTPException

from . import server


@pytest.fixture
def sparrow_routes(monkeypatch):
    monkeypatch.setattr(
        server,
        "_speaker_routes",
        {"es": ("sparrow", 1), "es-MX": ("sparrow", 2), "ar-SA": ("sparrow", 3)},
    )
    monkeypatch.setattr(
        server,
        "_lang_speaker_map",
        {"es-ES": "es", "ar": "ar-SA", "ar-SA": "ar-SA"},
    )


def test_regional_locale_without_speaker_falls_back_to_base(sparrow_routes) -> None:
    assert server._resolve_forced_language("es-CL") == ("es", "es", "sparrow")
    assert server._resolve_forced_language("ar-AE") == ("ar", "ar-SA", "sparrow")


def test_regional_locale_with_speaker_keeps_region(sparrow_routes) -> None:
    assert server._resolve_forced_language("es-MX") == ("es-MX", "es-MX", "sparrow")
    assert server._resolve_forced_language("es-ES") == ("es-ES", "es", "sparrow")


def test_unknown_language_is_rejected(sparrow_routes) -> None:
    with pytest.raises(HTTPException) as exc:
        server._resolve_forced_language("wuu-CN")
    assert exc.value.status_code == 400


def test_multilingual_plan_accepts_regional_fallback(sparrow_routes) -> None:
    plan, _ = server._plan_text_segments("Hola, ¿cómo estás?", None, forced_language="es-CL")
    assert [(seg["lang"], seg["speaker"]) for seg in plan] == [("es", "es")]
