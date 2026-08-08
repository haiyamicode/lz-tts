from __future__ import annotations

from ..ssml import PronunciationOperation
from . import server
from .server import SynthesizeRequest


def test_custom_pronunciation_uses_resolved_internal_model(monkeypatch) -> None:
    """Legacy query/GET model selection must also control the Sparrow render."""
    monkeypatch.setattr(server, "_language_at_source_position", lambda *_args: "en-US")
    monkeypatch.setattr(
        server,
        "_resolve_forced_language",
        lambda _language: ("en-US", "en-US", "sparrow-default"),
    )
    monkeypatch.setattr(server, "_configured_root_voice_for_voice_id", lambda _voice_id: None)

    route = server._ssml_sparrow_route(
        SynthesizeRequest(ssml="<speak>hello</speak>"),
        "hello",
        PronunciationOperation(0, 5, "ipa", "həˈləʊ"),
        "sparrow-internal",
    )

    assert route == (None, "sparrow-internal")
