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


def test_root_voice_pronunciation_uses_language_speaker_when_not_fixed(monkeypatch) -> None:
    root_voice = server.RootVoiceConfig(
        voice_id="root",
        model="sparrow-root",
    )
    monkeypatch.setattr(server, "_language_at_source_position", lambda *_args: "zh-CN")
    monkeypatch.setattr(
        server,
        "_resolve_forced_language",
        lambda _language: ("zh-CN", "zh", "sparrow-default"),
    )
    monkeypatch.setattr(
        server,
        "_configured_root_voice_for_voice_id",
        lambda _voice_id: root_voice,
    )

    route = server._ssml_sparrow_route(
        SynthesizeRequest(ssml="<speak>你好</speak>", voice_id="root"),
        "你好",
        PronunciationOperation(0, 2, "ipa", "ni xau"),
        None,
    )

    assert route == ("zh", "sparrow-root")
