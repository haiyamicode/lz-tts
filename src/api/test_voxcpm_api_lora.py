from __future__ import annotations

import pytest
from fastapi import HTTPException

from ..ssml import parse_ssml
from . import server
from .server import (
    BatchSynthesizeInputItem,
    RootVoiceConfig,
    VoxCPMConfig,
    _batch_item_pipeline,
    _batch_item_compatibility_key,
    _prepare_voxcpm_ipa_text,
    _shared_batch_from_items,
    _validate_batch_item,
    _voice_request_routes_to_voxcpm,
)


def test_voxcpm_release_defaults_use_stabilized_model_and_three_slots() -> None:
    config = VoxCPMConfig()

    assert config.model_path == "data/voxcpm2-stable"
    assert config.max_concurrent_loras == 3
    assert config.max_loras_per_request == 2


def test_voxcpm_batch_grouping_is_adapter_order_independent() -> None:
    first = BatchSynthesizeInputItem(
        text="first",
        model="voxcpm",
        voice_id="voice.first",
        reference_url="https://example.com/first.wav",
        voxcpm_loras=["accent-en-GB", "ipa"],
    )
    second = BatchSynthesizeInputItem(
        text="second",
        model="voxcpm",
        voice_id="voice.second",
        reference_url="https://example.com/second.wav",
        voxcpm_loras=["ipa", "accent-en-GB"],
    )
    different_adapter = BatchSynthesizeInputItem(
        text="third",
        model="voxcpm",
        voice_id="voice.third",
        reference_url="https://example.com/third.wav",
        voxcpm_loras=["accent-en-US"],
    )

    assert _batch_item_compatibility_key(first) == _batch_item_compatibility_key(second)
    assert _batch_item_compatibility_key(first) != _batch_item_compatibility_key(
        different_adapter
    )


def test_reference_voice_routing_uses_model_capabilities() -> None:
    request = {
        "voice_id": "product.voice",
        "reference_url": "https://example.com/reference.wav",
    }

    assert _voice_request_routes_to_voxcpm(**request, language="en-US", model=None)
    assert not _voice_request_routes_to_voxcpm(**request, language="bs-BA", model=None)
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="en-US",
        model="sparrow",
    )
    assert not _voice_request_routes_to_voxcpm(
        **request,
        language="bs-BA",
        model="voxcpm",
    )


def test_root_voice_with_reference_uses_voxcpm_or_seed_vc_by_language(
    monkeypatch,
) -> None:
    root = RootVoiceConfig(voice_id="root", model="sparrow-root")
    monkeypatch.setattr(
        server,
        "_configured_root_voice_for_voice_id",
        lambda voice_id: root if voice_id == "root" else None,
    )
    supported = BatchSynthesizeInputItem(
        text="hello",
        voice_id="root",
        reference_url="https://example.com/root.mp3",
        language="en-US",
    )
    unsupported = supported.model_copy(update={"language": "bs-BA"})

    assert _batch_item_pipeline(supported) == "voxcpm"
    assert _batch_item_pipeline(unsupported) == "sparrow_reference"
    shared = _shared_batch_from_items([(0, unsupported, unsupported.text or "")])
    assert shared.voice_id is None
    assert shared.reference_url == unsupported.reference_url


def test_explicit_voxcpm_requires_a_reference() -> None:
    item = BatchSynthesizeInputItem(text="hello", model="voxcpm", language="en-US")

    with pytest.raises(HTTPException, match="requires 'reference_url'"):
        _validate_batch_item(item, 0)


@pytest.mark.parametrize(
    ("language", "ssml", "visible_text"),
    [
        ("zh-CN", '<speak>请<phoneme alphabet="ipa" ph="ta˧˩˧kʰai̯˥">打开</phoneme>门。</speak>', "打开"),
        ("ja-JP", '<speak><phoneme alphabet="ipa" ph="to̞mato̞">トマト</phoneme>をください。</speak>', "トマト"),
        ("vi-VN", '<speak>Tôi thích <phoneme alphabet="ipa" ph="kaː fe˧˩">cà phê</phoneme>.</speak>', "cà phê"),
    ],
)
def test_non_english_voxcpm_ipa_keeps_native_visible_text_as_guide(
    language: str,
    ssml: str,
    visible_text: str,
) -> None:
    document = parse_ssml(ssml)

    controlled_text, detected_language, controls = _prepare_voxcpm_ipa_text(
        document, language
    )

    assert visible_text in controlled_text
    assert detected_language == language
    assert controls[0]["target_ipa"] == document.pronunciations[0].phonemes
