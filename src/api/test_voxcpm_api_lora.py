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
    _effective_voxcpm_lora_names,
    _prepare_voxcpm_ipa_text,
    _plan_synthesis_batches,
    _prepare_batchable_synthesis_request,
    _shared_batch_from_items,
    _validate_batch_item,
    _voice_request_routes_to_voxcpm,
)


def test_plain_speak_wrapper_is_prepared_for_batching() -> None:
    request = _prepare_batchable_synthesis_request(
        {
            "ssml": "<speak>Hello &amp; welcome.</speak>",
            "voice_id": "voice",
            "reference_url": "https://example.com/reference.wav",
        }
    )

    assert request is not None
    assert request.text == "Hello & welcome."
    assert request.ssml is None


def test_ssml_with_postprocessing_is_not_prepared_for_batching() -> None:
    request = _prepare_batchable_synthesis_request(
        {
            "ssml": '<speak>Hello<break time="1s"/>world.</speak>',
            "voice_id": "voice",
            "reference_url": "https://example.com/reference.wav",
        }
    )

    assert request is None


def test_voxcpm_release_defaults_use_stabilized_model_and_three_slots() -> None:
    config = VoxCPMConfig()

    assert config.model_path == "data/voxcpm2-stable"
    assert config.default_locales == []
    assert config.locale_loras == {}
    assert config.max_concurrent_loras == 3
    assert config.max_loras_per_request == 2


def test_voxcpm_batch_grouping_is_adapter_order_independent(monkeypatch) -> None:
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "applicable_loras",
        {
            "accent-en-US": "unused",
            "accent-en-GB": "unused",
            "ipa": "unused",
        },
    )
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


def test_batch_planner_combines_eight_voxcpm_voices() -> None:
    items = [
        BatchSynthesizeInputItem(
            text=f"Distinct synthesis request number {index}.",
            model="voxcpm",
            voice_id=f"voice.{index}",
            reference_url=f"https://example.com/reference-{index}.wav",
            language="zh-CN",
        )
        for index in range(8)
    ]

    plans = _plan_synthesis_batches(items)

    assert [(plan.pipeline, len(plan.records)) for plan in plans] == [("voxcpm", 8)]


def test_batch_planner_orders_backends_and_splits_incompatible_loras() -> None:
    voxcpm = BatchSynthesizeInputItem(
        text="VoxCPM",
        model="voxcpm",
        voice_id="voice.voxcpm",
        reference_url="https://example.com/voxcpm.wav",
        language="zh-CN",
    )
    different_lora = voxcpm.model_copy(
        update={
            "text": "Different adapter",
            "voxcpm_loras": ["accent-en-US"],
            "language": "en-US",
        }
    )
    sparrow = BatchSynthesizeInputItem(
        text="Sparrow",
        model="sparrow",
        language="bs-BA",
        language_override=True,
    )

    plans = _plan_synthesis_batches([voxcpm, different_lora, sparrow])

    assert [plan.pipeline for plan in plans] == [
        "sparrow_forced_language",
        "voxcpm",
        "voxcpm",
    ]


def test_reference_voice_routing_uses_model_capabilities() -> None:
    request = {
        "voice_id": "product.voice",
        "reference_url": "https://example.com/reference.wav",
    }

    assert _voice_request_routes_to_voxcpm(**request, language="en-US", model=None)
    assert not _voice_request_routes_to_voxcpm(**request, language="bs-BA", model=None)
    assert not _voice_request_routes_to_voxcpm(
        **request,
        language="en-US",
        model="sparrow",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="bs-BA",
        model="voxcpm",
    )

    explicit_voxcpm = BatchSynthesizeInputItem(
        text="Ovo je test.",
        model="voxcpm",
        language="bs-BA",
        language_override=True,
        **request,
    )
    assert _batch_item_pipeline(explicit_voxcpm) == "voxcpm"
    assert _shared_batch_from_items(
        [(0, explicit_voxcpm, explicit_voxcpm.text or "")]
    ).model == "voxcpm"


def test_locale_routing_respects_native_adapter_and_reference_accents(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "default_locales",
        ["zh-CN", "zh-HK", "yue-CN", "pt-PT"],
    )
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "locale_loras",
        {"en-US": "accent-en-US", "en-GB": "accent-en-GB"},
    )
    request = {
        "voice_id": "msa.en-US.AvaMultilingual",
        "reference_url": "https://example.com/ava.mp3",
        "model": None,
    }

    assert _voice_request_routes_to_voxcpm(
        **request,
        language="en-GB",
        reference_language="en-US",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="zh-CN",
        reference_language="en-US",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="zh-HK",
        reference_language="en-US",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="yue-CN",
        reference_language="en-US",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="pt-PT",
        reference_language="en-US",
    )
    assert not _voice_request_routes_to_voxcpm(
        **request,
        language="pt-BR",
        reference_language="en-US",
    )
    assert _voice_request_routes_to_voxcpm(
        **request,
        language="zh-HK",
        reference_language="zh-HK",
    )


def test_locale_adapter_is_only_applied_for_a_locale_override(monkeypatch) -> None:
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "applicable_loras",
        {"accent-en-US": "unused", "accent-en-GB": "unused"},
    )
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "locale_loras",
        {"en-US": "accent-en-US", "en-GB": "accent-en-GB"},
    )

    assert _effective_voxcpm_lora_names([], "en-US") == ()
    assert _effective_voxcpm_lora_names([], "en-GB", True) == (
        "accent-en-GB",
    )
    assert _effective_voxcpm_lora_names([], "en-US", True) == (
        "accent-en-US",
    )
    assert _effective_voxcpm_lora_names(["accent-en-US"], "en-US") == (
        "accent-en-US",
    )
    assert _effective_voxcpm_lora_names([], "en-GB") == ()
    assert _effective_voxcpm_lora_names([], "zh-CN", True) == ()


def test_batch_grouping_separates_native_and_overridden_accents(monkeypatch) -> None:
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "applicable_loras",
        {"accent-en-US": "unused", "accent-en-GB": "unused"},
    )
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "locale_loras",
        {"en-US": "accent-en-US", "en-GB": "accent-en-GB"},
    )
    native = BatchSynthesizeInputItem(
        text="Native American English.",
        model="voxcpm",
        voice_id="msa.en-US.AvaMultilingual",
        reference_url="https://example.com/ava.wav",
        reference_language="en-US",
        language="en-US",
    )
    overridden = native.model_copy(
        update={
            "text": "British accent override.",
            "language": "en-GB",
            "language_override": True,
        }
    )

    plans = _plan_synthesis_batches([native, overridden])

    assert [len(plan.records) for plan in plans] == [1, 1]


def test_native_british_voice_does_not_infer_an_accent_override(monkeypatch) -> None:
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "applicable_loras",
        {"accent-en-US": "unused", "accent-en-GB": "unused"},
    )
    monkeypatch.setattr(
        server._server_config.voxcpm,
        "locale_loras",
        {"en-US": "accent-en-US", "en-GB": "accent-en-GB"},
    )
    native = BatchSynthesizeInputItem(
        text="Native British English.",
        model="voxcpm",
        voice_id="msa.en-GB.AdaMultilingual",
        reference_url="https://example.com/ada.wav",
        reference_language="en-GB",
        language="en-GB",
    )

    assert _batch_item_compatibility_key(native) == _batch_item_compatibility_key(
        native.model_copy(update={"text": "Another native request."})
    )
    assert _effective_voxcpm_lora_names([], native.language, native.language_override) == ()


def test_root_voice_with_reference_skips_seed_vc_for_its_native_language(
    monkeypatch,
) -> None:
    root = RootVoiceConfig(
        voice_id="root",
        model="sparrow-root",
        speaker="bs-BA",
        languages=["bs-BA"],
    )
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
    native = supported.model_copy(update={"language": "bs-BA"})
    unsupported = supported.model_copy(update={"language": "as-IN"})

    assert _batch_item_pipeline(supported) == "voxcpm"
    assert _batch_item_pipeline(native) == "sparrow"
    assert _batch_item_pipeline(unsupported) == "sparrow_reference"
    native_shared = _shared_batch_from_items([(0, native, native.text or "")])
    assert native_shared.reference_url is None
    assert native_shared.language == "bs-BA"
    assert native_shared.languages == [None]

    native_override = native.model_copy(update={"language_override": True})
    assert _batch_item_pipeline(native_override) == "sparrow_forced_language"
    assert _shared_batch_from_items(
        [(0, native_override, native_override.text or "")]
    ).language == "bs-BA"
    shared = _shared_batch_from_items([(0, unsupported, unsupported.text or "")])
    assert shared.voice_id is None
    assert shared.reference_url == unsupported.reference_url

    native_without_language = native.model_copy(update={"language": None})
    assert _batch_item_pipeline(native_without_language) == "sparrow"
    assert _shared_batch_from_items(
        [(0, native_without_language, native_without_language.text or "")]
    ).reference_url is None


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
        document,
        language,
        language,
    )

    assert visible_text in controlled_text
    assert detected_language == language
    assert controls[0]["target_ipa"] == document.pronunciations[0].phonemes
