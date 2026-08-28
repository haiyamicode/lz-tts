from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import numpy as np

from .voxcpm_runtime import VoxCPMRuntime


class _FakeServer:
    def __init__(self) -> None:
        self.encoded: list[tuple[bytes, str]] = []
        self.references: list[bytes | None] = []
        self.generation_limits: list[int] = []
        self.seeds: list[int | None] = []
        self.lora_names: list[str | None] = []

    async def encode_latents(
        self,
        audio: bytes,
        audio_format: str,
        max_reference_seconds: float,
    ) -> bytes:
        assert max_reference_seconds == 25.0
        self.encoded.append((audio, audio_format))
        return b"latents:" + audio

    async def generate(self, **kwargs):
        self.references.append(kwargs["ref_audio_latents"])
        self.generation_limits.append(kwargs["max_generate_length"])
        self.seeds.append(kwargs.get("seed"))
        self.lora_names.append(kwargs.get("lora_name"))
        yield np.ones(16, dtype=np.float32)


class _FakeDurationBudget:
    def __init__(self) -> None:
        self.calls = []

    def predict_batch(self, texts, languages):
        self.calls.append((list(texts), list(languages)))
        return [
            {
                "seconds": index + 1.0,
                "estimated_tokens": 10 + index,
                "max_tokens": 17 + index * 12,
            }
            for index, _text in enumerate(texts)
        ]


def _settings() -> dict:
    return {
        "fallback_max_generate_length": 4096,
        "duration_budget": {
            "enabled": True,
            "preload": False,
        },
        "temperature": 1.0,
        "cfg_value": 2.0,
        "reference_cache_size": 8,
        "max_reference_seconds": 25.0,
    }


def test_reference_audio_uses_native_voxcpm_latents_and_cache() -> None:
    asyncio.run(_test_reference_audio_uses_native_voxcpm_latents_and_cache())


async def _test_reference_audio_uses_native_voxcpm_latents_and_cache() -> None:
    runtime = VoxCPMRuntime(_settings())
    duration_budget = _FakeDurationBudget()
    runtime._duration_budget = duration_budget
    server = _FakeServer()
    runtime.server = server

    first = await runtime.synthesize_batch(
        ["first", "second"],
        languages=["en", "vi"],
        seeds=[101, 202],
        reference_audio=b"wav-data",
        reference_format="wav",
    )
    await runtime.synthesize_batch(
        ["third"],
        reference_audio=b"wav-data",
        reference_format="wav",
    )

    assert len(first) == 2
    assert server.encoded == [(b"wav-data", "wav")]
    assert server.references == [b"latents:wav-data"] * 3
    assert server.generation_limits == [17, 29, 17]
    assert server.seeds == [101, 202, None]
    assert server.lora_names == [None, None, None]
    assert duration_budget.calls == [
        (["first", "second"], ["en", "vi"]),
        (["third"], [None]),
    ]


def test_disabled_duration_budget_uses_explicit_fallback() -> None:
    asyncio.run(_test_disabled_duration_budget_uses_explicit_fallback())


async def _test_disabled_duration_budget_uses_explicit_fallback() -> None:
    settings = _settings()
    settings["duration_budget"]["enabled"] = False
    settings["fallback_max_generate_length"] = 321
    runtime = VoxCPMRuntime(settings)
    server = _FakeServer()
    runtime.server = server

    await runtime.synthesize_batch(["one", "two"], languages=["en", "de"])

    assert server.generation_limits == [321, 321]


def test_per_item_references_are_encoded_once_and_routed_independently() -> None:
    asyncio.run(_test_per_item_references_are_encoded_once_and_routed_independently())


async def _test_per_item_references_are_encoded_once_and_routed_independently() -> None:
    runtime = VoxCPMRuntime(_settings())
    runtime._duration_budget = _FakeDurationBudget()
    server = _FakeServer()
    runtime.server = server

    await runtime.synthesize_batch(
        ["first", "second", "third"],
        reference_audios=[b"voice-a", b"voice-b", b"voice-a"],
        reference_formats=["wav", "mp3", "wav"],
    )

    assert server.encoded == [
        (b"voice-a", "wav"),
        (b"voice-b", "mp3"),
    ]
    assert server.references == [
        b"latents:voice-a",
        b"latents:voice-b",
        b"latents:voice-a",
    ]


def test_lora_names_are_forwarded_per_item() -> None:
    asyncio.run(_test_lora_names_are_forwarded_per_item())


async def _test_lora_names_are_forwarded_per_item() -> None:
    runtime = VoxCPMRuntime(_settings())
    runtime._duration_budget = _FakeDurationBudget()
    runtime._available_loras = {"accent-en-GB", "accent-en-US"}
    server = _FakeServer()
    runtime.server = server

    await runtime.synthesize_batch(
        ["first", "second"],
        lora_names=["accent-en-GB", "accent-en-US"],
    )

    assert server.lora_names == ["accent-en-GB", "accent-en-US"]


def test_runtime_canonicalizes_fused_qkv_target_order() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint = Path(temporary_directory) / "accent"
        checkpoint.mkdir()
        (checkpoint / "lora_config.json").write_text(
            json.dumps(
                {
                    "lora_config": {
                        "enable_lm": True,
                        "enable_dit": False,
                        "enable_proj": False,
                        "r": 64,
                        "alpha": 64,
                        "target_modules_lm": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    }
                }
            ),
            encoding="utf-8",
        )
        (checkpoint / "lora_weights.safetensors").touch()
        settings = _settings()
        settings["applicable_loras"] = {"accent": str(checkpoint)}
        settings["max_concurrent_loras"] = 3
        settings["max_loras_per_request"] = 2

        config = VoxCPMRuntime(settings)._load_lora_runtime_config()

    assert config.target_modules_lm == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert config.max_loras == 3


def test_runtime_reserves_rank_only_for_the_per_request_composition_limit() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        configured = {}
        for index, rank in enumerate((16, 32, 64)):
            checkpoint = Path(temporary_directory) / f"adapter-{index}"
            checkpoint.mkdir()
            (checkpoint / "lora_config.json").write_text(
                json.dumps(
                    {
                        "lora_config": {
                            "enable_lm": True,
                            "r": rank,
                            "alpha": rank,
                            "target_modules_lm": ["q_proj"],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (checkpoint / "lora_weights.safetensors").touch()
            configured[f"adapter-{index}"] = str(checkpoint)

        settings = _settings()
        settings["applicable_loras"] = configured
        settings["max_concurrent_loras"] = 3
        settings["max_loras_per_request"] = 2
        config = VoxCPMRuntime(settings)._load_lora_runtime_config()

    assert config.max_lora_rank == 96
