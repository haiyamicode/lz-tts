from __future__ import annotations

import asyncio

import numpy as np

from .voxcpm_runtime import VoxCPMRuntime


class _FakeServer:
    def __init__(self) -> None:
        self.encoded: list[tuple[bytes, str]] = []
        self.references: list[bytes | None] = []
        self.generation_limits: list[int] = []
        self.seeds: list[int | None] = []

    async def encode_latents(self, audio: bytes, audio_format: str) -> bytes:
        self.encoded.append((audio, audio_format))
        return b"reference-latents"

    async def generate(self, **kwargs):
        self.references.append(kwargs["ref_audio_latents"])
        self.generation_limits.append(kwargs["max_generate_length"])
        self.seeds.append(kwargs.get("seed"))
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
    assert server.references == [b"reference-latents"] * 3
    assert server.generation_limits == [17, 29, 17]
    assert server.seeds == [101, 202, None]
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
