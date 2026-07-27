"""Optimized nano-vLLM VoxCPM2 serving runtime."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


_LOGGER = logging.getLogger(__name__)
_KV_BLOCKS_ENV = "NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"


class VoxCPMRuntime:
    """Own one async nano-vLLM server pool and expose batched synthesis."""

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.server: Any | None = None
        self.sample_rate = 16000
        self._reference_latents: OrderedDict[str, bytes] = OrderedDict()
        self._reference_cache_lock = asyncio.Lock()
        self._duration_budget: Any | None = None
        self._duration_budget_init_lock = threading.Lock()
        self._duration_budget_predict_lock = asyncio.Lock()

    async def start(self) -> None:
        if self.server is not None:
            return

        from nanovllm_voxcpm import VoxCPM

        num_kvcache_blocks = int(self.settings["num_kvcache_blocks"])
        if num_kvcache_blocks <= 0:
            raise ValueError("voxcpm.num_kvcache_blocks must be greater than zero")
        os.environ[_KV_BLOCKS_ENV] = str(num_kvcache_blocks)

        self.server = VoxCPM.from_pretrained(
            model=str(self.settings["model_path"]),
            devices=[int(self.settings["device"])],
            inference_timesteps=int(self.settings["inference_timesteps"]),
            max_num_batched_tokens=int(self.settings["max_num_batched_tokens"]),
            max_num_seqs=int(self.settings["max_num_seqs"]),
            max_model_len=int(self.settings["max_model_len"]),
            gpu_memory_utilization=float(self.settings["gpu_memory_utilization"]),
            enforce_eager=bool(self.settings["enforce_eager"]),
        )
        await self.server.wait_for_ready()
        model_info = await self.server.get_model_info()
        self.sample_rate = int(model_info["sample_rate"])
        duration_settings = self.settings["duration_budget"]
        if duration_settings["enabled"] and duration_settings["preload"]:
            await asyncio.to_thread(self._get_duration_budget)
        _LOGGER.info(
            "VoxCPM nano-vLLM ready model=%s device=%s sample_rate=%d "
            "kv_blocks=%d max_num_seqs=%d",
            self.settings["model_path"],
            self.settings["device"],
            self.sample_rate,
            num_kvcache_blocks,
            self.settings["max_num_seqs"],
        )

    async def stop(self) -> None:
        server = self.server
        self.server = None
        self._duration_budget = None
        self._reference_latents.clear()
        if server is not None:
            await server.stop()

    def _get_duration_budget(self) -> Any:
        if self._duration_budget is not None:
            return self._duration_budget
        with self._duration_budget_init_lock:
            if self._duration_budget is not None:
                return self._duration_budget

            from src.duration_alignment import DpBudgetConfig, DurationAlignmentValidator

            settings = self.settings["duration_budget"]
            config_path = settings.get("config_path")
            validator = DurationAlignmentValidator(
                DpBudgetConfig(
                    checkpoint=Path(settings["checkpoint"]),
                    config_path=Path(config_path) if config_path else None,
                    device=str(settings["device"]),
                    language=str(settings["language"]),
                    noise_scale=float(settings["noise_scale"]),
                    length_scale=float(settings["length_scale"]),
                    token_rate=float(settings["token_rate"]),
                    samples=int(settings["samples"]),
                    upper_quantile=float(settings["upper_quantile"]),
                    min_margin=float(settings["min_margin"]),
                    max_margin=float(settings["max_margin"]),
                    min_extra_tokens=int(settings["min_extra_tokens"]),
                    max_extra_tokens=int(settings["max_extra_tokens"]),
                    language_profiles=dict(settings["language_profiles"]),
                    use_bert=bool(settings["use_bert"]),
                    enable_alignment_validation=False,
                )
            )
            validator.load()
            self._duration_budget = validator
            _LOGGER.info(
                "VoxCPM DP budget ready checkpoint=%s device=%s token_rate=%.2f "
                "max_margin=%.2f max_extra_tokens=%d",
                settings["checkpoint"],
                settings["device"],
                settings["token_rate"],
                settings["max_margin"],
                settings["max_extra_tokens"],
            )
            return validator

    async def _predict_generation_limits(
        self,
        texts: Sequence[str],
        languages: Sequence[str | None],
    ) -> tuple[list[int], list[dict[str, Any] | None]]:
        settings = self.settings["duration_budget"]
        if not settings["enabled"]:
            fallback = int(self.settings["fallback_max_generate_length"])
            return [fallback] * len(texts), [None] * len(texts)

        async with self._duration_budget_predict_lock:
            validator = await asyncio.to_thread(self._get_duration_budget)
            budgets = await asyncio.to_thread(
                validator.predict_batch,
                list(texts),
                list(languages),
            )
        limits = [int(budget["max_tokens"]) for budget in budgets]
        _LOGGER.info(
            "VoxCPM DP generation budgets: %s",
            [
                {
                    "language": language,
                    "expected_seconds": round(float(budget["seconds"]), 3),
                    "estimated_steps": int(budget["estimated_tokens"]),
                    "max_generate_length": limit,
                }
                for language, budget, limit in zip(languages, budgets, limits)
            ],
        )
        return limits, list(budgets)

    async def _encode_reference(self, audio: bytes, audio_format: str) -> bytes:
        if self.server is None:
            raise RuntimeError("VoxCPM runtime is not started")

        cache_key = hashlib.sha256(audio).hexdigest()
        async with self._reference_cache_lock:
            cached = self._reference_latents.get(cache_key)
            if cached is not None:
                self._reference_latents.move_to_end(cache_key)
                return cached

            latents = await self.server.encode_latents(audio, audio_format)
            self._reference_latents[cache_key] = latents
            cache_size = int(self.settings["reference_cache_size"])
            while len(self._reference_latents) > cache_size:
                self._reference_latents.popitem(last=False)
            return latents

    async def _synthesize_one(
        self,
        text: str,
        *,
        max_generate_length: int,
        seed: int | None = None,
        ref_audio_latents: bytes | None = None,
    ) -> np.ndarray:
        if self.server is None:
            raise RuntimeError("VoxCPM runtime is not started")
        generation_kwargs = {
            "target_text": text,
            "max_generate_length": max_generate_length,
            "temperature": float(self.settings["temperature"]),
            "cfg_value": float(self.settings["cfg_value"]),
            "ref_audio_latents": ref_audio_latents,
        }
        if seed is not None:
            generation_kwargs["seed"] = seed
        chunks = [
            np.asarray(chunk, dtype=np.float32)
            async for chunk in self.server.generate(**generation_kwargs)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    async def synthesize_batch(
        self,
        texts: Sequence[str],
        *,
        languages: Sequence[str | None] | None = None,
        seeds: Sequence[int | None] | None = None,
        reference_audio: bytes | None = None,
        reference_format: str = "wav",
    ) -> list[np.ndarray]:
        """Submit every item concurrently so nano-vLLM can schedule a real batch."""
        if not texts:
            return []
        if languages is None:
            languages = [None] * len(texts)
        if len(languages) != len(texts):
            raise ValueError("languages length must match texts length")
        if seeds is None:
            seeds = [None] * len(texts)
        if len(seeds) != len(texts):
            raise ValueError("seeds length must match texts length")
        generation_limits, _budgets = await self._predict_generation_limits(texts, languages)
        reference_latents = (
            await self._encode_reference(reference_audio, reference_format)
            if reference_audio is not None
            else None
        )
        return list(
            await asyncio.gather(
                *(
                    self._synthesize_one(
                        text,
                        max_generate_length=max_generate_length,
                        seed=seed,
                        ref_audio_latents=reference_latents,
                    )
                    for text, max_generate_length, seed in zip(texts, generation_limits, seeds)
                )
            )
        )


__all__ = ["VoxCPMRuntime"]
