"""Optimized nano-vLLM VoxCPM2 serving runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .voxcpm_lora import compose_voxcpm_loras

_LOGGER = logging.getLogger(__name__)
_KV_BLOCKS_ENV = "NANOVLLM_SERVERPOOL_NUM_KVCACHE_BLOCKS"


class VoxCPMRuntime:
    """Own one async nano-vLLM server pool and expose batched synthesis."""

    def __init__(self, settings: Mapping[str, Any]):
        self.settings = dict(settings)
        self.server: Any | None = None
        self.dtype: str | None = None
        self.sample_rate = 16000
        self.output_patch_samples = 2560
        self.ipa_fade_out_ratio = 0.2
        self._reference_latents: OrderedDict[str, bytes] = OrderedDict()
        self._reference_cache_lock = asyncio.Lock()
        self._duration_budget: Any | None = None
        self._duration_budget_init_lock = threading.Lock()
        self._duration_budget_predict_lock = asyncio.Lock()
        self._available_loras: set[str] = set()
        self._lora_paths: dict[str, Path] = {}
        self._lora_combinations: dict[tuple[str, ...], str] = {}
        self._lora_composition_lock = asyncio.Lock()

    @staticmethod
    def _resolve_path(path: str) -> Path:
        resolved = Path(path).expanduser()
        return resolved if resolved.is_absolute() else Path.cwd() / resolved

    def _load_lora_runtime_config(self) -> Any | None:
        configured = dict(self.settings.get("applicable_loras") or {})
        if not configured:
            return None

        from src.nanovllm_voxcpm.models.voxcpm2.config import LoRAConfig

        configs: list[dict[str, Any]] = []
        for name, path in configured.items():
            checkpoint_path = self._resolve_path(str(path))
            config_path = checkpoint_path / "lora_config.json"
            weights_path = checkpoint_path / "lora_weights.safetensors"
            if not config_path.is_file() or not weights_path.is_file():
                raise FileNotFoundError(
                    f"Configured VoxCPM LoRA {name!r} must contain lora_config.json "
                    f"and lora_weights.safetensors: {checkpoint_path}"
                )
            with config_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            lora_config = payload.get("lora_config")
            if not isinstance(lora_config, dict):
                raise TypeError(
                    f"Invalid VoxCPM LoRA config for {name!r}: {config_path}"
                )
            configs.append(lora_config)

        def union_list(key: str) -> list[str]:
            return list(
                dict.fromkeys(
                    item for config in configs for item in config.get(key, [])
                )
            )

        def canonical_qkv_targets(key: str) -> list[str]:
            targets = union_list(key)
            qkv_targets = ("q_proj", "k_proj", "v_proj")
            enabled = set(targets)
            return [target for target in qkv_targets if target in enabled] + [
                target for target in targets if target not in qkv_targets
            ]

        ranks = [int(config.get("r", config.get("rank", 0))) for config in configs]
        if any(rank <= 0 for rank in ranks):
            raise ValueError("Configured VoxCPM LoRAs must declare a positive rank")
        max_loras_per_request = int(self.settings.get("max_loras_per_request", 2))
        if max_loras_per_request <= 0:
            raise ValueError("voxcpm.max_loras_per_request must be greater than zero")
        max_rank = sum(sorted(ranks, reverse=True)[:max_loras_per_request])
        return LoRAConfig(
            enable_lm=any(bool(config.get("enable_lm")) for config in configs),
            enable_dit=any(bool(config.get("enable_dit")) for config in configs),
            enable_proj=any(bool(config.get("enable_proj")) for config in configs),
            max_loras=int(self.settings.get("max_concurrent_loras", 3)),
            max_lora_rank=max_rank,
            target_modules_lm=canonical_qkv_targets("target_modules_lm"),
            target_modules_dit=canonical_qkv_targets("target_modules_dit"),
            target_proj_modules=union_list("target_proj_modules"),
        )

    async def start(self) -> None:
        if self.server is not None:
            return

        from src.nanovllm_voxcpm import VoxCPM

        num_kvcache_blocks = int(self.settings["num_kvcache_blocks"])
        if num_kvcache_blocks <= 0:
            raise ValueError("voxcpm.num_kvcache_blocks must be greater than zero")
        os.environ[_KV_BLOCKS_ENV] = str(num_kvcache_blocks)
        dtype = str(self.settings.get("dtype", "auto"))

        configured_loras = dict(self.settings.get("applicable_loras") or {})
        self._lora_paths = {
            name: self._resolve_path(str(path))
            for name, path in configured_loras.items()
        }
        self.server = VoxCPM.from_pretrained(
            model=str(self.settings["model_path"]),
            dtype=dtype,
            devices=[int(self.settings["device"])],
            inference_timesteps=int(self.settings["inference_timesteps"]),
            max_num_batched_tokens=int(self.settings["max_num_batched_tokens"]),
            max_num_seqs=int(self.settings["max_num_seqs"]),
            max_model_len=int(self.settings["max_model_len"]),
            gpu_memory_utilization=float(self.settings["gpu_memory_utilization"]),
            enforce_eager=bool(self.settings["enforce_eager"]),
            lora_config=self._load_lora_runtime_config(),
            ipa_adapter_path=(
                str(self._resolve_path(str(self.settings["ipa_adapter_path"])))
                if self.settings.get("ipa_adapter_path")
                else None
            ),
        )
        await self.server.wait_for_ready()
        for name, path in configured_loras.items():
            await self.server.register_lora(name, str(self._resolve_path(str(path))))
            self._available_loras.add(name)
            self._lora_combinations[(name,)] = name
        model_info = await self.server.get_model_info()
        self.dtype = str(model_info["dtype"])
        self.sample_rate = int(model_info["sample_rate"])
        self.output_patch_samples = int(model_info["output_patch_samples"])
        if model_info.get("ipa_fade_out_ratio") is not None:
            self.ipa_fade_out_ratio = float(model_info["ipa_fade_out_ratio"])
        duration_settings = self.settings["duration_budget"]
        if duration_settings["enabled"] and duration_settings["preload"]:
            await asyncio.to_thread(self._get_duration_budget)
        _LOGGER.info(
            "VoxCPM nano-vLLM ready model=%s device=%s dtype=%s sample_rate=%d "
            "kv_blocks=%d max_num_seqs=%d loras=%s",
            self.settings["model_path"],
            self.settings["device"],
            self.dtype,
            self.sample_rate,
            num_kvcache_blocks,
            self.settings["max_num_seqs"],
            sorted(self._available_loras),
        )

    async def stop(self) -> None:
        server = self.server
        self.server = None
        self.dtype = None
        self._duration_budget = None
        self._reference_latents.clear()
        self._available_loras.clear()
        self._lora_paths.clear()
        self._lora_combinations.clear()
        if server is not None:
            await server.stop()

    async def resolve_lora_combination(self, lora_names: Sequence[str]) -> str | None:
        if not lora_names:
            return None
        names = tuple(sorted(lora_names))
        max_loras = int(self.settings.get("max_loras_per_request", 2))
        if len(names) > max_loras:
            raise ValueError(
                f"VoxCPM supports at most {max_loras} LoRAs per request"
            )
        unknown = sorted(set(names) - self._lora_paths.keys())
        if unknown:
            raise ValueError(f"VoxCPM LoRAs are not configured: {unknown}")
        cached = self._lora_combinations.get(names)
        if cached is not None:
            return cached
        if self.server is None:
            raise RuntimeError("VoxCPM runtime is not started")

        async with self._lora_composition_lock:
            cached = self._lora_combinations.get(names)
            if cached is not None:
                return cached
            digest = hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()[:16]
            runtime_name = f"combined-{digest}"
            cache_root = self._resolve_path(
                str(
                    self.settings.get(
                        "lora_composition_cache_path", "cache/voxcpm-lora-compositions"
                    )
                )
            )
            output_path = cache_root / runtime_name
            await asyncio.to_thread(
                compose_voxcpm_loras,
                [self._lora_paths[name] for name in names],
                output_path,
            )
            await self.server.register_lora(runtime_name, str(output_path))
            self._available_loras.add(runtime_name)
            self._lora_combinations[names] = runtime_name
            _LOGGER.info(
                "Registered composed VoxCPM LoRA name=%s components=%s",
                runtime_name,
                names,
            )
            return runtime_name

    def _get_duration_budget(self) -> Any:
        if self._duration_budget is not None:
            return self._duration_budget
        with self._duration_budget_init_lock:
            if self._duration_budget is not None:
                return self._duration_budget

            from src.duration_alignment import (
                DpBudgetConfig,
                DurationAlignmentValidator,
            )

            settings = self.settings["duration_budget"]
            config_path = settings.get("config_path")
            configured_device = str(settings["device"])
            duration_device = (
                f"cuda:{int(self.settings['device'])}"
                if configured_device.strip().lower() == "auto"
                else configured_device
            )
            validator = DurationAlignmentValidator(
                DpBudgetConfig(
                    checkpoint=Path(settings["checkpoint"]),
                    config_path=Path(config_path) if config_path else None,
                    device=duration_device,
                    language=str(settings["language"]),
                    length_scale=float(settings["length_scale"]),
                    token_rate=float(settings["token_rate"]),
                    min_margin=float(settings["min_margin"]),
                    max_margin=float(settings["max_margin"]),
                    min_extra_tokens=int(settings["min_extra_tokens"]),
                    max_extra_tokens=int(settings["max_extra_tokens"]),
                    soft_text_token_limit=int(settings["soft_text_token_limit"]),
                    hard_text_token_limit=int(settings["hard_text_token_limit"]),
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
                duration_device,
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

        cache_key = f"{audio_format}:{hashlib.sha256(audio).hexdigest()}"
        async with self._reference_cache_lock:
            cached = self._reference_latents.get(cache_key)
            if cached is not None:
                self._reference_latents.move_to_end(cache_key)
                return cached

        latents = await self.server.encode_latents(audio, audio_format)
        async with self._reference_cache_lock:
            cached = self._reference_latents.get(cache_key)
            if cached is not None:
                self._reference_latents.move_to_end(cache_key)
                return cached
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
        lora_name: str | None = None,
        ipa_controls: list[dict[str, Any]] | None = None,
        min_generate_length: int = 0,
    ) -> np.ndarray:
        if self.server is None:
            raise RuntimeError("VoxCPM runtime is not started")
        generation_kwargs = {
            "target_text": text,
            "max_generate_length": max_generate_length,
            "temperature": float(self.settings["temperature"]),
            "cfg_value": float(self.settings["cfg_value"]),
            "ref_audio_latents": ref_audio_latents,
            "lora_name": lora_name,
            "ipa_controls": ipa_controls,
            "min_generate_length": min_generate_length,
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

    async def synthesize_controlled(
        self,
        text: str,
        *,
        ipa_controls: list[dict[str, Any]],
        min_generate_length: int,
        max_generate_length: int,
        seed: int,
        reference_audio: bytes | None = None,
        reference_format: str = "wav",
        lora_name: str | None = None,
    ) -> np.ndarray:
        """Run one IPA-controlled request; nano-vLLM selects eager mode for it."""

        if not self.settings.get("ipa_adapter_path"):
            raise RuntimeError("VoxCPM IPA adapter is not configured")
        if lora_name is not None and lora_name not in self._available_loras:
            raise ValueError(f"VoxCPM LoRA is not registered: {lora_name}")
        reference_latents = (
            await self._encode_reference(reference_audio, reference_format)
            if reference_audio is not None
            else None
        )
        return await self._synthesize_one(
            text,
            max_generate_length=max_generate_length,
            seed=seed,
            ref_audio_latents=reference_latents,
            lora_name=lora_name,
            ipa_controls=ipa_controls,
            min_generate_length=min_generate_length,
        )

    async def synthesize_batch(
        self,
        texts: Sequence[str],
        *,
        languages: Sequence[str | None] | None = None,
        seeds: Sequence[int | None] | None = None,
        reference_audio: bytes | None = None,
        reference_format: str = "wav",
        reference_audios: Sequence[bytes | None] | None = None,
        reference_formats: Sequence[str] | None = None,
        lora_names: Sequence[str | None] | None = None,
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
        if lora_names is None:
            lora_names = [None] * len(texts)
        if len(lora_names) != len(texts):
            raise ValueError("lora_names length must match texts length")
        unknown_loras = sorted(
            {name for name in lora_names if name is not None} - self._available_loras
        )
        if unknown_loras:
            raise ValueError(f"VoxCPM LoRAs are not registered: {unknown_loras}")
        if reference_audio is not None and reference_audios is not None:
            raise ValueError("use either reference_audio or reference_audios, not both")
        if reference_audios is None:
            reference_audios = [reference_audio] * len(texts)
        if len(reference_audios) != len(texts):
            raise ValueError("reference_audios length must match texts length")
        if reference_formats is None:
            reference_formats = [reference_format] * len(texts)
        if len(reference_formats) != len(texts):
            raise ValueError("reference_formats length must match texts length")

        duration_started = time.perf_counter()
        generation_limits, _budgets = await self._predict_generation_limits(
            texts, languages
        )
        duration_wall_seconds = time.perf_counter() - duration_started

        unique_references: dict[tuple[str, str], tuple[bytes, str]] = {}
        reference_keys: list[tuple[str, str] | None] = []
        for audio, audio_format in zip(reference_audios, reference_formats):
            if audio is None:
                reference_keys.append(None)
                continue
            key = (hashlib.sha256(audio).hexdigest(), audio_format)
            unique_references.setdefault(key, (audio, audio_format))
            reference_keys.append(key)

        reference_started = time.perf_counter()
        encoded_references = await asyncio.gather(
            *(
                self._encode_reference(audio, audio_format)
                for audio, audio_format in unique_references.values()
            )
        )
        reference_latents_by_key = dict(zip(unique_references, encoded_references))
        reference_latents = [
            reference_latents_by_key[key] if key is not None else None
            for key in reference_keys
        ]
        reference_wall_seconds = time.perf_counter() - reference_started

        inference_started = time.perf_counter()
        outputs = list(
            await asyncio.gather(
                *(
                    self._synthesize_one(
                        text,
                        max_generate_length=max_generate_length,
                        seed=seed,
                        ref_audio_latents=item_reference_latents,
                        lora_name=lora_name,
                    )
                    for text, max_generate_length, seed, item_reference_latents, lora_name in zip(
                        texts,
                        generation_limits,
                        seeds,
                        reference_latents,
                        lora_names,
                    )
                )
            )
        )
        inference_wall_seconds = time.perf_counter() - inference_started
        _LOGGER.info(
            "VoxCPM runtime batch timing item_count=%d duration_budget_wall_seconds=%.4f "
            "reference_encode_wall_seconds=%.4f inference_wall_seconds=%.4f",
            len(texts),
            duration_wall_seconds,
            reference_wall_seconds,
            inference_wall_seconds,
        )
        return outputs


__all__ = ["VoxCPMRuntime"]
