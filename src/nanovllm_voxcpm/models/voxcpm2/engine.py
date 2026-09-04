from dataclasses import dataclass
import hashlib
import struct

import numpy as np
import torch
from transformers import LlamaTokenizerFast

from src.nanovllm_voxcpm.config import Config
from src.nanovllm_voxcpm.engine.llm_engine import LLMEngineBase
from src.nanovllm_voxcpm.engine.sequence import Sequence
from src.nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
from src.nanovllm_voxcpm.models.voxcpm2.lora_loader import load_voxcpm2_lora_checkpoint
from src.nanovllm_voxcpm.models.voxcpm2.runner import RunnerTask, VoxCPM2Payload, VoxCPM2Runner
from src.nanovllm_voxcpm.models.voxcpm2.utils import mask_multichar_chinese_tokens
from src.voxcpm_ipa_adapter import sparrow_phoneme_ids


def _ipa_cache_namespace(controls: list[dict]) -> bytes:
    digest = hashlib.sha256(b"voxcpm2-ipa-controls-v1")
    for control in controls:
        target_ipa = str(control["target_ipa"]).encode("utf-8")
        digest.update(struct.pack("<I", len(target_ipa)))
        digest.update(target_ipa)
        digest.update(
            struct.pack(
                "<qq?qq",
                int(control["controlled_start"]),
                int(control["controlled_end"]),
                bool(control.get("audio_enabled", True)),
                int(control.get("start_patch", -1)),
                int(control.get("target_patch_count", -1)),
            )
        )
        gates = np.asarray(control.get("gates", ()), dtype="<f4")
        digest.update(struct.pack("<I", gates.size))
        digest.update(gates.tobytes())
    return digest.digest()


@dataclass
class VoxCPM2SeqPayload:
    feats: list[np.ndarray]
    text_tokens: list[int]
    feat_masks: list[bool]
    generated_waveforms: list[np.ndarray]
    temperature: float
    cfg_value: float
    decode_pad: np.ndarray | None = None
    max_generate_length: int | None = None
    seed: int | None = None
    seed_step: int = 0
    ipa_memories: np.ndarray | None = None
    ipa_memory_mask: np.ndarray | None = None
    ipa_prefill_to_control: np.ndarray | None = None
    ipa_prefill_progress: np.ndarray | None = None
    ipa_prefill_gate: np.ndarray | None = None
    ipa_audio_to_control: np.ndarray | None = None
    ipa_audio_progress: np.ndarray | None = None
    ipa_audio_gate: np.ndarray | None = None
    min_generate_length: int = 0


class VoxCPM2Engine(LLMEngineBase):
    def __init__(self, config: Config[VoxCPM2Config]):
        self.n_decode_pad_frames = 12
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.audio_start_token = 101
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))
        super().__init__(VoxCPM2Runner, config, config.tensor_parallel_size)

    def register_lora(self, name: str, path: str) -> int:
        payload = load_voxcpm2_lora_checkpoint(path, tp_size=self.model_runner.world_size)
        return super().register_lora(name, payload)

    def preprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], is_prefill: bool) -> RunnerTask[VoxCPM2Payload]:
        payload = seq.custom_payload
        audio_index = payload.seed_step
        audio_active = (
            payload.ipa_audio_to_control is not None
            and audio_index < payload.ipa_audio_to_control.shape[0]
        )
        terminal_control = int(payload.ipa_audio_to_control[audio_index]) if audio_active else -1
        terminal_progress = float(payload.ipa_audio_progress[audio_index]) if audio_active else 0.0
        terminal_gate = float(payload.ipa_audio_gate[audio_index]) if audio_active else 0.0
        if is_prefill:
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            return RunnerTask(
                seq.block_table,
                len(seq),
                seq.num_cached_tokens,
                seq.block_size,
                VoxCPM2Payload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[seq.num_cached_tokens :], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][seq.num_cached_tokens :],
                    feat_masks=np.array(seq.custom_payload.feat_masks[seq.num_cached_tokens :], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                    seed=seq.custom_payload.seed,
                    seed_step=seq.custom_payload.seed_step,
                    ipa_memories=payload.ipa_memories,
                    ipa_memory_mask=payload.ipa_memory_mask,
                    ipa_transformer_to_control=(
                        payload.ipa_prefill_to_control[seq.num_cached_tokens :]
                        if payload.ipa_prefill_to_control is not None else None
                    ),
                    ipa_transformer_progress=(
                        payload.ipa_prefill_progress[seq.num_cached_tokens :]
                        if payload.ipa_prefill_progress is not None else None
                    ),
                    ipa_transformer_gate=(
                        payload.ipa_prefill_gate[seq.num_cached_tokens :]
                        if payload.ipa_prefill_gate is not None else None
                    ),
                    ipa_terminal_to_control=terminal_control,
                    ipa_terminal_progress=terminal_progress,
                    ipa_terminal_gate=terminal_gate,
                ),
                adapter_id=seq.adapter_id,
            )

        return RunnerTask(
            seq.block_table,
            len(seq),
            len(seq) - 1,
            seq.block_size,
            VoxCPM2Payload(
                text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                feats=seq.custom_payload.feats[-1][-1:],
                feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
                temperature=seq.custom_payload.temperature,
                cfg_value=seq.custom_payload.cfg_value,
                padding_decode=seq.custom_payload.decode_pad,
                seed=seq.custom_payload.seed,
                seed_step=seq.custom_payload.seed_step,
                ipa_memories=payload.ipa_memories,
                ipa_memory_mask=payload.ipa_memory_mask,
                ipa_transformer_to_control=np.array([terminal_control], dtype=np.int64),
                ipa_transformer_progress=np.array([terminal_progress], dtype=np.float32),
                ipa_transformer_gate=np.array([terminal_gate], dtype=np.float32),
                ipa_terminal_to_control=terminal_control,
                ipa_terminal_progress=terminal_progress,
                ipa_terminal_gate=terminal_gate,
            ),
            adapter_id=seq.adapter_id,
        )

    def postprocess_seq(self, seq: Sequence[VoxCPM2SeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        seq.append_token(latents.tobytes())
        seq.custom_payload.feats.append(latents[None])
        seq.custom_payload.text_tokens.append(0)
        seq.custom_payload.feat_masks.append(True)
        seq.custom_payload.seed_step += 1
        seq.custom_payload.generated_waveforms.append(waveforms)

        latents = latents.reshape(-1, self.feat_dim)
        if seq.custom_payload.decode_pad is not None:
            seq.custom_payload.decode_pad = np.concatenate([seq.custom_payload.decode_pad, latents], axis=0)[
                -self.n_decode_pad_frames :
            ]
        else:
            seq.custom_payload.decode_pad = latents[-self.n_decode_pad_frames :]

        if stop_flag == 1 and seq.custom_payload.seed_step >= seq.custom_payload.min_generate_length:
            seq.stoped = True
        elif (
            seq.custom_payload.max_generate_length is not None
            and len(seq.custom_payload.generated_waveforms) >= seq.custom_payload.max_generate_length
        ):
            seq.stoped = True

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_text: str = "",
        prompt_latents: np.ndarray | None = None,
        ref_audio_latents: np.ndarray | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        lora_name: str | None = None,
        seed: int | None = None,
        ipa_controls: list[dict] | None = None,
        min_generate_length: int = 0,
    ):
        if max_generate_length < 1:
            raise ValueError(f"max_generate_length must be >= 1, got {max_generate_length}")

        combined_text = prompt_text + target_text
        raw_text_tokens = self.tokenizer(combined_text)
        text_tokens = raw_text_tokens + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = [t for t in text_tokens]
        decode_pad = None
        ipa_memories = None
        ipa_memory_mask = None
        ipa_prefill_to_control = None
        ipa_prefill_progress = None
        ipa_prefill_gate = None
        ipa_audio_to_control = None
        ipa_audio_progress = None
        ipa_audio_gate = None
        cache_namespace = None

        if ipa_controls:
            if prompt_text:
                raise ValueError("IPA controls with prompt_text are not supported")
            # Offsets must live in the same token space as the prefill
            # sequence (the char-split wrapper tokenization); base-tokenizer
            # offsets misindex ipa_prefill_to_control for CJK text.
            offsets = self.tokenizer.tokenize_with_offsets(target_text)
            if len(offsets) != len(raw_text_tokens):
                raise ValueError("IPA span tokenization differs from VoxCPM target tokenization")
            ipa_prefill_to_control = np.full(len(text_tokens), -1, dtype=np.int64)
            ipa_prefill_progress = np.zeros(len(text_tokens), dtype=np.float32)
            ipa_prefill_gate = np.zeros(len(text_tokens), dtype=np.float32)
            ipa_audio_to_control = np.full(max_generate_length, -1, dtype=np.int64)
            ipa_audio_progress = np.zeros(max_generate_length, dtype=np.float32)
            ipa_audio_gate = np.zeros(max_generate_length, dtype=np.float32)
            phoneme_ids: list[list[int]] = []
            for control_index, control in enumerate(ipa_controls):
                start = int(control["controlled_start"])
                end = int(control["controlled_end"])
                token_indices = [
                    index for index, (token_start, token_end) in enumerate(offsets)
                    if token_end > start and token_start < end
                ]
                if not token_indices:
                    raise ValueError(f"IPA controlled span [{start}, {end}) has no VoxCPM text tokens")
                if any(ipa_prefill_to_control[index] >= 0 for index in token_indices):
                    raise ValueError("Overlapping IPA text controls are not supported")
                denom = max(1, len(token_indices) - 1)
                for order, token_index in enumerate(token_indices):
                    ipa_prefill_to_control[token_index] = control_index
                    ipa_prefill_progress[token_index] = order / denom
                    ipa_prefill_gate[token_index] = 1.0
                if bool(control.get("audio_enabled", True)):
                    start_patch = int(control["start_patch"])
                    gates = np.asarray(control["gates"], dtype=np.float32)
                    end_patch = min(max_generate_length, start_patch + gates.shape[0])
                    if start_patch < 0 or end_patch <= start_patch:
                        raise ValueError("IPA audio control falls outside the generation budget")
                    if np.any(ipa_audio_to_control[start_patch:end_patch] >= 0):
                        raise ValueError("Overlapping IPA audio controls are not supported")
                    count = end_patch - start_patch
                    ipa_audio_to_control[start_patch:end_patch] = control_index
                    target_patch_count = max(1, int(control["target_patch_count"]))
                    patch_indices = np.arange(count, dtype=np.float32)
                    ipa_audio_progress[start_patch:end_patch] = np.clip(
                        (patch_indices + 0.5) / target_patch_count,
                        0.0,
                        1.0,
                    )
                    ipa_audio_gate[start_patch:end_patch] = gates[:count]
                phoneme_ids.append(sparrow_phoneme_ids(str(control["target_ipa"])))
            ipa_memories, ipa_memory_mask = self.model_runner.encode_ipa_memories(phoneme_ids)
            cache_namespace = _ipa_cache_namespace(ipa_controls)

        if ref_audio_latents is not None:
            wav_latents = ref_audio_latents
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)

            audio_feat_pad = np.zeros((1, self.patch_size, self.feat_dim), dtype=np.float32)
            audio_feat = np.concatenate([audio_feat_pad, wav_latents, audio_feat_pad, audio_feat], axis=0)
            text_tokens = (
                [self.ref_audio_start_token]
                + ([0 for _ in range(wav_latents.shape[0])])
                + [self.ref_audio_end_token]
                + text_tokens
            )
            feat_masks = [False] + ([True for _ in range(wav_latents.shape[0])]) + [False] + feat_masks

            prepend_hash_tokens = (
                [self.ref_audio_start_token]
                + [wav_latents[i].tobytes() for i in range(wav_latents.shape[0])]
                + [self.ref_audio_end_token]
            )
            hash_tokens = prepend_hash_tokens + hash_tokens
            prefix = len(prepend_hash_tokens)
            if ipa_prefill_to_control is not None:
                ipa_prefill_to_control = np.pad(ipa_prefill_to_control, (prefix, 0), constant_values=-1)
                ipa_prefill_progress = np.pad(ipa_prefill_progress, (prefix, 0))
                ipa_prefill_gate = np.pad(ipa_prefill_gate, (prefix, 0))

        if prompt_latents is not None:
            wav_latents = prompt_latents
            decode_pad = wav_latents[-self.n_decode_pad_frames :]
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])
            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        prompt_len = len(hash_tokens)
        total_len_upper_bound = prompt_len + max_generate_length
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"Prompt is too long for max_model_len: prompt_len={prompt_len} > max_model_len={self.max_model_len}"
            )
        if total_len_upper_bound > self.max_model_len:
            raise ValueError(
                "Request may exceed max_model_len: "
                f"prompt_len({prompt_len}) + max_generate_length({max_generate_length}) = {total_len_upper_bound} "
                f"> max_model_len({self.max_model_len}). "
                "Reduce input length or max_generate_length, or increase max_model_len."
            )

        adapter_id = self.resolve_lora(lora_name)

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoxCPM2SeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
                seed=seed,
                seed_step=0,
                ipa_memories=ipa_memories,
                ipa_memory_mask=ipa_memory_mask,
                ipa_prefill_to_control=ipa_prefill_to_control,
                ipa_prefill_progress=ipa_prefill_progress,
                ipa_prefill_gate=ipa_prefill_gate,
                ipa_audio_to_control=ipa_audio_to_control,
                ipa_audio_progress=ipa_audio_progress,
                ipa_audio_gate=ipa_audio_gate,
                min_generate_length=max(0, int(min_generate_length)),
            ),
            lora_name=lora_name,
            adapter_id=adapter_id,
            cache_namespace=cache_namespace,
        )
        self.add_sequence(seq)

    def encode_latents(self, wav: torch.Tensor, align_size: int = -1) -> np.ndarray:
        if align_size == -1:
            align_size = self.patch_size * self.model_runner.vae.encoder_chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))
        return self.model_runner.encode_latents(wav)
