import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import numpy as np
import torch

from src.nanovllm_voxcpm.config import Config, resolve_torch_dtype
from src.nanovllm_voxcpm.engine.model_runner import BaseModelRunner, RunnerTask
from src.nanovllm_voxcpm.layers.audio_vae_v2 import AudioVAEV2
from src.nanovllm_voxcpm.models.voxcpm2.config import VoxCPM2Config
from src.nanovllm_voxcpm.models.voxcpm2.model import VoxCPM2Model
from src.nanovllm_voxcpm.utils.loader import load_model
from src.nanovllm_voxcpm.utils.seed import derive_step_seed
from src.voxcpm_ipa_adapter import IPAControl, load_nanovllm_ipa_adapter


@dataclass
class VoxCPM2Payload:
    text_tokens: np.ndarray | None = None
    feats: np.ndarray | None = None
    feat_masks: np.ndarray | None = None
    temperature: float = 1.0
    cfg_value: float = 1.0
    padding_decode: np.ndarray | None = None
    seed: int | None = None
    seed_step: int = 0
    ipa_memories: np.ndarray | None = None
    ipa_memory_mask: np.ndarray | None = None
    ipa_transformer_to_control: np.ndarray | None = None
    ipa_transformer_progress: np.ndarray | None = None
    ipa_transformer_gate: np.ndarray | None = None
    ipa_terminal_to_control: int = -1
    ipa_terminal_progress: float = 0.0
    ipa_terminal_gate: float = 0.0


class VoxCPM2Runner(BaseModelRunner):
    model: VoxCPM2Model
    dit_lora_seq_len_offset = 3

    def __init__(
        self,
        config: Config[VoxCPM2Config],
        rank: int,
        device_idx: int,
        distributed_port: int | None,
        event: Event | list[Event],
    ):
        self.inference_timesteps = config.model_config.inference_timesteps
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.lora_config = config.lora_config
        self.ipa_adapter_path = config.ipa_adapter_path
        self.ipa_adapter = None
        self.ipa_fade_out_ratio = 0.2
        super().__init__(config, rank, device_idx, distributed_port, event)

    @property
    def dtype(self) -> torch.dtype:
        return resolve_torch_dtype(self._config.model_config.dtype)

    def init_model(self, model_config: VoxCPM2Config, model_path: str):
        self.model = VoxCPM2Model(model_config, self.inference_timesteps, lora_config=self.lora_config)
        load_model(self.model, model_path)
        if self.ipa_adapter_path:
            if self.world_size != 1:
                raise ValueError("The VoxCPM IPA adapter currently requires tensor_parallel_size=1")
            self.ipa_adapter, self.ipa_fade_out_ratio = load_nanovllm_ipa_adapter(
                self.ipa_adapter_path, self.model
            )

        torch.set_default_dtype(torch.float32)
        try:
            self.vae = AudioVAEV2(config=model_config.audio_vae_config)
            vae_state_dict = torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"]
            self.vae.load_state_dict(vae_state_dict)
        finally:
            torch.set_default_dtype(self.dtype)

    def make_dummy_inputs(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        return {
            "text_tokens": torch.zeros(batch_size * length, dtype=torch.int64),
            "feat": torch.zeros(batch_size * length, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(batch_size * length, dtype=torch.bool),
            "temperature": torch.zeros(batch_size),
            "cfg_value": torch.zeros(batch_size),
            "z_noise": torch.zeros(batch_size, self.feat_dim, self.patch_size, dtype=self.dtype),
        }

    def make_dummy_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "latents": torch.zeros(batch_size, self.patch_size, self.feat_dim, dtype=self.dtype),
            "stop_flag": torch.zeros(batch_size, dtype=torch.int64),
        }

    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        wav = wav.to(torch.float32).cuda()
        return (
            self.vae.encode(wav, self.vae.sample_rate)
            .permute(0, 2, 1)
            .view(-1, self.feat_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

    def encode_ipa_memories(self, phoneme_ids: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
        if self.ipa_adapter is None:
            raise RuntimeError("VoxCPM IPA adapter is not configured")
        if not phoneme_ids or any(not item for item in phoneme_ids):
            raise ValueError("IPA controls must contain non-empty phoneme ID sequences")
        max_phones = max(len(item) for item in phoneme_ids)
        ids = torch.zeros(len(phoneme_ids), max_phones, dtype=torch.long, device="cuda")
        lengths = torch.tensor([len(item) for item in phoneme_ids], dtype=torch.long, device="cuda")
        for index, item in enumerate(phoneme_ids):
            ids[index, : len(item)] = torch.tensor(item, dtype=torch.long, device="cuda")
        with torch.inference_mode():
            memories, mask = self.ipa_adapter.phoneme_encoder(ids, lengths)
        return memories.float().cpu().numpy(), mask.cpu().numpy()

    def _ipa_control(
        self,
        seqs: list[RunnerTask[VoxCPM2Payload]],
        maps: list[np.ndarray],
        progresses: list[np.ndarray],
        gates: list[np.ndarray],
    ) -> IPAControl | None:
        if self.ipa_adapter is None or not any(bool((item >= 0).any()) for item in maps):
            return None
        rows = sum(item.shape[0] for item in maps)
        max_controls = max(
            payload.ipa_memories.shape[0] for payload in (seq.custom_payload for seq in seqs)
            if payload.ipa_memories is not None
        )
        max_phones = max(
            payload.ipa_memories.shape[1] for payload in (seq.custom_payload for seq in seqs)
            if payload.ipa_memories is not None
        )
        memory_dim = self.ipa_adapter.phoneme_encoder.hidden_channels
        memories = torch.zeros(rows, max_controls, max_phones, memory_dim, dtype=self.dtype, device="cuda")
        memory_mask = torch.zeros(rows, max_controls, max_phones, dtype=torch.bool, device="cuda")
        control_map = torch.full((rows, 1), -1, dtype=torch.long, device="cuda")
        progress = torch.zeros(rows, 1, dtype=self.dtype, device="cuda")
        gate = torch.zeros(rows, 1, dtype=self.dtype, device="cuda")
        offset = 0
        for seq, item_map, item_progress, item_gate in zip(seqs, maps, progresses, gates):
            count = item_map.shape[0]
            payload = seq.custom_payload
            if payload.ipa_memories is not None:
                n_controls, n_phones, _ = payload.ipa_memories.shape
                item_memories = torch.from_numpy(payload.ipa_memories).to(device="cuda", dtype=self.dtype)
                item_mask = torch.from_numpy(payload.ipa_memory_mask).to(device="cuda")
                memories[offset : offset + count, :n_controls, :n_phones] = item_memories.unsqueeze(0)
                memory_mask[offset : offset + count, :n_controls, :n_phones] = item_mask.unsqueeze(0)
            control_map[offset : offset + count, 0] = torch.from_numpy(item_map).to(device="cuda")
            progress[offset : offset + count, 0] = torch.from_numpy(item_progress).to(device="cuda", dtype=self.dtype)
            gate[offset : offset + count, 0] = torch.from_numpy(item_gate).to(device="cuda", dtype=self.dtype)
            offset += count
        control = IPAControl(memories, memory_mask, control_map, progress, gate)
        control.validate()
        return control

    @staticmethod
    def _has_active_ipa_control(payload: VoxCPM2Payload) -> bool:
        transformer_active = (
            payload.ipa_transformer_to_control is not None
            and bool((payload.ipa_transformer_to_control >= 0).any())
        )
        return transformer_active or payload.ipa_terminal_to_control >= 0

    def run(self, seqs: list[RunnerTask[VoxCPM2Payload]], is_prefill: bool):
        active = [self._has_active_ipa_control(seq.custom_payload) for seq in seqs]
        if any(active) and not all(active):
            outputs: list[dict | None] = [None] * len(seqs)
            for controlled in (False, True):
                indices = [index for index, value in enumerate(active) if value is controlled]
                group_outputs = self.run([seqs[index] for index in indices], is_prefill)
                for index, output in zip(indices, group_outputs, strict=True):
                    outputs[index] = output
            if any(output is None for output in outputs):
                raise RuntimeError("VoxCPM mixed IPA batch did not produce every output")
            return outputs

        positions = self.prepare_prefill_context(seqs) if is_prefill else self.prepare_decode_context(seqs)
        inputs = {"positions": positions}

        text_tokens = []
        feats = []
        feat_masks = []
        temperatures = []
        cfg_values = []
        for seq in seqs:
            payload = seq.custom_payload
            assert payload.text_tokens.shape[0] == payload.feats.shape[0]
            assert payload.text_tokens.shape[0] == payload.feat_masks.shape[0]
            text_tokens.append(payload.text_tokens)
            feats.append(payload.feats)
            feat_masks.append(payload.feat_masks)
            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)

        inputs["text_tokens"] = torch.from_numpy(np.concatenate(text_tokens, axis=0)).cuda(non_blocking=True)
        inputs["feat"] = torch.from_numpy(np.concatenate(feats, axis=0)).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = torch.from_numpy(np.concatenate(feat_masks, axis=0)).cuda(non_blocking=True)
        inputs["temperature"] = (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )
        inputs["cfg_value"] = (
            torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )

        bsz = len(seqs)
        seeded_rows = [
            (i, int(seq.custom_payload.seed), seq.custom_payload.seed_step)
            for i, seq in enumerate(seqs)
            if seq.custom_payload.seed is not None and seq.custom_payload.seed >= 0
        ]
        if len(seeded_rows) == bsz:
            z_noise = torch.empty((bsz, self.feat_dim, self.patch_size), dtype=self.dtype, device="cuda")
        else:
            z_noise = torch.randn((bsz, self.feat_dim, self.patch_size), dtype=self.dtype, device="cuda")

        for i, seed_val, seed_step in seeded_rows:
            generator = torch.Generator(device="cuda").manual_seed(derive_step_seed(seed_val, seed_step))
            z_noise[i] = torch.randn(
                (self.feat_dim, self.patch_size), generator=generator, dtype=self.dtype, device="cuda"
            )

        inputs["z_noise"] = z_noise

        transformer_maps: list[np.ndarray] = []
        transformer_progresses: list[np.ndarray] = []
        transformer_gates: list[np.ndarray] = []
        terminal_maps: list[np.ndarray] = []
        terminal_progresses: list[np.ndarray] = []
        terminal_gates: list[np.ndarray] = []
        for seq in seqs:
            payload = seq.custom_payload
            row_count = payload.text_tokens.shape[0]
            transformer_maps.append(
                payload.ipa_transformer_to_control
                if payload.ipa_transformer_to_control is not None
                else np.full(row_count, -1, dtype=np.int64)
            )
            transformer_progresses.append(
                payload.ipa_transformer_progress
                if payload.ipa_transformer_progress is not None
                else np.zeros(row_count, dtype=np.float32)
            )
            transformer_gates.append(
                payload.ipa_transformer_gate
                if payload.ipa_transformer_gate is not None
                else np.zeros(row_count, dtype=np.float32)
            )
            terminal_maps.append(np.array([payload.ipa_terminal_to_control], dtype=np.int64))
            terminal_progresses.append(np.array([payload.ipa_terminal_progress], dtype=np.float32))
            terminal_gates.append(np.array([payload.ipa_terminal_gate], dtype=np.float32))
        transformer_control = self._ipa_control(
            seqs, transformer_maps, transformer_progresses, transformer_gates
        )
        terminal_control = self._ipa_control(
            seqs, terminal_maps, terminal_progresses, terminal_gates
        )
        force_eager = transformer_control is not None or terminal_control is not None
        if self.ipa_adapter is None:
            outputs = self.run_model(inputs, is_prefill)
        else:
            with self.ipa_adapter.use_nanovllm_controls(transformer_control, terminal_control):
                outputs = self.run_model(inputs, is_prefill, force_eager=force_eager)
        latents = outputs["latents"]

        pad_lengths = [
            seq.custom_payload.padding_decode.shape[0] if seq.custom_payload.padding_decode is not None else 0
            for seq in seqs
        ]

        max_pad_decode = max(pad_lengths) + self.patch_size
        vae_decoder_inputs = torch.zeros(len(seqs), max_pad_decode, self.feat_dim, dtype=torch.float32, device="cuda")
        for i, seq in enumerate(seqs):
            pad_len = pad_lengths[i]
            if pad_len > 0:
                vae_decoder_inputs[i, :pad_len] = torch.from_numpy(seq.custom_payload.padding_decode).cuda(
                    non_blocking=True
                )
            vae_decoder_inputs[i, pad_len : pad_len + self.patch_size] = latents[i].to(torch.float32)

        vae_decoder_outputs = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))[:, 0, :].cpu().numpy()
        stop_flag = outputs["stop_flag"].cpu().tolist()
        ret_waveforms = []
        for i, pad_len in enumerate(pad_lengths):
            ret_waveforms.append(
                vae_decoder_outputs[
                    i,
                    pad_len * self.vae.decoder_chunk_size : (pad_len + self.patch_size) * self.vae.decoder_chunk_size,
                ]
            )

        np_latents = latents.to(torch.float32).cpu().numpy()
        return [
            {"latents": np_latents[i], "stop_flag": stop_flag[i], "waveforms": ret_waveforms[i]}
            for i in range(len(seqs))
        ]
