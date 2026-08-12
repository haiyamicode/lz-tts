"""Audio-step IPA conditioning for VoxCPM without speakable control tokens."""

from __future__ import annotations

import math
import json
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch
from torch import nn


_ENGLISH_IPA_RESPELLINGS = {
    "t͡ʃ": "ch",
    "d͡ʒ": "j",
    "tʃ": "ch",
    "dʒ": "j",
    "eɪ": "ay",
    "aɪ": "eye",
    "aʊ": "ow",
    "ɔɪ": "oy",
    "oʊ": "oh",
    "ɪə": "ear",
    "eə": "air",
    "ʊə": "oor",
    "iː": "ee",
    "uː": "oo",
    "ɑː": "ah",
    "ɔː": "aw",
    "ɜː": "er",
    "ɝ": "er",
    "ɚ": "er",
    "˞": "r",
    "θ": "th",
    "ð": "th",
    "ʃ": "sh",
    "ʒ": "zh",
    "ŋ": "ng",
    "ɡ": "g",
    "ɹ": "r",
    "ɾ": "t",
    "j": "y",
    "ɫ": "l",
    "ɬ": "l",
    "ʔ": "",
    "æ": "a",
    "ɑ": "ah",
    "ɒ": "o",
    "ɔ": "aw",
    "ə": "uh",
    "ɛ": "eh",
    "ɜ": "er",
    "ɪ": "ih",
    "ʊ": "oo",
    "ʌ": "uh",
    "ɐ": "uh",
    "ᵻ": "ih",
}
_IPA_NOTATION = frozenset("/ˈˌːˑ[](){}͡")
_IPA_PHONE_DELIMITERS = frozenset(
    {
        *"abcdefghijklmnopqrstuvwxyz0123456789",
        *_ENGLISH_IPA_RESPELLINGS,
    }
)


def _ipa_chunk_is_one_phone(chunk: str) -> bool:
    stripped = "".join(symbol for symbol in chunk if symbol not in _IPA_NOTATION)
    if not stripped:
        return True
    return stripped in _IPA_PHONE_DELIMITERS


def approximate_ipa_spelling(ipa: str, language: str = "en-US") -> str:
    """Convert IPA into a rough speakable spelling used only as an LM guide.

    Exact pronunciation remains in the IPA side channel. This spelling gives
    VoxCPM a text sequence with approximately matching sounds and word count,
    including phrase expansions whose visible source span is one token.
    """

    language_key = language.replace("_", "-").lower()
    if language_key not in {"en", "en-us", "en-gb"}:
        raise ValueError(
            f"IPA respelling currently supports English controls, got {language!r}"
        )
    normalized = unicodedata.normalize("NFC", ipa.strip())
    if not normalized:
        raise ValueError("IPA pronunciation must not be empty")

    chunks = normalized.split()
    # ASR-derived labels commonly put a space between every phone. Preserve
    # spaces only when chunks represent complete IPA words/phrases.
    preserve_word_boundaries = len(chunks) > 1 and not all(
        _ipa_chunk_is_one_phone(chunk) for chunk in chunks
    )
    source = " ".join(chunks) if preserve_word_boundaries else "".join(chunks)
    keys = sorted(_ENGLISH_IPA_RESPELLINGS, key=len, reverse=True)
    output: list[str] = []
    index = 0
    while index < len(source):
        symbol = source[index]
        if symbol.isspace():
            output.append(" ")
            index += 1
            continue
        if symbol in _IPA_NOTATION or unicodedata.combining(symbol):
            index += 1
            continue
        match = next((key for key in keys if source.startswith(key, index)), None)
        if match is not None:
            output.append(_ENGLISH_IPA_RESPELLINGS[match])
            index += len(match)
            continue
        if symbol.isascii() and symbol.isalnum():
            output.append(symbol)
            index += 1
            continue
        raise ValueError(
            f"No approximate English spelling for IPA symbol {symbol!r} in {ipa!r}"
        )
    spelling = " ".join("".join(output).split())
    if not spelling:
        raise ValueError(f"IPA pronunciation produced an empty spelling: {ipa!r}")
    return spelling


def replace_span_with_ipa_spelling(
    text: str,
    start: int,
    end: int,
    ipa: str,
    language: str = "en-US",
) -> tuple[str, str]:
    """Replace exactly one source span with its approximate IPA spelling."""

    if not 0 <= start < end <= len(text):
        raise ValueError(f"Invalid IPA source span [{start}, {end}) for {text!r}")
    spelling = approximate_ipa_spelling(ipa, language)
    return f"{text[:start]}{spelling}{text[end:]}", spelling


def cosine_fade_patch_gates(
    target_duration_seconds: float,
    patch_seconds: float,
    patch_count: int,
    fade_out_ratio: float,
) -> list[float]:
    """Keep full adapter strength for the target duration, then fade afterward."""

    if target_duration_seconds <= 0.0 or patch_seconds <= 0.0 or patch_count <= 0:
        raise ValueError("IPA gate durations and patch count must be positive")
    if not 0.0 <= fade_out_ratio <= 1.0:
        raise ValueError(f"IPA fade-out ratio must be in [0, 1], got {fade_out_ratio}")
    boundaries = torch.arange(patch_count + 1, dtype=torch.float64) * patch_seconds
    fade_duration = target_duration_seconds * fade_out_ratio
    control_duration = target_duration_seconds + fade_duration
    clipped = boundaries.clamp(0.0, control_duration)
    if fade_out_ratio == 0.0:
        cumulative = clipped
    else:
        fade_elapsed = (clipped - target_duration_seconds).clamp(0.0, fade_duration)
        fade_integral = (
            0.5 * fade_elapsed
            + fade_duration
            / (2.0 * math.pi)
            * torch.sin(math.pi * fade_elapsed / fade_duration)
        )
        cumulative = torch.where(
            clipped <= target_duration_seconds,
            clipped,
            target_duration_seconds + fade_integral,
        )
    return (
        ((cumulative[1:] - cumulative[:-1]) / patch_seconds)
        .float()
        .clamp(0.0, 1.0)
        .tolist()
    )


def resolve_ipa_control_schedules(
    spans: Sequence[dict[str, float]],
    target_durations_seconds: Sequence[float],
    *,
    baseline_patch_count: int,
    patch_seconds: float,
    fade_out_ratio: float,
) -> tuple[list[dict[str, object]], int]:
    """Map aligned source spans to non-overlapping controlled decode schedules."""

    if len(spans) != len(target_durations_seconds):
        raise ValueError("IPA alignment spans and target durations do not match")
    if baseline_patch_count <= 0 or patch_seconds <= 0.0:
        raise ValueError("IPA scheduling requires a positive baseline and patch duration")

    schedules: list[dict[str, object]] = []
    cumulative_shift = 0
    for control_index, (span, target_duration) in enumerate(
        zip(spans, target_durations_seconds, strict=True)
    ):
        target_duration = float(target_duration)
        if target_duration <= 0.0:
            raise ValueError("IPA target duration must be positive")
        baseline_start = max(0, math.floor(float(span["start_seconds"]) / patch_seconds))
        baseline_end = min(
            baseline_patch_count,
            max(
                baseline_start + 1,
                math.ceil(float(span["end_seconds"]) / patch_seconds),
            ),
        )
        baseline_span_patches = baseline_end - baseline_start
        target_patch_count = max(1, math.ceil(target_duration / patch_seconds))
        control_patch_count = max(
            target_patch_count,
            math.ceil(target_duration * (1.0 + fade_out_ratio) / patch_seconds),
        )
        start_patch = baseline_start + cumulative_shift
        duration_shift = target_patch_count - baseline_span_patches
        cumulative_shift += duration_shift
        schedules.append(
            {
                "control_index": control_index,
                "start_patch": start_patch,
                "baseline_start_patch": baseline_start,
                "baseline_end_patch": baseline_end,
                "baseline_span_patches": baseline_span_patches,
                "target_patch_count": target_patch_count,
                "control_patch_count": control_patch_count,
                "target_duration_seconds": target_duration,
                "duration_shift_patches": duration_shift,
            }
        )

    for index, schedule in enumerate(schedules[:-1]):
        next_start = int(schedules[index + 1]["start_patch"])
        target_end = int(schedule["start_patch"]) + int(schedule["target_patch_count"])
        if next_start < target_end:
            raise ValueError(
                "Predicted IPA target intervals overlap after duration adjustment: "
                f"control={index}, target_end={target_end}, next_start={next_start}"
            )
        schedule["control_patch_count"] = min(
            int(schedule["control_patch_count"]),
            next_start - int(schedule["start_patch"]),
        )

    expected_total = max(1, baseline_patch_count + cumulative_shift)
    for schedule in schedules:
        expected_total = max(
            expected_total,
            int(schedule["start_patch"]) + int(schedule["target_patch_count"]),
        )
    for schedule in schedules:
        patch_count = int(schedule["control_patch_count"])
        schedule["gates"] = cosine_fade_patch_gates(
            float(schedule["target_duration_seconds"]),
            patch_seconds,
            patch_count,
            fade_out_ratio,
        )
        schedule["expected_total_patches"] = expected_total
    return schedules, expected_total


def sparrow_phoneme_ids(ipa: str) -> list[int]:
    """Encode IPA with the same symbol IDs and boundary/blank contract as Sparrow."""

    from piper_phonemize import phoneme_ids_espeak

    normalized = unicodedata.normalize("NFD", ipa.strip())
    if not normalized:
        raise ValueError("IPA pronunciation must not be empty")
    missing: dict[str, int] = {}
    ids = list(phoneme_ids_espeak(list(normalized), missing_phonemes=missing))
    if missing:
        symbols = ", ".join(repr(symbol) for symbol in sorted(missing))
        raise ValueError(f"IPA contains symbols unsupported by Sparrow: {symbols}")
    return ids


class SparrowPhonemeEncoder(nn.Module):
    """Speaker-independent copy of Sparrow's phoneme embedding and encoder."""

    def __init__(
        self,
        *,
        num_symbols: int = 256,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        num_heads: int = 2,
        num_layers: int = 6,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        from src.piper.vits.attentions import Encoder

        self.hidden_channels = int(hidden_channels)
        self.embedding = nn.Embedding(int(num_symbols), self.hidden_channels)
        self.encoder = Encoder(
            self.hidden_channels,
            int(filter_channels),
            int(num_heads),
            int(num_layers),
            int(kernel_size),
            0.0,
            gin_channels=0,
        )
        self.warm_start_report: dict[str, object] | None = None

    def warm_start(self, checkpoint: str | Path) -> dict[str, object]:
        checkpoint = Path(checkpoint).expanduser().resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        state = payload.get("state_dict", payload)
        prefix = "model_g.enc_p."
        embedding_key = f"{prefix}emb.weight"
        if embedding_key not in state:
            raise KeyError(f"Sparrow checkpoint has no {embedding_key!r}")

        embedding_weight = state[embedding_key]
        if embedding_weight.shape != self.embedding.weight.shape:
            raise ValueError(
                "Sparrow phoneme embedding shape mismatch: "
                f"checkpoint={tuple(embedding_weight.shape)}, "
                f"expected={tuple(self.embedding.weight.shape)}"
            )
        self.embedding.load_state_dict({"weight": embedding_weight}, strict=True)

        encoder_prefix = f"{prefix}encoder."
        encoder_state = {
            key[len(encoder_prefix) :]: value
            for key, value in state.items()
            if key.startswith(encoder_prefix) and "spk_emb_linear" not in key
        }
        load_result = self.encoder.load_state_dict(encoder_state, strict=True)
        report = {
            "checkpoint": str(checkpoint),
            "embedding_shape": list(embedding_weight.shape),
            "encoder_tensors": len(encoder_state),
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
            "speaker_conditioning_loaded": False,
            "vits_projection_loaded": False,
            "bert_branch_loaded": False,
        }
        self.warm_start_report = report
        return report

    def forward(self, phoneme_ids: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if phoneme_ids.ndim != 2:
            raise ValueError(f"phoneme_ids must be [N,P], got {tuple(phoneme_ids.shape)}")
        if lengths.shape != (phoneme_ids.shape[0],):
            raise ValueError(
                f"phoneme lengths must be [{phoneme_ids.shape[0]}], got {tuple(lengths.shape)}"
            )

        positions = torch.arange(phoneme_ids.shape[1], device=phoneme_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        encoded = self.embedding(phoneme_ids) * math.sqrt(self.hidden_channels)
        encoded = self.encoder(
            encoded.transpose(1, 2) * valid.unsqueeze(1),
            valid.unsqueeze(1).to(encoded.dtype),
            g=None,
        ).transpose(1, 2)
        return encoded, valid


@dataclass(frozen=True)
class IPAControl:
    """Encoded IPA memories and their activation schedule over predicted audio patches."""

    memories: torch.Tensor  # [B,N,P,C]
    memory_mask: torch.Tensor  # [B,N,P]
    audio_to_control: torch.Tensor  # [B,T], -1 means no adapter delta
    audio_progress: torch.Tensor  # [B,T], normalized progress through the pronunciation
    audio_gate: torch.Tensor  # [B,T], adapter strength in [0, 1]
    prefill_to_control: torch.Tensor | None = None  # [B,S]
    prefill_progress: torch.Tensor | None = None  # [B,S]
    prefill_gate: torch.Tensor | None = None  # [B,S]
    training_lm_to_control: torch.Tensor | None = None  # [B,T], teacher-forcing input positions
    training_lm_progress: torch.Tensor | None = None  # [B,T]
    training_lm_gate: torch.Tensor | None = None  # [B,T]

    def validate(self) -> None:
        if self.memories.ndim != 4:
            raise ValueError(f"IPA memories must be [B,N,P,C], got {tuple(self.memories.shape)}")
        if self.memory_mask.shape != self.memories.shape[:3]:
            raise ValueError("IPA memory mask does not match IPA memories")
        schedule_shape = self.audio_to_control.shape
        if self.audio_to_control.ndim != 2:
            raise ValueError(f"IPA audio map must be [B,T], got {schedule_shape}")
        if self.audio_progress.shape != schedule_shape or self.audio_gate.shape != schedule_shape:
            raise ValueError("IPA progress/gate tensors do not match the audio map")
        if self.memories.shape[0] != schedule_shape[0]:
            raise ValueError("IPA memory batch does not match its audio schedule")
        active = self.audio_to_control >= 0
        if active.any() and int(self.audio_to_control[active].max()) >= self.memories.shape[1]:
            raise ValueError("IPA audio map refers to a missing pronunciation control")
        if not torch.isfinite(self.audio_progress).all() or not torch.isfinite(self.audio_gate).all():
            raise ValueError("IPA progress/gate contains a non-finite value")
        if bool(((self.audio_progress < 0) | (self.audio_progress > 1)).any()):
            raise ValueError("IPA audio progress must be in [0, 1]")
        if bool(((self.audio_gate < 0) | (self.audio_gate > 1)).any()):
            raise ValueError("IPA audio gate must be in [0, 1]")
        prefill_values = (
            self.prefill_to_control,
            self.prefill_progress,
            self.prefill_gate,
        )
        if any(value is not None for value in prefill_values):
            if not all(value is not None for value in prefill_values):
                raise ValueError("IPA prefill map, progress, and gate must be provided together")
            prefill_shape = self.prefill_to_control.shape
            if self.prefill_to_control.ndim != 2:
                raise ValueError(f"IPA prefill map must be [B,S], got {prefill_shape}")
            if (
                self.prefill_progress.shape != prefill_shape
                or self.prefill_gate.shape != prefill_shape
            ):
                raise ValueError("IPA prefill progress/gate do not match the prefill map")
            if prefill_shape[0] != self.memories.shape[0]:
                raise ValueError("IPA prefill schedule batch does not match IPA memories")
            prefill_active = self.prefill_to_control >= 0
            if (
                prefill_active.any()
                and int(self.prefill_to_control[prefill_active].max()) >= self.memories.shape[1]
            ):
                raise ValueError("IPA prefill map refers to a missing pronunciation control")
            if (
                not torch.isfinite(self.prefill_progress).all()
                or not torch.isfinite(self.prefill_gate).all()
            ):
                raise ValueError("IPA prefill progress/gate contains a non-finite value")
            if bool(((self.prefill_progress < 0) | (self.prefill_progress > 1)).any()):
                raise ValueError("IPA prefill progress must be in [0, 1]")
            if bool(((self.prefill_gate < 0) | (self.prefill_gate > 1)).any()):
                raise ValueError("IPA prefill gate must be in [0, 1]")
        training_lm_values = (
            self.training_lm_to_control,
            self.training_lm_progress,
            self.training_lm_gate,
        )
        if any(value is not None for value in training_lm_values):
            if not all(value is not None for value in training_lm_values):
                raise ValueError("IPA training LM map, progress, and gate must be provided together")
            if self.training_lm_to_control.shape != self.audio_to_control.shape:
                raise ValueError("IPA training LM map must match the packed audio schedule shape")
            if (
                self.training_lm_progress.shape != self.audio_to_control.shape
                or self.training_lm_gate.shape != self.audio_to_control.shape
            ):
                raise ValueError("IPA training LM progress/gate must match its map")

    def step_values(
        self,
        hidden_states: torch.Tensor,
        decode_step: int,
        *,
        transformer_lm: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return control index, progress, and gate matching one projection input."""

        if hidden_states.ndim == 3:
            batch, sequence, _ = hidden_states.shape
            if self.prefill_to_control is not None:
                schedule = self.prefill_to_control, self.prefill_progress, self.prefill_gate
            elif transformer_lm and self.training_lm_to_control is not None:
                schedule = (
                    self.training_lm_to_control,
                    self.training_lm_progress,
                    self.training_lm_gate,
                )
            else:
                schedule = self.audio_to_control, self.audio_progress, self.audio_gate
            control_map, progress, gate = schedule
            if control_map.shape != (batch, sequence):
                raise ValueError(
                    "IPA prefill/training schedule does not match VoxCPM sequence: "
                    f"schedule={tuple(control_map.shape)}, "
                    f"hidden={tuple(hidden_states.shape)}"
                )
            return control_map, progress, gate
        if hidden_states.ndim == 2:
            batch, _ = hidden_states.shape
            if self.audio_to_control.shape[0] != batch:
                raise ValueError("IPA decode schedule batch does not match VoxCPM batch")
            if decode_step >= self.audio_to_control.shape[1]:
                inactive = torch.full(
                    (batch,), -1, dtype=torch.long, device=hidden_states.device
                )
                zeros = torch.zeros(batch, dtype=hidden_states.dtype, device=hidden_states.device)
                return inactive, zeros, zeros
            return (
                self.audio_to_control[:, decode_step],
                self.audio_progress[:, decode_step],
                self.audio_gate[:, decode_step],
            )
        raise ValueError(
            "VoxCPM projection input must be [B,T,D] or [B,D], "
            f"got {tuple(hidden_states.shape)}"
        )


class AudioStepPronunciationBranch(nn.Module):
    """Low-rank phoneme cross-attention applied to selected audio prediction steps."""

    def __init__(
        self,
        model_dim: int,
        phoneme_dim: int,
        rank: int,
        *,
        progress_features: int = 8,
        position_sigma: float = 0.22,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"IPA adapter rank must be positive, got {rank}")
        if progress_features < 2 or progress_features % 2:
            raise ValueError("IPA progress_features must be an even integer >= 2")
        if position_sigma <= 0:
            raise ValueError("IPA position_sigma must be positive")
        self.rank = int(rank)
        self.progress_features = int(progress_features)
        self.position_sigma = float(position_sigma)
        self.input_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.query = nn.Linear(model_dim, self.rank, bias=False)
        self.progress_query = nn.Linear(self.progress_features, self.rank, bias=False)
        self.key = nn.Linear(phoneme_dim, self.rank, bias=False)
        self.value = nn.Linear(phoneme_dim, self.rank, bias=False)
        self.output = nn.Linear(self.rank, model_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def _progress_embedding(self, progress: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(
            1,
            self.progress_features // 2 + 1,
            device=progress.device,
            dtype=torch.float32,
        )
        angles = progress.float().unsqueeze(-1) * math.pi * frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        control: IPAControl | None,
        *,
        decode_step: int,
    ) -> torch.Tensor:
        if control is None:
            return hidden_states
        control_indices, progress, gate = control.step_values(hidden_states, decode_step)
        active_positions = ((control_indices >= 0) & (gate > 0)).nonzero(as_tuple=False)
        if active_positions.numel() == 0:
            return hidden_states

        if hidden_states.ndim == 3:
            batch_indices = active_positions[:, 0]
            step_indices = active_positions[:, 1]
            active_hidden = hidden_states[batch_indices, step_indices]
            active_control = control_indices[batch_indices, step_indices]
            active_progress = progress[batch_indices, step_indices]
            active_gate = gate[batch_indices, step_indices]
        else:
            batch_indices = active_positions[:, 0]
            step_indices = None
            active_hidden = hidden_states[batch_indices]
            active_control = control_indices[batch_indices]
            active_progress = progress[batch_indices]
            active_gate = gate[batch_indices]

        queries = self.query(self.input_norm(active_hidden))
        progress_embedding = self._progress_embedding(active_progress).to(
            device=queries.device,
            dtype=self.progress_query.weight.dtype,
        )
        queries = queries + self.progress_query(progress_embedding).to(queries.dtype)
        queries = queries.unsqueeze(1)
        memories = control.memories[batch_indices, active_control]
        memory_mask = control.memory_mask[batch_indices, active_control]
        keys = self.key(memories)
        values = self.value(memories)
        scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(self.rank)

        phone_positions = torch.arange(memories.shape[1], device=memories.device).unsqueeze(0)
        phone_lengths = memory_mask.sum(dim=-1).clamp_min(1)
        normalized_phone_positions = phone_positions / (phone_lengths - 1).clamp_min(1).unsqueeze(1)
        position_distance = normalized_phone_positions - active_progress.float().unsqueeze(1)
        position_bias = -0.5 * (position_distance / self.position_sigma).square()
        scores = scores + position_bias.unsqueeze(1).to(scores.dtype)
        scores = scores.masked_fill(~memory_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), values).squeeze(1)
        delta = self.output(attended) * active_gate.to(attended.dtype).unsqueeze(-1)

        result = hidden_states.clone()
        if step_indices is None:
            result[batch_indices] += delta.to(result.dtype)
        else:
            result[batch_indices, step_indices] += delta.to(result.dtype)
        return result


class IPAConditionalLoRABranch(nn.Module):
    """A LoRA delta whose rank-space activations attend to exact IPA memory."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        phoneme_dim: int,
        rank: int,
        *,
        alpha: float,
        progress_features: int = 8,
        position_sigma: float = 0.22,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"Conditional LoRA rank must be positive, got {rank}")
        if progress_features < 2 or progress_features % 2:
            raise ValueError("IPA progress_features must be an even integer >= 2")
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.progress_features = int(progress_features)
        self.position_sigma = float(position_sigma)
        self.input_norm = nn.LayerNorm(int(in_features), elementwise_affine=False)
        self.lora_a = nn.Linear(int(in_features), self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, int(out_features), bias=False)
        self.progress_query = nn.Linear(self.progress_features, self.rank, bias=False)
        self.ipa_key = nn.Linear(int(phoneme_dim), self.rank, bias=False)
        self.ipa_value = nn.Linear(int(phoneme_dim), self.rank, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def _progress_embedding(self, progress: torch.Tensor) -> torch.Tensor:
        frequencies = torch.arange(
            1,
            self.progress_features // 2 + 1,
            device=progress.device,
            dtype=torch.float32,
        )
        angles = progress.float().unsqueeze(-1) * math.pi * frequencies
        return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1)

    def forward(
        self,
        inputs: torch.Tensor,
        output: torch.Tensor,
        control: IPAControl | None,
        *,
        decode_step: int,
    ) -> torch.Tensor:
        if control is None:
            return output
        control_indices, progress, gate = control.step_values(
            inputs,
            decode_step,
            transformer_lm=True,
        )
        active_positions = ((control_indices >= 0) & (gate > 0)).nonzero(as_tuple=False)
        if active_positions.numel() == 0:
            return output

        if inputs.ndim == 3:
            batch_indices = active_positions[:, 0]
            step_indices = active_positions[:, 1]
            active_inputs = inputs[batch_indices, step_indices]
            active_control = control_indices[batch_indices, step_indices]
            active_progress = progress[batch_indices, step_indices]
            active_gate = gate[batch_indices, step_indices]
        else:
            batch_indices = active_positions[:, 0]
            step_indices = None
            active_inputs = inputs[batch_indices]
            active_control = control_indices[batch_indices]
            active_progress = progress[batch_indices]
            active_gate = gate[batch_indices]

        rank_states = self.lora_a(self.input_norm(active_inputs))
        progress_embedding = self._progress_embedding(active_progress).to(
            device=rank_states.device,
            dtype=self.progress_query.weight.dtype,
        )
        queries = rank_states + self.progress_query(progress_embedding).to(rank_states.dtype)
        memories = control.memories[batch_indices, active_control]
        memory_mask = control.memory_mask[batch_indices, active_control]
        keys = self.ipa_key(memories)
        values = self.ipa_value(memories)
        scores = torch.matmul(queries.unsqueeze(1), keys.transpose(1, 2)) / math.sqrt(self.rank)

        phone_positions = torch.arange(memories.shape[1], device=memories.device).unsqueeze(0)
        phone_lengths = memory_mask.sum(dim=-1).clamp_min(1)
        normalized_phone_positions = phone_positions / (phone_lengths - 1).clamp_min(1).unsqueeze(1)
        distance = normalized_phone_positions - active_progress.float().unsqueeze(1)
        scores = scores - 0.5 * (distance / self.position_sigma).square().unsqueeze(1).to(scores.dtype)
        scores = scores.masked_fill(~memory_mask.unsqueeze(1), torch.finfo(scores.dtype).min)
        attended = torch.matmul(torch.softmax(scores, dim=-1), values).squeeze(1)
        delta = self.lora_b(torch.nn.functional.silu(rank_states + attended))
        delta = delta * (self.scaling * active_gate.to(delta.dtype)).unsqueeze(-1)

        result = output.clone()
        if step_indices is None:
            result[batch_indices] += delta.to(result.dtype)
        else:
            result[batch_indices, step_indices] += delta.to(result.dtype)
        return result


class VoxCPMIPAAdapter(nn.Module):
    """Sparrow-warm-started side channel on VoxCPM audio-step LM conditioning."""

    def __init__(
        self,
        *,
        model_dim: int,
        rank: int,
        phoneme_encoder: SparrowPhonemeEncoder,
        progress_features: int = 8,
        position_sigma: float = 0.22,
        lora_alpha: float | None = None,
        lora_target_modules: Sequence[str] = (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        terminal_projection_enabled: bool = True,
    ) -> None:
        super().__init__()
        self.phoneme_encoder = phoneme_encoder
        self.rank = int(rank)
        self.lora_alpha = float(lora_alpha if lora_alpha is not None else rank)
        self.lora_target_modules = frozenset(str(name) for name in lora_target_modules)
        self.terminal_projection_enabled = bool(terminal_projection_enabled)
        self.progress_features = int(progress_features)
        self.position_sigma = float(position_sigma)
        branch_kwargs = {
            "model_dim": int(model_dim),
            "phoneme_dim": phoneme_encoder.hidden_channels,
            "rank": int(rank),
            "progress_features": int(progress_features),
            "position_sigma": float(position_sigma),
        }
        self.branches = nn.ModuleDict(
            {
                "base": AudioStepPronunciationBranch(**branch_kwargs),
                "residual": AudioStepPronunciationBranch(**branch_kwargs),
            }
        )
        self.lora_branches = nn.ModuleDict()
        self.lora_branch_names: dict[str, str] = {}
        self._control: IPAControl | None = None
        self._nano_transformer_control: IPAControl | None = None
        self._nano_terminal_control: IPAControl | None = None
        self._decode_step = 0
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, model: nn.Module) -> None:
        if self._hook_handles:
            raise RuntimeError("IPA adapter is already attached")
        for lm_name, lm in (("base_lm", model.base_lm), ("residual_lm", model.residual_lm)):
            for module_name, module in list(lm.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                short_name = module_name.rsplit(".", 1)[-1]
                if short_name not in self.lora_target_modules:
                    continue
                public_name = f"{lm_name}.{module_name}"
                storage_name = public_name.replace(".", "__")
                branch = IPAConditionalLoRABranch(
                    module.in_features,
                    module.out_features,
                    self.phoneme_encoder.hidden_channels,
                    self.rank,
                    alpha=self.lora_alpha,
                    progress_features=self.progress_features,
                    position_sigma=self.position_sigma,
                ).to(device=module.weight.device, dtype=module.weight.dtype)
                self.lora_branches[storage_name] = branch
                self.lora_branch_names[storage_name] = public_name

                def apply_lora(_module, inputs, output, *, branch_key=storage_name):
                    if len(inputs) != 1:
                        raise RuntimeError(f"Unexpected VoxCPM LoRA arguments for {branch_key}")
                    return self.lora_branches[branch_key](
                        inputs[0],
                        output,
                        self._control,
                        decode_step=self._decode_step,
                    )

                self._hook_handles.append(module.register_forward_hook(apply_lora))

        if not self.lora_branches:
            raise RuntimeError("No VoxCPM LM projections matched the conditional LoRA targets")

        for name, projection in (
            ("base", model.lm_to_dit_proj),
            ("residual", model.res_to_dit_proj),
        ):
            def apply_control(_module, inputs, *, branch=name):
                if len(inputs) != 1:
                    raise RuntimeError(f"Unexpected VoxCPM {branch} projection arguments")
                hidden_states = inputs[0]
                adapted = (
                    self.branches[branch](
                        hidden_states,
                        self._control,
                        decode_step=self._decode_step,
                    )
                    if self.terminal_projection_enabled
                    else hidden_states
                )
                if hidden_states.ndim == 2 and branch == "residual":
                    self._decode_step += 1
                return (adapted,)

            self._hook_handles.append(projection.register_forward_pre_hook(apply_control))

    def attach_nanovllm(self, model: nn.Module) -> None:
        """Attach trained HF projection branches to nano-vLLM's packed projections."""

        if self._hook_handles:
            raise RuntimeError("IPA adapter is already attached")

        def add_branch(storage_name: str, in_features: int, out_features: int) -> None:
            branch = IPAConditionalLoRABranch(
                in_features,
                out_features,
                self.phoneme_encoder.hidden_channels,
                self.rank,
                alpha=self.lora_alpha,
                progress_features=self.progress_features,
                position_sigma=self.position_sigma,
            )
            self.lora_branches[storage_name] = branch
            self.lora_branch_names[storage_name] = storage_name.replace("__", ".")

        def direct_hook(storage_name: str):
            def apply(_module, inputs, output):
                return self.lora_branches[storage_name](
                    inputs[0], output, self._nano_transformer_control, decode_step=0
                )

            return apply

        def split_hook(storage_names: tuple[str, ...], sizes: tuple[int, ...]):
            def apply(_module, inputs, output):
                pieces = output.split(sizes, dim=-1)
                return torch.cat(
                    [
                        self.lora_branches[name](
                            inputs[0], piece, self._nano_transformer_control, decode_step=0
                        )
                        for name, piece in zip(storage_names, pieces)
                    ],
                    dim=-1,
                )

            return apply

        for lm_name, lm in (("base_lm", model.base_lm), ("residual_lm", model.residual_lm)):
            for layer_index, layer in enumerate(lm.layers):
                prefix = f"{lm_name}__layers__{layer_index}"
                attention = layer.self_attn
                qkv_names = tuple(
                    f"{prefix}__self_attn__{name}" for name in ("q_proj", "k_proj", "v_proj")
                )
                qkv_sizes = (attention.q_size, attention.kv_size, attention.kv_size)
                qkv_in = int(attention.qkv_proj.weight.shape[1])
                for name, size in zip(qkv_names, qkv_sizes):
                    add_branch(name, qkv_in, int(size))
                self._hook_handles.append(
                    attention.qkv_proj.register_forward_hook(split_hook(qkv_names, qkv_sizes))
                )

                o_name = f"{prefix}__self_attn__o_proj"
                add_branch(
                    o_name,
                    int(attention.o_proj.weight.shape[1]),
                    int(attention.o_proj.weight.shape[0]),
                )
                self._hook_handles.append(attention.o_proj.register_forward_hook(direct_hook(o_name)))

                mlp = layer.mlp
                gate_up_names = tuple(
                    f"{prefix}__mlp__{name}" for name in ("gate_proj", "up_proj")
                )
                gate_up_sizes = tuple(int(size) for size in mlp.gate_up_proj.output_sizes)
                gate_up_in = int(mlp.gate_up_proj.weight.shape[1])
                for name, size in zip(gate_up_names, gate_up_sizes):
                    add_branch(name, gate_up_in, size)
                self._hook_handles.append(
                    mlp.gate_up_proj.register_forward_hook(split_hook(gate_up_names, gate_up_sizes))
                )

                down_name = f"{prefix}__mlp__down_proj"
                add_branch(
                    down_name,
                    int(mlp.down_proj.weight.shape[1]),
                    int(mlp.down_proj.weight.shape[0]),
                )
                self._hook_handles.append(mlp.down_proj.register_forward_hook(direct_hook(down_name)))

        if not self.lora_branches:
            raise RuntimeError("No nano-vLLM VoxCPM projections matched the IPA adapter")

        for name, projection in (("base", model.lm_to_dit_proj), ("residual", model.res_to_dit_proj)):
            def apply_terminal(_module, inputs, *, branch=name):
                hidden = inputs[0]
                if self.terminal_projection_enabled:
                    hidden = self.branches[branch](
                        hidden, self._nano_terminal_control, decode_step=0
                    )
                return (hidden,)

            self._hook_handles.append(projection.register_forward_pre_hook(apply_terminal))

    def detach(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def encode_controls(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_lengths: torch.Tensor,
        audio_to_control: torch.Tensor,
        audio_progress: torch.Tensor,
        audio_gate: torch.Tensor,
        *,
        prefill_to_control: torch.Tensor | None = None,
        prefill_progress: torch.Tensor | None = None,
        prefill_gate: torch.Tensor | None = None,
        training_lm_to_control: torch.Tensor | None = None,
        training_lm_progress: torch.Tensor | None = None,
        training_lm_gate: torch.Tensor | None = None,
    ) -> IPAControl:
        """Encode padded controls shaped [B,N,P] once for all adapter layers."""

        if phoneme_ids.ndim != 3:
            raise ValueError(f"batched IPA IDs must be [B,N,P], got {tuple(phoneme_ids.shape)}")
        batch, controls, phones = phoneme_ids.shape
        if phoneme_lengths.shape != (batch, controls):
            raise ValueError("IPA lengths do not match batched IPA IDs")
        flat_memory, flat_mask = self.phoneme_encoder(
            phoneme_ids.reshape(batch * controls, phones),
            phoneme_lengths.reshape(batch * controls),
        )
        control = IPAControl(
            memories=flat_memory.reshape(batch, controls, phones, -1),
            memory_mask=flat_mask.reshape(batch, controls, phones),
            audio_to_control=audio_to_control,
            audio_progress=audio_progress,
            audio_gate=audio_gate,
            prefill_to_control=prefill_to_control,
            prefill_progress=prefill_progress,
            prefill_gate=prefill_gate,
            training_lm_to_control=training_lm_to_control,
            training_lm_progress=training_lm_progress,
            training_lm_gate=training_lm_gate,
        )
        control.validate()
        return control

    @contextmanager
    def use_control(self, control: IPAControl | None) -> Iterator[None]:
        if self._control is not None:
            raise RuntimeError("Nested IPA adapter controls are not supported")
        self._control = control
        self._decode_step = 0
        try:
            yield
        finally:
            self._control = None
            self._decode_step = 0

    @contextmanager
    def use_nanovllm_controls(
        self,
        transformer_control: IPAControl | None,
        terminal_control: IPAControl | None,
    ) -> Iterator[None]:
        if self._nano_transformer_control is not None or self._nano_terminal_control is not None:
            raise RuntimeError("Nested nano-vLLM IPA controls are not supported")
        self._nano_transformer_control = transformer_control
        self._nano_terminal_control = terminal_control
        try:
            yield
        finally:
            self._nano_transformer_control = None
            self._nano_terminal_control = None


def load_nanovllm_ipa_adapter(
    checkpoint_dir: str | Path,
    model: nn.Module,
) -> tuple[VoxCPMIPAAdapter, float]:
    """Load and strictly attach a production conditional IPA adapter."""

    from safetensors.torch import load_file

    root = Path(checkpoint_dir).expanduser().resolve()
    config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_cfg = config["adapter"]
    encoder_cfg = config["phoneme_encoder"]
    encoder = SparrowPhonemeEncoder(
        num_symbols=int(encoder_cfg["num_symbols"]),
        hidden_channels=int(encoder_cfg["hidden_channels"]),
        filter_channels=int(encoder_cfg["filter_channels"]),
        num_heads=int(encoder_cfg["num_heads"]),
        num_layers=int(encoder_cfg["num_layers"]),
        kernel_size=int(encoder_cfg["kernel_size"]),
    )
    adapter = VoxCPMIPAAdapter(
        model_dim=int(model.base_lm.config.hidden_size),
        rank=int(adapter_cfg["rank"]),
        phoneme_encoder=encoder,
        progress_features=int(adapter_cfg["progress_features"]),
        position_sigma=float(adapter_cfg["position_sigma"]),
        lora_alpha=float(adapter_cfg["lora_alpha"]),
        lora_target_modules=adapter_cfg["lora_target_modules"],
        terminal_projection_enabled=bool(adapter_cfg["terminal_projection_enabled"]),
    )
    adapter.attach_nanovllm(model)
    adapter.load_state_dict(load_file(str(root / "adapter.safetensors")), strict=True)
    parameter = next(model.parameters())
    adapter.to(device=parameter.device, dtype=parameter.dtype).eval()
    return adapter, float(adapter_cfg.get("fade_out_ratio", 0.2))
