import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
from transformers import AutoTokenizer

from .hf_cache import resolve_hf_tokenizer_path


_DEFAULT_MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"
_LOGGER = logging.getLogger("vits.semantic")
_DEBUG_SEMANTIC = bool(int(os.environ.get("PIPER_SEMANTIC_DEBUG", "0")))

# Hugging Face tokenizers and Python multiprocessing don't mix well when
# tokenizers are initialized before a fork. Disable parallelism to avoid
# deadlocks and suppress the noisy warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _get_model_name() -> str:
    return os.environ.get("PIPER_SEMANTIC_MODEL_NAME", _DEFAULT_MODEL_NAME)


@dataclass
class SemanticBatch:
    """Container for batched semantic encoder inputs."""

    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    word2ph: Optional[torch.LongTensor] = None


class SemanticTokenizer:
    """Thin wrapper around a HuggingFace tokenizer with simple caching."""

    def __init__(self, model_name: Optional[str] = None, max_length: Optional[int] = None):
        self.model_name = model_name or _get_model_name()
        self.max_length = max_length
        local_files_only = any(
            os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
            for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
        )
        model_path = resolve_hf_tokenizer_path(self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            use_fast=True,
        )

    def encode_texts(
        self,
        texts: Optional[List[str]],
        phoneme_lengths: Optional[Sequence[int]] = None,
        word_spans: Optional[Sequence[Optional[Sequence[Sequence[int]]]]] = None,
    ) -> Optional[SemanticBatch]:
        """Tokenize a list of texts into a SemanticBatch, or None if texts is falsy."""
        if not texts:
            return None

        needs_word2ph = phoneme_lengths is not None
        tokenizer_kwargs = {
            "padding": True,
            "truncation": False,
            "return_tensors": "pt",
            "return_offsets_mapping": needs_word2ph,
            "return_special_tokens_mask": needs_word2ph,
        }
        if self.max_length is not None:
            tokenizer_kwargs["padding"] = "max_length"
            tokenizer_kwargs["truncation"] = True
            tokenizer_kwargs["max_length"] = self.max_length

        enc = self._tokenizer(texts, **tokenizer_kwargs)
        offsets = enc.pop("offset_mapping") if needs_word2ph else None
        special_tokens_mask = enc.pop("special_tokens_mask") if needs_word2ph else None

        word2ph_tensor = None
        if needs_word2ph:
            assert offsets is not None
            assert special_tokens_mask is not None
            spans = word_spans or [None] * len(texts)
            counts = [
                _build_word2ph_counts(
                    phoneme_length=int(phoneme_lengths[idx]),
                    offsets=offsets[idx].tolist(),
                    attention_mask=enc["attention_mask"][idx].tolist(),
                    special_tokens_mask=special_tokens_mask[idx].tolist(),
                    word_spans=spans[idx] if idx < len(spans) else None,
                )
                for idx in range(len(texts))
            ]
            word2ph_tensor = torch.LongTensor(counts)

        if _DEBUG_SEMANTIC:
            _LOGGER.debug(
                "SemanticTokenizer: model=%s, batch_size=%s, max_length=%s, example_text[0]=%r",
                self.model_name,
                len(texts),
                self.max_length,
                texts[0] if texts else None,
            )

        return SemanticBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            word2ph=word2ph_tensor,
        )


def _split_count(total: int, weights: Sequence[int]) -> List[int]:
    if total <= 0 or not weights:
        return [0 for _ in weights]

    positive_weights = [max(0, int(weight)) for weight in weights]
    weight_sum = sum(positive_weights)
    if weight_sum <= 0:
        base = total // len(weights)
        out = [base for _ in weights]
        for idx in range(total - base * len(weights)):
            out[idx] += 1
        return out

    out = [(total * weight) // weight_sum for weight in positive_weights]
    remainder = total - sum(out)
    ranked = sorted(
        range(len(weights)),
        key=lambda idx: (total * positive_weights[idx]) % weight_sum,
        reverse=True,
    )
    for idx in ranked[:remainder]:
        out[idx] += 1
    return out


def _token_overlaps(
    token_start: int,
    token_end: int,
    span_start: int,
    span_end: int,
) -> int:
    return max(0, min(token_end, span_end) - max(token_start, span_start))


def _normalize_word_spans(
    word_spans: Optional[Sequence[Sequence[int]]],
) -> List[tuple[int, int, int, int]]:
    spans: List[tuple[int, int, int, int]] = []
    if not word_spans:
        return spans

    for raw in word_spans:
        if raw is None or len(raw) < 4:
            continue
        text_start, text_end, ph_start, ph_end = [int(value) for value in raw[:4]]
        if text_end <= text_start or ph_end <= ph_start:
            continue
        spans.append((text_start, text_end, ph_start, ph_end))

    spans.sort(key=lambda item: (item[2], item[3], item[0], item[1]))
    return spans


def _build_word2ph_counts(
    phoneme_length: int,
    offsets: Sequence[Sequence[int]],
    attention_mask: Sequence[int],
    special_tokens_mask: Sequence[int],
    word_spans: Optional[Sequence[Sequence[int]]],
) -> List[int]:
    """Build word-to-phoneme token repeat counts for Piper phoneme ids.

    `word_spans` are 0-based [text_start, text_end, phoneme_start, phoneme_end]
    spans over raw eSpeak phonemes. Piper phoneme ids are laid out as:
    BOS, blank, (phoneme, blank)*, EOS. Therefore raw phoneme span [s, e)
    maps to id span [2 + 2*s, 2 + 2*e).
    """
    if phoneme_length <= 0:
        return [0 for _ in attention_mask]

    spans = _normalize_word_spans(word_spans)
    if not spans:
        raise ValueError(
            "word_spans are required when building phoneme-aligned BERT input"
        )

    active_tokens = [idx for idx, value in enumerate(attention_mask) if value]
    if not active_tokens:
        raise ValueError("tokenizer produced no active tokens for phoneme-aligned BERT input")

    first_special = next(
        (idx for idx in active_tokens if special_tokens_mask[idx]),
        active_tokens[0],
    )
    last_special = next(
        (idx for idx in reversed(active_tokens) if special_tokens_mask[idx]),
        active_tokens[-1],
    )

    counts = [0 for _ in attention_mask]
    previous_id_end = 0
    previous_word_token: Optional[int] = None

    for text_start, text_end, ph_start, ph_end in spans:
        id_start = min(max(0, 2 + (2 * ph_start)), phoneme_length)
        id_end = min(max(id_start, 2 + (2 * ph_end)), phoneme_length)

        gap_count = max(0, id_start - previous_id_end)
        if gap_count:
            counts[previous_word_token if previous_word_token is not None else first_special] += gap_count

        token_indices: List[int] = []
        weights: List[int] = []
        for token_idx in active_tokens:
            if special_tokens_mask[token_idx]:
                continue
            token_start, token_end = [int(value) for value in offsets[token_idx]]
            overlap = _token_overlaps(token_start, token_end, text_start, text_end)
            if overlap > 0:
                token_indices.append(token_idx)
                weights.append(overlap)

        span_count = max(0, id_end - id_start)
        if token_indices and span_count:
            for token_idx, count in zip(token_indices, _split_count(span_count, weights)):
                counts[token_idx] += count
            previous_word_token = token_indices[-1]
        elif span_count:
            raise ValueError(
                "word span does not overlap any non-special BERT token: "
                f"text_span=({text_start}, {text_end}), phoneme_span=({ph_start}, {ph_end})"
            )

        previous_id_end = id_end

    trailing_count = max(0, phoneme_length - previous_id_end)
    if trailing_count:
        counts[last_special if last_special is not None else active_tokens[-1]] += trailing_count

    if sum(counts) != phoneme_length:
        raise ValueError(
            f"word2ph count mismatch: sum={sum(counts)} phoneme_length={phoneme_length}"
        )

    return counts


def build_bert_input(
    texts: Optional[List[str]],
    tokenizer: Optional[SemanticTokenizer] = None,
    phoneme_lengths: Optional[Sequence[int]] = None,
    word_spans: Optional[Sequence[Optional[Sequence[Sequence[int]]]]] = None,
) -> Optional[Dict[str, torch.LongTensor]]:
    """Utility to create a dict suitable for BertTextEncoder from a list of texts."""
    if not texts:
        return None

    tok = tokenizer or SemanticTokenizer()
    batch = tok.encode_texts(
        texts,
        phoneme_lengths=phoneme_lengths,
        word_spans=word_spans,
    )
    if batch is None:
        return None

    result = {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
    }
    if batch.word2ph is not None:
        result["word2ph"] = batch.word2ph

    return result


def align_phone_features(
    hidden: torch.Tensor,
    word2ph: torch.Tensor,
    phone_len: int,
) -> torch.Tensor:
    """Expand token-level semantic features into phone-level features.

    Returns a tensor shaped ``[hidden_dim, phone_len]``, matching the sidecar
    feature tensors used by the Piper precomputed-BERT training path.
    """
    counts = torch.clamp(word2ph.to(device=hidden.device, dtype=torch.long), min=0)
    diff = int(phone_len) - int(counts.sum().item())
    if diff:
        active = torch.nonzero(counts > 0, as_tuple=False).flatten()
        adjust_idx = int(active[-1].item()) if active.numel() else max(0, counts.numel() - 1)
        counts = counts.clone()
        counts[adjust_idx] = torch.clamp(counts[adjust_idx] + diff, min=0)

    repeated = torch.repeat_interleave(hidden, counts, dim=0)
    if repeated.size(0) == 0:
        return hidden.new_zeros((hidden.size(-1), int(phone_len)))

    if repeated.size(0) < phone_len:
        pad = repeated.new_zeros((phone_len - repeated.size(0), repeated.size(1)))
        repeated = torch.cat([repeated, pad], dim=0)
    elif repeated.size(0) > phone_len:
        repeated = repeated[:phone_len]

    return repeated.transpose(0, 1).contiguous()
