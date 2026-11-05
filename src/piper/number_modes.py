"""Contextual reading mode for each number in a text.

The context replacer head with one output per mode, on the frozen multilingual
DistilBERT, picks how each number reads from its context: cardinal ("1982
apples"), year ("in 1982"), ordinal ("May 3") or digit by digit ("PIN 4821").
``text_norm.spoken`` then reads it in that mode with each language's rules.

Usage:
    from piper.number_modes import get_number_modes
    from text_norm.spoken import verbalize_spoken_text

    verbalize_spoken_text("In 1982 we had 1982 apples.", "en", get_number_modes())
    # "In nineteen eighty-two we had one thousand, nine hundred and eighty-two apples."
"""

from __future__ import annotations

import logging
import os
import threading
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from .context_replacer import (
    _DEFAULT_CONTEXT_HIDDEN,
    _DEFAULT_CONTEXT_MODEL,
    _MAX_BERT_TOKENS,
    ContextReplacerClassifier,
)
from .hf_cache import get_shared_hf_encoder, resolve_hf_tokenizer_path

_LOGGER = logging.getLogger(__name__)

ARCHITECTURE = "number_modes_v1"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "data" / "number-modes" / "best.pt"


@dataclass
class SpanWindows:
    """Token windows around spans: unique windows plus each span's window and tokens."""

    input_ids: torch.Tensor  # [windows, length]
    attention_mask: torch.Tensor  # [windows, length] bool
    window_index: torch.Tensor  # [spans]
    word_token_mask: torch.Tensor  # [spans, length] bool


def span_windows(tokenizer, text: str, spans: List[Tuple[int, int]]) -> SpanWindows:
    """Center a window of at most ``_MAX_BERT_TOKENS`` tokens on each span.

    The text is tokenized once; spans whose windows coincide (every span of a
    short text) share one encoder row.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        verbose=False,
    )
    ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    starts = [start for start, _ in offsets]
    budget = _MAX_BERT_TOKENS - 2

    windows: dict[Tuple[int, int], int] = {}
    span_rows: List[Tuple[int, int, int, int]] = []
    for start, end in spans:
        covered = [i for i, (left, right) in enumerate(offsets) if left < end and right > start]
        if not covered:
            nearest = min(bisect_left(starts, start), max(len(ids) - 1, 0))
            covered = [nearest]
        first, last = covered[0], min(covered[-1], covered[0] + budget - 1)
        if len(ids) <= budget:
            left = 0
        else:
            left = first - (budget - (last - first + 1)) // 2
            left = max(0, min(left, len(ids) - budget))
        right = min(len(ids), left + budget)
        window = windows.setdefault((left, right), len(windows))
        span_rows.append((window, left, first, last))

    length = max(right - left for left, right in windows) + 2 if windows else 2
    input_ids = torch.full((len(windows), length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(windows), length), dtype=torch.bool)
    for (left, right), row in windows.items():
        window_ids = [tokenizer.cls_token_id, *ids[left:right], tokenizer.sep_token_id]
        input_ids[row, : len(window_ids)] = torch.tensor(window_ids)
        attention_mask[row, : len(window_ids)] = True

    word_token_mask = torch.zeros((len(span_rows), length), dtype=torch.bool)
    for i, (_, left, first, last) in enumerate(span_rows):
        word_token_mask[i, first - left + 1 : last - left + 2] = True
    window_index = torch.tensor([row for row, *_ in span_rows], dtype=torch.long)
    return SpanWindows(input_ids, attention_mask, window_index, word_token_mask)


class NumberModeResolver:
    """``text_norm.spoken.NumberModes`` backed by the trained classifier."""

    def __init__(self, checkpoint_path: Path = DEFAULT_CHECKPOINT, device: Optional[str] = None):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.labels: List[str] = []
        self._model: Optional[ContextReplacerClassifier] = None
        self._bert = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            _LOGGER.info("Loading number mode classifier from %s", self.checkpoint_path)
            ckpt = torch.load(self.checkpoint_path, map_location=self.device)
            context_model = ckpt.get("context_model_name", _DEFAULT_CONTEXT_MODEL)
            self.labels = list(ckpt["labels"])

            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                resolve_hf_tokenizer_path(context_model),
                local_files_only=True,
                use_fast=True,
            )
            self._bert = get_shared_hf_encoder(
                context_model,
                device=self.device,
                dtype=torch.float32,
                local_files_only=any(
                    os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
                    for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
                ),
            )
            model = ContextReplacerClassifier(
                context_hidden_size=ckpt.get("context_hidden_size", _DEFAULT_CONTEXT_HIDDEN),
                num_labels=len(self.labels),
            ).to(self.device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            self._model = model

    def __call__(self, text: str, spans: List[Tuple[int, int]]) -> List[str]:
        if not spans:
            return []
        self.load()
        windows = span_windows(self._tokenizer, text, spans)
        with torch.inference_mode():
            hidden = self._bert(
                input_ids=windows.input_ids.to(self.device),
                attention_mask=windows.attention_mask.to(self.device),
            ).last_hidden_state
            index = windows.window_index.to(self.device)
            logits = self._model(
                hidden.index_select(0, index),
                windows.attention_mask.to(self.device).index_select(0, index),
                windows.word_token_mask.to(self.device),
            )
        return [self.labels[i] for i in logits.argmax(dim=-1).tolist()]


_resolver: Optional[NumberModeResolver] = None
_resolver_lock = threading.Lock()


def get_number_modes(
    checkpoint_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> NumberModeResolver:
    """Get or create the global number mode resolver."""
    global _resolver
    with _resolver_lock:
        if _resolver is None:
            _resolver = NumberModeResolver(checkpoint_path or DEFAULT_CHECKPOINT, device)
        return _resolver
