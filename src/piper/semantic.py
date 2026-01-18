import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


_DEFAULT_MODEL_NAME = "distilbert-base-multilingual-cased"
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


class SemanticTokenizer:
    """Thin wrapper around a HuggingFace tokenizer with simple caching."""

    def __init__(self, model_name: Optional[str] = None, max_length: int = 128):
        self.model_name = model_name or _get_model_name()
        self.max_length = max_length
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def encode_texts(self, texts: Optional[List[str]]) -> Optional[SemanticBatch]:
        """Tokenize a list of texts into a SemanticBatch, or None if texts is falsy."""
        if not texts:
            return None

        enc = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

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
        )


def build_bert_input(
    texts: Optional[List[str]],
    tokenizer: Optional[SemanticTokenizer] = None,
) -> Optional[Dict[str, torch.LongTensor]]:
    """Utility to create a dict suitable for BertTextEncoder from a list of texts."""
    if not texts:
        return None

    tok = tokenizer or SemanticTokenizer()
    batch = tok.encode_texts(texts)
    if batch is None:
        return None

    return {
        "input_ids": batch.input_ids,
        "attention_mask": batch.attention_mask,
    }
