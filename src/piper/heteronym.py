"""Heteronym disambiguation phonemizer.

Uses a trained BERT-based model to resolve heteronym pronunciations based on context.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

_LOGGER = logging.getLogger(__name__)

# Model architecture constants (must match training)
_BERT_HIDDEN = 768
_EMBED_DIM = 256
_DROPOUT = 0.1


class _HeteronymClassifier(nn.Module):
    """Lightweight classifier for heteronym disambiguation."""

    def __init__(self, vocab_size: int, num_variants: int, variant_phoneme_ids: torch.Tensor):
        super().__init__()

        self.context_proj = nn.Sequential(
            nn.Linear(_BERT_HIDDEN, _EMBED_DIM),
            nn.ReLU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(_EMBED_DIM, _EMBED_DIM),
        )

        self.phoneme_embed = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=_EMBED_DIM,
            nhead=4,
            dim_feedforward=_EMBED_DIM * 4,
            dropout=_DROPOUT,
            batch_first=True,
        )
        self.phoneme_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.phoneme_proj = nn.Linear(_EMBED_DIM, _EMBED_DIM)

        self.register_buffer("variant_phoneme_ids", variant_phoneme_ids)
        self.num_variants = num_variants
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_variants(self):
        embedded = self.phoneme_embed(self.variant_phoneme_ids)
        mask = self.variant_phoneme_ids == 0
        encoded = self.phoneme_encoder(embedded, src_key_padding_mask=mask)
        lengths = (~mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (encoded * (~mask).unsqueeze(-1)).sum(dim=1) / lengths
        return nn.functional.normalize(self.phoneme_proj(pooled), dim=-1)

    def forward(self, word_emb, sample_variant_map, sample_variant_mask):
        context_emb = nn.functional.normalize(self.context_proj(word_emb), dim=-1)
        variant_embs = self.encode_variants()
        all_scores = torch.matmul(context_emb, variant_embs.T) / self.temperature.abs().clamp(min=0.01)
        scores = torch.gather(all_scores, 1, sample_variant_map)
        scores = scores.masked_fill(~sample_variant_mask, float("-inf"))
        return scores


class HeteronymResolver:
    """Resolves heteronym pronunciations using a trained model."""

    def __init__(
        self,
        checkpoint_path: Path,
        heretonyms_path: Path,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._model: Optional[_HeteronymClassifier] = None
        self._bert = None
        self._tokenizer = None
        self._heretonyms: Dict[str, List[str]] = {}
        self._variant_list: List[str] = []
        self._variant_to_idx: Dict[str, int] = {}
        self._phoneme_to_idx: Dict[str, int] = {}

        self._checkpoint_path = checkpoint_path
        self._heretonyms_path = heretonyms_path
        self._loaded = False

    def _load(self):
        """Lazy load model and resources."""
        if self._loaded:
            return

        _LOGGER.info(f"Loading heteronym model from {self._checkpoint_path}")

        # Load heretonyms dictionary
        with open(self._heretonyms_path) as f:
            for line in f:
                data = json.loads(line)
                self._heretonyms[data["word"].lower()] = data.get("variants", [])

        # Load checkpoint
        ckpt = torch.load(self._checkpoint_path, map_location=self.device)
        self._phoneme_to_idx = ckpt["phoneme_to_idx"]
        self._variant_list = ckpt["variant_list"]
        self._variant_to_idx = {v: i for i, v in enumerate(self._variant_list)}

        # Build variant phoneme IDs
        max_len = max(len(v) for v in self._variant_list)
        variant_phoneme_ids = torch.zeros(len(self._variant_list), max_len, dtype=torch.long)
        for i, v in enumerate(self._variant_list):
            for j, p in enumerate(v):
                variant_phoneme_ids[i, j] = self._phoneme_to_idx.get(p, 0)

        # Load model
        self._model = _HeteronymClassifier(
            vocab_size=len(self._phoneme_to_idx),
            num_variants=len(self._variant_list),
            variant_phoneme_ids=variant_phoneme_ids.to(self.device),
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # Load BERT
        from transformers import AutoModel, AutoTokenizer

        bert_model = "distilbert-base-multilingual-cased"
        self._tokenizer = AutoTokenizer.from_pretrained(bert_model)
        self._bert = AutoModel.from_pretrained(bert_model).to(self.device)
        self._bert.eval()

        self._loaded = True
        _LOGGER.info("Heteronym model loaded successfully")

    def get_heteronyms(self) -> Dict[str, List[str]]:
        """Return the heteronyms dictionary."""
        self._load()
        return self._heretonyms

    def resolve(self, text: str, word: str, word_start: int, word_end: int) -> Optional[str]:
        """Resolve a heteronym's pronunciation given its context.

        Args:
            text: The full sentence (clean, no markers)
            word: The heteronym word
            word_start: Character start index of the word in text
            word_end: Character end index of the word in text

        Returns:
            The predicted phoneme variant (IPA), or None if word is not a known heteronym.
        """
        self._load()

        word_lower = word.lower()
        variants = self._heretonyms.get(word_lower)
        if not variants:
            return None

        # Tokenize and get BERT embedding
        encoding = self._tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

        # Find token indices for the word
        word_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start < word_end and end > word_start:
                word_indices.append(i)

        # Check if all variants are in the trained model
        # Skip words whose variants weren't included in training
        for v in variants:
            if v not in self._variant_to_idx:
                _LOGGER.debug(
                    "Skipping heteronym '%s': variant '%s' not in trained model",
                    word, v
                )
                return None

        with torch.no_grad():
            hidden = self._bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            if word_indices:
                word_emb = hidden[0, word_indices, :].mean(dim=0, keepdim=True)
            else:
                word_emb = hidden[0, 0:1, :]

            # Build variant map
            var_map = torch.tensor(
                [[self._variant_to_idx[v] for v in variants]], device=self.device
            )
            var_mask = torch.ones_like(var_map, dtype=torch.bool)

            # Pad if needed
            max_v = self._model.num_variants
            if var_map.size(1) < max_v:
                pad = torch.zeros(1, max_v - var_map.size(1), dtype=torch.long, device=self.device)
                var_map = torch.cat([var_map, pad], dim=1)
                var_mask = torch.cat([var_mask, torch.zeros_like(pad, dtype=torch.bool)], dim=1)

            logits = self._model(word_emb, var_map, var_mask)
            pred_idx = logits.argmax(dim=-1).item()

        return variants[pred_idx]

    def resolve_all(self, text: str) -> List[Tuple[str, int, int, str]]:
        """Find and resolve all heteronyms in text.

        Args:
            text: Input text

        Returns:
            List of (word, start, end, phoneme) tuples for each resolved heteronym.
        """
        self._load()

        results = []
        # Find all words and check if they're heteronyms
        for match in re.finditer(r"\b(\w+)\b", text):
            word = match.group(1)
            if word.lower() in self._heretonyms:
                start, end = match.start(), match.end()
                phoneme = self.resolve(text, word, start, end)
                if phoneme:
                    results.append((word, start, end, phoneme))

        return results


# Global resolver instance (lazy initialized)
_resolver: Optional[HeteronymResolver] = None


def get_resolver(
    checkpoint_path: Optional[Path] = None,
    heretonyms_path: Optional[Path] = None,
    device: Optional[str] = None,
) -> HeteronymResolver:
    """Get or create the global heteronym resolver.

    Args:
        checkpoint_path: Path to model checkpoint. Defaults to package resources.
        heretonyms_path: Path to heretonyms.jsonl. Defaults to package resources.
        device: Device to use ('cpu', 'cuda', etc.)

    Returns:
        HeteronymResolver instance
    """
    global _resolver

    if _resolver is None:
        if checkpoint_path is None:
            # Default paths relative to package
            pkg_dir = Path(__file__).parent.parent.parent
            checkpoint_path = pkg_dir / "data" / "heteronyms" / "best.pt"
        if heretonyms_path is None:
            pkg_dir = Path(__file__).parent.parent.parent
            heretonyms_path = pkg_dir / "data" / "heteronyms" / "heretonyms.jsonl"

        _resolver = HeteronymResolver(checkpoint_path, heretonyms_path, device)

    return _resolver
