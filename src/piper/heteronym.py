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
_DEFAULT_CONTEXT_MODEL = "distilbert-base-multilingual-cased"
_DEFAULT_CONTEXT_HIDDEN = 768
_EMBED_DIM = 256
_DROPOUT = 0.1
_MAX_BERT_TOKENS = 128
_ARCHITECTURE = "attention_v3"


class _HeteronymClassifier(nn.Module):
    """Attention-based classifier for heteronym disambiguation."""

    def __init__(
        self,
        vocab_size: int,
        num_variants: int,
        variant_phoneme_ids: torch.Tensor,
        context_hidden_size: int = _DEFAULT_CONTEXT_HIDDEN,
    ):
        super().__init__()

        max_phoneme_len = variant_phoneme_ids.size(1)
        self.context_proj = nn.Sequential(
            nn.Linear(context_hidden_size, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
        )
        self.target_proj = nn.Sequential(
            nn.Linear(context_hidden_size, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
        )

        self.phoneme_embed = nn.Embedding(vocab_size, _EMBED_DIM, padding_idx=0)
        self.variant_id_embed = nn.Embedding(num_variants, _EMBED_DIM)
        self.variant_cls = nn.Parameter(torch.zeros(1, 1, _EMBED_DIM))
        self.variant_pos_embed = nn.Parameter(torch.randn(1, max_phoneme_len + 1, _EMBED_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=_EMBED_DIM,
            nhead=4,
            dim_feedforward=_EMBED_DIM * 4,
            dropout=_DROPOUT,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.variant_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.variant_norm = nn.LayerNorm(_EMBED_DIM)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=_EMBED_DIM,
            num_heads=4,
            dropout=_DROPOUT,
            batch_first=True,
        )
        self.score = nn.Sequential(
            nn.Linear(_EMBED_DIM * 4, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(_EMBED_DIM, 1),
        )

        self.register_buffer("variant_phoneme_ids", variant_phoneme_ids)
        self.num_variants = num_variants

    def encode_variants(self):
        phoneme_emb = self.phoneme_embed(self.variant_phoneme_ids)
        cls = self.variant_cls.expand(phoneme_emb.size(0), -1, -1)
        encoded = torch.cat([cls, phoneme_emb], dim=1) + self.variant_pos_embed
        phoneme_mask = self.variant_phoneme_ids == 0
        cls_mask = torch.zeros(phoneme_mask.size(0), 1, dtype=torch.bool, device=phoneme_mask.device)
        padding_mask = torch.cat([cls_mask, phoneme_mask], dim=1)
        encoded = self.variant_encoder(encoded, src_key_padding_mask=padding_mask)
        variant_ids = torch.arange(self.num_variants, device=self.variant_phoneme_ids.device)
        return self.variant_norm(encoded[:, 0] + self.variant_id_embed(variant_ids))

    @staticmethod
    def _masked_mean(values, mask):
        mask = mask.bool()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / lengths

    def forward(self, context_hidden, context_mask, word_token_mask, sample_variant_map, sample_variant_mask):
        context = self.context_proj(context_hidden)
        target_raw = self._masked_mean(context_hidden, word_token_mask & context_mask)
        target = self.target_proj(target_raw)

        variant_embs = self.encode_variants()
        batch_size, max_variants = sample_variant_map.shape
        candidates = variant_embs[sample_variant_map.reshape(-1)].view(batch_size, max_variants, -1)
        flat_candidates = candidates.reshape(batch_size * max_variants, 1, _EMBED_DIM)
        flat_context = context.unsqueeze(1).expand(-1, max_variants, -1, -1).reshape(
            batch_size * max_variants,
            context.size(1),
            _EMBED_DIM,
        )
        flat_context_mask = context_mask.unsqueeze(1).expand(-1, max_variants, -1).reshape(
            batch_size * max_variants,
            context_mask.size(1),
        )
        attended, _ = self.cross_attention(
            flat_candidates,
            flat_context,
            flat_context,
            key_padding_mask=~flat_context_mask.bool(),
            need_weights=False,
        )
        attended = attended.squeeze(1).view(batch_size, max_variants, _EMBED_DIM)
        target = target.unsqueeze(1).expand(-1, max_variants, -1)
        features = torch.cat([candidates, attended, target, candidates * target], dim=-1)
        scores = self.score(features).squeeze(-1)
        return scores.masked_fill(~sample_variant_mask, float("-inf"))


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
        self._context_model_name = _DEFAULT_CONTEXT_MODEL

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
        self._context_model_name = ckpt.get("context_model_name", _DEFAULT_CONTEXT_MODEL)
        context_hidden_size = ckpt.get("context_hidden_size", _DEFAULT_CONTEXT_HIDDEN)

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
            context_hidden_size=context_hidden_size,
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # Load frozen context encoder
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._context_model_name)
        self._bert = AutoModel.from_pretrained(self._context_model_name).to(self.device)
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

        encoding = self._tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=_MAX_BERT_TOKENS,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device).bool()
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

        # Find token indices for the word
        word_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if start < word_end and end > word_start:
                word_indices.append(i)
        word_token_mask = torch.zeros(1, _MAX_BERT_TOKENS, dtype=torch.bool, device=self.device)
        if word_indices:
            word_token_mask[0, word_indices] = True
        else:
            word_token_mask[0, 0] = True

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

            logits = self._model(hidden, attention_mask, word_token_mask, var_map, var_mask)
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
