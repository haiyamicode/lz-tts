"""Train a contextual text replacement classifier.

Uses a frozen DistilBERT encoder + lightweight classification head to decide
whether a matched token should be replaced in its given context.

Architecture inspired by the heteronym disambiguation model (src/piper/heteronym.py)
but simplified to binary approve/reject instead of multi-variant scoring.

Usage:
    uv run python scripts/train_context_replacer.py \
        --rules data/replacements/rules.jsonl \
        --data local/datasets/context-replace-corpus.jsonl \
        --output data/replacements/best.pt

    # Filter by language
    uv run python scripts/train_context_replacer.py \
        --rules data/replacements/rules.jsonl \
        --data local/datasets/context-replace-corpus.jsonl \
        --language vi-VN \
        --output data/replacements/best.pt

Corpus format (JSONL, ⟦token⟧ markers):
    {"text": "hệ thống ⟦AI⟧ mới ra mắt", "token": "AI", "label": 1, "language": "vi-VN"}
    {"text": "⟦AI⟧ hỏi gì vậy?", "token": "AI", "label": 0, "language": "vi-VN"}

Rules format (JSONL):
    {"token": "AI", "language": "vi-VN", "replacement": "ây ai"}
"""

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOGGER = logging.getLogger(__name__)

_DEFAULT_CONTEXT_MODEL = "distilbert/distilbert-base-multilingual-cased"
_DEFAULT_CONTEXT_HIDDEN = 768
_EMBED_DIM = 256
_DROPOUT = 0.1
_MAX_BERT_TOKENS = 128
_ARCHITECTURE = "context_replacer_v1"
_MARKER_RE = re.compile(r"⟦([^⟧]+)⟧")


class ContextReplacerClassifier(nn.Module):
    """Binary classifier: approve (1) or reject (0) a text replacement in context."""

    def __init__(self, context_hidden_size: int = _DEFAULT_CONTEXT_HIDDEN):
        super().__init__()

        self.target_proj = nn.Sequential(
            nn.Linear(context_hidden_size, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
        )

        self.context_proj = nn.Sequential(
            nn.Linear(context_hidden_size, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=_EMBED_DIM,
            num_heads=4,
            dropout=_DROPOUT,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(_EMBED_DIM * 4, _EMBED_DIM),
            nn.LayerNorm(_EMBED_DIM),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(_EMBED_DIM, _EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(_DROPOUT),
            nn.Linear(_EMBED_DIM // 2, 1),
        )

    def forward(
        self,
        context_hidden: torch.Tensor,
        context_mask: torch.Tensor,
        word_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        context = self.context_proj(context_hidden)
        target_raw = self._masked_mean(context_hidden, word_token_mask & context_mask)
        target = self.target_proj(target_raw)

        target_query = target.unsqueeze(1)
        attended, _ = self.cross_attention(
            target_query,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        attended = attended.squeeze(1)

        features = torch.cat(
            [target, attended, target * attended, target - attended],
            dim=-1,
        )
        return self.classifier(features).squeeze(-1)

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)
        return (values * mask.unsqueeze(-1)).sum(dim=1) / lengths


def _parse_marker_text(raw_text: str) -> Tuple[str, List[Tuple[str, int, int]]]:
    """Parse ⟦token⟧ markers from text.

    Returns:
        (clean_text, [(token, start, end), ...]) where start/end are char offsets
        in the clean text (markers stripped).
    """
    matches = []
    offset = 0
    clean_parts = []
    last_end = 0

    for m in _MARKER_RE.finditer(raw_text):
        token = m.group(1)
        clean_parts.append(raw_text[last_end:m.start()])
        start = len("".join(clean_parts)) - len(clean_parts[-1]) if clean_parts else 0
        start = sum(len(p) for p in clean_parts[:-1])
        clean_parts.append(token)
        end = sum(len(p) for p in clean_parts)
        last_end = m.end()
        matches.append((token, start, end))

    clean_parts.append(raw_text[last_end:])
    clean_text = "".join(clean_parts)

    return clean_text, matches


def load_corpus(path: Path, language: Optional[str] = None) -> List[Dict]:
    """Load corpus with ⟦token⟧ markers, converting to (sentence, token, label) triples.

    Args:
        path: Path to corpus JSONL.
        language: If set, only load samples matching this language code.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if language and entry.get("language") != language:
                continue
            raw_text = entry["text"]
            label = entry["label"]

            clean_text, token_spans = _parse_marker_text(raw_text)
            for token, start, end in token_spans:
                samples.append({
                    "sentence": clean_text,
                    "token": token,
                    "label": label,
                    "language": entry.get("language", ""),
                })
    return samples


def load_rules(path: Path) -> List[Dict]:
    rules = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rules.append(json.loads(line))
    return rules


class ReplacementDataset(Dataset):
    """Dataset of (sentence, token, label) triples for training."""

    def __init__(self, samples: List[Dict], tokenizer, max_length: int = _MAX_BERT_TOKENS):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence = sample["sentence"]
        token = sample["token"]
        label = float(sample["label"])

        encoding = self.tokenizer(
            sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0).bool()
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()

        word_token_mask = torch.zeros(self.max_length, dtype=torch.bool)

        pattern = re.compile(r"\b" + re.escape(token) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(sentence):
            word_start = m.start()
            word_end = m.end()
            for tok_idx, (start, end) in enumerate(offset_mapping):
                if start < word_end and end > word_start:
                    word_token_mask[tok_idx] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "word_token_mask": word_token_mask,
            "label": torch.tensor(label, dtype=torch.float),
        }


def train(
    rules_path: Path,
    data_path: Path,
    output_path: Path,
    language: Optional[str] = None,
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_split: float = 0.15,
    patience: int = 7,
    device: Optional[str] = None,
    context_model: str = _DEFAULT_CONTEXT_MODEL,
):
    if device:
        dev = torch.device(device)
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info("Using device: %s", dev)

    rules = load_rules(rules_path)
    _LOGGER.info("Loaded %d replacement rules", len(rules))

    samples = load_corpus(data_path, language=language)
    _LOGGER.info("Loaded %d samples from %s", len(samples), data_path)

    from transformers import AutoModel, AutoTokenizer

    _LOGGER.info("Loading context model: %s", context_model)
    tokenizer = AutoTokenizer.from_pretrained(context_model, use_fast=True)
    bert = AutoModel.from_pretrained(context_model).to(dev)
    bert.eval()
    for param in bert.parameters():
        param.requires_grad = False

    random.shuffle(samples)
    val_size = max(1, int(len(samples) * val_split))
    val_samples = samples[:val_size]
    train_samples = samples[val_size:]
    _LOGGER.info("Train: %d, Val: %d", len(train_samples), len(val_samples))

    train_ds = ReplacementDataset(train_samples, tokenizer)
    val_ds = ReplacementDataset(val_samples, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ContextReplacerClassifier(
        context_hidden_size=_DEFAULT_CONTEXT_HIDDEN,
    ).to(dev)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
    )
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            word_token_mask = batch["word_token_mask"].to(dev)
            labels = batch["label"].to(dev)

            with torch.no_grad():
                hidden = bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

            logits = model(hidden, attention_mask, word_token_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * input_ids.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += input_ids.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev)
                attention_mask = batch["attention_mask"].to(dev)
                word_token_mask = batch["word_token_mask"].to(dev)
                labels = batch["label"].to(dev)

                hidden = bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state

                logits = model(hidden, attention_mask, word_token_mask)
                loss = criterion(logits, labels)

                val_loss += loss.item() * input_ids.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += input_ids.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        _LOGGER.info(
            "Epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f  lr=%.2e",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
            scheduler.get_last_lr()[0],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            no_improve = 0

            rule_tokens = [r["token"] for r in rules]
            rule_replacements = {r["token"]: r["replacement"] for r in rules}

            ckpt = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "architecture": _ARCHITECTURE,
                "context_model_name": context_model,
                "context_hidden_size": _DEFAULT_CONTEXT_HIDDEN,
                "rule_tokens": rule_tokens,
                "rule_replacements": rule_replacements,
                "language": language or "",
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(ckpt, output_path)
            _LOGGER.info("Saved checkpoint to %s (val_loss=%.4f)", output_path, best_val_loss)
        else:
            no_improve += 1
            if no_improve >= patience:
                _LOGGER.info("Early stopping after %d epochs without improvement", patience)
                break

    _LOGGER.info(
        "Training complete. Best val_loss=%.4f, val_acc=%.4f, saved to %s",
        best_val_loss, best_val_acc, output_path,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rules", type=Path, required=True, help="Replacement rules JSONL")
    parser.add_argument("--data", type=Path, required=True, help="Corpus JSONL with ⟦token⟧ markers")
    parser.add_argument("--output", type=Path, required=True, help="Output checkpoint path")
    parser.add_argument("--language", type=str, default=None, help="Filter corpus by language code (e.g. vi-VN)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--context-model", type=str, default=_DEFAULT_CONTEXT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(
        rules_path=args.rules,
        data_path=args.data,
        output_path=args.output,
        language=args.language,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        patience=args.patience,
        device=args.device,
        context_model=args.context_model,
    )


if __name__ == "__main__":
    main()
