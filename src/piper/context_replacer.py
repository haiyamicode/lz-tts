"""Context-aware text replacement resolver.

Uses ICU word segmentation + a lookup dict built at load time to find matching
tokens, then either auto-replaces (always_replace) or sends a batch through a
trained binary classifier.

Usage:
    from piper.context_replacer import get_replacer

    replacer = get_replacer()
    result = replacer.apply_replacements("hệ thống AI mới ra mắt", language="vi-VN")
    # result: "hệ thống ây ai mới ra mắt"
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .hf_cache import get_shared_hf_encoder, resolve_hf_model_path
from .word_segmentation import icu_word_spans as _icu_word_spans

_LOGGER = logging.getLogger(__name__)

_DEFAULT_CONTEXT_MODEL = "distilbert/distilbert-base-multilingual-cased"
_DEFAULT_CONTEXT_HIDDEN = 768
_EMBED_DIM = 256
_DROPOUT = 0.1
_MAX_BERT_TOKENS = 128


@dataclass
class _RuleEntry:
    token: str
    replacement: str
    language: str = ""
    always_replace: bool = False


_TRANSFORMS: Dict[str, Callable] = {}


def _get_num2words(lang: str) -> Callable:
    """Get a num2words function for a specific language, cached."""
    key = f"num2words:{lang}"
    if key not in _TRANSFORMS:
        from num2words import num2words as _n2w

        def _convert(s: str, _lang: str = lang) -> str:
            normalized = s.strip()
            if re.fullmatch(r"\d{1,3}(?:[.,]\d{3}){1,}", normalized):
                value = int(re.sub(r"[.,]", "", normalized))
            elif re.fullmatch(r"\d+[.,]\d+", normalized):
                value = float(normalized.replace(",", "."))
            else:
                value = int(normalized)
            return _n2w(value, lang=_lang)

        _TRANSFORMS[key] = _convert
    return _TRANSFORMS[key]


@dataclass
class _PatternRule:
    pattern: re.Pattern
    replacement_template: str
    language: str = ""
    always_replace: bool = True
    transforms: Dict[int, Callable] = field(default_factory=dict)

    def expand(self, match: re.Match) -> str:
        result = self.replacement_template
        for i, group in enumerate(match.groups(), 1):
            value = group
            fn = self.transforms.get(i)
            if fn:
                value = fn(group)
            result = result.replace(f"${i}", value)
        return result


@dataclass
class _Match:
    start: int
    end: int
    word: str
    rule: _RuleEntry
    text_idx: int = 0


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


def _segment_and_find_matches(
    text: str,
    text_idx: int,
    lookup: Dict[str, _RuleEntry],
    language: Optional[str] = None,
) -> List[_Match]:
    """Segment text with ICU, then space-split safety, then look up in dict."""
    locale = language or "und"
    icu_spans = _icu_word_spans(text, locale)

    matches: List[_Match] = []
    for span_start, span_end, piece in icu_spans:
        sub_start = span_start
        for sub_piece in piece.split():
            sub_end = sub_start + len(sub_piece)
            entry = lookup.get(sub_piece.lower())
            if entry is not None:
                if not (language and entry.language and entry.language != language):
                    matches.append(_Match(
                        start=sub_start,
                        end=sub_end,
                        word=sub_piece,
                        rule=_RuleEntry(
                            token=entry.token,
                            replacement=entry.replacement,
                            language=entry.language,
                            always_replace=entry.always_replace,
                        ),
                        text_idx=text_idx,
                    ))
            sub_start = sub_end + 1

    matches.sort(key=lambda m: m.start)
    return matches


def _find_pattern_matches(
    text: str,
    text_idx: int,
    pattern_rules: List[_PatternRule],
    language: Optional[str] = None,
) -> List[_Match]:
    """Run regex pattern rules against text, return matches with computed replacements."""
    matches: List[_Match] = []
    for pr in pattern_rules:
        if language and pr.language and pr.language != language:
            continue
        for m in pr.pattern.finditer(text):
            replacement = pr.expand(m)
            matches.append(_Match(
                start=m.start(),
                end=m.end(),
                word=m.group(0),
                rule=_RuleEntry(
                    token=m.group(0),
                    replacement=replacement,
                    language=pr.language,
                    always_replace=pr.always_replace,
                ),
                text_idx=text_idx,
            ))
    matches.sort(key=lambda m: m.start)
    return matches


class ContextReplacer:
    """Resolves text replacements using ICU segmentation + trained classifier."""

    def __init__(
        self,
        checkpoint_path: Path,
        rules_path: Optional[Path] = None,
        device: Optional[str] = None,
        threshold: float = 0.5,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = threshold
        self._model: Optional[ContextReplacerClassifier] = None
        self._bert = None
        self._tokenizer = None
        self._context_model_name = _DEFAULT_CONTEXT_MODEL

        self._lookup: Dict[str, _RuleEntry] = {}
        self._pattern_rules: List[_PatternRule] = []

        self._checkpoint_path = checkpoint_path
        self._rules_path = rules_path
        self._loaded = False

    def _load(self):
        if self._loaded:
            return

        _LOGGER.info("Loading context replacer from %s", self._checkpoint_path)

        ckpt = torch.load(self._checkpoint_path, map_location=self.device)
        self._context_model_name = ckpt.get("context_model_name", _DEFAULT_CONTEXT_MODEL)
        context_hidden_size = ckpt.get("context_hidden_size", _DEFAULT_CONTEXT_HIDDEN)

        # Build lookup from checkpoint rules + rules file
        ckpt_tokens = ckpt.get("rule_tokens", [])
        ckpt_replacements = ckpt.get("rule_replacements", {})
        for token in ckpt_tokens:
            self._lookup[token.lower()] = _RuleEntry(
                token=token,
                replacement=ckpt_replacements.get(token, ""),
            )

        if self._rules_path and self._rules_path.exists():
            with open(self._rules_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rule = json.loads(line)
                    if "pattern" in rule:
                        lang = rule.get("language", "")
                        base_lang = lang.split("-")[0] if lang else "en"
                        raw_transforms = rule.get("transforms", {})
                        resolved_transforms: Dict[int, Callable] = {}
                        for group_str, fn_name in raw_transforms.items():
                            group_idx = int(group_str)
                            if fn_name == "num2words":
                                resolved_transforms[group_idx] = _get_num2words(base_lang)
                        self._pattern_rules.append(_PatternRule(
                            pattern=re.compile(rule["pattern"]),
                            replacement_template=rule["replacement"],
                            language=lang,
                            always_replace=rule.get("always_replace", True),
                            transforms=resolved_transforms,
                        ))
                    elif "token" in rule:
                        key = rule["token"].lower()
                        self._lookup[key] = _RuleEntry(
                            token=rule["token"],
                            replacement=rule.get("replacement", ""),
                            language=rule.get("language", ""),
                            always_replace=rule.get("always_replace", False),
                        )

        if not self._lookup and not self._pattern_rules:
            _LOGGER.warning("No replacement rules loaded")

        self._model = ContextReplacerClassifier(
            context_hidden_size=context_hidden_size,
        ).to(self.device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        from transformers import AutoTokenizer

        tokenizer_path = resolve_hf_model_path(self._context_model_name)
        local_files_only = any(
            os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}
            for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=local_files_only,
            use_fast=True,
        )
        self._bert = get_shared_hf_encoder(
            self._context_model_name,
            device=self.device,
            dtype=torch.float32,
            local_files_only=local_files_only,
        )

        self._loaded = True
        _LOGGER.info("Context replacer loaded (%d rules)", len(self._lookup))

    def load(self) -> None:
        self._load()

    def _predict_matches(
        self,
        texts: List[str],
        matches: List[_Match],
    ) -> List[bool]:
        if not matches:
            return []

        encoding = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=_MAX_BERT_TOKENS,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device).bool()
        offset_mapping = encoding["offset_mapping"].tolist()

        with torch.no_grad():
            hidden = self._bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            batch_indices = torch.tensor(
                [m.text_idx for m in matches],
                dtype=torch.long,
                device=self.device,
            )
            context_hidden = hidden.index_select(0, batch_indices)
            context_mask = attention_mask.index_select(0, batch_indices)
            word_token_mask = torch.zeros(
                (len(matches), _MAX_BERT_TOKENS),
                dtype=torch.bool,
                device=self.device,
            )

            for i, m in enumerate(matches):
                token_indices = [
                    tok_idx
                    for tok_idx, (start, end) in enumerate(offset_mapping[m.text_idx])
                    if start < m.end and end > m.start
                ]
                word_token_mask[i, token_indices or [0]] = True

            logits = self._model(context_hidden, context_mask, word_token_mask)
            probs = torch.sigmoid(logits)
            approved = (probs >= self.threshold).detach().cpu().tolist()

        return [bool(a) for a in approved]

    def _resolve_matches(self, texts: List[str], all_matches: List[_Match]) -> List[_Match]:
        """Split into always_replace vs need_classify, run classifier batch, return approved."""
        always = [m for m in all_matches if m.rule.always_replace]
        classify = [m for m in all_matches if not m.rule.always_replace]

        results = list(always)
        if classify:
            approved = self._predict_matches(texts, classify)
            results.extend(m for m, ok in zip(classify, approved) if ok)

        results.sort(key=lambda m: (m.text_idx, m.start))
        return results

    def _find_all_matches(self, text: str, text_idx: int, language: Optional[str] = None) -> List[_Match]:
        matches = _segment_and_find_matches(text, text_idx, self._lookup, language)
        matches.extend(_find_pattern_matches(text, text_idx, self._pattern_rules, language))
        matches.sort(key=lambda m: m.start)
        return matches

    def resolve_all(self, text: str, language: Optional[str] = None) -> List[Tuple[str, int, int, str]]:
        """Find and resolve all replaceable tokens in text."""
        self._load()
        matches = self._find_all_matches(text, 0, language)
        approved = self._resolve_matches([text], matches)
        return [(m.word, m.start, m.end, m.rule.replacement) for m in approved]

    def resolve_all_many(
        self, texts: List[str], language: Optional[str] = None,
    ) -> List[List[Tuple[str, int, int, str]]]:
        """Resolve replacements for multiple texts with one BERT pass."""
        self._load()
        all_matches: List[_Match] = []
        for text_idx, text in enumerate(texts):
            all_matches.extend(self._find_all_matches(text, text_idx, language))

        if not all_matches:
            return [[] for _ in texts]

        approved = self._resolve_matches(texts, all_matches)
        results: List[List[Tuple[str, int, int, str]]] = [[] for _ in texts]
        for m in approved:
            results[m.text_idx].append((m.word, m.start, m.end, m.rule.replacement))
        return results

    def apply_replacements(self, text: str, language: Optional[str] = None) -> str:
        """Apply approved replacements and return modified text."""
        replacements = self.resolve_all(text, language=language)
        if not replacements:
            return text
        result = text
        for _token, start, end, replacement in reversed(replacements):
            result = result[:start] + replacement + result[end:]
        return result

    def apply_replacements_many(self, texts: List[str], language: Optional[str] = None) -> List[str]:
        """Apply approved replacements to multiple texts."""
        all_replacements = self.resolve_all_many(texts, language=language)
        results = []
        for text, replacements in zip(texts, all_replacements):
            if not replacements:
                results.append(text)
                continue
            result = text
            for _token, start, end, replacement in reversed(replacements):
                result = result[:start] + replacement + result[end:]
            results.append(result)
        return results


_replacer: Optional[ContextReplacer] = None


def get_replacer(
    checkpoint_path: Optional[Path] = None,
    rules_path: Optional[Path] = None,
    device: Optional[str] = None,
    threshold: float = 0.5,
) -> ContextReplacer:
    """Get or create the global context replacer."""
    global _replacer

    if _replacer is None:
        if checkpoint_path is None:
            pkg_dir = Path(__file__).parent.parent.parent
            checkpoint_path = pkg_dir / "data" / "replacements" / "best.pt"
        if rules_path is None:
            pkg_dir = Path(__file__).parent.parent.parent
            rules_path = pkg_dir / "data" / "replacements" / "rules.jsonl"

        _replacer = ContextReplacer(checkpoint_path, rules_path, device, threshold)

    return _replacer
