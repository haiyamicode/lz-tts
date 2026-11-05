"""Train the number mode classifier (piper.number_modes).

The context replacer head with one output per mode (cardinal, year, ordinal,
digits) on the frozen multilingual DistilBERT. Features come from the same
``span_windows`` the resolver uses at inference, computed once; only the head
trains, on each hand-written line plus copies with resampled numbers, and its
weights are averaged over the second half of training. 5-fold cross-validation
(stratified by language, each line held out once without its copies) scores that
procedure; the head then retrains on every line.

Usage:
    uv run python scripts/train_number_modes.py \
        --data scripts/number_modes_samples.tsv \
        --output data/number-modes/best.pt

Sample format (TSV, ``#`` comments): ``<language>\\t<text>``; every digit run
inside ``⟦...:X⟧`` gets label X (C cardinal, Y year, O ordinal, D digits).
"""

import argparse
import logging
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from piper.context_replacer import (  # noqa: E402
    _DEFAULT_CONTEXT_HIDDEN,
    _DEFAULT_CONTEXT_MODEL,
    ContextReplacerClassifier,
)
from piper.hf_cache import get_shared_hf_encoder, resolve_hf_tokenizer_path  # noqa: E402
from piper.number_modes import ARCHITECTURE, span_windows  # noqa: E402
from text_norm.spoken import NUMBER_MODES, verbalize_spoken_text  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOGGER = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"⟦([^⟧]*?):([CYOD])⟧")
_CODES = {"C": "cardinal", "Y": "year", "O": "ordinal", "D": "digits"}


def load_lines(path: Path) -> list[tuple[str, str]]:
    return [
        tuple(line.split("\t", 1))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def parse(language: str, raw: str) -> dict:
    text, spans, labels = [], [], []
    cursor = length = 0
    for match in _MARKER_RE.finditer(raw):
        text.append(raw[cursor : match.start()])
        length += match.start() - cursor
        inner = match.group(1)
        for digits in re.finditer(r"\d+", inner):
            spans.append((length + digits.start(), length + digits.end()))
            labels.append(NUMBER_MODES.index(_CODES[match.group(2)]))
        text.append(inner)
        length += len(inner)
        cursor = match.end()
    text.append(raw[cursor:])
    return {"language": language, "text": "".join(text), "spans": spans, "labels": labels}


def resample(raw: str, rng: random.Random) -> str:
    """The same line with fresh numbers of the same shape, so the frame decides the mode.

    Codes get random digits, 4-digit years a random year (ascending inside one
    marker), counts of 3+ digits a random count (4-digit ones mostly year-like);
    small numbers and zero-led runs stay.
    """

    def marker(match: re.Match) -> str:
        code, inner = match.group(2), match.group(1)
        years = iter(sorted(rng.randint(1800, 2039) for _ in re.finditer(r"\d{4}", inner)))

        def number(digits: re.Match) -> str:
            run = digits.group(0)
            if run.startswith("0") and code != "D":
                return run
            if code == "D":
                first = "0" if run.startswith("0") else str(rng.randint(1, 9))
                return first + "".join(str(rng.randint(0, 9)) for _ in run[1:])
            if code == "Y" and len(run) == 4:
                return str(next(years))
            if code == "C" and len(run) == 4 and rng.random() < 0.7:
                return str(rng.randint(1000, 2099))
            if code == "C" and len(run) >= 3:
                return str(rng.randint(10 ** (len(run) - 1), 10 ** len(run) - 1))
            return run

        return "⟦" + re.sub(r"\d+", number, inner) + f":{code}⟧"

    return _MARKER_RE.sub(marker, raw)


@torch.no_grad()
def encode(samples: list[dict], tokenizer, bert, device) -> None:
    """Attach (hidden, attention mask, word mask, label) per span to each sample."""
    for sample in samples:
        windows = span_windows(tokenizer, sample["text"], sample["spans"])
        hidden = bert(
            input_ids=windows.input_ids.to(device),
            attention_mask=windows.attention_mask.to(device),
        ).last_hidden_state.cpu()
        sample["features"] = [
            (hidden[row], windows.attention_mask[row], windows.word_token_mask[i], label)
            for i, (row, label) in enumerate(zip(windows.window_index.tolist(), sample["labels"]))
        ]


def collate(features: list[tuple]) -> tuple[torch.Tensor, ...]:
    length = max(hidden.size(0) for hidden, *_ in features)
    hidden = torch.zeros(len(features), length, features[0][0].size(-1))
    attention = torch.zeros(len(features), length, dtype=torch.bool)
    words = torch.zeros(len(features), length, dtype=torch.bool)
    for i, (h, a, w, _) in enumerate(features):
        hidden[i, : h.size(0)] = h
        attention[i, : a.size(0)] = a
        words[i, : w.size(0)] = w
    labels = torch.tensor([label for *_, label in features])
    return hidden, attention, words, labels


def run_epoch(model, features, device, optimizer=None, batch_size=32):
    criterion = nn.CrossEntropyLoss()
    model.train(optimizer is not None)
    if optimizer is not None:
        features = random.sample(features, len(features))
    total_loss, predictions = 0.0, []
    for start in range(0, len(features), batch_size):
        hidden, attention, words, labels = (t.to(device) for t in collate(features[start : start + batch_size]))
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(hidden, attention, words)
            loss = criterion(logits, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=-1).tolist())
    return total_loss / len(features), predictions


def fit(features, epochs, lr, device, held_out=None):
    """Train a fresh head and average its weights over the second half of training.

    With ``held_out``, also return (loss, predictions) per epoch, from the
    averaged head once averaging has started.
    """
    model = ContextReplacerClassifier(_DEFAULT_CONTEXT_HIDDEN, num_labels=len(NUMBER_MODES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    averaged = torch.optim.swa_utils.AveragedModel(model)
    curve = []
    for epoch in range(epochs):
        run_epoch(model, features, device, optimizer)
        if epoch >= epochs // 2:
            averaged.update_parameters(model)
        if held_out is not None:
            curve.append(run_epoch(averaged.module if epoch >= epochs // 2 else model, held_out, device))
    return averaged.module, curve


def spoken(sample: dict, labels: list[int]) -> str:
    """The sample read with these modes (spans the engine finds on its own read as cardinals)."""
    modes = dict(zip(sample["spans"], (NUMBER_MODES[label] for label in labels)))
    return verbalize_spoken_text(
        sample["text"], sample["language"], lambda _, spans: [modes.get(span, "cardinal") for span in spans]
    )


def report(held_out_samples, predictions) -> None:
    rows = [(s, i) for s in held_out_samples for i in range(len(s["labels"]))]
    per_language = defaultdict(lambda: [0, 0])
    confusion = [[0] * len(NUMBER_MODES) for _ in NUMBER_MODES]
    errors, audible = [], 0
    for (sample, i), predicted in zip(rows, predictions):
        label = sample["labels"][i]
        per_language[sample["language"]][0] += predicted == label
        per_language[sample["language"]][1] += 1
        confusion[label][predicted] += 1
        if predicted != label:
            # Many confusions say the same words: years in French, a day before its month in Chinese.
            swapped = list(sample["labels"])
            swapped[i] = predicted
            same = spoken(sample, swapped) == spoken(sample, sample["labels"])
            audible += not same
            start, end = sample["spans"][i]
            errors.append(
                f"  [{sample['language']}] {sample['text'][:start]}⟦{sample['text'][start:end]}⟧"
                f"{sample['text'][end:]}  gold={NUMBER_MODES[label]} predicted={NUMBER_MODES[predicted]}"
                + ("  (same words)" if same else "")
            )
    correct = sum(c for c, _ in per_language.values())
    total = sum(t for _, t in per_language.values())
    print(f"\nCross-validated accuracy: {correct}/{total} = {correct / total:.3f}; errors that change the words: {audible}")
    for language, (c, t) in sorted(per_language.items()):
        print(f"  {language:4s} {c:3d}/{t:3d} = {c / t:.3f}")
    print("\nConfusion (rows gold, columns predicted):")
    print(" " * 10 + "".join(f"{mode:>10s}" for mode in NUMBER_MODES))
    for mode, row in zip(NUMBER_MODES, confusion):
        print(f"{mode:>10s}" + "".join(f"{n:10d}" for n in row))
    print("\nHeld-out errors:" if errors else "\nNo held-out errors.")
    print("\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("scripts/number_modes_samples.tsv"))
    parser.add_argument("--output", type=Path, default=Path("data/number-modes/best.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--variants", type=int, default=4, help="Resampled copies of each training line")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    lines = load_lines(args.data)
    rng = random.Random(args.seed)
    originals = [parse(language, raw) for language, raw in lines]
    variants = [[parse(language, resample(raw, rng)) for _ in range(args.variants)] for language, raw in lines]
    _LOGGER.info(
        "%d lines (%d spans), %d resampled variants each",
        len(lines), sum(len(s["spans"]) for s in originals), args.variants,
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve_hf_tokenizer_path(_DEFAULT_CONTEXT_MODEL), use_fast=True)
    bert = get_shared_hf_encoder(_DEFAULT_CONTEXT_MODEL, device=device, dtype=torch.float32, local_files_only=True)
    encode(originals + [v for line_variants in variants for v in line_variants], tokenizer, bert, device)

    def features(indices, with_variants=True):
        return [
            f
            for i in indices
            for sample in ((originals[i], *variants[i]) if with_variants else (originals[i],))
            for f in sample["features"]
        ]

    # Folds stratified by language; every line is held out once, without its variants.
    fold_of = {}
    by_language = defaultdict(list)
    for i, (language, _) in enumerate(lines):
        by_language[language].append(i)
    for indices in by_language.values():
        random.shuffle(indices)
        for n, i in enumerate(indices):
            fold_of[i] = n % args.folds
    folds = []
    for fold in range(args.folds):
        held = [i for i in range(len(lines)) if fold_of[i] == fold]
        train = [i for i in range(len(lines)) if fold_of[i] != fold]
        _, curve = fit(features(train), args.epochs, args.lr, device, features(held, with_variants=False))
        folds.append((held, curve))
        _LOGGER.info("fold %d: %d held-out lines", fold + 1, len(held))

    labels = [f[3] for held, _ in folds for f in features(held, with_variants=False)]
    for epoch in range(args.epochs):
        loss = sum(curve[epoch][0] * len(features(held, with_variants=False)) for held, curve in folds) / len(labels)
        predictions = [p for _, curve in folds for p in curve[epoch][1]]
        accuracy = sum(p == label for p, label in zip(predictions, labels)) / len(labels)
        _LOGGER.info("epoch %d  cross-validated loss %.4f  accuracy %.3f", epoch + 1, loss, accuracy)
    report([originals[i] for held, _ in folds for i in held], predictions)

    torch.manual_seed(args.seed)
    model, _ = fit(features(range(len(lines))), args.epochs, args.lr, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "labels": list(NUMBER_MODES),
            "architecture": ARCHITECTURE,
            "context_model_name": _DEFAULT_CONTEXT_MODEL,
            "context_hidden_size": _DEFAULT_CONTEXT_HIDDEN,
            "epochs": args.epochs,
            "cross_validated_loss": loss,
            "cross_validated_accuracy": accuracy,
        },
        args.output,
    )
    _LOGGER.info("Saved %s (all %d lines, %d epochs)", args.output, len(lines), args.epochs)


if __name__ == "__main__":
    main()
