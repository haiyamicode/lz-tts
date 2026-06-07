#!/usr/bin/env python3
"""Precompute phone-level BERT features for Piper datasets.

This converts an existing Piper `dataset.jsonl`/`dataset.parquet` into a compact
`dataset.parquet` with sidecar `.bert.pt` files. The model receives dense
phone-aligned BERT tensors instead of running tokenization/BERT/repeat alignment
inside the training step.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModel

from src.piper.semantic import SemanticTokenizer, build_bert_input

_LOGGER = logging.getLogger("precompute_piper_bert_features")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".parquet":
        return pq.read_table(path).to_pylist()

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _batched(items: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _feature_path(features_dir: Path, row: dict[str, Any], index: int) -> Path:
    audio_path = row.get("audio_path") or row.get("audio_norm_path") or f"utt_{index:08d}"
    stem = Path(str(audio_path)).stem or f"utt_{index:08d}"
    speaker = row.get("speaker") or row.get("speaker_id")
    prefix = f"{speaker}_" if speaker not in (None, "") else ""
    return features_dir / f"{prefix}{index:08d}_{stem}.bert.pt"


def _align_phone_features(
    hidden: torch.Tensor,
    word2ph: torch.Tensor,
    phone_len: int,
) -> torch.Tensor:
    counts = torch.clamp(word2ph.to(torch.long).cpu(), min=0)
    diff = int(phone_len) - int(counts.sum().item())
    if diff:
        active = torch.nonzero(counts > 0, as_tuple=False).flatten()
        adjust_idx = int(active[-1].item()) if active.numel() else max(0, counts.numel() - 1)
        counts = counts.clone()
        counts[adjust_idx] = torch.clamp(counts[adjust_idx] + diff, min=0)

    repeated = torch.repeat_interleave(hidden.cpu(), counts, dim=0)
    if repeated.size(0) == 0:
        return hidden.new_zeros((hidden.size(-1), int(phone_len))).cpu()

    if repeated.size(0) < phone_len:
        pad = repeated.new_zeros((phone_len - repeated.size(0), repeated.size(1)))
        repeated = torch.cat([repeated, pad], dim=0)
    elif repeated.size(0) > phone_len:
        repeated = repeated[:phone_len]

    return repeated.transpose(0, 1).contiguous()


def precompute(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.input_dataset:
        input_path = Path(args.input_dataset).resolve()
    else:
        jsonl_path = dataset_dir / "dataset.jsonl"
        parquet_path = dataset_dir / "dataset.parquet"
        input_path = jsonl_path if jsonl_path.exists() else parquet_path
    output_path = Path(args.output_dataset).resolve() if args.output_dataset else dataset_dir / "dataset.parquet"
    features_dir = Path(args.features_dir).resolve() if args.features_dir else dataset_dir / "bert_features"
    features_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    if args.max_phoneme_ids is not None:
        max_phoneme_ids = int(args.max_phoneme_ids)
        before = len(rows)
        rows = [
            row
            for row in rows
            if len(row.get("phoneme_ids") or []) <= max_phoneme_ids
        ]
        _LOGGER.info(
            "Filtered rows by max_phoneme_ids=%s: kept=%s dropped=%s",
            max_phoneme_ids,
            len(rows),
            before - len(rows),
        )
        if not rows:
            raise ValueError(f"No rows remain after max_phoneme_ids={max_phoneme_ids} filter")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    _LOGGER.info("Loading semantic model: %s on %s", args.model_name, device)
    tokenizer = SemanticTokenizer(model_name=args.model_name, max_length=args.max_tokens)
    model = AutoModel.from_pretrained(
        args.model_name,
        local_files_only=bool(args.local_files_only),
    ).to(device).eval()
    hidden_size = int(getattr(model.config, "hidden_size"))

    out_rows: list[dict[str, Any]] = []
    total = len(rows)
    with torch.inference_mode():
        for batch_start, batch in enumerate(_batched(rows, int(args.batch_size))):
            base_index = batch_start * int(args.batch_size)
            texts = [str(row.get("text") or "") for row in batch]
            phoneme_lengths = [len(row["phoneme_ids"]) for row in batch]
            word_spans = [row.get("word_spans") for row in batch]
            bert_input = build_bert_input(
                texts,
                tokenizer,
                phoneme_lengths=phoneme_lengths,
                word_spans=word_spans,
            )
            if bert_input is None:
                raise ValueError("Failed to build BERT input")

            input_ids = bert_input["input_ids"].to(device)
            attention_mask = bert_input["attention_mask"].to(device)
            word2ph = bert_input["word2ph"]
            hidden = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

            for item_idx, row in enumerate(batch):
                global_idx = base_index + item_idx
                phone_len = phoneme_lengths[item_idx]
                features_path = _feature_path(features_dir, row, global_idx)
                if args.overwrite or not features_path.exists():
                    features = _align_phone_features(
                        hidden[item_idx],
                        word2ph[item_idx],
                        phone_len=phone_len,
                    )
                    if args.storage_dtype == "float16":
                        features = features.half()
                    torch.save(features, features_path)

                out_row = dict(row)
                out_row["bert_path"] = str(features_path)
                out_row["bert_dim"] = hidden_size
                out_row["bert_model_name"] = args.model_name
                out_rows.append(out_row)

            done = min(total, base_index + len(batch))
            if done == total or done % max(1, int(args.log_every)) == 0:
                _LOGGER.info("Precomputed BERT features: %s/%s", done, total)

    table = pa.Table.from_pylist(out_rows)
    pq.write_table(table, output_path, compression=args.compression)
    _LOGGER.info("Wrote %s rows to %s", len(out_rows), output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, help="Piper dataset directory")
    parser.add_argument("--input-dataset", help="Input dataset.jsonl or dataset.parquet")
    parser.add_argument("--output-dataset", help="Output dataset.parquet")
    parser.add_argument("--features-dir", help="Directory for sidecar .bert.pt feature tensors")
    parser.add_argument("--model-name", default="distilbert-base-multilingual-cased")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-phoneme-ids", type=int)
    parser.add_argument("--storage-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    precompute(parser.parse_args())


if __name__ == "__main__":
    main()
