#!/usr/bin/env python3
"""Prepare a Starling voice-clone JSONL dataset from randomspeech manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SEED_VC_ROOT = PROJECT_ROOT / "src" / "seed_vc_runtime"
if str(SEED_VC_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_VC_ROOT))

from hf_utils import load_custom_model_from_hf  # noqa: E402
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402
from src.piper.hf_cache import resolve_hf_model_path  # noqa: E402
from src.piper.preprocess import phonemize_texts_for_speaker  # noqa: E402
from src.piper.semantic import SemanticTokenizer, align_phone_features, build_bert_input  # noqa: E402


DEFAULT_LANG_TO_ID = {
    "en": 1,
    "ja": 2,
    "ko": 3,
    "zh": 4,
    "de": 5,
    "tr": 6,
    "ru": 7,
    "fr": 8,
    "es": 9,
    "pt": 10,
    "it": 11,
    "pl": 12,
    "nl": 13,
    "ar": 14,
    "sv": 15,
    "cs": 16,
    "id": 17,
    "th": 18,
    "vi": 19,
    "hi": 20,
    "uk": 21,
    "ro": 22,
}


def read_manifest(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_row(file, row: dict) -> None:
    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_campplus(device: torch.device) -> CAMPPlus:
    checkpoint_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model.eval().to(device)
    return model


def embedding_name(audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{audio_path.stem}_{digest}.npy"


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    if path.suffix == ".pt":
        audio = torch.load(path, map_location="cpu", weights_only=True).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        return audio
    audio_np, source_rate = sf.read(path, always_2d=True, dtype="float32")
    audio = torch.from_numpy(audio_np.T.copy())
    if source_rate != sample_rate:
        audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


@torch.inference_mode()
def extract_campplus(model: CAMPPlus, audio_path: Path, sample_rate: int, device: torch.device) -> np.ndarray:
    audio = load_audio(audio_path, sample_rate)
    audio_16k = torchaudio.functional.resample(audio, sample_rate, 16000)
    feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = model(feat.unsqueeze(0).to(device)).squeeze(0).detach().float().cpu().numpy()
    if embedding.shape != (192,):
        raise ValueError(f"Expected CAMP++ embedding shape (192,), got {embedding.shape} for {audio_path}")
    return embedding.astype(np.float32)


def cache_audio(path: Path, output_dir: Path, sample_rate: int) -> str:
    output_path = output_dir / f"{path.stem}_{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:16]}.pt"
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(load_audio(path, sample_rate).contiguous(), output_path)
    return str(output_path.resolve())


def feature_name(row: dict, index: int) -> str:
    source_id = str(row.get("source_utt_id") or row.get("utt_id") or f"utt_{index:08d}")
    digest = hashlib.sha1(
        f"{source_id}|{row.get('speaker')}|{row.get('text')}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{source_id}_{digest}.bert.pt"


def convert_frontend_row(row: dict) -> dict:
    ids = [int(value) for value in row["phoneme_ids"]]
    if not ids:
        raise ValueError("phonemized text has no supported Starling phoneme IDs")
    word_spans = row.get("word_spans")
    if not word_spans:
        raise ValueError("frontend did not return word_spans for semantic feature alignment")
    return {
        "text": str(row["text"]),
        "phoneme_ids": ids,
        "word_spans": word_spans,
    }


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_frontend_cache(
    rows: list[dict],
    piper_config: Path,
    batch_size: int,
    desc: str,
) -> dict[tuple[str, str], dict]:
    by_lang: dict[str, list[str]] = {}
    for row in rows:
        lang = str(row.get("lang") or "en").lower()
        text = str(row["text"])
        if text not in by_lang.setdefault(lang, []):
            by_lang[lang].append(text)

    cache: dict[tuple[str, str], dict] = {}
    total = sum(len(texts) for texts in by_lang.values())
    with tqdm(total=total, desc=desc) as progress:
        for lang in sorted(by_lang):
            for text_batch in batched(by_lang[lang], batch_size):
                try:
                    frontend_rows = phonemize_texts_for_speaker(
                        text_batch,
                        piper_config,
                        speaker_label=lang,
                        neural=True,
                    )
                except Exception as exc:  # noqa: BLE001
                    sample = text_batch[0] if text_batch else "<empty>"
                    raise RuntimeError(
                        f"Frontend preprocessing failed for language={lang}, sample_text={sample[:120]!r}"
                    ) from exc
                if len(frontend_rows) != len(text_batch):
                    raise RuntimeError(
                        f"Frontend returned {len(frontend_rows)} rows for {len(text_batch)} texts in language={lang}"
                    )
                for text, frontend in zip(text_batch, frontend_rows):
                    cache[(lang, text)] = convert_frontend_row(frontend)
                progress.update(len(text_batch))
    return cache


def split_rows(rows: list[dict], valid_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_lang: dict[str, list[dict]] = {}
    for row in rows:
        by_lang.setdefault(str(row.get("lang") or "en").lower(), []).append(row)
    train: list[dict] = []
    valid: list[dict] = []
    for lang_rows in by_lang.values():
        shuffled = list(lang_rows)
        rng.shuffle(shuffled)
        valid_count = max(1, int(round(len(shuffled) * valid_ratio))) if len(shuffled) > 1 else 1
        valid.extend(shuffled[:valid_count])
        train.extend(shuffled[valid_count:])
    if not train and valid:
        train.append(valid.pop())
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def make_output_row(
    row: dict,
    embedding_paths: dict[str, str],
    audio_paths: dict[str, str],
    prompt_audio_paths: dict[str, str],
    lang_to_id: dict[str, int],
    phoneme_cache: dict[tuple[str, str], dict],
) -> dict:
    lang = str(row.get("lang") or "en").lower()
    audio_path = str(Path(row.get("wav_path") or row.get("audio_path", "")).resolve())
    prompt_audio_path = str(Path(row["reference_voice"]).resolve())
    text = str(row["text"])
    cache_key = (lang, text)
    prepared = phoneme_cache.get(cache_key)
    if prepared is None:
        raise KeyError(f"Missing frontend cache entry for {cache_key}")
    return {
        "audio_path": audio_paths[audio_path],
        "text": prepared["text"],
        "source_text": text,
        "phoneme_ids": prepared["phoneme_ids"],
        "word_spans": prepared["word_spans"],
        "speaker_id": lang_to_id[lang],
        "speaker": lang,
        "prompt_audio_path": prompt_audio_paths[prompt_audio_path],
        "speaker_embedding_path": embedding_paths[prompt_audio_path],
        "source_utt_id": row.get("utt_id"),
        "source_text_id": row.get("source_text_id"),
        "preprocess_source": "piper_frontend",
    }


def build_output_rows(
    rows: list[dict],
    embedding_paths: dict[str, str],
    audio_paths: dict[str, str],
    prompt_audio_paths: dict[str, str],
    lang_to_id: dict[str, int],
    phoneme_cache: dict[tuple[str, str], dict],
    desc: str,
) -> tuple[list[dict], Counter]:
    counts: Counter = Counter()
    output_rows: list[dict] = []
    for row in tqdm(rows, desc=desc):
        lang = str(row.get("lang") or "en").lower()
        try:
            output_row = make_output_row(
                row,
                embedding_paths,
                audio_paths,
                prompt_audio_paths,
                lang_to_id,
                phoneme_cache,
            )
        except Exception as exc:  # noqa: BLE001
            source_id = row.get("utt_id") or row.get("source_text_id") or "<unknown>"
            raise RuntimeError(f"Frontend preprocessing failed for {source_id} ({lang})") from exc
        output_rows.append(output_row)
        counts[output_row["speaker"]] += 1
    return output_rows, counts


def write_output_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            write_jsonl_row(file, row)


@torch.inference_mode()
def add_bert_features(
    rows: list[dict],
    features_dir: Path,
    model_name: str,
    max_tokens: int | None,
    batch_size: int,
    device: torch.device,
    storage_dtype: str,
) -> None:
    if not rows:
        raise ValueError("No rows available for BERT feature precompute")
    features_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = SemanticTokenizer(model_name=model_name, max_length=max_tokens)
    model_path = resolve_hf_model_path(model_name, require_weights=True)
    model = AutoModel.from_pretrained(model_path).to(device).eval()
    hidden_size = int(getattr(model.config, "hidden_size"))

    for start in tqdm(range(0, len(rows), batch_size), desc="BERT sidecars"):
        batch = rows[start : start + batch_size]
        texts = [str(row["text"]) for row in batch]
        phoneme_lengths = [len(row["phoneme_ids"]) for row in batch]
        word_spans = [row.get("word_spans") for row in batch]
        bert_input = build_bert_input(
            texts,
            tokenizer,
            phoneme_lengths=phoneme_lengths,
            word_spans=word_spans,
        )
        if bert_input is None or "word2ph" not in bert_input:
            raise RuntimeError("Failed to build phoneme-aligned BERT input")
        hidden = model(
            input_ids=bert_input["input_ids"].to(device),
            attention_mask=bert_input["attention_mask"].to(device),
        ).last_hidden_state
        word2ph = bert_input["word2ph"].to(device)

        for item_idx, row in enumerate(batch):
            global_idx = start + item_idx
            features = align_phone_features(
                hidden[item_idx],
                word2ph[item_idx],
                phone_len=phoneme_lengths[item_idx],
            ).detach().float().cpu()
            if features.shape != (hidden_size, phoneme_lengths[item_idx]):
                raise ValueError(
                    f"BERT feature shape mismatch for {row.get('source_utt_id')}: "
                    f"got {tuple(features.shape)}, expected {(hidden_size, phoneme_lengths[item_idx])}"
                )
            if storage_dtype == "float16":
                features = features.half()
            output_path = features_dir / feature_name(row, global_idx)
            torch.save(features.contiguous(), output_path)
            row["bert_path"] = str(output_path.resolve())
            row["bert_dim"] = hidden_size
            row["bert_model_name"] = model_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--piper-config", type=Path, default=PROJECT_ROOT / "data" / "lzspeech-starling" / "config.json")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--semantic-model-name", default="distilbert/distilbert-base-multilingual-cased")
    parser.add_argument("--semantic-max-tokens", type=int, default=256)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-storage-dtype", choices=("float32", "float16"), default="float16")
    parser.add_argument("--frontend-batch-size", type=int, default=64)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    if args.purge and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.piper_config = args.piper_config.resolve()
    if not args.piper_config.exists():
        raise FileNotFoundError(f"Piper/Starling frontend config not found: {args.piper_config}")

    source_rows = read_manifest(args.input_manifest)
    rows = [row for row in source_rows if row.get("status") == "generated" and row.get("reference_voice")]
    if not rows:
        raise ValueError(f"No generated rows with reference_voice in {args.input_manifest}")
    unknown_langs = sorted({str(row.get("lang") or "en").lower() for row in rows} - set(DEFAULT_LANG_TO_ID))
    if unknown_langs:
        raise ValueError(f"No speaker/language IDs configured for: {unknown_langs}")

    target_audio_dir = args.output_dir / "audio_24k"
    reference_audio_dir = args.output_dir / "reference_audio_24k"
    audio_paths = {
        str(Path(row.get("wav_path") or row.get("audio_path", "")).resolve()): cache_audio(
            Path(row.get("wav_path") or row.get("audio_path", "")).resolve(),
            target_audio_dir,
            args.sample_rate,
        )
        for row in tqdm(rows, desc="Cache target audio")
    }
    reference_paths = sorted({str(Path(row["reference_voice"]).resolve()) for row in rows})
    prompt_audio_paths = {
        reference: cache_audio(Path(reference), reference_audio_dir, args.sample_rate)
        for reference in tqdm(reference_paths, desc="Cache reference audio")
    }

    device = torch.device(args.device)
    embedding_dir = args.output_dir / "campplus_embeddings"
    embedding_dir.mkdir(parents=True, exist_ok=True)
    model = load_campplus(device)
    embedding_paths: dict[str, str] = {}
    for reference in tqdm(reference_paths, desc="CAMP++ reference embeddings"):
        reference_path = Path(prompt_audio_paths[reference])
        output_path = embedding_dir / embedding_name(Path(reference))
        if not output_path.exists():
            np.save(output_path, extract_campplus(model, reference_path, args.sample_rate, device))
        embedding_paths[reference] = str(output_path.resolve())

    train_source, valid_source = split_rows(rows, args.valid_ratio, args.seed)
    frontend_cache = build_frontend_cache(
        train_source + valid_source,
        args.piper_config,
        args.frontend_batch_size,
        "Frontend rows",
    )
    train_rows, train_counts = build_output_rows(
        train_source,
        embedding_paths,
        audio_paths,
        prompt_audio_paths,
        DEFAULT_LANG_TO_ID,
        frontend_cache,
        "Assemble train rows",
    )
    valid_rows, valid_counts = build_output_rows(
        valid_source,
        embedding_paths,
        audio_paths,
        prompt_audio_paths,
        DEFAULT_LANG_TO_ID,
        frontend_cache,
        "Assemble val rows",
    )
    if not train_rows or not valid_rows:
        raise ValueError(f"Invalid split after preprocessing: train={len(train_rows)} valid={len(valid_rows)}")

    add_bert_features(
        train_rows + valid_rows,
        args.output_dir / "bert_features",
        args.semantic_model_name,
        args.semantic_max_tokens,
        args.bert_batch_size,
        device,
        args.bert_storage_dtype,
    )
    missing_bert = [row for row in train_rows + valid_rows if not row.get("bert_path")]
    if missing_bert:
        raise RuntimeError(f"BERT sidecar generation failed for {len(missing_bert)} rows")

    write_output_rows(args.output_dir / "train.jsonl", train_rows)
    write_output_rows(args.output_dir / "val.jsonl", valid_rows)

    counts = train_counts + valid_counts
    summary = {
        "source_manifest": str(args.input_manifest.resolve()),
        "piper_config": str(args.piper_config),
        "sample_rate": args.sample_rate,
        "total": len(train_rows) + len(valid_rows),
        "train": len(train_rows),
        "valid": len(valid_rows),
        "prompt_embedding_dim": 192,
        "semantic_model_name": args.semantic_model_name,
        "semantic_max_tokens": args.semantic_max_tokens,
        "bert_storage_dtype": args.bert_storage_dtype,
        "frontend_batch_size": args.frontend_batch_size,
        "langs": dict(sorted(counts.items())),
        "language_id_map": DEFAULT_LANG_TO_ID,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(train_rows)} train rows")
    print(f"wrote {len(valid_rows)} valid rows")
    print(f"output: {args.output_dir}")


if __name__ == "__main__":
    main()
