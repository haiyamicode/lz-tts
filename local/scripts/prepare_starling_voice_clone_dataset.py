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
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import local

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SEED_VC_ROOT = PROJECT_ROOT / "src" / "seed_vc_runtime"
if str(SEED_VC_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_VC_ROOT))

from src.piper.hf_cache import resolve_hf_model_path  # noqa: E402
from src.piper.preprocess import phonemize_texts_for_speaker  # noqa: E402
from src.piper.semantic import SemanticTokenizer, align_phone_features, build_bert_input  # noqa: E402
from src.starling.utils.audio import normalize_audio_rms  # noqa: E402
from precompute_starling_campplus_embeddings import (  # noqa: E402
    embedding_name as campplus_embedding_name,
    run_worker_pool as run_campplus_worker_pool,
)


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

_AUDIO_THREAD_STATE = local()


def read_manifest(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path).to_dict(orient="records")
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_row(file, row: dict) -> None:
    file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_audio(
    path: Path,
    sample_rate: int,
    rms_normalize_audio: bool = False,
    rms_target: float = 0.1,
    rms_peak_limit: float = 0.99,
    rms_eps: float = 1e-6,
) -> torch.Tensor:
    if path.suffix == ".pt":
        audio = torch.load(path, map_location="cpu", weights_only=True).float()
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
    else:
        audio_np, source_rate = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(audio_np.T.copy())
        if source_rate != sample_rate:
            rates = (source_rate, sample_rate)
            resamplers = getattr(_AUDIO_THREAD_STATE, "resamplers", None)
            if resamplers is None:
                resamplers = {}
                _AUDIO_THREAD_STATE.resamplers = resamplers
            resampler = resamplers.get(rates)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(source_rate, sample_rate)
                resamplers[rates] = resampler
            audio = resampler(audio)
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
    if rms_normalize_audio:
        audio = normalize_audio_rms(audio, rms_target, rms_peak_limit, rms_eps)
    return audio.contiguous()


def cache_audio(
    path: Path,
    output_dir: Path,
    sample_rate: int,
    rms_normalize_audio: bool = False,
    rms_target: float = 0.1,
    rms_peak_limit: float = 0.99,
    rms_eps: float = 1e-6,
) -> str:
    output_path = output_dir / f"{path.stem}_{hashlib.sha1(str(path.resolve()).encode('utf-8')).hexdigest()[:16]}.pt"
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
        torch.save(
            load_audio(
                path,
                sample_rate,
                rms_normalize_audio=rms_normalize_audio,
                rms_target=rms_target,
                rms_peak_limit=rms_peak_limit,
                rms_eps=rms_eps,
            ),
            temporary_path,
        )
        temporary_path.replace(output_path)
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
    use_same_utterance_as_reference: bool,
) -> dict:
    lang = str(row.get("lang") or "en").lower()
    audio_path = str(Path(row.get("wav_path") or row.get("audio_path", "")).resolve())
    prompt_audio_path = (
        audio_path
        if use_same_utterance_as_reference
        else str(Path(row["reference_voice"]).resolve())
    )
    text = str(row["text"])
    cache_key = (lang, text)
    prepared = phoneme_cache.get(cache_key)
    if prepared is None:
        raise KeyError(f"Missing frontend cache entry for {cache_key}")
    output_row = {
        "audio_path": audio_paths[audio_path],
        "text": prepared["text"],
        "source_text": text,
        "phoneme_ids": prepared["phoneme_ids"],
        "word_spans": prepared["word_spans"],
        "speaker_id": lang_to_id[lang],
        "speaker": lang,
        "prompt_audio_path": prompt_audio_paths[prompt_audio_path],
        "source_utt_id": row.get("utt_id"),
        "source_text_id": row.get("source_text_id"),
        "preprocess_source": "piper_frontend",
    }
    output_row["speaker_embedding_path"] = embedding_paths[prompt_audio_path]
    return output_row


def build_output_rows(
    rows: list[dict],
    embedding_paths: dict[str, str],
    audio_paths: dict[str, str],
    prompt_audio_paths: dict[str, str],
    lang_to_id: dict[str, int],
    phoneme_cache: dict[tuple[str, str], dict],
    desc: str,
    use_same_utterance_as_reference: bool,
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
                use_same_utterance_as_reference,
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
    parser.add_argument("--audio-workers", type=int, default=8)
    parser.add_argument("--campplus-workers", type=int, default=5)
    parser.set_defaults(rms_normalize_audio=True)
    parser.add_argument("--rms-normalize-audio", dest="rms_normalize_audio", action="store_true")
    parser.add_argument("--no-rms-normalize-audio", dest="rms_normalize_audio", action="store_false")
    parser.add_argument("--rms-target", type=float, default=0.1)
    parser.add_argument("--rms-peak-limit", type=float, default=0.99)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--use-same-utterance-as-reference", action="store_true")
    parser.add_argument("--same-utterance-reference-embedding-count", type=int, default=5)
    parser.add_argument("--same-utterance-reference-min-ratio", type=float, default=0.2)
    parser.add_argument("--same-utterance-reference-max-ratio", type=float, default=0.5)
    parser.add_argument("--same-utterance-reference-short-threshold-seconds", type=float, default=5.0)
    parser.add_argument("--same-utterance-reference-short-ratio", type=float, default=0.5)
    parser.add_argument("--purge", action="store_true")
    args = parser.parse_args()

    if args.same_utterance_reference_embedding_count < 1:
        raise ValueError("same-utterance-reference-embedding-count must be at least 1")
    if args.audio_workers < 1 or args.campplus_workers < 1:
        raise ValueError("audio-workers and campplus-workers must be at least 1")

    if args.purge and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.piper_config = args.piper_config.resolve()
    if not args.piper_config.exists():
        raise FileNotFoundError(f"Piper/Starling frontend config not found: {args.piper_config}")

    source_rows = read_manifest(args.input_manifest)
    rows = [
        row
        for row in source_rows
        if row.get("status") == "generated"
        and (args.use_same_utterance_as_reference or row.get("reference_voice"))
    ]
    if not rows:
        requirement = "generated rows" if args.use_same_utterance_as_reference else "generated rows with reference_voice"
        raise ValueError(f"No {requirement} in {args.input_manifest}")
    unknown_langs = sorted({str(row.get("lang") or "en").lower() for row in rows} - set(DEFAULT_LANG_TO_ID))
    if unknown_langs:
        raise ValueError(f"No speaker/language IDs configured for: {unknown_langs}")

    target_audio_dir = args.output_dir / "audio_24k"
    source_audio_paths = list(
        dict.fromkeys(
            str(Path(row.get("wav_path") or row.get("audio_path", "")).resolve())
            for row in rows
        )
    )
    cache_target_audio = partial(
        cache_audio,
        output_dir=target_audio_dir,
        sample_rate=args.sample_rate,
        rms_normalize_audio=args.rms_normalize_audio,
        rms_target=args.rms_target,
        rms_peak_limit=args.rms_peak_limit,
        rms_eps=args.rms_eps,
    )
    original_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(max_workers=args.audio_workers) as executor:
            cached_audio_paths = list(
                tqdm(
                    executor.map(cache_target_audio, map(Path, source_audio_paths)),
                    total=len(source_audio_paths),
                    desc="Cache target audio",
                )
            )
    finally:
        torch.set_num_threads(original_num_threads)
    audio_paths = dict(zip(source_audio_paths, cached_audio_paths))
    device = torch.device(args.device)
    embedding_paths: dict[str, str] = {}
    if args.use_same_utterance_as_reference:
        prompt_audio_paths = dict(audio_paths)
        embedding_dir = args.output_dir / "campplus_embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        cached_source_paths = sorted({Path(path) for path in audio_paths.values()}, key=str)
        pending_source_paths = [
            path
            for path in cached_source_paths
            if not (embedding_dir / campplus_embedding_name(path)).exists()
        ]
        if pending_source_paths:
            campplus_args = argparse.Namespace(
                **{
                    **vars(args),
                    "num_workers": args.campplus_workers,
                    "embeddings_per_utterance": args.same_utterance_reference_embedding_count,
                }
            )
            run_campplus_worker_pool(
                pending_source_paths,
                embedding_dir,
                [str(device)],
                campplus_args,
            )
        for source_path in source_audio_paths:
            cached_audio_path = Path(audio_paths[source_path])
            output_path = embedding_dir / campplus_embedding_name(cached_audio_path)
            if not output_path.is_file():
                raise FileNotFoundError(f"Missing CAMP++ embedding bank: {output_path}")
            embedding_paths[source_path] = str(output_path.resolve())
    else:
        reference_audio_dir = args.output_dir / "reference_audio_24k"
        reference_paths = sorted({str(Path(row["reference_voice"]).resolve()) for row in rows})
        prompt_audio_paths = {
            reference: audio_paths[reference]
            for reference in reference_paths
            if reference in audio_paths
        }
        external_references = [
            reference for reference in reference_paths if reference not in audio_paths
        ]
        cache_reference_audio = partial(
            cache_audio,
            output_dir=reference_audio_dir,
            sample_rate=args.sample_rate,
            rms_normalize_audio=args.rms_normalize_audio,
            rms_target=args.rms_target,
            rms_peak_limit=args.rms_peak_limit,
            rms_eps=args.rms_eps,
        )
        torch.set_num_threads(1)
        try:
            with ThreadPoolExecutor(max_workers=args.audio_workers) as executor:
                cached_reference_paths = list(
                    tqdm(
                        executor.map(cache_reference_audio, map(Path, external_references)),
                        total=len(external_references),
                        desc="Cache external reference audio",
                    )
                )
        finally:
            torch.set_num_threads(original_num_threads)
        prompt_audio_paths.update(zip(external_references, cached_reference_paths))
        embedding_dir = args.output_dir / "campplus_embeddings"
        embedding_dir.mkdir(parents=True, exist_ok=True)
        cached_reference_paths = sorted(
            {Path(prompt_audio_paths[reference]) for reference in reference_paths},
            key=str,
        )
        pending_reference_paths = [
            path
            for path in cached_reference_paths
            if not (embedding_dir / campplus_embedding_name(path)).exists()
        ]
        if pending_reference_paths:
            campplus_args = argparse.Namespace(
                **{
                    **vars(args),
                    "num_workers": args.campplus_workers,
                    "embeddings_per_utterance": 1,
                }
            )
            run_campplus_worker_pool(
                pending_reference_paths,
                embedding_dir,
                [str(device)],
                campplus_args,
            )
        for reference in reference_paths:
            cached_reference = Path(prompt_audio_paths[reference])
            output_path = embedding_dir / campplus_embedding_name(cached_reference)
            if not output_path.is_file():
                raise FileNotFoundError(f"Missing CAMP++ embedding: {output_path}")
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
        args.use_same_utterance_as_reference,
    )
    valid_rows, valid_counts = build_output_rows(
        valid_source,
        embedding_paths,
        audio_paths,
        prompt_audio_paths,
        DEFAULT_LANG_TO_ID,
        frontend_cache,
        "Assemble val rows",
        args.use_same_utterance_as_reference,
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
        "rms_normalize_audio": args.rms_normalize_audio,
        "rms_target": args.rms_target,
        "rms_peak_limit": args.rms_peak_limit,
        "rms_eps": args.rms_eps,
        "use_same_utterance_as_reference": args.use_same_utterance_as_reference,
        "same_utterance_reference_embedding_count": args.same_utterance_reference_embedding_count,
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
