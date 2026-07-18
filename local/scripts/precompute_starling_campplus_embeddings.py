#!/usr/bin/env python3
"""Precompute CAMP++ speaker embeddings for Starling JSONL manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing as mp
import random
import sys
import traceback
from pathlib import Path
from queue import Empty

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SEED_VC_ROOT = PROJECT_ROOT / "src" / "seed_vc_runtime"
if str(SEED_VC_ROOT) not in sys.path:
    sys.path.insert(0, str(SEED_VC_ROOT))

from hf_utils import load_custom_model_from_hf  # noqa: E402
from modules.campplus.DTDNN import CAMPPlus  # noqa: E402
from src.starling.data.text_mel_datamodule import sample_same_utterance_reference  # noqa: E402


def load_campplus(device: torch.device) -> CAMPPlus:
    checkpoint_path = load_custom_model_from_hf(
        "funasr/campplus",
        "campplus_cn_common.bin",
        config_filename=None,
    )
    model = CAMPPlus(feat_dim=80, embedding_size=192)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model.eval().to(device)
    return model


def load_audio(path: Path, sample_rate: int) -> torch.Tensor:
    if path.suffix == ".pt":
        audio = torch.load(path, map_location="cpu", weights_only=True).float()
    else:
        audio, source_rate = torchaudio.load(path)
        if source_rate != sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, sample_rate)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    return audio


def embedding_name(audio_path: Path) -> str:
    digest = hashlib.sha1(str(audio_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return f"{audio_path.stem}_{digest}.npy"


@torch.inference_mode()
def extract_embedding(model: CAMPPlus, audio: torch.Tensor, sample_rate: int, device: torch.device) -> np.ndarray:
    feat = extract_features(audio, sample_rate)
    embedding = model(feat.unsqueeze(0).to(device)).squeeze(0).detach().float().cpu().numpy()
    if embedding.shape != (192,):
        raise ValueError(f"Expected CAMP++ embedding shape (192,), got {embedding.shape}")
    return embedding.astype(np.float32)


def extract_features(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    audio_16k = torchaudio.functional.resample(audio, sample_rate, 16000)
    feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
    return feat - feat.mean(dim=0, keepdim=True)


def extract_embedding_bank(
    model: CAMPPlus,
    audio: torch.Tensor,
    sample_rate: int,
    device: torch.device,
    count: int,
    min_ratio: float,
    max_ratio: float,
    short_threshold_seconds: float,
    short_ratio: float,
    rng: random.Random,
) -> np.ndarray:
    reference_audios = []
    for _ in range(count):
        reference_audios.append(
            sample_same_utterance_reference(
                audio,
                sample_rate,
                min_ratio,
                max_ratio,
                short_threshold_seconds,
                short_ratio,
                randomize=True,
                rng=rng,
            )
        )
    if len({reference_audio.shape[-1] for reference_audio in reference_audios}) == 1:
        features = []
        for reference_audio in reference_audios:
            audio_16k = torchaudio.functional.resample(reference_audio, sample_rate, 16000)
            feat = kaldi.fbank(audio_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
            features.append(feat - feat.mean(dim=0, keepdim=True))
        embeddings = model(torch.stack(features).to(device)).detach().float().cpu().numpy()
    else:
        embeddings = [
            extract_embedding(model, reference_audio, sample_rate, device)
            for reference_audio in reference_audios
        ]
    return np.asarray(embeddings, dtype=np.float32)


@torch.inference_mode()
def extract_embedding_banks_batched(
    model: CAMPPlus,
    audio_paths: list[Path],
    sample_rate: int,
    device: torch.device,
    count: int,
    min_ratio: float,
    max_ratio: float,
    short_threshold_seconds: float,
    short_ratio: float,
    seed: int,
    batch_size: int,
) -> dict[Path, np.ndarray]:
    features_by_frames: dict[int, list[tuple[Path, int, torch.Tensor]]] = defaultdict(list)
    banks = {path: np.empty((count, 192), dtype=np.float32) for path in audio_paths}

    for audio_path in audio_paths:
        audio = load_audio(audio_path, sample_rate)
        seed_material = f"{seed}|{audio_path.resolve()}".encode("utf-8")
        rng = random.Random(int(hashlib.sha1(seed_material).hexdigest()[:16], 16))
        for embedding_index in range(count):
            reference_audio = sample_same_utterance_reference(
                audio,
                sample_rate,
                min_ratio,
                max_ratio,
                short_threshold_seconds,
                short_ratio,
                randomize=True,
                rng=rng,
            )
            features = extract_features(reference_audio, sample_rate)
            features_by_frames[features.shape[0]].append((audio_path, embedding_index, features))

    # CAMP++ performs unmasked pooling inside its CAM blocks. Only equal-length
    # features may share a batch without changing the resulting embeddings.
    for items in features_by_frames.values():
        for start in range(0, len(items), batch_size):
            batch = items[start : start + batch_size]
            batch_features = torch.stack([item[2] for item in batch]).to(device)
            embeddings = model(batch_features).detach().float().cpu().numpy()
            for (audio_path, embedding_index, _), embedding in zip(batch, embeddings):
                banks[audio_path][embedding_index] = embedding
    return banks


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary_path.replace(path)


def update_manifests(
    rows_by_manifest: list[tuple[Path, list[dict]]],
    path_to_embedding: dict[str, str],
) -> None:
    for manifest_path, rows in rows_by_manifest:
        changed = 0
        for row in rows:
            embedding_path = path_to_embedding.get(row["audio_path"])
            if embedding_path is None or not Path(embedding_path).is_file():
                continue
            if row.get("speaker_embedding_path") != embedding_path:
                row["speaker_embedding_path"] = embedding_path
                changed += 1
        if changed:
            write_jsonl(manifest_path, rows)
            print(f"updated {manifest_path}: {changed} rows")


def save_numpy_atomic(path: Path, array: np.ndarray) -> None:
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("wb") as file:
        np.save(file, array)
    temporary_path.replace(path)


def precompute_worker(
    worker_index: int,
    audio_paths: list[Path],
    output_dir: Path,
    device_names: list[str],
    args: argparse.Namespace,
    result_queue: mp.Queue,
) -> None:
    try:
        torch.set_num_threads(1)
        device = torch.device(device_names[worker_index % len(device_names)])
        if device.type == "cuda" and device.index is not None:
            torch.cuda.set_device(device)
        model = load_campplus(device)
        for audio_path in audio_paths:
            audio = load_audio(audio_path, args.sample_rate)
            if args.embeddings_per_utterance == 1:
                embedding = extract_embedding(model, audio, args.sample_rate, device)
            else:
                seed_material = f"{args.seed}|{audio_path.resolve()}".encode("utf-8")
                rng = random.Random(int(hashlib.sha1(seed_material).hexdigest()[:16], 16))
                embedding = extract_embedding_bank(
                    model,
                    audio,
                    args.sample_rate,
                    device,
                    args.embeddings_per_utterance,
                    args.same_utterance_reference_min_ratio,
                    args.same_utterance_reference_max_ratio,
                    args.same_utterance_reference_short_threshold_seconds,
                    args.same_utterance_reference_short_ratio,
                    rng,
                )
            save_numpy_atomic(output_dir / embedding_name(audio_path), embedding)
            result_queue.put(("completed", str(audio_path)))
    except BaseException:
        result_queue.put(("error", traceback.format_exc()))
    finally:
        result_queue.put(("worker_done", worker_index))


def run_worker_pool(
    audio_paths: list[Path],
    output_dir: Path,
    device_names: list[str],
    args: argparse.Namespace,
) -> int:
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=max(64, args.num_workers * 4))
    worker_paths = [audio_paths[index :: args.num_workers] for index in range(args.num_workers)]
    workers = [
        context.Process(
            target=precompute_worker,
            args=(index, paths, output_dir, device_names, args, result_queue),
            name=f"campplus-worker-{index}",
        )
        for index, paths in enumerate(worker_paths)
        if paths
    ]
    for worker in workers:
        worker.start()

    completed = 0
    finished_workers = 0
    error: str | None = None
    progress = tqdm(total=len(audio_paths), desc="CAMP++ embeddings")
    while finished_workers < len(workers) and error is None:
        try:
            status, payload = result_queue.get(timeout=1.0)
        except Empty:
            failed = [
                worker
                for worker in workers
                if worker.exitcode is not None and worker.exitcode != 0
            ]
            if failed:
                error = (
                    "CAMP++ workers exited without reporting an error: "
                    f"{[worker.exitcode for worker in failed]}"
                )
            continue
        if status == "completed":
            completed += 1
            progress.update(1)
        elif status == "worker_done":
            finished_workers += 1
        elif status == "error":
            error = payload
    progress.close()
    if error is not None:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
    for worker in workers:
        worker.join()
    if error is not None:
        raise RuntimeError(f"CAMP++ worker failed:\n{error}")
    failed = [worker for worker in workers if worker.exitcode != 0]
    if failed:
        raise RuntimeError(f"CAMP++ workers exited abnormally: {[worker.exitcode for worker in failed]}")
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new", type=int, default=None, help="Stop after writing this many new embeddings.")
    parser.add_argument("--embeddings-per-utterance", type=int, default=1)
    parser.add_argument("--utterance-chunk-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated worker devices, for example cuda:0,cuda:1,cuda:2.",
    )
    parser.add_argument("--same-utterance-reference-min-ratio", type=float, default=0.2)
    parser.add_argument("--same-utterance-reference-max-ratio", type=float, default=0.5)
    parser.add_argument("--same-utterance-reference-short-threshold-seconds", type=float, default=5.0)
    parser.add_argument("--same-utterance-reference-short-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--skip-manifest-update", action="store_true")
    parser.add_argument("--update-manifests-only", action="store_true")
    args = parser.parse_args()

    if args.embeddings_per_utterance < 1:
        raise ValueError("embeddings-per-utterance must be at least 1")
    if args.utterance_chunk_size < 1 or args.batch_size < 1 or args.num_workers < 1:
        raise ValueError("utterance-chunk-size and batch-size must be at least 1")
    if args.skip_manifest_update and args.update_manifests_only:
        raise ValueError("--skip-manifest-update and --update-manifests-only are mutually exclusive")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_manifest = [(path, read_jsonl(path)) for path in args.manifests]
    audio_paths = sorted({Path(row["audio_path"]) for _, rows in rows_by_manifest for row in rows})
    path_to_embedding = {
        str(audio_path): str((args.output_dir / embedding_name(audio_path)).resolve())
        for audio_path in audio_paths
    }

    if args.update_manifests_only:
        update_manifests(rows_by_manifest, path_to_embedding)
        return

    written = 0
    pending_paths = [
        path
        for path in audio_paths
        if args.force or not (args.output_dir / embedding_name(path)).exists()
    ]
    if args.max_new is not None:
        pending_paths = pending_paths[: args.max_new]

    if args.num_workers > 1:
        device_names = (
            [item.strip() for item in args.devices.split(",") if item.strip()]
            if args.devices
            else [str(device)]
        )
        if not device_names:
            raise ValueError("--devices must contain at least one device")
        written = run_worker_pool(pending_paths, args.output_dir, device_names, args)
    else:
        model = load_campplus(device)
        progress = tqdm(total=len(pending_paths), desc="CAMP++ embeddings")
        for start in range(0, len(pending_paths), args.utterance_chunk_size):
            chunk_paths = pending_paths[start : start + args.utterance_chunk_size]
            if args.embeddings_per_utterance == 1:
                banks = {
                    path: extract_embedding(
                        model,
                        load_audio(path, args.sample_rate),
                        args.sample_rate,
                        device,
                    )
                    for path in chunk_paths
                }
            else:
                banks = extract_embedding_banks_batched(
                    model,
                    chunk_paths,
                    args.sample_rate,
                    device,
                    args.embeddings_per_utterance,
                    args.same_utterance_reference_min_ratio,
                    args.same_utterance_reference_max_ratio,
                    args.same_utterance_reference_short_threshold_seconds,
                    args.same_utterance_reference_short_ratio,
                    args.seed,
                    args.batch_size,
                )
            for audio_path, embedding in banks.items():
                save_numpy_atomic(args.output_dir / embedding_name(audio_path), embedding)
                written += 1
            progress.update(len(chunk_paths))
        progress.close()

    print(f"wrote {written} new embeddings")
    if not args.skip_manifest_update:
        update_manifests(rows_by_manifest, path_to_embedding)


if __name__ == "__main__":
    main()
