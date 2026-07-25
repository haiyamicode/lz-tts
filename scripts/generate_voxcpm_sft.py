#!/usr/bin/env python3
"""Generate a resumable, quality-gated VoxCPM2 self-SFT dataset."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from scipy.signal import resample_poly
from tqdm.auto import tqdm


def _absolute(path: str) -> Path:
    return Path(to_absolute_path(os.path.expanduser(path))).resolve()


def _generation_limit(model_path: Path, max_audio_seconds: float) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    audio_config = config["audio_vae_config"]
    encoder_stride = math.prod(audio_config["encoder_rates"])
    step_seconds = (
        config["patch_size"] * encoder_stride / audio_config["sample_rate"]
    )
    return max(1, math.floor(max_audio_seconds / step_seconds))


async def _generate_one(
    server: Any,
    task: dict[str, Any],
    generation: dict[str, Any],
    *,
    prompt_id: str | None = None,
) -> dict[str, Any]:
    chunks = []
    started = time.perf_counter()
    try:
        async for chunk in server.generate(
            target_text=task["text"],
            max_generate_length=int(generation["max_generate_length"]),
            temperature=float(generation["temperature"]),
            cfg_value=float(generation["cfg_value"]),
            seed=int(task["seed"]),
            prompt_id=prompt_id,
        ):
            chunks.append(chunk)
        audio = (
            np.concatenate(chunks).astype(np.float32, copy=False)
            if chunks
            else np.zeros(0, dtype=np.float32)
        )
        max_samples = math.floor(
            float(generation["max_audio_seconds"])
            * int(generation["sample_rate"])
        )
        return {
            "task": task,
            "audio": np.ascontiguousarray(audio[:max_samples]),
            "generation_wall_seconds": time.perf_counter() - started,
            "chunks": len(chunks),
            "error": None,
        }
    except BaseException as exc:
        return {
            "task": task,
            "audio": np.zeros(0, dtype=np.float32),
            "generation_wall_seconds": time.perf_counter() - started,
            "chunks": 0,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


async def _generate_voice_group(
    server: Any,
    group: dict[str, Any],
    generation: dict[str, Any],
) -> list[dict[str, Any]]:
    anchor_task = group["anchor_task"]
    tasks = group["tasks"]
    anchor_result = None
    prompt_audio_path = group.get("prompt_audio_path")
    anchor_task_id = int(anchor_task["task_id"])

    if any(int(task["task_id"]) == anchor_task_id for task in tasks):
        anchor_result = await _generate_one(server, anchor_task, generation)
        results = [anchor_result]
    else:
        results = []

    if prompt_audio_path is not None:
        prompt_audio = Path(prompt_audio_path).read_bytes()
    elif anchor_result is not None:
        if anchor_result["error"] is not None or not anchor_result["audio"].size:
            anchor_error = anchor_result["error"] or "empty anchor audio"
            for task in tasks:
                if int(task["task_id"]) == anchor_task_id:
                    continue
                results.append(
                    {
                        "task": task,
                        "audio": np.zeros(0, dtype=np.float32),
                        "generation_wall_seconds": 0.0,
                        "chunks": 0,
                        "error": f"VoiceAnchorError: {anchor_error}",
                    }
                )
            return sorted(results, key=lambda result: result["task"]["task_id"])
        prompt_audio = _wav_bytes(
            anchor_result["audio"],
            int(generation["sample_rate"]),
        )
    else:
        regenerated_anchor = await _generate_one(server, anchor_task, generation)
        if regenerated_anchor["error"] is not None or not regenerated_anchor[
            "audio"
        ].size:
            anchor_error = regenerated_anchor["error"] or "empty anchor audio"
            for task in tasks:
                if int(task["task_id"]) == anchor_task_id:
                    continue
                results.append(
                    {
                        "task": task,
                        "audio": np.zeros(0, dtype=np.float32),
                        "generation_wall_seconds": 0.0,
                        "chunks": 0,
                        "error": f"VoiceAnchorError: {anchor_error}",
                    }
                )
            return sorted(results, key=lambda result: result["task"]["task_id"])
        prompt_audio = _wav_bytes(
            regenerated_anchor["audio"],
            int(generation["sample_rate"]),
        )

    follower_tasks = [
        task for task in tasks if int(task["task_id"]) != anchor_task_id
    ]
    if not follower_tasks:
        return results

    prompt_id = await server.add_prompt(
        prompt_audio,
        "wav",
        str(anchor_task["text"]),
    )
    try:
        results.extend(
            await asyncio.gather(
                *[
                    _generate_one(
                        server,
                        task,
                        generation,
                        prompt_id=prompt_id,
                    )
                    for task in follower_tasks
                ]
            )
        )
    finally:
        with contextlib.suppress(Exception):
            await server.remove_prompt(prompt_id)
    return sorted(results, key=lambda result: result["task"]["task_id"])


async def _worker_async(
    worker_id: int,
    gpu: int,
    worker_config: dict[str, Any],
    input_queue: Any,
    output_queue: Any,
) -> None:
    sys.path.insert(0, worker_config["nanovllm_path"])
    from nanovllm_voxcpm import VoxCPM

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices != str(gpu):
        raise RuntimeError(
            f"Worker {worker_id} expected CUDA_VISIBLE_DEVICES={gpu}, "
            f"got {visible_devices!r}"
        )
    logical_gpu = 0
    torch.cuda.set_device(logical_gpu)
    server = None
    try:
        server = VoxCPM.from_pretrained(
            model=worker_config["model_path"],
            devices=[logical_gpu],
            inference_timesteps=int(worker_config["inference_timesteps"]),
            max_num_batched_tokens=int(worker_config["max_num_batched_tokens"]),
            max_num_seqs=int(worker_config["batch_size"]),
            max_model_len=int(worker_config["max_model_len"]),
            gpu_memory_utilization=float(worker_config["gpu_memory_utilization"]),
            enforce_eager=bool(worker_config["enforce_eager"]),
        )
        await server.wait_for_ready()
        model_info = await server.get_model_info()
        sample_rate = int(model_info["sample_rate"])
        output_queue.put(
            {
                "kind": "ready",
                "worker_id": worker_id,
                "gpu": gpu,
                "sample_rate": sample_rate,
            }
        )

        generation = {
            **worker_config,
            "sample_rate": sample_rate,
        }
        while True:
            message = await asyncio.to_thread(input_queue.get)
            if message is None:
                break
            batch_started = time.perf_counter()
            group_results = await asyncio.gather(
                *[
                    _generate_voice_group(server, group, generation)
                    for group in message["groups"]
                ]
            )
            results = [
                result
                for group_result in group_results
                for result in group_result
            ]
            output_queue.put(
                {
                    "kind": "batch",
                    "worker_id": worker_id,
                    "gpu": gpu,
                    "batch_id": message["batch_id"],
                    "batch_wall_seconds": time.perf_counter() - batch_started,
                    "results": results,
                }
            )
    except BaseException as exc:
        output_queue.put(
            {
                "kind": "worker_error",
                "worker_id": worker_id,
                "gpu": gpu,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        if server is not None:
            await server.stop()


def _worker_entry(
    worker_id: int,
    gpu: int,
    worker_config: dict[str, Any],
    input_queue: Any,
    output_queue: Any,
) -> None:
    asyncio.run(
        _worker_async(
            worker_id,
            gpu,
            worker_config,
            input_queue,
            output_queue,
        )
    )


class ScoreQ:
    def __init__(self, model_path: Path, threads: int):
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def score(self, audio: np.ndarray, sample_rate: int) -> float:
        if sample_rate != 16_000:
            ratio = Fraction(16_000, sample_rate)
            audio = resample_poly(
                audio,
                ratio.numerator,
                ratio.denominator,
            ).astype(np.float32)
        padded_length = math.ceil(len(audio) / 320) * 320
        if padded_length != len(audio):
            audio = np.pad(audio, (0, padded_length - len(audio)))
        return float(
            self.session.run(
                None,
                {self.input_name: audio.astype(np.float32, copy=False)[None, :]},
            )[0].item()
        )


def _build_validator(cfg: DictConfig) -> Any:
    from src.qwen_dp_budget import DpBudgetConfig, QwenDpBudget

    validator = QwenDpBudget(
        DpBudgetConfig(
            checkpoint=_absolute(cfg.validation.mas.checkpoint),
            device=str(cfg.validation.device),
            language="multilingual",
            noise_scale=float(cfg.validation.mas.noise_scale),
            length_scale=float(cfg.validation.mas.length_scale),
            token_rate=12.0,
            samples=int(cfg.validation.mas.duration_samples),
            upper_quantile=0.90,
            min_margin=1.0,
            max_margin=1.35,
            min_extra_tokens=0,
            max_extra_tokens=72,
            use_bert=True,
            enable_alignment_validation=True,
        )
    )
    validator.load()
    return validator


def _build_asr(cfg: DictConfig) -> Any:
    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, str(cfg.validation.asr.dtype))
    return Qwen3ASRModel.from_pretrained(
        str(cfg.validation.asr.model),
        dtype=dtype,
        device_map=str(cfg.validation.device),
        attn_implementation=str(cfg.validation.asr.attn_implementation),
        max_inference_batch_size=int(cfg.validation.asr.batch_size),
        max_new_tokens=int(cfg.validation.asr.max_new_tokens),
    )


def _load_corpora(cfg: DictConfig) -> dict[str, list[str]]:
    root = _absolute(cfg.source.root)
    corpora = {}
    for language in cfg.source.languages:
        path = root / f"{language}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing corpus for {language}: {path}")
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        corpora[str(language)] = [line for line in lines if line]
        if not corpora[str(language)]:
            raise ValueError(f"Corpus has no non-empty lines: {path}")
    return corpora


def _build_prompt_orders(
    corpora: dict[str, list[str]],
    seed: int,
) -> dict[str, list[int]]:
    orders = {}
    for language, lines in corpora.items():
        order = list(range(len(lines)))
        language_seed = int.from_bytes(language.encode("ascii"), "little")
        random.Random(seed + language_seed).shuffle(order)
        order = [
            source_index
            for source_index in order
            if any(character.isalnum() for character in lines[source_index])
        ]
        if not order:
            raise ValueError(
                f"Corpus has no utterances containing letters or numbers: {language}"
            )
        orders[language] = order
    return orders


def _task_for_id(
    task_id: int,
    corpora: dict[str, list[str]],
    orders: dict[str, list[int]],
    languages: list[str],
    seed: int,
    samples_per_voice: int,
    voice_anchor_selection: str,
) -> dict[str, Any]:
    voice_group_id = task_id // samples_per_voice
    voice_sample_index = task_id % samples_per_voice
    language = languages[voice_group_id % len(languages)]
    language_group_position = voice_group_id // len(languages)
    order = orders[language]
    first_language_position = language_group_position * samples_per_voice
    candidates = []
    for candidate_index in range(samples_per_voice):
        language_position = first_language_position + candidate_index
        source_index = order[language_position % len(order)]
        candidates.append(
            (
                language_position,
                source_index,
                corpora[language][source_index],
                candidate_index,
            )
        )
    if voice_anchor_selection == "longest_text":
        candidates.sort(key=lambda item: (-len(item[2]), item[3]))
    elif voice_anchor_selection != "first":
        raise ValueError(
            f"Unsupported generation.voice_anchor_selection: "
            f"{voice_anchor_selection!r}"
        )
    language_position, source_index, text, _ = candidates[voice_sample_index]
    corpus_round = language_position // len(order)
    generation_seed = (seed + task_id * 1_000_003) % (2**31 - 1)
    voice_seed = (seed + voice_group_id * 1_000_033) % (2**31 - 1)
    return {
        "task_id": task_id,
        "voice_group_id": voice_group_id,
        "voice_sample_index": voice_sample_index,
        "voice_anchor_task_id": voice_group_id * samples_per_voice,
        "voice_seed": voice_seed,
        "language": language,
        "text": text,
        "source_line": source_index + 1,
        "corpus_round": corpus_round,
        "seed": generation_seed,
    }


def _prepare_tasks(tasks: list[dict[str, Any]], cfg: DictConfig) -> None:
    from src.text_norm import prepare_tts_texts

    source_texts = [task["text"] for task in tasks]
    locales = [
        str(cfg.validation.languages[task["language"]].locale)
        for task in tasks
    ]
    prepared_texts = prepare_tts_texts(
        source_texts,
        locales,
        normalization_enabled=bool(
            cfg.text_processing.normalization_enabled
        ),
        normalization_profile=str(
            cfg.text_processing.normalization_profile
        ),
        context_replacements_enabled=bool(
            cfg.text_processing.context_replacements_enabled
        ),
        context_replacer_device=(
            str(cfg.text_processing.context_replacer_device)
            if cfg.text_processing.context_replacer_device is not None
            else None
        ),
    )
    for task, source_text, prepared_text in zip(
        tasks,
        source_texts,
        prepared_texts,
    ):
        task["source_text"] = source_text
        task["text"] = prepared_text


def _tasks_have_word_spans(
    tasks: list[dict[str, Any]],
    cfg: DictConfig,
) -> tuple[bool, list[int]]:
    from src.piper.preprocess import phonemize_text_for_infer

    invalid_task_ids = []
    for task in tasks:
        locale = str(cfg.validation.languages[task["language"]].locale)
        phoneme_result = phonemize_text_for_infer(
            task["text"],
            {
                "language": {"code": locale},
                "espeak": {"voice": locale, "primary": "en-us"},
            },
            neural=False,
        )
        if not phoneme_result.get("word_spans"):
            invalid_task_ids.append(int(task["task_id"]))
    return not invalid_task_ids, invalid_task_ids


def _repair_jsonl(path: Path) -> None:
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    last_newline = data.rfind(b"\n")
    repaired = data[: last_newline + 1] if last_newline >= 0 else b""
    with path.open("wb") as handle:
        handle.write(repaired)
        handle.flush()
        os.fsync(handle.fileno())


def _load_completed(
    accepted_manifest: Path,
    rejected_manifest: Path,
) -> tuple[dict[int, dict[str, Any]], int, int]:
    completed = {}
    counts = []
    for path in (accepted_manifest, rejected_manifest):
        _repair_jsonl(path)
        count = 0
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    task_id = int(row["task_id"])
                    if task_id in completed:
                        raise RuntimeError(f"Duplicate completed task_id={task_id}")
                    completed[task_id] = row
                    count += 1
        counts.append(count)
    return completed, counts[0], counts[1]


def _atomic_write_audio(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sf.write(temporary, audio, sample_rate, format="WAV", subtype="PCM_16")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_batch(
    results: list[dict[str, Any]],
    sample_rate: int,
    validator: Any,
    asr_model: Any,
    scoreq: ScoreQ,
    cfg: DictConfig,
) -> list[dict[str, Any]]:
    from src.tts_wer import calculate_multilingual_phoneme_adjusted_wer

    usable_indices = [
        index
        for index, result in enumerate(results)
        if result["error"] is None and result["audio"].size > 0
    ]
    budgets = {}
    transcriptions = {}
    if usable_indices:
        texts = [results[index]["task"]["text"] for index in usable_indices]
        locales = [
            str(cfg.validation.languages[results[index]["task"]["language"]].locale)
            for index in usable_indices
        ]
        predicted = validator.predict_batch(texts, languages=locales)
        budgets = dict(zip(usable_indices, predicted))
        asr_languages = [
            str(cfg.validation.languages[results[index]["task"]["language"]].asr)
            for index in usable_indices
        ]
        asr_results = asr_model.transcribe(
            audio=[(results[index]["audio"], sample_rate) for index in usable_indices],
            language=asr_languages,
        )
        if len(asr_results) != len(usable_indices):
            raise RuntimeError(
                f"Qwen3-ASR returned {len(asr_results)} rows for "
                f"{len(usable_indices)} inputs"
            )
        transcriptions = dict(zip(usable_indices, asr_results))

    validated = []
    for index, result in enumerate(results):
        task = result["task"]
        language = task["language"]
        audio = result["audio"]
        failures = []
        if result["error"] is not None:
            failures.append("generation")
            validation = {
                "valid": False,
                "reason": "generation_error",
                "error": result["error"],
            }
            asr_text = ""
            asr_language = ""
            wer = None
            score = None
            expected_seconds = None
        elif audio.size == 0:
            failures.append("generation")
            validation = {"valid": False, "reason": "empty_audio"}
            asr_text = ""
            asr_language = ""
            wer = None
            score = None
            expected_seconds = None
        else:
            budget = budgets[index]
            expected_seconds = float(budget["p50_seconds"])
            validation = validator.validate_alignment(
                text=task["text"],
                wav_data=audio,
                sample_rate=sample_rate,
                language=str(cfg.validation.languages[language].locale),
                expected_seconds=expected_seconds,
                duration_tolerance=float(cfg.validation.mas.duration_tolerance),
                reject_zero_phoneme_duration=True,
            )
            if not bool(validation.get("valid", False)):
                failures.append("mas")

            transcription = transcriptions[index]
            asr_text = str(transcription.text or "").strip()
            asr_language = str(transcription.language or "").strip()
            wer = calculate_multilingual_phoneme_adjusted_wer(
                task["text"],
                asr_text,
                language,
            )
            if float(wer["phoneme_adjusted_wer"]) > float(
                cfg.validation.wer.max_error_rate
            ):
                failures.append("wer")

            score = scoreq.score(audio, sample_rate)
            if score < float(cfg.validation.scoreq.thresholds[language]):
                failures.append("scoreq")

        validated.append(
            {
                **result,
                "accepted": not failures,
                "failed_gates": failures,
                "mas": validation,
                "expected_seconds": expected_seconds,
                "asr_text": asr_text,
                "asr_language": asr_language,
                "wer": wer,
                "scoreq": score,
            }
        )
    return validated


def _record_result(
    result: dict[str, Any],
    output_dir: Path,
    sample_rate: int,
    cfg: DictConfig,
    language_ids: dict[str, int],
    reference_audio: str | None,
) -> dict[str, Any]:
    task = result["task"]
    accepted = bool(result["accepted"])
    category = "accepted" if accepted else "rejected"
    audio_path = None
    if result["audio"].size > 0:
        audio_path = (
            output_dir
            / f"{category}_audio"
            / task["language"]
            / f"task-{int(task['task_id']):012d}.wav"
        )
        _atomic_write_audio(audio_path, result["audio"], sample_rate)

    row = {
        "task_id": int(task["task_id"]),
        "voice_group_id": int(task["voice_group_id"]),
        "voice_sample_index": int(task["voice_sample_index"]),
        "voice_anchor_task_id": int(task["voice_anchor_task_id"]),
        "voice_seed": int(task["voice_seed"]),
        "text": task["text"],
        "source_text": task["source_text"],
        "audio": str(audio_path) if audio_path else None,
        "duration": float(result["audio"].size / sample_rate),
        "language": task["language"],
        "dataset_id": language_ids[task["language"]],
        "source_line": int(task["source_line"]),
        "corpus_round": int(task["corpus_round"]),
        "seed": int(task["seed"]),
        "reference_audio": reference_audio,
        "decision": category,
        "failed_gates": result["failed_gates"],
        "generation_error": result["error"],
        "generation_wall_seconds": float(result["generation_wall_seconds"]),
        "generation_chunks": int(result["chunks"]),
        "worker_id": int(result["worker_id"]),
        "gpu": int(result["gpu"]),
        "expected_seconds": result["expected_seconds"],
        "mas": result["mas"],
        "asr_text": result["asr_text"],
        "asr_language": result["asr_language"],
        "wer": result["wer"],
        "scoreq": result["scoreq"],
        "thresholds": {
            "mas_duration_tolerance": float(
                cfg.validation.mas.duration_tolerance
            ),
            "phoneme_adjusted_wer_max": float(
                cfg.validation.wer.max_error_rate
            ),
            "scoreq_min": float(
                cfg.validation.scoreq.thresholds[task["language"]]
            ),
        },
    }
    _append_jsonl(output_dir / f"{category}.jsonl", row)
    return row


def _summary(
    *,
    accepted: int,
    rejected: int,
    started: float,
    completed: dict[int, dict[str, Any]],
    workers: list[dict[str, Any]],
    target: int | None,
) -> dict[str, Any]:
    total = accepted + rejected
    return {
        "accepted": accepted,
        "rejected": rejected,
        "attempts": total,
        "acceptance_rate": accepted / total if total else None,
        "max_accepted_items": target,
        "completed_task_ids": len(completed),
        "process_wall_seconds": time.perf_counter() - started,
        "workers": workers,
        "updated_at_unix": time.time(),
    }


@hydra.main(
    version_base=None,
    config_path="../local/configs/voxcpm",
    config_name="sft_generate",
)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("VoxCPM SFT generation requires CUDA")

    lz_tts_root = _absolute(cfg.paths.lz_tts_root)
    sys.path.insert(0, str(lz_tts_root))
    model_path = _absolute(cfg.model.pretrained_path)
    nanovllm_path = _absolute(cfg.paths.nanovllm_voxcpm)
    output_dir = _absolute(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    languages = [str(language) for language in cfg.source.languages]
    missing_validation = [
        language
        for language in languages
        if language not in cfg.validation.languages
        or language not in cfg.validation.scoreq.thresholds
    ]
    if missing_validation:
        raise ValueError(
            f"Missing validation settings for languages: {missing_validation}"
        )

    gpu_ids = [int(gpu) for gpu in cfg.workers.gpus]
    memory_utilizations = [
        float(value) for value in cfg.workers.gpu_memory_utilization
    ]
    enforce_eager_values = [
        bool(value) for value in cfg.workers.enforce_eager
    ]
    if len(gpu_ids) != len(memory_utilizations):
        raise ValueError(
            "workers.gpus and workers.gpu_memory_utilization must have equal length"
        )
    if len(gpu_ids) != len(enforce_eager_values):
        raise ValueError(
            "workers.gpus and workers.enforce_eager must have equal length"
        )
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("Each generation worker must use a distinct GPU")

    accepted_manifest = output_dir / "accepted.jsonl"
    rejected_manifest = output_dir / "rejected.jsonl"
    completed, accepted_count, rejected_count = _load_completed(
        accepted_manifest,
        rejected_manifest,
    )
    samples_per_voice = int(cfg.generation.samples_per_voice)
    if samples_per_voice <= 0:
        raise ValueError("generation.samples_per_voice must be positive")
    if samples_per_voice > int(cfg.workers.batch_size):
        raise ValueError(
            "generation.samples_per_voice cannot exceed workers.batch_size"
        )
    voice_anchor_selection = str(cfg.generation.voice_anchor_selection)
    if voice_anchor_selection not in {"first", "longest_text"}:
        raise ValueError(
            "generation.voice_anchor_selection must be first or longest_text"
        )
    original_config_path = output_dir / "config.yaml"
    if completed and original_config_path.is_file():
        original_config = OmegaConf.load(original_config_path)
        original_samples_per_voice = int(
            OmegaConf.select(
                original_config,
                "generation.samples_per_voice",
                default=1,
            )
        )
        if original_samples_per_voice != samples_per_voice:
            raise ValueError(
                "Cannot resume with a different samples_per_voice setting: "
                f"{original_samples_per_voice} != {samples_per_voice}"
            )
        original_anchor_selection = str(
            OmegaConf.select(
                original_config,
                "generation.voice_anchor_selection",
                default="first",
            )
        )
        if original_anchor_selection != voice_anchor_selection:
            raise ValueError(
                "Cannot resume with a different voice_anchor_selection: "
                f"{original_anchor_selection} != {voice_anchor_selection}"
            )
    max_accepted = (
        int(cfg.max_accepted_items)
        if cfg.max_accepted_items is not None
        else None
    )
    if max_accepted is not None and max_accepted <= 0:
        raise ValueError("max_accepted_items must be null or a positive integer")
    if max_accepted is not None and accepted_count >= max_accepted:
        print(
            f"Target already reached: accepted={accepted_count} "
            f"target={max_accepted}",
            flush=True,
        )
        return

    corpora = _load_corpora(cfg)
    orders = _build_prompt_orders(corpora, int(cfg.seed))
    generation_limit = _generation_limit(
        model_path,
        float(cfg.generation.max_audio_seconds),
    )
    run_config = OmegaConf.to_container(cfg, resolve=True)
    run_config["resolved"] = {
        "generation_max_generate_length": generation_limit,
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "corpus_sizes": {
            language: len(lines) for language, lines in corpora.items()
        },
        "single_writer": True,
        "reference_conditioning": None,
        "samples_per_voice": samples_per_voice,
    }
    if not (output_dir / "config.yaml").exists():
        OmegaConf.save(OmegaConf.create(run_config), output_dir / "config.yaml")
    OmegaConf.save(
        OmegaConf.create(run_config),
        output_dir / "config.latest.yaml",
    )

    context = mp.get_context("spawn")
    output_queue = context.Queue(maxsize=max(2, len(gpu_ids) * 2))
    input_queues = [context.Queue(maxsize=1) for _ in gpu_ids]
    processes = []
    worker_descriptions = []
    inherited_environment = {
        name: os.environ.get(name)
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "TORCHINDUCTOR_CACHE_DIR",
            "TRITON_CACHE_DIR",
        )
    }
    try:
        for worker_id, (gpu, memory_utilization, enforce_eager) in enumerate(
            zip(gpu_ids, memory_utilizations, enforce_eager_values)
        ):
            worker_config = {
                "model_path": str(model_path),
                "nanovllm_path": str(nanovllm_path),
                "inference_timesteps": int(cfg.generation.inference_timesteps),
                "max_num_batched_tokens": int(
                    cfg.generation.max_num_batched_tokens
                ),
                "batch_size": int(cfg.workers.batch_size),
                "max_model_len": int(cfg.generation.max_model_len),
                "gpu_memory_utilization": memory_utilization,
                "enforce_eager": enforce_eager,
                "temperature": float(cfg.generation.temperature),
                "cfg_value": float(cfg.generation.cfg_value),
                "max_audio_seconds": float(cfg.generation.max_audio_seconds),
                "max_generate_length": generation_limit,
            }
            process = context.Process(
                target=_worker_entry,
                args=(
                    worker_id,
                    gpu,
                    worker_config,
                    input_queues[worker_id],
                    output_queue,
                ),
                name=f"voxcpm-sft-gpu-{gpu}",
            )
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = (
                f"/tmp/torchinductor_voxcpm_sft_gpu{gpu}"
            )
            os.environ["TRITON_CACHE_DIR"] = (
                f"/tmp/triton_voxcpm_sft_gpu{gpu}"
            )
            process.start()
            processes.append(process)
            worker_descriptions.append(
                {
                    "worker_id": worker_id,
                    "gpu": gpu,
                    "logical_gpu": 0,
                    "gpu_memory_utilization": memory_utilization,
                    "enforce_eager": enforce_eager,
                }
            )
    finally:
        for name, value in inherited_environment.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    ready_workers = set()
    sample_rate = None
    try:
        while len(ready_workers) < len(processes):
            message = output_queue.get()
            if message["kind"] == "worker_error":
                raise RuntimeError(
                    f"Worker {message['worker_id']} failed during startup: "
                    f"{message['error']}"
                )
            if message["kind"] != "ready":
                raise RuntimeError(f"Unexpected startup message: {message['kind']}")
            ready_workers.add(int(message["worker_id"]))
            worker_sample_rate = int(message["sample_rate"])
            if sample_rate is None:
                sample_rate = worker_sample_rate
            elif sample_rate != worker_sample_rate:
                raise RuntimeError("Generation workers disagree on sample rate")
            print(
                f"worker={message['worker_id']} gpu={message['gpu']} ready "
                f"sample_rate={worker_sample_rate}",
                flush=True,
            )
        assert sample_rate is not None

        torch.cuda.set_device(torch.device(str(cfg.validation.device)))
        validator = _build_validator(cfg)
        asr_model = _build_asr(cfg)
        scoreq = ScoreQ(
            _absolute(cfg.validation.scoreq.model_path),
            int(cfg.validation.scoreq.cpu_threads),
        )

        started = time.perf_counter()
        next_voice_group_id = 0
        batch_id = 0
        inflight: dict[int, int] = {}
        skipped_task_ids: set[int] = set()
        progress = tqdm(
            total=max_accepted,
            initial=accepted_count if max_accepted is not None else 0,
            unit="accepted",
            dynamic_ncols=True,
        )

        def group_is_complete(voice_group_id: int) -> bool:
            first_task_id = voice_group_id * samples_per_voice
            return all(
                first_task_id + sample_index in completed
                or first_task_id + sample_index in skipped_task_ids
                for sample_index in range(samples_per_voice)
            )

        def dispatch(worker_id: int) -> bool:
            nonlocal next_voice_group_id, batch_id
            if max_accepted is not None:
                remaining_capacity = (
                    max_accepted
                    - accepted_count
                    - sum(inflight.values())
                )
                if remaining_capacity <= 0:
                    return False
                group_count = min(
                    max(1, int(cfg.workers.batch_size) // samples_per_voice),
                    math.ceil(remaining_capacity / samples_per_voice),
                )
            else:
                group_count = max(
                    1,
                    int(cfg.workers.batch_size) // samples_per_voice,
                )

            groups = []
            pending_count = 0
            while len(groups) < group_count:
                while group_is_complete(next_voice_group_id):
                    next_voice_group_id += 1
                first_task_id = next_voice_group_id * samples_per_voice
                all_tasks = [
                    _task_for_id(
                        first_task_id + sample_index,
                        corpora,
                        orders,
                        languages,
                        int(cfg.seed),
                        samples_per_voice,
                        voice_anchor_selection,
                    )
                    for sample_index in range(samples_per_voice)
                ]
                _prepare_tasks(all_tasks, cfg)
                has_word_spans, invalid_task_ids = _tasks_have_word_spans(
                    all_tasks,
                    cfg,
                )
                if not has_word_spans:
                    skipped_task_ids.update(invalid_task_ids)
                    print(
                        "Skipping source entries before generation: "
                        f"voice_group_id={next_voice_group_id} "
                        f"task_ids_without_word_spans={invalid_task_ids}",
                        flush=True,
                    )
                valid_tasks = [
                    task
                    for task in all_tasks
                    if int(task["task_id"]) not in skipped_task_ids
                ]
                if not valid_tasks:
                    next_voice_group_id += 1
                    continue
                anchor_task = valid_tasks[0]
                anchor_task_id = int(anchor_task["task_id"])
                for sample_index, task in enumerate(valid_tasks):
                    task["voice_sample_index"] = sample_index
                    task["voice_anchor_task_id"] = anchor_task_id
                pending_tasks = [
                    task
                    for task in valid_tasks
                    if int(task["task_id"]) not in completed
                ]
                if not pending_tasks:
                    next_voice_group_id += 1
                    continue
                anchor_row = completed.get(anchor_task_id)
                prompt_audio_path = (
                    anchor_row.get("audio")
                    if anchor_row is not None and anchor_row.get("audio")
                    else None
                )
                if prompt_audio_path is not None and not Path(
                    prompt_audio_path
                ).is_file():
                    prompt_audio_path = None
                groups.append(
                    {
                        "anchor_task": anchor_task,
                        "tasks": pending_tasks,
                        "prompt_audio_path": prompt_audio_path,
                    }
                )
                pending_count += len(pending_tasks)
                next_voice_group_id += 1
            input_queues[worker_id].put(
                {"batch_id": batch_id, "groups": groups}
            )
            inflight[worker_id] = pending_count
            batch_id += 1
            return True

        for worker_id in range(len(processes)):
            dispatch(worker_id)

        while inflight:
            message = output_queue.get()
            if message["kind"] == "worker_error":
                raise RuntimeError(
                    f"Worker {message['worker_id']} failed: {message['error']}"
                )
            if message["kind"] != "batch":
                continue
            worker_id = int(message["worker_id"])
            inflight.pop(worker_id, None)
            for result in message["results"]:
                result["worker_id"] = worker_id
                result["gpu"] = int(message["gpu"])
            validated = _validate_batch(
                message["results"],
                sample_rate,
                validator,
                asr_model,
                scoreq,
                cfg,
            )
            for result in validated:
                task = result["task"]
                reference_audio = None
                if int(task["voice_sample_index"]) > 0:
                    anchor_row = completed.get(int(task["voice_anchor_task_id"]))
                    if anchor_row is not None:
                        reference_audio = anchor_row.get("audio")
                row = _record_result(
                    result,
                    output_dir,
                    sample_rate,
                    cfg,
                    {language: index for index, language in enumerate(languages)},
                    reference_audio,
                )
                completed[int(row["task_id"])] = row
                if row["decision"] == "accepted":
                    accepted_count += 1
                    progress.update(1)
                else:
                    rejected_count += 1

            progress.set_postfix(
                accepted=accepted_count,
                rejected=rejected_count,
                rate=(
                    f"{accepted_count / (accepted_count + rejected_count):.1%}"
                    if accepted_count + rejected_count
                    else "-"
                ),
                refresh=True,
            )
            _atomic_write_json(
                output_dir / "summary.json",
                _summary(
                    accepted=accepted_count,
                    rejected=rejected_count,
                    started=started,
                    completed=completed,
                    workers=worker_descriptions,
                    target=max_accepted,
                ),
            )
            dispatch(worker_id)
        progress.close()
    finally:
        for input_queue in input_queues:
            try:
                input_queue.put_nowait(None)
            except queue.Full:
                pass
        for process in processes:
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)


if __name__ == "__main__":
    main()
