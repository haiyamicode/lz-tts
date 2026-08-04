#!/usr/bin/env python3
"""Generate one cross-language VoxCPM reference for every accent target."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.voxcpm_accent_filter import AccentFilterPolicy


def _path(value: str) -> Path:
    return Path(to_absolute_path(os.path.expanduser(value))).resolve()


def _load_source_manifest(
    path: Path,
    policy: AccentFilterPolicy | None = None,
) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"audio", "text", "speaker_id", "accent", "utterance_id"}
            missing = required.difference(row)
            if missing:
                raise ValueError(f"{path}:{line_number} missing fields: {sorted(missing)}")
            utterance_id = str(row["utterance_id"])
            if utterance_id in seen:
                raise ValueError(f"Duplicate utterance_id in source manifest: {utterance_id}")
            audio = Path(str(row["audio"])).resolve()
            if not audio.is_file():
                raise FileNotFoundError(f"Source audio does not exist: {audio}")
            text = " ".join(str(row["text"]).split())
            if not text:
                raise ValueError(f"{path}:{line_number} has empty text")
            if policy is not None:
                policy.require_row(row, source=f"{path}:{line_number}")
            rows.append({**row, "audio": str(audio), "text": text, "utterance_id": utterance_id})
            seen.add(utterance_id)
    if not rows:
        raise ValueError(f"Source manifest is empty: {path}")
    return rows


def _load_corpus(
    path: Path,
    languages: list[str],
    min_characters: int,
    max_characters: int,
) -> dict[str, list[dict[str, Any]]]:
    requested = set(languages)
    corpus: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("|", 2)
            if len(fields) != 3 or fields[1].lower() not in requested:
                continue
            text = " ".join(fields[2].split())
            if (
                min_characters <= len(text) <= max_characters
                and any(character.isalnum() for character in text)
                and '""' not in text
            ):
                corpus[fields[1].lower()].append(
                    {"source_id": fields[0], "source_line": line_number, "text": text}
                )
    missing = requested.difference(corpus)
    if missing:
        raise ValueError(f"Corpus has no entries for languages: {sorted(missing)}")
    return dict(corpus)


def _stable_int(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")


def _build_tasks(cfg: DictConfig) -> list[dict[str, Any]]:
    policy = AccentFilterPolicy.from_mapping(
        OmegaConf.to_container(cfg.accent_filter, resolve=True)
    )
    targets = _load_source_manifest(
        _path(cfg.source.manifest),
        policy=policy,
    )
    included_accents = {str(value) for value in cfg.source.get("include_accents", [])}
    if included_accents:
        targets = [target for target in targets if str(target["accent"]) in included_accents]
        if not targets:
            raise ValueError(f"Source manifest has no targets for accents: {sorted(included_accents)}")
    languages = [str(value) for value in cfg.source.reference_languages]
    corpus = _load_corpus(
        _path(cfg.source.corpus_manifest),
        languages,
        int(cfg.source.text_min_characters),
        int(cfg.source.text_max_characters),
    )
    seed = int(cfg.seed)
    tasks = []
    for target in sorted(targets, key=lambda row: str(row["utterance_id"])):
        utterance_id = str(target["utterance_id"])
        language = languages[_stable_int(f"language:{seed}:{utterance_id}") % len(languages)]
        choices = corpus[language]
        source = choices[_stable_int(f"text:{seed}:{utterance_id}:{language}") % len(choices)]
        tasks.append(
            {
                **target,
                "target_audio": target["audio"],
                "target_text": target["text"],
                "reference_language": language,
                "reference_text": source["text"],
                "reference_text_source_id": source["source_id"],
                "reference_text_source_line": source["source_line"],
                "prompt_audio": target["audio"],
                "prompt_text": target["text"],
                "prompt_utterance_id": utterance_id,
                "seed": _stable_int(f"generation:{seed}:{utterance_id}") % (2**31 - 1),
            }
        )
    if bool(cfg.text_processing.normalization_enabled):
        from src.text_norm import prepare_tts_texts

        locales = {
            str(key): str(value)
            for key, value in OmegaConf.to_container(
                cfg.text_processing.locales,
                resolve=True,
            ).items()
        }
        prepared = prepare_tts_texts(
            [task["reference_text"] for task in tasks],
            [locales[task["reference_language"]] for task in tasks],
            normalization_enabled=True,
            normalization_profile=str(cfg.text_processing.normalization_profile),
            context_replacements_enabled=False,
        )
        for task, text in zip(tasks, prepared):
            task["reference_text"] = text
    return tasks


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed = {}
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row["utterance_id"])
            if row.get("reference_audio") and Path(row["reference_audio"]).is_file():
                completed[key] = row
    return completed


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, path)


def _save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    sf.write(temporary, audio, sample_rate, format="WAV", subtype="PCM_16")
    os.replace(temporary, path)


def _generation_limit(model_path: Path, max_audio_seconds: float) -> int:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    audio_config = config["audio_vae_config"]
    stride = math.prod(audio_config["encoder_rates"])
    seconds_per_step = config["patch_size"] * stride / audio_config["sample_rate"]
    return max(1, math.floor(max_audio_seconds / seconds_per_step))


def _build_manifests(
    output_dir: Path,
    references: dict[str, dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, int]:
    val_fraction = float(cfg.dataset.validation_speaker_fraction)
    seed = int(cfg.seed)
    speakers_by_accent: dict[str, list[str]] = defaultdict(list)
    speaker_accents = {
        (str(row["speaker_id"]), str(row["accent"]))
        for row in references.values()
    }
    for speaker_id, accent in speaker_accents:
        speakers_by_accent[accent].append(speaker_id)
    validation_speakers: set[tuple[str, str]] = set()
    for accent, speaker_ids in speakers_by_accent.items():
        ranked = sorted(
            speaker_ids,
            key=lambda speaker_id: _stable_int(f"split:{seed}:{accent}:{speaker_id}"),
        )
        count = max(1, round(len(ranked) * val_fraction)) if val_fraction > 0 else 0
        validation_speakers.update((accent, speaker_id) for speaker_id in ranked[:count])
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for row in sorted(references.values(), key=lambda item: str(item["utterance_id"])):
        speaker_id = str(row["speaker_id"])
        accent = str(row["accent"])
        split = "val" if (accent, speaker_id) in validation_speakers else "train"
        manifests[split].append(
            {
                "audio": row["target_audio"],
                "text": row["target_text"],
                "duration": row.get("duration"),
                "ref_audio": row["reference_audio"],
                "ref_duration": row["reference_duration"],
                "ref_text": row["reference_text"],
                "language": "en",
                "reference_language": row["reference_language"],
                "speaker_id": speaker_id,
                "accent": accent,
                "source_accent": row.get("source_accent"),
                "accent_classifier_label": row.get("accent_classifier_label"),
                "accent_classifier_target": row.get("accent_classifier_target"),
                "accent_classifier_confidence": row.get("accent_classifier_confidence"),
                "accent_classifier_clips": row.get("accent_classifier_clips"),
                "accent_classifier_confirming_clips": row.get(
                    "accent_classifier_confirming_clips"
                ),
                "accent_classifier_confirms_source": row.get(
                    "accent_classifier_confirms_source"
                ),
                "accent_source_metadata_agrees": row.get("accent_source_metadata_agrees"),
                "accent_sample_classifier_label": row.get(
                    "accent_sample_classifier_label"
                ),
                "accent_sample_classifier_target": row.get(
                    "accent_sample_classifier_target"
                ),
                "accent_sample_classifier_confidence": row.get(
                    "accent_sample_classifier_confidence"
                ),
                "accent_sample_classifier_passes": row.get(
                    "accent_sample_classifier_passes"
                ),
                "utterance_id": row["utterance_id"],
                "prompt_utterance_id": row["prompt_utterance_id"],
                "dataset_id": 0,
            }
        )
    output_rows = dict(manifests)
    for split, rows in manifests.items():
        for accent in sorted({row["accent"] for row in rows}):
            output_rows[f"{split}_{accent}"] = [row for row in rows if row["accent"] == accent]
    for name, rows in output_rows.items():
        temporary = output_dir / f".{name}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, output_dir / f"{name}.jsonl")
    return {name: len(rows) for name, rows in output_rows.items()}


async def _run(cfg: DictConfig) -> None:
    output_dir = _path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    references_path = output_dir / "references.jsonl"
    tasks = _build_tasks(cfg)
    tasks_by_id = {str(task["utterance_id"]): task for task in tasks}
    completed = _load_completed(references_path)
    stale = sorted(set(completed).difference(tasks_by_id))
    if stale:
        if not bool(cfg.dataset.get("prune_stale_references", False)):
            raise ValueError(
                f"{references_path} contains {len(stale)} rows outside the current source manifest"
            )
        audio_root = (output_dir / "audio").resolve()
        for utterance_id in stale:
            audio_path = Path(str(completed[utterance_id].get("reference_audio", ""))).resolve()
            if audio_path.is_relative_to(audio_root):
                audio_path.unlink(missing_ok=True)
            del completed[utterance_id]
    for utterance_id, row in completed.items():
        task = tasks_by_id[utterance_id]
        if (
            str(row.get("prompt_utterance_id", "")) != utterance_id
            or Path(str(row.get("prompt_audio", ""))).resolve()
            != Path(str(task["prompt_audio"])).resolve()
        ):
            raise ValueError(
                f"{references_path} contains an incompatible completed row for {utterance_id}"
            )
        completed[utterance_id] = {
            **task,
            "reference_audio": row["reference_audio"],
            "reference_duration": row["reference_duration"],
            "reference_sample_rate": row.get("reference_sample_rate"),
        }
    _write_jsonl(
        references_path,
        sorted(completed.values(), key=lambda row: str(row["utterance_id"])),
    )
    pending = [task for task in tasks if str(task["utterance_id"]) not in completed]
    if pending and bool(cfg.generation.get("generate_missing_references", True)):
        nanovllm_path = str(_path(cfg.paths.nanovllm_voxcpm))
        if nanovllm_path not in sys.path:
            sys.path.insert(0, nanovllm_path)
        from nanovllm_voxcpm import VoxCPM

        model_path = _path(cfg.model.pretrained_path)
        max_generate_length = _generation_limit(
            model_path, float(cfg.generation.max_audio_seconds)
        )
        gpu = int(cfg.worker.gpu)
        torch.cuda.set_device(gpu)
        server = VoxCPM.from_pretrained(
            model=str(model_path),
            devices=[gpu],
            inference_timesteps=int(cfg.generation.inference_timesteps),
            max_num_batched_tokens=int(cfg.worker.max_num_batched_tokens),
            max_num_seqs=int(cfg.worker.batch_size),
            max_model_len=int(cfg.worker.max_model_len),
            gpu_memory_utilization=float(cfg.worker.gpu_memory_utilization),
            enforce_eager=bool(cfg.worker.enforce_eager),
        )
        await server.wait_for_ready()
        sample_rate = int((await server.get_model_info())["sample_rate"])
        try:
            await _generate_pending(
                cfg,
                output_dir,
                references_path,
                pending,
                completed,
                server,
                sample_rate,
                max_generate_length,
            )
        finally:
            await server.stop()
    manifest_counts = _build_manifests(output_dir, completed, cfg)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    summary = {
        "speakers": len({row["speaker_id"] for row in completed.values()}),
        "references": len(completed),
        "eligible_targets": len(tasks),
        "missing_references": len(tasks) - len(completed),
        "one_reference_per_target": True,
        "manifests": manifest_counts,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


async def _generate_pending(
    cfg: DictConfig,
    output_dir: Path,
    references_path: Path,
    pending: list[dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    server: Any,
    sample_rate: int,
    max_generate_length: int,
) -> None:
    tasks_total = len(completed) + len(pending)
    try:
        progress = tqdm(
            total=tasks_total,
            initial=len(completed),
            unit="ref",
            desc="one-to-one cross references",
            dynamic_ncols=True,
        )
        for offset in range(0, len(pending), int(cfg.worker.batch_size)):
            batch = pending[offset : offset + int(cfg.worker.batch_size)]
            prompt_ids = await asyncio.gather(
                *(
                    server.add_prompt(
                        Path(task["prompt_audio"]).read_bytes(),
                        Path(task["prompt_audio"]).suffix.lstrip(".").lower(),
                        task["prompt_text"],
                    )
                    for task in batch
                )
            )
            try:
                async def generate(
                    task: dict[str, Any], prompt_id: str
                ) -> tuple[dict[str, Any], np.ndarray, str | None]:
                    error = None
                    for attempt in range(int(cfg.generation.max_attempts)):
                        try:
                            chunks = []
                            async for chunk in server.generate(
                                target_text=task["reference_text"],
                                prompt_id=prompt_id,
                                max_generate_length=max_generate_length,
                                temperature=float(cfg.generation.temperature),
                                cfg_value=float(cfg.generation.cfg_value),
                                seed=int(task["seed"]) + attempt,
                            ):
                                chunks.append(chunk)
                            audio = (
                                np.concatenate(chunks).astype(np.float32, copy=False)
                                if chunks
                                else np.zeros(0, np.float32)
                            )
                            audio = audio[: round(float(cfg.generation.max_audio_seconds) * sample_rate)]
                            if audio.size >= round(float(cfg.generation.min_audio_seconds) * sample_rate):
                                return task, audio, None
                            error = f"audio shorter than {float(cfg.generation.min_audio_seconds):.2f}s"
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                    return task, np.zeros(0, np.float32), error

                results = await asyncio.gather(
                    *(generate(task, prompt_id) for task, prompt_id in zip(batch, prompt_ids))
                )
                for task, audio, error in results:
                    if error is not None:
                        _append_jsonl(output_dir / "failures.jsonl", {**task, "error": error})
                        progress.update(1)
                        continue
                    audio_path = (
                        output_dir
                        / "audio"
                        / str(task["accent"])
                        / str(task["speaker_id"])
                        / f"{task['utterance_id']}-{task['reference_language']}.wav"
                    )
                    _save_audio(audio_path, audio, sample_rate)
                    row = {
                        **task,
                        "reference_audio": str(audio_path),
                        "reference_duration": float(audio.size / sample_rate),
                        "reference_sample_rate": sample_rate,
                    }
                    _append_jsonl(references_path, row)
                    completed[str(task["utterance_id"])] = row
                    progress.update(1)
            finally:
                await asyncio.gather(
                    *(server.remove_prompt(prompt_id) for prompt_id in prompt_ids),
                    return_exceptions=True,
                )
        progress.close()
    finally:
        if "progress" in locals():
            progress.close()


@hydra.main(version_base=None, config_path="../local/configs/voxcpm", config_name="accent_refs_generate")
def main(cfg: DictConfig) -> None:
    asyncio.run(_run(cfg))


if __name__ == "__main__":
    main()
