#!/usr/bin/env python3
"""Hydra frontend for VoxCPM's standard LoRA SFT trainer."""

from __future__ import annotations

import importlib.util
import fcntl
import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.voxcpm_accent_filter import AccentFilterPolicy


def _path(value: str) -> str:
    return str(Path(to_absolute_path(os.path.expanduser(value))).resolve())


def _validate_one_to_one_references(
    path: str,
    expected_accent: str = "",
    policy: AccentFilterPolicy | None = None,
) -> int:
    seen = set()
    count = 0
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if policy is not None:
                policy.require_row(
                    row,
                    source=f"{path}:{line_number}",
                    expected_accent=expected_accent,
                )
            elif expected_accent and str(row.get("accent", "")) != expected_accent:
                raise ValueError(
                    f"{path}:{line_number} has accent {row.get('accent')!r}; "
                    f"expected {expected_accent!r}"
                )
            utterance_id = str(row.get("utterance_id", ""))
            prompt_utterance_id = str(row.get("prompt_utterance_id", ""))
            if not utterance_id or prompt_utterance_id != utterance_id:
                raise ValueError(
                    f"{path}:{line_number} is not a one-to-one reference: "
                    f"utterance_id={utterance_id!r}, prompt_utterance_id={prompt_utterance_id!r}"
                )
            if utterance_id in seen:
                raise ValueError(f"{path}:{line_number} duplicates utterance_id {utterance_id!r}")
            target_audio = Path(str(row.get("audio", "")))
            if not target_audio.is_file():
                raise FileNotFoundError(f"{path}:{line_number} missing target audio: {target_audio}")
            ref_audio = Path(str(row.get("ref_audio", "")))
            if not ref_audio.is_file():
                raise FileNotFoundError(f"{path}:{line_number} missing ref_audio: {ref_audio}")
            seen.add(utterance_id)
            count += 1
    if count == 0:
        raise ValueError(f"Training manifest is empty: {path}")
    return count


def _wait_for_manifest(path: str, poll_seconds: int) -> None:
    manifest = Path(path)
    while not manifest.is_file():
        if poll_seconds <= 0:
            raise FileNotFoundError(manifest)
        print(f"Waiting for training manifest: {manifest}", flush=True)
        time.sleep(poll_seconds)


@contextmanager
def _exclusive_training_lock(path: str):
    if not path:
        yield
        return
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        print(f"Waiting for training GPU lock: {lock_path}", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
        print(f"Acquired training GPU lock: {lock_path}", flush=True)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_upstream_trainer(voxcpm_root: Path) -> Any:
    path = voxcpm_root / "scripts" / "train_voxcpm_finetune.py"
    spec = importlib.util.spec_from_file_location("voxcpm_finetune_upstream", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load VoxCPM trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@hydra.main(version_base=None, config_path="../local/configs/voxcpm", config_name="accent_lora_train")
def main(cfg: DictConfig) -> None:
    train_manifest = _path(cfg.data.train_manifest)
    val_manifest = (
        _path(cfg.data.val_manifest)
        if bool(cfg.training.validation_enabled) and cfg.data.val_manifest
        else ""
    )
    batch_size = int(cfg.training.batch_size)
    grad_accum_steps = int(cfg.training.grad_accum_steps)
    expected_accent = str(cfg.data.get("expected_accent", ""))
    policy = AccentFilterPolicy.from_mapping(
        OmegaConf.to_container(cfg.accent_filter, resolve=True)
    )
    _wait_for_manifest(train_manifest, int(cfg.data.get("wait_for_manifest_seconds", 0)))
    validation_args = {
        "expected_accent": expected_accent,
        "policy": policy,
    }
    examples = _validate_one_to_one_references(train_manifest, **validation_args)
    validation_examples = (
        _validate_one_to_one_references(val_manifest, **validation_args) if val_manifest else 0
    )
    steps_per_epoch = math.ceil(examples / max(1, batch_size * grad_accum_steps))
    num_iters = int(cfg.training.num_iters) if cfg.training.num_iters is not None else steps_per_epoch * int(cfg.training.epochs)
    if num_iters <= 0:
        raise ValueError("Training must contain at least one optimizer step")

    output_dir = Path(_path(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8")
    run = {
        "train_examples": examples,
        "validation_examples": validation_examples,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "steps_per_epoch": steps_per_epoch,
        "num_iters": num_iters,
    }
    (output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(run, indent=2))

    with _exclusive_training_lock(str(cfg.training.get("gpu_lock_path", ""))):
        trainer = _load_upstream_trainer(Path(_path(cfg.paths.voxcpm_root)))
        trainer.train(
            pretrained_path=_path(cfg.model.pretrained_path),
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            sample_rate=int(cfg.training.sample_rate),
            out_sample_rate=int(cfg.training.output_sample_rate),
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            num_workers=int(cfg.training.num_workers),
            num_iters=num_iters,
            log_interval=int(cfg.training.log_interval),
            valid_interval=int(cfg.training.validation_interval),
            save_interval=int(cfg.training.save_interval),
            retain_interval=int(cfg.training.retain_interval),
            learning_rate=float(cfg.optimizer.learning_rate),
            weight_decay=float(cfg.optimizer.weight_decay),
            warmup_steps=int(cfg.optimizer.warmup_steps),
            max_steps=num_iters,
            max_batch_tokens=int(cfg.training.max_batch_tokens),
            max_grad_norm=float(cfg.optimizer.max_grad_norm),
            save_path=str(output_dir / "checkpoints"),
            tensorboard=str(output_dir / "tensorboard"),
            lambdas=OmegaConf.to_container(cfg.objective, resolve=True),
            lora=OmegaConf.to_container(cfg.lora, resolve=True),
            config_path="",
            distribute=False,
        )


if __name__ == "__main__":
    main()
