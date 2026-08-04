"""Hydra entry point for Piper/VITS training."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .train import train_from_args

_LOGGER = logging.getLogger(__package__)

_SECTIONS = (
    "dataset",
    "dataloader",
    "trainer",
    "model",
    "optimizer",
    "checkpoint",
    "quality_monitor",
)

_DEFAULTS: dict[str, Any] = {
    "accelerator": "auto",
    "devices": "auto",
    "precision": "32",
    "max_epochs": 10_000,
    "default_root_dir": None,
    "log_every_n_steps": 50,
    "progress_log_every_n_steps": 0,
    "enable_progress_bar": True,
    "epoch_summary_log": True,
    "cudnn_benchmark": False,
    "gradient_clip_val": 0.0,
    "accumulate_grad_batches": 1,
    "checkpoint_epochs": 1,
    "keep_last_checkpoints": 5,
    "retain_every": 0,
    "batch_size": 1,
    "num_workers": 1,
    "use_length_buckets": False,
    "bucket_boundaries": None,
    "validation_split": 0.1,
    "num_test_examples": 5,
    "max_phoneme_ids": None,
    "quality": "high",
    "resume_from_single_speaker_checkpoint": None,
    "init_from_checkpoint": None,
    "resume_from_checkpoint": None,
    "init_partial_from_checkpoint": None,
    "init_partial_include_prefixes": None,
    "init_partial_exclude_prefixes": ("dec.",),
    "speaker_embedding_init_map": None,
    "utmos_enabled": False,
    "utmos_every_n_epochs": 10,
    "utmos_num_samples": 10,
    "utmos_output_dir": None,
    "utmos_python": "local/utmos_probe/.venv/bin/python",
    "utmos_worker": "local/utmos_probe/utmos_stdin_worker.py",
    "utmos_cuda_visible_devices": None,
    "utmos_noise_scale": 0.667,
    "utmos_length_scale": 1.0,
    "utmos_noise_w": 0.8,
    "utmos_sdp_ratio": 0.2,
    "seed": 1234,
}


@hydra.main(
    version_base=None,
    config_path="../../local/configs/piper",
    config_name="train",
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _args_from_config(cfg)
    _configure_run_file_logging(args.default_root_dir)
    _LOGGER.info(
        "Piper training: dataset=%s root=%s epochs=%s batch=%s precision=%s "
        "progress_bar=%s cudnn_benchmark=%s",
        args.dataset_dir,
        args.default_root_dir,
        args.max_epochs,
        args.batch_size,
        args.precision,
        args.enable_progress_bar,
        args.cudnn_benchmark,
    )
    train_from_args(args)


def _configure_run_file_logging(root_dir: str | None) -> None:
    if not root_dir:
        return
    log_path = Path(root_dir) / "train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved = str(log_path.resolve())
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if getattr(handler, "baseFilename", None) == resolved:
                return

    file_handler = logging.FileHandler(resolved, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    )
    root_logger.addHandler(file_handler)


def _args_from_config(cfg: DictConfig) -> SimpleNamespace:
    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise TypeError("Hydra config must resolve to a dictionary")

    args = dict(_DEFAULTS)
    for section in _SECTIONS:
        section_values = raw.get(section) or {}
        if not isinstance(section_values, dict):
            raise TypeError(f"Hydra section '{section}' must be a dictionary")
        args.update(section_values)

    for key, value in raw.items():
        if key not in _SECTIONS and key != "hydra":
            args[key] = value

    args["resume_from_checkpoint"] = _resolve_resume_checkpoint(
        args.get("resume_from_checkpoint"), args.get("default_root_dir")
    )

    return SimpleNamespace(**args)


def _resolve_resume_checkpoint(value: Any, root_dir: str | None) -> str | None:
    if value != "latest":
        return value

    if not root_dir:
        return None

    latest = _find_latest_checkpoint(Path(root_dir))
    if latest is None:
        _LOGGER.info("No checkpoint found under %s; starting a fresh run", root_dir)
        return None

    _LOGGER.info("Resuming latest checkpoint: %s", latest)
    return str(latest)


def _find_latest_checkpoint(root_dir: Path) -> Path | None:
    if not root_dir.exists():
        return None

    candidates = list((root_dir / "lightning_logs").rglob("*.ckpt"))
    candidates.extend(root_dir.glob("checkpoints/*.ckpt"))
    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)


if __name__ == "__main__":
    main()
