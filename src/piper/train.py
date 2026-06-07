"""Piper/VITS training entry point for this repository.

This is a small Lightning 2.x compatible equivalent of Piper's
``piper_train.__main__``. It intentionally keeps semantic/BERT conditioning off
unless ``--use-bert`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path, PosixPath
import subprocess
import time
from typing import Any

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .vits.utils import audio_float_to_int16
from .vits.wavfile import write as write_wav
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# Lightning checkpoints written before PyTorch's weights_only default change can
# contain Path objects in trainer metadata. Allowlist only that local type so
# trusted in-repo resume checkpoints keep loading with the safe unpickler.
torch.serialization.add_safe_globals([PosixPath])


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog="python -m src.piper.train")
    parser.add_argument(
        "--dataset-dir", required=True, help="Path to pre-processed dataset directory"
    )
    parser.add_argument(
        "--checkpoint-epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--quality",
        default="high",
        choices=("x-low", "medium", "high", "custom", "bert-87m"),
        help="Quality/size of model",
    )
    parser.add_argument(
        "--resume_from_single_speaker_checkpoint",
        help="Convert a single-speaker checkpoint to multi-speaker and resume training",
    )
    parser.add_argument(
        "--init_from_checkpoint",
        help="Initialize weights from another checkpoint without optimizer state",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        help="Resume a Lightning run, including optimizer state",
    )
    parser.add_argument(
        "--init_partial_from_checkpoint",
        help="Initialize compatible generator weights from another checkpoint without optimizer state",
    )
    parser.add_argument(
        "--init_partial_include_prefixes",
        nargs="*",
        help=(
            "Whitelist model_g prefixes to copy for partial initialization. "
            "If omitted, all compatible generator tensors except excluded prefixes are considered."
        ),
    )
    parser.add_argument(
        "--init_partial_exclude_prefixes",
        nargs="*",
        default=("dec.",),
        help="model_g prefixes to skip for partial initialization",
    )
    parser.add_argument("--seed", type=int, default=1234)

    add_trainer_args(parser)
    add_quality_monitor_args(parser)
    VitsModel.add_model_specific_args(parser)
    args = parser.parse_args()
    train_from_args(args, parser=parser)


def train_from_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if isinstance(args.devices, str) and args.devices.isdigit():
        args.devices = int(args.devices)

    if args.resume_from_single_speaker_checkpoint and args.init_from_checkpoint:
        message = (
            "--resume_from_single_speaker_checkpoint cannot be combined with "
            "--init_from_checkpoint"
        )
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.default_root_dir is None:
        args.default_root_dir = str(dataset_dir)

    torch.backends.cudnn.benchmark = bool(getattr(args, "cudnn_benchmark", False))
    torch.manual_seed(args.seed)

    config_path = dataset_dir / "config.json"
    parquet_path = dataset_dir / "dataset.parquet"
    jsonl_path = dataset_dir / "dataset.jsonl"
    dataset_path = parquet_path if parquet_path.exists() else jsonl_path
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Expected {parquet_path} or {jsonl_path} for Piper training"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
        num_symbols = int(config["num_symbols"])
        num_speakers = int(config["num_speakers"])
        sample_rate = int(config["audio"]["sample_rate"])
        speaker_id_map = config.get("speaker_id_map") or {}

    model_args = vars(args).copy()
    apply_quality_preset(model_args, args.quality)

    model = VitsModel(
        num_symbols=num_symbols,
        num_speakers=num_speakers,
        sample_rate=sample_rate,
        dataset=[dataset_path],
        speaker_id_map=speaker_id_map,
        **model_args,
    )

    if args.resume_from_single_speaker_checkpoint:
        if num_speakers <= 1:
            message = (
                "--resume_from_single_speaker_checkpoint is only for multi-speaker models"
            )
            if parser is not None:
                parser.error(message)
            raise ValueError(message)
        initialize_from_checkpoint(
            model,
            args.resume_from_single_speaker_checkpoint,
            expect_multispeaker=False,
        )

    if args.init_from_checkpoint:
        initialize_from_checkpoint(model, args.init_from_checkpoint)

    if args.init_partial_from_checkpoint:
        initialize_compatible_generator_from_checkpoint(
            model,
            args.init_partial_from_checkpoint,
            include_prefixes=tuple(args.init_partial_include_prefixes or ()),
            exclude_prefixes=tuple(args.init_partial_exclude_prefixes or ()),
        )

    callbacks = []
    if args.checkpoint_epochs and args.checkpoint_epochs > 0:
        callbacks.append(
            ModelCheckpoint(
                every_n_epochs=args.checkpoint_epochs,
                save_top_k=-1,
                save_last=True,
            )
        )
    if getattr(args, "utmos_enabled", False):
        callbacks.append(
            UtmosQualityCallback(
                every_n_epochs=int(args.utmos_every_n_epochs),
                num_samples=int(args.utmos_num_samples),
                output_dir=args.utmos_output_dir,
                python_bin=args.utmos_python,
                worker_path=args.utmos_worker,
                cuda_visible_devices=getattr(args, "utmos_cuda_visible_devices", None),
                noise_scale=float(args.utmos_noise_scale),
                length_scale=float(args.utmos_length_scale),
                noise_w=float(args.utmos_noise_w),
                sdp_ratio=float(args.utmos_sdp_ratio),
            )
        )
    if bool(getattr(args, "epoch_summary_log", True)):
        callbacks.append(EpochSummaryCallback())

    logger = TensorBoardLogger(save_dir=args.default_root_dir)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=bool(getattr(args, "enable_progress_bar", True)),
        benchmark=torch.backends.cudnn.benchmark,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


def add_trainer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="32")
    parser.add_argument("--max_epochs", type=int, default=10_000)
    parser.add_argument("--default_root_dir")
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--progress-log-every-n-steps", type=int, default=0)
    parser.add_argument(
        "--enable-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--epoch-summary-log",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--use-length-buckets", action="store_true")
    parser.add_argument("--bucket-boundaries", nargs="*", type=int)


def add_quality_monitor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--utmos-enabled", action="store_true")
    parser.add_argument("--utmos-every-n-epochs", type=int, default=10)
    parser.add_argument("--utmos-num-samples", type=int, default=10)
    parser.add_argument("--utmos-output-dir")
    parser.add_argument(
        "--utmos-python",
        default="local/utmos_probe/.venv/bin/python",
    )
    parser.add_argument(
        "--utmos-worker",
        default="local/utmos_probe/utmos_stdin_worker.py",
    )
    parser.add_argument("--utmos-cuda-visible-devices")
    parser.add_argument("--utmos-noise-scale", type=float, default=0.667)
    parser.add_argument("--utmos-length-scale", type=float, default=1.0)
    parser.add_argument("--utmos-noise-w", type=float, default=0.8)
    parser.add_argument("--utmos-sdp-ratio", type=float, default=0.2)


def apply_quality_preset(args: dict[str, Any], quality: str) -> None:
    if quality == "x-low":
        args["hidden_channels"] = 96
        args["inter_channels"] = 96
        args["filter_channels"] = 384
    elif quality == "high":
        args["resblock"] = "1"
        args["resblock_kernel_sizes"] = (3, 7, 11)
        args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        args["upsample_rates"] = (8, 8, 2, 2)
        args["upsample_initial_channel"] = 512
        args["upsample_kernel_sizes"] = (16, 16, 4, 4)
    elif quality == "bert-87m":
        args["hidden_channels"] = 288
        args["inter_channels"] = 288
        args["filter_channels"] = 1152
        args["n_heads"] = 4
        args["n_layers"] = 8
        args["resblock"] = "1"
        args["resblock_kernel_sizes"] = (3, 7, 11)
        args["resblock_dilation_sizes"] = (
            (1, 3, 5),
            (1, 3, 5),
            (1, 3, 5),
        )
        args["upsample_rates"] = (8, 8, 2, 2)
        args["upsample_initial_channel"] = 640
        args["upsample_kernel_sizes"] = (16, 16, 4, 4)


def load_state_dict(model, saved_state_dict, allow_speaker_mismatch: bool = False):
    state_dict = model.state_dict()
    new_state_dict = {}
    speaker_keys = {"emb_g.weight"}

    missing_keys = []
    for key, value in state_dict.items():
        if key in saved_state_dict:
            new_state_dict[key] = saved_state_dict[key]
        elif allow_speaker_mismatch and key in speaker_keys:
            _LOGGER.info("Speaker embedding %s will be initialized", key)
            new_state_dict[key] = value
        else:
            missing_keys.append(key)

    extra_keys = set(saved_state_dict.keys()) - set(state_dict.keys())
    if allow_speaker_mismatch:
        extra_keys = {key for key in extra_keys if key not in speaker_keys}

    if missing_keys:
        raise ValueError(
            f"Checkpoint is missing {len(missing_keys)} required keys: {missing_keys}"
        )

    if extra_keys:
        raise ValueError(
            f"Checkpoint has {len(extra_keys)} unexpected keys: {list(extra_keys)}"
        )

    model.load_state_dict(new_state_dict)


def initialize_from_checkpoint(
    model: VitsModel, checkpoint_path: str | Path, expect_multispeaker: bool = True
) -> None:
    if expect_multispeaker:
        _LOGGER.info("Initializing model weights from %s", checkpoint_path)
    else:
        _LOGGER.info(
            "Converting single-speaker checkpoint for multi-speaker training: %s",
            checkpoint_path,
        )

    source_model = VitsModel.load_from_checkpoint(
        str(checkpoint_path), dataset=None, weights_only=False
    )

    source_use_bert = bool(getattr(source_model.hparams, "use_bert", False))
    target_use_bert = bool(getattr(model.hparams, "use_bert", False))
    if source_use_bert != target_use_bert:
        raise ValueError(
            f"BERT configuration mismatch: checkpoint has use_bert={source_use_bert}, "
            f"current training has use_bert={target_use_bert}."
        )

    source_map = _extract_speaker_map(source_model.hparams)
    target_map = _extract_speaker_map(model.hparams)
    generator_state = source_model.model_g.state_dict()

    remapped_embedding = _remap_speaker_embeddings(
        model, generator_state, source_map, target_map, str(checkpoint_path)
    )
    if remapped_embedding is not None:
        generator_state = dict(generator_state)
        generator_state["emb_g.weight"] = remapped_embedding

    load_state_dict(model.model_g, generator_state, allow_speaker_mismatch=True)
    load_state_dict(model.model_d, source_model.model_d.state_dict())


def initialize_compatible_generator_from_checkpoint(
    model: VitsModel,
    checkpoint_path: str | Path,
    include_prefixes: tuple[str, ...] = (),
    exclude_prefixes: tuple[str, ...] = ("dec.",),
) -> None:
    _LOGGER.info("Partially initializing whitelisted model_g weights from %s", checkpoint_path)
    if include_prefixes:
        _LOGGER.info("Partial init include prefixes: %s", ", ".join(include_prefixes))
    if exclude_prefixes:
        _LOGGER.info("Partial init exclude prefixes: %s", ", ".join(exclude_prefixes))

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_hparams = checkpoint.get("hyper_parameters") or {}
    source_use_bert = bool(checkpoint_hparams.get("use_bert", False))
    target_use_bert = bool(getattr(model.hparams, "use_bert", False))
    if source_use_bert != target_use_bert:
        _LOGGER.warning(
            "Partial checkpoint has use_bert=%s but target has use_bert=%s",
            source_use_bert,
            target_use_bert,
        )

    raw_state = checkpoint.get("state_dict")
    if not isinstance(raw_state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain a state_dict")

    source_state = {
        key.removeprefix("model_g."): value
        for key, value in raw_state.items()
        if key.startswith("model_g.")
    }
    remapped_embedding = _remap_speaker_embeddings(
        model,
        source_state,
        _extract_speaker_map_from_checkpoint_hparams(checkpoint_hparams),
        _extract_speaker_map(model.hparams),
        str(checkpoint_path),
    )
    if remapped_embedding is not None:
        source_state["emb_g.weight"] = remapped_embedding

    target_state = model.model_g.state_dict()
    target_uses_duration_blend = any(key.startswith("sdp.") for key in target_state)
    source_uses_duration_blend = any(key.startswith("sdp.") for key in source_state)
    if target_uses_duration_blend and not source_uses_duration_blend:
        remapped_sdp = 0
        for key, value in list(source_state.items()):
            if not key.startswith("dp."):
                continue
            sdp_key = "sdp." + key.removeprefix("dp.")
            target_value = target_state.get(sdp_key)
            if target_value is None:
                continue
            if tuple(value.shape) != tuple(target_value.shape):
                continue
            source_state[sdp_key] = value
            remapped_sdp += 1
        if remapped_sdp:
            _LOGGER.info(
                "Remapped %s old stochastic duration predictor tensor(s) from dp.* to sdp.*",
                remapped_sdp,
            )

    copied_state = {}
    copied = []
    skipped_not_included = []
    skipped_excluded = []
    skipped_missing = []
    skipped_shape = []
    resized = []

    for key, target_value in target_state.items():
        if include_prefixes and not any(key.startswith(prefix) for prefix in include_prefixes):
            copied_state[key] = target_value
            skipped_not_included.append(key)
            continue

        if any(key.startswith(prefix) for prefix in exclude_prefixes):
            copied_state[key] = target_value
            skipped_excluded.append(key)
            continue

        if (
            target_uses_duration_blend
            and not source_uses_duration_blend
            and key.startswith("dp.")
        ):
            copied_state[key] = target_value
            skipped_missing.append(key)
            continue

        source_value = source_state.get(key)
        if source_value is None:
            copied_state[key] = target_value
            skipped_missing.append(key)
            continue

        if tuple(source_value.shape) != tuple(target_value.shape):
            resized_value = _copy_overlapping_tensor(source_value, target_value)
            if resized_value is None:
                copied_state[key] = target_value
                skipped_shape.append(
                    (key, tuple(source_value.shape), tuple(target_value.shape))
                )
                continue

            copied_state[key] = resized_value
            resized.append((key, tuple(source_value.shape), tuple(target_value.shape)))
            continue

        copied_state[key] = source_value
        copied.append(key)

    model.model_g.load_state_dict(copied_state, strict=True)
    _LOGGER.info(
        "Partial init copied %s tensor(s), resized %s tensor(s); "
        "skipped not_included=%s excluded=%s missing=%s shape=%s",
        len(copied),
        len(resized),
        len(skipped_not_included),
        len(skipped_excluded),
        len(skipped_missing),
        len(skipped_shape),
    )
    if resized:
        preview = ", ".join(
            f"{key}: {src}->{dst}" for key, src, dst in resized[:12]
        )
        _LOGGER.info("Shape-resized tensors preview: %s", preview)
    if skipped_shape:
        preview = ", ".join(
            f"{key}: {src}->{dst}" for key, src, dst in skipped_shape[:12]
        )
        _LOGGER.info("Shape-skipped tensors preview: %s", preview)


def _copy_overlapping_tensor(
    source_value: torch.Tensor,
    target_value: torch.Tensor,
) -> torch.Tensor | None:
    if source_value.ndim != target_value.ndim:
        return None
    if source_value.ndim == 0:
        return None

    copied = target_value.detach().clone()
    source = source_value.detach().to(dtype=copied.dtype)
    slices = tuple(
        slice(0, min(int(src_dim), int(dst_dim)))
        for src_dim, dst_dim in zip(source.shape, copied.shape)
    )
    copied[slices] = source[slices]
    return copied


def _extract_speaker_map_from_checkpoint_hparams(hparams):
    raw_map = None
    if isinstance(hparams, dict):
        raw_map = hparams.get("speaker_id_map")
    return _normalize_speaker_map(raw_map)


def _extract_speaker_map(hparams):
    raw_map = getattr(hparams, "speaker_id_map", None)
    normalized = _normalize_speaker_map(raw_map)
    if normalized:
        return normalized

    dataset_dir = getattr(hparams, "dataset_dir", None)
    if dataset_dir:
        config_path = Path(dataset_dir) / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as config_file:
                config = json.load(config_file)
                speaker_map = config.get("speaker_id_map")
                if speaker_map:
                    return {
                        str(key): int(value)
                        for key, value in sorted(
                            speaker_map.items(), key=lambda item: item[1]
                        )
                    }

    return None


def _normalize_speaker_map(raw_map):
    if not raw_map:
        return None
    return {
        str(key): int(value)
        for key, value in sorted(raw_map.items(), key=lambda item: item[1])
    }


def _remap_speaker_embeddings(
    model: VitsModel, source_state, source_map, target_map, source_label: str
):
    if (
        "emb_g.weight" not in source_state
        or model.hparams.num_speakers <= 1
        or not hasattr(model.model_g, "emb_g")
    ):
        return None

    saved_weight = source_state["emb_g.weight"]
    new_weight = model.model_g.emb_g.weight.detach().clone()
    copied = 0

    if source_map and target_map:
        source_lookup = {speaker: idx for speaker, idx in source_map.items()}
        for speaker, target_idx in target_map.items():
            source_idx = source_lookup.get(speaker)
            if source_idx is None:
                continue
            if source_idx >= saved_weight.shape[0] or target_idx >= new_weight.shape[0]:
                continue
            new_weight[target_idx] = saved_weight[source_idx]
            copied += 1

        dropped = sorted(set(source_map) - set(target_map))
        added = sorted(set(target_map) - set(source_map))
        if dropped:
            _LOGGER.info("Dropping %s speaker(s): %s", len(dropped), ", ".join(dropped))
        if added:
            _LOGGER.info("Initializing new speaker(s): %s", ", ".join(added))
        _LOGGER.info("Copied %s/%s speaker embedding(s)", copied, len(target_map))
    else:
        count = min(saved_weight.shape[0], new_weight.shape[0])
        new_weight[:count] = saved_weight[:count]
        copied = count
        _LOGGER.warning("Speaker map missing; copied %s embedding(s) by index", count)

    if copied == 0:
        _LOGGER.warning("No speaker embeddings were copied from %s", source_label)

    return new_weight


class EpochSummaryCallback(Callback):
    def __init__(self) -> None:
        super().__init__()
        self._epoch_start_time = 0.0
        self._epoch_start_step = 0

    def on_train_epoch_start(self, trainer: Trainer, pl_module: VitsModel) -> None:
        self._epoch_start_time = time.perf_counter()
        self._epoch_start_step = int(trainer.global_step)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: VitsModel) -> None:
        if not getattr(trainer, "is_global_zero", True):
            return
        elapsed = max(0.0, time.perf_counter() - self._epoch_start_time)
        steps = max(0, int(trainer.global_step) - self._epoch_start_step)
        batches = self._format_batches(trainer.num_training_batches)
        _LOGGER.info(
            "Epoch %s/%s: global_step=%s batches=%s/%s elapsed=%.1fs%s",
            int(trainer.current_epoch) + 1,
            trainer.max_epochs,
            int(trainer.global_step),
            steps,
            batches,
            elapsed,
            self._format_metrics(trainer.callback_metrics),
        )

    @staticmethod
    def _format_batches(value: Any) -> str:
        try:
            return str(int(value))
        except (TypeError, ValueError, OverflowError):
            return str(value)

    @staticmethod
    def _format_metrics(metrics: dict[str, Any]) -> str:
        names = (
            "loss_gen_all",
            "loss_disc_all",
            "loss_dur",
            "loss_dur_sdp",
            "loss_dur_dp",
            "val_loss",
            "utmos/mean",
        )
        parts = []
        for name in names:
            value = metrics.get(name)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                value = value.detach().float().cpu().item()
            try:
                parts.append(f"{name}={float(value):.4f}")
            except (TypeError, ValueError):
                continue
        return (" " + " ".join(parts)) if parts else ""


class UtmosQualityCallback(Callback):
    def __init__(
        self,
        every_n_epochs: int,
        num_samples: int,
        output_dir: str | None,
        python_bin: str,
        worker_path: str,
        cuda_visible_devices: str | None,
        noise_scale: float,
        length_scale: float,
        noise_w: float,
        sdp_ratio: float,
    ) -> None:
        super().__init__()
        self.every_n_epochs = max(1, every_n_epochs)
        self.num_samples = max(1, num_samples)
        self.output_dir = output_dir
        self.python_bin = python_bin
        self.worker_path = worker_path
        self.cuda_visible_devices = cuda_visible_devices
        self.scales = [noise_scale, length_scale, noise_w, sdp_ratio]

    def on_train_epoch_end(self, trainer: Trainer, pl_module: VitsModel) -> None:
        epoch = int(trainer.current_epoch) + 1
        if epoch % self.every_n_epochs != 0 and epoch != int(trainer.max_epochs):
            return
        if not getattr(trainer, "is_global_zero", True):
            return

        sample_dataset = getattr(pl_module, "_test_dataset", None)
        if sample_dataset is None or len(sample_dataset) == 0:
            _LOGGER.warning("UTMOS monitor has no held-out samples to synthesize")
            return

        root = Path(
            self.output_dir
            or Path(trainer.default_root_dir) / "quality" / "utmos_samples"
        )
        sample_dir = root / f"epoch_{epoch:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()
        wav_paths: list[Path] = []
        rows: list[dict[str, Any]] = []
        try:
            with torch.inference_mode():
                for idx in range(min(self.num_samples, len(sample_dataset))):
                    utt = sample_dataset[idx]
                    wav_path = sample_dir / f"{idx:03d}.wav"
                    text = utt.phoneme_ids.unsqueeze(0).to(pl_module.device)
                    text_lengths = torch.LongTensor([len(utt.phoneme_ids)]).to(
                        pl_module.device
                    )
                    sid = (
                        utt.speaker_id.to(pl_module.device)
                        if utt.speaker_id is not None
                        else None
                    )
                    bert_input = self._build_bert_input(pl_module, utt)
                    audio = pl_module(
                        text,
                        text_lengths,
                        self.scales,
                        sid=sid,
                        bert_input=bert_input,
                    )
                    audio_np = audio.detach().float().cpu().numpy().reshape(-1)
                    audio_i16 = audio_float_to_int16(audio_np)
                    write_wav(str(wav_path), int(pl_module.hparams.sample_rate), audio_i16)
                    wav_paths.append(wav_path)
                    rows.append(
                        {
                            "index": idx,
                            "path": str(wav_path),
                            "speaker_id": int(utt.speaker_id.item())
                            if utt.speaker_id is not None
                            else "",
                            "text": utt.text or "",
                        }
                    )
        finally:
            if was_training:
                pl_module.train()

        scores = self._score_wavs(wav_paths)
        numeric_scores = []
        for row in rows:
            score = scores.get(row["path"])
            row["utmos"] = score if score is not None else ""
            if score is not None:
                numeric_scores.append(float(score))

        self._write_scores(sample_dir, rows)
        if not numeric_scores:
            _LOGGER.warning("UTMOS monitor produced no scores for epoch %s", epoch)
            return

        mean_score = float(np.mean(numeric_scores))
        min_score = float(np.min(numeric_scores))
        max_score = float(np.max(numeric_scores))
        _LOGGER.info(
            "UTMOS epoch %s: mean=%.4f min=%.4f max=%.4f n=%s",
            epoch,
            mean_score,
            min_score,
            max_score,
            len(numeric_scores),
        )
        if trainer.logger is not None:
            writer = trainer.logger.experiment
            writer.add_scalar("utmos/mean", mean_score, epoch)
            writer.add_scalar("utmos/min", min_score, epoch)
            writer.add_scalar("utmos/max", max_score, epoch)

    def _build_bert_input(self, pl_module: VitsModel, utt):
        features = getattr(utt, "bert_features", None)
        if features is not None:
            return {"features": features.unsqueeze(0).to(pl_module.device)}

        text = getattr(utt, "text", None)
        if not getattr(pl_module.hparams, "use_bert", False) or not text:
            return None
        if getattr(pl_module.hparams, "bert_features_precomputed", False):
            raise ValueError(
                "UTMOS sample is missing precomputed BERT features; regenerate dataset.parquet with bert_path entries"
            )

        from .semantic import SemanticTokenizer, build_bert_input

        if pl_module._semantic_tokenizer is None:
            model_name = getattr(pl_module.hparams, "bert_model_name", None)
            pl_module._semantic_tokenizer = SemanticTokenizer(model_name=model_name)

        phoneme_ids = getattr(utt, "phoneme_ids", None)
        phoneme_length = int(phoneme_ids.size(0)) if phoneme_ids is not None else None
        bert_dict = build_bert_input(
            [text],
            pl_module._semantic_tokenizer,
            phoneme_lengths=[phoneme_length] if phoneme_length is not None else None,
            word_spans=[getattr(utt, "word_spans", None)],
        )
        if bert_dict is None:
            return None
        return {
            key: value.to(pl_module.device)
            for key, value in bert_dict.items()
        }

    def _score_wavs(self, wav_paths: list[Path]) -> dict[str, float]:
        python_bin = Path(self.python_bin)
        worker_path = Path(self.worker_path)
        if not python_bin.exists() or not worker_path.exists():
            _LOGGER.warning(
                "UTMOS worker unavailable: python=%s worker=%s",
                python_bin,
                worker_path,
            )
            return {}

        env = os.environ.copy()
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
        proc = subprocess.Popen(
            [str(python_bin), str(worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        assert proc.stdin is not None
        assert proc.stdout is not None

        scores: dict[str, float] = {}
        try:
            ready_line = proc.stdout.readline()
            if not ready_line:
                stderr = proc.stderr.read() if proc.stderr else ""
                _LOGGER.warning(
                    "UTMOS worker exited before ready%s",
                    f": {stderr.strip()}" if stderr.strip() else "",
                )
                return {}
            _LOGGER.debug("UTMOS worker: %s", ready_line.strip())
            for idx, wav_path in enumerate(wav_paths):
                request = {"id": idx, "path": str(wav_path)}
                try:
                    proc.stdin.write(json.dumps(request) + "\n")
                    proc.stdin.flush()
                except BrokenPipeError:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    _LOGGER.warning(
                        "UTMOS worker pipe closed while scoring %s%s",
                        wav_path,
                        f": {stderr.strip()}" if stderr.strip() else "",
                    )
                    break
                response_line = proc.stdout.readline()
                if not response_line:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    _LOGGER.warning(
                        "UTMOS worker stopped before response for %s%s",
                        wav_path,
                        f": {stderr.strip()}" if stderr.strip() else "",
                    )
                    break
                response = json.loads(response_line)
                if "mos_score" in response:
                    scores[str(wav_path)] = float(response["mos_score"])
                else:
                    _LOGGER.warning("UTMOS scoring error: %s", response)
        finally:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            if proc.poll() is None:
                proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
            stderr = proc.stderr.read() if proc.stderr and not proc.stderr.closed else ""
            if stderr.strip():
                _LOGGER.debug("UTMOS worker stderr: %s", stderr.strip())

        return scores

    @staticmethod
    def _write_scores(sample_dir: Path, rows: list[dict[str, Any]]) -> None:
        with (sample_dir / "utmos_scores.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["index", "path", "speaker_id", "text", "utmos"]
            )
            writer.writeheader()
            writer.writerows(rows)
        with (sample_dir / "utmos_scores.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
