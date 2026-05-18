"""Piper/VITS training entry point for this repository.

This is a small Lightning 2.x compatible equivalent of Piper's
``piper_train.__main__``. It intentionally keeps semantic/BERT conditioning off
unless ``--use-bert`` is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)


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
        choices=("x-low", "medium", "high"),
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
    parser.add_argument("--seed", type=int, default=1234)

    add_trainer_args(parser)
    VitsModel.add_model_specific_args(parser)
    args = parser.parse_args()
    if isinstance(args.devices, str) and args.devices.isdigit():
        args.devices = int(args.devices)

    if args.resume_from_single_speaker_checkpoint and args.init_from_checkpoint:
        parser.error(
            "--resume_from_single_speaker_checkpoint cannot be combined with "
            "--init_from_checkpoint"
        )

    dataset_dir = Path(args.dataset_dir).resolve()
    if args.default_root_dir is None:
        args.default_root_dir = str(dataset_dir)

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)

    config_path = dataset_dir / "config.json"
    dataset_path = dataset_dir / "dataset.jsonl"

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
            parser.error(
                "--resume_from_single_speaker_checkpoint is only for multi-speaker models"
            )
        initialize_from_checkpoint(
            model,
            args.resume_from_single_speaker_checkpoint,
            expect_multispeaker=False,
        )

    if args.init_from_checkpoint:
        initialize_from_checkpoint(model, args.init_from_checkpoint)

    callbacks = []
    if args.checkpoint_epochs and args.checkpoint_epochs > 0:
        callbacks.append(ModelCheckpoint(every_n_epochs=args.checkpoint_epochs))

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
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)


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


def _extract_speaker_map(hparams):
    raw_map = getattr(hparams, "speaker_id_map", None)
    if raw_map:
        return {
            str(key): int(value)
            for key, value in sorted(raw_map.items(), key=lambda item: item[1])
        }

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


if __name__ == "__main__":
    main()
