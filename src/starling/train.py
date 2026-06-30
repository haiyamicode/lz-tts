import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.starling import utils

log = utils.get_pylogger(__name__)


def _latest_output_checkpoint(output_dir: str) -> Optional[Path]:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    last_checkpoint = checkpoint_dir / "last.ckpt"
    if last_checkpoint.is_file():
        return last_checkpoint

    checkpoints = [path for path in checkpoint_dir.glob("*.ckpt") if path.is_file()]
    if not checkpoints:
        return None

    return max(checkpoints, key=lambda path: path.stat().st_mtime)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")  # pylint: disable=protected-access
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # pylint: disable=protected-access
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    resume_from_output = bool(cfg.get("resume_from_output_checkpoint", True))
    resume_ckpt_path = _latest_output_checkpoint(cfg.paths.output_dir) if cfg.get("train") and resume_from_output else None
    if resume_ckpt_path is not None:
        cfg.ckpt_path = str(resume_ckpt_path)
        log.info(f"Resuming training from output checkpoint: {cfg.ckpt_path}")
    elif cfg.get("ckpt_path"):
        log.info(f"Resuming training from configured checkpoint: {cfg.ckpt_path}")
    elif cfg.get("init_from_checkpoint"):
        import torch

        init_ckpt = torch.load(cfg.init_from_checkpoint, map_location="cpu", weights_only=False)
        strict = bool(cfg.get("init_strict", True))
        incompatible = model.load_state_dict(init_ckpt["state_dict"], strict=strict)
        if not strict:
            missing = list(getattr(incompatible, "missing_keys", []))
            unexpected = list(getattr(incompatible, "unexpected_keys", []))
            if missing:
                log.info("Initializer checkpoint missing %d model keys; newly initialized modules will train from scratch", len(missing))
                log.debug("Missing keys: %s", missing)
            if unexpected:
                log.info("Initializer checkpoint has %d unused model keys", len(unexpected))
                log.debug("Unexpected keys: %s", unexpected)
        log.info(f"Loaded initial weights from {cfg.init_from_checkpoint}")
        cfg.ckpt_path = None

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")  # pylint: disable=protected-access
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"), weights_only=False)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../local/configs/starling", config_name="train_andrew_edge_multilingual.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    utils.extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
