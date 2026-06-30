import os
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import lightning as L
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PROJECT_ROOT", str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.starling import utils
from src.starling.eval import run_configured_eval

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


def _resolve_checkpoint(cfg: DictConfig) -> Path:
    eval_cfg = cfg.get("eval") or {}
    configured = eval_cfg.get("checkpoint")
    if configured:
        return Path(str(configured))

    if cfg.get("resume_from_output_checkpoint", True):
        latest = _latest_output_checkpoint(cfg.paths.output_dir)
        if latest is not None:
            return latest

    if cfg.get("ckpt_path"):
        return Path(str(cfg.ckpt_path))

    raise FileNotFoundError("Starling eval requires eval.checkpoint, ckpt_path, or an output checkpoint")


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    eval_cfg = cfg.get("eval")
    if not eval_cfg or not bool(eval_cfg.get("enabled", False)):
        raise ValueError("Set eval.enabled=true to run Starling eval")

    checkpoint_path = _resolve_checkpoint(cfg)

    log.info(f"Instantiating model <{cfg.model._target_}>")  # pylint: disable=protected-access
    model = hydra.utils.instantiate(cfg.model)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    log.info(f"Loaded Starling checkpoint for eval: {checkpoint_path}")

    metrics = run_configured_eval(cfg, model, checkpoint_path=checkpoint_path)
    return metrics, {"cfg": cfg, "model": model, "checkpoint_path": checkpoint_path}


@hydra.main(version_base="1.3", config_path="../../local/configs/starling", config_name="train_voice_clone_multilingual.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    evaluate(cfg)
    return None


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
