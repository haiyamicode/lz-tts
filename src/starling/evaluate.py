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

_CHECKPOINT_ENV = "STARLING_EVAL_CHECKPOINT"


def _pop_checkpoint_arg(argv: list[str]) -> list[str]:
    """Remove --checkpoint before Hydra parses argv.

    Lightning checkpoint filenames commonly contain "=" (for example
    step=00433000.ckpt), which is awkward as a Hydra override value. Keep the
    checkpoint path as a normal CLI option and leave all remaining args for
    Hydra.
    """
    cleaned = [argv[0]]
    index = 1
    while index < len(argv):
        arg = argv[index]
        if arg == "--checkpoint":
            if index + 1 >= len(argv):
                raise ValueError("--checkpoint requires a path")
            os.environ[_CHECKPOINT_ENV] = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--checkpoint="):
            os.environ[_CHECKPOINT_ENV] = arg.split("=", 1)[1]
            index += 1
            continue
        cleaned.append(arg)
        index += 1
    return cleaned


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
    cli_checkpoint = os.environ.get(_CHECKPOINT_ENV)
    if cli_checkpoint:
        return Path(cli_checkpoint)

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
def _hydra_main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)
    evaluate(cfg)
    return None


def main() -> Optional[float]:
    sys.argv = _pop_checkpoint_arg(sys.argv)
    return _hydra_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
