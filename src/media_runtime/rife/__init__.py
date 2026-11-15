import os
from pathlib import Path

import torch

from .train_log import Model


# data/rife-v4.26 of the lz-tts repo (src/media_runtime/rife -> repo root); RIFE_MODEL_DIR overrides.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[3] / "data" / "rife-v4.26"


def _default_model_dir() -> Path:
    return Path(os.getenv("RIFE_MODEL_DIR") or _DEFAULT_MODEL_DIR)


def load_model(model_dir: str | os.PathLike[str] | None = None, **kwargs) -> Model:
    path = Path(model_dir) if model_dir else _default_model_dir()
    pkl_path = path / "flownet.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"RIFE model weights not found at {pkl_path}. "
            f"Download from https://github.com/hzwer/Practical-RIFE and place in {path}."
        )

    model = Model(**kwargs)

    if torch.cuda.is_available():
        state = torch.load(str(pkl_path))
    else:
        state = torch.load(str(pkl_path), map_location="cpu")

    clean = {k.replace("module.", ""): v for k, v in state.items()}
    model.flownet.load_state_dict(clean, strict=False)
    model.eval()
    return model


__all__ = ["Model", "load_model"]
