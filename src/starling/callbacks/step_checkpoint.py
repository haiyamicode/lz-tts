from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule


class StepCheckpointCallback(Callback):
    """Save step checkpoints with recent retention plus sparse milestones."""

    _STEP_RE = re.compile(r"^step=(?P<step>\d+)(?:-v\d+)?\.ckpt$")

    def __init__(
        self,
        dirpath: str,
        every_n_train_steps: int = 1000,
        keep_last: int = 3,
        retain_every_n_train_steps: int = 10000,
        save_last: bool = True,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.every_n_train_steps = max(1, int(every_n_train_steps))
        self.keep_last = max(1, int(keep_last))
        self.retain_every_n_train_steps = max(0, int(retain_every_n_train_steps))
        self.save_last = bool(save_last)
        self._last_saved_step = -1
        self._train_step = 0

    def state_dict(self) -> dict[str, int]:
        return {
            "last_saved_step": int(self._last_saved_step),
            "train_step": int(self._train_step),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._last_saved_step = int(state_dict.get("last_saved_step", -1))
        self._train_step = int(state_dict.get("train_step", 0))

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        self._train_step += 1
        step = int(self._train_step)
        if step <= 0 or step == self._last_saved_step:
            return
        if step % self.every_n_train_steps != 0:
            return

        self._save_checkpoint(trainer, step)

    def _save_checkpoint(self, trainer: Trainer, step: int) -> None:
        checkpoint_dir = Path(self.dirpath)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"step={step:08d}.ckpt"

        trainer.save_checkpoint(str(checkpoint_path))
        self._last_saved_step = step
        if self.save_last:
            self._update_last_checkpoint(checkpoint_path, checkpoint_dir / "last.ckpt")
        self._prune_old_checkpoints(checkpoint_dir)
        trainer.print(
            f"[checkpoint] saved {checkpoint_path} "
            f"(keep_last={self.keep_last}, retain_every={self.retain_every_n_train_steps or 'disabled'})"
        )

    @staticmethod
    def _update_last_checkpoint(checkpoint_path: Path, last_path: Path) -> None:
        if last_path.exists() or last_path.is_symlink():
            last_path.unlink()
        try:
            os.link(checkpoint_path, last_path)
        except OSError:
            shutil.copy2(checkpoint_path, last_path)

    def _prune_old_checkpoints(self, checkpoint_dir: Path) -> None:
        checkpoints: list[tuple[int, Path]] = []
        for path in checkpoint_dir.glob("step=*.ckpt"):
            match = self._STEP_RE.match(path.name)
            if match is None:
                continue
            checkpoints.append((int(match.group("step")), path))

        checkpoints.sort(key=lambda item: item[0], reverse=True)
        recent_paths = {path for _, path in checkpoints[: self.keep_last]}

        for step, path in checkpoints[self.keep_last :]:
            if path in recent_paths or self._is_retained_step(step):
                continue
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _is_retained_step(self, step: int) -> bool:
        return (
            self.retain_every_n_train_steps > 0
            and step > 0
            and step % self.retain_every_n_train_steps == 0
        )
