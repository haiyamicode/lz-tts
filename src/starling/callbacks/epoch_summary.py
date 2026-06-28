from __future__ import annotations

from datetime import timedelta
from time import monotonic

import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule


class EpochSummaryCallback(Callback):
    def __init__(self, precision: int = 4):
        super().__init__()
        self.precision = int(precision)
        self._epoch_start: float | None = None

    def on_train_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._epoch_start = monotonic()

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if trainer.sanity_checking or not trainer.is_global_zero:
            return

        metrics = trainer.callback_metrics
        epoch = int(trainer.current_epoch) + 1
        fields = [
            f"epoch={epoch:04d}",
            f"elapsed={self._elapsed()}",
        ]
        for display_name, metric_name in (
            ("train_loss", "train_loss"),
            ("val_loss", "val_loss"),
            ("diff_loss", "diff_loss"),
            ("dur_loss", "dur_loss"),
        ):
            value = metrics.get(metric_name)
            if value is not None:
                fields.append(f"{display_name}={self._format_metric(value)}")

        trainer.print("[epoch] " + " ".join(fields))

    def _elapsed(self) -> str:
        if self._epoch_start is None:
            return "0:00:00"
        seconds = int(monotonic() - self._epoch_start)
        return str(timedelta(seconds=seconds))

    def _format_metric(self, value) -> str:
        if isinstance(value, torch.Tensor):
            value = value.detach().float().cpu().item()
        return f"{float(value):.{self.precision}f}"
