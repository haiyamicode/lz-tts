"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from src.starling import utils
from src.starling.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
            data_statistics = {
                "mel_mean": 0.0,
                "mel_std": 1.0,
            }

        self.register_buffer("mel_mean", torch.tensor(data_statistics["mel_mean"]))
        self.register_buffer("mel_std", torch.tensor(data_statistics["mel_std"]))

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        semantic_features = batch.get("bert_features")

        outputs = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            spks=spks,
            out_size=self.out_size,
            durations=batch["durations"],
            semantic_features=semantic_features,
            prompt_mel=batch.get("prompt_mel"),
            prompt_mel_lengths=batch.get("prompt_mel_lengths"),
            prompt_embedding=batch.get("prompt_embedding"),
        )
        dur_loss, prior_loss, diff_loss, *extra = outputs
        metrics = {}
        if extra and isinstance(extra[-1], dict):
            metrics.update(extra[-1])
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }, metrics

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict, metric_dict = self.get_losses(batch)
        batch_size = int(batch["x"].shape[0])

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in metric_dict.items():
            self.log(
                f"sub_loss/train_{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "train_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "diff_loss",
            loss_dict["diff_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "dur_loss",
            loss_dict["dur_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict, metric_dict = self.get_losses(batch)
        batch_size = int(batch["x"].shape[0])
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        for name, value in metric_dict.items():
            self.log(
                f"sub_loss/val_{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=batch_size,
            )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            "val_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            n_log_samples = min(2, one_batch["x"].shape[0])
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(n_log_samples):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(n_log_samples):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                semantic_features = None
                if one_batch.get("bert_features") is not None:
                    semantic_features = one_batch["bert_features"][i].unsqueeze(0).to(self.device)
                    semantic_features = semantic_features[:, :, : x_lengths.item()]
                prompt_mel = None
                prompt_mel_lengths = None
                if one_batch.get("prompt_mel") is not None:
                    prompt_mel_lengths = one_batch["prompt_mel_lengths"][i].unsqueeze(0).to(self.device)
                    prompt_mel = one_batch["prompt_mel"][i].unsqueeze(0).to(self.device)
                    prompt_mel = prompt_mel[:, :, : prompt_mel_lengths.item()]
                prompt_embedding = None
                if one_batch.get("prompt_embedding") is not None:
                    prompt_embedding = one_batch["prompt_embedding"][i].unsqueeze(0).to(self.device)
                output = self.synthesise(
                    x[:, : x_lengths.item()],
                    x_lengths,
                    n_timesteps=10,
                    spks=spks,
                    semantic_features=semantic_features,
                    prompt_mel=prompt_mel,
                    prompt_mel_lengths=prompt_mel_lengths,
                    prompt_embedding=prompt_embedding,
                )
                y_enc, y_dec = output["encoder_outputs"], output["decoder_outputs"]
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )

    def on_before_optimizer_step(self, optimizer):
        if getattr(self.hparams, "log_grad_norm", False):
            self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
