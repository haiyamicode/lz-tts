from __future__ import annotations

import csv
import json
import os
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from lightning.pytorch import Callback, Trainer
from lightning.pytorch.core import LightningModule

from src.starling.data.text_mel_datamodule import TextMelBatchCollate


class MatchaUtmosQualityCallback(Callback):
    def __init__(
        self,
        enabled: bool = False,
        every_n_epochs: int = 5,
        train_num_samples: int = 5,
        val_num_samples: int = 5,
        output_dir: str | None = None,
        python_bin: str = "/mnt/data/lz-tts/local/utmos_probe/.venv/bin/python",
        worker_path: str = "/mnt/data/lz-tts/local/utmos_probe/utmos_stdin_worker.py",
        cuda_visible_devices: str | None = None,
        vocoder: str = "vocos24k",
        sample_rate: int = 24000,
        n_timesteps: int = 32,
        temperature: float = 1.0,
        length_scale: float = 1.0,
        noise_scale_w: float | None = None,
        sdp_ratio: float | None = None,
        score_final_epoch: bool = True,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.every_n_epochs = max(1, int(every_n_epochs))
        self.train_num_samples = max(0, int(train_num_samples))
        self.val_num_samples = max(0, int(val_num_samples))
        self.output_dir = output_dir
        self.python_bin = python_bin
        self.worker_path = worker_path
        self.cuda_visible_devices = cuda_visible_devices
        self.vocoder_name = vocoder
        self.sample_rate = int(sample_rate)
        self.n_timesteps = int(n_timesteps)
        self.temperature = float(temperature)
        self.length_scale = float(length_scale)
        self.noise_scale_w = noise_scale_w
        self.sdp_ratio = sdp_ratio
        self.score_final_epoch = bool(score_final_epoch)
        self._vocoder = None
        self._sample_indices: dict[str, list[int]] = {}

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled:
            return
        datamodule = trainer.datamodule
        if datamodule is None:
            raise RuntimeError("Matcha UTMOS callback requires a datamodule")
        if not hasattr(datamodule, "trainset") or not hasattr(datamodule, "validset"):
            datamodule.setup("fit")
        self._sample_indices = {
            "train": list(range(min(self.train_num_samples, len(datamodule.trainset)))),
            "val": list(range(min(self.val_num_samples, len(datamodule.validset)))),
        }

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.enabled or trainer.sanity_checking or not trainer.is_global_zero:
            return
        epoch = int(trainer.current_epoch) + 1
        should_score = epoch % self.every_n_epochs == 0
        if self.score_final_epoch:
            should_score = should_score or epoch == int(trainer.max_epochs)
        if not should_score:
            return

        root = Path(self.output_dir or Path(trainer.default_root_dir) / "quality" / "utmos_samples")
        epoch_dir = root / f"epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        was_training = pl_module.training
        pl_module.eval()
        try:
            with torch.inference_mode():
                train_rows = self._run_split(trainer, pl_module, "train", epoch_dir)
                val_rows = self._run_split(trainer, pl_module, "val", epoch_dir)
        finally:
            if was_training:
                pl_module.train()

        self._log_summary(trainer, epoch, train_rows, val_rows)

    def _run_split(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        split: str,
        epoch_dir: Path,
    ) -> list[dict[str, Any]]:
        datamodule = trainer.datamodule
        dataset = datamodule.trainset if split == "train" else datamodule.validset
        indices = self._sample_indices.get(split, [])
        split_dir = epoch_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        wav_paths: list[Path] = []
        for sample_idx in indices:
            item = dataset[sample_idx]
            batch = TextMelBatchCollate(datamodule.hparams.n_spks)([item])
            wav_path = split_dir / f"{sample_idx:03d}.wav"
            self._synthesise_batch(pl_module, batch, wav_path)
            wav_paths.append(wav_path)
            rows.append(
                {
                    "split": split,
                    "index": sample_idx,
                    "path": str(wav_path),
                    "text": item.get("x_text", ""),
                    "filepath": item.get("filepath", ""),
                    "utmos": "",
                }
            )

        scores = self._score_wavs(wav_paths)
        for row in rows:
            score = scores.get(row["path"])
            row["utmos"] = score if score is not None else ""
        self._write_rows(split_dir, rows)
        return rows

    def _synthesise_batch(self, pl_module: LightningModule, batch: dict[str, Any], wav_path: Path) -> None:
        device = pl_module.device
        x_lengths = batch["x_lengths"].to(device)
        x_length = int(x_lengths[0].item())
        x = batch["x"][:, :x_length].to(device)
        spks = batch["spks"].to(device) if batch["spks"] is not None else None

        semantic_features = batch.get("bert_features")
        if semantic_features is None:
            raise ValueError("Matcha UTMOS synthesis requires precomputed aligned BERT features")
        semantic_features = semantic_features[:, :, :x_length].to(device)

        output = pl_module.synthesise(
            x,
            x_lengths,
            n_timesteps=self.n_timesteps,
            temperature=self.temperature,
            spks=spks,
            length_scale=self.length_scale,
            semantic_features=semantic_features,
            noise_scale_w=self.noise_scale_w,
            sdp_ratio=self.sdp_ratio,
        )
        audio = self._decode_mel(output["mel"]).squeeze().detach().float().cpu().numpy()
        sf.write(wav_path, np.clip(audio, -1.0, 1.0), self.sample_rate, subtype="PCM_24")

    def _decode_mel(self, mel: torch.Tensor) -> torch.Tensor:
        vocoder = self._load_vocoder(mel.device)
        if self.vocoder_name == "vocos24k":
            return vocoder.decode(mel).clamp(-1, 1)
        raise ValueError(f"Unsupported Matcha MOS vocoder: {self.vocoder_name}")

    def _load_vocoder(self, device):
        if self._vocoder is None:
            if self.vocoder_name != "vocos24k":
                raise ValueError(f"Unsupported Matcha MOS vocoder: {self.vocoder_name}")
            from vocos import Vocos  # pylint: disable=import-outside-toplevel

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, module="vocos.pretrained")
                self._vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval().to(device)
        return self._vocoder

    def _score_wavs(self, wav_paths: list[Path]) -> dict[str, float]:
        python_bin = Path(self.python_bin)
        worker_path = Path(self.worker_path)
        if not python_bin.exists() or not worker_path.exists():
            raise RuntimeError(
                f"UTMOS scoring is enabled, but python_bin={python_bin} or worker_path={worker_path} does not exist"
            )

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
                raise RuntimeError(f"UTMOS worker exited before ready: {stderr.strip()}")
            for idx, wav_path in enumerate(wav_paths):
                proc.stdin.write(json.dumps({"id": idx, "path": str(wav_path)}) + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
                if not line:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    raise RuntimeError(f"UTMOS worker stopped before scoring {wav_path}: {stderr.strip()}")
                response = json.loads(line)
                if "mos_score" not in response:
                    raise RuntimeError(f"UTMOS scoring failed for {wav_path}: {response}")
                scores[str(wav_path)] = float(response["mos_score"])
        finally:
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
        return scores

    @staticmethod
    def _write_rows(split_dir: Path, rows: list[dict[str, Any]]) -> None:
        with (split_dir / "utmos_scores.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "index", "path", "text", "filepath", "utmos"])
            writer.writeheader()
            writer.writerows(rows)
        with (split_dir / "utmos_scores.json").open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _mean_score(rows: list[dict[str, Any]]) -> float | None:
        values = [float(row["utmos"]) for row in rows if row.get("utmos") != ""]
        return float(np.mean(values)) if values else None

    def _log_summary(
        self,
        trainer: Trainer,
        epoch: int,
        train_rows: list[dict[str, Any]],
        val_rows: list[dict[str, Any]],
    ) -> None:
        train_mean = self._mean_score(train_rows)
        val_mean = self._mean_score(val_rows)
        if trainer.logger is not None:
            writer = trainer.logger.experiment
            if train_mean is not None:
                writer.add_scalar("mos/train_mean", train_mean, epoch)
            if val_mean is not None:
                writer.add_scalar("mos/val_mean", val_mean, epoch)
        if train_mean is not None:
            trainer.print(f"[mos] epoch={epoch:04d} split=train mean={train_mean:.4f} n={len(train_rows)}")
        if val_mean is not None:
            trainer.print(f"[mos] epoch={epoch:04d} split=val mean={val_mean:.4f} n={len(val_rows)}")
