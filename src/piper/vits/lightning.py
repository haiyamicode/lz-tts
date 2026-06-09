import argparse
import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
from torch import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from .commons import slice_segments
from .mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from .models import MultiPeriodDiscriminator, SynthesizerTrn

# Training-only imports (optional for inference)
try:
    from .dataset import Batch, LengthBucketBatchSampler, PiperDataset, UtteranceCollate
    from .losses import discriminator_loss, feature_loss, generator_loss, kl_loss
except ImportError:
    Batch = None
    PiperDataset = None
    LengthBucketBatchSampler = None
    UtteranceCollate = None
    discriminator_loss = None
    feature_loss = None
    generator_loss = None
    kl_loss = None

_LOGGER = logging.getLogger("vits.lightning")
_DEBUG_SEMANTIC = bool(int(os.environ.get("PIPER_SEMANTIC_DEBUG", "0")))


class VitsModel(pl.LightningModule):
    def __init__(
        self,
        num_symbols: int,
        num_speakers: int,
        # audio
        resblock="2",
        resblock_kernel_sizes=(3, 5, 7),
        resblock_dilation_sizes=(
            (1, 2),
            (2, 6),
            (3, 12),
        ),
        upsample_rates=(8, 8, 4),
        upsample_initial_channel=256,
        upsample_kernel_sizes=(16, 16, 8),
        # mel
        filter_length: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        mel_channels: int = 80,
        sample_rate: int = 22050,
        sample_bytes: int = 2,
        channels: int = 1,
        mel_fmin: float = 0.0,
        mel_fmax: Optional[float] = None,
        # model
        inter_channels: int = 192,
        hidden_channels: int = 192,
        filter_channels: int = 768,
        n_heads: int = 2,
        n_layers: int = 6,
        kernel_size: int = 3,
        p_dropout: float = 0.1,
        n_layers_q: int = 3,
        use_spectral_norm: bool = False,
        gin_channels: int = 0,
        use_sdp: bool = True,
        use_duration_blend: bool = False,
        duration_blend_sdp_ratio: float = 0.2,
        duration_sdp_loss_weight: float = 1.0,
        duration_dp_loss_weight: float = 1.0,
        segment_size: int = 8192,
        # training
        dataset: Optional[List[Union[str, Path]]] = None,
        learning_rate: float = 2e-4,
        betas: Tuple[float, float] = (0.8, 0.99),
        eps: float = 1e-9,
        batch_size: int = 1,
        lr_decay: float = 0.999875,
        init_lr_ratio: float = 1.0,
        warmup_epochs: int = 0,
        c_mel: int = 45,
        c_kl: float = 1.0,
        grad_clip: Optional[float] = None,
        num_workers: int = 1,
        seed: int = 1234,
        num_test_examples: int = 5,
        validation_split: float = 0.1,
        max_phoneme_ids: Optional[int] = None,
        speaker_id_map: Optional[dict] = None,
        use_length_buckets: bool = False,
        bucket_boundaries: Optional[List[int]] = None,
        # semantic / BERT options
        use_bert: bool = False,
        bert_model_name: Optional[str] = None,
        bert_hidden_size: int = 768,
        freeze_bert: bool = True,
        bert_features_precomputed: bool = False,
        bert_fusion_weight: float = 0.5,
        semantic_fusion_mode: Optional[str] = None,
        use_spk_conditioned_encoder: bool = False,
        speaker_condition_layer: int = 2,
        # inference-only (no discriminator, no datasets)
        inference_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        if speaker_id_map is None:
            speaker_id_map = {}
        else:
            speaker_id_map = {
                str(key): int(value) for key, value in speaker_id_map.items()
            }
        self.hparams.speaker_id_map = speaker_id_map

        if (self.hparams.num_speakers > 1) and (self.hparams.gin_channels <= 0):
            # Default gin_channels for multi-speaker model
            self.hparams.gin_channels = 512

        # Determine semantic encoder model name (if enabled)
        bert_model = bert_model_name or os.environ.get(
            "PIPER_SEMANTIC_MODEL_NAME", "distilbert-base-multilingual-cased"
        )

        # Set up models
        self.model_g = SynthesizerTrn(
            n_vocab=self.hparams.num_symbols,
            spec_channels=self.hparams.filter_length // 2 + 1,
            segment_size=self.hparams.segment_size // self.hparams.hop_length,
            inter_channels=self.hparams.inter_channels,
            hidden_channels=self.hparams.hidden_channels,
            filter_channels=self.hparams.filter_channels,
            n_heads=self.hparams.n_heads,
            n_layers=self.hparams.n_layers,
            kernel_size=self.hparams.kernel_size,
            p_dropout=self.hparams.p_dropout,
            resblock=self.hparams.resblock,
            resblock_kernel_sizes=self.hparams.resblock_kernel_sizes,
            resblock_dilation_sizes=self.hparams.resblock_dilation_sizes,
            upsample_rates=self.hparams.upsample_rates,
            upsample_initial_channel=self.hparams.upsample_initial_channel,
            upsample_kernel_sizes=self.hparams.upsample_kernel_sizes,
            n_speakers=self.hparams.num_speakers,
            gin_channels=self.hparams.gin_channels,
            use_sdp=self.hparams.use_sdp,
            use_duration_blend=self.hparams.use_duration_blend,
            duration_blend_sdp_ratio=self.hparams.duration_blend_sdp_ratio,
            use_bert=use_bert,
            bert_model=bert_model,
            bert_hidden_size=bert_hidden_size,
            freeze_bert=freeze_bert,
            bert_features_precomputed=bert_features_precomputed,
            bert_fusion_weight=bert_fusion_weight,
            semantic_fusion_mode=semantic_fusion_mode,
            use_spk_conditioned_encoder=use_spk_conditioned_encoder,
            speaker_condition_layer=speaker_condition_layer,
        )
        if inference_only:
            self.model_d = None
        else:
            self.model_d = MultiPeriodDiscriminator(
                use_spectral_norm=self.hparams.use_spectral_norm
            )

        # Dataset splits
        self._train_dataset: Optional[Dataset] = None
        self._val_dataset: Optional[Dataset] = None
        self._test_dataset: Optional[Dataset] = None
        if not inference_only:
            self._load_datasets(validation_split, num_test_examples, max_phoneme_ids)

        # State kept between training optimizers
        self._y = None
        self._y_hat = None
        # Lazily-initialized semantic tokenizer (only when use_bert=True)
        self._semantic_tokenizer = None

        # Enable manual optimization for multiple optimizers (Lightning 2.0+)
        self.automatic_optimization = False

        if use_bert:
            _LOGGER.info(
                "Semantic encoder enabled: model=%s alignment=word2ph",
                bert_model,
            )
        if use_spk_conditioned_encoder and self.hparams.num_speakers > 1:
            _LOGGER.info(
                "Speaker-conditioned text encoder enabled: gin_channels=%s layer=%s",
                self.hparams.gin_channels,
                speaker_condition_layer,
            )

    def _load_datasets(
        self,
        validation_split: float,
        num_test_examples: int,
        max_phoneme_ids: Optional[int] = None,
    ):
        if self.hparams.dataset is None:
            _LOGGER.debug("No dataset to load")
            return

        full_dataset = PiperDataset(
            self.hparams.dataset,
            max_phoneme_ids=max_phoneme_ids,
            hop_length=self.hparams.hop_length,
        )
        dataset_size = len(full_dataset)
        if dataset_size <= 0:
            raise ValueError("No utterances available after dataset filtering")

        num_test_examples = max(0, min(int(num_test_examples), max(0, dataset_size - 1)))

        valid_set_size = int(dataset_size * validation_split)
        if validation_split > 0 and valid_set_size == 0:
            valid_set_size = 1

        max_valid_size = max(0, dataset_size - num_test_examples - 1)
        valid_set_size = min(valid_set_size, max_valid_size)

        train_set_size = dataset_size - valid_set_size - num_test_examples
        if train_set_size <= 0:
            raise ValueError(
                "Dataset split leaves no training samples: "
                f"total={dataset_size}, test={num_test_examples}, val={valid_set_size}"
            )

        _LOGGER.info(
            "Dataset split: train=%s test=%s val=%s",
            train_set_size,
            num_test_examples,
            valid_set_size,
        )

        self._train_dataset, self._test_dataset, self._val_dataset = random_split(
            full_dataset, [train_set_size, num_test_examples, valid_set_size]
        )

    def forward(self, text, text_lengths, scales, sid=None, bert_input=None):
        noise_scale = scales[0]
        length_scale = scales[1]
        noise_scale_w = scales[2]
        sdp_ratio = (
            scales[3]
            if len(scales) > 3
            else getattr(self.hparams, "duration_blend_sdp_ratio", 0.2)
        )
        audio, *_ = self.model_g.infer(
            text,
            text_lengths,
            noise_scale=noise_scale,
            length_scale=length_scale,
            noise_scale_w=noise_scale_w,
            sdp_ratio=sdp_ratio,
            sid=sid,
            bert_input=bert_input,
        )

        return audio

    def train_dataloader(self):
        num_workers = int(self.hparams.num_workers)
        collate_fn = UtteranceCollate(
            is_multispeaker=self.hparams.num_speakers > 1,
            segment_size=self.hparams.segment_size,
        )
        if getattr(self.hparams, "use_length_buckets", False):
            boundaries = list(
                getattr(self.hparams, "bucket_boundaries", None)
                or [0, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 2000]
            )
            return DataLoader(
                self._train_dataset,
                collate_fn=collate_fn,
                num_workers=num_workers,
                batch_sampler=LengthBucketBatchSampler(
                    self._train_dataset,
                    batch_size=int(self.hparams.batch_size),
                    boundaries=boundaries,
                    shuffle=True,
                ),
                pin_memory=torch.cuda.is_available(),
                persistent_workers=num_workers > 0,
            )

        return DataLoader(
            self._train_dataset,
            collate_fn=collate_fn,
            num_workers=num_workers,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self):
        num_workers = int(self.hparams.num_workers)
        return DataLoader(
            self._val_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def test_dataloader(self):
        num_workers = int(self.hparams.num_workers)
        return DataLoader(
            self._test_dataset,
            collate_fn=UtteranceCollate(
                is_multispeaker=self.hparams.num_speakers > 1,
                segment_size=self.hparams.segment_size,
            ),
            num_workers=num_workers,
            batch_size=self.hparams.batch_size,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=num_workers > 0,
        )

    def training_step(self, batch: Batch, batch_idx: int):
        # Manual optimization for multiple optimizers (Lightning 2.0+)
        g_opt, d_opt = self.optimizers()
        profile_limit = int(os.environ.get("PIPER_PROFILE_BATCHES", "0") or 0)
        profile = batch_idx < profile_limit

        def mark() -> float:
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            return time.perf_counter()

        t0 = mark()

        # Generator step
        self.toggle_optimizer(g_opt)
        try:
            loss_gen_all = self.training_step_g(batch)
            t_gen_loss = mark()
            g_opt.zero_grad()
            if profile and os.environ.get("PIPER_PROFILE_BACKWARD_TABLE"):
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=True,
                    profile_memory=False,
                    with_stack=False,
                ) as prof:
                    self.manual_backward(loss_gen_all)
                _LOGGER.info(
                    "Generator backward profiler:\n%s",
                    prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=int(
                            os.environ.get("PIPER_PROFILE_BACKWARD_ROWS", "24")
                        ),
                    ),
                )
            else:
                self.manual_backward(loss_gen_all)
            if self.hparams.grad_clip:
                self.clip_gradients(g_opt, gradient_clip_val=self.hparams.grad_clip)
            g_opt.step()
            t_gen_step = mark()
        finally:
            self.untoggle_optimizer(g_opt)

        # Discriminator step
        self.toggle_optimizer(d_opt)
        try:
            loss_disc_all = self.training_step_d(batch)
            t_disc_loss = mark()
            d_opt.zero_grad()
            self.manual_backward(loss_disc_all)
            if self.hparams.grad_clip:
                self.clip_gradients(d_opt, gradient_clip_val=self.hparams.grad_clip)
            d_opt.step()
            t_disc_step = mark()
        finally:
            self.untoggle_optimizer(d_opt)

        if profile:
            param_device = next(self.model_g.parameters()).device
            _LOGGER.info(
                "Batch profile: epoch=%s batch=%s global_step=%s "
                "devices=(model=%s phonemes=%s spec=%s audio=%s) "
                "lengths=(phone_max=%s spec_max=%s audio_max=%s) "
                "times=(gen_loss=%.3fs gen_backward_step=%.3fs "
                "disc_loss=%.3fs disc_backward_step=%.3fs total=%.3fs)",
                int(self.current_epoch) + 1,
                batch_idx + 1,
                int(self.global_step),
                param_device,
                batch.phoneme_ids.device,
                batch.spectrograms.device,
                batch.audios.device,
                int(batch.phoneme_lengths.max().item()),
                int(batch.spectrogram_lengths.max().item()),
                int(batch.audio_lengths.max().item()),
                t_gen_loss - t0,
                t_gen_step - t_gen_loss,
                t_disc_loss - t_gen_step,
                t_disc_step - t_disc_loss,
                t_disc_step - t0,
            )

    def training_step_g(self, batch: Batch):
        x, x_lengths, y, _, spec, spec_lengths, speaker_ids = (
            batch.phoneme_ids,
            batch.phoneme_lengths,
            batch.audios,
            batch.audio_lengths,
            batch.spectrograms,
            batch.spectrogram_lengths,
            batch.speaker_ids if batch.speaker_ids is not None else None,
        )

        bert_input = None
        if getattr(self.hparams, "use_bert", False):
            if batch.bert_features is not None:
                bert_input = {"features": batch.bert_features.to(x.device)}
            elif getattr(self.hparams, "bert_features_precomputed", False):
                raise ValueError(
                    "model.use_bert=true with precomputed BERT features, but the batch has no bert_features. "
                    "Run the Piper BERT feature precompute step and train from dataset.parquet."
                )
            elif batch.texts is not None:
                # Import semantic helpers lazily so that transformers is only
                # required when the semantic encoder is actually used.
                from ..semantic import SemanticTokenizer, build_bert_input

                if self._semantic_tokenizer is None:
                    # Use the same model id as the generator's BERT encoder when provided,
                    # otherwise fall back to environment/defaults inside SemanticTokenizer.
                    model_name = getattr(self.hparams, "bert_model_name", None)
                    self._semantic_tokenizer = SemanticTokenizer(model_name=model_name)

                phoneme_lengths = [int(length.item()) for length in x_lengths]
                bert_input = build_bert_input(
                    batch.texts,
                    self._semantic_tokenizer,
                    phoneme_lengths=phoneme_lengths,
                    word_spans=batch.word_spans,
                )

                if bert_input is not None:
                    # Move semantic inputs to the same device as the phoneme ids / model.
                    dev = x.device
                    bert_input = {
                        key: value.to(dev)
                        for key, value in bert_input.items()
                    }

                    if _DEBUG_SEMANTIC:
                        # Approximate effective text length from attention mask
                        attn = bert_input["attention_mask"]
                        eff_len = int(attn[0].sum().item()) if attn.size(0) > 0 else 0
                        _LOGGER.debug(
                            "training_step_g: x_len[0]=%s, spec_len[0]=%s, "
                            "bert_input_ids_shape=%s, bert_attn_shape=%s, bert_eff_len[0]=%s, device=%s, example_text[0]=%r",
                            int(x_lengths[0].item()) if x_lengths.numel() > 0 else -1,
                            int(spec_lengths[0].item()) if spec_lengths.numel() > 0 else -1,
                            tuple(bert_input["input_ids"].shape),
                            tuple(bert_input["attention_mask"].shape),
                            eff_len,
                            dev,
                            batch.texts[0] if batch.texts else None,
                        )

        model_g_out = self.model_g(
            x,
            x_lengths,
            spec,
            spec_lengths,
            sid=speaker_ids,
            bert_input=bert_input,
        )
        (
            y_hat,
            l_length,
            _attn,
            ids_slice,
            _x_mask,
            z_mask,
            (_z, z_p, m_p, logs_p, _m_q, logs_q),
            *maybe_duration_losses,
        ) = model_g_out
        duration_losses = maybe_duration_losses[0] if maybe_duration_losses else None
        self._y_hat = y_hat

        # STFT runs through complex tensors. ComplexHalf is slow/experimental on
        # older CUDA targets, so keep the mel loss path in fp32 under AMP.
        with autocast(self.device.type, enabled=False):
            mel = spec_to_mel_torch(
                spec.float(),
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
            y_mel = slice_segments(
                mel,
                ids_slice,
                self.hparams.segment_size // self.hparams.hop_length,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(),
                self.hparams.filter_length,
                self.hparams.mel_channels,
                self.hparams.sample_rate,
                self.hparams.hop_length,
                self.hparams.win_length,
                self.hparams.mel_fmin,
                self.hparams.mel_fmax,
            )
        y = slice_segments(
            y,
            ids_slice * self.hparams.hop_length,
            self.hparams.segment_size,
        )  # slice

        # Save for training_step_d
        self._y = y

        _y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.model_d(y, y_hat)

        with autocast(self.device.type, enabled=False):
            # Generator loss
            loss_dur_sdp = None
            loss_dur_dp = None
            if duration_losses is not None:
                loss_dur_sdp = (
                    torch.sum(duration_losses["sdp"].float())
                    * float(getattr(self.hparams, "duration_sdp_loss_weight", 1.0))
                )
                loss_dur_dp = (
                    torch.sum(duration_losses["dp"].float())
                    * float(getattr(self.hparams, "duration_dp_loss_weight", 1.0))
                )
                loss_dur = loss_dur_sdp + loss_dur_dp
            else:
                loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.hparams.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.hparams.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, _losses_gen = generator_loss(y_d_hat_g)
            loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl

            self.log("loss_gen_all", loss_gen_all)
            self.log("loss_dur", loss_dur)
            if loss_dur_sdp is not None and loss_dur_dp is not None:
                self.log("loss_dur_sdp", loss_dur_sdp)
                self.log("loss_dur_dp", loss_dur_dp)

            return loss_gen_all

    def training_step_d(self, batch: Batch):
        # From training_step_g
        y = self._y
        y_hat = self._y_hat
        y_d_hat_r, y_d_hat_g, _, _ = self.model_d(y, y_hat.detach())

        with autocast(self.device.type, enabled=False):
            # Discriminator
            loss_disc, _losses_disc_r, _losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            self.log("loss_disc_all", loss_disc_all)

            return loss_disc_all

    def validation_step(self, batch: Batch, batch_idx: int):
        val_loss = self.training_step_g(batch) + self.training_step_d(batch)
        self.log("val_loss", val_loss)

        if batch_idx != 0:
            return val_loss

        # Generate audio examples
        for utt_idx, test_utt in enumerate(self._test_dataset):
            text = test_utt.phoneme_ids.unsqueeze(0).to(self.device)
            text_lengths = torch.LongTensor([len(test_utt.phoneme_ids)]).to(self.device)
            if bool(getattr(self.hparams, "use_duration_blend", False)):
                scales = [
                    0.6,
                    1.0,
                    0.8,
                    float(getattr(self.hparams, "duration_blend_sdp_ratio", 0.2)),
                ]
            else:
                scales = [0.667, 1.0, 0.8]
            sid = (
                test_utt.speaker_id.to(self.device)
                if test_utt.speaker_id is not None
                else None
            )

            bert_input = None
            if getattr(self.hparams, "use_bert", False):
                if test_utt.bert_features is not None:
                    bert_input = {
                        "features": test_utt.bert_features.unsqueeze(0).to(self.device)
                    }
                elif getattr(self.hparams, "bert_features_precomputed", False):
                    raise ValueError(
                        "model.use_bert=true with precomputed BERT features, but the held-out sample has no bert_features"
                    )
                elif test_utt.text:
                    from ..semantic import SemanticTokenizer, build_bert_input

                    if self._semantic_tokenizer is None:
                        model_name = getattr(self.hparams, "bert_model_name", None)
                        self._semantic_tokenizer = SemanticTokenizer(model_name=model_name)

                    bert_dict = build_bert_input(
                        [test_utt.text],
                        self._semantic_tokenizer,
                        phoneme_lengths=[len(test_utt.phoneme_ids)],
                        word_spans=[test_utt.word_spans],
                    )
                    if bert_dict is not None:
                        bert_input = {
                            key: value.to(self.device)
                            for key, value in bert_dict.items()
                        }

            test_audio = self(
                text,
                text_lengths,
                scales,
                sid=sid,
                bert_input=bert_input,
            ).detach()

            # TensorBoard warns and clips if generated samples exceed [-1, 1].
            peak = test_audio.abs().amax().clamp_min(0.01)
            test_audio = (test_audio.float() / peak).clamp(-1.0, 1.0)

            tag = test_utt.text or str(utt_idx)
            self.logger.experiment.add_audio(
                tag, test_audio, sample_rate=self.hparams.sample_rate
            )

        return val_loss

    def configure_optimizers(self):
        g_opt = torch.optim.AdamW(
            self.model_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )
        d_opt = torch.optim.AdamW(
            self.model_d.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            eps=self.hparams.eps,
        )

        g_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            g_opt, gamma=self.hparams.lr_decay
        )
        d_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            d_opt, gamma=self.hparams.lr_decay
        )

        return [g_opt, d_opt], [g_scheduler, d_scheduler]

    def on_train_epoch_end(self):
        # Manually step schedulers with manual optimization
        schedulers = self.lr_schedulers()
        if schedulers:
            for scheduler in schedulers:
                scheduler.step()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("VitsModel")
        parser.add_argument("--batch-size", type=int, required=True)
        parser.add_argument("--num-workers", type=int, default=1)
        parser.add_argument("--validation-split", type=float, default=0.1)
        parser.add_argument("--num-test-examples", type=int, default=5)
        parser.add_argument(
            "--max-phoneme-ids",
            type=int,
            help="Exclude utterances with phoneme id lists longer than this",
        )
        #
        parser.add_argument("--hidden-channels", type=int, default=192)
        parser.add_argument("--inter-channels", type=int, default=192)
        parser.add_argument("--filter-channels", type=int, default=768)
        parser.add_argument("--n-layers", type=int, default=6)
        parser.add_argument("--n-heads", type=int, default=2)
        parser.add_argument("--use-sdp", action=argparse.BooleanOptionalAction, default=True)
        parser.add_argument(
            "--use-duration-blend",
            action="store_true",
            help="Train both stochastic and deterministic duration predictors and blend them at inference.",
        )
        parser.add_argument("--duration-blend-sdp-ratio", type=float, default=0.2)
        parser.add_argument("--duration-sdp-loss-weight", type=float, default=1.0)
        parser.add_argument("--duration-dp-loss-weight", type=float, default=1.0)
        # semantic / BERT options
        parser.add_argument(
            "--use-bert",
            action="store_true",
            help="Enable semantic encoder (BERT) branch in the text encoder",
        )
        parser.add_argument(
            "--bert-model-name",
            type=str,
            help="HuggingFace model id or local path for the semantic encoder (defaults to PIPER_SEMANTIC_MODEL_NAME or a multilingual DistilBERT)",
        )
        parser.add_argument(
            "--bert-features-precomputed",
            action="store_true",
            help="Use precomputed phone-level BERT features from the dataset instead of running BERT during training",
        )
        parser.add_argument(
            "--bert-fusion-weight",
            type=float,
            default=0.5,
            help="Legacy cross-attention BERT fusion weight for older checkpoints",
        )
        parser.add_argument(
            "--semantic-fusion-mode",
            choices=["aligned", "legacy_cross_attention"],
            help="Semantic fusion mode. Defaults to aligned for precomputed BERT features and legacy cross-attention otherwise.",
        )
        #
        return parent_parser
