import datetime as dt
import math
import random

import torch

from src.starling import utils
from src.starling.models.baselightningmodule import BaseLightningClass
from src.starling.models.components.flow_matching import CFM
from src.starling.models.components.text_encoder import TextEncoder
from src.starling.utils import monotonic_align
from src.starling.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)


class MatchaTTS(BaseLightningClass):  # 🍵
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_feats,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size,
        optimizer=None,
        scheduler=None,
        prior_loss=True,
        use_precomputed_durations=False,
        freeze_spk_emb=False,
        prompt_mel_conditioning=None,
        prompt_embedding_encoder=None,
        condition_dim=None,
        voice_conditioning_cfg=None,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.condition_dim = int(spk_emb_dim if condition_dim is None else condition_dim)
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        self.prompt_mel_enabled = bool(prompt_mel_conditioning and prompt_mel_conditioning.get("enabled", False))
        self.prompt_embedding_enabled = bool(prompt_embedding_encoder and prompt_embedding_encoder.get("enabled", False))
        voice_conditioning_cfg = voice_conditioning_cfg or {}
        self.voice_cfg_drop_prob = float(voice_conditioning_cfg.get("cfg_drop_prob", 0.0))
        self.voice_cfg_inference_scale = float(voice_conditioning_cfg.get("inference_scale", 1.0))
        if not 0.0 <= self.voice_cfg_drop_prob <= 1.0:
            raise ValueError(f"voice_conditioning_cfg.cfg_drop_prob must be in [0, 1], got {self.voice_cfg_drop_prob}")
        condition_enabled = n_spks > 1 or self.prompt_embedding_enabled
        encoder_condition_n_spks = n_spks if n_spks > 1 else (2 if condition_enabled else 1)
        decoder_condition_n_spks = n_spks if n_spks > 1 else (2 if condition_enabled else 1)

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, self.condition_dim)
            if freeze_spk_emb:
                self.spk_emb.weight.requires_grad_(False)
        if self.prompt_embedding_enabled:
            input_dim = int(prompt_embedding_encoder.get("input_dim", self.condition_dim))
            hidden_dim = int(prompt_embedding_encoder.get("hidden_dim", max(self.condition_dim * 4, input_dim)))
            self.prompt_embedding_proj = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Dropout(prompt_embedding_encoder.get("dropout", 0.0)),
                torch.nn.Linear(hidden_dim, self.condition_dim),
            )
            if prompt_embedding_encoder.get("zero_init", False):
                torch.nn.init.zeros_(self.prompt_embedding_proj[-1].weight)
                torch.nn.init.zeros_(self.prompt_embedding_proj[-1].bias)

        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            encoder_condition_n_spks,
            self.condition_dim,
        )

        self.decoder = CFM(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=decoder_condition_n_spks,
            spk_emb_dim=self.condition_dim,
        )

        self.update_data_statistics(data_statistics)

    def _voice_condition_drop_mask(self, batch_size, device):
        if not self.training or self.voice_cfg_drop_prob <= 0.0:
            return None
        return torch.rand(batch_size, device=device) < self.voice_cfg_drop_prob

    @staticmethod
    def _apply_voice_drop(x, voice_drop_mask):
        if x is None or voice_drop_mask is None or not bool(voice_drop_mask.any()):
            return x
        x = x.clone()
        x[voice_drop_mask] = 0
        return x

    def _conditions(self, spks=None, prompt_embedding=None, voice_drop_mask=None, return_parts=False):
        speaker_cond = None
        voice_cond = None
        if self.n_spks > 1:
            if spks is None:
                raise ValueError("spks is required when n_spks > 1")
            speaker_cond = self.spk_emb(spks.long())
        if self.prompt_embedding_enabled and prompt_embedding is not None:
            dtype = speaker_cond.dtype if speaker_cond is not None else self.prompt_embedding_proj[1].weight.dtype
            voice_cond = self.prompt_embedding_proj(prompt_embedding.to(device=self.device, dtype=dtype))
            if voice_drop_mask is not None:
                voice_cond = self._apply_voice_drop(voice_cond, voice_drop_mask)

        cond = speaker_cond
        if voice_cond is not None:
            cond = voice_cond if cond is None else cond + voice_cond.to(dtype=cond.dtype)
        if return_parts:
            return cond, speaker_cond, voice_cond
        return cond

    @torch.inference_mode()
    def synthesise(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        spks=None,
        length_scale=1.0,
        semantic_features=None,
        noise_scale_w=None,
        sdp_ratio=None,
        prompt_mel=None,
        prompt_mel_lengths=None,
        prompt_embedding=None,
        voice_cfg_scale=None,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            spks (bool, optional): speaker ids.
                shape: (batch_size,)
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.

        Returns:
            dict: {
                "encoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Average mel spectrogram generated by the encoder
                "decoder_outputs": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Refined mel spectrogram improved by the CFM
                "attn": torch.Tensor, shape: (batch_size, max_text_length, max_mel_length),
                # Alignment map between text and mel spectrogram
                "mel": torch.Tensor, shape: (batch_size, n_feats, max_mel_length),
                # Denormalized mel spectrogram
                "mel_lengths": torch.Tensor, shape: (batch_size,),
                # Lengths of mel spectrograms
                "rtf": float,
                # Real-time factor
            }
        """
        # For RTF computation
        t = dt.datetime.now()

        spks, speaker_cond, voice_cond = self._conditions(spks, prompt_embedding, return_parts=True)
        if voice_cfg_scale is None:
            voice_cfg_scale = self.voice_cfg_inference_scale
        voice_cfg_scale = float(voice_cfg_scale)
        cfg_spks = None
        cfg_cond = None
        if voice_cfg_scale != 1.0 and (voice_cond is not None or prompt_mel is not None):
            cfg_spks = speaker_cond
            if cfg_spks is None:
                reference = voice_cond if voice_cond is not None else spks
                if reference is not None:
                    cfg_spks = torch.zeros_like(reference)
            cfg_cond = {}

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        if self.encoder.use_sdp:
            mu_x, logw_dp, x_mask, duration_hidden = self.encoder(
                x,
                x_lengths,
                spks,
                semantic_features=semantic_features,
                return_duration_hidden=True,
            )
            logw = self.encoder.blend_duration_logw(
                logw_dp,
                duration_hidden,
                x_mask,
                sdp_ratio=sdp_ratio,
                noise_scale_w=noise_scale_w,
                condition=spks,
            )
        else:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, semantic_features=semantic_features)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Generate sample tracing the probability flow
        decoder_outputs = self.decoder(
            mu_y,
            y_mask,
            n_timesteps,
            temperature,
            spks,
            prompt_mel=prompt_mel,
            prompt_mel_lengths=prompt_mel_lengths,
            voice_cfg_scale=voice_cfg_scale,
            cfg_spks=cfg_spks,
            cfg_cond=cfg_cond,
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)

        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        spks=None,
        out_size=None,
        cond=None,
        durations=None,
        semantic_features=None,
        prompt_mel=None,
        prompt_mel_lengths=None,
        prompt_embedding=None,
    ):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotonic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. flow matching loss: loss between mel-spectrogram and decoder outputs.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
                shape: (batch_size, max_text_length)
            x_lengths (torch.Tensor): lengths of texts in batch.
                shape: (batch_size,)
            y (torch.Tensor): batch of corresponding mel-spectrograms.
                shape: (batch_size, n_feats, max_mel_length)
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
                shape: (batch_size,)
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
            spks (torch.Tensor, optional): speaker ids.
                shape: (batch_size,)
        """
        voice_drop_mask = self._voice_condition_drop_mask(x.shape[0], x.device)
        spks = self._conditions(spks, prompt_embedding, voice_drop_mask=voice_drop_mask)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        if self.encoder.use_sdp:
            mu_x, logw, x_mask, duration_hidden = self.encoder(
                x,
                x_lengths,
                spks,
                semantic_features=semantic_features,
                return_duration_hidden=True,
            )
        else:
            mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, semantic_features=semantic_features)
            duration_hidden = None
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const

                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()  # b, t_text, T_mel

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        # refered to as prior loss in the paper
        target_durations = torch.sum(attn.unsqueeze(1), -1) * x_mask
        logw_ = torch.log(1e-8 + target_durations) * x_mask
        dur_dp_loss = duration_loss(logw, logw_, x_lengths)
        if self.encoder.use_sdp:
            dur_sdp_loss = self.encoder.stochastic_duration_loss(
                duration_hidden,
                x_mask,
                target_durations,
                condition=spks,
            )
            dur_loss = dur_dp_loss + dur_sdp_loss
        else:
            dur_sdp_loss = None
            dur_loss = dur_dp_loss

        # Cut a small segment of mel-spectrogram in order to increase batch size
        #   - "Hack" taken from Grad-TTS, in case of Grad-TTS, we cannot train batch size 32 on a 24GB GPU without it
        #   - Do not need this hack for Matcha-TTS, but it works with it as well
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor(
                [torch.tensor(random.choice(range(start, end)) if end > start else 0) for start, end in offset_ranges]
            ).to(y_lengths)
            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]

            y_cut_lengths = torch.stack(y_cut_lengths).to(y_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths, out_size).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of the decoder
        diff_loss, _ = self.decoder.compute_loss(
            x1=y,
            mask=y_mask,
            mu=mu_y,
            spks=spks,
            cond=cond,
            prompt_mel=prompt_mel,
            prompt_mel_lengths=prompt_mel_lengths,
            prompt_drop_mask=voice_drop_mask,
        )

        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0

        duration_losses = {"dur_dp_loss": dur_dp_loss}
        if dur_sdp_loss is not None:
            duration_losses["dur_sdp_loss"] = dur_sdp_loss

        return dur_loss, prior_loss, diff_loss, attn, duration_losses
