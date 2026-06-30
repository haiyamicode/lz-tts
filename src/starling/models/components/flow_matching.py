from abc import ABC

import torch
import torch.nn.functional as F

from src.starling.models.components.decoder import Decoder
from src.starling.models.components.f5_dit_decoder import MatchaF5DiTDecoder
from src.starling.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4

        self.estimator = None

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
        prompt_mel=None,
        prompt_mel_lengths=None,
    ):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if prompt_mel is not None:
            mu, mask, cond, prompt_mask, prompt_max_length = self._prepend_prompt(
                mu,
                mask,
                prompt_mel,
                prompt_mel_lengths,
                cond,
            )
        else:
            prompt_mask = None
            prompt_max_length = 0

        z = torch.randn_like(mu) * temperature
        if prompt_mask is not None:
            z = z.masked_fill(prompt_mask, 0.0)
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        sample = self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            prompt_mask=prompt_mask,
        )
        return sample[..., prompt_max_length:] if prompt_max_length else sample

    def solve_euler(self, x, t_span, mu, mask, spks, cond, prompt_mask=None):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []

        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)

            x = x + dt * dphi_dt
            if prompt_mask is not None:
                x = x.masked_fill(prompt_mask, 0.0)
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None, prompt_mel=None, prompt_mel_lengths=None):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """
        if prompt_mel is not None:
            return self._compute_prompt_loss(x1, mask, mu, spks, cond, prompt_mel, prompt_mel_lengths)

        b, _, _ = mu.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks, cond), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y

    def _prepend_prompt(self, mu, mask, prompt_mel, prompt_mel_lengths=None, cond=None):
        if prompt_mel.dim() != 3:
            raise ValueError(f"Expected prompt_mel to be 3D, got {tuple(prompt_mel.shape)}")
        if prompt_mel.shape[0] != mu.shape[0] or prompt_mel.shape[1] != mu.shape[1]:
            raise ValueError(
                f"Prompt mel shape {tuple(prompt_mel.shape)} is incompatible with mu shape {tuple(mu.shape)}"
            )
        prompt_mel = prompt_mel.to(device=mu.device, dtype=mu.dtype)
        batch_size, n_feats, prompt_max_length = prompt_mel.shape
        target_max_length = mu.shape[-1]
        if prompt_mel_lengths is None:
            prompt_mel_lengths = torch.full(
                (batch_size,),
                prompt_max_length,
                dtype=torch.long,
                device=mu.device,
            )
        else:
            prompt_mel_lengths = prompt_mel_lengths.to(device=mu.device, dtype=torch.long)

        full_mu = torch.zeros(
            batch_size,
            n_feats,
            prompt_max_length + target_max_length,
            dtype=mu.dtype,
            device=mu.device,
        )
        full_mu[..., prompt_max_length:] = mu

        full_mask = torch.zeros(
            batch_size,
            1,
            prompt_max_length + target_max_length,
            dtype=mask.dtype,
            device=mask.device,
        )
        full_mask[..., prompt_max_length:] = mask
        prompt_mask_1d = torch.arange(prompt_max_length, device=mu.device).unsqueeze(0) < prompt_mel_lengths.unsqueeze(1)
        full_mask[..., :prompt_max_length] = prompt_mask_1d.unsqueeze(1).to(dtype=mask.dtype)

        prompt_x = torch.zeros_like(full_mu)
        prompt_x[..., :prompt_max_length] = prompt_mel * prompt_mask_1d.unsqueeze(1).to(dtype=prompt_mel.dtype)
        prompt_mask = torch.zeros_like(full_mask, dtype=torch.bool)
        prompt_mask[..., :prompt_max_length] = prompt_mask_1d.unsqueeze(1)

        cond = dict(cond or {})
        cond["prompt_x"] = prompt_x
        return full_mu, full_mask, cond, prompt_mask, prompt_max_length

    def _compute_prompt_loss(self, x1, mask, mu, spks, cond, prompt_mel, prompt_mel_lengths):
        full_mu, full_mask, cond, prompt_mask, prompt_max_length = self._prepend_prompt(
            mu,
            mask,
            prompt_mel,
            prompt_mel_lengths,
            cond,
        )
        batch_size, n_feats, target_max_length = x1.shape
        full_x1 = torch.zeros_like(full_mu)
        full_x1[..., :prompt_max_length] = cond["prompt_x"][..., :prompt_max_length]
        full_x1[..., prompt_max_length:] = x1

        t = torch.rand([batch_size, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(full_x1)
        z = z.masked_fill(prompt_mask, 0.0)

        y = (1 - (1 - self.sigma_min) * t) * z + t * full_x1
        y = y.masked_fill(prompt_mask, 0.0)
        u = full_x1 - (1 - self.sigma_min) * z

        estimator_out = self.estimator(y, full_mask, full_mu, t.squeeze(), spks, cond)
        target_mask = torch.zeros_like(full_mask)
        target_mask[..., prompt_max_length:] = mask
        loss = F.mse_loss(estimator_out * target_mask, u * target_mask, reduction="sum") / (
            torch.sum(target_mask) * n_feats
        )
        return loss, y[..., prompt_max_length : prompt_max_length + target_max_length]


class CFM(BASECFM):
    def __init__(self, in_channels, out_channel, cfm_params, decoder_params, n_spks=1, spk_emb_dim=64):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )

        decoder_params = dict(decoder_params)
        decoder_type = decoder_params.pop("decoder_type", "unet")
        prompt_mel_condition = bool(decoder_params.get("prompt_mel_condition", False))
        in_channels = in_channels + (out_channel if prompt_mel_condition else 0)
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        global_speaker_condition = bool(decoder_params.pop("global_speaker_condition", False))
        if global_speaker_condition:
            decoder_params["global_cond_dim"] = spk_emb_dim if n_spks > 1 else None
        # Just change the architecture of the estimator here
        if decoder_type == "unet":
            self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        elif decoder_type == "f5_dit":
            self.estimator = MatchaF5DiTDecoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        else:
            raise ValueError(f"Unknown decoder_type={decoder_type}")
