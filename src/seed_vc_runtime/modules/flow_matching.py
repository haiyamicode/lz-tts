from abc import ABC

import torch
import torch.nn.functional as F

from modules.diffusion_transformer import DiT
from modules.commons import sequence_mask

from tqdm import tqdm

class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5):
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
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate)

    def solve_euler(self, x, x_lens, prompt, mu, style, f0, t_span, inference_cfg_rate=0.5):
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
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0
        for step in tqdm(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                stacked_x = torch.cat([x, x], dim=0)
                stacked_t = t.expand(stacked_x.size(0))
                stacked_x_lens = torch.cat([x_lens, x_lens], dim=0)

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, stacked_x_lens, stacked_t, stacked_style, stacked_mu,
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.expand(x.size(0)), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        return sol[-1]
    def forward(self, x1, x_lens, prompt_lens, mu, style):
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
        b, _, t = x1.shape

        # random timestep
        t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        prompt = torch.zeros_like(x1)
        for bib in range(b):
            prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
            # range covered by prompt are set to 0
            y[bib, :, :prompt_lens[bib]] = 0
            if self.zero_prompt_speech_token:
                mu[bib, :, :prompt_lens[bib]] = 0

        estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)
        loss = 0
        for bib in range(b):
            loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
        loss /= b

        return loss, estimator_out + (1 - self.sigma_min) * z

    def _run_teacher_trajectory(self, z, x_lens, prompt_x, prompt_len, mu, style,
                                  teacher_estimator, n_steps, cfg_rate):
        """
        Run teacher model for full trajectory from noise z to output.

        This is a non-tqdm version of solve_euler for use in training.

        Args:
            z: Initial noise (B, C, T)
            x_lens: Sequence lengths
            prompt_x: Prompt mel (B, C, T)
            prompt_len: Length of prompt region
            mu: Conditioning (B, T, D_cond)
            style: Style embedding (B, D)
            teacher_estimator: Teacher model
            n_steps: Number of euler steps
            cfg_rate: CFG rate

        Returns:
            x1_teacher: Final output after n_steps (B, C, T)
        """
        device = z.device
        t_span = torch.linspace(0, 1, n_steps + 1, device=device)

        x = z.clone()
        x[..., :prompt_len] = 0
        t = t_span[0]

        for step in range(1, len(t_span)):
            dt = t_span[step] - t_span[step - 1]

            if cfg_rate > 0:
                # CFG: stack conditioned and unconditioned
                null_prompt_x = torch.zeros_like(prompt_x)
                null_style = torch.zeros_like(style)
                null_mu = torch.zeros_like(mu)

                vel_cond = teacher_estimator(x, prompt_x, x_lens, t.unsqueeze(0).expand(x.size(0)), style, mu)
                vel_uncond = teacher_estimator(x, null_prompt_x, x_lens, t.unsqueeze(0).expand(x.size(0)), null_style, null_mu)
                dphi_dt = (1.0 + cfg_rate) * vel_cond - cfg_rate * vel_uncond
            else:
                dphi_dt = teacher_estimator(x, prompt_x, x_lens, t.unsqueeze(0).expand(x.size(0)), style, mu)

            x = x + dt * dphi_dt
            t = t + dt
            x[..., :prompt_len] = 0

        return x

    def distill_forward(self, x1, x_lens, prompt_lens, mu, style,
                        teacher_estimator, num_teacher_steps=30, cfg_rate=0.7):
        """
        Reflow Distillation for Flow Matching.

        Based on: "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"
        Paper: https://arxiv.org/abs/2209.03003

        The idea is simple:
        1. Sample noise z
        2. Run teacher for full trajectory: z → x1_teacher (N steps)
        3. Train student on STRAIGHT paths from z to x1_teacher
           - The straight-line velocity is: u = x1_teacher - z
           - This is constant regardless of timestep t

        After training, the flow becomes straighter, enabling fewer steps.

        Args:
            x1 (torch.Tensor): Target mel spectrogram (used for prompt creation)
                shape: (batch_size, n_feats, mel_timesteps)
            x_lens (torch.Tensor): lengths of each sequence
            prompt_lens (torch.Tensor): lengths of prompts
            mu (torch.Tensor): conditioning from length regulator
            style (torch.Tensor): style embedding from CAMPPlus
            teacher_estimator: frozen teacher DiT model
            num_teacher_steps: number of steps for teacher trajectory (default 30)
            cfg_rate: CFG rate (default 0.7, matching inference.py)

        Returns:
            loss: Reflow distillation loss
        """
        b, _, seq_len = x1.shape
        device = mu.device

        # Clone mu to avoid in-place modification
        mu = mu.clone()

        # For simplicity, use the minimum prompt_len for batch
        prompt_len = prompt_lens.min().item()

        # Prepare prompt_x from target (reference mel)
        prompt_x = torch.zeros_like(x1)
        for bib in range(b):
            prompt_x[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]

        # Zero prompt region in mu if configured
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        # ============================================================
        # Step 1: Sample noise z
        # ============================================================
        z = torch.randn_like(x1)
        z[..., :prompt_len] = 0

        # ============================================================
        # Step 2: Run teacher for full trajectory to get x1_teacher
        # ============================================================
        with torch.no_grad():
            x1_teacher = self._run_teacher_trajectory(
                z=z,
                x_lens=x_lens,
                prompt_x=prompt_x,
                prompt_len=prompt_len,
                mu=mu,
                style=style,
                teacher_estimator=teacher_estimator,
                n_steps=num_teacher_steps,
                cfg_rate=cfg_rate,
            )

        # ============================================================
        # Step 3: Compute straight-line target velocity
        # ============================================================
        # For a straight path from z to x1_teacher:
        # x_t = (1-t)*z + t*x1_teacher
        # dx/dt = x1_teacher - z  (constant velocity)
        u_target = x1_teacher - z  # This is the straight-line velocity

        # ============================================================
        # Step 4: Sample random timestep and create x_t
        # ============================================================
        t = torch.rand(b, device=device)
        t_expand = t.view(b, 1, 1)

        # Interpolate along straight path: x_t = (1-t)*z + t*x1_teacher
        x_t = (1 - t_expand) * z + t_expand * x1_teacher
        x_t[..., :prompt_len] = 0

        # ============================================================
        # Step 5: Student predicts velocity at (t, x_t)
        # ============================================================
        # Note: No CFG during student training - student learns the combined velocity
        v_student = self.estimator(
            x_t, prompt_x, x_lens, t, style, mu
        )

        # ============================================================
        # Step 6: Compute MSE loss between student velocity and target
        # ============================================================
        # Loss = MSE(v_student, u_target) on non-prompt regions
        loss = 0
        for bib in range(b):
            valid_region = slice(prompt_len, x_lens[bib].item())
            loss += F.mse_loss(
                v_student[bib, :, valid_region],
                u_target[bib, :, valid_region]
            )
        loss /= b

        return loss


class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")
