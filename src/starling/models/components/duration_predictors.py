"""Duration predictors used by Matcha encoders."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.starling.models.components.conditioning import FiLM1d


class LayerNorm(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class DDSConv(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        n_layers: int,
        p_dropout: float = 0.0,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.condition_dim = int(condition_dim or 0)
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList()
        self.convs_1x1 = nn.ModuleList()
        self.norms_1 = nn.ModuleList()
        self.norms_2 = nn.ModuleList()
        self.films_1 = nn.ModuleList()
        self.films_2 = nn.ModuleList()
        for i in range(n_layers):
            dilation = kernel_size**i
            padding = (kernel_size * dilation - dilation) // 2
            self.convs_sep.append(
                nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=dilation, padding=padding)
            )
            self.convs_1x1.append(nn.Conv1d(channels, channels, 1))
            self.norms_1.append(LayerNorm(channels))
            self.norms_2.append(LayerNorm(channels))
            self.films_1.append(FiLM1d(self.condition_dim, channels) if self.condition_dim > 0 else nn.Identity())
            self.films_2.append(FiLM1d(self.condition_dim, channels) if self.condition_dim > 0 else nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if g is not None:
            x = x + g
        for i in range(self.n_layers):
            y = self.convs_sep[i](x * x_mask)
            y = self.norms_1[i](y)
            if self.condition_dim > 0:
                y = self.films_1[i](y, condition, x_mask)
            y = F.gelu(y)
            y = self.convs_1x1[i](y)
            y = self.norms_2[i](y)
            if self.condition_dim > 0:
                y = self.films_2[i](y, condition, x_mask)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class Log(nn.Module):
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        if reverse:
            return torch.exp(x) * x_mask
        y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
        logdet = torch.sum(-y, [1, 2])
        return y, logdet


class Flip(nn.Module):
    def forward(self, x: torch.Tensor, *args, reverse: bool = False, **kwargs):
        x = torch.flip(x, [1])
        if reverse:
            return x
        logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        return x, logdet


class ElementwiseAffine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, reverse: bool = False, **kwargs):
        if reverse:
            return (x - self.m) * torch.exp(-self.logs) * x_mask
        y = (self.m + torch.exp(self.logs) * x) * x_mask
        logdet = torch.sum(self.logs * x_mask, [1, 2])
        return y, logdet


def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    spline_fn = rational_quadratic_spline if tails is None else unconstrained_rational_quadratic_spline
    spline_kwargs = {} if tails is None else {"tails": tails, "tail_bound": tail_bound}
    return spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )


def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bin_locations[..., bin_locations.size(-1) - 1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails != "linear":
        raise RuntimeError(f"{tails} tails are not implemented.")

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., unnormalized_derivatives.size(-1) - 1] = constant
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    inside_outputs, inside_logabsdet = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )
    outputs[inside_interval_mask] = inside_outputs.to(inputs.dtype)
    logabsdet[inside_interval_mask] = inside_logabsdet.to(inputs.dtype)
    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=1e-3,
    min_bin_height=1e-3,
    min_derivative=1e-3,
):
    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., cumwidths.size(-1) - 1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., cumheights.size(-1) - 1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    bin_idx = searchsorted(cumheights if inverse else cumwidths, inputs)[..., None]
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    input_delta = (heights / widths).gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        a = a + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives
        b = b - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
        c = -input_delta * (inputs - input_cumheights)
        discriminant = b.pow(2) - 4 * a * c
        root = (2 * c) / (-b - torch.sqrt(discriminant.clamp_min(0.0)))
        outputs = root * input_bin_widths + input_cumwidths
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet

    theta = (inputs - input_cumwidths) / input_bin_widths
    theta_one_minus_theta = theta * (1 - theta)
    numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
    denominator = input_delta + (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
    outputs = input_cumheights + numerator / denominator
    derivative_numerator = input_delta.pow(2) * (
        input_derivatives_plus_one * theta.pow(2)
        + 2 * input_delta * theta_one_minus_theta
        + input_derivatives * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return outputs, logabsdet


class ConvFlow(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        n_layers: int,
        num_bins: int = 10,
        tail_bound: float = 5.0,
        condition_dim: int = 0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2
        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers, p_dropout=0.0, condition_dim=condition_dim)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: torch.Tensor | None = None,
        reverse: bool = False,
        condition: torch.Tensor | None = None,
    ):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g, condition=condition)
        h = self.proj(h) * x_mask
        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)
        unnormalized_widths = h[..., : self.num_bins] / math.sqrt(self.pre.out_channels)
        unnormalized_heights = h[..., self.num_bins : 2 * self.num_bins] / math.sqrt(self.pre.out_channels)
        unnormalized_derivatives = h[..., 2 * self.num_bins :]
        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1,
            unnormalized_widths,
            unnormalized_heights,
            unnormalized_derivatives,
            inverse=reverse,
            tails="linear",
            tail_bound=self.tail_bound,
        )
        x = torch.cat([x0, x1], 1) * x_mask
        if reverse:
            return x
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        return x, logdet


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        condition_dim: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels
        self.condition_dim = int(condition_dim or 0)
        self.log_flow = Log()
        self.flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(n_flows):
            self.flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3, condition_dim=self.condition_dim))
            self.flows.append(Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = DDSConv(
            filter_channels,
            kernel_size,
            n_layers=3,
            p_dropout=p_dropout,
            condition_dim=self.condition_dim,
        )
        self.post_flows = nn.ModuleList([ElementwiseAffine(2)])
        for _ in range(4):
            self.post_flows.append(ConvFlow(2, filter_channels, kernel_size, n_layers=3, condition_dim=self.condition_dim))
            self.post_flows.append(Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = DDSConv(
            filter_channels,
            kernel_size,
            n_layers=3,
            p_dropout=p_dropout,
            condition_dim=self.condition_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        w: torch.Tensor | None = None,
        reverse: bool = False,
        noise_scale: float = 1.0,
        condition: torch.Tensor | None = None,
    ):
        x = torch.detach(x)
        x = self.pre(x)
        x = self.convs(x, x_mask, condition=condition)
        x = self.proj(x) * x_mask

        if reverse:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]
            z = torch.randn(x.size(0), 2, x.size(2), dtype=x.dtype, device=x.device) * noise_scale
            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=True, condition=condition)
            z0, _ = torch.split(z, [1, 1], 1)
            return z0 * x_mask

        if w is None:
            raise ValueError("StochasticDurationPredictor training requires target durations")

        logdet_tot_q = 0
        h_w = self.post_pre(w)
        h_w = self.post_convs(h_w, x_mask, condition=condition)
        h_w = self.post_proj(h_w) * x_mask
        e_q = torch.randn(w.size(0), 2, w.size(2), dtype=x.dtype, device=x.device) * x_mask
        z_q = e_q
        for flow in self.post_flows:
            z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w), condition=condition)
            logdet_tot_q += logdet_q
        z_u, z1 = torch.split(z_q, [1, 1], 1)
        u = torch.sigmoid(z_u) * x_mask
        z0 = (w - u) * x_mask
        logdet_tot_q += torch.sum((F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2])
        logq = torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2]) - logdet_tot_q

        logdet_tot = 0
        z0, logdet = self.log_flow(z0, x_mask)
        logdet_tot += logdet
        z = torch.cat([z0, z1], 1)
        for flow in self.flows:
            z, logdet = flow(z, x_mask, g=x, condition=condition)
            logdet_tot += logdet
        nll = torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2]) - logdet_tot
        return nll + logq
