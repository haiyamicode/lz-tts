import math
import os
import typing

import torch
from torch import nn
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from . import attentions, commons, modules, monotonic_align
from .voice_adapter import (
    install_conditioning_lora,
    remove_conditioning_lora,
    set_conditioning_lora_enabled,
    unwrap_lora_base,
)
from .commons import get_padding, init_weights

_DEBUG_SEMANTIC = bool(int(os.environ.get("PIPER_SEMANTIC_DEBUG", "0")))


class StochasticDurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        filter_channels = in_channels  # it needs to be removed from future version.
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.log_flow = modules.Log()
        self.flows = nn.ModuleList()
        self.flows.append(modules.ElementwiseAffine(2))
        for i in range(n_flows):
            self.flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.flows.append(modules.Flip())

        self.post_pre = nn.Conv1d(1, filter_channels, 1)
        self.post_proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.post_convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        self.post_flows = nn.ModuleList()
        self.post_flows.append(modules.ElementwiseAffine(2))
        for i in range(4):
            self.post_flows.append(
                modules.ConvFlow(2, filter_channels, kernel_size, n_layers=3)
            )
            self.post_flows.append(modules.Flip())

        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.proj = nn.Conv1d(filter_channels, filter_channels, 1)
        self.convs = modules.DDSConv(
            filter_channels, kernel_size, n_layers=3, p_dropout=p_dropout
        )
        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, filter_channels, 1)

    def forward(self, x, x_mask, w=None, g=None, reverse=False, noise_scale=1.0):
        x = torch.detach(x)
        x = self.pre(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.convs(x, x_mask)
        x = self.proj(x) * x_mask

        if not reverse:
            flows = self.flows
            assert w is not None

            logdet_tot_q = 0
            h_w = self.post_pre(w)
            h_w = self.post_convs(h_w, x_mask)
            h_w = self.post_proj(h_w) * x_mask
            e_q = torch.randn(w.size(0), 2, w.size(2)).type_as(x) * x_mask
            z_q = e_q
            for flow in self.post_flows:
                z_q, logdet_q = flow(z_q, x_mask, g=(x + h_w))
                logdet_tot_q += logdet_q
            z_u, z1 = torch.split(z_q, [1, 1], 1)
            u = torch.sigmoid(z_u) * x_mask
            z0 = (w - u) * x_mask
            logdet_tot_q += torch.sum(
                (F.logsigmoid(z_u) + F.logsigmoid(-z_u)) * x_mask, [1, 2]
            )
            logq = (
                torch.sum(-0.5 * (math.log(2 * math.pi) + (e_q**2)) * x_mask, [1, 2])
                - logdet_tot_q
            )

            logdet_tot = 0
            z0, logdet = self.log_flow(z0, x_mask)
            logdet_tot += logdet
            z = torch.cat([z0, z1], 1)
            for flow in flows:
                z, logdet = flow(z, x_mask, g=x, reverse=reverse)
                logdet_tot = logdet_tot + logdet
            nll = (
                torch.sum(0.5 * (math.log(2 * math.pi) + (z**2)) * x_mask, [1, 2])
                - logdet_tot
            )
            return nll + logq  # [b]
        else:
            flows = list(reversed(self.flows))
            flows = flows[:-2] + [flows[-1]]  # remove a useless vflow
            z = torch.randn(x.size(0), 2, x.size(2)).type_as(x) * noise_scale

            for flow in flows:
                z = flow(z, x_mask, g=x, reverse=reverse)
            z0, z1 = torch.split(z, [1, 1], 1)
            logw = z0
            return logw


class DurationPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        filter_channels: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(
            in_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(
            filter_channels, filter_channels, kernel_size, padding=kernel_size // 2
        )
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, in_channels, 1)

    def forward(self, x, x_mask, g=None):
        x = torch.detach(x)
        if g is not None:
            g = torch.detach(g)
            x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask


class TextEncoder(nn.Module):
    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        gin_channels: int = 0,
        speaker_condition_layer: int = 2,
    ):
        super().__init__()
        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            speaker_condition_layer=speaker_condition_layer,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x = self.emb(x) * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)

        x = self.encoder(x * x_mask, x_mask, g=g)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask


class BertTextEncoder(nn.Module):
    """
    Text encoder with optional semantic (BERT-like) conditioning.

    This is a drop-in replacement for TextEncoder:

        forward(x, x_lengths, bert_input=None) -> (x, m, logs, x_mask)

    where `bert_input`, when provided, is a dict with:

        - 'input_ids': LongTensor [B, T_text]
        - 'attention_mask': LongTensor [B, T_text]
        - 'word2ph': LongTensor [B, T_text], repeat counts that sum to T_phone
    """

    def __init__(
        self,
        n_vocab: int,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        bert_model: str = "distilbert/distilbert-base-multilingual-cased",
        bert_hidden_size: int = 768,
        freeze_bert: bool = True,
        bert_features_precomputed: bool = False,
        fusion_weight: float = 0.5,
        semantic_fusion_mode: typing.Optional[str] = None,
        gin_channels: int = 0,
        speaker_condition_layer: int = 2,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.freeze_bert = bool(freeze_bert)
        self.bert_features_precomputed = bool(bert_features_precomputed)
        self.fusion_weight = float(fusion_weight)
        self.semantic_fusion_mode = (
            semantic_fusion_mode
            or ("aligned" if self.bert_features_precomputed else "legacy_cross_attention")
        )
        self._debug_once = False

        # === Phoneme branch (same structure as TextEncoder) ===
        self.emb = nn.Embedding(n_vocab, hidden_channels)
        nn.init.normal_(self.emb.weight, 0.0, hidden_channels**-0.5)

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
            gin_channels=gin_channels,
            speaker_condition_layer=speaker_condition_layer,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

        self.bert = None
        if self.bert_features_precomputed:
            bert_hidden = int(bert_hidden_size)
        else:
            # === Semantic branch (BERT encoder) ===
            # The model is loaded by identifier so it can be swapped easily.
            from transformers import AutoModel  # lazy import to avoid hard dependency when unused
            from ..hf_cache import resolve_hf_model_path

            if _DEBUG_SEMANTIC:
                print(f"[BertTextEncoder] Loading semantic model: {bert_model}")

            self.bert = AutoModel.from_pretrained(
                resolve_hf_model_path(bert_model, require_weights=True)
            )

            if self.freeze_bert:
                for param in self.bert.parameters():
                    param.requires_grad = False
                self.bert.eval()

            bert_hidden = getattr(self.bert.config, "hidden_size", bert_hidden_size)
        self.bert_projection = nn.Linear(bert_hidden, hidden_channels)

        self.cross_attention = None
        if self.semantic_fusion_mode == "legacy_cross_attention":
            # Compatibility path for older lzspeech-bert checkpoints.
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_channels,
                num_heads=n_heads,
                dropout=p_dropout,
                batch_first=True,
            )

        self.layer_norm = nn.LayerNorm(hidden_channels)

    def _expand_semantic_features(
        self,
        semantic_features: torch.Tensor,
        bert_mask: torch.Tensor,
        x_lengths: torch.Tensor,
        word2ph: torch.Tensor,
        max_phone_len: int,
    ) -> torch.Tensor:
        word2ph = word2ph.to(device=semantic_features.device, dtype=torch.long)

        aligned = semantic_features.new_zeros(
            semantic_features.size(0),
            max_phone_len,
            semantic_features.size(-1),
        )
        for b in range(semantic_features.size(0)):
            phone_len = int(x_lengths[b].item())
            counts = torch.clamp(word2ph[b], min=0)
            diff = phone_len - int(counts.sum().item())
            if diff:
                active = torch.nonzero(bert_mask[b].bool(), as_tuple=False).flatten()
                adjust_idx = int(active[-1].item()) if active.numel() else 0
                counts = counts.clone()
                counts[adjust_idx] = torch.clamp(counts[adjust_idx] + diff, min=0)

            repeated = torch.repeat_interleave(semantic_features[b], counts, dim=0)
            if repeated.size(0) == 0:
                continue
            copy_len = min(phone_len, repeated.size(0), max_phone_len)
            aligned[b, :copy_len] = repeated[:copy_len]

        return aligned

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        bert_input: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
        g: typing.Optional[torch.Tensor] = None,
    ):
        # === Phoneme branch ===
        x_embed = self.emb(x) * math.sqrt(self.hidden_channels)  # [B, T_phone, H]
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x_embed.size(1)), 1
        ).type_as(x_embed)

        if self.semantic_fusion_mode == "legacy_cross_attention":
            x_ph = torch.transpose(x_embed, 1, -1)  # [B, H, T_phone]
            x_enc = self.encoder(x_ph * x_mask, x_mask, g=g)

            if _DEBUG_SEMANTIC and not self._debug_once:
                print(
                    f"[BertTextEncoder] legacy x_enc shape={tuple(x_enc.shape)}, "
                    f"x_mask shape={tuple(x_mask.shape)}, "
                    f"bert_input={'yes' if bert_input is not None else 'no'}"
                )
                self._debug_once = True

            if bert_input is not None:
                if self.bert is None or self.cross_attention is None:
                    raise ValueError(
                        "Legacy cross-attention semantic fusion requires a runtime BERT model"
                    )

                bert_ids = bert_input["input_ids"]
                bert_mask = bert_input["attention_mask"]
                if self.freeze_bert:
                    with torch.no_grad():
                        bert_output = self.bert(
                            input_ids=bert_ids,
                            attention_mask=bert_mask,
                        )
                else:
                    bert_output = self.bert(
                        input_ids=bert_ids,
                        attention_mask=bert_mask,
                    )

                semantic_features = self.bert_projection(bert_output.last_hidden_state)
                x_seq = x_enc.transpose(1, 2)
                aligned, _ = self.cross_attention(
                    query=x_seq,
                    key=semantic_features,
                    value=semantic_features,
                    key_padding_mask=~bert_mask.bool(),
                )
                fused = (1.0 - self.fusion_weight) * x_seq + self.fusion_weight * aligned
                x_enc = self.layer_norm(fused).transpose(1, 2)

            stats = self.proj(x_enc) * x_mask
            m, logs = torch.split(stats, self.out_channels, dim=1)
            return x_enc, m, logs, x_mask

        if _DEBUG_SEMANTIC and not self._debug_once:
            print(
                f"[BertTextEncoder] x_embed shape={tuple(x_embed.shape)}, "
                f"x_mask shape={tuple(x_mask.shape)}, "
                f"bert_input={'yes' if bert_input is not None else 'no'}"
            )
            self._debug_once = True

        if bert_input is not None:
            if "features" in bert_input:
                raw_features = bert_input["features"]
                if raw_features.dim() != 3:
                    raise ValueError(
                        f"Precomputed BERT features must be 3D, got {tuple(raw_features.shape)}"
                    )
                if raw_features.size(1) == self.bert_projection.in_features:
                    # [B, H_bert, T_phone] -> [B, T_phone, H_bert]
                    raw_features = raw_features.transpose(1, 2)
                elif raw_features.size(2) != self.bert_projection.in_features:
                    raise ValueError(
                        "Precomputed BERT feature dim mismatch: "
                        f"expected {self.bert_projection.in_features}, got {tuple(raw_features.shape)}"
                    )

                semantic_features = self.bert_projection(raw_features)
                if semantic_features.size(1) == x_embed.size(1):
                    aligned = semantic_features
                elif semantic_features.size(1) > x_embed.size(1):
                    aligned = semantic_features[:, : x_embed.size(1)]
                else:
                    aligned = F.pad(
                        semantic_features,
                        (0, 0, 0, x_embed.size(1) - semantic_features.size(1)),
                    )
            else:
                if self.bert is None:
                    raise ValueError(
                        "This model expects precomputed BERT features, but bert_input['features'] is missing"
                    )

                bert_ids = bert_input["input_ids"]
                bert_mask = bert_input["attention_mask"]
                word2ph = bert_input.get("word2ph")
                if word2ph is None:
                    raise ValueError(
                        "Semantic input requires word2ph alignment counts. "
                        "Build bert_input with phoneme_lengths and word_spans."
                    )

                if self.freeze_bert:
                    with torch.no_grad():
                        bert_output = self.bert(
                            input_ids=bert_ids,
                            attention_mask=bert_mask,
                        )
                else:
                    bert_output = self.bert(
                        input_ids=bert_ids,
                        attention_mask=bert_mask,
                    )

                # [B, T_text, H_bert] -> [B, T_text, H]
                semantic_features = self.bert_projection(bert_output.last_hidden_state)

                aligned = self._expand_semantic_features(
                    semantic_features=semantic_features,
                    bert_mask=bert_mask,
                    x_lengths=x_lengths,
                    word2ph=word2ph,
                    max_phone_len=x_embed.size(1),
                )

            if _DEBUG_SEMANTIC and not self._debug_once:
                bsz, t_phone, _ = x_embed.shape
                _, t_text, _ = semantic_features.shape
                print(
                    f"[BertTextEncoder] word2ph: B={bsz}, T_phone={t_phone}, "
                    f"T_semantic={t_text}, precomputed={'features' in bert_input}"
                )

            x_embed = self.layer_norm(x_embed + aligned)

        x = torch.transpose(x_embed, 1, -1)  # [B, H, T_phone]
        x_enc = self.encoder(x * x_mask, x_mask, g=g)

        # === Project to prior mean/log-std as usual ===
        stats = self.proj(x_enc) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x_enc, m, logs, x_mask

    def unfreeze_bert(self):
        if self.bert is None:
            raise RuntimeError("Cannot unfreeze BERT because this encoder uses precomputed features")
        self.freeze_bert = False
        for param in self.bert.parameters():
            param.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_bert and self.bert is not None:
            self.bert.eval()
        return self


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(
            commons.sequence_mask(x_lengths, x.size(2)), 1
        ).type_as(x)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel: int,
        resblock: typing.Optional[str],
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        gin_channels: int = 0,
    ):
        super(Generator, self).__init__()
        self.LRELU_SLOPE = 0.1
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock_module = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock_module(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i, up in enumerate(self.ups):
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            x = up(x)
            xs = torch.zeros(1)
            for j, resblock in enumerate(self.resblocks):
                index = j - (i * self.num_kernels)
                if index == 0:
                    xs = resblock(x)
                elif (index > 0) and (index < self.num_kernels):
                    xs += resblock(x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(unwrap_lora_base(l))
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        super(DiscriminatorP, self).__init__()
        self.LRELU_SLOPE = 0.1
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        self.LRELU_SLOPE = 0.1
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab: int,
        spec_channels: int,
        segment_size: int,
        inter_channels: int,
        hidden_channels: int,
        filter_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: float,
        resblock: str,
        resblock_kernel_sizes: typing.Tuple[int, ...],
        resblock_dilation_sizes: typing.Tuple[typing.Tuple[int, ...], ...],
        upsample_rates: typing.Tuple[int, ...],
        upsample_initial_channel: int,
        upsample_kernel_sizes: typing.Tuple[int, ...],
        n_speakers: int = 1,
        gin_channels: int = 0,
        use_sdp: bool = True,
        use_duration_blend: bool = False,
        duration_blend_sdp_ratio: float = 0.2,
        # Semantic options
        use_bert: bool = False,
        bert_model: str = "distilbert/distilbert-base-multilingual-cased",
        bert_hidden_size: int = 768,
        freeze_bert: bool = True,
        bert_features_precomputed: bool = False,
        bert_fusion_weight: float = 0.5,
        semantic_fusion_mode: typing.Optional[str] = None,
        use_spk_conditioned_encoder: bool = False,
        speaker_condition_layer: int = 2,
    ):

        super().__init__()
        self.n_vocab = n_vocab
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels
        self.use_sdp = use_sdp
        self.use_duration_blend = bool(use_duration_blend)
        self.duration_blend_sdp_ratio = float(duration_blend_sdp_ratio)
        self.use_spk_conditioned_encoder = bool(use_spk_conditioned_encoder)
        enc_gin_channels = (
            gin_channels
            if self.use_spk_conditioned_encoder and n_speakers > 1 and gin_channels > 0
            else 0
        )

        # Text encoder: phoneme-only or phoneme+semantic (BertTextEncoder)
        if use_bert:
            self.enc_p = BertTextEncoder(
                n_vocab,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                bert_model=bert_model,
                bert_hidden_size=bert_hidden_size,
                freeze_bert=freeze_bert,
                bert_features_precomputed=bert_features_precomputed,
                fusion_weight=bert_fusion_weight,
                semantic_fusion_mode=semantic_fusion_mode,
                gin_channels=enc_gin_channels,
                speaker_condition_layer=speaker_condition_layer,
            )
        else:
            self.enc_p = TextEncoder(
                n_vocab,
                inter_channels,
                hidden_channels,
                filter_channels,
                n_heads,
                n_layers,
                kernel_size,
                p_dropout,
                gin_channels=enc_gin_channels,
                speaker_condition_layer=speaker_condition_layer,
            )
        self.dec = Generator(
            inter_channels,
            resblock,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            upsample_rates,
            upsample_initial_channel,
            upsample_kernel_sizes,
            gin_channels=gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels
        )

        if self.use_duration_blend:
            self.sdp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )
        elif use_sdp:
            self.dp = StochasticDurationPredictor(
                hidden_channels, 192, 3, 0.5, 4, gin_channels=gin_channels
            )
        else:
            self.dp = DurationPredictor(
                hidden_channels, 256, 3, 0.5, gin_channels=gin_channels
            )

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

        self.voice_adapter_embedding: typing.Optional[nn.Parameter] = None
        self.voice_adapter_speaker_id: typing.Optional[int] = None
        self.voice_adapter_target_modules: typing.Tuple[str, ...] = ()
        self.voice_adapter_active = False

    def configure_voice_adapter(
        self,
        speaker_id: int,
        target_modules: typing.Sequence[str],
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        if self.n_speakers <= 1 or not hasattr(self, "emb_g"):
            raise ValueError("Voice adapters require a multi-speaker Sparrow model")
        if not 0 <= int(speaker_id) < self.n_speakers:
            raise ValueError(
                f"Voice adapter speaker id {speaker_id} is outside [0, {self.n_speakers})"
            )
        if self.voice_adapter_embedding is not None:
            raise ValueError("A voice adapter is already configured")

        self.voice_adapter_speaker_id = int(speaker_id)
        self.voice_adapter_target_modules = install_conditioning_lora(
            self,
            target_modules=target_modules,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        initial_embedding = self.emb_g.weight[self.voice_adapter_speaker_id].detach().clone()
        self.voice_adapter_embedding = nn.Parameter(initial_embedding)
        self.voice_adapter_active = True

    def set_voice_adapter_enabled(self, enabled: bool) -> None:
        if self.voice_adapter_embedding is None:
            if enabled:
                raise RuntimeError("No voice adapter is installed")
            return
        set_conditioning_lora_enabled(
            self,
            self.voice_adapter_target_modules,
            enabled,
        )
        self.voice_adapter_active = bool(enabled)

    def remove_voice_adapter(self) -> None:
        if self.voice_adapter_embedding is None:
            return
        remove_conditioning_lora(self, self.voice_adapter_target_modules)
        self.voice_adapter_embedding = None
        self.voice_adapter_speaker_id = None
        self.voice_adapter_target_modules = ()
        self.voice_adapter_active = False

    def reset_voice_adapter_embedding(self) -> None:
        if self.voice_adapter_embedding is None:
            return
        assert self.voice_adapter_speaker_id is not None
        with torch.no_grad():
            self.voice_adapter_embedding.copy_(
                self.emb_g.weight[self.voice_adapter_speaker_id]
            )

    def _speaker_conditioning(self, sid: torch.Tensor) -> torch.Tensor:
        if self.voice_adapter_embedding is None or not self.voice_adapter_active:
            return self.emb_g(sid).unsqueeze(-1)

        assert self.voice_adapter_speaker_id is not None
        if bool((sid != self.voice_adapter_speaker_id).any()):
            observed = sorted(set(int(value) for value in sid.detach().cpu().view(-1)))
            raise ValueError(
                "Voice adapter batches may only contain speaker id "
                f"{self.voice_adapter_speaker_id}; observed {observed}"
            )
        return self.voice_adapter_embedding.view(1, -1, 1).expand(
            sid.size(0), -1, -1
        )

    def forward(
        self,
        x,
        x_lengths,
        y,
        y_lengths,
        sid=None,
        bert_input: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
    ):

        if self.n_speakers > 1:
            g = self._speaker_conditioning(sid)  # [b, h, 1]
        else:
            g = None

        if isinstance(self.enc_p, BertTextEncoder):
            x, m_p, logs_p, x_mask = self.enc_p(
                x,
                x_lengths,
                bert_input=bert_input,
                g=g,
            )
        else:
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p)  # [b, d, t]
            neg_cent1 = torch.sum(
                -0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent2 = torch.matmul(
                -0.5 * (z_p**2).transpose(1, 2), s_p_sq_r
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(
                z_p.transpose(1, 2), (m_p * s_p_sq_r)
            )  # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(
                -0.5 * (m_p**2) * s_p_sq_r, [1], keepdim=True
            )  # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = (
                monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
                .unsqueeze(1)
                .detach()
            )

        w = attn.sum(2)
        duration_losses = None
        if self.use_duration_blend:
            l_length_sdp = self.sdp(x, x_mask, w, g=g)
            l_length_sdp = l_length_sdp / torch.sum(x_mask)
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length_dp = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )
            l_length = l_length_sdp + l_length_dp
            duration_losses = {"sdp": l_length_sdp, "dp": l_length_dp}
        elif self.use_sdp:
            l_length = self.dp(x, x_mask, w, g=g)
            l_length = l_length / torch.sum(x_mask)
        else:
            logw_ = torch.log(w + 1e-6) * x_mask
            logw = self.dp(x, x_mask, g=g)
            l_length = torch.sum((logw - logw_) ** 2, [1, 2]) / torch.sum(
                x_mask
            )  # for averaging

        # expand prior
        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        o = self.dec(z_slice, g=g)
        return (
            o,
            l_length,
            attn,
            ids_slice,
            x_mask,
            y_mask,
            (z, z_p, m_p, logs_p, m_q, logs_q),
            duration_losses,
        )

    def infer(
        self,
        x,
        x_lengths,
        sid=None,
        noise_scale=0.667,
        length_scale=1,
        noise_scale_w=0.8,
        sdp_ratio: typing.Optional[float] = None,
        max_len=None,
        bert_input: typing.Optional[typing.Dict[str, torch.Tensor]] = None,
    ):
        if self.n_speakers > 1:
            assert sid is not None, "Missing speaker id"
            g = self._speaker_conditioning(sid)  # [b, h, 1]
        else:
            g = None

        if isinstance(self.enc_p, BertTextEncoder):
            x, m_p, logs_p, x_mask = self.enc_p(
                x,
                x_lengths,
                bert_input=bert_input,
                g=g,
            )
        else:
            x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, g=g)

        if self.use_duration_blend:
            ratio = (
                self.duration_blend_sdp_ratio
                if sdp_ratio is None
                else float(sdp_ratio)
            )
            ratio = min(1.0, max(0.0, ratio))
            logw = (
                self.sdp(
                    x,
                    x_mask,
                    g=g,
                    reverse=True,
                    noise_scale=noise_scale_w,
                )
                * ratio
                + self.dp(x, x_mask, g=g) * (1.0 - ratio)
            )
        elif self.use_sdp:
            logw = self.dp(x, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(
            commons.sequence_mask(y_lengths, y_lengths.max()), 1
        ).type_as(x_mask)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask)

        m_p = torch.matmul(attn.squeeze(1), m_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn.squeeze(1), logs_p.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)

        return o, attn, y_mask, (z, z_p, m_p, logs_p)

    def voice_conversion(self, y, y_lengths, sid_src, sid_tgt):
        assert self.n_speakers > 1, "n_speakers have to be larger than 1."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)
