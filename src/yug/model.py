# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""The Yug architecture.

A patch-based, decoder-only transformer that forecasts directly: every input
patch produces a fixed block of future values at several quantile levels in one
forward pass. There is no token vocabulary and no sampling loop inside the
model — autoregression happens one *block* at a time, in the pipeline.

Layout, per decoder layer:

* causal self-attention over target patches, with RoPE on the patch index
* (multivariate only) self-attention over covariate patches, cross-variate
  attention, target->covariate cross-attention, and a learned compression of
  the covariate stack back into the target stream
* a SwiGLU MLP

Wrapping the stack is :class:`CausalReversibleInstanceNorm`: a cumulative
Welford scan that normalises each patch by the statistics of everything at or
before it, and denormalises the head's output the same way. Because it is
strictly causal, appending patches cannot disturb earlier positions — which is
what makes the cached rollout in :mod:`yug.engine` valid.

Module and attribute names in this file are part of the checkpoint format:
renaming any of them breaks ``load_state_dict`` against published weights.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "Yug_Model",
    "Yug_RoPEEmbedding",
    "Yug_ResidualBlock",
    "Yug_FrequencyEmbedding",
    "Yug_SelfAttention",
    "Yug_CrossAttention",
    "Yug_CrossVariate_Attention",
    "Yug_MLP",
    "Yug_Variate_MLP",
    "Yug_DecoderLayer",
    "CausalReversibleInstanceNorm",
]


class Yug_RoPEEmbedding(nn.Module):
    """Rotary position embedding, keyed on the absolute patch index.

    Pairs adjacent dimensions (``repeat_interleave(2)``) rather than splitting
    the vector in half, so ``rotate_half`` swaps within each pair.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = 0
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None

    def _update_cache(self, seq_len: int, device, dtype):
        """Grow the cos/sin tables on demand. Values at index i never change."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.to(dtype))
            emb = freqs.repeat_interleave(2, dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        assert self._cos_cached is not None and self._sin_cached is not None
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """``[x0, x1] -> [-x1, x0]`` within each adjacent pair."""
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_rotated = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        return x_rotated.reshape(*x.shape[:-2], -1)

    def apply_rotary_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        cos, sin = self._update_cache(seq_len, x.device, x.dtype)
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        return (x * cos) + (self.rotate_half(x) * sin)


class Yug_ResidualBlock(nn.Module):
    """Gated two-layer MLP with a linear skip: ``out(act(in(x))) + res(x)``.

    ``is_output_layer`` shrinks the initialisation of all three projections so
    an untrained head starts near zero rather than emitting a random explosion
    that the denormaliser would then scale by the series' standard deviation.
    """

    def __init__(
        self,
        in_features: int = 32,
        hidden_features: int = 1280,
        out_features: int = 1280,
        bias: bool = True,
        is_output_layer: bool = False,
    ) -> None:
        super().__init__()
        self.input_layer = nn.Linear(in_features, hidden_features, bias=bias)
        self.activation = nn.SiLU()
        self.output_layer = nn.Linear(hidden_features, out_features, bias=bias)
        self.residual_layer = nn.Linear(in_features, out_features, bias=bias)

        if is_output_layer:
            nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.input_layer.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.residual_layer.weight, mean=0.0, std=0.02)
            if bias:
                nn.init.zeros_(self.output_layer.bias)
                nn.init.zeros_(self.input_layer.bias)
                nn.init.zeros_(self.residual_layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        main = self.input_layer(x)
        main = self.activation(main)
        main = self.output_layer(main)
        residual = self.residual_layer(x)
        return main + residual


class Yug_FrequencyEmbedding(nn.Module):
    """Learned bucket embedding modulated by a sinusoidal encoding of the scale.

    The bucket ("hourly") is categorical and learned; the multiplier ("every
    15") is continuous and encoded sinusoidally on a log scale so that widely
    separated multipliers stay distinguishable. The two combine by Hadamard
    product, which lets the scale gate the bucket rather than merely offset it.
    """

    def __init__(self, freq_num: int = 9, out_features: int = 1280) -> None:
        super().__init__()
        # 9 buckets: secondly, minutely, hourly, daily, weekly, monthly,
        # quarterly, yearly, unknown.
        self.freq_embedding = nn.Embedding(freq_num, out_features)
        self.out_features = out_features

    def sinusoidal_encoding(self, positions, d_model: int = 1280, device="cuda"):
        if d_model % 2 != 0:
            d_model = d_model - 1

        angle_rates = 1 / torch.pow(
            10000.0,
            (2 * (torch.arange(d_model, dtype=torch.bfloat16, device=device) // 2))
            / float(d_model),
        )

        if isinstance(positions, (int, float)):
            positions = torch.tensor([positions], dtype=torch.bfloat16, device=device)
        else:
            positions = positions.to(dtype=torch.bfloat16, device=device)

        # log1p keeps a 1x and a 1440x multiplier on comparable footing.
        positions = torch.log1p(positions)

        angle_rads = positions.unsqueeze(1) * angle_rates.unsqueeze(0)

        pos_encoding = torch.zeros_like(angle_rads)
        pos_encoding[:, 0::2] = torch.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = torch.cos(angle_rads[:, 1::2])

        return pos_encoding

    def forward(self, freq_id, scale=1.0) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(freq_id, (int, float)):
            freq_id = torch.tensor([int(freq_id)], dtype=torch.int32, device=device)
        else:
            freq_id = freq_id.to(dtype=torch.int32, device=device)

        core_freq_emb = self.freq_embedding(freq_id)
        sinusoidal_emb = self.sinusoidal_encoding(
            scale, d_model=self.out_features, device=device
        ).to(core_freq_emb.dtype)

        combined = core_freq_emb * sinusoidal_emb

        if combined.shape[0] == 1:
            combined = combined.squeeze(0)
        return combined


class Yug_SelfAttention(nn.Module):
    """Pre-norm causal self-attention over patches, with RoPE.

    Accepts ``(B, S, D)`` or ``(B, V, S, D)``; the variate axis is folded into
    the batch before the SDPA call because the fused kernels require exactly
    four dimensions.
    """

    def __init__(
        self,
        num_attn_heads: int = 8,
        d_model: int = 1280,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert d_model % num_attn_heads == 0
        self.num_attn_heads = num_attn_heads
        self.d_model = d_model
        self.head_dim = d_model // num_attn_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.rope = Yug_RoPEEmbedding(dim=self.head_dim)
        # Applied to the attention output rather than inside SDPA: the fused
        # kernels reject a non-zero dropout_p.
        self.attn_dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.RMSNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor, attn_mask=None) -> torch.Tensor:
        univariate = False
        if x.dim() < 4:
            univariate = True
            batch_size, seq_len, d_model = x.size()
            x = x.view(batch_size, 1, seq_len, d_model)

        batch_size, num_variates, seq_len, d_model = x.size()
        x = self.layer_norm1(x)

        Q = (
            self.query(x)
            .view(batch_size, num_variates, seq_len, self.num_attn_heads, self.head_dim)
            .transpose(2, 3)
        )
        K = (
            self.key(x)
            .view(batch_size, num_variates, seq_len, self.num_attn_heads, self.head_dim)
            .transpose(2, 3)
        )
        V = (
            self.value(x)
            .view(batch_size, num_variates, seq_len, self.num_attn_heads, self.head_dim)
            .transpose(2, 3)
        )

        orig = Q.shape
        Q = Q.reshape(-1, seq_len, self.head_dim)
        K = K.reshape(-1, seq_len, self.head_dim)
        Q = self.rope.apply_rotary_pos_emb(Q)
        K = self.rope.apply_rotary_pos_emb(K)
        Q = Q.reshape(orig)
        K = K.reshape(orig)

        causal_bias = torch.zeros(seq_len, seq_len, device=x.device, dtype=Q.dtype)
        causal_bias.masked_fill_(
            ~torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)),
            float("-inf"),
        )
        attn_bias = causal_bias.view(1, 1, 1, seq_len, seq_len)

        if attn_mask is not None:
            if attn_mask.dim() == 3:  # (B, V, S)
                pad_bias = attn_mask.unsqueeze(2).unsqueeze(3)
            else:  # (B, S)
                pad_bias = attn_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)

            pad_bias_full = torch.zeros(
                batch_size,
                num_variates,
                self.num_attn_heads,
                seq_len,
                seq_len,
                device=x.device,
                dtype=Q.dtype,
            )
            pad_bias_full.masked_fill_(pad_bias == 0, float("-inf"))
            attn_bias = attn_bias + pad_bias_full

        b_v = batch_size * num_variates
        q_4d = Q.reshape(b_v, self.num_attn_heads, seq_len, self.head_dim)
        k_4d = K.reshape(b_v, self.num_attn_heads, seq_len, self.head_dim)
        v_4d = V.reshape(b_v, self.num_attn_heads, seq_len, self.head_dim)

        attn_bias_4d = attn_bias.expand(
            batch_size, num_variates, self.num_attn_heads, seq_len, seq_len
        ).reshape(b_v, self.num_attn_heads, seq_len, seq_len)

        attn_output = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=attn_bias_4d, dropout_p=0.0
        )

        attn_output = (
            attn_output.reshape(
                batch_size, num_variates, self.num_attn_heads, seq_len, self.head_dim
            )
            .transpose(2, 3)
            .contiguous()
            .view(batch_size, num_variates, seq_len, d_model)
        )
        attn_output = self.out(attn_output)
        attn_output = self.attn_dropout(attn_output)

        # nan_to_num before the output mask: a padding query whose entire
        # causal window is also padding gives softmax([-inf, ...]) = NaN. That
        # position is zeroed by the mask anyway, so NaN -> 0 is correct — but
        # it has to happen before the multiply or the NaN survives it.
        if attn_mask is not None:
            if attn_mask.dim() == 3:
                output_mask = attn_mask.unsqueeze(-1)
            else:
                output_mask = attn_mask.unsqueeze(1).unsqueeze(-1)
            attn_output = attn_output.nan_to_num(0.0) * output_mask.to(attn_output.dtype)

        return (
            attn_output.view(batch_size, seq_len, d_model) if univariate else attn_output
        )


class Yug_CrossAttention(nn.Module):
    """Target patches attend into the covariate stack, causally in time."""

    def __init__(
        self,
        num_attn_heads: int = 8,
        d_model: int = 1280,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert d_model % num_attn_heads == 0
        self.num_attn_heads = num_attn_heads
        self.d_model = d_model
        self.head_dim = d_model // num_attn_heads

        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm1 = nn.RMSNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.RMSNorm(d_model, eps=1e-6)
        self.attn_dropout = nn.Dropout(dropout)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=eps)

    def forward(self, q_mem, kv_mem, attn_mask_q=None, attn_mask_kv=None):
        batch_size, num_variates, seq_len_kv, d_model = kv_mem.size()
        batch_size, seq_len_q, _ = q_mem.size()

        q_mem = self.layer_norm1(q_mem)
        kv_mem = self.layer_norm2(kv_mem)

        Q = (
            self.query(q_mem)
            .view(batch_size, 1, seq_len_q, self.num_attn_heads, self.head_dim)
            .transpose(2, 3)
        )
        K = (
            self.key(kv_mem)
            .view(
                batch_size, num_variates, seq_len_kv, self.num_attn_heads, self.head_dim
            )
            .transpose(2, 3)
        )
        V = (
            self.value(kv_mem)
            .view(
                batch_size, num_variates, seq_len_kv, self.num_attn_heads, self.head_dim
            )
            .transpose(2, 3)
        )

        Q = self.q_norm(Q)
        K = self.k_norm(K)

        Q = Q.expand(
            batch_size, num_variates, self.num_attn_heads, seq_len_q, self.head_dim
        )

        attn_bias = torch.zeros(
            batch_size,
            num_variates,
            self.num_attn_heads,
            seq_len_q,
            seq_len_kv,
            device=q_mem.device,
            dtype=Q.dtype,
        )

        causal_mask = ~torch.tril(
            torch.ones(seq_len_q, seq_len_kv, device=q_mem.device, dtype=torch.bool)
        )
        attn_bias.masked_fill_(
            causal_mask.view(1, 1, 1, seq_len_q, seq_len_kv), float("-inf")
        )

        if attn_mask_q is not None and attn_mask_kv is not None:
            q_mask = attn_mask_q[:, None, None, :, None].bool()
            kv_mask = attn_mask_kv[:, :, None, None, :].bool()
            attn_bias.masked_fill_(~(q_mask & kv_mask), float("-inf"))
        elif attn_mask_kv is not None:
            kv_mask = attn_mask_kv[:, :, None, None, :].bool()
            attn_bias.masked_fill_(~kv_mask, float("-inf"))

        b_v = batch_size * num_variates
        q_4d = Q.reshape(b_v, self.num_attn_heads, seq_len_q, self.head_dim)
        k_4d = K.reshape(b_v, self.num_attn_heads, seq_len_kv, self.head_dim)
        v_4d = V.reshape(b_v, self.num_attn_heads, seq_len_kv, self.head_dim)
        attn_bias_4d = attn_bias.reshape(b_v, self.num_attn_heads, seq_len_q, seq_len_kv)

        attn_output = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=attn_bias_4d, dropout_p=0.0
        )

        attn_output = (
            attn_output.reshape(
                batch_size, num_variates, self.num_attn_heads, seq_len_q, self.head_dim
            )
            .transpose(2, 3)
            .contiguous()
            .view(batch_size, num_variates, seq_len_q, d_model)
        )
        attn_output = self.out(attn_output)
        attn_output = self.attn_dropout(attn_output)

        if attn_mask_q is not None:
            output_mask = attn_mask_q[:, None, :, None]
            attn_output = attn_output.nan_to_num(0.0) * output_mask.to(attn_output.dtype)

        return attn_output


class Yug_CrossVariate_Attention(nn.Module):
    """Attention *across* variates at each timestep, not across time.

    The variate axis is permuted into the sequence position so each timestep
    independently mixes its channels; time stays a batch dimension.
    """

    def __init__(
        self, hidden_dim: int = 1280, num_heads: int = 8, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.o_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, variate_attn_mask=None) -> torch.Tensor:
        batch, num_vars, seq_len, hidden_dim = x.shape

        x_collapsed = x.permute(0, 2, 1, 3)  # (B, S, V, D)
        x_collapsed = self.norm1(x_collapsed)

        q = self.q_proj(x_collapsed)
        k = self.k_proj(x_collapsed)
        v = self.v_proj(x_collapsed)

        q = q.reshape(batch, seq_len, num_vars, self.num_heads, self.head_dim).transpose(
            2, 3
        )
        k = k.reshape(batch, seq_len, num_vars, self.num_heads, self.head_dim).transpose(
            2, 3
        )
        v = v.reshape(batch, seq_len, num_vars, self.num_heads, self.head_dim).transpose(
            2, 3
        )

        attn_bias = None
        if variate_attn_mask is not None:
            # mask == 0 means a padding variate -> -inf.
            mask = variate_attn_mask.permute(0, 2, 1).unsqueeze(2).unsqueeze(3)
            attn_bias = torch.zeros(
                batch,
                seq_len,
                self.num_heads,
                num_vars,
                num_vars,
                device=x.device,
                dtype=q.dtype,
            )
            attn_bias.masked_fill_(mask == 0, float("-inf"))

        b_s = batch * seq_len
        q_4d = q.reshape(b_s, self.num_heads, num_vars, self.head_dim)
        k_4d = k.reshape(b_s, self.num_heads, num_vars, self.head_dim)
        v_4d = v.reshape(b_s, self.num_heads, num_vars, self.head_dim)
        attn_bias_4d = (
            attn_bias.reshape(b_s, self.num_heads, num_vars, num_vars)
            if attn_bias is not None
            else None
        )

        attn_output = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=attn_bias_4d, dropout_p=0.0
        )

        attn_output = (
            attn_output.reshape(batch, seq_len, self.num_heads, num_vars, self.head_dim)
            .transpose(2, 3)
            .contiguous()
            .view(batch, seq_len, num_vars, hidden_dim)
            .permute(0, 2, 1, 3)
        )
        # If every variate at a position is masked the softmax is NaN.
        attn_output = attn_output.nan_to_num(0.0)
        attn_output = self.attn_dropout(self.o_proj(attn_output))
        return attn_output


class Yug_MLP(nn.Module):
    """SwiGLU feed-forward block.

    ``down_proj`` is initialised at ``0.02 / sqrt(2 * num_decoder_layers)`` so
    the summed residual variance stays roughly constant as depth grows.
    """

    def __init__(
        self,
        hidden_size: int = 1280,
        eps: float = 1e-6,
        expansion: float = 1,
        num_decoder_layers: int = 8,
    ) -> None:
        super().__init__()
        # expansion can be fractional (e.g. 1.25) for some checkpoints; nn.Linear
        # needs int dims, so round rather than pass the float straight through.
        expanded_size = round(hidden_size * expansion)
        self.gate_proj = nn.Linear(hidden_size, expanded_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, expanded_size, bias=False)
        self.down_proj = nn.Linear(expanded_size, hidden_size, bias=False)

        std = 0.02 / math.sqrt(2 * num_decoder_layers)
        nn.init.normal_(self.down_proj.weight, mean=0.0, std=std)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, paddings=None) -> torch.Tensor:
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        if paddings is not None:
            x = x * (1.0 - paddings[..., None])
        return x


class Yug_Variate_MLP(nn.Module):
    """Compresses the covariate stack into one vector per timestep.

    Several learned latent queries pool over the variate axis, so different
    queries can specialise on different channel combinations instead of forcing
    everything through a single softmax bottleneck. A ReZero gate
    (``out_gate``, initialised to zero) means the block starts as a plain
    masked mean and only earns influence during training.
    """

    def __init__(
        self,
        hidden_size: int = 1280,
        rank: int = 128,
        num_heads: int = 8,
        num_latent: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert rank % num_heads == 0, "rank must be divisible by num_heads"
        self.rank = rank
        self.num_heads = num_heads
        self.num_latent = num_latent
        self.head_dim = rank // num_heads

        self.latent_q = nn.Parameter(torch.randn(1, num_heads, num_latent, self.head_dim))
        nn.init.trunc_normal_(self.latent_q, std=0.02)

        self.k_proj = nn.Linear(hidden_size, rank, bias=False)
        self.v_proj = nn.Linear(hidden_size, rank, bias=False)
        self.out_proj = nn.Linear(rank * num_latent, hidden_size, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.pre_norm = nn.RMSNorm(hidden_size, eps=eps, elementwise_affine=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=eps, elementwise_affine=False)

        self.out_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x, x_variate_cross_attn, attn_mask=None):
        batch, num_vars, seq_len, hidden_size = x.shape
        x_normed = self.pre_norm(x)

        if attn_mask is not None:
            m = attn_mask.permute(0, 2, 1).unsqueeze(-1).to(x_variate_cross_attn.dtype)
            cross_perm = x_variate_cross_attn.permute(0, 2, 1, 3)
            denom = m.sum(dim=2).clamp(min=1.0)
            pooled_residual = (cross_perm * m).sum(dim=2) / denom
        else:
            pooled_residual = x_variate_cross_attn.mean(dim=1)

        K = self.k_proj(x_normed).permute(0, 2, 1, 3)
        V = self.v_proj(x_variate_cross_attn).permute(0, 2, 1, 3)

        b_s = batch * seq_len
        k_4d = (
            K.reshape(b_s, num_vars, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        v_4d = (
            V.reshape(b_s, num_vars, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )
        q_4d = self.q_norm(self.latent_q).expand(b_s, -1, -1, -1)

        mask_4d = None
        if attn_mask is not None:
            # (B*S, 1, 1, V) broadcasts over every latent query.
            mask_4d = attn_mask.permute(0, 2, 1).bool().reshape(b_s, 1, 1, num_vars)

        attn_out = F.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, attn_mask=mask_4d, is_causal=False, dropout_p=0.0
        )

        out = attn_out.transpose(1, 2).contiguous()
        out = out.reshape(batch, seq_len, self.num_latent * self.rank)

        out = self.out_proj(out)
        out = self.attn_dropout(out)
        return pooled_residual + self.out_gate * out


class Yug_DecoderLayer(nn.Module):
    """One decoder block. The covariate path is skipped entirely when
    ``variates is None``, so univariate inference never touches it."""

    def __init__(
        self,
        num_attn_heads: int = 8,
        d_model: int = 1280,
        dropout: float = 0.1,
        eps: float = 1e-6,
        kernel_size: int = 3,
        num_decoder_layers: int = 8,
        expansion: float = 1,
    ) -> None:
        super().__init__()

        self.self_attn_tar = Yug_SelfAttention(
            num_attn_heads=num_attn_heads, d_model=d_model, dropout=dropout
        )
        self.self_attn_var = Yug_SelfAttention(
            num_attn_heads=num_attn_heads, d_model=d_model, dropout=dropout
        )
        self.cross_attn_1 = Yug_CrossAttention(
            num_attn_heads=num_attn_heads, d_model=d_model, dropout=dropout
        )
        self.cross_variate_attention = Yug_CrossVariate_Attention(
            hidden_dim=d_model, num_heads=num_attn_heads, dropout=dropout
        )
        self.variate_mlp = Yug_Variate_MLP(hidden_size=d_model, eps=eps)
        self.mlp = Yug_MLP(
            hidden_size=d_model,
            eps=eps,
            num_decoder_layers=num_decoder_layers,
            expansion=expansion,
        )
        self.inject_dropout = nn.Dropout(dropout)

        self.variate_inject_norm = nn.RMSNorm(d_model, eps=eps, elementwise_affine=False)
        self.output_tar_norm = nn.RMSNorm(d_model, eps=eps, elementwise_affine=False)

        self.variate_ffn_norm = nn.RMSNorm(d_model, eps=eps)
        self.variate_ffn = Yug_MLP(
            hidden_size=d_model,
            num_decoder_layers=num_decoder_layers,
            expansion=expansion,
        )
        self.variate_ffn_drop = nn.Dropout(dropout)

    def forward(
        self, target, variates=None, attn_mask_target=None, attn_mask_variates=None
    ):
        residual = target
        target = self.self_attn_tar(target, attn_mask=attn_mask_target)
        target = residual + target

        if variates is not None:
            variates = variates + self.self_attn_var(
                variates, attn_mask=attn_mask_variates
            )
            cross_variates = variates + self.cross_variate_attention(
                variates, variate_attn_mask=attn_mask_variates
            )
            variates = variates + self.cross_attn_1(
                target,
                variates,
                attn_mask_q=attn_mask_target,
                attn_mask_kv=attn_mask_variates,
            )

            variates_compressed = self.variate_mlp(
                variates,
                x_variate_cross_attn=cross_variates,
                attn_mask=attn_mask_variates,
            )
            variates_compressed = self.variate_inject_norm(variates_compressed)

            var_residual = variates
            variates = self.variate_ffn_norm(variates)
            variates = var_residual + self.variate_ffn_drop(self.variate_ffn(variates))

            target = target + self.inject_dropout(variates_compressed)

        residual_for_mlp = target
        target = self.output_tar_norm(target)
        target = residual_for_mlp + self.mlp(target)

        return target, variates


class CausalReversibleInstanceNorm(nn.Module):
    """Reversible instance norm whose statistics are causal.

    A standard RevIn normalises by the whole window's mean and variance, which
    leaks future information into every position. Here a vectorised Welford
    scan over the flattened ``(T, P)`` axis gives patch ``t`` the statistics of
    everything at or before ``t`` and nothing after — so the same module is
    valid for training, for a full forward pass, and for an incremental rollout
    that appends patches one block at a time.

    Masked positions are zeroed after normalisation. Without that, a padding
    sentinel normalises to an enormous value and reaches the network as a real
    signal.
    """

    def __init__(self, eps: float = 1e-4, affine: bool = False) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, 1))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, 1))

    def _expand_mask(self, mask, x):
        if mask.ndim == x.ndim:
            return mask.float()
        if mask.ndim == x.ndim - 1:
            return mask.unsqueeze(-1).expand_as(x).float()
        raise ValueError(f"mask.ndim={mask.ndim} incompatible with x.ndim={x.ndim}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        is_variate = x.ndim == 4
        if is_variate:
            b, v, t, p = x.shape
            x = x.reshape(b * v, t, p)
            if mask is not None:
                mask = mask.reshape(b * v, *mask.shape[2:])

        b_, t, p = x.shape
        orig_dtype = x.dtype
        x_fp32 = x.float()

        if mask is None:
            mask = torch.ones(b_, t, p, device=x.device, dtype=torch.float32)

        mask_expanded = self._expand_mask(mask, x_fp32)
        x_flat = torch.where(mask_expanded.bool(), x_fp32, torch.zeros_like(x_fp32)).view(
            b_, -1
        )
        mask_f = mask_expanded.reshape(b_, -1)

        cum_w = torch.cumsum(mask_f, 1)
        cum_v = torch.cumsum(x_flat, 1)
        safe_n = cum_w.clamp(min=1.0)
        cm = cum_v / safe_n

        sm = torch.zeros_like(cm)
        sm[:, 1:] = cm[:, :-1]
        m2 = torch.cumsum((x_flat - sm) * (x_flat - cm) * mask_f, 1)
        c_std = torch.sqrt(m2 / safe_n + self.eps)

        # One statistic per patch: the running value at the patch's last element.
        mean_fp32 = cm.view(b_, t, p)[:, :, -1:]
        std_fp32 = c_std.view(b_, t, p)[:, :, -1:]

        x_norm_fp32 = (x_fp32 - mean_fp32) / std_fp32
        x_norm_fp32 = x_norm_fp32 * mask_expanded
        x_norm = x_norm_fp32.to(orig_dtype)

        if self.affine:
            x_norm = x_norm * self.affine_weight + self.affine_bias

        if is_variate:
            x_norm = x_norm.reshape(b, v, t, p)
            mean_fp32 = mean_fp32.reshape(b, v, t, 1)
            std_fp32 = std_fp32.reshape(b, v, t, 1)

        return x_norm, mean_fp32, std_fp32

    def denormalize(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        # Restore the raw data scale in fp32: a valid forecast can exceed
        # fp16's range even when every normalised activation is representable.
        x = x.float()
        if self.affine:
            x = (x - self.affine_bias.float()) / self.affine_weight.float()
        if x.ndim == 5:
            b, t, k, q, d = x.shape
            idx = (
                torch.arange(t, device=mean.device).unsqueeze(1)
                + torch.arange(k, device=mean.device).unsqueeze(0)
            ).clamp(max=t - 1)
            mean = mean[:, idx, :].unsqueeze(-1)
            std = std[:, idx, :].unsqueeze(-1)
        else:
            for _ in range(x.ndim - mean.ndim):
                mean = mean.unsqueeze(-1)
                std = std.unsqueeze(-1)
        return x * std.float() + mean.float()


class Yug_Model(nn.Module):
    """The full forecaster.

    ``forward`` takes patched input and returns ``(predictions, mean, std)``,
    where ``predictions`` is ``(B, T, output_patches * n_quantiles * patch_len)``
    — a full forecast block anchored at *every* input patch. Rollout code uses
    only the last one; training uses all of them.

    ``num_input_patches`` marks the boundary between observed history and
    already-generated patches. Positions beyond it stay valid attention tokens
    but are excluded from the normalisation statistics, so a rollout's own
    output cannot drift the scale it is being normalised against.
    """

    def __init__(self, config: Mapping[str, Any] | Any) -> None:
        super().__init__()

        # Accept a YugConfig or the plain dict the reference scripts use.
        if hasattr(config, "to_arch_dict"):
            config = config.to_arch_dict()
        config = dict(config)
        self.config = config

        self.input_layer = Yug_ResidualBlock(
            in_features=config["patch_len"] * 2,
            hidden_features=config["hidden_dim"],
            out_features=config["d_model"],
            bias=True,
        )
        self.freq_embedding = Yug_FrequencyEmbedding(
            freq_num=config["freq_num"], out_features=config["d_model"]
        )
        self.decoder_layers = nn.ModuleList(
            [
                Yug_DecoderLayer(
                    num_attn_heads=config["num_attn_heads"],
                    d_model=config["d_model"],
                    dropout=config["dropout"],
                    eps=config["eps"],
                    num_decoder_layers=config["num_decoder_layers"],
                    expansion=config.get("expansion", 1),
                )
                for _ in range(config["num_decoder_layers"])
            ]
        )

        self.output_layer = Yug_ResidualBlock(
            in_features=config["d_model"],
            hidden_features=config["hidden_dim"],
            out_features=config["output_dim"] * len(config["quantiles"]),
            bias=True,
            is_output_layer=True,
        )

        self.input_norm_tar = nn.RMSNorm(config["d_model"], eps=config.get("eps", 1e-5))
        self.input_norm_var = nn.RMSNorm(config["d_model"], eps=config.get("eps", 1e-5))
        self.output_norm = nn.RMSNorm(config["d_model"], eps=config.get("eps", 1e-5))

        # Retained for checkpoint compatibility: this module carries trained
        # parameters that exist in published state dicts, but `forward` does not
        # currently call it. Removing it would make `load_state_dict` reject
        # every released checkpoint. Do not delete without a format bump.
        self.output_patch_norm = nn.LayerNorm(
            config["patch_len"], eps=config.get("eps", 1e-5)
        )

        self._output_patches = config["output_dim"] // config["patch_len"]
        self._num_quantiles = len(config["quantiles"])
        self._patch_len = config["patch_len"]

        self.CausalRevIn = CausalReversibleInstanceNorm(eps=config["eps"])

    def forward(
        self,
        target: torch.Tensor,
        variates: torch.Tensor | None = None,
        attn_mask_target: torch.Tensor | None = None,
        attn_mask_variates: torch.Tensor | None = None,
        freq_id=8,
        scale=1,
        num_input_patches: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if attn_mask_target is None:
            attn_mask_target = torch.ones_like(target)

        # The statistics mask is different: it stops at the observed history.
        if num_input_patches is not None:
            if isinstance(num_input_patches, torch.Tensor):
                cutoffs = num_input_patches.to(target.device).reshape(-1, 1, 1)
                positions = torch.arange(target.shape[-2], device=target.device).reshape(
                    1, -1, 1
                )
                stats_mask_target = attn_mask_target * (positions < cutoffs)
            else:
                stats_mask_target = attn_mask_target.clone()
                stats_mask_target[:, num_input_patches:, :] = 0
        else:
            stats_mask_target = attn_mask_target

        target, target_mean, target_std = self.CausalRevIn(target, stats_mask_target)
        target = target.to(self.input_layer.input_layer.weight.dtype)
        # Every patch — observed or generated — is a valid attention token, so
        # the decoder mask uses the original mask untouched.
        validity_target = attn_mask_target.to(dtype=target.dtype)
        target = torch.cat([target, validity_target], dim=-1)

        target = self.input_layer(target)

        freq_emb = self.freq_embedding(freq_id, scale=scale)
        if freq_emb.dim() == 1:
            freq_emb = freq_emb.unsqueeze(0)
        freq_emb = freq_emb.unsqueeze(1)  # (B, 1, d_model)

        target = target + freq_emb
        target = self.input_norm_tar(target)

        patch_attn_mask_target = attn_mask_target.sum(dim=-1) > 0

        if variates is not None:
            if attn_mask_variates is None:
                attn_mask_variates = torch.ones_like(variates)

            # Same history/generated split as the target: in multivariate
            # rollout the covariate channels are extended alongside it.
            if num_input_patches is not None:
                if isinstance(num_input_patches, torch.Tensor):
                    cutoffs = num_input_patches.to(variates.device).reshape(-1, 1, 1, 1)
                    positions = torch.arange(
                        variates.shape[-2], device=variates.device
                    ).reshape(1, 1, -1, 1)
                    stats_mask_variates = attn_mask_variates * (positions < cutoffs)
                else:
                    stats_mask_variates = attn_mask_variates.clone()
                    stats_mask_variates[:, :, num_input_patches:, :] = 0
            else:
                stats_mask_variates = attn_mask_variates

            variates, variates_mean, variates_std = self.CausalRevIn(
                variates, stats_mask_variates
            )
            variates = variates.to(self.input_layer.input_layer.weight.dtype)
            validity_variates = attn_mask_variates.to(dtype=variates.dtype)
            variates = torch.cat([variates, validity_variates], dim=-1)
            variates = self.input_layer(variates)

            freq_emb_var = freq_emb.unsqueeze(1)  # (B, 1, 1, d_model)
            variates = variates + freq_emb_var
            variates = self.input_norm_var(variates)

            attn_mask_variates = attn_mask_variates.sum(dim=-1) > 0

        for layer in self.decoder_layers:
            target, variates = layer(
                target,
                variates,
                attn_mask_target=patch_attn_mask_target,
                attn_mask_variates=attn_mask_variates,
            )

        target = self.output_norm(target)
        target = self.output_layer(target)
        target = self.CausalRevIn.denormalize(target, target_mean, target_std)
        return target, target_mean, target_std
