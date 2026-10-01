# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Prefix-KV-cached incremental decoder for the univariate rollout.

Why it exists
-------------
A rollout draws ``num_samples`` trajectories over ``n_steps`` blocks. The naive
loop re-runs the whole model over the whole series for every path at every
step, then keeps exactly one patch of the result — the last. For a 2048-point
context (128 patches) and a 512-step horizon (8 blocks of 4 patches) that is
8 x 100 forward passes over ~144 patches, ~99% of which recomputes a prefix
that never changes.

Why it is valid
---------------
The univariate model is strictly causal at the patch level:

* :class:`CausalReversibleInstanceNorm` is a cumulative scan — patch ``t``'s
  statistics depend only on patches ``<= t``.
* ``input_layer``, ``input_norm_tar``, ``output_tar_norm``, ``mlp``,
  ``output_norm`` and ``output_layer`` are all position-wise.
* ``self_attn_tar`` is ``tril``-masked with RoPE on the *absolute* patch index,
  so appending patches cannot alter earlier positions.
* ``denormalize`` receives a 3-D tensor here, taking its per-patch branch.

So the context can be encoded once and its per-layer K/V reused by every path
at every step, with only the newly generated patches pushed through.

Memory
------
The context's K/V are kept once, at batch 1. Each path stores only the patches
it generated itself (at most ``(n_steps - 1) * output_patches`` of them).
Attention for an appended patch is one softmax over ``[shared context | own
tail]`` — the same attention as over the concatenated keys, without ever
materialising ``num_samples`` copies of the context. Memory therefore does not
scale with ``num_samples x context length``, which is what lets a very long
context (tens of thousands of patches) run at all.

Gaps
----
A patch with no observed value (an all-NaN gap) is legal input, handled the
way :meth:`Yug_Model.forward` handles it: it is never an attention key, and
its own attention output is zero (its residual stream still flows through the
MLP). Series with gaps therefore stay on this engine rather than falling back
to the exact path.

Numerical contract
------------------
This is *mathematically* identical to :meth:`Yug_Model.forward`, not
bitwise identical. Bitwise identity is unreachable for any reshaping of this
model: cuBLAS and the SDPA kernels pick different reduction orders per tensor
shape, so even the unmodified model returns different last bits at batch 1
versus batch 100 (~7e-6 apart). Deviations here sit at that same fp32 floor.
Use ``engine="exact"`` in the pipeline when you need the reference's exact
batch shapes reproduced bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["CachedUnivariateEngine"]


class _RevInState:
    """Running Welford quantities at a chunk boundary.

    :class:`CausalReversibleInstanceNorm` reduces over the flattened ``(T, P)``
    axis, so continuing the scan across a chunk boundary needs only these four
    values: total weight, total sum, running M2, and the last cumulative mean.
    """

    __slots__ = ("w", "v", "m2", "cm")

    def __init__(self, w, v, m2, cm) -> None:
        self.w, self.v, self.m2, self.cm = w, v, m2, cm

    def expand(self, n: int) -> _RevInState:
        """Broadcast a batch-1 state across ``n`` rollout paths."""
        return _RevInState(
            *(t.expand(n).contiguous() for t in (self.w, self.v, self.m2, self.cm))
        )


class CachedUnivariateEngine:
    """Incremental univariate decoder over a shared, cached context prefix.

    Usage::

        eng = CachedUnivariateEngine(model)
        qf = eng.begin(ctx, ctx_mask, freq_ids, scalings,
                       n_paths=100, max_patches=...)   # (1, Q, output_len)
        qf = eng.extend(new_patches)                   # (B, Q, output_len)

    One engine instance is bound to one context. Call :meth:`begin` again to
    start a new series; the RoPE table is reused across series.
    """

    def __init__(self, model) -> None:
        self.m = model
        cfg = model.config
        self.H = cfg["num_attn_heads"]
        self.D = cfg["d_model"]
        self.hd = self.D // self.H
        self.P = cfg["patch_len"]
        self.O = cfg["output_dim"] // cfg["patch_len"]
        self.Q = len(cfg["quantiles"])
        self.eps = model.CausalRevIn.eps
        self.layers = model.decoder_layers
        self.device = next(model.parameters()).device

        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._rope_len = 0

        # Populated by begin().
        # Per-layer K/V of the shared context, each (1, H, S0, hd).
        self._pk: list[torch.Tensor] = []
        self._pv: list[torch.Tensor] = []
        # Per-layer K/V each path generated itself, each (B, H, tail_max, hd).
        self._sk: list[torch.Tensor] = []
        self._sv: list[torch.Tensor] = []
        self._n_prefix = 0  # S0: patches in the shared context
        self._valid: torch.Tensor | None = None  # (S0,) context patches with data
        self._all_valid = True
        self._pos = 0
        self._freq_emb: torch.Tensor | None = None
        self._state: _RevInState | None = None

    # ------------------------------------------------------------------ RoPE
    def _rope(self, total_len: int, dtype: torch.dtype):
        """cos/sin for absolute positions ``0 .. total_len-1``.

        Reproduces ``Yug_RoPEEmbedding._update_cache`` exactly, including
        the ``repeat_interleave(2)`` pairing and building ``t`` in the
        activation dtype. Index ``i``'s value does not depend on ``total_len``,
        so one oversized table serves every chunk.
        """
        if (
            self._cos is not None
            and self._rope_len >= total_len
            and self._cos.dtype == dtype
        ):
            return self._cos, self._sin
        rope = self.layers[0].self_attn_tar.rope
        t = torch.arange(total_len, device=self.device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, rope.inv_freq.to(dtype))
        emb = freqs.repeat_interleave(2, dim=-1)
        self._cos, self._sin, self._rope_len = emb.cos(), emb.sin(), total_len
        return self._cos, self._sin

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_rot = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        return x_rot.reshape(*x.shape[:-2], -1)

    def _apply_rope(self, x: torch.Tensor, start: int, length: int) -> torch.Tensor:
        """``x`` is ``(B, H, S, hd)`` at absolute positions ``start .. start+S-1``."""
        cos, sin = self._rope(start + length, x.dtype)
        c = cos[start : start + length].view(1, 1, length, self.hd)
        s = sin[start : start + length].view(1, 1, length, self.hd)
        return (x * c) + (self._rotate_half(x) * s)

    # ----------------------------------------------------------------- RevIn
    def _revin(self, x: torch.Tensor, mask: torch.Tensor, state: _RevInState | None):
        """Causal Welford scan over one chunk, continuing ``state``.

        Mirrors ``CausalReversibleInstanceNorm.forward`` term for term; the only
        difference is that the running sums start from the previous chunk
        instead of from zero.
        """
        b, t, p = x.shape
        x32 = x.float()
        m32 = mask.float()

        x_flat = torch.where(m32.bool(), x32, torch.zeros_like(x32)).view(b, -1)
        mask_f = m32.reshape(b, -1)

        if state is None:
            cum_w = torch.cumsum(mask_f, 1)
            cum_v = torch.cumsum(x_flat, 1)
        else:
            cum_w = torch.cumsum(mask_f, 1) + state.w.unsqueeze(1)
            cum_v = torch.cumsum(x_flat, 1) + state.v.unsqueeze(1)

        safe_n = cum_w.clamp(min=1.0)
        cm = cum_v / safe_n

        sm = torch.empty_like(cm)
        sm[:, 0] = 0.0 if state is None else state.cm
        sm[:, 1:] = cm[:, :-1]

        m2 = torch.cumsum((x_flat - sm) * (x_flat - cm) * mask_f, 1)
        if state is not None:
            m2 = m2 + state.m2.unsqueeze(1)

        c_std = torch.sqrt(m2 / safe_n + self.eps)

        mean = cm.view(b, t, p)[:, :, -1:]
        std = c_std.view(b, t, p)[:, :, -1:]

        x_norm = ((x32 - mean) / std) * m32

        nxt = _RevInState(cum_w[:, -1], cum_v[:, -1], m2[:, -1], cm[:, -1])
        return x_norm.to(x.dtype), mean, std, nxt

    # --------------------------------------------------------------- encoder
    def _embed(
        self, patches: torch.Tensor, mask: torch.Tensor, state: _RevInState | None
    ):
        """``(B, S, patch_len)`` -> token embeddings ``(B, S, D)``."""
        validity = mask.to(dtype=patches.dtype)
        x, mean, std, nxt = self._revin(patches, mask, state)
        activation_dtype = self.m.input_layer.input_layer.weight.dtype
        x = x.to(activation_dtype)
        validity = validity.to(activation_dtype)
        x = torch.cat([x, validity], dim=-1)
        x = self.m.input_layer(x)
        x = x + self._freq_emb
        x = self.m.input_norm_tar(x)
        return x, mean, std, nxt

    def _attend_shared_prefix(
        self,
        li: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        start: int,
    ) -> torch.Tensor:
        """Attention for newly appended tokens over ``[shared context | own tail]``.

        ``q``, ``k``, ``v`` are ``(B, H, S, hd)`` at absolute positions
        ``start .. start+S-1``, with ``start >= S0``. The context K/V exist once
        (batch 1); each path keeps only its own generated tail. The softmax runs
        over the union of the two key sets, so this is the same attention as
        SDPA over the concatenated keys without materialising ``B`` copies of
        the context.

        Every new query sits after the whole context, so it sees all of it (bar
        empty patches); inside the tail the causal rule is applied by position.
        The ``B`` paths' queries are stacked along the query axis for the
        context part, so the context K/V are read once, not ``B`` times.
        """
        b, h, s, hd = q.shape
        s0 = self._n_prefix
        lo = start - s0  # where these tokens land in the tail
        n_tail = lo + s

        self._sk[li][:, :, lo:n_tail] = k
        self._sv[li][:, :, lo:n_tail] = v
        tk, tv = self._sk[li][:, :, :n_tail], self._sv[li][:, :, :n_tail]
        pk, pv = self._pk[li][0], self._pv[li][0]  # (H, S0, hd)

        scale = hd**-0.5
        qf = q.transpose(0, 1).reshape(h, b * s, hd)
        s_ctx = torch.matmul(qf, pk.transpose(-1, -2)) * scale  # (H, B*S, S0)
        if not self._all_valid:
            assert self._valid is not None
            # Empty patches are never keys.
            s_ctx = s_ctx.masked_fill(~self._valid, float("-inf"))

        qi = torch.arange(start, start + s, device=q.device).unsqueeze(1)
        ki = s0 + torch.arange(n_tail, device=q.device).unsqueeze(0)
        s_tail = (torch.matmul(q, tk.transpose(-1, -2)) * scale).masked_fill(
            ki > qi, float("-inf")
        )  # (B, H, S, n_tail)
        s_tail = s_tail.transpose(0, 1).reshape(h, b * s, n_tail)

        p = torch.softmax(
            torch.cat([s_ctx, s_tail], dim=-1), dim=-1, dtype=torch.float32
        ).to(q.dtype)  # (H, B*S, S0 + n_tail)
        out_ctx = torch.matmul(p[..., :s0], pv)  # (H, B*S, hd)
        out_tail = torch.matmul(
            p[..., s0:].reshape(h, b, s, n_tail).transpose(0, 1), tv
        )  # (B, H, S, hd)
        return out_ctx.reshape(h, b, s, hd).transpose(0, 1) + out_tail

    def _decode(self, x: torch.Tensor, start: int) -> torch.Tensor:
        """Run the decoder stack over ``x`` ``(B, S, D)`` at absolute offset ``start``.

        ``start == 0`` encodes the shared context (batch 1) and keeps its K/V.
        ``start > 0`` appends new tokens: their K/V go to each path's own tail,
        and they attend over the shared context plus that tail.
        """
        b, s, _ = x.shape

        for li, layer in enumerate(self.layers):
            sa = layer.self_attn_tar

            residual = x
            xn = sa.layer_norm1(x)

            q = sa.query(xn).view(b, s, self.H, self.hd).transpose(1, 2)
            k = sa.key(xn).view(b, s, self.H, self.hd).transpose(1, 2)
            v = sa.value(xn).view(b, s, self.H, self.hd).transpose(1, 2)

            q = self._apply_rope(q, start, s)
            k = self._apply_rope(k, start, s)

            if start == 0:
                self._pk[li], self._pv[li] = k.contiguous(), v.contiguous()
                if self._all_valid:
                    # Square and fully causal with every patch valid, so
                    # is_causal lets SDPA pick a flash kernel instead of
                    # materialising an (B, H, S, S) additive bias.
                    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    assert self._valid is not None
                    keep = torch.ones(s, s, dtype=torch.bool, device=x.device).tril_()
                    keep &= self._valid  # empty patches are never keys
                    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=keep)
                    # ...and an empty patch's own row emits nothing (a row with
                    # no valid key at all comes out NaN; the model zeroes those
                    # too).
                    attn = attn.nan_to_num(0.0) * self._valid.view(1, 1, s, 1).to(
                        attn.dtype
                    )
            else:
                attn = self._attend_shared_prefix(li, q, k, v, start)

            attn = attn.transpose(1, 2).contiguous().view(b, s, self.D)
            x = residual + sa.out(attn)

            r = x
            x = layer.output_tar_norm(x)
            x = r + layer.mlp(x)

        return x

    def _head(
        self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> torch.Tensor:
        """Final patch only -> ``(B, Q, output_patches * patch_len)``."""
        h = x[:, -1:]
        h = self.m.output_norm(h)
        h = self.m.output_layer(h)
        h = self.m.CausalRevIn.denormalize(h, mean[:, -1:], std[:, -1:])
        b = h.shape[0]
        h = h.view(b, self.O, self.Q, self.P)
        return h.permute(0, 2, 1, 3).reshape(b, self.Q, self.O * self.P)

    # ------------------------------------------------------------------- API
    def begin(
        self,
        patches: torch.Tensor,
        mask: torch.Tensor,
        freq_id: torch.Tensor,
        scale: torch.Tensor,
        n_paths: int,
        max_patches: int,
    ) -> torch.Tensor:
        """Encode the shared context once at batch 1 and seed the KV cache.

        Returns the context's final-patch forecast, ``(1, Q, output_len)``.
        Every path sees the same context at step 0, so one result is shared —
        the paths only diverge once they start sampling.
        """
        if patches.dim() != 3 or patches.shape[0] != 1:
            raise ValueError(
                f"begin() expects a single context of shape (1, S, patch_len), "
                f"got {tuple(patches.shape)}"
            )

        s = patches.shape[1]

        # Patches holding no observed value at all (interior gaps). Left-padding
        # only blanks part of patch 0, so the common case has none of them.
        self._valid = (mask.sum(-1) > 0)[0]  # (S,)
        self._all_valid = bool(self._valid.all())

        fe = self.m.freq_embedding(freq_id, scale=scale)
        if fe.dim() == 1:
            fe = fe.unsqueeze(0)
        self._freq_emb = fe.unsqueeze(1)  # (1, 1, D)

        nl = len(self.layers)

        # The context's K/V are stored once (batch 1, filled by the pass below);
        # only the patches generated during the rollout get a per-path cache.
        cache_dtype = self.m.input_layer.input_layer.weight.dtype
        self._n_prefix = s
        self._pk = [torch.empty(0)] * nl
        self._pv = [torch.empty(0)] * nl
        tail_max = max(max_patches - s, 0)
        self._sk = [
            torch.empty(
                n_paths, self.H, tail_max, self.hd, device=self.device, dtype=cache_dtype
            )
            for _ in range(nl)
        ]
        self._sv = [torch.empty_like(t) for t in self._sk]

        x, mean, std, state = self._embed(patches, mask, None)
        x = self._decode(x, start=0)

        self._pos = s
        self._state = state.expand(n_paths)
        return self._head(x, mean, std)

    def extend(self, patches: torch.Tensor) -> torch.Tensor:
        """Append ``(B, S, patch_len)`` newly generated patches and forecast.

        Generated patches are always fully observed. Returns the new
        final-patch forecast, ``(B, Q, output_len)``.
        """
        if self._state is None:
            raise RuntimeError("call begin() before extend()")
        b, s, _ = patches.shape
        mask = torch.ones_like(patches, dtype=torch.bool)
        x, mean, std, self._state = self._embed(patches, mask, self._state)
        x = self._decode(x, start=self._pos)
        self._pos += s
        return self._head(x, mean, std)
