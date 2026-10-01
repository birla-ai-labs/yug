# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: frequency encoding, patch padding, and quantile sampling.

Everything here is pure and independently testable — no model, no device, no
network. The interesting piece is :class:`QuantileSampler`, which turns the
model's quantile head into sampled trajectories.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "normalize_freq",
    "freq_id",
    "pad_to_patch_multiple",
    "patchify",
    "QuantileSampler",
    "FREQ_CORE",
]

# The nine frequency buckets the embedding table was trained with.
FREQ_CORE = {
    "S": 0,  # secondly
    "T": 1,  # minutely
    "H": 2,  # hourly
    "D": 3,  # daily
    "W": 4,  # weekly
    "M": 5,  # monthly
    "Q": 6,  # quarterly
    "A": 7,  # annually
    "U": 8,  # unknown / other
}

# pandas >= 2.2 renamed several offset aliases. Map them back onto the codes the
# model was trained on so a user passing a modern freq string is not silently
# bucketed as "unknown".
_PANDAS_TO_MODEL = {
    "MIN": "T",
    "ME": "M",
    "MS": "M",
    "QE": "Q",
    "QS": "Q",
    "YE": "A",
    "YS": "A",
    "AE": "A",
    "AS": "A",
    "BA": "A",
    "BAS": "A",
}


def normalize_freq(freq: str | None) -> str:
    """Map a pandas offset alias onto the model's frequency vocabulary.

    ``"15MIN" -> "15T"``, ``"ME" -> "M"``, ``"W-SUN" -> "W"``. Anything
    unrecognised is passed through upper-cased and will land in the "unknown"
    bucket at :func:`freq_id` time.
    """
    if not isinstance(freq, str) or not freq:
        return "D"
    freq = freq.split("-")[0]
    match = re.fullmatch(r"(\d+)?(.*)", freq)
    scale_str, core = match.groups() if match else ("", freq)
    core_mapped = _PANDAS_TO_MODEL.get(core.upper(), core.upper())
    return f"{scale_str}{core_mapped}" if scale_str else core_mapped


def freq_id(freq: str) -> list[int]:
    """``"15T" -> [1, 15]``: the embedding index and its integer multiplier.

    Unparseable input falls back to ``[8, 1]`` — the "unknown" bucket at scale
    one — rather than raising, so a stray frequency degrades the forecast
    instead of aborting a sweep.
    """
    if not isinstance(freq, str):
        return [8, 1]
    base = freq.split("-")[0]
    match = re.fullmatch(r"(\d+)?([A-Z]+)", base)
    if not match:
        return [8, 1]
    scale_str, core = match.groups()
    scale = int(scale_str) if scale_str else 1
    if core == "MS":
        core = "M"
    return [FREQ_CORE.get(core, 8), scale]


def pad_to_patch_multiple(
    target: np.ndarray, patch_len: int, side: str = "left"
) -> tuple[np.ndarray, int]:
    """Grow the last axis to a whole number of patches using NaN sentinels.

    No observation is ever discarded. The NaNs become ``False`` in the validity
    mask at :func:`patchify` time, so padded positions are excluded from both
    attention and the running normalisation statistics.

    Returns ``(padded, pad_len)``; ``pad_len`` is 0 when no padding was needed.
    """
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    seq_len = target.shape[-1]
    rem = seq_len % patch_len
    if rem == 0:
        return target, 0

    pad_len = patch_len - rem
    pad_width = [(0, 0)] * target.ndim
    pad_width[-1] = (pad_len, 0) if side == "left" else (0, pad_len)

    padded = np.pad(
        target.astype(np.float32), pad_width, mode="constant", constant_values=np.nan
    )
    return padded, pad_len


def patchify(arr: np.ndarray, patch_len: int) -> tuple[np.ndarray, np.ndarray]:
    """``(N, T)`` float array -> ``(values, mask)`` shaped ``(N, T // patch_len, patch_len)``.

    NaN becomes 0.0 in the values and ``False`` in the mask. ``T`` must already
    be a multiple of ``patch_len`` (see :func:`pad_to_patch_multiple`).
    """
    if arr.shape[-1] % patch_len:
        raise ValueError(
            f"sequence length {arr.shape[-1]} is not a multiple of patch_len {patch_len}; "
            f"call pad_to_patch_multiple first"
        )
    n_patches = arr.shape[-1] // patch_len
    nan_mask = np.isnan(arr)
    values = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    n = arr.shape[0]
    return (
        np.ascontiguousarray(values.reshape(n, n_patches, patch_len)),
        np.ascontiguousarray((~nan_mask).reshape(n, n_patches, patch_len)),
    )


class QuantileSampler:
    """Draws trajectories from the model's predicted quantile band.

    The head emits a handful of quantile levels per step. To roll the model
    forward autoregressively we need *samples*, not a band, so each path draws
    one uniform ``u`` per block and reads the value at that level:

    * ``u`` inside ``[q_min, q_max]`` — piecewise-linear interpolation between
      the two bracketing predicted quantiles (inverse-transform sampling).
    * ``u`` outside that range — a Gaussian tail, scaled so the predicted
      ``q_min``/``q_max`` spread matches the normal's, centred on the median.

    One uniform is shared across every step within a block, which keeps a drawn
    path smooth instead of jittering between the quantile levels at each step.

    Known discontinuity at the band edge
    ------------------------------------
    The Gaussian tail is fitted to the band's *width* in normal units, not
    pinned to its endpoints, so the two branches only meet where the predicted
    band happens to be Gaussian. For a skewed or heavy-tailed band the sampled
    value jumps at ``u = q_min`` and ``u = q_max`` and is not monotone in ``u``
    across those points. With the released levels that affects the
    ``2 * q_min`` fraction of draws that land in a tail (20% at 0.1/0.9), so it
    shows up in the extremes of the returned band rather than near the median.

    This is inherited from the reference implementation and is preserved here
    deliberately: changing it would silently move every previously published
    forecast. Treat it as a known limitation of the extreme quantiles, not as
    something to patch locally.

    Vectorised across paths and timesteps. The float64 promotions below are not
    incidental: they reproduce ``np.interp``'s internal precision, which makes
    this bit-for-bit identical to the scalar reference implementation.
    """

    def __init__(self, quantile_levels: Sequence[float]) -> None:
        levels = [float(q) for q in quantile_levels]
        if list(levels) != sorted(levels):
            raise ValueError(f"quantile_levels must be ascending, got {levels}")
        if len(levels) < 2:
            raise ValueError("need at least two quantile levels to sample")

        self.levels = np.asarray(levels, dtype=np.float64)
        self.median_idx = int(np.argmin(np.abs(self.levels - 0.5)))
        # Width of the predicted band measured in standard normal units, used
        # to convert the band into a sigma for the tail draws.
        #
        # Deliberately left as np.float64 rather than a Python float: under
        # NEP 50 a float32 array divided by a Python float stays float32, but
        # divided by an np.float64 it promotes to float64. The reference
        # implementation promotes, and the tail values differ in the last bits
        # if we do not. Do not "simplify" this with float().
        self.ppf_span = norm.ppf(self.levels[-1]) - norm.ppf(self.levels[0])

    def __call__(self, qf: np.ndarray, us: np.ndarray) -> np.ndarray:
        """``qf`` ``(N, Q, out)`` ascending in Q, ``us`` ``(N,)`` -> ``(N, out)``."""
        n, n_q, out = qf.shape
        if us.shape != (n,):
            raise ValueError(f"expected us of shape ({n},), got {us.shape}")
        if n_q != self.levels.size:
            raise ValueError(
                f"forecast has {n_q} quantiles but sampler was built for "
                f"{self.levels.size}"
            )

        q = self.levels
        lo, hi = float(q[0]), float(q[-1])
        res = np.empty((n, out), dtype=np.float32)

        interior = (us >= lo) & (us <= hi)

        idx = np.nonzero(interior)[0]
        if idx.size:
            u = us[idx]
            qf64 = qf[idx].astype(np.float64)
            j = np.searchsorted(q, u, side="right") - 1
            # np.interp returns the endpoint verbatim at x == xp[-1] rather
            # than extrapolating from the final interval; mirror that.
            at_hi = j >= n_q - 1
            j = np.clip(j, 0, n_q - 2)
            rows = np.arange(idx.size)
            a = qf64[rows, j]
            b = qf64[rows, j + 1]
            slope = (b - a) / (q[j + 1] - q[j])[:, None]
            vals = slope * (u - q[j])[:, None] + a
            if at_hi.any():
                vals[at_hi] = qf64[rows[at_hi], n_q - 1]
            res[idx] = vals.astype(np.float32)

        idx = np.nonzero(~interior)[0]
        if idx.size:
            median = qf[idx, self.median_idx]
            sigma = (qf[idx, -1] - qf[idx, 0]) / self.ppf_span
            res[idx] = (median + norm.ppf(us[idx])[:, None] * sigma).astype(np.float32)

        return res
