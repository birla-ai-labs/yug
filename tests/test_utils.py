# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Utility correctness: frequency mapping, patch padding, quantile sampling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from yug.utils import (
    QuantileSampler,
    freq_id,
    normalize_freq,
    pad_to_patch_multiple,
    patchify,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("D", "D"),
        ("15MIN", "15T"),
        ("ME", "M"),  # pandas >= 2.2 spelling of month-end
        ("MS", "M"),
        ("QE", "Q"),
        ("YE", "A"),
        ("W-SUN", "W"),  # anchored offsets drop their anchor
        ("H", "H"),
        (None, "D"),  # missing frequency falls back to daily
        ("", "D"),
    ],
)
def test_normalize_freq(raw, expected):
    assert normalize_freq(raw) == expected


@pytest.mark.parametrize(
    "freq, expected",
    [
        ("D", [3, 1]),
        ("15T", [1, 15]),
        ("H", [2, 1]),
        ("M", [5, 1]),
        ("MS", [5, 1]),
        ("nonsense", [8, 1]),  # unknown -> the "other" bucket, never a raise
        (None, [8, 1]),
    ],
)
def test_freq_id(freq, expected):
    assert freq_id(freq) == expected


def test_padding_is_a_no_op_when_already_aligned():
    arr = np.zeros((1, 32), dtype=np.float32)
    padded, pad_len = pad_to_patch_multiple(arr, 16)
    assert pad_len == 0
    assert padded is arr


@pytest.mark.parametrize("side, nan_at", [("left", 0), ("right", -1)])
def test_padding_adds_nans_on_the_requested_side(side, nan_at):
    arr = np.arange(20, dtype=np.float32).reshape(1, 20)
    padded, pad_len = pad_to_patch_multiple(arr, 16, side)
    assert pad_len == 12
    assert padded.shape == (1, 32)
    assert np.isnan(padded[0, nan_at])
    # No observation is ever dropped.
    assert np.array_equal(padded[~np.isnan(padded)], arr.ravel())


def test_padding_rejects_an_unknown_side():
    with pytest.raises(ValueError, match="left"):
        pad_to_patch_multiple(np.zeros((1, 20), dtype=np.float32), 16, "middle")


def test_patchify_marks_nans_invalid_and_zeroes_them():
    arr = np.array([[1.0, np.nan, 3.0, 4.0]], dtype=np.float32)
    values, mask = patchify(arr, 2)
    assert values.shape == mask.shape == (1, 2, 2)
    assert values[0, 0, 1] == 0.0  # NaN zeroed in the values
    assert mask.tolist() == [[[True, False], [True, True]]]


def test_patchify_refuses_a_misaligned_series():
    with pytest.raises(ValueError, match="multiple of patch_len"):
        patchify(np.zeros((1, 20), dtype=np.float32), 16)


# --------------------------------------------------------------- the sampler
LEVELS = [0.1, 0.3, 0.5, 0.9]


def _reference_sample(qf_row, u, levels):
    """Scalar reference: what np.interp / scipy would give, one path at a time."""
    q = np.asarray(levels)
    if q[0] <= u <= q[-1]:
        return np.array(
            [np.interp(u, q, qf_row[:, t]) for t in range(qf_row.shape[1])],
            dtype=np.float32,
        )
    sigma = (qf_row[-1] - qf_row[0]) / (norm.ppf(q[-1]) - norm.ppf(q[0]))
    median = qf_row[int(np.argmin(np.abs(q - 0.5)))]
    return (median + norm.ppf(u) * sigma).astype(np.float32)


def test_sampler_is_bit_exact_against_the_scalar_reference():
    """The vectorised sampler must not merely be close.

    It replaced a per-timestep np.interp loop, so any last-bit drift would show
    up as an unexplained diff against previously published forecasts. The
    float64 promotions in QuantileSampler exist for exactly this reason.
    """
    rng = np.random.default_rng(0)
    sampler = QuantileSampler(LEVELS)

    for _ in range(50):
        n = int(rng.integers(1, 6))
        qf = np.sort(
            rng.normal(scale=rng.uniform(0.1, 50.0), size=(n, 4, 17)).astype(np.float32),
            axis=1,
        )
        us = rng.random(n)
        got = sampler(qf, us)
        expected = np.stack([_reference_sample(qf[i], us[i], LEVELS) for i in range(n)])
        assert np.array_equal(got, expected)


@pytest.mark.parametrize("u", LEVELS)
def test_sampler_is_exact_on_quantile_boundaries(u):
    """u landing exactly on a level must return that level, not extrapolate."""
    sampler = QuantileSampler(LEVELS)
    qf = np.sort(
        np.random.default_rng(1).normal(size=(1, 4, 9)).astype(np.float32), axis=1
    )
    got = sampler(qf, np.array([u]))
    level_idx = LEVELS.index(u)
    assert np.allclose(got[0], qf[0, level_idx], atol=1e-6)


def test_sampler_uses_gaussian_tails_beyond_the_predicted_band():
    """Draws outside [q_min, q_max] must extend past the band, not clamp to it."""
    sampler = QuantileSampler(LEVELS)
    qf = np.tile(np.array([-2.0, -1.0, 0.0, 2.0], dtype=np.float32)[:, None], (1, 1, 5))
    assert (sampler(qf, np.array([0.99]))[0] > qf[0, -1]).all()
    assert (sampler(qf, np.array([0.01]))[0] < qf[0, 0]).all()


def test_sampler_is_monotone_inside_the_predicted_band():
    """Within [q_min, q_max] a larger uniform must never give a smaller value."""
    sampler = QuantileSampler(LEVELS)
    qf = np.sort(
        np.random.default_rng(2).normal(size=(1, 4, 6)).astype(np.float32), axis=1
    )
    us = np.linspace(LEVELS[0], LEVELS[-1], 40)
    drawn = np.stack([sampler(qf, np.array([u]))[0] for u in us])
    assert np.all(np.diff(drawn, axis=0) >= -1e-6)


def test_sampler_is_monotone_within_each_tail():
    sampler = QuantileSampler(LEVELS)
    qf = np.sort(
        np.random.default_rng(3).normal(size=(1, 4, 6)).astype(np.float32), axis=1
    )
    for us in (
        np.linspace(0.001, LEVELS[0] - 1e-4, 20),
        np.linspace(LEVELS[-1] + 1e-4, 0.999, 20),
    ):
        drawn = np.stack([sampler(qf, np.array([u]))[0] for u in us])
        assert np.all(np.diff(drawn, axis=0) >= -1e-6)


def test_sampler_jumps_at_the_band_edge_for_a_skewed_band():
    """Documents a known limitation rather than asserting it is absent.

    The Gaussian tail matches the band's width, not its endpoints, so the two
    branches disagree at q_min/q_max whenever the band is not Gaussian. This
    is inherited from the reference implementation and is preserved on purpose;
    the test exists so that changing it can never happen silently.
    """
    sampler = QuantileSampler(LEVELS)
    # Tight core, very fat upper tail -> strongly non-Gaussian.
    qf = np.tile(np.array([-1.0, -0.9, 0.0, 10.0], dtype=np.float32)[:, None], (1, 1, 3))
    at_edge = sampler(qf, np.array([LEVELS[-1]]))[0]
    just_past = sampler(qf, np.array([LEVELS[-1] + 1e-4]))[0]
    assert np.all(just_past < at_edge)  # the documented downward jump
    assert np.allclose(at_edge, qf[0, -1])  # the edge itself is still exact


def test_sampler_validates_its_inputs():
    with pytest.raises(ValueError, match="ascending"):
        QuantileSampler([0.9, 0.1])
    with pytest.raises(ValueError, match="at least two"):
        QuantileSampler([0.5])

    sampler = QuantileSampler(LEVELS)
    qf = np.zeros((3, 4, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        sampler(qf, np.zeros(2))
    with pytest.raises(ValueError, match="quantiles"):
        sampler(np.zeros((3, 2, 5), dtype=np.float32), np.zeros(3))
