# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""End-to-end behaviour of the public entry point.

Runs against a staged local checkpoint, so the whole file works offline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from yug import QuantileForecast, YugConfig, YugPipeline
from yug.pipeline import _unit_scale_affine


# ----------------------------------------------------------------- loading
def test_from_pretrained_reads_a_hub_shaped_directory(pipeline, tiny_config):
    assert isinstance(pipeline, YugPipeline)
    assert pipeline.quantile_levels == tiny_config.quantiles
    assert pipeline.device.type == "cpu"


def test_save_pretrained_round_trips(pipeline, tmp_path, series):
    out = pipeline.save_pretrained(tmp_path / "ckpt")
    assert (out / "config.json").exists()
    assert (out / "model.safetensors").exists()

    reloaded = YugPipeline.from_pretrained(out, device_map="cpu")
    a = pipeline.predict(series, 32, num_samples=4, seed=0).values
    b = reloaded.predict(series, 32, num_samples=4, seed=0).values
    assert np.array_equal(a, b)


def test_missing_weights_fail_with_a_clear_message(tmp_path):
    (tmp_path / "config.json").write_text(YugConfig().to_dict().__repr__())
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        YugPipeline.from_pretrained(tmp_path, device_map="cpu")


def test_a_mismatched_config_is_rejected_rather_than_silently_loaded(
    staged_checkpoint, tiny_config
):
    """Loading wider weights under a narrower config must not half-succeed."""
    wrong = YugConfig(**{**tiny_config.to_arch_dict(), "num_decoder_layers": 4})
    with pytest.raises(RuntimeError, match="missing"):
        YugPipeline.from_pretrained(
            staged_checkpoint, device_map="cpu", model_config=wrong
        )


# ----------------------------------------------------------------- predict
@pytest.mark.parametrize("horizon", [1, 8, 16, 17, 64])
def test_horizon_is_respected_exactly(pipeline, series, horizon):
    """Including horizons that are not a multiple of the model's block size."""
    forecast = pipeline.predict(series, horizon, num_samples=4, seed=0)
    assert forecast.values.shape == (1, 3, horizon)
    assert np.isfinite(forecast.values).all()


@pytest.mark.parametrize(
    "make_context, expected",
    [
        (lambda s: s, 1),  # 1-D array
        (lambda s: np.stack([s, s]), 2),  # 2-D array, one series per row
        (lambda s: [s, s[:128], s[:100]], 3),  # ragged list
        (lambda s: list(map(float, s[:64])), 1),  # plain Python list of floats
        (lambda s: torch.from_numpy(s), 1),  # torch tensor
    ],
)
def test_accepted_context_shapes(pipeline, series, make_context, expected):
    forecast = pipeline.predict(make_context(series), 16, num_samples=4, seed=0)
    assert forecast.values.shape[0] == expected


def test_quantile_band_is_ordered(pipeline, series):
    """q0.1 <= q0.5 <= q0.9 at every step — the band must never cross."""
    values = pipeline.predict(series, 48, num_samples=32, seed=0).values
    assert (np.diff(values, axis=1) >= -1e-5).all()


def test_same_seed_is_reproducible_and_different_seeds_are_not(pipeline, series):
    a = pipeline.predict(series, 32, num_samples=8, seed=42).values
    b = pipeline.predict(series, 32, num_samples=8, seed=42).values
    c = pipeline.predict(series, 32, num_samples=8, seed=43).values
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_cached_and_exact_engines_agree(pipeline, series):
    """The cached engine is mathematically equivalent, not bitwise identical.

    Bitwise identity is unreachable for any reshaping of this model, so the bar
    is the fp32 kernel-noise floor relative to the signal.
    """
    cached = pipeline.predict(series, 64, num_samples=16, seed=0, engine="cached").values
    exact = pipeline.predict(series, 64, num_samples=16, seed=0, engine="exact").values
    scale = np.abs(exact).mean()
    assert np.abs(cached - exact).max() <= 1e-4 * max(scale, 1.0)


def test_cached_engine_handles_interior_gaps(pipeline, series, tiny_config, monkeypatch):
    """A series with all-NaN patches stays on the cached engine and agrees with
    the exact one, which runs the model's own forward over the same gaps."""
    gappy = series.copy()
    p = tiny_config.patch_len
    gappy[3 * p : 6 * p] = np.nan  # three whole patches with no observed value
    gappy[10 * p + 2 : 12 * p] = np.nan

    def no_fallback(*args, **kwargs):
        raise AssertionError("cached rollout fell back to the exact engine")

    exact = pipeline.predict(gappy, 40, num_samples=16, seed=0, engine="exact").values
    monkeypatch.setattr(pipeline, "_rollout_exact", no_fallback)
    cached = pipeline.predict(gappy, 40, num_samples=16, seed=0, engine="cached").values
    assert np.isfinite(cached).all()
    scale = np.abs(exact).mean()
    assert np.abs(cached - exact).max() <= 1e-4 * max(scale, 1.0)


def test_more_samples_do_not_change_the_median_much(pipeline, series):
    """A sanity check that the band is a Monte Carlo estimate, not noise."""
    few = pipeline.predict(series, 32, num_samples=8, seed=0).median
    many = pipeline.predict(series, 32, num_samples=64, seed=0).median
    assert np.isfinite(few).all() and np.isfinite(many).all()
    assert np.corrcoef(few, many)[0, 1] > 0.5


def test_context_longer_than_the_limit_is_truncated_not_rejected(pipeline, series):
    long_series = np.tile(series, 8)
    forecast = pipeline.predict(long_series, 16, num_samples=4, seed=0, context_length=64)
    assert forecast.values.shape == (1, 3, 16)


def test_default_context_reads_only_the_latest_2048_points(pipeline, series):
    """Anything older than the trained window must not change the forecast."""
    rng = np.random.default_rng(0)
    recent = np.tile(series, 8)  # exactly 2048 points
    older = rng.normal(0, 50, 1000).astype(np.float32)
    with_history = np.concatenate([older, recent])
    a = pipeline.predict(with_history, 16, num_samples=4, seed=0).values
    b = pipeline.predict(recent, 16, num_samples=4, seed=0).values
    assert np.array_equal(a, b)


def test_multivariate_context_is_accepted(pipeline, series):
    covariates = np.stack([series, np.roll(series, 3), np.roll(series, -3)])
    forecast = pipeline.predict([covariates], 16, num_samples=4, seed=0)
    assert forecast.values.shape == (1, 3, 16)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("engine", ["cached", "exact"])
def test_reduced_precision_inference(staged_checkpoint, series, dtype, engine):
    pipeline = YugPipeline.from_pretrained(
        staged_checkpoint, device_map="cpu", dtype=dtype
    )
    forecast = pipeline.predict(series, 17, num_samples=4, seed=0, engine=engine)
    assert forecast.values.shape == (1, 3, 17)
    assert np.isfinite(forecast.values).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_multivariate_inference(staged_checkpoint, series, dtype):
    pipeline = YugPipeline.from_pretrained(
        staged_checkpoint, device_map="cpu", dtype=dtype
    )
    context = np.stack([series, np.roll(series, 3)])
    forecast = pipeline.predict([context], 17, num_samples=4, seed=0)
    assert forecast.values.shape == (1, 3, 17)
    assert np.isfinite(forecast.values).all()


# ------------------------------------------------------------------ errors
def test_prediction_length_is_required(pipeline, series):
    with pytest.raises(ValueError, match="prediction_length is required"):
        pipeline.predict(series)


def test_context_shorter_than_one_patch_is_padded_not_rejected(
    pipeline, series, tiny_config
):
    """M4 yearly has series shorter than a patch; they must still be forecast."""
    short = series[: tiny_config.patch_len - 3]
    forecast = pipeline.predict(short, 8, num_samples=4, seed=0)
    assert forecast.values.shape == (1, 3, 8)
    assert np.isfinite(forecast.values).all()


def test_empty_context_is_rejected(pipeline):
    with pytest.raises(ValueError, match="empty"):
        pipeline.predict([np.array([], dtype=np.float32)], 8)


# -------------------------------------------------------- constant contexts
def test_constant_context_forecasts_the_constant(pipeline):
    flat = np.full(64, 42.5, dtype=np.float32)
    values = pipeline.predict(flat, 20, num_samples=4, seed=0).values
    assert (values == 42.5).all()


def test_constant_context_ignores_missing_values(pipeline):
    flat = np.full(64, 3.0, dtype=np.float32)
    flat[[0, 10, 40]] = np.nan
    values = pipeline.predict(flat, 8, num_samples=4, seed=0).values
    assert (values == 3.0).all()


def test_constant_check_sees_the_whole_series_not_just_the_window(pipeline, series):
    """Flat over the last context_length points, but not before: not constant."""
    tail_flat = np.concatenate([series, np.full(64, 7.0, dtype=np.float32)])
    values = pipeline.predict(
        tail_flat, 16, num_samples=8, seed=0, context_length=32
    ).values
    assert not (values == 7.0).all()


def test_constant_context_keeps_the_rng_stream_aligned(pipeline, series):
    """A skipped series consumes the draws its rollout would have made."""
    flat = np.full(64, 1.0, dtype=np.float32)
    after_flat = pipeline.predict([flat, series], 40, num_samples=8, seed=0).values[1]
    after_series = pipeline.predict(
        [series[::-1].copy(), series], 40, num_samples=8, seed=0
    ).values[1]
    assert np.array_equal(after_flat, after_series)


# ------------------------------------------------------- sub-floor contexts
# The causal RevIn cannot represent a deviation below sqrt(eps) (1e-3 at the
# published eps=1e-6), so a tiny-amplitude series used to reach the model
# flattened and come back as noise. The pipeline now forecasts such a context at
# unit scale and inverts the transform on the band.
def test_tiny_amplitude_context_is_scale_equivariant(pipeline, series):
    """A sub-floor series must forecast like the same shape at unit scale."""
    tiny = (series * 1e-6).astype(np.float32)
    at_unit = pipeline.predict(series, 32, num_samples=8, seed=0).values
    at_tiny = pipeline.predict(tiny, 32, num_samples=8, seed=0).values
    assert np.allclose(at_tiny, at_unit * 1e-6, rtol=1e-4, atol=0.0)


def test_the_rescale_is_what_makes_it_equivariant(pipeline, series, monkeypatch):
    """Without the guard the same shape at 1e-6 gives a materially different band."""
    monkeypatch.setattr("yug.pipeline._unit_scale_affine", lambda target, floor: None)
    tiny = (series * 1e-6).astype(np.float32)
    at_unit = pipeline.predict(series, 32, num_samples=8, seed=0).values
    at_tiny = pipeline.predict(tiny, 32, num_samples=8, seed=0).values
    assert not np.allclose(at_tiny, at_unit * 1e-6, rtol=1e-4, atol=0.0)


def test_ordinary_scale_context_is_left_untouched(pipeline, series):
    """Above the floor the guard must not fire at all."""
    assert _unit_scale_affine(series, pipeline._revin_std_floor) is None
    assert _unit_scale_affine(series * 1e9, pipeline._revin_std_floor) is None


@pytest.mark.parametrize("scale", [1e-7, 1.0, 1e7])
def test_multivariate_context_survives_any_scale(pipeline, series, scale):
    stack = np.stack([series, series * 0.5]).astype(np.float32) * scale
    values = pipeline.predict(stack, 16, num_samples=4, seed=0).values
    assert np.isfinite(values).all()


def test_a_generator_seed_spans_several_calls(pipeline, series):
    one_call = pipeline.predict([series, series[:200]], 24, num_samples=8, seed=3).values
    rng = np.random.default_rng(3)
    first = pipeline.predict(series, 24, num_samples=8, seed=rng).values
    second = pipeline.predict(series[:200], 24, num_samples=8, seed=rng).values
    assert np.array_equal(one_call, np.concatenate([first, second]))


def test_unknown_engine_is_rejected(pipeline, series):
    with pytest.raises(ValueError, match="cached"):
        pipeline.predict(series, 8, engine="turbo")


def test_defaults_from_load_time_are_used(staged_checkpoint, series):
    pipe = YugPipeline.from_pretrained(
        staged_checkpoint, device_map="cpu", prediction_length=24, num_samples=4
    )
    assert pipe.predict(series).values.shape == (1, 3, 24)


# ------------------------------------------------------- the forecast object
def test_forecast_accessors(pipeline, series):
    forecast = pipeline.predict(
        [series, series], 16, num_samples=4, seed=0, item_ids=["a", "b"]
    )
    assert len(forecast) == 2
    assert forecast.horizon == 16
    assert forecast.median.shape == (2, 16)

    one = forecast[0]
    assert one.median.shape == (16,)
    assert one.quantile(0.9).shape == (16,)

    band = one.interval()
    assert (band["lower"] <= band["upper"]).all()

    frame = forecast.to_dataframe()
    assert len(frame) == 32
    assert set(frame["item_id"]) == {"a", "b"}


def test_asking_for_an_untrained_quantile_says_which_exist(pipeline, series):
    forecast = pipeline.predict(series, 8, num_samples=4, seed=0)
    with pytest.raises(ValueError, match="available levels"):
        forecast.quantile(0.99)


def test_quantile_forecast_validates_its_shape():
    with pytest.raises(ValueError, match="quantiles"):
        QuantileForecast(np.zeros((1, 3, 5)), [0.1, 0.9])
