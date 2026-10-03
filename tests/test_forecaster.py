# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""The batched single-shot layer and the GluonTS adapter."""

from __future__ import annotations

import numpy as np
import pytest

from yug import ForecastResult, InferenceConfig, YugForecaster


@pytest.fixture
def forecaster(tiny_model, tiny_config) -> YugForecaster:
    cfg = InferenceConfig.from_model_config(tiny_config, device="cpu", batch_size=4)
    return YugForecaster(model=tiny_model, cfg=cfg)


def test_single_series_forecast_shape(forecaster, series, tiny_config):
    result = forecaster.predict(series, freq="D")
    assert result.median().shape == (tiny_config.output_dim,)
    assert result.last_window_forecast().shape == (
        tiny_config.num_quantiles,
        tiny_config.output_dim,
    )


def test_batch_of_ragged_series(forecaster, series):
    records = [
        {"target": series, "freq": "D"},
        {"target": series[:128], "freq": "H"},
        {"target": series[:100], "freq": "15min"},
    ]
    result = forecaster.predict_batch(records)
    assert result.predictions.shape[0] == 3
    for i in range(3):
        assert np.isfinite(result.last_window_forecast(i)).all()


def test_long_series_forecast_is_independent_of_short_batch_companion(forecaster, series):
    alone = forecaster.predict_batch([{"target": series, "freq": "D"}])
    batched = forecaster.predict_batch(
        [
            {"target": series, "freq": "D"},
            {"target": series[:64], "freq": "D"},
        ]
    )
    assert np.allclose(
        alone.last_window_forecast(0),
        batched.last_window_forecast(0),
        rtol=1e-5,
        atol=1e-5,
    )


def test_every_anchor_carries_a_forecast(forecaster, series, tiny_config):
    """all_windows_median gives an in-sample backtest for free."""
    result = forecaster.predict(series, freq="D")
    windows = result.all_windows_median()
    n_patches = len(series) // tiny_config.patch_len
    assert windows.shape == (n_patches, tiny_config.output_dim)
    # The last anchor is the one that forecasts the future.
    assert np.allclose(windows[-1], result.median())


def test_from_checkpoint_loads_a_staged_directory(staged_checkpoint, tiny_config, series):
    cfg = InferenceConfig.from_model_config(tiny_config, device="cpu")
    forecaster = YugForecaster.from_checkpoint(
        staged_checkpoint, model_config=tiny_config, inference_cfg=cfg
    )
    assert np.isfinite(forecaster.predict(series, freq="D").median()).all()


def test_covariates_change_the_forecast(forecaster, series):
    plain = forecaster.predict(series, freq="D").median()
    with_cov = forecaster.predict(
        series, freq="D", variates=np.stack([np.roll(series, 4)])
    ).median()
    assert not np.allclose(plain, with_cov)


def test_a_zero_patch_series_fails_loudly_instead_of_returning_zeros():
    """The exact trap that used to produce silent all-zero forecasts."""
    result = ForecastResult(
        predictions=np.zeros((1, 1, 2, 3, 8), dtype=np.float32),
        quantiles=[0.1, 0.5, 0.9],
        patch_len=8,
        output_len=16,
        n_patches_per_series=[0],
    )
    with pytest.raises(ValueError, match="refusing to emit"):
        result.last_window_forecast(0)


def test_rejects_a_2d_series(forecaster, series):
    with pytest.raises(ValueError, match="1-D"):
        forecaster.predict(np.stack([series, series]), freq="D")


def test_rejects_an_unknown_mode(forecaster, series):
    with pytest.raises(ValueError, match="univariate"):
        forecaster.predict_batch([{"target": series, "freq": "D"}], mode="sideways")


# ------------------------------------------------------------ GluonTS adapter
gluonts = pytest.importorskip("gluonts", reason="the gluonts extra is not installed")


def test_gluonts_adapter_emits_quantile_forecasts(pipeline, series):
    import pandas as pd
    from gluonts.model.forecast import QuantileForecast

    from yug.gluonts_adapter import YugPredictor

    predictor = YugPredictor(pipeline, freq="D", prediction_length=16, num_samples=4)
    dataset = [
        {
            "target": series,
            "start": pd.Period("2020-01-01", freq="D"),
            "item_id": "series-0",
        }
    ]
    forecasts = list(predictor.predict(dataset))

    assert len(forecasts) == 1
    forecast = forecasts[0]
    assert isinstance(forecast, QuantileForecast)
    assert forecast.item_id == "series-0"
    assert forecast.forecast_array.shape == (3, 16)
    # The forecast starts immediately after the last observation.
    assert forecast.start_date == pd.Period("2020-01-01", freq="D") + len(series)


def test_gluonts_adapter_walks_one_rng_stream(pipeline, series):
    """Entries share one stream, as in a single pipeline call over all of them.

    That is how the reference predictor behind the published GIFT-Eval scores
    draws its trajectories; re-seeding each entry does not reproduce them.
    """
    import pandas as pd

    from yug.gluonts_adapter import YugPredictor

    targets = [series, series[:200], series[::-1].copy()]
    dataset = [
        {"target": t, "start": pd.Period("2020-01-01", freq="D"), "item_id": str(i)}
        for i, t in enumerate(targets)
    ]
    predictor = YugPredictor(
        pipeline, freq="D", prediction_length=24, num_samples=8, seed=5
    )
    adapter = np.stack([f.forecast_array for f in predictor.predict(dataset)])
    direct = pipeline.predict(targets, 24, freq="D", num_samples=8, seed=5).values
    assert np.array_equal(adapter, direct)


def test_gluonts_adapter_defaults_to_the_trained_context_window(pipeline):
    from yug.gluonts_adapter import YugPredictor

    assert YugPredictor(pipeline).context_length == 2048


def test_gluonts_adapter_is_zero_shot_only(pipeline):
    from yug.gluonts_adapter import YugPredictor

    predictor = YugPredictor(pipeline, prediction_length=8)
    with pytest.raises(NotImplementedError, match="finetune"):
        predictor.train()


def test_gluonts_adapter_runs_make_evaluation_predictions(pipeline, series):
    """The predictor is drop-in for make_evaluation_predictions.

    Covers the two contract points that used to crash: the harness reads
    predictor.lead_time to size the holdout window, and calls
    predict(dataset, num_samples=...). The result is scored through the
    GluonTS Evaluator to confirm a full forecast flows end to end.
    """
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from gluonts.evaluation import Evaluator
    from gluonts.evaluation.backtest import make_evaluation_predictions

    from yug.gluonts_adapter import YugPredictor

    predictor = YugPredictor(pipeline, freq="D", prediction_length=16, num_samples=4)
    assert predictor.lead_time == 0

    ds = ListDataset(
        [{"start": pd.Period("2020-01-01", freq="D"), "target": series}], freq="D"
    )
    forecasts, tss = make_evaluation_predictions(ds, predictor=predictor, num_samples=8)
    forecasts, tss = list(forecasts), list(tss)
    assert len(forecasts) == 1

    # num_workers=0 keeps the Evaluator single-process; metric values are
    # irrelevant here, only that a finite result flows through.
    agg, _ = Evaluator(quantiles=[0.1, 0.5, 0.9], num_workers=0)(iter(tss), iter(forecasts))
    assert np.isfinite(agg["MASE"])
