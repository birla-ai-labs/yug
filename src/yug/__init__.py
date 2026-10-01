# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Yug — a foundation model for zero-shot probabilistic time-series forecasting.

    from yug import YugPipeline

    pipe = YugPipeline.from_pretrained("birlaailabs/yug")
    forecast = pipe.predict(context, prediction_length=64)
    forecast.median

Everything below :class:`YugPipeline` is available for the cases it does
not cover — :class:`YugForecaster` for raw per-anchor head output,
:class:`Yug_Model` for the network itself — but most users need only the
pipeline and the forecast it returns.
"""

from .__about__ import __version__
from .base import BaseForecastPipeline, QuantileForecast
from .configs import InferenceConfig, PredictionConfig, YugConfig
from .engine import CachedUnivariateEngine
from .forecaster import ForecastResult, YugForecaster
from .model import Yug_Model
from .pipeline import YugPipeline
from .utils import QuantileSampler, freq_id, normalize_freq

#: Alias matching the class name used in the reference research scripts.
Yug = Yug_Model

__all__ = [
    "__version__",
    # Primary entry point
    "YugPipeline",
    "QuantileForecast",
    "YugConfig",
    # Configuration
    "InferenceConfig",
    "PredictionConfig",
    # Lower-level building blocks
    "BaseForecastPipeline",
    "Yug_Model",
    "Yug",
    "YugForecaster",
    "ForecastResult",
    "CachedUnivariateEngine",
    "QuantileSampler",
    "normalize_freq",
    "freq_id",
]


def __getattr__(name: str):
    """Expose the GluonTS adapter lazily so gluonts stays an optional extra."""
    if name == "YugPredictor":
        from .gluonts_adapter import YugPredictor

        return YugPredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
