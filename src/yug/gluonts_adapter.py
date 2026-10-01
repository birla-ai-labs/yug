# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""GluonTS-compatible predictor, for benchmark harnesses.

GIFT-Eval and most public time-series benchmarks drive a GluonTS ``Predictor``:
they hand it an iterable of entries and expect an iterator of ``Forecast``
objects back. :class:`YugPredictor` is that shim over
:class:`~yug.pipeline.YugPipeline`.

Requires the ``gluonts`` extra::

    pip install 'yug[gluonts]'
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from .pipeline import YugPipeline
from .utils import normalize_freq

logger = logging.getLogger(__name__)

__all__ = ["YugPredictor"]


def _entry_freq(entry: dict, fallback: str) -> str:
    """Prefer the frequency carried by the entry's own start period."""
    start = entry.get("start")
    freqstr = getattr(start, "freqstr", None)
    if freqstr:
        return normalize_freq(freqstr)
    return normalize_freq(fallback)


class YugPredictor:
    """GluonTS ``Predictor`` interface over the Yug pipeline.

    Zero-shot only: ``train`` and ``finetune`` raise. Fine-tune a checkpoint
    separately, then load the result here.

    Example::

        predictor = YugPredictor.from_pretrained(
            "birlaailabs/yug", freq="H", prediction_length=48,
        )
        forecasts = list(predictor.predict(dataset))
    """

    def __init__(
        self,
        pipeline: YugPipeline,
        *,
        freq: str = "D",
        prediction_length: int | None = None,
        num_samples: int = 100,
        context_length: int | None = 8192,
        engine: str = "cached",
        pad_side: str = "left",
        seed: int = 0,
        target_index: int = 0,
    ) -> None:
        self.pipeline = pipeline
        self.freq = str(freq)
        self.prediction_length = prediction_length
        self.num_samples = int(num_samples)
        self.context_length = context_length
        self.engine = engine
        self.pad_side = pad_side
        self.seed = int(seed)
        self.target_index = int(target_index)

        self.quantile_levels: list[float] = list(pipeline.quantile_levels)
        self.quantile_keys = [str(q) for q in self.quantile_levels]

    @classmethod
    def from_pretrained(
        cls, model_id: str, *, device_map: str | None = None, **kwargs: Any
    ) -> YugPredictor:
        """Load a checkpoint and wrap it. Predictor options pass through."""
        pipeline_keys = {"dtype", "revision", "cache_dir", "token", "model_config"}
        pipe_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in pipeline_keys}
        pipeline = YugPipeline.from_pretrained(
            model_id, device_map=device_map, **pipe_kwargs
        )
        return cls(pipeline, **kwargs)

    # ------------------------------------------------------------ interface
    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "YugPredictor is zero-shot; finetune a checkpoint "
            "separately, then load the resulting checkpoint."
        )

    def finetune(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "YugPredictor is zero-shot; finetune a checkpoint "
            "separately, then load the resulting checkpoint."
        )

    def predict(
        self, dataset: Iterable[dict], prediction_length: int | None = None
    ) -> Iterator:
        """Yield one GluonTS ``QuantileForecast`` per dataset entry."""
        try:
            from gluonts.model.forecast import QuantileForecast
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "the GluonTS adapter needs gluonts: pip install 'yug[gluonts]'"
            ) from exc

        horizon = (
            prediction_length if prediction_length is not None else self.prediction_length
        )
        if horizon is None:
            raise ValueError(
                "prediction_length missing: pass it to predict(), or to the constructor."
            )
        horizon = int(horizon)

        n = 0
        for entry in dataset:
            target = np.asarray(entry["target"], dtype=np.float32)
            forecast = self.pipeline.predict(
                [target],  # a one-element batch
                horizon,
                freq=_entry_freq(entry, self.freq),
                num_samples=self.num_samples,
                context_length=self.context_length,
                engine=self.engine,
                pad_side=self.pad_side,
                # Per-entry seed keeps a sweep reproducible whatever order or
                # parallelism the harness uses to walk the dataset.
                seed=self.seed + n,
            )

            seq_len = target.shape[-1] if target.ndim == 2 else len(target)
            yield QuantileForecast(
                forecast_arrays=forecast.values[0],
                start_date=entry["start"] + seq_len,
                forecast_keys=self.quantile_keys,
                item_id=entry.get("item_id"),
            )
            n += 1

        logger.info(
            "predict(): emitted %d forecast(s) (horizon=%d, engine=%s)",
            n,
            horizon,
            self.engine,
        )

    def __repr__(self) -> str:
        return (
            f"YugPredictor(freq={self.freq!r}, engine={self.engine!r}, "
            f"num_samples={self.num_samples}, "
            f"quantile_levels={self.quantile_levels})"
        )
