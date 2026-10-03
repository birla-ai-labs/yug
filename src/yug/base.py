# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""The contract every Yug pipeline obeys, and the type they return.

Splitting this out is what lets several checkpoints, sizes, or future variants
share one user experience: load with ``from_pretrained``, call ``predict``, get
a :class:`QuantileForecast` back.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["ContextLike", "QuantileForecast", "BaseForecastPipeline"]

#: Anything accepted as forecast context: one series, several series, or a
#: covariate stack. Concrete shapes are normalised in the pipeline.
ContextLike = (
    np.ndarray | Sequence[float] | Sequence[Sequence[float]] | Sequence[np.ndarray]
)


class QuantileForecast:
    """A probabilistic forecast: quantile levels by horizon, per series.

    ``values`` is ``(n_series, n_quantiles, horizon)``, ascending in the
    quantile axis. Levels are exactly those the checkpoint was trained to emit
    — the head's width is fixed at training time, so you cannot ask for a level
    that was not trained.

    Index it to get one series::

        forecast[0].median            # (horizon,)
        forecast[0].quantile(0.9)     # (horizon,)
    """

    def __init__(
        self,
        values: np.ndarray,
        quantile_levels: Sequence[float],
        item_ids: Sequence[Any] | None = None,
    ) -> None:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2:
            values = values[None, ...]
        if values.ndim != 3:
            raise ValueError(
                f"values must be (n_series, n_quantiles, horizon), got {values.shape}"
            )
        if values.shape[1] != len(quantile_levels):
            raise ValueError(
                f"values has {values.shape[1]} quantiles but {len(quantile_levels)} "
                f"levels were given"
            )
        if item_ids is not None and len(item_ids) != values.shape[0]:
            raise ValueError(f"got {len(item_ids)} item_ids for {values.shape[0]} series")

        self.values = values
        self.quantile_levels = [float(q) for q in quantile_levels]
        self.item_ids = list(item_ids) if item_ids is not None else None

    # ------------------------------------------------------------- geometry
    @property
    def n_series(self) -> int:
        return self.values.shape[0]

    @property
    def horizon(self) -> int:
        return self.values.shape[2]

    def __len__(self) -> int:
        return self.n_series

    def __repr__(self) -> str:
        return (
            f"QuantileForecast(n_series={self.n_series}, horizon={self.horizon}, "
            f"quantile_levels={self.quantile_levels})"
        )

    def __getitem__(self, i: int) -> QuantileForecast:
        return QuantileForecast(
            self.values[i : i + 1],
            self.quantile_levels,
            None if self.item_ids is None else self.item_ids[i : i + 1],
        )

    # -------------------------------------------------------------- accessors
    def _level_index(self, q: float) -> int:
        for i, level in enumerate(self.quantile_levels):
            if abs(level - q) < 1e-6:
                return i
        raise ValueError(
            f"quantile {q} was not produced by this checkpoint; available levels "
            f"are {self.quantile_levels}"
        )

    @property
    def median(self) -> np.ndarray:
        """The 0.5 level, or the nearest level if 0.5 was not trained.

        ``(horizon,)`` for a single series, ``(n_series, horizon)`` otherwise.
        """
        idx = int(np.argmin(np.abs(np.asarray(self.quantile_levels) - 0.5)))
        out = self.values[:, idx, :]
        return out[0] if self.n_series == 1 else out

    def quantile(self, q: float) -> np.ndarray:
        """One quantile level across the horizon."""
        out = self.values[:, self._level_index(q), :]
        return out[0] if self.n_series == 1 else out

    def interval(
        self, lower: float | None = None, upper: float | None = None
    ) -> dict[str, np.ndarray]:
        """A prediction band. Defaults to the widest trained pair."""
        lower = self.quantile_levels[0] if lower is None else lower
        upper = self.quantile_levels[-1] if upper is None else upper
        return {"lower": self.quantile(lower), "upper": self.quantile(upper)}

    def to_dataframe(self):
        """Long-format frame: one row per (item, step), one column per level.

        Requires pandas, which is an optional dependency of this package.
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                "to_dataframe() needs pandas: pip install 'yug[pandas]'"
            ) from exc

        frames = []
        for i in range(self.n_series):
            item = self.item_ids[i] if self.item_ids is not None else i
            data = {"item_id": item, "step": np.arange(self.horizon)}
            for j, level in enumerate(self.quantile_levels):
                data[str(level)] = self.values[i, j]
            frames.append(pd.DataFrame(data))
        return pd.concat(frames, ignore_index=True)


class BaseForecastPipeline(abc.ABC):
    """Interface shared by every Yug pipeline."""

    #: Quantile levels this pipeline emits, ascending.
    quantile_levels: list[float]

    @classmethod
    @abc.abstractmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device_map: str | None = None,
        **kwargs: Any,
    ) -> BaseForecastPipeline:
        """Load a checkpoint by Hub id or local path."""

    @abc.abstractmethod
    def predict(
        self,
        context: ContextLike,
        prediction_length: int | None = None,
    ) -> QuantileForecast:
        """Forecast ``prediction_length`` steps beyond each context series.

        Implementations may add keyword-only options on top of these, but must
        accept at least these two positionally.
        """
