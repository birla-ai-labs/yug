# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Batched single-shot forecasting: patch, collate, run, reshape.

This is the layer directly above the network. It turns records into padded
patch tensors, runs one forward pass, and returns a
:class:`ForecastResult` holding a forecast block anchored at every input patch.

It does *not* roll forward. One call gives you ``output_dim`` future points; for
longer horizons use :class:`~yug.pipeline.YugPipeline`, which
samples and feeds the result back in.

Most users should not need this module — reach for it when you want the raw
per-patch head output, e.g. for backtesting every anchor in one pass.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

from .configs import InferenceConfig, YugConfig
from .model import Yug_Model
from .utils import freq_id as _freq_id
from .utils import normalize_freq

logger = logging.getLogger(__name__)

__all__ = ["ForecastResult", "YugForecaster", "load_state_dict_from_file"]


def load_state_dict_from_file(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a ``.safetensors`` checkpoint into a plain state dict.

    Handles the two wrappers training checkpoints commonly carry: an outer
    ``model_state_dict`` key, and the ``_orig_mod.`` prefix that
    ``torch.compile`` adds to every parameter name.
    """
    raw: dict[str, Any] = load_file(str(path))
    # Training checkpoints sometimes nest the weights one level down.
    state_dict: dict[str, torch.Tensor] = raw.get("model_state_dict", raw)
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


class _UnivariateInferenceDataset(Dataset):
    """Records -> per-series patch arrays with a validity mask."""

    def __init__(self, records: list[dict], cfg: InferenceConfig) -> None:
        self.records = records
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        series = np.asarray(r["target"], dtype=np.float32)
        seq_len = len(series)
        fid = _freq_id(normalize_freq(r.get("freq", "D")))

        nan_mask = np.isnan(series)
        series = np.nan_to_num(series, nan=0.0, posinf=0.0, neginf=0.0)

        rem = seq_len % self.cfg.patch_len
        pad_len = (self.cfg.patch_len - rem) if rem else 0
        if pad_len:
            series = np.pad(series, (0, pad_len), constant_values=self.cfg.pad_token)

        mask = np.zeros(len(series), dtype=np.int32)
        mask[:seq_len] = 1
        mask[:seq_len] &= (~nan_mask).astype(np.int32)

        n_patches = len(series) // self.cfg.patch_len
        return {
            "patches": series.reshape(n_patches, self.cfg.patch_len),
            "mask": mask.reshape(n_patches, self.cfg.patch_len).astype(np.int16),
            "freq_id": fid,
            "n_patches": n_patches,
            "series_idx": idx,
        }


class _MultivariateInferenceDataset(Dataset):
    """As above, but every channel of a ``(V, T)`` record is patched."""

    def __init__(
        self, records: list[dict], cfg: InferenceConfig, target_index: int = 0
    ) -> None:
        self.records = records
        self.cfg = cfg
        self.target_index = target_index

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        series = np.asarray(r["target"], dtype=np.float32)
        if series.ndim == 1:
            series = series.reshape(1, -1)

        n_variables, seq_len = series.shape
        fid = _freq_id(normalize_freq(r.get("freq", "D")))

        n_patches = None
        padded_vars, masks = [], []
        for v in range(n_variables):
            vs = series[v]
            nan_mask = np.isnan(vs)
            vs = np.nan_to_num(vs, nan=0.0, posinf=0.0, neginf=0.0)

            rem = seq_len % self.cfg.patch_len
            pad_len = (self.cfg.patch_len - rem) if rem else 0
            if pad_len:
                vs = np.pad(vs, (0, pad_len), constant_values=self.cfg.pad_token)

            m = np.zeros(len(vs), dtype=np.int32)
            m[:seq_len] = 1
            m[:seq_len] &= (~nan_mask).astype(np.int32)

            n_patches = len(vs) // self.cfg.patch_len
            padded_vars.append(vs.reshape(n_patches, self.cfg.patch_len))
            masks.append(m.reshape(n_patches, self.cfg.patch_len))

        return {
            "patches": np.stack(padded_vars, axis=0),
            "mask": np.stack(masks, axis=0).astype(np.int16),
            "freq_id": fid,
            "n_patches": n_patches,
            "n_variables": n_variables,
            "series_idx": idx,
        }


def _collate_univariate_inference(batch: list[dict]) -> dict:
    """Right-pad every series in the batch to the longest one."""
    max_p = max(b["n_patches"] for b in batch)
    patch_len = batch[0]["patches"].shape[-1]
    n = len(batch)

    inp_arr = np.zeros((n, max_p, patch_len), dtype=np.float32)
    mask_arr = np.zeros((n, max_p, patch_len), dtype=np.int16)

    for i, b in enumerate(batch):
        k = b["n_patches"]
        inp_arr[i, :k] = b["patches"]
        mask_arr[i, :k] = b["mask"]

    return {
        "input": torch.tensor(inp_arr, dtype=torch.float32),
        "input_padding_mask": torch.tensor(mask_arr, dtype=torch.bool),
        "freq_id": torch.tensor([b["freq_id"] for b in batch], dtype=torch.long),
        "n_patches": torch.tensor([b["n_patches"] for b in batch], dtype=torch.long),
        "series_idx": torch.tensor([b["series_idx"] for b in batch], dtype=torch.long),
    }


def _collate_multivariate_inference(batch: list[dict], target_index: int = 0) -> dict:
    max_p = max(b["n_patches"] for b in batch)
    patch_len = batch[0]["patches"].shape[-1]
    n_var = batch[0]["patches"].shape[0]
    n = len(batch)

    all_patches = np.zeros((n, n_var, max_p, patch_len), dtype=np.float32)
    all_masks = np.zeros((n, n_var, max_p, patch_len), dtype=np.int16)

    for i, b in enumerate(batch):
        k = b["n_patches"]
        all_patches[i, :, :k] = b["patches"]
        all_masks[i, :, :k] = b["mask"]

    var_indices = [v for v in range(n_var) if v != target_index]

    return {
        "input_target": torch.tensor(all_patches[:, target_index], dtype=torch.float32),
        "input_variates": torch.tensor(all_patches[:, var_indices], dtype=torch.float32),
        "input_padding_mask_target": torch.tensor(
            all_masks[:, target_index], dtype=torch.bool
        ),
        "input_padding_mask_variates": torch.tensor(
            all_masks[:, var_indices], dtype=torch.bool
        ),
        "freq_id": torch.tensor([b["freq_id"] for b in batch], dtype=torch.long),
        "n_patches": torch.tensor([b["n_patches"] for b in batch], dtype=torch.long),
        "series_idx": torch.tensor([b["series_idx"] for b in batch], dtype=torch.long),
    }


class ForecastResult:
    """A forecast block anchored at every input patch.

    ``predictions`` is ``(n_series, n_patches, output_patches, n_quantiles,
    patch_len)``. Anchor ``t`` holds the model's forecast of the ``output_dim``
    points following patch ``t``, so the last anchor is the one that forecasts
    the future and the earlier anchors are in-sample backtests, free.
    """

    def __init__(
        self,
        predictions: np.ndarray,
        quantiles: list[float],
        patch_len: int,
        output_len: int,
        n_patches_per_series: list[int],
    ) -> None:
        self.predictions = predictions
        self.quantiles = list(quantiles)
        self.patch_len = patch_len
        self.output_len = output_len
        self.n_patches_per_series = list(n_patches_per_series)

    def __repr__(self) -> str:
        return (
            f"ForecastResult(n_series={self.predictions.shape[0]}, "
            f"quantiles={self.quantiles}, output_len={self.output_len})"
        )

    def _q_idx(self, q: float) -> int:
        for i, qv in enumerate(self.quantiles):
            if abs(qv - q) < 1e-6:
                return i
        raise ValueError(f"Quantile {q} not found in {self.quantiles}")

    def _median_idx(self) -> int:
        for i, q in enumerate(self.quantiles):
            if abs(q - 0.5) < 1e-6:
                return i
        return len(self.quantiles) // 2

    def last_window_forecast(self, series_i: int = 0) -> np.ndarray:
        """``(n_quantiles, output_len)`` from the final real patch."""
        n = self.n_patches_per_series[series_i]
        if n <= 0:
            # n - 1 == -1 would silently read the last (zero-filled) slot of
            # the dense array and emit an all-zero forecast. Fail loudly: this
            # is exactly the trap that produces silent zero predictions.
            raise ValueError(
                f"series {series_i} produced 0 usable patches — refusing to emit "
                f"an all-zero forecast (input shorter than one patch, or its "
                f"batch was dropped upstream)."
            )
        last = self.predictions[series_i, n - 1]
        arr = last.transpose(1, 0, 2).reshape(len(self.quantiles), -1)
        return arr[:, : self.output_len]

    def median(self, series_i: int = 0) -> np.ndarray:
        return self.last_window_forecast(series_i)[self._median_idx()]

    def quantile(self, q: float, series_i: int = 0) -> np.ndarray:
        return self.last_window_forecast(series_i)[self._q_idx(q)]

    def interval(self, series_i: int = 0) -> dict[str, np.ndarray]:
        """Outermost predicted band as ``{"lower": ..., "upper": ...}``."""
        lw = self.last_window_forecast(series_i)
        return {"lower": lw[0], "upper": lw[-1]}

    def all_windows_median(self, series_i: int = 0) -> np.ndarray:
        """Median forecast from every anchor: ``(n_patches, output_len)``."""
        n = self.n_patches_per_series[series_i]
        q = self._median_idx()
        return self.predictions[series_i, :n, :, q, :].reshape(n, -1)[
            :, : self.output_len
        ]


class YugForecaster:
    """Runs one forward pass over a batch of records."""

    def __init__(self, model: Yug_Model, cfg: InferenceConfig) -> None:
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        model_config: dict | YugConfig | None = None,
        inference_cfg: InferenceConfig | None = None,
    ) -> YugForecaster:
        """Load weights from a local ``.safetensors`` file (or a directory holding one)."""
        config = model_config if model_config is not None else YugConfig()
        if isinstance(config, YugConfig):
            arch = config.to_arch_dict()
            cfg = inference_cfg or InferenceConfig.from_model_config(config)
        else:
            arch = dict(config)
            cfg = inference_cfg or InferenceConfig(
                patch_len=arch["patch_len"],
                output_len=arch["output_dim"],
                quantiles=list(arch["quantiles"]),
            )

        path = Path(checkpoint_path)
        if path.is_dir():
            candidate = path / "model.safetensors"
            if not candidate.exists():
                raise FileNotFoundError(f"No model.safetensors inside {path}")
            path = candidate

        logger.info("Loading checkpoint: %s", path)
        state_dict = load_state_dict_from_file(path)

        model = Yug_Model(config=arch)
        model.load_state_dict(state_dict)
        model.to(torch.device(cfg.device)).to(torch.float32).eval()

        del state_dict
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cls(model=model, cfg=cfg)

    def predict(
        self,
        series: np.ndarray | Sequence[float],
        freq: str = "D",
        variates: np.ndarray | None = None,
        target_index: int = 0,
        num_input_patches_override: int | None = None,
    ) -> ForecastResult:
        """Forecast one series, optionally conditioned on covariates."""
        series = np.asarray(series, dtype=np.float32)
        if series.ndim != 1:
            raise ValueError(f"series must be 1-D, got shape {series.shape}")

        if variates is not None:
            variates = np.asarray(variates, dtype=np.float32)
            if variates.ndim == 1:
                variates = variates.reshape(1, -1)
            combined = np.insert(variates, target_index, series, axis=0)
            return self._predict_records(
                [{"target": combined, "freq": freq}],
                mode="multivariate",
                target_index=target_index,
                num_input_patches_override=num_input_patches_override,
            )
        return self._predict_records(
            [{"target": series, "freq": freq}],
            mode="univariate",
            num_input_patches_override=num_input_patches_override,
        )

    def predict_batch(
        self,
        records: list[dict],
        mode: str = "univariate",
        target_index: int = 0,
        num_input_patches_override: int | None = None,
    ) -> ForecastResult:
        """Forecast many records at once. Each is ``{"target": ..., "freq": ...}``."""
        return self._predict_records(
            records,
            mode=mode,
            target_index=target_index,
            num_input_patches_override=num_input_patches_override,
        )

    def _predict_records(
        self,
        records: list[dict],
        mode: str = "univariate",
        target_index: int = 0,
        num_input_patches_override: int | None = None,
    ) -> ForecastResult:
        cfg = self.cfg
        n_quantiles = len(cfg.quantiles)
        device = self.device

        if mode == "univariate":
            dataset: Dataset = _UnivariateInferenceDataset(records, cfg)
            collate = _collate_univariate_inference
        elif mode == "multivariate":
            dataset = _MultivariateInferenceDataset(records, cfg, target_index)

            def collate(b):
                return _collate_multivariate_inference(b, target_index)

        else:
            raise ValueError(f"mode must be 'univariate' or 'multivariate', got {mode!r}")

        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
        )

        results_store: dict[int, np.ndarray] = {}
        n_patches_store: dict[int, int] = {}

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                series_idxs = batch["series_idx"].tolist()
                n_patches_b = batch["n_patches"].tolist()
                freq_ids = batch["freq_id"][:, 0].to(device)
                scalings = batch["freq_id"][:, 1].to(device)

                if num_input_patches_override is not None:
                    num_input_patches = min(
                        int(num_input_patches_override), min(n_patches_b)
                    )
                else:
                    num_input_patches = batch["n_patches"].to(device)

                if mode == "univariate":
                    inp = batch["input"].to(device, dtype=torch.float32)
                    attn_mask = batch["input_padding_mask"].to(device)
                    pred, _, _ = self.model(
                        inp,
                        freq_id=freq_ids,
                        scale=scalings,
                        attn_mask_target=attn_mask,
                        num_input_patches=num_input_patches,
                    )
                else:
                    inp_t = batch["input_target"].to(device, dtype=torch.float32)
                    inp_v = batch["input_variates"].to(device, dtype=torch.float32)
                    mask_t = batch["input_padding_mask_target"].to(device)
                    mask_v = batch["input_padding_mask_variates"].to(device)
                    pred, _, _ = self.model(
                        inp_t,
                        inp_v,
                        freq_id=freq_ids,
                        scale=scalings,
                        attn_mask_target=mask_t,
                        attn_mask_variates=mask_v,
                        num_input_patches=num_input_patches,
                    )

                pred = pred.to(torch.float32)
                b, p, _ = pred.shape
                o, ln = cfg.output_patches, cfg.patch_len
                expected = o * n_quantiles * ln
                if pred.shape[-1] != expected:
                    raise RuntimeError(
                        f"Model head emitted {pred.shape[-1]} values per patch, but "
                        f"InferenceConfig expects O({o}) x Q({n_quantiles}) x "
                        f"L({ln}) = {expected}. output_len and quantiles must match "
                        f"the trained checkpoint — never derive output_len from the "
                        f"prediction horizon."
                    )
                pred = pred.reshape(b, p, o, n_quantiles, ln)

                pred_np = pred.cpu().numpy()
                for i, (sid, n_real) in enumerate(
                    zip(series_idxs, n_patches_b, strict=True)
                ):
                    results_store[sid] = pred_np[i, :n_real]
                    n_patches_store[sid] = n_real

        if not results_store:
            raise RuntimeError(
                "No batches produced predictions — refusing to return an all-zero array."
            )

        n_series = len(records)
        max_patches = max(n_patches_store.values())
        o, q, ln = cfg.output_patches, n_quantiles, cfg.patch_len
        out = np.zeros((n_series, max_patches, o, q, ln), dtype=np.float32)
        n_patches_list = []
        for sid in range(n_series):
            arr = results_store.get(sid)
            if arr is not None:
                out[sid, : arr.shape[0]] = arr
            n_patches_list.append(n_patches_store.get(sid, 0))

        return ForecastResult(
            predictions=out,
            quantiles=cfg.quantiles,
            patch_len=cfg.patch_len,
            output_len=cfg.output_len,
            n_patches_per_series=n_patches_list,
        )
