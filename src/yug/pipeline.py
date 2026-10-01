# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""The user-facing pipeline: load a checkpoint, get a forecast.

    from yug import YugPipeline

    pipe = YugPipeline.from_pretrained("birlaailabs/yug")
    forecast = pipe.predict(context, prediction_length=64)
    forecast.median

How a horizon longer than one block is produced
-----------------------------------------------
The network emits ``output_dim`` future points per forward pass, as a set of
quantile levels. For a longer horizon the pipeline rolls forward: it draws
``num_samples`` independent trajectories, each of which repeatedly samples a
block from the predicted quantile band, appends the *sample* (not the median)
to its own context, and forecasts again. The returned band is the empirical
quantile across those trajectories, so it accumulates rollout uncertainty
rather than pretending each block is independent.

Feeding back the sample rather than the median matters: a median-feedback
rollout collapses toward a smooth line and reports a band far too narrow to be
honest about multi-block uncertainty.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .base import BaseForecastPipeline, ContextLike, QuantileForecast
from .configs import InferenceConfig, PredictionConfig, YugConfig, default_device
from .engine import CachedUnivariateEngine
from .forecaster import YugForecaster, load_state_dict_from_file
from .model import Yug_Model
from .utils import (
    QuantileSampler,
    freq_id,
    normalize_freq,
    pad_to_patch_multiple,
    patchify,
)

logger = logging.getLogger(__name__)

__all__ = ["YugPipeline"]

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


class YugPipeline(BaseForecastPipeline):
    """Zero-shot probabilistic forecasting with Yug.

    Construct with :meth:`from_pretrained`. The pipeline is stateless between
    calls apart from the loaded weights, so one instance can serve many series.
    """

    def __init__(
        self,
        model: Yug_Model,
        config: YugConfig,
        *,
        device: str | torch.device | None = None,
        prediction_config: PredictionConfig | None = None,
    ) -> None:
        self.config = config
        self.model = model.eval()
        self.device = torch.device(device or next(model.parameters()).device)
        self.defaults = prediction_config or PredictionConfig()

        self.quantile_levels: list[float] = list(config.quantiles)
        self._sampler = QuantileSampler(self.quantile_levels)

        self._patch_len = config.patch_len
        self._output_len = config.output_dim
        self._output_patches = config.output_patches
        self._n_quantiles = config.num_quantiles

        # Lazily built; reused across series so the RoPE table is built once.
        self._engine: CachedUnivariateEngine | None = None
        self._forecaster: YugForecaster | None = None

    def __repr__(self) -> str:
        return (
            f"YugPipeline(device={self.device}, "
            f"quantile_levels={self.quantile_levels}, "
            f"patch_len={self._patch_len}, output_dim={self._output_len})"
        )

    # ------------------------------------------------------------- loading
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        device_map: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        model_config: dict | YugConfig | None = None,
        **prediction_defaults: Any,
    ) -> YugPipeline:
        """Load from a Hugging Face Hub id or a local directory.

        ``model_id``
            ``"birlaailabs/yug"``, or a path to a directory
            holding ``config.json`` and ``model.safetensors``, or a path
            straight to a ``.safetensors`` file (then pass ``model_config``).

        ``device_map``
            ``"cuda"``, ``"cpu"``, ``"cuda:1"``, ... Defaults to CUDA when an
            accelerator is visible.

        ``model_config``
            Overrides the published ``config.json``. Needed only for bare
            checkpoint files that ship without one.

        Extra keyword arguments become the defaults for :meth:`predict` — e.g.
        ``num_samples=200``, ``engine="exact"``.
        """
        device = torch.device(device_map or default_device())
        if device.type == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable — falling back to CPU")
            device = torch.device("cpu")

        config, weights_path = cls._resolve(
            model_id,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            model_config=model_config,
        )

        model = Yug_Model(config=config.to_arch_dict())
        state_dict = load_state_dict_from_file(weights_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            raise RuntimeError(
                f"Checkpoint {weights_path} is missing {len(missing)} parameter(s) "
                f"the architecture needs, e.g. {sorted(missing)[:5]}. The config "
                f"and the weights do not describe the same model."
            )
        if unexpected:
            # Extra tensors are survivable — a newer checkpoint carrying
            # optimiser or EMA state, say — so warn rather than refuse.
            logger.warning(
                "Ignoring %d unexpected tensor(s) in the checkpoint, e.g. %s",
                len(unexpected),
                sorted(unexpected)[:5],
            )

        model.to(device).to(dtype).eval()

        return cls(
            model,
            config,
            device=device,
            prediction_config=PredictionConfig(**prediction_defaults),
        )

    @staticmethod
    def _resolve(
        model_id: str,
        *,
        revision: str | None,
        cache_dir: str | None,
        token: str | bool | None,
        model_config: dict | YugConfig | None,
    ):
        """Return ``(YugConfig, weights_path)`` for a local or Hub id."""
        path = Path(model_id)

        if path.is_file():
            if model_config is None:
                sibling = path.parent / CONFIG_NAME
                if not sibling.exists():
                    raise FileNotFoundError(
                        f"{path} is a bare checkpoint and no {CONFIG_NAME} sits "
                        f"beside it; pass model_config=... explicitly."
                    )
                config = YugConfig.from_json_file(sibling)
            else:
                config = _as_config(model_config)
            return config, path

        if path.is_dir():
            weights = path / WEIGHTS_NAME
            if not weights.exists():
                raise FileNotFoundError(f"No {WEIGHTS_NAME} inside {path}")
            config = (
                _as_config(model_config)
                if model_config is not None
                else YugConfig.from_json_file(path / CONFIG_NAME)
            )
            return config, weights

        # Not on disk — treat it as a Hub repo id.
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(
                f"{model_id!r} is not a local path, so it is being read as a "
                f"Hugging Face repo id, which needs huggingface_hub: "
                f"pip install huggingface_hub"
            ) from exc

        def fetch(filename: str) -> Path:
            return Path(
                hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                )
            )

        config = (
            _as_config(model_config)
            if model_config is not None
            else YugConfig.from_json_file(fetch(CONFIG_NAME))
        )
        return config, fetch(WEIGHTS_NAME)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        """Write ``config.json`` and ``model.safetensors`` into a directory.

        This produces exactly the layout :meth:`from_pretrained` reads and that
        a Hugging Face model repository expects, so it is the staging step
        before publishing a checkpoint.
        """
        from safetensors.torch import save_file

        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)
        self.config.to_json_file(out / CONFIG_NAME)

        state = {
            k: v.detach().cpu().contiguous() for k, v in self.model.state_dict().items()
        }
        save_file(state, str(out / WEIGHTS_NAME))
        return out

    # ------------------------------------------------------------- predict
    def predict(
        self,
        context: ContextLike,
        prediction_length: int | None = None,
        *,
        freq: str | Sequence[str] = "D",
        num_samples: int | None = None,
        context_length: int | None = None,
        engine: str | None = None,
        pad_side: str | None = None,
        seed: int | None = None,
        item_ids: Sequence[Any] | None = None,
    ) -> QuantileForecast:
        """Forecast ``prediction_length`` steps beyond each context series.

        ``context``
            One series (1-D), several equal-length series (2-D, one per row),
            or a list of 1-D arrays of differing lengths.

        ``freq``
            Sampling frequency as a pandas offset alias (``"D"``, ``"15min"``,
            ``"ME"``). One string, or one per series. This is a real input to
            the model, not bookkeeping — it selects a learned embedding, so a
            wrong value degrades the forecast.

        ``num_samples``
            Trajectories to draw. More narrows Monte Carlo noise in the
            returned band, at linear cost.

        ``seed``
            Makes the call reproducible. The stream advances across series
            within one call, so a given ``(seed, context, settings)`` always
            gives the same answer.

        Returns a :class:`~yug.base.QuantileForecast` of shape
        ``(n_series, n_quantiles, prediction_length)``.
        """
        d = self.defaults
        horizon = (
            prediction_length if prediction_length is not None else d.prediction_length
        )
        if horizon is None:
            raise ValueError(
                "prediction_length is required: pass it to predict(), or set it "
                "as a default via from_pretrained(prediction_length=...)."
            )
        horizon = int(horizon)
        if horizon < 1:
            raise ValueError(f"prediction_length must be >= 1, got {horizon}")

        n_samples = int(num_samples if num_samples is not None else d.num_samples)
        ctx_limit = context_length if context_length is not None else d.context_length
        engine_mode = engine or d.engine
        side = pad_side or d.pad_side
        if engine_mode not in ("cached", "exact"):
            raise ValueError(f"engine must be 'cached' or 'exact', got {engine_mode!r}")

        series_list = _as_series_list(context)
        freqs = _broadcast_freq(freq, len(series_list))
        rng = np.random.default_rng(d.seed if seed is None else seed)

        bands = []
        with torch.inference_mode():
            for target, f in zip(series_list, freqs, strict=True):
                bands.append(
                    self._predict_one(
                        target,
                        normalize_freq(f),
                        horizon,
                        n_samples=n_samples,
                        context_length=ctx_limit,
                        engine=engine_mode,
                        pad_side=side,
                        rng=rng,
                    )
                )

        return QuantileForecast(
            np.stack(bands, axis=0), self.quantile_levels, item_ids=item_ids
        )

    # --------------------------------------------------------------- rollout
    def _predict_one(
        self,
        target: np.ndarray,
        freq: str,
        horizon: int,
        *,
        n_samples: int,
        context_length: int | None,
        engine: str,
        pad_side: str,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """One series -> ``(n_quantiles, horizon)``."""
        multivariate = target.ndim == 2

        if context_length and target.shape[-1] > context_length:
            target = target[..., -context_length:]

        if target.shape[-1] < self._patch_len:
            raise ValueError(
                f"context has {target.shape[-1]} point(s) but the model reads whole "
                f"patches of {self._patch_len}; supply at least one full patch."
            )

        target, _ = pad_to_patch_multiple(target, self._patch_len, pad_side)

        # Blocks needed to cover the horizon; the tail is trimmed afterwards.
        n_steps = -(-horizon // self._output_len)

        if multivariate:
            paths = self._rollout_multivariate(target, freq, n_steps, n_samples, rng)
        elif engine == "cached":
            paths = self._rollout_cached(target, freq, n_steps, n_samples, rng)
        else:
            paths = self._rollout_exact(target, freq, n_steps, n_samples, rng)

        traj = paths[:, :horizon]
        return np.quantile(traj, self.quantile_levels, axis=0).astype(np.float32)

    def _freq_tensors(self, freq: str, n: int):
        fid, scale = freq_id(freq)
        return (
            torch.tensor([fid] * n, dtype=torch.long, device=self.device),
            torch.tensor([scale] * n, dtype=torch.long, device=self.device),
        )

    def _to_device(self, values: np.ndarray, mask: np.ndarray):
        return (
            torch.from_numpy(values).to(self.device, dtype=torch.float32),
            torch.from_numpy(mask).to(self.device),
        )

    def _rollout_cached(
        self,
        target: np.ndarray,
        freq: str,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Encode the context once, then advance only the newly drawn patches."""
        patch_len = self._patch_len
        n_ctx_patches = target.shape[-1] // patch_len

        values, mask = patchify(target[None, :], patch_len)
        ctx, ctx_mask = self._to_device(values, mask)

        # The cached path needs every patch to be a valid attention token. Left
        # padding only blanks part of patch 0, so that holds — but a series with
        # an interior all-NaN patch does not, and falls back.
        if not bool((ctx_mask.sum(-1) > 0).all()):
            logger.debug("interior all-NaN patch — falling back to the exact engine")
            return self._rollout_exact(target, freq, n_steps, n_paths, rng)

        freq_ids, scalings = self._freq_tensors(freq, 1)

        if self._engine is None:
            self._engine = CachedUnivariateEngine(self.model)
        eng = self._engine

        max_patches = n_ctx_patches + (n_steps - 1) * self._output_patches
        collected = np.empty((n_paths, n_steps * self._output_len), dtype=np.float32)

        # Step 0: every path holds the identical context, so one forward pass
        # serves them all. The paths only diverge through their sampled u.
        qf_device = eng.begin(
            ctx,
            ctx_mask,
            freq_ids,
            scalings,
            n_paths=n_paths,
            max_patches=max_patches,
        )
        qf = np.broadcast_to(
            np.sort(qf_device.float().cpu().numpy(), axis=1),
            (n_paths, self._n_quantiles, self._output_len),
        )

        for step in range(n_steps):
            if step:
                qf = np.sort(qf_device.float().cpu().numpy(), axis=1)
            us = rng.random(n_paths)
            block = self._sampler(qf, us)
            collected[:, step * self._output_len : (step + 1) * self._output_len] = block

            if step < n_steps - 1:
                new = torch.from_numpy(
                    np.ascontiguousarray(
                        block.reshape(n_paths, self._output_patches, self._patch_len)
                    )
                ).to(self.device, dtype=torch.float32)
                qf_device = eng.extend(new)

        return collected

    def _rollout_exact(
        self,
        target: np.ndarray,
        freq: str,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Re-encode the full series each step, reproducing reference shapes."""
        patch_len = self._patch_len
        freq_ids, scalings = self._freq_tensors(freq, n_paths)

        paths = np.repeat(target[None, :], n_paths, axis=0)
        collected = np.empty((n_paths, n_steps * self._output_len), dtype=np.float32)

        for step in range(n_steps):
            values, mask = patchify(paths, patch_len)
            inp, attn_mask = self._to_device(values, mask)
            qf_device = self._last_patch_qf(inp, attn_mask, freq_ids, scalings)
            qf = np.sort(qf_device.float().cpu().numpy(), axis=1)

            us = rng.random(n_paths)
            block = self._sampler(qf, us)
            collected[:, step * self._output_len : (step + 1) * self._output_len] = block

            if step < n_steps - 1:
                paths = np.concatenate([paths, block], axis=1)

        return collected

    def _last_patch_qf(self, inp, mask, freq_ids, scalings) -> torch.Tensor:
        """Forward pass, sliced to the final patch *on device* before the copy.

        The head emits a forecast at every anchor, but a rollout uses only the
        last. Slicing before the transfer moves ~1/n_patches of the data.
        """
        n_patches = inp.shape[1]
        pred, _, _ = self.model(
            inp,
            freq_id=freq_ids,
            scale=scalings,
            attn_mask_target=mask,
            num_input_patches=n_patches,
        )
        b, p, w = pred.shape
        expected = self._output_patches * self._n_quantiles * self._patch_len
        if w != expected:
            raise RuntimeError(
                f"Model head emitted {w} values per patch, but the config expects "
                f"O({self._output_patches}) x Q({self._n_quantiles}) x "
                f"L({self._patch_len}) = {expected}."
            )
        last = pred[:, p - 1].reshape(
            b, self._output_patches, self._n_quantiles, self._patch_len
        )
        return last.permute(0, 2, 1, 3).reshape(b, self._n_quantiles, self._output_len)

    def _rollout_multivariate(
        self,
        target: np.ndarray,
        freq: str,
        n_steps: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Covariate-conditioned rollout.

        The KV cache is univariate-only, so this re-runs the full forecaster
        each step. Drawn blocks are tiled across every channel, which assumes
        covariates track the target — good enough to extend the window, but the
        reason covariate forecasts degrade faster with horizon than univariate
        ones.
        """
        if self._forecaster is None:
            inf_cfg = InferenceConfig.from_model_config(
                self.config, batch_size=n_paths, num_workers=0, device=str(self.device)
            )
            self._forecaster = YugForecaster(model=self.model, cfg=inf_cfg)

        paths = [target.copy() for _ in range(n_paths)]
        collected = np.empty((n_paths, n_steps * self._output_len), dtype=np.float32)

        for step in range(n_steps):
            records = [{"target": p, "freq": freq} for p in paths]
            result = self._forecaster.predict_batch(records, mode="multivariate")
            qf = np.sort(
                np.stack(
                    [result.last_window_forecast(series_i=i) for i in range(n_paths)]
                ),
                axis=1,
            )

            us = rng.random(n_paths)
            block = self._sampler(qf, us)
            collected[:, step * self._output_len : (step + 1) * self._output_len] = block

            if step < n_steps - 1:
                for i in range(n_paths):
                    tile = np.tile(block[i], (paths[i].shape[0], 1))
                    paths[i] = np.concatenate([paths[i], tile], axis=1)

        return collected


# --------------------------------------------------------------------- helpers
def _as_config(config: dict | YugConfig) -> YugConfig:
    return config if isinstance(config, YugConfig) else YugConfig.from_dict(config)


def _as_series_list(context) -> list[np.ndarray]:
    """Normalise the accepted context shapes into a list of float32 arrays."""
    if isinstance(context, np.ndarray):
        if context.ndim == 1:
            return [context.astype(np.float32)]
        if context.ndim == 2:
            return [row.astype(np.float32) for row in context]
        raise ValueError(
            f"context array must be 1-D or 2-D, got {context.ndim} dimensions"
        )

    if isinstance(context, torch.Tensor):
        return _as_series_list(context.detach().cpu().numpy())

    if isinstance(context, (list, tuple)):
        if not context:
            raise ValueError("context is empty")
        first = context[0]
        if np.isscalar(first) or (isinstance(first, np.generic) and np.ndim(first) == 0):
            return [np.asarray(context, dtype=np.float32)]
        return [np.asarray(s, dtype=np.float32) for s in context]

    # pandas Series / DataFrame and anything else array-like.
    arr = np.asarray(getattr(context, "values", context), dtype=np.float32)
    return _as_series_list(arr)


def _broadcast_freq(freq: str | Sequence[str], n: int) -> list[str]:
    if isinstance(freq, str):
        return [freq] * n
    freqs = list(freq)
    if len(freqs) != n:
        raise ValueError(f"got {len(freqs)} freq values for {n} series")
    return freqs
