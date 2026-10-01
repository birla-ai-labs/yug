# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration objects for Yug.

Three configs, deliberately separated by lifetime:

``YugConfig``
    The *architecture*. Fixed at training time, published as ``config.json``
    next to the weights. Changing any field invalidates a checkpoint.

``InferenceConfig``
    How a batch is fed to the network (device, batch size, workers). Free to
    change per run; has no effect on the numbers the model produces.

``PredictionConfig``
    How the autoregressive rollout is driven (context window, number of sampled
    paths, RNG seed, engine). Affects the forecast, but not the weights.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

__all__ = ["YugConfig", "InferenceConfig", "PredictionConfig", "default_device"]


def default_device() -> str:
    """``"cuda"`` when an accelerator is present, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


# The exact key set the architecture reads. Anything else in a config.json is
# metadata (``model_type``, ``architectures``, ``transformers_version``, ...)
# and is carried alongside rather than passed into the modules.
_ARCH_KEYS = (
    "patch_len",
    "output_dim",
    "hidden_dim",
    "d_model",
    "freq_num",
    "num_attn_heads",
    "dropout",
    "eps",
    "kernel_size",
    "num_decoder_layers",
    "expansion",
    "quantiles",
)


@dataclass
class YugConfig:
    """Architecture hyper-parameters. Serialised as the Hub's ``config.json``.

    ``quantiles`` is not a runtime choice: the output head emits
    ``output_dim * len(quantiles)`` values per patch, so its width is frozen
    when the checkpoint is trained.

    ``kernel_size`` is carried for checkpoint-metadata fidelity; the current
    decoder does not read it.

    ``expansion`` is the MLP hidden-dim multiplier. It may be fractional (the
    V19 checkpoint trains at 1.25); :class:`~yug.model.Yug_MLP` rounds
    ``hidden_size * expansion`` to the nearest int for the actual layer width.
    """

    patch_len: int = 16
    output_dim: int = 64
    hidden_dim: int = 960
    d_model: int = 960
    freq_num: int = 9
    num_attn_heads: int = 8
    dropout: float = 0.1
    eps: float = 1e-6
    kernel_size: int = 3
    num_decoder_layers: int = 12
    expansion: float = 1.25
    quantiles: list[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )

    model_type: str = "yug"

    def __post_init__(self) -> None:
        self.quantiles = [float(q) for q in self.quantiles]
        if not self.quantiles:
            raise ValueError("quantiles must be a non-empty list")
        if self.quantiles != sorted(self.quantiles):
            raise ValueError(f"quantiles must be ascending, got {self.quantiles}")
        if self.output_dim % self.patch_len:
            raise ValueError(
                f"output_dim ({self.output_dim}) must be a multiple of "
                f"patch_len ({self.patch_len})"
            )
        if self.d_model % self.num_attn_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_attn_heads ({self.num_attn_heads})"
            )

    # ------------------------------------------------------------- derived
    @property
    def output_patches(self) -> int:
        """Patches emitted per forward pass (``output_dim // patch_len``)."""
        return self.output_dim // self.patch_len

    @property
    def num_quantiles(self) -> int:
        return len(self.quantiles)

    @property
    def head_width(self) -> int:
        """Values the output head emits per input patch."""
        return self.output_patches * self.num_quantiles * self.patch_len

    # --------------------------------------------------------- (de)serialise
    def to_dict(self) -> dict[str, Any]:
        """Full dict, including metadata keys. Written to ``config.json``."""
        return asdict(self)

    def to_arch_dict(self) -> dict[str, Any]:
        """Only the keys the network modules consume.

        This is the dict handed to :class:`~yug.model.Yug_Model`;
        keeping it exact is what makes released checkpoints loadable.
        """
        return {k: getattr(self, k) for k in _ARCH_KEYS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> YugConfig:
        """Build from a dict, ignoring keys this version does not know about."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_json_file(cls, path: str | Path) -> YugConfig:
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_json_file(self, path: str | Path, indent: int = 2) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=indent)
            fh.write("\n")


@dataclass
class InferenceConfig:
    """Batching and placement. Does not change the forecast values.

    ``pad_token`` is the sentinel the batching path writes into trailing
    positions; those positions are masked out, so its value never reaches the
    loss or the output — it only has to be distinguishable.
    """

    patch_len: int = 16
    output_len: int = 64
    pad_token: float = 111181.0
    batch_size: int = 32
    num_workers: int = 0
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.3, 0.5, 0.9])
    device: str = field(default_factory=default_device)

    @property
    def output_patches(self) -> int:
        return self.output_len // self.patch_len

    @classmethod
    def from_model_config(cls, config: YugConfig, **overrides: Any) -> InferenceConfig:
        """Derive the batching config from an architecture config."""
        base: dict[str, Any] = {
            "patch_len": config.patch_len,
            "output_len": config.output_dim,
            "quantiles": list(config.quantiles),
        }
        base.update(overrides)
        return cls(**base)


@dataclass
class PredictionConfig:
    """Rollout behaviour for :meth:`YugPipeline.predict`.

    ``num_samples``
        Independent sampled trajectories. The returned quantile band is the
        empirical quantile across these paths, so raising it narrows Monte
        Carlo noise at linear cost.

    ``engine``
        ``"cached"`` encodes the context once and reuses its per-layer K/V,
        pushing only newly generated patches through the network (~9x faster).
        ``"exact"`` reproduces the reference implementation's tensor shapes and
        is bitwise identical to it. See :mod:`yug.engine`.

    ``pad_side``
        Series are grown to a whole number of patches with masked sentinels.
        ``"left"`` keeps the forecast origin on the final real observation and
        is the only side you should normally use; ``"right"`` places the origin
        ``pad_len`` steps past the last real point.
    """

    context_length: int | None = 8192
    prediction_length: int | None = None
    num_samples: int = 100
    engine: str = "cached"
    pad_side: str = "left"
    seed: int = 0
    quantile_levels: list[float] | None = None

    def __post_init__(self) -> None:
        if self.engine not in ("cached", "exact"):
            raise ValueError(f"engine must be 'cached' or 'exact', got {self.engine!r}")
        if self.pad_side not in ("left", "right"):
            raise ValueError(f"pad_side must be 'left' or 'right', got {self.pad_side!r}")
        if self.num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got {self.num_samples}")
