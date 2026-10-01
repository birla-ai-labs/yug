# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures.

Every fixture here is tiny, synthetic, CPU-only and offline. The suite must
stay runnable on a laptop with no GPU and no network, because that is what CI
runs on and what a contributor has in front of them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from yug import YugConfig, YugPipeline
from yug.model import Yug_Model


@pytest.fixture(scope="session")
def tiny_config() -> YugConfig:
    """A model small enough to run in milliseconds, same shape as the real one."""
    return YugConfig(
        patch_len=8,
        output_dim=16,
        hidden_dim=32,
        d_model=32,
        freq_num=9,
        num_attn_heads=4,
        dropout=0.0,
        eps=1e-6,
        kernel_size=3,
        num_decoder_layers=2,
        quantiles=[0.1, 0.5, 0.9],
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: YugConfig) -> Yug_Model:
    torch.manual_seed(0)
    return Yug_Model(config=tiny_config.to_arch_dict()).eval()


@pytest.fixture(scope="session")
def staged_checkpoint(tmp_path_factory, tiny_model, tiny_config):
    """A directory laid out exactly like a published Hub repo.

    Exercising ``from_pretrained`` against this is what proves the load path
    works without downloading anything.
    """
    out = tmp_path_factory.mktemp("checkpoint")
    YugPipeline(tiny_model, tiny_config, device="cpu").save_pretrained(out)
    return out


@pytest.fixture
def pipeline(staged_checkpoint) -> YugPipeline:
    return YugPipeline.from_pretrained(staged_checkpoint, device_map="cpu")


@pytest.fixture(scope="session")
def series() -> np.ndarray:
    """A seasonal series with trend — 256 points, cleanly patch-aligned."""
    t = np.arange(256, dtype=np.float32)
    return (5.0 * np.sin(2 * np.pi * t / 32.0) + 0.02 * t + 10.0).astype(np.float32)
