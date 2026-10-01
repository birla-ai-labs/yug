# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Configuration behaviour: derived values, validation, round-tripping."""

from __future__ import annotations

import json

import pytest

from yug import InferenceConfig, PredictionConfig, YugConfig


def test_defaults_match_the_released_architecture():
    cfg = YugConfig()
    assert cfg.patch_len == 16
    assert cfg.output_dim == 64
    assert cfg.d_model == 960
    assert cfg.num_decoder_layers == 12
    assert cfg.expansion == 1.25
    assert cfg.quantiles == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def test_derived_geometry():
    cfg = YugConfig()
    assert cfg.output_patches == 4
    assert cfg.num_quantiles == 9
    # The head emits output_patches x quantiles x patch_len values per anchor.
    assert cfg.head_width == 4 * 9 * 16


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"quantiles": []}, "non-empty"),
        ({"quantiles": [0.9, 0.1]}, "ascending"),
        ({"output_dim": 60, "patch_len": 16}, "multiple"),
        ({"d_model": 100, "num_attn_heads": 8}, "divisible"),
    ],
)
def test_invalid_configs_are_rejected_at_construction(kwargs, message):
    with pytest.raises(ValueError, match=message):
        YugConfig(**kwargs)


def test_arch_dict_is_exactly_what_the_network_reads():
    cfg = YugConfig()
    arch = cfg.to_arch_dict()
    # Metadata must not leak into the module constructors.
    assert "model_type" not in arch
    assert set(arch) == {
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
    }


def test_json_round_trip(tmp_path):
    cfg = YugConfig(d_model=64, hidden_dim=64, num_attn_heads=8)
    path = tmp_path / "config.json"
    cfg.to_json_file(path)
    assert json.loads(path.read_text())["d_model"] == 64
    assert YugConfig.from_json_file(path) == cfg


def test_unknown_keys_are_ignored_so_newer_configs_still_load():
    cfg = YugConfig.from_dict(
        {"d_model": 960, "transformers_version": "9.9", "future_field": 1}
    )
    assert cfg.d_model == 960


def test_inference_config_derives_from_the_architecture():
    model_cfg = YugConfig()
    inf = InferenceConfig.from_model_config(model_cfg, batch_size=7, device="cpu")
    assert inf.patch_len == model_cfg.patch_len
    assert inf.output_len == model_cfg.output_dim
    assert inf.quantiles == model_cfg.quantiles
    assert inf.output_patches == model_cfg.output_patches
    assert inf.batch_size == 7


@pytest.mark.parametrize(
    "kwargs", [{"engine": "turbo"}, {"pad_side": "middle"}, {"num_samples": 0}]
)
def test_prediction_config_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        PredictionConfig(**kwargs)


def test_default_context_length_is_the_trained_window():
    """2048 is what Yug was trained on and what its published scores used."""
    assert PredictionConfig().context_length == 2048


@pytest.mark.network
def test_staged_hub_config_matches_the_released_defaults():
    """The config.json we publish must describe the architecture the code builds.

    These two drift apart silently — someone edits a default here, or the JSON
    there — and the failure only shows up as a load error for users. Catch it
    against the real Hub config, cached locally like any other download.
    """
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    staged = Path(hf_hub_download("birlaailabs/yug", "config.json"))

    published = YugConfig.from_json_file(staged)
    assert published.to_arch_dict() == YugConfig().to_arch_dict()

    # The Hub needs these for its own tooling; from_dict ignores them, so they
    # would not otherwise be checked.
    raw = json.loads(staged.read_text())
    assert raw["architectures"] == ["Yug_Model"]
    assert raw["library_name"] == "yug"
