# Copyright 2026 Birla AI Labs
# SPDX-License-Identifier: Apache-2.0
"""Architecture invariants.

These are the properties the rest of the package is built on — most importantly
patch-level causality, which is the entire justification for the cached rollout
engine. If any of these break, the fast engine is silently wrong.
"""

from __future__ import annotations

import torch

from yug import YugConfig
from yug.model import Yug_Model


def _inputs(cfg: YugConfig, batch=2, patches=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, patches, cfg.patch_len, generator=g)
    mask = torch.ones(batch, patches, cfg.patch_len, dtype=torch.bool)
    return x, mask


def test_forward_emits_a_forecast_block_at_every_anchor(tiny_model, tiny_config):
    x, mask = _inputs(tiny_config, patches=6)
    with torch.inference_mode():
        pred, mean, std = tiny_model(
            x,
            attn_mask_target=mask,
            freq_id=torch.tensor([3, 3]),
            scale=torch.tensor([1, 1]),
            num_input_patches=6,
        )
    assert pred.shape == (2, 6, tiny_config.head_width)
    assert mean.shape == std.shape == (2, 6, 1)
    assert torch.isfinite(pred).all()


def test_output_is_patch_level_causal(tiny_model, tiny_config):
    """Appending patches must not disturb earlier anchors.

    This is what lets yug.engine encode a context once and reuse its
    K/V. Exact equality is unreachable — cuBLAS picks different reduction
    orders per shape — so the bar is the fp32 kernel-noise floor.
    """
    x, mask = _inputs(tiny_config, patches=6, seed=1)
    extra = torch.randn(
        2, 2, tiny_config.patch_len, generator=torch.Generator().manual_seed(9)
    )
    x_long = torch.cat([x, extra], dim=1)
    mask_long = torch.ones(2, 8, tiny_config.patch_len, dtype=torch.bool)

    kw = {"freq_id": torch.tensor([3, 3]), "scale": torch.tensor([1, 1])}
    with torch.inference_mode():
        short, _, _ = tiny_model(x, attn_mask_target=mask, num_input_patches=6, **kw)
        long, _, _ = tiny_model(
            x_long, attn_mask_target=mask_long, num_input_patches=8, **kw
        )

    drift = (short - long[:, :6]).abs().max().item()
    scale = short.abs().max().item()
    assert drift <= 1e-4 * max(scale, 1.0), f"causality violated: drift={drift}"


def test_masked_positions_do_not_leak_into_the_statistics(tiny_model, tiny_config):
    """Changing a masked-out value must not change the forecast.

    Padding sentinels are extreme by design; if they reached the running mean
    and standard deviation, every padded series would be normalised wrongly.
    """
    x, mask = _inputs(tiny_config, patches=4, seed=2)
    mask[:, 0, :] = False

    polluted = x.clone()
    polluted[:, 0, :] = 111181.0  # the sentinel the batching path writes

    kw = {
        "freq_id": torch.tensor([3, 3]),
        "scale": torch.tensor([1, 1]),
        "num_input_patches": 4,
    }
    with torch.inference_mode():
        a, _, _ = tiny_model(x, attn_mask_target=mask, **kw)
        b, _, _ = tiny_model(polluted, attn_mask_target=mask, **kw)

    assert torch.allclose(a, b, atol=1e-4), "a masked value changed the output"


def test_num_input_patches_freezes_the_normalisation_statistics(tiny_model, tiny_config):
    """Generated patches stay attention tokens but must not move the scale.

    Without this, a rollout renormalises against its own output and drifts.
    """
    x, mask = _inputs(tiny_config, patches=6, seed=3)
    kw = {"freq_id": torch.tensor([3, 3]), "scale": torch.tensor([1, 1])}
    with torch.inference_mode():
        _, mean_frozen, std_frozen = tiny_model(
            x, attn_mask_target=mask, num_input_patches=3, **kw
        )
        _, mean_all, std_all = tiny_model(
            x, attn_mask_target=mask, num_input_patches=6, **kw
        )

    # Statistics agree up to the freeze point and diverge after it.
    assert torch.allclose(mean_frozen[:, :3], mean_all[:, :3], atol=1e-5)
    assert not torch.allclose(mean_frozen[:, 3:], mean_all[:, 3:], atol=1e-5)
    # Past the freeze the statistics simply hold their last observed value.
    assert torch.allclose(
        std_frozen[:, 3:], std_frozen[:, 2:3].expand_as(std_frozen[:, 3:]), atol=1e-5
    )


def test_multivariate_forward_runs_and_matches_univariate_shape(tiny_model, tiny_config):
    x, mask = _inputs(tiny_config, patches=5, seed=4)
    g = torch.Generator().manual_seed(5)
    variates = torch.randn(2, 3, 5, tiny_config.patch_len, generator=g)
    var_mask = torch.ones(2, 3, 5, tiny_config.patch_len, dtype=torch.bool)

    with torch.inference_mode():
        pred, _, _ = tiny_model(
            x,
            variates,
            attn_mask_target=mask,
            attn_mask_variates=var_mask,
            freq_id=torch.tensor([3, 3]),
            scale=torch.tensor([1, 1]),
            num_input_patches=5,
        )
    assert pred.shape == (2, 5, tiny_config.head_width)
    assert torch.isfinite(pred).all()


def test_output_patch_norm_is_retained_for_checkpoint_compatibility(tiny_model):
    """It carries trained parameters even though forward does not call it.

    Dropping the module would make load_state_dict reject every published
    checkpoint, so its presence is deliberate and load-bearing.
    """
    assert "output_patch_norm.weight" in tiny_model.state_dict()


def test_model_accepts_a_config_object_or_a_plain_dict(tiny_config):
    from_obj = Yug_Model(config=tiny_config)
    from_dict = Yug_Model(config=tiny_config.to_arch_dict())
    assert from_obj.state_dict().keys() == from_dict.state_dict().keys()


def test_released_architecture_parameter_count_is_stable():
    """A change here means the published weights no longer fit the code."""
    model = Yug_Model(config=YugConfig().to_arch_dict())
    assert sum(p.numel() for p in model.parameters()) == 271_843_820
