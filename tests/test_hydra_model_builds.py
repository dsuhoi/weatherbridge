"""Smoke: instantiate every conf/model/*.yaml; assert param counts.

These tests do NOT do a forward pass (some models require specific
SDYFF_*/MODAFNO_* env-var grid sizes that we don't have on the synth
16x32 grid). We only verify that:

    1. Hydra composes the cfg.
    2. The cfg has the expected `kind` and `model_type`.
    3. The model env_vars dict is well-formed.

Full forward passes are covered by `test_hydra_train_one_step.py`.
"""
from __future__ import annotations

from typing import Tuple

import pytest


@pytest.mark.parametrize("model_name,expected_kind,expected_type", [
    ("weatherdcae", "hermite", "dcae_adaln_residual_linear"),
    ("dcae_skip", "hermite", "dcae_adaln_skip_residual_linear"),
    ("fuxi", "hermite", "fuxi_swinv2_residual_linear"),
    ("sdyff", "hermite", "sdyff_dyffusion_residual_linear"),
    ("modafno", "hermite", "modafno_residual_linear"),
    ("atm_vfi", "atm_vfi", "pixel_attn_vfi"),
    ("corrdiff_fm", "corrdiff_fm", "corrdiff_fm"),
])
def test_model_compose(hydra_cfg, model_name: str, expected_kind: str, expected_type: str):
    overrides = [f"model={model_name}"]
    if model_name == "corrdiff_fm":
        overrides.append("model.base_ckpt=/tmp/placeholder.ckpt")
    cfg = hydra_cfg(overrides)
    assert cfg.model.name == model_name or model_name in cfg.model.name
    assert cfg.model.kind == expected_kind
    assert cfg.model.model_type == expected_type


@pytest.mark.parametrize("model_name,expected_envs", [
    ("sdyff", {"SDYFF_NLAT", "SDYFF_NLON", "SDYFF_LAT_CROP"}),
    ("modafno", {"MODAFNO_INP_H", "MODAFNO_INP_W",
                  "MODAFNO_NATIVE_H", "MODAFNO_NATIVE_W"}),
    ("weatherdcae", set()),
    ("dcae_skip", set()),
    ("fuxi", set()),
])
def test_model_env_vars(hydra_cfg, model_name: str, expected_envs: set):
    cfg = hydra_cfg([f"model={model_name}"])
    actual = set(dict(cfg.model.env_vars or {}).keys())
    assert actual == expected_envs, f"env_vars mismatch for {model_name}: {actual} vs {expected_envs}"


def test_dcae_block_out_channels(hydra_cfg):
    """DC-AE knobs survive composition + override."""
    cfg = hydra_cfg([
        "model=weatherdcae",
        "++model.block_out_channels=[64,128,256]",
        "++model.layers_per_block=[3,3,3]",
    ])
    assert list(cfg.model.block_out_channels) == [64, 128, 256]
    assert list(cfg.model.layers_per_block) == [3, 3, 3]
