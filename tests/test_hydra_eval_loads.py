"""Smoke: instantiate every conf/eval/*.yaml; verify Hydra→argv translation.

We don't actually run the eval — that requires production climatology zarr
and trained ckpts. We just verify that:

    1. Each eval cfg composes.
    2. `eval.py:_build_argv(cfg)` produces a non-empty argv with the
       expected flags for the underlying script.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from eval import _build_argv, _normalize_models  # noqa: E402


ALL_EVAL_KINDS = [
    "memmap_6h",
    "memmap_12h",
    "crps_ensemble",
    "sh_spectra",
    "region_season_6h",
    "region_season_12h",
    "sh_spectra_12h",
    "sh_spectra_ens",
    "bootstrap_ci",
]


@pytest.mark.parametrize("eval_name", ALL_EVAL_KINDS)
def test_eval_compose(hydra_cfg, eval_name: str):
    overrides = [f"eval={eval_name}"]
    if eval_name in ("memmap_6h", "memmap_12h",
                     "region_season_6h", "region_season_12h"):
        overrides.append("++eval.models='m1:logs/foo/last.ckpt'")
    elif eval_name == "crps_ensemble":
        overrides.extend([
            "++eval.fm_ckpt=logs/foo/fm.ckpt",
            "++eval.base_ckpt=logs/foo/base.ckpt",
        ])
    elif eval_name in ("sh_spectra", "sh_spectra_ens"):
        overrides.append("++eval.ckpt=logs/foo/last.ckpt")
    elif eval_name == "sh_spectra_12h":
        overrides.extend([
            "++eval.ckpt=logs/foo/last.ckpt",
            "++eval.model_name=test_model",
        ])
    # bootstrap_ci needs no required overrides — uses defaults.

    cfg = hydra_cfg(overrides, config_name="eval")
    assert cfg.eval.name == eval_name
    assert cfg.eval.kind == eval_name

    argv = _build_argv(cfg)
    assert argv, f"empty argv for {eval_name}"
    flat = " ".join(argv)
    print(f"{eval_name}: argv = {flat}")

    if eval_name == "memmap_6h":
        assert "--memmap-dir" in argv
        assert "--out-acc-dir" in argv
        assert "--out-rmse-dir" in argv
    elif eval_name == "memmap_12h":
        assert "--out-dir" in argv
        assert "--paper-tag" in argv
        assert "--eval-hours" in argv
    elif eval_name == "crps_ensemble":
        assert "--fm_ckpt" in argv
        assert "--base_ckpt" in argv
        assert "--mode" in argv
    elif eval_name in ("sh_spectra", "sh_spectra_ens"):
        assert "--ckpt" in argv
        assert "--lmax" in argv
    elif eval_name == "region_season_6h":
        assert "--memmap-dir" in argv
        assert "--models" in argv
        assert "--out-dir" in argv
        # 6h variant does NOT pass --max-tau-hours / --eval-hours / --keep-n-channels
        assert "--max-tau-hours" not in argv
        assert "--eval-hours" not in argv
        assert "--keep-n-channels" not in argv
    elif eval_name == "region_season_12h":
        assert "--memmap-dir" in argv
        assert "--models" in argv
        assert "--max-tau-hours" in argv
        assert "--eval-hours" in argv
        assert "--keep-n-channels" in argv
    elif eval_name == "sh_spectra_12h":
        assert "--ckpt" in argv
        assert "--taus" in argv
        assert "--model-name" in argv
        assert "--lmax" in argv
    elif eval_name == "bootstrap_ci":
        assert "--bootstraps" in argv
        assert "--seed" in argv


@pytest.mark.parametrize("horizon", ["6h", "12h", "legacy"])
def test_eval_bootstrap_ci_horizon(hydra_cfg, horizon: str):
    cfg = hydra_cfg(
        ["eval=bootstrap_ci", f"++eval.horizon={horizon}"],
        config_name="eval",
    )
    argv = _build_argv(cfg)
    assert argv, f"empty argv for bootstrap_ci horizon={horizon}"
    assert "--bootstraps" in argv
    if horizon == "6h":
        # both out-24ch/out-27ch are optional unless bs_out_24ch/27 set
        # but --seed should always be there
        assert "--seed" in argv
    elif horizon == "12h":
        assert "--seed" in argv
    elif horizon == "legacy":
        # Only --bootstraps (and optional --out) are produced
        pass


def test_normalize_models_string():
    assert _normalize_models("a:p1,b:p2") == "a:p1,b:p2"
    assert _normalize_models("a:p1\n,b:p2 ") == "a:p1,b:p2"


def test_normalize_models_list():
    assert _normalize_models(["a:p1", "b:p2"]) == "a:p1,b:p2"


def test_eval_memmap_6h_missing_models_raises(hydra_cfg):
    """memmap_6h with empty models should raise during argv build."""
    cfg = hydra_cfg(["eval=memmap_6h"], config_name="eval")
    with pytest.raises(ValueError):
        _build_argv(cfg)
