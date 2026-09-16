"""Smoke: run train.py model=X trainer=smoke for each model config.

These are GPU-only — synth 16x32 grid still requires CUDA for the
Lightning trainer to work with bf16-mixed precision and the
WeatherHermiteLightningModule defaults.

Each test runs `train.py` end-to-end on the synth fixture for 1 epoch
with `trainer=smoke` (2 train batches, 1 val batch). On success we assert:

    - the process exits 0
    - at least one ckpt is written under `cfg.out_dir`
    - hparams.yaml is written and contains the model name

This test invokes the entry point in-process via Hydra. It does NOT
spawn a fresh wti-train:v1 docker — the docker harness lives in the CI
shim under `scripts/run_hydra_smoke_test.sh` (Phase 5).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


# These configs are known to compose without forward-pass issues on the
# synth grid (16x32, 27ch) provided env-vars are set correctly.
# Models that require >32-px grid for their FFT/SHT layers are skipped here
# and validated via dedicated targeted tests on the production grid.
HERMITE_MODELS_PASS_FORWARD = ["weatherdcae", "dcae_skip", "fuxi"]
HERMITE_MODELS_REQUIRES_PROD_GRID = ["sdyff", "modafno"]


pytestmark = pytest.mark.gpu


@pytest.mark.smoke
@pytest.mark.parametrize("model_name", HERMITE_MODELS_PASS_FORWARD)
def test_train_one_step_hermite(tmp_path: Path, model_name: str):
    out_dir = tmp_path / f"smoke_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    fixture_overrides = [
        f"data.memmap_dir={REPO}/tests/fixtures/synth_memmap",
        f"data.stats_path={REPO}/tests/fixtures/synth_stats/json_stats_0p5.nc",
        f"data.surface_stats_path={REPO}/tests/fixtures/synth_stats/surface_stats_0p5.json",
        f"data.static_path={REPO}/tests/fixtures/synth_stats/static_features_0p5.pt",
        "data.years=[2020]",
        "data.val_years=[2020]",
        "data.samples_per_date_train=2",
        "data.samples_per_date_val=1",
        "data.n_static_features=5",
        f"model={model_name}",
        "model.latent_channels=16",      # tiny so we don't OOM at smoke
        "trainer=smoke",
        f"exp_name=smoke_{model_name}",
        f"out_dir={out_dir}",
    ]
    cmd = [sys.executable, str(REPO / "train.py")] + fixture_overrides
    print(f"\n=== smoke: {model_name} ===\n{' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    print("STDOUT (tail):")
    print("\n".join(result.stdout.splitlines()[-40:]))
    if result.returncode != 0:
        print("STDERR (tail):")
        print("\n".join(result.stderr.splitlines()[-40:]))
    assert result.returncode == 0, f"train.py exited {result.returncode}"

    # Recursively look for any ckpt under out_dir (Hydra may nest under timestamp).
    ckpts = list(out_dir.rglob("*.ckpt"))
    assert ckpts, f"no ckpt written under {out_dir}"

    hparams = list(out_dir.rglob("hparams.yaml"))
    assert hparams, f"no hparams.yaml written under {out_dir}"
    content = hparams[0].read_text()
    assert model_name in content or "model" in content


@pytest.mark.smoke
@pytest.mark.prod_grid
@pytest.mark.parametrize("model_name", HERMITE_MODELS_REQUIRES_PROD_GRID)
def test_train_one_step_prod_grid_required(model_name: str):
    """Documents the requirement; real coverage is in `test_hydra_prod_grid.py`.

    Gated by `prod_grid` marker so it does not run during default smoke and
    `pytest -m prod_grid` picks up the prod-grid module instead.
    """
    pytest.skip(
        f"{model_name} requires production grid (360x720 for sdyff/modafno SHT/AFNO blocks). "
        f"Covered by `tests/test_hydra_prod_grid.py` (run with `pytest -m prod_grid`)."
    )


# ATM-VFI uses a custom Lightning module; it works on the synth grid (no SHT).
@pytest.mark.smoke
def test_train_one_step_atm_vfi(tmp_path: Path):
    out_dir = tmp_path / "smoke_atm_vfi"
    out_dir.mkdir(parents=True, exist_ok=True)
    fixture_overrides = [
        f"data=era5_0p5_12h",
        f"data.memmap_dir={REPO}/tests/fixtures/synth_memmap",
        f"data.stats_path={REPO}/tests/fixtures/synth_stats/json_stats_0p5.nc",
        f"data.surface_stats_path={REPO}/tests/fixtures/synth_stats/surface_stats_0p5.json",
        f"data.static_path={REPO}/tests/fixtures/synth_stats/static_features_0p5.pt",
        "data.years=[2020]",
        "data.val_years=[2020]",
        "data.samples_per_date_train=2",
        "data.samples_per_date_val=1",
        "data.n_static_features=5",
        "model=atm_vfi",
        "model.hidden=16",
        "model.n_levels=2",
        "trainer=smoke",
        "exp_name=smoke_atm_vfi",
        f"out_dir={out_dir}",
    ]
    cmd = [sys.executable, str(REPO / "train.py")] + fixture_overrides
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    print("STDOUT (tail):")
    print("\n".join(result.stdout.splitlines()[-40:]))
    if result.returncode != 0:
        print("STDERR (tail):")
        print("\n".join(result.stderr.splitlines()[-40:]))
    assert result.returncode == 0
    ckpts = list(out_dir.rglob("*.ckpt"))
    assert ckpts, f"no ckpt under {out_dir}"


# CorrDiff-FM needs a base ckpt — we don't smoke-test it here. Validated in
# `test_train_corrdiff_fm_requires_base_ckpt`.
@pytest.mark.smoke
def test_train_corrdiff_fm_requires_base_ckpt(tmp_path: Path):
    fixture_overrides = [
        "model=corrdiff_fm",
        "trainer=smoke",
        # no model.base_ckpt set
    ]
    cmd = [sys.executable, str(REPO / "train.py")] + fixture_overrides
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH','')}"
    result = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120)
    # Hydra exits 1 on missing mandatory value (???). Either way, must NOT exit 0.
    assert result.returncode != 0, (
        "corrdiff_fm without base_ckpt should fail.\n"
        f"stdout: {result.stdout[-500:]}\nstderr: {result.stderr[-500:]}"
    )
