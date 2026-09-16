"""Shared pytest fixtures: synth memmap + Hydra config factory for smoke tests."""
from __future__ import annotations

import os
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIX_DIR = Path(__file__).resolve().parent / "fixtures"
BIN_PATH = FIX_DIR / "synth_memmap" / "wb2_2020.bin"
BUILDER = FIX_DIR / "build_synth_memmap.py"

# Ensure repo root on sys.path so `train.py`, `eval.py`, conf/* are importable.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


@pytest.fixture(scope="session", autouse=True)
def _ensure_synth_fixture() -> None:
    """Build the synthetic ERA5 memmap fixture on demand (.bin is gitignored)."""
    if BIN_PATH.exists():
        return
    if not BUILDER.exists():
        return  # caller doesn't need the fixture
    if importlib.util.find_spec("torch") is None:
        return  # Pure protocol tests do not need the tensor-backed fixture.
    subprocess.run([sys.executable, str(BUILDER)], check=True)


@pytest.fixture(scope="session")
def synth_memmap_dir() -> str:
    """Absolute path to the synthetic memmap dir."""
    return str(FIX_DIR / "synth_memmap")


@pytest.fixture(scope="session")
def synth_stats() -> dict:
    """Path overrides pointing at the synthetic stats fixture."""
    return {
        "stats_path": str(FIX_DIR / "synth_stats" / "json_stats_0p5.nc"),
        "surface_stats_path": str(FIX_DIR / "synth_stats" / "surface_stats_0p5.json"),
        "static_path": str(FIX_DIR / "synth_stats" / "static_features_0p5.pt"),
    }


@pytest.fixture()
def hydra_cfg(synth_memmap_dir: str, synth_stats: dict):
    """Factory for a Hydra DictConfig composed against `conf/` with synth
    fixture path overrides applied.

    Usage:
        cfg = hydra_cfg(["model=fuxi", "trainer=smoke"])
    """
    import hydra
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    conf_dir = str(REPO / "conf")

    def _factory(overrides=None, config_name: str = "train"):
        overrides = list(overrides or [])
        # Default fixture overrides — synth memmap + stats + tiny grid.
        fixture_overrides = [
            f"data.memmap_dir={synth_memmap_dir}",
            f"data.stats_path={synth_stats['stats_path']}",
            f"data.surface_stats_path={synth_stats['surface_stats_path']}",
            f"data.static_path={synth_stats['static_path']}",
            "data.years=[2020]",
            "data.val_years=[2020]",
            "data.samples_per_date_train=2",
            "data.samples_per_date_val=1",
            "data.n_static_features=5",  # synth fixture has 5 static channels
        ]
        # `train` config has a trainer group, `eval` does not.
        if config_name == "train":
            fixture_overrides.append("trainer=smoke")
        all_overrides = fixture_overrides + overrides
        # Use a fresh GlobalHydra each call.
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=conf_dir):
            cfg = compose(config_name=config_name, overrides=all_overrides)
        return cfg

    return _factory


def _have_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    return _have_cuda()


# Add markers that skip prod-grid tests by default + on missing GPU.
def pytest_collection_modifyitems(config, items):
    has_gpu = _have_cuda()
    skip_gpu = pytest.mark.skip(reason="no CUDA GPU available")
    # If the user did not explicitly pass `-m prod_grid` (or a marker expression
    # that includes it), skip prod_grid-marked tests. This keeps them out of the
    # default smoke run while still allowing the dedicated `make prod-grid-test`
    # target (and `pytest -m prod_grid`) to pick them up.
    marker_expr = config.getoption("-m", default="") or ""
    prod_grid_requested = "prod_grid" in marker_expr
    skip_prod = pytest.mark.skip(
        reason="prod_grid test — run with `pytest -m prod_grid` (requires real 360x720 memmap)"
    )
    for item in items:
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)
        if "prod_grid" in item.keywords and not prod_grid_requested:
            item.add_marker(skip_prod)
