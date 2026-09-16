"""Smoke: instantiate every conf/data/*.yaml against the synth fixture."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("data_name", ["era5_0p5_6h"])
def test_data_compose(hydra_cfg, data_name: str):
    """The data group composes without errors and has the expected fields."""
    cfg = hydra_cfg([f"data={data_name}"])
    assert cfg.data.name == data_name
    assert cfg.data.max_tau_hours >= 1
    assert cfg.data.delta_t_hours > 0
    assert isinstance(list(cfg.data.years), list)
    assert isinstance(list(cfg.data.train_hours), list)


def test_data_6h_loader_instantiate(hydra_cfg):
    """ERA5MemmapDataset can be built from the era5_0p5_6h cfg + synth memmap."""
    cfg = hydra_cfg(["data=era5_0p5_6h"])

    from weather_time_interp.memmap_dataset import ERA5MemmapDataset

    ds = ERA5MemmapDataset(
        memmap_dir=str(cfg.data.memmap_dir),
        years=list(cfg.data.years),
        max_tau_hours=int(cfg.data.max_tau_hours),
        samples_per_date=int(cfg.data.samples_per_date),
        train=True,
        train_hours=list(cfg.data.train_hours),
        static_path=str(cfg.data.static_path),
        stats_path=str(cfg.data.stats_path),
        surface_stats_path=str(cfg.data.surface_stats_path),
    )
    assert len(ds) > 0, "synth memmap should produce >0 windows"
    sample = ds[0]
    # Check expected keys + tensor shapes.
    for key in ("x0", "x1", "target", "tau"):
        assert key in sample, f"missing key {key} from dataset sample"

    # Channels-first format.
    assert sample["x0"].dim() == 3
    n_ch = sample["x0"].shape[0]
    assert n_ch in (24, 27), f"unexpected channel count {n_ch}"


def test_data_12h_compose(hydra_cfg):
    cfg = hydra_cfg(["data=era5_0p5_12h"])
    assert cfg.data.delta_t_hours == 12.0
    assert 4 in list(cfg.data.eval_hours)  # held-out tau is in eval set
    assert 4 not in list(cfg.data.train_hours)  # but NOT in train set (oddskip)


def test_release_memmap_pages_preserves_source_values(
    tmp_path: Path,
) -> None:
    from weather_time_interp.memmap_dataset import release_memmap_time_slices

    path = tmp_path / "samples.bin"
    shape = (8, 3, 16, 16)
    array = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
    array[:] = np.arange(array.size, dtype=np.float32).reshape(shape)
    array.flush()
    expected = np.array(array[3], copy=True)

    assert release_memmap_time_slices(array, (3,))
    assert np.array_equal(np.array(array[3], copy=True), expected)
