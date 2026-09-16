from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import xarray as xr
from torch.utils.data import Dataset

from dataset import ERA5ResNetODEDataset
from tools.train.train_capacity_matched_6h import (
    TauRescaleAnd24chWrapper,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


class _SingleWindow(Dataset):
    def __init__(self, tau_hour: int):
        self.tau_hour = int(tau_hour)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        assert index == 0
        field = torch.arange(27 * 2 * 4, dtype=torch.float32).reshape(
            27,
            2,
            4,
        )
        return {
            "x0": field,
            "x1": field + 1.0,
            "target": field + 0.5,
            "tau": torch.tensor([self.tau_hour / 6.0]),
            "tau_hour": torch.tensor([self.tau_hour]),
        }


@pytest.mark.parametrize(
    ("window_hours", "tau_hour", "expected_tau"),
    ((6, 2, 1.0 / 3.0), (12, 4, 1.0 / 3.0), (12, 8, 2.0 / 3.0)),
)
def test_tau_wrapper_uses_anchor_interval_and_keeps_paper_channels(
    window_hours: int,
    tau_hour: int,
    expected_tau: float,
) -> None:
    wrapped = TauRescaleAnd24chWrapper(
        _SingleWindow(tau_hour),
        delta_t=window_hours,
    )

    sample = wrapped[0]

    torch.testing.assert_close(
        sample["tau"],
        torch.tensor([expected_tau]),
    )
    for key in ("x0", "x1", "target"):
        assert sample[key].shape == (24, 2, 4)


def test_zarr_dataset_normalizes_tau_by_configured_anchor_interval(
    tmp_path,
    monkeypatch,
) -> None:
    weather = xr.Dataset(
        coords={
            "time": range(48),
            "level": [1000],
            "latitude": [0.0],
            "longitude": [0.0],
        }
    )
    monkeypatch.setattr(xr, "open_zarr", lambda *args, **kwargs: weather)
    stats_path = tmp_path / "stats.nc"
    xr.Dataset(
        {
            "climate_statistics": (
                ("stats", "params"),
                [[0.0], [1.0]],
            )
        },
        coords={"stats": ["mean", "std"], "params": ["T1000"]},
    ).to_netcdf(stats_path)

    dataset = ERA5ResNetODEDataset(
        data_dir=str(tmp_path),
        years=[2020],
        variables=["T"],
        pressure_levels=[1000],
        max_tau_hours=12,
        samples_per_date=1,
        train=False,
        eval_hours=[4, 8],
        stats_path=str(stats_path),
    )

    assert dataset.active_hours == [4, 8]
    assert dataset.active_taus == pytest.approx([1.0 / 3.0, 2.0 / 3.0])
    assert sorted({entry[3] for entry in dataset.index}) == pytest.approx(
        [1.0 / 3.0, 2.0 / 3.0]
    )


def test_memmap_dataset_normalizes_tau_by_configured_anchor_interval(
    tmp_path,
) -> None:
    memmap_root = tmp_path / "memmap"
    memmap_root.mkdir()
    shape = (48, 27, 1, 2)
    array = np.memmap(
        memmap_root / "wb2_2020.bin",
        dtype=np.float32,
        mode="w+",
        shape=shape,
    )
    array[:] = 0.0
    array.flush()
    (memmap_root / "wb2_2020.json").write_text(
        json.dumps(
            {
                "T": shape[0],
                "H": shape[2],
                "W": shape[3],
                "n_channels": shape[1],
            }
        )
    )
    pressure_channels = [
        f"{variable}{level}"
        for variable in ("T", "U", "V", "Q", "Z")
        for level in (1000, 925, 850, 700)
    ]
    stats_path = tmp_path / "stats.nc"
    xr.Dataset(
        {
            "climate_statistics": (
                ("stats", "params"),
                np.stack(
                    (
                        np.zeros(len(pressure_channels), dtype=np.float32),
                        np.ones(len(pressure_channels), dtype=np.float32),
                    )
                ),
            )
        },
        coords={"stats": ["mean", "std"], "params": pressure_channels},
    ).to_netcdf(stats_path)
    surface_path = tmp_path / "surface.json"
    surface_path.write_text(
        json.dumps(
            {
                channel: {"mean": 0.0, "std": 1.0}
                for channel in ("t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv")
            }
        )
    )

    dataset = ERA5MemmapDataset(
        memmap_dir=str(memmap_root),
        years=[2020],
        max_tau_hours=12,
        samples_per_date=1,
        train=False,
        eval_hours=[4, 8],
        stats_path=str(stats_path),
        surface_stats_path=str(surface_path),
    )

    assert sorted({entry[3] for entry in dataset.index}) == pytest.approx(
        [1.0 / 3.0, 2.0 / 3.0]
    )
