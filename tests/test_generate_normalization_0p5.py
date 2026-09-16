from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from tools.data.generate_normalization_0p5 import (
    PL_VARIABLES,
    SURFACE_VARIABLES,
    _spatial_stats,
    _streaming_spatial_stats,
    _streaming_spatial_stats_by_level,
    generate,
)


def test_cli_defaults_do_not_target_frozen_training_artifacts() -> None:
    source = Path("tools/data/generate_normalization_0p5.py").read_text()
    assert (
        'default=Path("metrics/normalization_replay_v1/generated_stats.nc")'
        in source
    )
    assert (
        'default=Path("metrics/normalization_replay_v1/generated_surface.json")'
        in source
    )


def test_streaming_spatial_stats_matches_array_reference() -> None:
    rng = np.random.default_rng(20260820)
    values = rng.normal(size=(4, 7, 5, 8))
    values[1, 3, 2, 4] = np.nan
    latitude = np.linspace(80.0, -80.0, values.shape[-2])
    weights = np.cos(np.deg2rad(latitude))
    weights /= weights.sum()
    data = xr.DataArray(
        values,
        dims=("hour", "dayofyear", "latitude", "longitude"),
    )

    expected = _spatial_stats(values, weights, values.shape[-1])
    actual = _streaming_spatial_stats(
        data,
        weights,
        values.shape[-1],
        day_chunk=3,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_streaming_pressure_stats_match_per_level_reference() -> None:
    rng = np.random.default_rng(20260821)
    values = rng.normal(size=(4, 7, 3, 5, 8)).astype(np.float32)
    values[2, 5, 1, 3, 6] = np.nan
    latitude = np.linspace(80.0, -80.0, values.shape[-2])
    weights = np.cos(np.deg2rad(latitude))
    weights /= weights.sum()
    data = xr.DataArray(
        values,
        dims=("hour", "dayofyear", "level", "latitude", "longitude"),
    )

    expected = [
        _spatial_stats(values[:, :, level], weights, values.shape[-1])
        for level in range(values.shape[2])
    ]
    actual = _streaming_spatial_stats_by_level(
        data,
        weights,
        values.shape[-1],
        day_chunk=3,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-10)


def test_generate_writes_named_statistics_coordinate(tmp_path, monkeypatch) -> None:
    hour = np.arange(4)
    day = np.arange(366)
    level = np.array([1000, 925, 850, 700])
    latitude = np.array([45.0, -45.0])
    longitude = np.array([0.0, 180.0])
    pressure_shape = (4, 366, 4, 2, 2)
    surface_shape = (4, 366, 2, 2)
    data_vars = {
        name: (
            ("hour", "dayofyear", "level", "latitude", "longitude"),
            np.arange(np.prod(pressure_shape), dtype=np.float32).reshape(
                pressure_shape
            ),
        )
        for name in PL_VARIABLES
    }
    data_vars.update(
        {
            name: (
                ("hour", "dayofyear", "latitude", "longitude"),
                np.arange(np.prod(surface_shape), dtype=np.float32).reshape(
                    surface_shape
                ),
            )
            for name in SURFACE_VARIABLES
        }
    )
    dataset = xr.Dataset(
        data_vars,
        coords={
            "hour": hour,
            "dayofyear": day,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
    monkeypatch.setattr(xr, "open_zarr", lambda *_args, **_kwargs: dataset)
    pressure = tmp_path / "stats.nc"
    surface = tmp_path / "surface.json"

    generate(tmp_path / "climatology.zarr", pressure, surface)

    with xr.open_dataset(pressure) as generated:
        assert generated.climate_statistics.dims == ("stats", "params")
        assert generated.stats.values.tolist() == ["mean", "std"]
        assert generated.sizes["params"] == 20
    assert surface.is_file()
