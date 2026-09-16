from __future__ import annotations

import numpy as np

from tools.downstream.eval_diurnal_amplitude import build_grid as diurnal_grid
from tools.downstream.eval_physics_consistency import build_grid as physics_grid
from tools.downstream.eval_wind_capacity_factor import build_grid as wind_grid
from weather_time_interp.grid import (
    latitude_strip_weights,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


def test_wb2_block_average_coordinates_follow_native_pair_means() -> None:
    latitude = wb2_block_average_latitudes()
    longitude = wb2_block_average_longitudes()
    assert latitude.shape == (360,)
    assert longitude.shape == (720,)
    np.testing.assert_allclose(latitude[[0, -1]], [89.875, -89.625])
    np.testing.assert_allclose(longitude[[0, -1]], [0.125, 359.625])
    np.testing.assert_allclose(np.diff(latitude), -0.5)
    np.testing.assert_allclose(np.diff(longitude), 0.5)


def test_downstream_tools_use_the_canonical_memmap_grid() -> None:
    expected_latitude = wb2_block_average_latitudes()
    expected_longitude = wb2_block_average_longitudes()

    for build_grid in (diurnal_grid, physics_grid, wind_grid):
        latitude, longitude = build_grid(360, 720)
        np.testing.assert_array_equal(latitude, expected_latitude)
        np.testing.assert_array_equal(longitude, expected_longitude)


def test_strip_weights_cover_the_sphere_and_record_asymmetry() -> None:
    weights = latitude_strip_weights(wb2_block_average_latitudes())
    np.testing.assert_allclose(weights.sum(), 2.0, rtol=0.0, atol=1e-14)
    assert np.all(weights > 0.0)
    assert not np.allclose(weights, weights[::-1])
