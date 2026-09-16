from __future__ import annotations

import torch

from tools.baselines.numerical_baselines import (
    _build_lat_lon_grids,
    _sample_field_at_departure,
)


def test_numerical_baseline_uses_canonical_block_average_centres() -> None:
    lat, lon = _build_lat_lon_grids(360, 720, torch.device("cpu"), torch.float32)

    torch.testing.assert_close(lat[[0, -1]], torch.tensor([89.875, -89.625]))
    torch.testing.assert_close(lon[[0, -1]], torch.tensor([0.125, 359.625]))


def test_periodic_sampler_is_identity_at_canonical_cell_centres() -> None:
    height, width = 360, 720
    lat, lon = _build_lat_lon_grids(
        height, width, torch.device("cpu"), torch.float32
    )
    dep_lat = lat.view(1, 1, height, 1).expand(1, 1, height, width)
    dep_lon = lon.view(1, 1, 1, width).expand(1, 1, height, width)
    field = (
        torch.arange(height, dtype=torch.float32).view(1, 1, height, 1)
        + torch.arange(width, dtype=torch.float32).view(1, 1, 1, width) / width
    )

    sampled = _sample_field_at_departure(field, dep_lat, dep_lon)

    torch.testing.assert_close(sampled, field, rtol=0.0, atol=4e-5)


def test_periodic_sampler_interpolates_across_longitude_seam() -> None:
    height, width = 360, 720
    lat, lon = _build_lat_lon_grids(
        height, width, torch.device("cpu"), torch.float32
    )
    field = torch.zeros(1, 1, height, width)
    field[..., -1] = 2.0
    dep_lat = lat.view(1, 1, height, 1).expand(1, 1, height, width)
    dep_lon = lon.view(1, 1, 1, width).expand_as(dep_lat).clone()
    dep_lon[..., 0] = lon[0] - 0.25

    sampled = _sample_field_at_departure(field, dep_lat, dep_lon)

    torch.testing.assert_close(sampled[..., 0], torch.ones_like(sampled[..., 0]))
