"""Coordinate definitions for the stored WeatherBench-2 block-average grid."""

from __future__ import annotations

import numpy as np


WB2_BLOCK_GRID_NAME = "wb2_0p25_2x2_block_average_v1"


def wb2_block_average_latitudes(nlat: int = 360) -> np.ndarray:
    """Return row coordinates implied by the retained native WB2 samples.

    The preprocessing averages adjacent rows of the native 90..-90 degree
    721-point grid after dropping the final -90 degree row.
    """
    if nlat != 360:
        raise ValueError("the canonical WB2 block-average grid has 360 rows")
    return 89.875 - 0.5 * np.arange(nlat, dtype=np.float64)


def wb2_block_average_longitudes(nlon: int = 720) -> np.ndarray:
    """Return column coordinates implied by adjacent native longitude means."""
    if nlon != 720:
        raise ValueError("the canonical WB2 block-average grid has 720 columns")
    return 0.125 + 0.5 * np.arange(nlon, dtype=np.float64)


def latitude_strip_weights(latitude_degrees: np.ndarray) -> np.ndarray:
    """Return exact spherical strip areas for descending latitude centres.

    Midpoints define interior row boundaries and the two poles close the
    domain. The returned longitude-independent weights integrate a constant to
    two, matching the latitude factor in a spherical surface integral.
    """
    latitude = np.asarray(latitude_degrees, dtype=np.float64)
    if latitude.ndim != 1 or latitude.size < 2:
        raise ValueError("latitude coordinates must be a one-dimensional vector")
    if not np.all(np.isfinite(latitude)) or not np.all(np.diff(latitude) < 0.0):
        raise ValueError("latitudes must be finite and strictly north-to-south")
    if latitude[0] >= 90.0 or latitude[-1] <= -90.0:
        raise ValueError("latitude centres must lie strictly between the poles")
    boundaries = np.empty(latitude.size + 1, dtype=np.float64)
    boundaries[0] = 90.0
    boundaries[-1] = -90.0
    boundaries[1:-1] = 0.5 * (latitude[:-1] + latitude[1:])
    weights = np.sin(np.deg2rad(boundaries[:-1])) - np.sin(
        np.deg2rad(boundaries[1:])
    )
    if np.any(weights <= 0.0) or not np.isclose(weights.sum(), 2.0, atol=1e-14):
        raise RuntimeError("invalid spherical latitude-strip weights")
    return weights
