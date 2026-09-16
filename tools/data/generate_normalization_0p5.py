#!/usr/bin/env python3
"""Independently reconstruct the 0.5-degree normalization statistics.

Project records associate the frozen training artifacts with a 1990--2019
hourly ERA5 climatology sampled at 00, 06, 12 and 18 UTC. The original
artifact generator was not retained. This reconstruction uses cosine-latitude
weights and the documented 1.3 standard-deviation multiplier, but it is a
provenance audit, not the source of truth for trained checkpoints. Never use
its output to overwrite the hash-bound files under ``data/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


PL_VARIABLES = {
    "t": "T",
    "u": "U",
    "v": "V",
    "q": "Q",
    "z": "Z",
}
SURFACE_VARIABLES = ("t2m", "u10", "v10", "mslp", "sst", "tcc")
STD_MULTIPLIER = 1.3
TCWV_STATS = {"mean": 19.0, "std": 16.59}
TISR_STATS = {"mean": 1071964.4, "std": 1437247.4}


def _spatial_stats(
    values: np.ndarray,
    latitude_weights: np.ndarray,
    n_longitudes: int,
) -> tuple[float, float]:
    filled = np.nan_to_num(values, nan=float(np.nanmean(values)))
    weights = (latitude_weights[:, None] / n_longitudes).reshape(
        (1,) * (filled.ndim - 2) + (-1, 1)
    )
    mean = float((filled * weights).sum(axis=(-2, -1)).mean())
    second_moment = float((filled**2 * weights).sum(axis=(-2, -1)).mean())
    variance = max(second_moment - mean**2, 1.0e-12)
    return mean, float(np.sqrt(variance) * STD_MULTIPLIER)


def _streaming_spatial_stats(
    data: xr.DataArray,
    latitude_weights: np.ndarray,
    n_longitudes: int,
    *,
    day_chunk: int = 46,
) -> tuple[float, float]:
    """Match ``_spatial_stats`` without materialising a global field."""
    if data.dims[-2:] != ("latitude", "longitude"):
        raise ValueError("spatial dimensions must be latitude then longitude")
    if data.dims[:2] != ("hour", "dayofyear"):
        raise ValueError("sample dimensions must be hour then dayofyear")

    weights = np.asarray(
        latitude_weights[:, None] / n_longitudes,
        dtype=np.float64,
    ).reshape(1, 1, -1, 1)
    finite_sum = 0.0
    finite_count = 0
    weighted_finite_sum = 0.0
    weighted_finite_second = 0.0
    missing_weight = 0.0
    sample_count = 0
    for start in range(0, data.sizes["dayofyear"], day_chunk):
        values = np.asarray(
            data.isel(dayofyear=slice(start, start + day_chunk)).values
        )
        finite = np.isfinite(values)
        finite_values = np.where(finite, values, 0.0)
        finite_sum += float(finite_values.sum(dtype=np.float64))
        finite_count += int(finite.sum())
        weighted_finite_sum += float(
            np.einsum(
                "...ij,...ij->",
                finite_values,
                np.broadcast_to(weights, values.shape),
                dtype=np.float64,
                optimize=True,
            )
        )
        np.square(finite_values, out=finite_values)
        weighted_finite_second += float(
            np.einsum(
                "...ij,...ij->",
                finite_values,
                np.broadcast_to(weights, values.shape),
                dtype=np.float64,
                optimize=True,
            )
        )
        finite_spatial_weight = float(
            np.einsum(
                "...ij,...ij->",
                finite,
                np.broadcast_to(weights, values.shape),
                dtype=np.float64,
                optimize=True,
            )
        )
        missing_weight += values.shape[0] * values.shape[1] - finite_spatial_weight
        sample_count += int(values.shape[0] * values.shape[1])
    if finite_count == 0:
        raise ValueError("normalization source contains no finite values")
    fill_value = finite_sum / finite_count
    weighted_sum = weighted_finite_sum + fill_value * missing_weight
    weighted_second = weighted_finite_second + fill_value**2 * missing_weight
    mean = weighted_sum / sample_count
    second_moment = weighted_second / sample_count
    variance = max(second_moment - mean**2, 1.0e-12)
    return mean, float(np.sqrt(variance) * STD_MULTIPLIER)


def _streaming_spatial_stats_by_level(
    data: xr.DataArray,
    latitude_weights: np.ndarray,
    n_longitudes: int,
    *,
    day_chunk: int = 46,
) -> list[tuple[float, float]]:
    """Read each pressure-variable Zarr chunk once for all four levels."""
    if data.dims != (
        "hour",
        "dayofyear",
        "level",
        "latitude",
        "longitude",
    ):
        raise ValueError("unexpected pressure-variable dimension order")
    weights = np.asarray(
        latitude_weights[:, None] / n_longitudes,
        dtype=np.float64,
    ).reshape(1, 1, -1, 1)
    n_levels = data.sizes["level"]
    finite_sum = np.zeros(n_levels, dtype=np.float64)
    finite_count = np.zeros(n_levels, dtype=np.int64)
    weighted_finite_sum = np.zeros(n_levels, dtype=np.float64)
    weighted_finite_second = np.zeros(n_levels, dtype=np.float64)
    missing_weight = np.zeros(n_levels, dtype=np.float64)
    sample_count = 0

    for start in range(0, data.sizes["dayofyear"], day_chunk):
        values = np.asarray(
            data.isel(dayofyear=slice(start, start + day_chunk)).values
        )
        sample_count += int(values.shape[0] * values.shape[1])
        for level_index in range(n_levels):
            level_values = values[:, :, level_index]
            finite = np.isfinite(level_values)
            finite_values = np.where(finite, level_values, 0.0)
            finite_sum[level_index] += finite_values.sum(dtype=np.float64)
            finite_count[level_index] += finite.sum(dtype=np.int64)
            broadcast_weights = np.broadcast_to(weights, level_values.shape)
            weighted_finite_sum[level_index] += np.einsum(
                "...ij,...ij->",
                finite_values,
                broadcast_weights,
                dtype=np.float64,
                optimize=True,
            )
            np.square(finite_values, out=finite_values)
            weighted_finite_second[level_index] += np.einsum(
                "...ij,...ij->",
                finite_values,
                broadcast_weights,
                dtype=np.float64,
                optimize=True,
            )
            finite_spatial_weight = np.einsum(
                "...ij,...ij->",
                finite,
                broadcast_weights,
                dtype=np.float64,
                optimize=True,
            )
            missing_weight[level_index] += (
                level_values.shape[0] * level_values.shape[1]
                - finite_spatial_weight
            )

    if np.any(finite_count == 0):
        raise ValueError("normalization source contains an empty pressure level")
    fill_value = finite_sum / finite_count
    weighted_sum = weighted_finite_sum + fill_value * missing_weight
    weighted_second = weighted_finite_second + fill_value**2 * missing_weight
    mean = weighted_sum / sample_count
    variance = np.maximum(weighted_second / sample_count - mean**2, 1.0e-12)
    std = np.sqrt(variance) * STD_MULTIPLIER
    return [(float(mu), float(sigma)) for mu, sigma in zip(mean, std)]


def generate(
    climatology_path: Path,
    pressure_output: Path,
    surface_output: Path,
) -> None:
    dataset = xr.open_zarr(climatology_path, consolidated=True)
    required = set(PL_VARIABLES) | set(SURFACE_VARIABLES)
    missing = sorted(required - set(dataset.data_vars))
    if missing:
        raise ValueError(f"climatology is missing variables: {missing}")
    if tuple(int(level) for level in dataset.level.values) != (1000, 925, 850, 700):
        raise ValueError("expected pressure levels 1000, 925, 850 and 700 hPa")
    if dataset.sizes.get("hour") != 4 or dataset.sizes.get("dayofyear") != 366:
        raise ValueError("expected a four-cycle, 366-day hourly climatology")

    latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
    latitude_weights = np.cos(np.deg2rad(latitude))
    latitude_weights /= latitude_weights.sum()
    n_longitudes = int(dataset.sizes["longitude"])

    names: list[str] = []
    means: list[float] = []
    standard_deviations: list[float] = []
    levels = tuple(int(level) for level in dataset.level.values)
    for source_name, prefix in PL_VARIABLES.items():
        print(f"pressure variable {source_name}", flush=True)
        level_statistics = _streaming_spatial_stats_by_level(
            dataset[source_name], latitude_weights, n_longitudes
        )
        for level, (mean, std) in zip(levels, level_statistics):
            names.append(f"{prefix}{level}")
            means.append(mean)
            standard_deviations.append(std)

    pressure_output.parent.mkdir(parents=True, exist_ok=True)
    statistics = xr.DataArray(
        np.stack([means, standard_deviations]).astype(np.float32),
        dims=("stats", "params"),
        coords={"stats": ["mean", "std"], "params": names},
    )
    xr.Dataset({"climate_statistics": statistics}).to_netcdf(pressure_output)

    # The checked-in JSON preserves the rounding used by every trained model.
    rounding = {
        "t2m": (1, 1),
        "u10": (2, 2),
        "v10": (2, 2),
        "mslp": (1, 1),
        "sst": (1, 2),
        "tcc": (3, 2),
        "tcwv": (1, 2),
    }
    surface: dict[str, dict[str, float]] = {}
    for name in SURFACE_VARIABLES:
        print(f"surface variable {name}", flush=True)
        mean, std = _streaming_spatial_stats(
            dataset[name], latitude_weights, n_longitudes
        )
        mean_digits, std_digits = rounding[name]
        surface[name] = {
            "mean": round(mean, mean_digits),
            "std": round(std, std_digits),
        }
    # These channels were absent from the retained climatology store. Preserve
    # the constants consumed by the historical checkpoints rather than
    # inventing an unverifiable reconstruction.
    surface["tcwv"] = TCWV_STATS
    surface["tisr"] = TISR_STATS
    surface_output.parent.mkdir(parents=True, exist_ok=True)
    surface_output.write_text(
        json.dumps(surface, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology", type=Path, required=True)
    parser.add_argument(
        "--pressure-output",
        type=Path,
        default=Path("metrics/normalization_replay_v1/generated_stats.nc"),
    )
    parser.add_argument(
        "--surface-output",
        type=Path,
        default=Path("metrics/normalization_replay_v1/generated_surface.json"),
    )
    args = parser.parse_args()
    generate(args.climatology, args.pressure_output, args.surface_output)


if __name__ == "__main__":
    main()
