#!/usr/bin/env python3
"""Build the frozen sparse 2022 WeatherBench-2 evaluation memmap.

The output has the standard full-year logical shape expected by
``ERA5MemmapDataset``, but only the predeclared hours occupy disk blocks.
Evaluation must use the matching economy filter and one 00 UTC anchor per
selected day. Values are fetched only after the frozen manifest is validated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = (
    "weatherbench2/datasets/era5/"
    "1959-2023_01_10-full_37-1h-0p25deg-chunk-1.zarr"
)
BASE_EPOCH = np.datetime64("1959-01-01T00:00:00")
LEVELS = np.array(
    [
        1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175,
        200, 225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700,
        750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000,
    ],
    dtype=np.int32,
)
TARGET_LEVELS = (1000, 925, 850, 700)
PL_VARIABLES = (
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "geopotential",
)
SURFACE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
)
CHANNEL_NAMES = tuple(
    f"{name}{level}"
    for name in ("T", "U", "V", "Q", "Z")
    for level in TARGET_LEVELS
) + ("t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text())
    if manifest.get("status") not in (
        "frozen_before_data_access",
        "frozen_before_nwp_model_evaluation",
    ):
        raise ValueError("holdout manifest is not frozen")
    if int(manifest.get("year", -1)) != 2022:
        raise ValueError("only the frozen 2022 holdout is supported")
    if manifest.get("source") != f"gs://{ROOT}":
        raise ValueError("holdout source does not match the downloader")
    if manifest.get("fields") != list(CHANNEL_NAMES[:24]):
        raise ValueError("holdout field order does not match the model")
    if manifest.get("status") == "frozen_before_nwp_model_evaluation":
        parent = Path(str(manifest.get("parent_manifest", "")))
        if not parent.is_absolute():
            parent = Path.cwd() / parent
        if (
            not parent.is_file()
            or _sha256(parent) != manifest.get("parent_manifest_sha256")
        ):
            raise ValueError("NWP holdout parent manifest identity mismatch")
    return manifest


def selected_relative_hours(
    year: int,
    days: Iterable[int],
    anchor_hours: Iterable[int],
    max_horizon: int,
) -> list[int]:
    year_start = np.datetime64(f"{year}-01-01T00:00:00")
    selected: set[int] = set()
    for month in range(1, 13):
        for day in days:
            for anchor_hour in anchor_hours:
                anchor = np.datetime64(
                    f"{year}-{month:02d}-{int(day):02d}T{int(anchor_hour):02d}:00:00"
                )
                t0 = int((anchor - year_start) / np.timedelta64(1, "h"))
                selected.update(range(t0, t0 + int(max_horizon) + 1))
    return sorted(selected)


def selected_relative_hours_from_inits(
    year: int,
    init_times: Iterable[str],
    max_horizon: int,
) -> list[int]:
    """Expand explicit forecast initialisations over one fixed lead range."""
    year_start = np.datetime64(f"{year}-01-01T00:00:00")
    selected: set[int] = set()
    previous: np.datetime64 | None = None
    for value in init_times:
        instant = np.datetime64(str(value), "h")
        if int(str(instant)[:4]) != year or (
            previous is not None and instant <= previous
        ):
            raise ValueError("init_times must be unique, sorted, and in year")
        previous = instant
        t0 = int((instant - year_start) / np.timedelta64(1, "h"))
        selected.update(range(t0, t0 + int(max_horizon) + 1))
    if previous is None:
        raise ValueError("init_times must not be empty")
    return sorted(selected)


def _coarsen(values: np.ndarray) -> np.ndarray:
    values = values[..., :720, :1440]
    shape = values.shape[:-2] + (360, 2, 720, 2)
    return np.nanmean(values.reshape(shape), axis=(-3, -1)).astype(np.float32)


def _absolute_hour(year: int, relative_hour: int) -> int:
    instant = np.datetime64(f"{year}-01-01T00:00:00") + np.timedelta64(
        int(relative_hour), "h"
    )
    return int((instant - BASE_EPOCH) / np.timedelta64(1, "h"))


def _download_pressure_variable(
    array: zarr.Array,
    relative_hours: list[int],
    year: int,
    level_indices: list[int],
    workers: int,
):
    def fetch(relative_hour: int) -> tuple[int, np.ndarray]:
        absolute = _absolute_hour(year, relative_hour)
        selected = np.asarray(
            array.oindex[absolute, level_indices, :, :]
        )
        return relative_hour, _coarsen(selected)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, hour) for hour in relative_hours]
        for future in as_completed(futures):
            yield future.result()


def _download_surface_variable(
    array: zarr.Array,
    relative_hours: list[int],
    year: int,
    workers: int,
):
    def fetch(relative_hour: int) -> tuple[int, np.ndarray]:
        absolute = _absolute_hour(year, relative_hour)
        return relative_hour, _coarsen(np.asarray(array[absolute]))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, hour) for hour in relative_hours]
        for future in as_completed(futures):
            yield future.result()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _marker_matches(marker: Path, manifest_sha256: str) -> bool:
    if not marker.is_file():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("manifest_sha256") == manifest_sha256
        and int(payload.get("selected_hour_count", -1)) > 0
    )


def main() -> None:
    import gcsfs
    import zarr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("repro/postselection_holdout_2022.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/wb2_0p5_postselection"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    manifest = _load_manifest(args.manifest)
    year = int(manifest["year"])
    if "init_times" in manifest:
        init_times = [str(value) for value in manifest["init_times"]]
        max_horizon = int(manifest["maximum_forecast_lead_hours"])
        relative_hours = selected_relative_hours_from_inits(
            year,
            init_times,
            max_horizon,
        )
        expected_windows = len(init_times)
    else:
        days = [int(value) for value in manifest["day_of_month"]]
        anchor_hours = [int(value) for value in manifest["anchor_hours_utc"]]
        max_horizon = int(manifest["maximum_anchor_spacing_hours"])
        relative_hours = selected_relative_hours(
            year, days, anchor_hours, max_horizon
        )
        expected_windows = 12 * len(days) * len(anchor_hours)
    if len(relative_hours) != expected_windows * (max_horizon + 1):
        raise ValueError("selected holdout windows unexpectedly overlap")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    state_dir = args.out_dir / f"wb2_{year}.state"
    state_dir.mkdir(exist_ok=True)
    data_path = args.out_dir / f"wb2_{year}.bin"
    metadata_path = args.out_dir / f"wb2_{year}.json"
    hours_in_year = int(
        (
            np.datetime64(f"{year + 1}-01-01T00:00:00")
            - np.datetime64(f"{year}-01-01T00:00:00")
        )
        / np.timedelta64(1, "h")
    )
    if min(relative_hours) < 0 or max(relative_hours) >= hours_in_year:
        raise ValueError("selected holdout hours cross the declared year")
    channel_names = tuple(str(value) for value in manifest["fields"])
    if channel_names != CHANNEL_NAMES[: len(channel_names)]:
        raise ValueError("manifest channels must follow the canonical prefix")
    shape = (hours_in_year, len(channel_names), 360, 720)
    expected_size = int(np.prod(shape, dtype=np.int64)) * 4
    if data_path.exists() and data_path.stat().st_size != expected_size:
        raise ValueError(f"wrong existing memmap size: {data_path}")
    if not data_path.exists():
        with data_path.open("wb") as stream:
            stream.truncate(expected_size)

    memmap = np.memmap(data_path, dtype=np.float32, mode="r+", shape=shape)
    fs = gcsfs.GCSFileSystem(token="anon")
    level_indices = [int(np.where(LEVELS == level)[0][0]) for level in TARGET_LEVELS]
    manifest_sha256 = _sha256(args.manifest)

    for variable_index, variable in enumerate(PL_VARIABLES):
        marker = state_dir / f"{variable}.complete.json"
        if _marker_matches(marker, manifest_sha256):
            continue
        source = zarr.open(fs.get_mapper(f"{ROOT}/{variable}"), mode="r")
        channel_slice = slice(variable_index * 4, variable_index * 4 + 4)
        for relative_hour, values in _download_pressure_variable(
            source, relative_hours, year, level_indices, args.workers
        ):
            memmap[relative_hour, channel_slice] = values
        memmap.flush()
        _write_json_atomic(
            marker,
            {
                "variable": variable,
                "selected_hour_count": len(relative_hours),
                "manifest_sha256": manifest_sha256,
            },
        )

    for surface_index, variable in enumerate(SURFACE_VARIABLES, start=20):
        marker = state_dir / f"{variable}.complete.json"
        if _marker_matches(marker, manifest_sha256):
            continue
        source = zarr.open(fs.get_mapper(f"{ROOT}/{variable}"), mode="r")
        for relative_hour, values in _download_surface_variable(
            source, relative_hours, year, args.workers
        ):
            memmap[relative_hour, surface_index] = values
        memmap.flush()
        _write_json_atomic(
            marker,
            {
                "variable": variable,
                "selected_hour_count": len(relative_hours),
                "manifest_sha256": manifest_sha256,
            },
        )

    del memmap
    _write_json_atomic(
        metadata_path,
        {
            "year": year,
            "T": hours_in_year,
            "n_channels": len(channel_names),
            "H": 360,
            "W": 720,
            "dtype": "float32",
            "channel_names": list(channel_names),
            "shape": list(shape),
            "size_bytes": expected_size,
            "sparse_file": True,
            "selected_relative_hours": relative_hours,
            "selected_window_count": expected_windows,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": manifest_sha256,
            "builder_sha256": _sha256(Path(__file__)),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "path": str(data_path),
                "logical_size_bytes": expected_size,
                "selected_hours": len(relative_hours),
                "selected_windows": expected_windows,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
