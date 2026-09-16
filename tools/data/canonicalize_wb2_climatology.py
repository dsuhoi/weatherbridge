#!/usr/bin/env python3
"""Bind the existing WB2 climatology payload to its true cell-centre grid.

The legacy downloader averaged native rows ``[0, 1], [2, 3], ...`` after
dropping the final south-pole row, but wrote nominal 0.5-degree coordinates.
The data payload is therefore already on the canonical 2x2 block-average grid;
only its coordinate metadata is wrong.  This tool verifies that relationship
against the public WB2 source, makes a hard-link clone, fixes the two coordinate
arrays, and records a fail-closed provenance manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


DEFAULT_SOURCE_URI = (
    "gs://weatherbench2/datasets/era5-hourly-climatology/"
    "1990-2019_6h_1440x721.zarr"
)
MANIFEST_NAME = ".canonical_climatology_manifest.json"
PL_VARIABLES = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}
SURFACE_VARIABLES = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "mslp": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
    "tcc": "total_cloud_cover",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def source_metadata(source_uri: str) -> tuple[str, int]:
    import gcsfs

    if not source_uri.startswith("gs://"):
        raise ValueError("the canonical source must be a public gs:// Zarr")
    fs = gcsfs.GCSFileSystem(token="anon")
    remote = source_uri.removeprefix("gs://").rstrip("/") + "/.zmetadata"
    payload = fs.cat(remote)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _legacy_coordinates(dataset: Any) -> None:
    expected_latitude = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    expected_longitude = np.linspace(0.0, 359.5, 720, dtype=np.float32)
    if not np.array_equal(dataset.latitude.values, expected_latitude):
        raise ValueError("input does not have the expected legacy latitude labels")
    if not np.array_equal(dataset.longitude.values, expected_longitude):
        raise ValueError("input does not have the expected legacy longitude labels")


def _validate_shapes(dataset: Any) -> None:
    expected_dims = {
        "hour": 4,
        "dayofyear": 366,
        "level": 4,
        "latitude": 360,
        "longitude": 720,
    }
    for name, size in expected_dims.items():
        if int(dataset.sizes.get(name, -1)) != size:
            raise ValueError(f"unexpected climatology dimension {name}")
    if list(map(int, dataset.level.values)) != [1000, 925, 850, 700]:
        raise ValueError("unexpected pressure-level order")
    expected_vars = set(PL_VARIABLES) | set(SURFACE_VARIABLES)
    if set(dataset.data_vars) != expected_vars:
        raise ValueError(
            f"unexpected climatology variables: {sorted(dataset.data_vars)}"
        )


def _source_grid(source: Any) -> None:
    expected_latitude = np.linspace(90.0, -90.0, 721, dtype=np.float32)
    expected_longitude = np.linspace(0.0, 359.75, 1440, dtype=np.float32)
    if not np.allclose(source.latitude.values, expected_latitude, atol=0.0):
        raise ValueError("WB2 source latitude grid changed")
    if not np.allclose(source.longitude.values, expected_longitude, atol=0.0):
        raise ValueError("WB2 source longitude grid changed")


def _sample_equivalence(legacy: Any, source: Any) -> dict[str, Any]:
    """Compare every stored channel at representative block locations."""
    samples = ((0, 0), (179, 359), (359, 719))
    max_abs_error = 0.0
    checked = 0
    per_variable: dict[str, float] = {}
    source_levels = list(map(int, source.level.values))

    for short, long_name in {**PL_VARIABLES, **SURFACE_VARIABLES}.items():
        variable_max = 0.0
        levels: list[int | None] = [None]
        if short in PL_VARIABLES:
            levels = [1000, 925, 850, 700]
        for level in levels:
            old_field = legacy[short].isel(hour=0, dayofyear=0)
            new_field = source[long_name].isel(hour=0, dayofyear=0)
            if level is not None:
                old_field = old_field.sel(level=level)
                new_field = new_field.isel(level=source_levels.index(level))
            for row, column in samples:
                old_value = float(old_field.isel(latitude=row, longitude=column))
                block = np.asarray(
                    new_field.isel(
                        latitude=slice(2 * row, 2 * row + 2),
                        longitude=slice(2 * column, 2 * column + 2),
                    ).values,
                    dtype=np.float64,
                )
                expected = float(np.nanmean(block))
                if np.isnan(old_value) and np.isnan(expected):
                    error = 0.0
                else:
                    if not np.isclose(old_value, expected, rtol=2e-6, atol=1e-7):
                        raise ValueError(
                            f"payload mismatch for {short} level={level} "
                            f"row={row} column={column}: {old_value} vs {expected}"
                        )
                    error = abs(old_value - expected)
                variable_max = max(variable_max, error)
                max_abs_error = max(max_abs_error, error)
                checked += 1
        per_variable[short] = variable_max
    return {
        "hour_index": 0,
        "dayofyear_index": 0,
        "output_cells": [list(sample) for sample in samples],
        "values_checked": checked,
        "all_channels_checked": True,
        "rtol": 2e-6,
        "atol": 1e-7,
        "max_abs_error": max_abs_error,
        "max_abs_error_by_variable": per_variable,
    }


def _hardlink_clone(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=os.link)
    for name in ("latitude", "longitude"):
        shutil.rmtree(destination / name)
        shutil.copytree(source / name, destination / name)
    for name in (".zattrs", ".zmetadata"):
        target = destination / name
        target.unlink()
        shutil.copy2(source / name, target)


def _payload_inventory(root: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    size = 0
    for variable in sorted(set(PL_VARIABLES) | set(SURFACE_VARIABLES)):
        for path in sorted((root / variable).rglob("*")):
            if not path.is_file() or path.name.startswith(".z"):
                continue
            relative = path.relative_to(root).as_posix()
            stat = path.stat()
            digest.update(f"{relative}\0{stat.st_size}\n".encode())
            count += 1
            size += stat.st_size
    return {
        "data_chunk_count": count,
        "data_chunk_total_size_bytes": size,
        "data_chunk_layout_sha256": digest.hexdigest(),
        "storage": "hard_link_clone_of_validated_legacy_payload",
    }


def canonicalize(legacy_path: Path, output_path: Path, source_uri: str) -> dict[str, Any]:
    import xarray as xr
    import zarr

    legacy_path = legacy_path.resolve()
    if not legacy_path.is_dir():
        raise FileNotFoundError(legacy_path)
    legacy = xr.open_zarr(str(legacy_path), consolidated=True)
    source = xr.open_zarr(
        source_uri,
        storage_options={"token": "anon"},
        consolidated=True,
    )
    try:
        _validate_shapes(legacy)
        _legacy_coordinates(legacy)
        _source_grid(source)
        if legacy.attrs.get("source") != source_uri:
            raise ValueError("legacy source URI does not match requested WB2 source")
        if legacy.attrs.get("climatology_window") != "1990-2019":
            raise ValueError("unexpected climatology window")
        sample_check = _sample_equivalence(legacy, source)
    finally:
        legacy.close()
        source.close()

    source_metadata_sha256, source_metadata_size = source_metadata(source_uri)
    legacy_metadata_sha256 = sha256_file(legacy_path / ".zmetadata")
    _hardlink_clone(legacy_path, output_path)
    latitude = wb2_block_average_latitudes()
    longitude = wb2_block_average_longitudes()
    group = zarr.open_group(str(output_path), mode="a")
    group["latitude"][:] = latitude
    group["longitude"][:] = longitude
    group.attrs.update(
        {
            "grid_name": WB2_BLOCK_GRID_NAME,
            "latitude_order": "north_to_south",
            "coordinate_transform": (
                "drop_native_south_pole_then_unweighted_2x2_block_average"
            ),
            "legacy_coordinate_labels_corrected": True,
        }
    )
    zarr.consolidate_metadata(str(output_path))

    builder_path = Path(__file__).resolve()
    manifest = {
        "schema_version": 1,
        "archive_role": "acc_climatology",
        "source": {
            "uri": source_uri,
            "consolidated_metadata_sha256": source_metadata_sha256,
            "consolidated_metadata_size_bytes": source_metadata_size,
            "climatology_window": "1990-2019",
            "native_grid": "0.25_degree_721x1440_north_to_south",
        },
        "legacy_store": {
            "path": str(legacy_path),
            "consolidated_metadata_sha256": legacy_metadata_sha256,
            "declared_coordinate_error": (
                "payload used 2x2 block centres but labels used nominal 0.5-degree nodes"
            ),
        },
        "grid": {
            "name": WB2_BLOCK_GRID_NAME,
            "latitude_order": "north_to_south",
            "latitude_sha256": sha256_array(latitude),
            "longitude_sha256": sha256_array(longitude),
            "transform": (
                "drop_native_south_pole_then_unweighted_2x2_block_average"
            ),
        },
        "sample_equivalence": sample_check,
        "payload": _payload_inventory(output_path),
        "builder": {
            "path": str(builder_path),
            "sha256": sha256_file(builder_path),
        },
    }
    atomic_json(output_path / MANIFEST_NAME, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE_URI)
    args = parser.parse_args()
    manifest = canonicalize(args.legacy, args.output, args.source)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
