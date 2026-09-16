"""Shared calendar and provenance checks for six-hourly climatology."""
from __future__ import annotations

import calendar
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


CANONICAL_CLIMATOLOGY_MANIFEST = ".canonical_climatology_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_climatology_archive(dataset: Any, path: str | Path) -> dict[str, Any]:
    """Fail closed unless ACC climatology is on the canonical source grid."""
    root = Path(path).resolve()
    latitude = np.asarray(dataset.latitude.values, dtype=np.float64)
    longitude = np.asarray(dataset.longitude.values, dtype=np.float64)
    expected_latitude = wb2_block_average_latitudes()
    expected_longitude = wb2_block_average_longitudes()
    if not np.array_equal(latitude, expected_latitude):
        raise ValueError(
            "ACC climatology latitude coordinates are not canonical 2x2 block centres"
        )
    if not np.array_equal(longitude, expected_longitude):
        raise ValueError(
            "ACC climatology longitude coordinates are not canonical 2x2 block centres"
        )
    manifest_path = root / CANONICAL_CLIMATOLOGY_MANIFEST
    if not manifest_path.is_file():
        raise ValueError(f"ACC climatology lacks {CANONICAL_CLIMATOLOGY_MANIFEST}")
    manifest = json.loads(manifest_path.read_text())
    grid = manifest.get("grid", {})
    if (
        manifest.get("schema_version") != 1
        or manifest.get("archive_role") != "acc_climatology"
        or grid.get("name") != WB2_BLOCK_GRID_NAME
        or grid.get("latitude_order") != "north_to_south"
        or grid.get("latitude_sha256") != _sha256_array(latitude)
        or grid.get("longitude_sha256") != _sha256_array(longitude)
        or manifest.get("source", {}).get("climatology_window") != "1990-2019"
        or not manifest.get("sample_equivalence", {}).get("all_channels_checked")
    ):
        raise ValueError("ACC climatology canonical manifest is inconsistent")
    builder_path = Path(__file__).resolve().parents[1] / "data" / (
        "canonicalize_wb2_climatology.py"
    )
    if (
        not builder_path.is_file()
        or manifest.get("builder", {}).get("sha256") != _sha256_file(builder_path)
    ):
        raise ValueError("ACC climatology builder provenance is stale")
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "grid_name": WB2_BLOCK_GRID_NAME,
        "latitude_order": "north_to_south",
        "latitude_sha256": _sha256_array(latitude),
        "longitude_sha256": _sha256_array(longitude),
        "source_uri": manifest["source"]["uri"],
        "source_metadata_sha256": manifest["source"][
            "consolidated_metadata_sha256"
        ],
        "sample_values_checked": int(
            manifest["sample_equivalence"]["values_checked"]
        ),
    }


def climatology_time_weights(
    *,
    day_of_year: int,
    hour: float,
    year: int,
    climatology_days: int,
) -> tuple[int, int, int, int, float]:
    """Return hour/day indices and weight for linear climatology lookup."""
    days_in_year = 366 if calendar.isleap(int(year)) else 365
    if climatology_days < days_in_year:
        raise ValueError(
            f"climatology has {climatology_days} days, "
            f"but year {year} requires {days_in_year}"
        )
    if not 1 <= int(day_of_year) <= days_in_year:
        raise ValueError(
            f"day_of_year={day_of_year} is invalid for year {year}"
        )
    if not 0.0 <= float(hour) < 24.0:
        raise ValueError("hour must be in [0, 24)")

    hour0 = int(float(hour) // 6)
    hour1 = (hour0 + 1) % 4
    weight = (float(hour) - hour0 * 6.0) / 6.0
    day0 = int(day_of_year) - 1
    day1 = day0
    if hour1 == 0:
        day1 = (day0 + 1) % days_in_year
    return hour0, hour1, day0, day1, weight
