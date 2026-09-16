import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools.eval import climatology
from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)


def _array(values):
    return SimpleNamespace(values=values)


def _dataset(latitude=None, longitude=None):
    return SimpleNamespace(
        latitude=_array(
            wb2_block_average_latitudes() if latitude is None else latitude
        ),
        longitude=_array(
            wb2_block_average_longitudes() if longitude is None else longitude
        ),
    )


def _sha(values):
    payload = np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()
    return hashlib.sha256(payload).hexdigest()


def _write_manifest(root: Path, builder_sha: str) -> None:
    latitude = wb2_block_average_latitudes()
    longitude = wb2_block_average_longitudes()
    payload = {
        "schema_version": 1,
        "archive_role": "acc_climatology",
        "source": {
            "uri": "gs://weatherbench2/example.zarr",
            "consolidated_metadata_sha256": "a" * 64,
            "climatology_window": "1990-2019",
        },
        "grid": {
            "name": WB2_BLOCK_GRID_NAME,
            "latitude_order": "north_to_south",
            "latitude_sha256": _sha(latitude),
            "longitude_sha256": _sha(longitude),
        },
        "sample_equivalence": {
            "all_channels_checked": True,
            "values_checked": 78,
        },
        "builder": {"sha256": builder_sha},
    }
    (root / climatology.CANONICAL_CLIMATOLOGY_MANIFEST).write_text(
        json.dumps(payload)
    )


def test_validate_climatology_archive_accepts_bound_grid(tmp_path):
    builder = (
        Path(climatology.__file__).resolve().parents[1]
        / "data"
        / "canonicalize_wb2_climatology.py"
    )
    expected_builder = hashlib.sha256(builder.read_bytes()).hexdigest()
    _write_manifest(tmp_path, expected_builder)
    result = climatology.validate_climatology_archive(_dataset(), tmp_path)
    assert result["grid_name"] == WB2_BLOCK_GRID_NAME
    assert result["sample_values_checked"] == 78


def test_validate_climatology_archive_rejects_legacy_coordinates(tmp_path):
    latitude = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    with pytest.raises(ValueError, match="latitude coordinates"):
        climatology.validate_climatology_archive(
            _dataset(latitude=latitude),
            tmp_path,
        )
