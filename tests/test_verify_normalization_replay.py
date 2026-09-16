from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tools.data.verify_normalization_replay import verify_replay


def _pressure(path: Path, value: float) -> None:
    xr.Dataset(
        {
            "climate_statistics": (
                ("stats", "params"),
                np.asarray([[value], [2.0]], dtype=np.float32),
            )
        },
        coords={"stats": ["mean", "std"], "params": ["T1000"]},
    ).to_netcdf(path)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    climatology = tmp_path / "climatology.zarr"
    climatology.mkdir()
    (climatology / ".zmetadata").write_text("{}\n")
    generated_pressure = tmp_path / "generated.nc"
    reference_pressure = tmp_path / "reference.nc"
    _pressure(generated_pressure, 1.0)
    _pressure(reference_pressure, 1.0)
    surface = {"t2m": {"mean": 1.0, "std": 2.0}}
    generated_surface = tmp_path / "generated.json"
    reference_surface = tmp_path / "reference.json"
    generated_surface.write_text(json.dumps(surface))
    reference_surface.write_text(json.dumps(surface))
    generator = tmp_path / "generator.py"
    generator.write_text("print('generator')\n")
    return {
        "climatology_path": climatology,
        "generated_pressure": generated_pressure,
        "generated_surface": generated_surface,
        "reference_pressure": reference_pressure,
        "reference_surface": reference_surface,
        "generator_path": generator,
    }


def test_verify_replay_binds_source_and_matches_values(tmp_path: Path) -> None:
    report = verify_replay(**_inputs(tmp_path))

    assert report["status"] == "complete"
    assert report["pressure"]["max_absolute_difference"] == 0.0
    assert report["surface"]["exact_match"]
    assert len(report["climatology_zmetadata_sha256"]) == 64


def test_verify_replay_rejects_pressure_drift(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _pressure(inputs["generated_pressure"], 1.1)

    with pytest.raises(ValueError, match="pressure statistics differ"):
        verify_replay(**inputs)


def test_verify_replay_records_non_exact_reconstruction(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    _pressure(inputs["generated_pressure"], 1.1)
    generated_surface = json.loads(inputs["generated_surface"].read_text())
    generated_surface["t2m"]["std"] = 2.1
    inputs["generated_surface"].write_text(json.dumps(generated_surface))

    report = verify_replay(**inputs, require_exact=False)

    assert report["status"] == "not_bit_exact"
    assert not report["pressure"]["exact_within_tolerance"]
    assert report["pressure"]["worst_channel"] == "T1000"
    assert report["surface"]["differing_channels"] == ["t2m"]
