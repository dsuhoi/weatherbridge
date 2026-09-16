from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from tools.data.verify_normalization_provenance import verify


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(repo_root: Path) -> Path:
    data_root = repo_root / "data"
    data_root.mkdir()
    pressure = data_root / "json_stats_0p5.nc"
    surface = data_root / "surface_stats_0p5.json"
    pressure_channels = ["T1000"]
    surface_channels = ["t2m"]
    xr.Dataset(
        {
            "climate_statistics": (
                ("stats", "params"),
                np.asarray([[0.0], [1.0]], dtype=np.float32),
            )
        },
        coords={
            "stats": ["mean", "std"],
            "params": pressure_channels,
        },
    ).to_netcdf(pressure)
    surface.write_text(
        json.dumps({"t2m": {"mean": 0.0, "std": 1.0}})
    )
    payload = {
        "schema_version": 1,
        "source": {
            "declared_period": [1979, 2019],
            "generator_retained": False,
        },
        "artifacts": {
            "pressure_level_stats": {
                "path": "data/json_stats_0p5.nc",
                "size_bytes": pressure.stat().st_size,
                "sha256": _sha256(pressure),
            },
            "surface_stats": {
                "path": "data/surface_stats_0p5.json",
                "size_bytes": surface.stat().st_size,
                "sha256": _sha256(surface),
            },
        },
        "expected_pressure_channels": pressure_channels,
        "expected_surface_channels": surface_channels,
        "provenance_status": "synthetic test",
    }
    path = repo_root / "normalization.json"
    path.write_text(json.dumps(payload))
    return path


def test_verify_normalization_identity_and_temporal_separation(
    tmp_path: Path,
) -> None:
    report = verify(
        _manifest(tmp_path),
        tmp_path,
        (2020, 2021),
    )

    assert report["verified"]
    assert report["declared_period"] == [1979, 2019]
    assert not report["generator_retained"]


def test_verify_rejects_evaluation_year_overlap(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="overlaps evaluation years"):
        verify(
            _manifest(tmp_path),
            tmp_path,
            (2019, 2020),
        )


def test_verify_rejects_artifact_drift(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"]["surface_stats"]["sha256"] = "0" * 64
    changed = tmp_path / "changed_normalization.json"
    changed.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="hash mismatch"):
        verify(changed, tmp_path, (2020, 2021))


def test_verify_accepts_hash_bound_symlinked_data_directory(
    tmp_path: Path,
) -> None:
    storage = tmp_path / "storage"
    storage.mkdir()
    source_manifest = _manifest(storage)
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "data").symlink_to(storage / "data", target_is_directory=True)
    manifest = overlay / "normalization.json"
    manifest.write_text(source_manifest.read_text())

    report = verify(manifest, overlay, (2020, 2021))

    assert report["verified"]
    artifact = Path(report["artifacts"]["pressure_level_stats"]["path"])
    assert artifact.resolve().is_relative_to(storage.resolve())


def test_verify_rejects_lexical_path_escape(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["artifacts"]["surface_stats"]["path"] = "../surface_stats.json"
    changed = tmp_path / "escaped-normalization.json"
    changed.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="escapes repository root"):
        verify(changed, tmp_path, (2020, 2021))


def test_verify_checks_retained_generator_hash(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    generator = tmp_path / "generator.py"
    generator.write_text("print('ok')\n")
    payload = json.loads(manifest.read_text())
    payload["source"]["generator_retained"] = True
    payload["source"]["generator"] = {
        "path": "generator.py",
        "sha256": _sha256(generator),
    }
    manifest.write_text(json.dumps(payload))

    assert verify(manifest, tmp_path, (2020, 2021))["generator_retained"]
    generator.write_text("print('changed')\n")
    with pytest.raises(ValueError, match="generator hash mismatch"):
        verify(manifest, tmp_path, (2020, 2021))


def test_verify_binds_independent_reconstruction_report(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    generator = tmp_path / "generator.py"
    generator.write_text("print('audit')\n")
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "not_bit_exact"}))
    payload = json.loads(manifest.read_text())
    payload["source"]["independent_reconstruction"] = {
        "path": "generator.py",
        "generator_sha256": _sha256(generator),
        "report": "report.json",
        "report_sha256": _sha256(report),
        "status": "not_bit_exact",
    }
    manifest.write_text(json.dumps(payload))

    result = verify(manifest, tmp_path, (2020, 2021))
    assert result["independent_reconstruction_status"] == "not_bit_exact"

    report.write_text(json.dumps({"status": "complete"}))
    with pytest.raises(ValueError, match="report hash mismatch"):
        verify(manifest, tmp_path, (2020, 2021))
