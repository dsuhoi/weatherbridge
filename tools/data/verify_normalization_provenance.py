#!/usr/bin/env python3
"""Verify normalization identity and temporal separation from evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import xarray as xr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_path(repo_root: Path, relative: str) -> Path:
    root = Path(os.path.abspath(repo_root))
    path = Path(os.path.abspath(root / relative))
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"artifact escapes repository root: {relative}") from error
    return path


def verify(
    manifest_path: Path,
    repo_root: Path,
    evaluation_years: tuple[int, ...],
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported normalization manifest schema")
    period = manifest.get("source", {}).get("declared_period")
    if (
        not isinstance(period, list)
        or len(period) != 2
        or not all(isinstance(value, int) for value in period)
        or period[0] > period[1]
    ):
        raise ValueError("invalid declared normalization period")
    if not evaluation_years:
        raise ValueError("at least one evaluation year is required")
    overlap = [
        year
        for year in evaluation_years
        if period[0] <= year <= period[1]
    ]
    if overlap:
        raise ValueError(
            f"normalization period overlaps evaluation years: {overlap}"
        )

    generator = manifest.get("source", {}).get("generator")
    generator_retained = bool(
        manifest.get("source", {}).get("generator_retained")
    )
    if generator_retained:
        if not isinstance(generator, dict):
            raise ValueError("retained normalization generator is not described")
        generator_path = _artifact_path(repo_root, str(generator.get("path", "")))
        if _sha256(generator_path) != str(generator.get("sha256", "")):
            raise ValueError("normalization generator hash mismatch")

    reconstruction = manifest.get("source", {}).get(
        "independent_reconstruction"
    )
    reconstruction_status = None
    if reconstruction is not None:
        if not isinstance(reconstruction, dict):
            raise ValueError("independent reconstruction is not described")
        reconstruction_generator = _artifact_path(
            repo_root, str(reconstruction.get("path", ""))
        )
        if _sha256(reconstruction_generator) != str(
            reconstruction.get("generator_sha256", "")
        ):
            raise ValueError("reconstruction generator hash mismatch")
        reconstruction_report = _artifact_path(
            repo_root, str(reconstruction.get("report", ""))
        )
        if _sha256(reconstruction_report) != str(
            reconstruction.get("report_sha256", "")
        ):
            raise ValueError("normalization reconstruction report hash mismatch")
        report = json.loads(reconstruction_report.read_text())
        reconstruction_status = report.get("status")
        if reconstruction_status != reconstruction.get("status"):
            raise ValueError("normalization reconstruction status mismatch")

    verified_artifacts: dict[str, dict[str, Any]] = {}
    for name, expected in manifest.get("artifacts", {}).items():
        path = _artifact_path(repo_root, str(expected["path"]))
        actual = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if actual["size_bytes"] != int(expected["size_bytes"]):
            raise ValueError(f"{name}: normalization artifact size mismatch")
        if actual["sha256"] != str(expected["sha256"]):
            raise ValueError(f"{name}: normalization artifact hash mismatch")
        verified_artifacts[name] = actual
    if set(verified_artifacts) != {"pressure_level_stats", "surface_stats"}:
        raise ValueError("normalization manifest has an incomplete artifact set")

    pressure_path = Path(verified_artifacts["pressure_level_stats"]["path"])
    with xr.open_dataset(pressure_path) as dataset:
        if "climate_statistics" not in dataset:
            raise ValueError("pressure statistics lack climate_statistics")
        values = dataset["climate_statistics"]
        channels = tuple(str(value) for value in values["params"].values)
        stats = tuple(str(value) for value in values["stats"].values)
        expected_channels = tuple(manifest["expected_pressure_channels"])
        if channels != expected_channels:
            raise ValueError("pressure statistics channel order mismatch")
        if stats != ("mean", "std"):
            raise ValueError("pressure statistics must contain mean then std")
        means = values.sel(stats="mean").values
        stds = values.sel(stats="std").values
        if not all(math.isfinite(float(value)) for value in means.flat):
            raise ValueError("pressure means must be finite")
        if not all(
            math.isfinite(float(value)) and float(value) > 0.0
            for value in stds.flat
        ):
            raise ValueError("pressure standard deviations must be positive")

    surface_path = Path(verified_artifacts["surface_stats"]["path"])
    surface = json.loads(surface_path.read_text())
    if tuple(surface) != tuple(manifest["expected_surface_channels"]):
        raise ValueError("surface statistics channel order mismatch")
    for channel, values in surface.items():
        if not all(
            key in values and math.isfinite(float(values[key]))
            for key in ("mean", "std")
        ):
            raise ValueError(f"{channel}: invalid surface statistics")
        if float(values["std"]) <= 0.0:
            raise ValueError(f"{channel}: standard deviation must be positive")

    return {
        "schema_version": 1,
        "verified": True,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "declared_period": period,
        "evaluation_years": list(evaluation_years),
        "artifacts": verified_artifacts,
        "generator_retained": generator_retained,
        "independent_reconstruction_status": reconstruction_status,
        "provenance_status": manifest.get("provenance_status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/normalization_provenance.json"),
    )
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--evaluation-years", default="2020,2021")
    args = parser.parse_args()
    years = tuple(
        int(value)
        for value in args.evaluation_years.split(",")
        if value.strip()
    )
    print(
        json.dumps(
            verify(args.manifest, args.repo_root, years),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
