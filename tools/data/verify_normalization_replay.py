#!/usr/bin/env python3
"""Audit an independent normalization reconstruction against frozen artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pressure_values(path: Path) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    with xr.open_dataset(path) as dataset:
        values = dataset["climate_statistics"].load()
    return (
        tuple(str(value) for value in values["params"].values),
        tuple(str(value) for value in values["stats"].values),
        np.asarray(values.values, dtype=np.float64),
    )


def verify_replay(
    *,
    climatology_path: Path,
    generated_pressure: Path,
    generated_surface: Path,
    reference_pressure: Path,
    reference_surface: Path,
    generator_path: Path,
    relative_tolerance: float = 5.0e-7,
    require_exact: bool = True,
) -> dict[str, Any]:
    for path in (
        generated_pressure,
        generated_surface,
        reference_pressure,
        reference_surface,
        generator_path,
    ):
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"missing replay input: {path}")
    zmetadata = climatology_path / ".zmetadata"
    if not zmetadata.is_file():
        raise ValueError(f"consolidated Zarr metadata is missing: {zmetadata}")

    generated_channels, generated_stats, generated_values = _pressure_values(
        generated_pressure
    )
    reference_channels, reference_stats, reference_values = _pressure_values(
        reference_pressure
    )
    if generated_channels != reference_channels:
        raise ValueError("replayed pressure-channel order differs from reference")
    if generated_stats != reference_stats:
        raise ValueError("replayed pressure-stat order differs from reference")
    if generated_values.shape != reference_values.shape:
        raise ValueError("replayed pressure-stat shape differs from reference")
    absolute_difference = np.abs(generated_values - reference_values)
    denominator = np.maximum(np.abs(reference_values), np.finfo(np.float64).tiny)
    relative_difference = absolute_difference / denominator
    pressure_exact = bool(
        np.allclose(
            generated_values,
            reference_values,
            rtol=relative_tolerance,
            atol=0.0,
        )
    )
    if require_exact and not pressure_exact:
        index = np.unravel_index(
            int(np.argmax(relative_difference)), relative_difference.shape
        )
        raise ValueError(
            "replayed pressure statistics differ from reference: "
            f"index={index}, generated={generated_values[index]}, "
            f"reference={reference_values[index]}, "
            f"relative_difference={relative_difference[index]}"
        )

    generated_surface_values = json.loads(generated_surface.read_text())
    reference_surface_values = json.loads(reference_surface.read_text())
    differing = sorted(
        key
        for key in set(generated_surface_values) | set(reference_surface_values)
        if generated_surface_values.get(key) != reference_surface_values.get(key)
    )
    surface_exact = not differing
    if require_exact and not surface_exact:
        raise ValueError(
            "replayed surface statistics differ from reference: " + ", ".join(differing)
        )

    status = "complete" if pressure_exact and surface_exact else "not_bit_exact"
    worst_index = np.unravel_index(
        int(np.argmax(relative_difference)), relative_difference.shape
    )

    return {
        "schema_version": 1,
        "status": status,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "climatology_path": str(climatology_path),
        "climatology_zmetadata_sha256": _sha256(zmetadata),
        "generator_path": str(generator_path),
        "generator_sha256": _sha256(generator_path),
        "relative_tolerance": relative_tolerance,
        "pressure": {
            "channels": list(reference_channels),
            "stats": list(reference_stats),
            "max_absolute_difference": float(absolute_difference.max()),
            "max_relative_difference": float(relative_difference.max()),
            "exact_within_tolerance": pressure_exact,
            "worst_stat": reference_stats[worst_index[0]],
            "worst_channel": reference_channels[worst_index[1]],
            "worst_generated_value": float(generated_values[worst_index]),
            "worst_reference_value": float(reference_values[worst_index]),
            "generated_sha256": _sha256(generated_pressure),
            "reference_sha256": _sha256(reference_pressure),
        },
        "surface": {
            "channels": list(reference_surface_values),
            "exact_match": surface_exact,
            "differing_channels": differing,
            "generated_sha256": _sha256(generated_surface),
            "reference_sha256": _sha256(reference_surface),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology", type=Path, required=True)
    parser.add_argument("--generated-pressure", type=Path, required=True)
    parser.add_argument("--generated-surface", type=Path, required=True)
    parser.add_argument(
        "--reference-pressure", type=Path, default=Path("data/json_stats_0p5.nc")
    )
    parser.add_argument(
        "--reference-surface", type=Path, default=Path("data/surface_stats_0p5.json")
    )
    parser.add_argument(
        "--generator",
        type=Path,
        default=Path("tools/data/generate_normalization_0p5.py"),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--record-mismatch",
        action="store_true",
        help="Write a not_bit_exact audit instead of rejecting value drift.",
    )
    args = parser.parse_args()
    report = verify_replay(
        climatology_path=args.climatology,
        generated_pressure=args.generated_pressure,
        generated_surface=args.generated_surface,
        reference_pressure=args.reference_pressure,
        reference_surface=args.reference_surface,
        generator_path=args.generator,
        require_exact=not args.record_mismatch,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.out)
    print(json.dumps({"status": report["status"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
