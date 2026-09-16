#!/usr/bin/env python3
"""Verify and fingerprint the frozen sparse 2022 evaluation memmap."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.data.build_postselection_2022_memmap import (
    CHANNEL_NAMES,
    PL_VARIABLES,
    SURFACE_VARIABLES,
    _sha256,
)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def verify_memmap(
    manifest_path: Path,
    memmap_dir: Path,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    year = int(manifest.get("year", -1))
    if manifest.get("status") != "frozen_before_data_access" or year != 2022:
        raise ValueError("expected the frozen 2022 holdout manifest")
    manifest_sha256 = _sha256(manifest_path)

    metadata_path = memmap_dir / f"wb2_{year}.json"
    data_path = memmap_dir / f"wb2_{year}.bin"
    state_dir = memmap_dir / f"wb2_{year}.state"
    metadata = json.loads(metadata_path.read_text())
    shape = tuple(int(metadata[key]) for key in ("T", "n_channels", "H", "W"))
    expected_shape = (8760, len(CHANNEL_NAMES), 360, 720)
    if shape != expected_shape:
        raise ValueError(f"unexpected memmap shape: {shape}")
    if data_path.stat().st_size != math.prod(shape) * 4:
        raise ValueError("memmap byte size does not match float32 metadata")
    if metadata.get("manifest_sha256") != manifest_sha256:
        raise ValueError("metadata is not bound to the frozen manifest")
    relative_hours = [int(value) for value in metadata["selected_relative_hours"]]
    if len(relative_hours) != 48 * 13 or len(set(relative_hours)) != len(relative_hours):
        raise ValueError("selected holdout hours are incomplete or duplicated")
    if int(metadata.get("selected_window_count", -1)) != 48:
        raise ValueError("selected holdout window count is not 48")

    marker_hashes: dict[str, str] = {}
    for variable in (*PL_VARIABLES, *SURFACE_VARIABLES):
        marker = state_dir / f"{variable}.complete.json"
        payload = json.loads(marker.read_text())
        if (
            payload.get("manifest_sha256") != manifest_sha256
            or int(payload.get("selected_hour_count", -1)) != len(relative_hours)
        ):
            raise ValueError(f"invalid completion marker for {variable}")
        marker_hashes[variable] = _sha256(marker)

    data = np.memmap(data_path, dtype=np.float32, mode="r", shape=shape)
    digest = hashlib.sha256()
    channel_nonzero = np.zeros(24, dtype=bool)
    for relative_hour in relative_hours:
        values = np.asarray(data[relative_hour, :24], dtype="<f4", order="C")
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite holdout value at relative hour {relative_hour}")
        channel_nonzero |= np.any(values != 0.0, axis=(1, 2))
        digest.update(int(relative_hour).to_bytes(4, "little", signed=False))
        digest.update(values.tobytes(order="C"))
        if np.any(np.asarray(data[relative_hour, 24:]) != 0.0):
            raise ValueError("unused sparse channels must remain zero")
    del data
    missing_channels = [
        CHANNEL_NAMES[index]
        for index, nonzero in enumerate(channel_nonzero)
        if not nonzero
    ]
    if missing_channels:
        raise ValueError(f"holdout channels contain only zeros: {missing_channels}")

    return {
        "schema_version": 1,
        "verified": True,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "metadata_path": str(metadata_path.resolve()),
        "metadata_sha256": _sha256(metadata_path),
        "data_path": str(data_path.resolve()),
        "logical_size_bytes": data_path.stat().st_size,
        "data_mtime_ns": data_path.stat().st_mtime_ns,
        "selected_data_sha256": digest.hexdigest(),
        "selected_hour_count": len(relative_hours),
        "selected_window_count": 48,
        "field_count": 24,
        "marker_sha256": marker_hashes,
        "native_grid": {
            "latitude_first_last_degrees": [90.0, -90.0],
            "longitude_first_last_degrees": [0.0, 359.75],
            "shape": [721, 1440],
        },
        "stored_block_average_grid": {
            "latitude_first_last_degrees": [89.875, -89.625],
            "longitude_first_last_degrees": [0.125, 359.625],
            "shape": [360, 720],
            "operation": "unweighted mean of adjacent 2x2 native samples after dropping latitude -90",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--memmap-dir", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    result = verify_memmap(args.manifest, args.memmap_dir)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(args.out_json, result)
    print(json.dumps({"verified": True, "selected_data_sha256": result["selected_data_sha256"]}))


if __name__ == "__main__":
    main()
