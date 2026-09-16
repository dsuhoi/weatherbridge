#!/usr/bin/env python3
"""Export a provenance-bound cyclone window for the graphical overview."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

METHOD_SLUGS = {
    "Linear": "linear",
    "WeatherDCAE-14M": "weatherdcae_14m",
    "WeatherBridge": "weatherbridge",
}
REQUIRED_CHANNELS = ("mslp", "u10", "v10")


def wb2_block_average_latitudes() -> np.ndarray:
    return 89.875 - 0.5 * np.arange(360, dtype=np.float64)


def wb2_block_average_longitudes() -> np.ndarray:
    return 0.125 + 0.5 * np.arange(720, dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(fields_dir: Path, event_id: str, method: str) -> Path:
    return fields_dir / f"{event_id}__{METHOD_SLUGS[method]}__tau3.npz"


def load_artifact(path: Path, event_id: str, method: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as artifact:
        if artifact["event_id"].item() != event_id:
            raise ValueError(f"{path}: event mismatch")
        if artifact["model_name"].item() != method:
            raise ValueError(f"{path}: method mismatch")
        if int(artifact["tau"].item()) != 3:
            raise ValueError(f"{path}: expected tau=3")
        channels = artifact["channels"].tolist()
        if tuple(channels) != REQUIRED_CHANNELS:
            raise ValueError(f"{path}: unexpected channels {channels}")
        return {
            "event_label": artifact["event_label"].item(),
            "init_time": artifact["init_time"].item(),
            "bbox": np.asarray(artifact["bbox"], dtype=np.float32),
            "lat": np.asarray(artifact["lat"], dtype=np.float64),
            "lon": np.asarray(artifact["lon"], dtype=np.float64),
            "prediction": np.asarray(artifact["prediction"], dtype=np.float32),
            "target": np.asarray(artifact["target"], dtype=np.float32),
            "provenance": json.loads(artifact["provenance_json"].item()),
        }


def align_coordinates(
    full: np.ndarray, crop: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray, float]:
    """Align legacy edge-labelled event coordinates to WB2 cell centres."""
    for offset in (0.0, 0.125, -0.125):
        corrected = crop + offset
        indices = np.asarray(
            [int(np.argmin(np.abs(full - value))) for value in corrected],
            dtype=np.int64,
        )
        if len(np.unique(indices)) != len(indices):
            continue
        if np.allclose(full[indices], corrected, rtol=0.0, atol=1.0e-6):
            return indices, corrected, offset
    raise ValueError(f"{name}: artifact does not align with the WB2 grid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--fields-dir", type=Path, required=True)
    parser.add_argument("--memmap", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifacts = {
        method: load_artifact(
            artifact_path(args.fields_dir, args.event_id, method),
            args.event_id,
            method,
        )
        for method in METHOD_SLUGS
    }
    reference = artifacts["Linear"]
    for method, artifact in artifacts.items():
        for key in ("event_label", "init_time", "bbox", "lat", "lon", "target"):
            if not np.array_equal(artifact[key], reference[key]):
                raise ValueError(f"{method}: inconsistent {key}")

    metadata = json.loads(args.metadata.read_text())
    init = datetime.fromisoformat(reference["init_time"]).replace(
        tzinfo=timezone.utc
    )
    if metadata.get("year") != init.year or metadata.get("dtype") != "float32":
        raise ValueError("memmap metadata does not match the event")
    shape = tuple(int(value) for value in metadata["shape"])
    expected_hours = 8784 if init.year % 4 == 0 else 8760
    if shape != (expected_hours, 27, 360, 720):
        raise ValueError(f"unexpected memmap shape {shape}")
    channel_names = list(metadata["channel_names"])
    channel_indices = [channel_names.index(name) for name in REQUIRED_CHANNELS]

    artifact_lat = np.asarray(reference["lat"])
    artifact_lon = np.asarray(reference["lon"])
    lat_indices, lat, lat_offset = align_coordinates(
        wb2_block_average_latitudes(), artifact_lat, "latitude"
    )
    lon_indices, lon, lon_offset = align_coordinates(
        wb2_block_average_longitudes(), artifact_lon, "longitude"
    )
    year_start = datetime(init.year, 1, 1, tzinfo=timezone.utc)
    start_index = int((init - year_start).total_seconds() // 3600)
    if start_index < 0 or start_index + 6 >= shape[0]:
        raise ValueError("event window falls outside the memmap")

    raw = np.memmap(args.memmap, dtype=np.float32, mode="r", shape=shape)
    era5 = np.empty((7, 3, lat.size, lon.size), dtype=np.float32)
    for hour in range(7):
        for output_index, channel_index in enumerate(channel_indices):
            values = raw[start_index + hour, channel_index]
            era5[hour, output_index] = values[np.ix_(lat_indices, lon_indices)]
    era5[:, 0] *= 0.01

    target = np.asarray(reference["target"], dtype=np.float32).copy()
    target[0] *= 0.01
    target_delta = np.max(np.abs(era5[3] - target), axis=(1, 2))
    if target_delta[0] > 2.0e-3 or np.any(target_delta[1:] > 2.0e-5):
        raise ValueError(f"artifact target mismatch: {target_delta.tolist()}")

    predictions = np.stack(
        [artifacts[method]["prediction"] for method in METHOD_SLUGS]
    ).astype(np.float32)
    predictions[:, 0] *= 0.01
    provenance = {
        "schema_version": 1,
        "event_id": args.event_id,
        "event_label": reference["event_label"],
        "init_time": reference["init_time"],
        "query_hour": 3,
        "methods": list(METHOD_SLUGS),
        "channels": list(REQUIRED_CHANNELS),
        "coordinate_correction_degrees": {
            "latitude": lat_offset,
            "longitude": lon_offset,
        },
        "target_max_abs_difference": target_delta.tolist(),
        "source_sha256": {
            str(args.metadata): sha256_file(args.metadata),
            **{
                str(artifact_path(args.fields_dir, args.event_id, method)): (
                    sha256_file(artifact_path(args.fields_dir, args.event_id, method))
                )
                for method in METHOD_SLUGS
            },
        },
        "method_provenance": {
            method: artifact["provenance"]
            for method, artifact in artifacts.items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_id=np.asarray(args.event_id),
        event_label=np.asarray(reference["event_label"]),
        init_time=np.asarray(reference["init_time"]),
        latitude=lat.astype(np.float32),
        longitude=lon.astype(np.float32),
        hours=np.arange(7, dtype=np.int8),
        methods=np.asarray(list(METHOD_SLUGS)),
        channels=np.asarray(REQUIRED_CHANNELS),
        era5=era5,
        midpoint_predictions=predictions,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "size_bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "event": reference["event_label"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
