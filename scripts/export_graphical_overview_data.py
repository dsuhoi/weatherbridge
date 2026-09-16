#!/usr/bin/env python3
"""Export provenance-bound weather fields for the graphical paper overview."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from weather_time_interp.grid import (
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)

EXPECTED_INIT_TIME = "2020-09-07T00:00:00"
EXPECTED_TAUS = np.arange(1, 6, dtype=np.int32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_case(path: Path, field: str) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        if data["methods"].tolist() != ["WeatherBridge"]:
            raise ValueError(f"{path}: unexpected model label")
        if data["init_time"].item() != EXPECTED_INIT_TIME:
            raise ValueError(f"{path}: unexpected initial time")
        if not np.array_equal(data["taus"], EXPECTED_TAUS):
            raise ValueError(f"{path}: unexpected query-hour grid")
        provenance = json.loads(data["provenance_json"].item())
        if provenance.get("model_label") != "WeatherBridge":
            raise ValueError(f"{path}: invalid model provenance")
        return {
            "field": field,
            "lat": np.asarray(data["lat"], dtype=np.float64),
            "lon": np.asarray(data["lon"], dtype=np.float64),
            "truth": np.asarray(data["truth"], dtype=np.float32),
            "prediction": np.asarray(data["panels"][0], dtype=np.float32),
            "provenance": provenance,
        }


def crop_indices(
    full: np.ndarray,
    expected: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    low = float(np.min(expected)) - 1e-6
    high = float(np.max(expected)) + 1e-6
    indices = np.flatnonzero((full >= low) & (full <= high))
    if not np.allclose(full[indices], expected, rtol=0.0, atol=1e-6):
        raise ValueError(f"{name}: case grid does not match WB2 coordinates")
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memmap",
        type=Path,
        default=Path("/tmp/wb2_0p5_cache/wb2_2020.bin"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/tmp/wb2_0p5_cache/wb2_2020.json"),
    )
    parser.add_argument(
        "--base-mslp",
        type=Path,
        default=Path("demo/precomputed/typhoon_haishen/East_Asia__mslp.npz"),
    )
    parser.add_argument(
        "--weatherbridge-mslp",
        type=Path,
        default=Path("metrics/case_studies_weatherbridge/typhoon_haishen/mslp.npz"),
    )
    parser.add_argument(
        "--weatherbridge-q1000",
        type=Path,
        default=Path("metrics/case_studies_weatherbridge/typhoon_haishen/q1000.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "paper/graphical_overview/data/haishen_overview.npz"
        ),
    )
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text())
    if metadata.get("year") != 2020 or metadata.get("dtype") != "float32":
        raise ValueError("unexpected memmap metadata")
    shape = tuple(int(value) for value in metadata["shape"])
    if shape != (8784, 27, 360, 720):
        raise ValueError(f"unexpected memmap shape {shape}")
    channels = list(metadata["channel_names"])
    mslp_index = channels.index("mslp")
    q1000_index = channels.index("Q1000")

    mslp = load_case(args.weatherbridge_mslp, "mslp")
    q1000 = load_case(args.weatherbridge_q1000, "Q1000")
    for coordinate in ("lat", "lon"):
        if not np.array_equal(mslp[coordinate], q1000[coordinate]):
            raise ValueError(f"MSLP and Q1000 disagree on {coordinate}")

    latitude = np.asarray(mslp["lat"])
    longitude = np.asarray(mslp["lon"])
    full_latitude = wb2_block_average_latitudes()
    full_longitude = wb2_block_average_longitudes()
    lat_indices = crop_indices(full_latitude, latitude, name="latitude")
    lon_indices = crop_indices(full_longitude, longitude, name="longitude")

    init = datetime.fromisoformat(EXPECTED_INIT_TIME).replace(
        tzinfo=timezone.utc
    )
    year_start = datetime(init.year, 1, 1, tzinfo=timezone.utc)
    init_index = int((init - year_start).total_seconds() // 3600)
    if init_index < 0 or init_index + 6 >= shape[0]:
        raise ValueError("case time lies outside the memmap")

    data = np.memmap(args.memmap, dtype=np.float32, mode="r", shape=shape)

    def read_field(hour: int, channel: int, scale: float) -> np.ndarray:
        values = data[init_index + hour, channel]
        return np.asarray(
            values[np.ix_(lat_indices, lon_indices)] * scale,
            dtype=np.float32,
        )

    mslp_trajectory = np.stack(
        [read_field(hour, mslp_index, 0.01) for hour in range(7)]
    )
    q1000_trajectory = np.stack(
        [read_field(hour, q1000_index, 1.0) for hour in range(7)]
    )
    mslp_truth = np.asarray(mslp["truth"])
    q1000_truth = np.asarray(q1000["truth"])
    mslp_max_abs_error = float(
        np.max(np.abs(mslp_trajectory[1:6] - mslp_truth))
    )
    q1000_max_abs_error = float(
        np.max(np.abs(q1000_trajectory[1:6] - q1000_truth))
    )
    if mslp_max_abs_error > 2e-3:
        raise ValueError(
            f"case MSLP target differs from memmap by {mslp_max_abs_error}"
        )
    if q1000_max_abs_error > 2e-7:
        raise ValueError(
            f"case Q1000 target differs from memmap by {q1000_max_abs_error}"
        )

    with np.load(args.base_mslp, allow_pickle=True) as base:
        methods = list(base["methods"])
        if not np.array_equal(base["taus"], EXPECTED_TAUS):
            raise ValueError("historical case has a different query grid")
        base_latitude = np.asarray(base["lat"], dtype=np.float64) + 0.125
        base_longitude = np.asarray(base["lon"], dtype=np.float64) + 0.125
        if not np.allclose(base_latitude, latitude, rtol=0.0, atol=1e-6):
            raise ValueError("historical case latitude does not match")
        if not np.allclose(base_longitude, longitude, rtol=0.0, atol=1e-6):
            raise ValueError("historical case longitude does not match")
        base_panels = np.asarray(base["panels"], dtype=np.float32)
        historical_truth = base_panels[methods.index("ERA5")]
        mslp_linear = base_panels[methods.index("Bilinear")]
    if not np.allclose(historical_truth, mslp_truth, rtol=0.0, atol=1e-4):
        raise ValueError("WeatherBridge and historical ERA5 targets differ")

    fractions = EXPECTED_TAUS.astype(np.float32)[:, None, None] / 6.0
    q1000_linear = (
        (1.0 - fractions) * q1000_trajectory[0]
        + fractions * q1000_trajectory[6]
    ).astype(np.float32)

    provenance = {
        "schema_version": 1,
        "event": "Typhoon Haishen",
        "initial_time": EXPECTED_INIT_TIME,
        "query_hours": EXPECTED_TAUS.tolist(),
        "grid": {
            "name": "wb2_0p25_2x2_block_average_v1",
            "shape": [int(latitude.size), int(longitude.size)],
            "latitude_first_last": [float(latitude[0]), float(latitude[-1])],
            "longitude_first_last": [
                float(longitude[0]),
                float(longitude[-1]),
            ],
        },
        "target_verification": {
            "mslp_max_abs_hpa": mslp_max_abs_error,
            "q1000_max_abs_kg_kg-1": q1000_max_abs_error,
        },
        "source_sha256": {
            str(args.metadata): sha256_file(args.metadata),
            str(args.base_mslp): sha256_file(args.base_mslp),
            str(args.weatherbridge_mslp): sha256_file(
                args.weatherbridge_mslp
            ),
            str(args.weatherbridge_q1000): sha256_file(
                args.weatherbridge_q1000
            ),
        },
        "weatherbridge_model": mslp["provenance"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        latitude=latitude.astype(np.float32),
        longitude=longitude.astype(np.float32),
        taus=EXPECTED_TAUS,
        mslp_anchor0=mslp_trajectory[0],
        mslp_anchor6=mslp_trajectory[6],
        mslp_truth=mslp_truth,
        mslp_linear=mslp_linear,
        mslp_weatherbridge=np.asarray(mslp["prediction"]),
        q1000_anchor0=q1000_trajectory[0],
        q1000_anchor6=q1000_trajectory[6],
        q1000_truth=q1000_truth,
        q1000_linear=q1000_linear,
        q1000_weatherbridge=np.asarray(q1000["prediction"]),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "size_bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "mslp_target_max_abs_hpa": mslp_max_abs_error,
                "q1000_target_max_abs_kg_kg-1": q1000_max_abs_error,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
