#!/usr/bin/env python3
"""Build a provenance-bound, north-to-south HRES forecast memmap archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)
from weather_time_interp.normalization import PAPER_CHANNELS_24


DEFAULT_SOURCE = (
    "gs://weatherbench2/datasets/hres/"
    "2016-2022-0012-1440x721.zarr"
)
PRESSURE_VARIABLES = {
    "T": "temperature",
    "U": "u_component_of_wind",
    "V": "v_component_of_wind",
    "Q": "specific_humidity",
    "Z": "geopotential",
}
SURFACE_VARIABLES = {
    "t2m": "2m_temperature",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "mslp": "mean_sea_level_pressure",
}
LEGACY_QC_POLICY = (
    "finite_only_2x2_mean_require_at_least_3_of_4_native_values"
)
QC_POLICY = (
    "finite_only_2x2_mean_require_at_least_2_of_4_native_values_"
    "and_nonfinite_fraction_le_1e-6"
)
MAX_SOURCE_NONFINITE_FRACTION = 1.0e-6
LEGACY_BUILDER_SHA256 = {
    "723d194dc769fd0ae45638263fa5025f0bb56bf79a3c7b91f11095c8299bfe23"
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_init_times(value: str) -> list[np.datetime64]:
    raw = [item.strip() for item in value.split(",") if item.strip()]
    if not raw:
        raise ValueError("--init-times must contain at least one timestamp")
    result = [np.datetime64(item, "h") for item in raw]
    if len(result) != len(set(map(str, result))):
        raise ValueError("--init-times contains duplicates")
    if result != sorted(result):
        raise ValueError("--init-times must be sorted")
    return result


def lead_hours(coordinate) -> np.ndarray:
    """Decode the WB2 forecast-period coordinate without unit guessing."""
    values = np.asarray(coordinate.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        hours = values / np.timedelta64(1, "h")
        if not np.all(np.isfinite(hours)) or not np.array_equal(
            hours,
            np.rint(hours),
        ):
            raise ValueError("forecast periods are not integral hours")
        result = np.rint(hours).astype(np.int64)
    elif np.issubdtype(values.dtype, np.integer):
        if coordinate.attrs.get("units") != "hours":
            raise ValueError("integer forecast periods must declare units=hours")
        result = values.astype(np.int64, copy=False)
    else:
        raise ValueError(f"unsupported forecast-period dtype {values.dtype}")
    return np.asarray(result, dtype=np.int64)


def canonical_source_indices(
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source indices and exact 2x2 block-average coordinates."""
    lat = np.asarray(latitude, dtype=np.float64)
    lon = np.asarray(longitude, dtype=np.float64)
    expected_lat = np.linspace(-90.0, 90.0, 721, dtype=np.float64)
    expected_lon = np.arange(1440, dtype=np.float64) * 0.25
    if lat.shape != expected_lat.shape or not np.array_equal(lat, expected_lat):
        raise ValueError("HRES latitude must be -90..90 at 0.25 degree")
    if lon.shape != expected_lon.shape or not np.array_equal(lon, expected_lon):
        raise ValueError("HRES longitude must be 0..359.75 at 0.25 degree")

    # Reverse before trimming, so the dropped row is the south pole and the
    # resulting centres match the ERA5 preprocessing contract exactly.
    lat_indices = np.arange(720, 0, -1, dtype=np.int64)
    lon_indices = np.arange(1440, dtype=np.int64)
    coarse_lat = lat[lat_indices].reshape(360, 2).mean(axis=1)
    coarse_lon = lon[lon_indices].reshape(720, 2).mean(axis=1)
    np.testing.assert_array_equal(coarse_lat, wb2_block_average_latitudes())
    np.testing.assert_array_equal(coarse_lon, wb2_block_average_longitudes())
    return lat_indices, lon_indices, coarse_lat, coarse_lon


def consolidated_metadata(source: str) -> tuple[str, int]:
    import fsspec

    with fsspec.open(f"{source.rstrip('/')}/.zmetadata", "rb", token="anon") as stream:
        payload = stream.read()
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _channel_field(dataset, channel: str):
    if channel in SURFACE_VARIABLES:
        variable = SURFACE_VARIABLES[channel]
        if variable not in dataset:
            raise ValueError(f"HRES source lacks {variable}")
        return dataset[variable]
    variable = PRESSURE_VARIABLES[channel[0]]
    level = int(channel[1:])
    if variable not in dataset:
        raise ValueError(f"HRES source lacks {variable}")
    if level not in set(int(value) for value in dataset.level.values):
        raise ValueError(f"HRES source lacks {level} hPa")
    return dataset[variable].sel(level=level)


def _coarsen_channel(
    field,
    lat_indices: np.ndarray,
    lon_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    selected = field.isel(latitude=lat_indices, longitude=lon_indices)
    values = np.asarray(
        selected.transpose(
            "prediction_timedelta",
            "latitude",
            "longitude",
        ).values,
        dtype=np.float32,
    )
    if values.shape[1:] != (720, 1440):
        raise ValueError(f"unexpected selected field shape {values.shape}")
    return finite_block_mean(values)


def finite_block_mean(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, int | float]]:
    """Average finite 2x2 values under a strict sparse-missingness gate."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (720, 1440):
        raise ValueError("values must have shape (lead, 720, 1440)")
    blocks = values.reshape(values.shape[0], 360, 2, 720, 2)
    finite = np.isfinite(blocks)
    counts = finite.sum(axis=(2, 4), dtype=np.int8)
    minimum = int(counts.min())
    if minimum < 2:
        bad = np.argwhere(counts < 2)
        raise ValueError(
            "HRES block has fewer than two finite native values; "
            f"first={bad[0].tolist()} count={int(counts[tuple(bad[0])])}"
        )
    sums = np.where(finite, blocks, 0.0).sum(
        axis=(2, 4),
        dtype=np.float32,
    )
    output = sums / counts.astype(np.float32)
    source_nonfinite = int((~finite).sum())
    source_nonfinite_fraction = float(source_nonfinite / finite.size)
    if source_nonfinite_fraction > MAX_SOURCE_NONFINITE_FRACTION:
        raise ValueError(
            "HRES source non-finite fraction exceeds the fixed QC limit; "
            f"fraction={source_nonfinite_fraction:.12g} "
            f"limit={MAX_SOURCE_NONFINITE_FRACTION:.12g}"
        )
    affected = int((counts < 4).sum())
    qc: dict[str, int | float] = {
        "source_nonfinite_values": source_nonfinite,
        "affected_output_values": affected,
        "minimum_finite_native_values_per_output": minimum,
        "output_nonfinite_values": int((~np.isfinite(output)).sum()),
        "source_nonfinite_fraction": source_nonfinite_fraction,
    }
    return output, qc


def _validated_existing_record(
    binary_path: Path,
    sidecar_path: Path,
    *,
    source: str,
    source_metadata_sha256: str,
    builder_sha256: str,
    grid: dict[str, Any],
) -> dict[str, Any] | None:
    if not binary_path.exists() and not sidecar_path.exists():
        return None
    if not binary_path.is_file() or not sidecar_path.is_file():
        raise FileExistsError(
            f"partial existing output for {binary_path.stem}"
        )
    sidecar = json.loads(sidecar_path.read_text())
    stat = binary_path.stat()
    recorded = sidecar.get("binary", {})
    source_qc = sidecar.get("source_qc", {})
    channel_qc = source_qc.get("channels", {})
    sidecar_schema = sidecar.get("schema_version")
    sidecar_policy = source_qc.get("policy")
    sidecar_builder_sha256 = sidecar.get("builder", {}).get("sha256")
    legacy_record = (
        sidecar_schema == 3
        and sidecar_policy == LEGACY_QC_POLICY
        and sidecar_builder_sha256 in LEGACY_BUILDER_SHA256
        and int(source_qc.get("minimum_finite_native_values_per_output", -1))
        >= 3
    )
    current_record = (
        sidecar_schema == 4
        and sidecar_policy == QC_POLICY
        and sidecar_builder_sha256 == builder_sha256
        and int(source_qc.get("minimum_finite_native_values_per_output", -1))
        >= 2
        and max(
            (
                float(item.get("source_nonfinite_fraction", float("inf")))
                for item in channel_qc.values()
            ),
            default=float("inf"),
        )
        <= MAX_SOURCE_NONFINITE_FRACTION
    )
    if (
        not (legacy_record or current_record)
        or sidecar.get("grid") != grid
        or sidecar.get("source", {}).get("uri") != source
        or sidecar.get("source", {}).get("consolidated_metadata_sha256")
        != source_metadata_sha256
        or stat.st_size != int(recorded.get("size_bytes", -1))
        or stat.st_mtime_ns != int(recorded.get("mtime_ns", -1))
        or set(channel_qc) != set(PAPER_CHANNELS_24)
        or int(source_qc.get("output_nonfinite_values", -1)) != 0
    ):
        raise ValueError(f"stale existing output {binary_path}")
    actual_sha256 = sha256_file(binary_path)
    if actual_sha256 != recorded.get("sha256"):
        raise ValueError(f"corrupt existing output {binary_path}")
    return {
        "sidecar": sidecar_path.name,
        "sidecar_sha256": sha256_file(sidecar_path),
        "binary_sha256": actual_sha256,
        "binary_size_bytes": stat.st_size,
        "binary_mtime_ns": stat.st_mtime_ns,
    }


def build_archive(
    source: str,
    out_dir: Path,
    init_times: list[np.datetime64],
) -> dict[str, Any]:
    import xarray as xr

    dataset = xr.open_zarr(
        source,
        storage_options={"token": "anon"},
        consolidated=True,
    )
    lat_indices, lon_indices, latitude, longitude = canonical_source_indices(
        dataset.latitude.values,
        dataset.longitude.values,
    )
    source_metadata_sha256, source_metadata_size = consolidated_metadata(source)
    available_times = set(
        np.asarray(dataset.time.values).astype("datetime64[h]").tolist()
    )
    missing = [str(value) for value in init_times if value.tolist() not in available_times]
    if missing:
        raise ValueError(f"HRES source lacks requested init times: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)
    builder_path = Path(__file__).resolve()
    builder_sha256 = sha256_file(builder_path)
    grid = {
        "name": WB2_BLOCK_GRID_NAME,
        "latitude_order": "north_to_south",
        "latitude_sha256": sha256_array(latitude),
        "longitude_sha256": sha256_array(longitude),
    }
    files: dict[str, Any] = {}
    for init_time in init_times:
        tag = np.datetime_as_string(init_time, unit="h").replace(":", "")
        stem = f"init_{tag}"
        binary_path = out_dir / f"{stem}.bin"
        sidecar_path = out_dir / f"{stem}.json"
        existing = _validated_existing_record(
            binary_path,
            sidecar_path,
            source=source,
            source_metadata_sha256=source_metadata_sha256,
            builder_sha256=builder_sha256,
            grid=grid,
        )
        if existing is not None:
            files[binary_path.name] = existing
            print(f"[{stem}] validated existing output", flush=True)
            continue
        init_dataset = dataset.sel(time=init_time)
        leads = lead_hours(init_dataset.prediction_timedelta)
        if leads.ndim != 1 or not np.array_equal(
            leads,
            np.arange(0, 241, 6, dtype=np.int64),
        ):
            raise ValueError(f"{init_time}: unexpected HRES lead grid")

        temporary = binary_path.with_name(f".{binary_path.name}.{os.getpid()}.tmp")
        shape = (len(leads), len(PAPER_CHANNELS_24), 360, 720)
        try:
            output = np.memmap(
                temporary,
                dtype="float32",
                mode="w+",
                shape=shape,
            )
            channel_qc: dict[str, dict[str, int | float]] = {}
            for channel_index, channel in enumerate(PAPER_CHANNELS_24):
                coarse, qc = _coarsen_channel(
                    _channel_field(init_dataset, channel),
                    lat_indices,
                    lon_indices,
                )
                output[:, channel_index] = coarse
                channel_qc[channel] = qc
                output.flush()
                print(
                    f"[{stem}] {channel_index + 1:02d}/24 {channel} "
                    f"missing_native={qc['source_nonfinite_values']}",
                    flush=True,
                )
            if not np.isfinite(output).all():
                raise ValueError(f"{init_time}: non-finite values in HRES output")
            output.flush()
            del output
            os.replace(temporary, binary_path)
        finally:
            temporary.unlink(missing_ok=True)

        binary_stat = binary_path.stat()
        binary_sha256 = sha256_file(binary_path)
        sidecar = {
            "schema_version": 4,
            "init_time": np.datetime_as_string(init_time, unit="h"),
            "lead_hours": leads.tolist(),
            "shape": list(shape),
            "channel_order": list(PAPER_CHANNELS_24),
            "channels_available": list(PAPER_CHANNELS_24),
            "dtype": "float32",
            "grid": grid,
            "source": {
                "uri": source,
                "consolidated_metadata_sha256": source_metadata_sha256,
                "consolidated_metadata_size_bytes": source_metadata_size,
                "native_grid": "0.25_degree_721x1440_south_to_north",
                "transform": (
                    "reverse_latitude_then_drop_south_pole_then_"
                    "finite_only_unweighted_2x2_block_average"
                ),
            },
            "source_qc": {
                "policy": QC_POLICY,
                "channels": channel_qc,
                "source_nonfinite_values": sum(
                    int(item["source_nonfinite_values"])
                    for item in channel_qc.values()
                ),
                "affected_output_values": sum(
                    int(item["affected_output_values"])
                    for item in channel_qc.values()
                ),
                "minimum_finite_native_values_per_output": min(
                    int(item["minimum_finite_native_values_per_output"])
                    for item in channel_qc.values()
                ),
                "output_nonfinite_values": sum(
                    int(item["output_nonfinite_values"])
                    for item in channel_qc.values()
                ),
                "maximum_source_nonfinite_fraction_per_channel": max(
                    float(item["source_nonfinite_fraction"])
                    for item in channel_qc.values()
                ),
            },
            "builder": {
                "path": str(builder_path),
                "sha256": builder_sha256,
            },
            "binary": {
                "size_bytes": binary_stat.st_size,
                "mtime_ns": binary_stat.st_mtime_ns,
                "sha256": binary_sha256,
            },
        }
        atomic_json(sidecar_path, sidecar)
        files[binary_path.name] = {
            "sidecar": sidecar_path.name,
            "sidecar_sha256": sha256_file(sidecar_path),
            "binary_sha256": binary_sha256,
            "binary_size_bytes": binary_stat.st_size,
            "binary_mtime_ns": binary_stat.st_mtime_ns,
        }
        print(f"[{stem}] complete sha256={binary_sha256}", flush=True)

    manifest = {
        "schema_version": 2,
        "archive_kind": "weatherbench2_ifs_hres_forecast_anchors",
        "source": {
            "uri": source,
            "consolidated_metadata_sha256": source_metadata_sha256,
            "consolidated_metadata_size_bytes": source_metadata_size,
        },
        "grid": grid,
        "builder": {"path": str(builder_path), "sha256": builder_sha256},
        "source_qc_policy": QC_POLICY,
        "init_times": [
            np.datetime_as_string(value, unit="h") for value in init_times
        ],
        "files": files,
    }
    atomic_json(out_dir / "forecast_archive_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--init-times",
        required=True,
        help="Sorted comma-separated UTC initialisation timestamps.",
    )
    args = parser.parse_args()
    manifest = build_archive(
        args.source,
        args.out_dir,
        parse_init_times(args.init_times),
    )
    print(
        json.dumps(
            {
                "archive": str(args.out_dir),
                "n_inits": len(manifest["init_times"]),
            }
        )
    )


if __name__ == "__main__":
    main()
