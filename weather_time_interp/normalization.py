"""Canonical ERA5 normalisation shared by training and evaluation tools."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PAPER_CHANNELS_24: tuple[str, ...] = (
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
)
STATIC_FEATURES_3: tuple[str, ...] = (
    "land_sea_mask",
    "normalized_orography",
    "cosine_latitude",
)
STATIC_COSINE_LATITUDE_GRID = "legacy_symmetric_89p75_to_minus89p75"


def static_feature_provenance(path: str | Path) -> dict[str, Any]:
    """Fingerprint and semantically verify the three-channel static tensor."""
    provenance: dict[str, Any] = dict(file_provenance(path))
    try:
        import torch
    except ImportError as error:  # pragma: no cover - production has torch
        raise RuntimeError("PyTorch is required to verify static features") from error
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict):
        tensors = [value for value in payload.values() if hasattr(value, "shape")]
        if len(tensors) != 1:
            raise ValueError("static feature artifact must contain one tensor")
        payload = tensors[0]
    if not hasattr(payload, "detach"):
        raise ValueError("static feature artifact is not a tensor")
    values = np.asarray(payload.detach().cpu(), dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != len(STATIC_FEATURES_3):
        raise ValueError(
            "static feature tensor must have shape (3, latitude, longitude)"
        )
    semantic: dict[str, Any] = {
        "feature_names": list(STATIC_FEATURES_3),
        "shape": list(values.shape),
    }
    if values.shape[1:] == (360, 720):
        legacy_latitude = np.linspace(89.75, -89.75, 360, dtype=np.float64)
        expected = np.cos(np.deg2rad(legacy_latitude))[:, None]
        actual = values[2]
        if not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "production cosine-latitude static feature has unknown geometry"
            )
        block_latitude = np.linspace(89.875, -89.625, 360, dtype=np.float64)
        semantic.update(
            {
                "cosine_latitude_grid": STATIC_COSINE_LATITUDE_GRID,
                "latitude_degrees_first_last": [89.75, -89.75],
                "offset_from_wb2_block_centres_degrees": -0.125,
                "max_abs_value_difference_from_block_centres": float(
                    np.max(
                        np.abs(
                            expected[:, 0]
                            - np.cos(np.deg2rad(block_latitude))
                        )
                    )
                ),
            }
        )
    else:
        semantic["cosine_latitude_grid"] = "nonproduction_grid_not_verified"
    provenance["semantic"] = semantic
    return provenance


@dataclass(frozen=True)
class ChannelStats:
    channel_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        shape = (1,) * (values.ndim - 3) + (-1, 1, 1)
        return (values - self.mean.reshape(shape)) / self.std.reshape(shape)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        shape = (1,) * (values.ndim - 3) + (-1, 1, 1)
        return values * self.std.reshape(shape) + self.mean.reshape(shape)


def file_provenance(path: str | Path) -> dict[str, str | int]:
    """Return a stable fingerprint for an evaluation input."""
    resolved = Path(path).resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def zarr_store_provenance(
    path: str | Path,
    *,
    sampled_data_files: int = 32,
    sampled_bytes_per_file: int = 1024 * 1024,
) -> dict[str, Any]:
    """Fingerprint Zarr metadata, layout, state, and bounded chunk slices."""
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise ValueError(f"{resolved}: expected a Zarr directory")
    if sampled_data_files < 1:
        raise ValueError("sampled_data_files must be positive")
    if sampled_bytes_per_file < 3:
        raise ValueError("sampled_bytes_per_file must be at least 3")

    metadata_names = {
        ".zarray",
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "zarr.json",
    }
    files = sorted(
        candidate
        for candidate in resolved.rglob("*")
        if candidate.is_file()
    )
    if not files:
        raise ValueError(f"{resolved}: empty Zarr directory")

    metadata_digest = hashlib.sha256()
    layout_digest = hashlib.sha256()
    state_digest = hashlib.sha256()
    data_files: list[Path] = []
    total_size = 0
    metadata_count = 0
    for candidate in files:
        relative = candidate.relative_to(resolved).as_posix()
        stat = candidate.stat()
        total_size += stat.st_size
        layout_digest.update(
            f"{relative}\0{stat.st_size}\n".encode("utf-8")
        )
        state_digest.update(
            f"{relative}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
        if candidate.name in metadata_names:
            metadata_count += 1
            metadata_digest.update(f"{relative}\0".encode("utf-8"))
            metadata_digest.update(candidate.read_bytes())
            metadata_digest.update(b"\n")
        else:
            data_files.append(candidate)

    if not data_files:
        raise ValueError(f"{resolved}: Zarr store has no data chunks")
    if len(data_files) <= sampled_data_files:
        sampled = data_files
    elif sampled_data_files == 1:
        sampled = [data_files[len(data_files) // 2]]
    else:
        sampled = [
            data_files[
                round(
                    index
                    * (len(data_files) - 1)
                    / (sampled_data_files - 1)
                )
            ]
            for index in range(sampled_data_files)
        ]

    sampled_digest = hashlib.sha256()
    sampled_paths: list[str] = []
    sampled_total_bytes = 0
    for candidate in sampled:
        relative = candidate.relative_to(resolved).as_posix()
        sampled_paths.append(relative)
        size = candidate.stat().st_size
        segment_size = min(
            size,
            max(1, sampled_bytes_per_file // 3),
        )
        offsets = sorted(
            {
                0,
                max(0, (size - segment_size) // 2),
                max(0, size - segment_size),
            }
        )
        sampled_digest.update(
            f"{relative}\0{size}\0".encode("utf-8")
        )
        with candidate.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(segment_size)
                sampled_total_bytes += len(chunk)
                sampled_digest.update(
                    f"{offset}\0{len(chunk)}\0".encode("utf-8")
                )
                sampled_digest.update(chunk)
        sampled_digest.update(b"\n")

    identity = {
        "path": str(resolved),
        "metadata_sha256": metadata_digest.hexdigest(),
        "layout_sha256": layout_digest.hexdigest(),
        "mtime_state_sha256": state_digest.hexdigest(),
        "sampled_chunk_slices_sha256": sampled_digest.hexdigest(),
        "sampled_data_file_count": len(sampled),
        "sampled_bytes_per_file_limit": sampled_bytes_per_file,
    }
    return {
        **identity,
        "cache_identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "file_count": len(files),
        "metadata_file_count": metadata_count,
        "data_file_count": len(data_files),
        "total_size_bytes": total_size,
        "sampled_data_file_count": len(sampled),
        "sampled_bytes_per_file_limit": sampled_bytes_per_file,
        "sampled_total_bytes": sampled_total_bytes,
        "sampled_data_paths": sampled_paths,
    }


def load_channel_stats(
    stats_path: str | Path,
    surface_stats_path: str | Path,
    channel_names: Sequence[str] = PAPER_CHANNELS_24,
) -> ChannelStats:
    """Load means and standard deviations using the training dataset schema."""
    import xarray as xr

    names = tuple(channel_names)
    pressure_names = [name for name in names if name not in {"t2m", "u10", "v10", "mslp"}]
    with xr.open_dataset(stats_path) as dataset:
        stats = dataset["climate_statistics"].sel(params=pressure_names)
        pressure_mean = {
            name: float(value)
            for name, value in zip(pressure_names, stats.isel(stats=0).values)
        }
        pressure_std = {
            name: float(value)
            for name, value in zip(pressure_names, stats.isel(stats=1).values)
        }

    with open(surface_stats_path) as handle:
        surface = json.load(handle)

    mean = np.empty(len(names), dtype=np.float32)
    std = np.empty(len(names), dtype=np.float32)
    for index, name in enumerate(names):
        if name in surface:
            mean[index] = float(surface[name]["mean"])
            std[index] = float(surface[name]["std"])
        else:
            mean[index] = pressure_mean[name]
            std[index] = pressure_std[name]
    std = np.maximum(std, np.float32(1e-6))
    return ChannelStats(names, mean, std)
