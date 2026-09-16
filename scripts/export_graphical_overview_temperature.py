#!/usr/bin/env python3
"""Export a provenance-bound temperature event for the graphical overview."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tools.eval.capmatched_loader import load_capmatched_checkpoint
from weather_time_interp.memmap_dataset import ERA5MemmapDataset

YEAR = 2021
INIT_TIME = datetime(2021, 6, 28, 18, tzinfo=timezone.utc)
LATITUDE = np.arange(61.875, 29.874, -0.5, dtype=np.float32)
LONGITUDE = np.arange(210.125, 275.126, 0.5, dtype=np.float32)
FIELDS = {"T850": 2, "t2m": 20}
FULL_LAT = (89.875 - 0.5 * np.arange(360)).astype(np.float32)
FULL_LON = (0.125 + 0.5 * np.arange(720)).astype(np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path_like: str | Path) -> dict[str, str | int]:
    path = Path(path_like).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def memmap_identity(raw: np.ndarray) -> dict[str, str | int]:
    filename = getattr(raw, "filename", None)
    if filename is None:
        return {"path": "unavailable"}
    path = Path(filename).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def axis_indices(full: np.ndarray, selected: Iterable[float]) -> np.ndarray:
    selected_array = np.asarray(list(selected), dtype=np.float32)
    indices = np.asarray(
        [int(np.argmin(np.abs(full - value))) for value in selected_array],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("requested coordinates do not map one-to-one")
    if not np.allclose(full[indices], selected_array, rtol=0.0, atol=1.0e-6):
        raise ValueError("requested coordinates are not memmap cell centres")
    return indices


def normalise_24(dataset: ERA5MemmapDataset, raw: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.array(raw[:24], copy=True)).float()
    n_pl = dataset.in_channels
    pl = (tensor[:n_pl] - dataset.mu) / dataset.sigma
    surface = (
        tensor[n_pl:] - dataset.surface_mu[: 24 - n_pl]
    ) / dataset.surface_sigma[: 24 - n_pl]
    return torch.nan_to_num(
        torch.cat([pl, surface], dim=0),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def physical_24(
    dataset: ERA5MemmapDataset,
    normalised: torch.Tensor,
) -> torch.Tensor:
    n_pl = dataset.in_channels
    pl = normalised[:n_pl] * dataset.sigma + dataset.mu
    surface = (
        normalised[n_pl:] * dataset.surface_sigma[: 24 - n_pl]
        + dataset.surface_mu[: 24 - n_pl]
    )
    return torch.cat([pl, surface], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device(args.device)
    model, model_type = load_capmatched_checkpoint(
        str(args.checkpoint),
        device,
        static_path=args.static_path,
    )
    if str(getattr(model, "arch", "")) != "flow_pp3":
        raise ValueError(f"expected flow_pp3, got {getattr(model, 'arch', None)}")

    dataset = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[YEAR],
        max_tau_hours=6,
        samples_per_date=1,
        train=False,
        eval_hours=[1, 2, 3, 4, 5],
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    year_start = datetime(YEAR, 1, 1, tzinfo=timezone.utc)
    t0 = int((INIT_TIME - year_start).total_seconds() // 3600)
    raw = dataset.memmaps[YEAR]
    raw_x0 = np.ascontiguousarray(raw[t0, :24])
    raw_xT = np.ascontiguousarray(raw[t0 + 6, :24])
    x0 = normalise_24(dataset, raw_x0).unsqueeze(0).to(device)
    xT = normalise_24(dataset, raw_xT).unsqueeze(0).to(device)
    static = torch.load(
        args.static_path,
        map_location=device,
        weights_only=False,
    )[:3].unsqueeze(0)
    query_hours = np.arange(1, 6, dtype=np.int8)
    condition = torch.tensor([6.0], device=device)
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    predictions = []
    with torch.no_grad(), autocast:
        for query_hour in query_hours:
            tau = torch.tensor(
                [float(query_hour) / 6.0],
                device=device,
            )
            output = model(x0, xT, tau, condition, static=static)
            prediction_norm = output[0] if isinstance(output, tuple) else output
            predictions.append(
                physical_24(
                    dataset,
                    prediction_norm[0].float().cpu(),
                ).numpy()
            )

    lat_indices = axis_indices(FULL_LAT, LATITUDE)
    lon_indices = axis_indices(FULL_LON, LONGITUDE)

    def crop(field: np.ndarray) -> np.ndarray:
        return np.asarray(
            field[np.ix_(lat_indices, lon_indices)],
            dtype=np.float32,
        )

    era5 = np.stack(
        [
            np.stack(
                [crop(raw[t0 + hour, channel]) for channel in FIELDS.values()]
            )
            for hour in range(7)
        ]
    )
    interior_predictions = np.stack(
        [
            np.stack(
                [crop(prediction[channel]) for channel in FIELDS.values()]
            )
            for prediction in predictions
        ]
    )
    midpoint_prediction = interior_predictions[2]
    midpoint_target = era5[3]
    interior_targets = era5[query_hours]
    interior_rmse = np.sqrt(
        np.mean(
            (interior_predictions - interior_targets) ** 2,
            axis=(-2, -1),
        )
    )
    midpoint_rmse = interior_rmse[2]
    if not np.all(np.isfinite(era5)) or not np.all(
        np.isfinite(interior_predictions)
    ):
        raise ValueError("temperature export contains non-finite values")

    provenance = {
        "schema_version": 2,
        "event_id": "pacific_northwest_heatwave",
        "event_label": "Pacific Northwest heatwave",
        "init_time": INIT_TIME.isoformat(),
        "query_hours": query_hours.tolist(),
        "fields": list(FIELDS),
        "model_label": "WeatherBridge",
        "model_type": model_type,
        "model_arch": str(model.arch),
        "checkpoint": file_provenance(args.checkpoint),
        "static": file_provenance(args.static_path),
        "stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
        "memmap": memmap_identity(raw),
        "x0_24_sha256": array_sha256(raw_x0),
        "xT_24_sha256": array_sha256(raw_xT),
        "era5_crop_sha256": array_sha256(era5),
        "prediction_crops_sha256": array_sha256(interior_predictions),
        "latitude_sha256": array_sha256(LATITUDE),
        "longitude_sha256": array_sha256(LONGITUDE),
        "grid_convention": "wb2_0p25_pair_average_cell_centres_v1",
        "exporter": file_provenance(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        event_id=np.asarray("pacific_northwest_heatwave"),
        event_label=np.asarray("Pacific Northwest heatwave"),
        init_time=np.asarray(INIT_TIME.isoformat()),
        fields=np.asarray(list(FIELDS)),
        latitude=LATITUDE,
        longitude=LONGITUDE,
        hours=np.arange(7, dtype=np.int8),
        era5=era5,
        query_hours=query_hours,
        interior_predictions=interior_predictions,
        interior_rmse=interior_rmse.astype(np.float32),
        midpoint_prediction=midpoint_prediction,
        midpoint_rmse=midpoint_rmse.astype(np.float32),
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "size_bytes": args.output.stat().st_size,
                "sha256": sha256_file(args.output),
                "rmse_k_by_query_hour": {
                    str(int(query_hour)): dict(
                        zip(
                            FIELDS,
                            interior_rmse[index].tolist(),
                            strict=True,
                        )
                    )
                    for index, query_hour in enumerate(query_hours)
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
