#!/usr/bin/env python3
"""Export WeatherBridge tiles for the Haishen and Ida paper case studies."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from tools.eval.capmatched_loader import load_capmatched_checkpoint
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.normalization import file_provenance


TAUS = np.arange(1, 6, dtype=np.int32)
# Centres produced by averaging adjacent 0.25-degree WeatherBench-2 cells
# after dropping the terminal -90-degree row.
FULL_LAT = (89.875 - 0.5 * np.arange(360)).astype(np.float32)
FULL_LON = (0.125 + 0.5 * np.arange(720)).astype(np.float32)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _memmap_identity(raw: np.ndarray) -> dict[str, str | int]:
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


def _normalise_24(dataset: ERA5MemmapDataset, raw: np.ndarray) -> torch.Tensor:
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


def _physical_24(dataset: ERA5MemmapDataset, normalised: torch.Tensor) -> torch.Tensor:
    n_pl = dataset.in_channels
    pl = normalised[:n_pl] * dataset.sigma + dataset.mu
    surface = (
        normalised[n_pl:] * dataset.surface_sigma[: 24 - n_pl]
        + dataset.surface_mu[: 24 - n_pl]
    )
    return torch.cat([pl, surface], dim=0)


def _axis_indices(full: np.ndarray, selected: Iterable[float]) -> np.ndarray:
    selected_array = np.asarray(list(selected), dtype=np.float32)
    indices = np.asarray(
        [int(np.argmin(np.abs(full - value))) for value in selected_array],
        dtype=np.int64,
    )
    if len(np.unique(indices)) != len(indices):
        raise ValueError("requested case coordinates do not map one-to-one")
    if not np.allclose(full[indices], selected_array, rtol=0.0, atol=1.0e-6):
        raise ValueError("requested case coordinates are not memmap cell centres")
    return indices


def _write_case(
    *,
    model: torch.nn.Module,
    model_label: str,
    model_type: str,
    checkpoint: str,
    memmap_dir: str,
    static_path: str,
    stats_path: str,
    surface_stats_path: str,
    year: int,
    init_time: datetime,
    lat: np.ndarray,
    lon: np.ndarray,
    channels: dict[str, int],
    output_dir: Path,
    device: torch.device,
) -> None:
    dataset = ERA5MemmapDataset(
        memmap_dir=memmap_dir,
        years=[year],
        max_tau_hours=6,
        samples_per_date=1,
        train=False,
        eval_hours=TAUS.tolist(),
        static_path=static_path,
        stats_path=stats_path,
        surface_stats_path=surface_stats_path,
    )
    year_start = datetime(year, 1, 1)
    t0 = int((init_time - year_start).total_seconds() // 3600)
    raw = dataset.memmaps[year]
    raw_x0 = np.ascontiguousarray(raw[t0, :24])
    raw_xT = np.ascontiguousarray(raw[t0 + 6, :24])
    x0 = _normalise_24(dataset, raw_x0).unsqueeze(0).to(device)
    xT = _normalise_24(dataset, raw_xT).unsqueeze(0).to(device)
    static = torch.load(
        static_path,
        map_location=device,
        weights_only=False,
    )[:3].unsqueeze(0)

    lat_idx = _axis_indices(FULL_LAT, lat)
    lon_idx = _axis_indices(FULL_LON, lon)
    predictions = {name: [] for name in channels}
    targets = {name: [] for name in channels}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tau in TAUS:
            tau_norm = torch.tensor([float(tau) / 6.0], device=device)
            cond = torch.tensor([6.0], device=device)
            output = model(x0, xT, tau_norm, cond, static=static)
            pred_norm = output[0] if isinstance(output, tuple) else output
            pred_phys = _physical_24(dataset, pred_norm[0].float().cpu())
            target_phys = torch.from_numpy(
                np.array(raw[t0 + int(tau), :24], copy=True)
            ).float()
            for name, channel_idx in channels.items():
                pred = pred_phys[channel_idx].numpy()[np.ix_(lat_idx, lon_idx)]
                target = target_phys[channel_idx].numpy()[np.ix_(lat_idx, lon_idx)]
                if name == "mslp":
                    pred = pred / 100.0
                    target = target / 100.0
                predictions[name].append(pred.astype(np.float32))
                targets[name].append(target.astype(np.float32))

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in channels:
        panels = np.stack(predictions[name], axis=0)
        truth = np.stack(targets[name], axis=0)
        if not np.all(np.isfinite(panels)) or not np.all(np.isfinite(truth)):
            raise ValueError(f"{name}: case export contains non-finite values")
        rmse = np.sqrt(np.mean((panels - truth) ** 2, axis=(-2, -1)))
        output_path = output_dir / f"{name}.npz"
        provenance = {
            "schema_version": 3,
            "model_label": model_label,
            "model_type": model_type,
            "model_arch": str(getattr(model, "arch", "unknown")),
            "checkpoint": file_provenance(checkpoint),
            "static": file_provenance(static_path),
            "stats": file_provenance(stats_path),
            "surface_stats": file_provenance(surface_stats_path),
            "memmap": _memmap_identity(raw),
            "year": year,
            "t0_index": t0,
            "x0_24_sha256": _array_sha256(raw_x0),
            "xT_24_sha256": _array_sha256(raw_xT),
            "target_crop_sha256": _array_sha256(truth),
            "latitude_sha256": _array_sha256(lat),
            "longitude_sha256": _array_sha256(lon),
            "grid_convention": "wb2_0p25_pair_average_cell_centres_v1",
            "exporter": file_provenance(__file__),
        }
        np.savez(
            output_path,
            methods=np.asarray([model_label]),
            taus=TAUS,
            lat=lat,
            lon=lon,
            panels=panels[None],
            truth=truth,
            rmse=rmse[None],
            checkpoint=checkpoint,
            model_type=model_type,
            model_arch=str(getattr(model, "arch", "unknown")),
            init_time=init_time.isoformat(),
            provenance_json=json.dumps(provenance, sort_keys=True),
        )
        print(f"wrote {output_path} rmse={rmse.tolist()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-label", default="WeatherBridge")
    parser.add_argument(
        "--expected-arch",
        choices=("flow_pp3", "flow_pp3_detail"),
        required=True,
    )
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument(
        "--output-root",
        default="metrics/case_studies_weatherbridge",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    device = torch.device(args.device)
    model, model_type = load_capmatched_checkpoint(
        args.checkpoint,
        device,
        static_path=args.static_path,
    )
    if model_type not in {"weatherbridge", "weatherbridge_flow_spectral"} and not (
        model_type.startswith("capmatched_flow")
    ):
        raise ValueError(f"expected WeatherBridge checkpoint, got {model_type}")
    if model.arch != args.expected_arch:
        raise ValueError(
            f"expected arch {args.expected_arch}, checkpoint records {model.arch}"
        )
    _write_case(
        model=model,
        model_label=args.model_label,
        model_type=model_type,
        checkpoint=args.checkpoint,
        memmap_dir=args.memmap_dir,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
        year=2020,
        init_time=datetime(2020, 9, 7, 0),
        lat=np.arange(54.875, 19.874, -0.5, dtype=np.float32),
        lon=np.arange(100.125, 145.126, 0.5, dtype=np.float32),
        channels={"mslp": 23},
        output_dir=output_root / "typhoon_haishen",
        device=device,
    )
    _write_case(
        model=model,
        model_label=args.model_label,
        model_type=model_type,
        checkpoint=args.checkpoint,
        memmap_dir=args.memmap_dir,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
        year=2021,
        init_time=datetime(2021, 8, 29, 12),
        lat=np.arange(59.875, 19.874, -0.5, dtype=np.float32),
        lon=np.arange(230.125, 300.126, 0.5, dtype=np.float32),
        channels={"u10": 21, "v10": 22},
        output_dir=output_root / "hurricane_ida_2021",
        device=device,
    )


if __name__ == "__main__":
    main()
