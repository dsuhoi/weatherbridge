#!/usr/bin/env python3
"""Export matched-model 10-m wind fields for Storm Ciara."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tools.eval.batch_eval_memmap import load_model_safe
from tools.eval.export_weatherbridge_cases import (
    FULL_LAT,
    FULL_LON,
    TAUS,
    _array_sha256,
    _axis_indices,
    _memmap_identity,
    _normalise_24,
    _physical_24,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.normalization import file_provenance


YEAR = 2020
INIT_TIME = datetime(2020, 2, 9, 9)
LAT = np.arange(64.875, 44.874, -0.5, dtype=np.float32)
LON = np.concatenate(
    (
        np.arange(335.125, 360.0, 0.5, dtype=np.float32),
        np.arange(0.125, 20.126, 0.5, dtype=np.float32),
    )
)
U10_INDEX = 21
V10_INDEX = 22


def _crop(field: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    return field[np.ix_(lat_idx, lon_idx)].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device(args.device)
    model, model_type = load_model_safe(str(checkpoint), device, {})
    dataset = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[YEAR],
        max_tau_hours=6,
        samples_per_date=1,
        train=False,
        eval_hours=TAUS.tolist(),
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    year_start = datetime(YEAR, 1, 1)
    t0 = int((INIT_TIME - year_start).total_seconds() // 3600)
    raw = dataset.memmaps[YEAR]
    raw_x0 = np.ascontiguousarray(raw[t0, :24])
    raw_xT = np.ascontiguousarray(raw[t0 + 6, :24])
    x0 = _normalise_24(dataset, raw_x0).unsqueeze(0).to(device)
    xT = _normalise_24(dataset, raw_xT).unsqueeze(0).to(device)
    static = torch.load(
        args.static_path,
        map_location=device,
        weights_only=False,
    )[:3].unsqueeze(0)
    lat_idx = _axis_indices(FULL_LAT, LAT)
    lon_idx = _axis_indices(FULL_LON, LON)

    anchor0_u = _crop(raw_x0[U10_INDEX], lat_idx, lon_idx)
    anchor0_v = _crop(raw_x0[V10_INDEX], lat_idx, lon_idx)
    anchorT_u = _crop(raw_xT[U10_INDEX], lat_idx, lon_idx)
    anchorT_v = _crop(raw_xT[V10_INDEX], lat_idx, lon_idx)
    pred_u: list[np.ndarray] = []
    pred_v: list[np.ndarray] = []
    truth_u: list[np.ndarray] = []
    truth_v: list[np.ndarray] = []
    linear_u: list[np.ndarray] = []
    linear_v: list[np.ndarray] = []

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for tau in TAUS:
            fraction = float(tau) / 6.0
            tau_norm = torch.tensor([fraction], device=device)
            cond = torch.tensor([6.0], device=device)
            output = model(x0, xT, tau_norm, cond, static=static)
            pred_norm = output[0] if isinstance(output, tuple) else output
            pred_phys = _physical_24(dataset, pred_norm[0].float().cpu()).numpy()
            target = np.array(raw[t0 + int(tau), :24], copy=True)
            pred_u.append(_crop(pred_phys[U10_INDEX], lat_idx, lon_idx))
            pred_v.append(_crop(pred_phys[V10_INDEX], lat_idx, lon_idx))
            truth_u.append(_crop(target[U10_INDEX], lat_idx, lon_idx))
            truth_v.append(_crop(target[V10_INDEX], lat_idx, lon_idx))
            linear_u.append((1.0 - fraction) * anchor0_u + fraction * anchorT_u)
            linear_v.append((1.0 - fraction) * anchor0_v + fraction * anchorT_v)

    arrays = {
        "prediction_u": np.stack(pred_u),
        "prediction_v": np.stack(pred_v),
        "truth_u": np.stack(truth_u),
        "truth_v": np.stack(truth_v),
        "linear_u": np.stack(linear_u),
        "linear_v": np.stack(linear_v),
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise ValueError("Storm Ciara export contains non-finite values")

    vector_rmse = np.sqrt(
        np.mean(
            (arrays["prediction_u"] - arrays["truth_u"]) ** 2
            + (arrays["prediction_v"] - arrays["truth_v"]) ** 2,
            axis=(-2, -1),
        )
    )
    provenance = {
        "schema_version": 1,
        "event": "Storm Ciara",
        "model_label": args.model_label,
        "model_type": model_type,
        "model_arch": str(getattr(model, "arch", "unknown")),
        "checkpoint": file_provenance(checkpoint),
        "static": file_provenance(args.static_path),
        "stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
        "memmap": _memmap_identity(raw),
        "year": YEAR,
        "t0_index": t0,
        "x0_24_sha256": _array_sha256(raw_x0),
        "xT_24_sha256": _array_sha256(raw_xT),
        "truth_u_sha256": _array_sha256(arrays["truth_u"]),
        "truth_v_sha256": _array_sha256(arrays["truth_v"]),
        "linear_u_sha256": _array_sha256(arrays["linear_u"]),
        "linear_v_sha256": _array_sha256(arrays["linear_v"]),
        "latitude_sha256": _array_sha256(LAT),
        "longitude_sha256": _array_sha256(LON),
        "grid_convention": "wb2_0p25_pair_average_cell_centres_v1",
        "exporter": file_provenance(__file__),
    }
    output_dir = Path(args.output_root) / "storm_ciara_2020"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wind10.npz"
    np.savez(
        output_path,
        methods=np.asarray([args.model_label]),
        taus=TAUS,
        lat=LAT,
        lon=LON,
        init_time=INIT_TIME.isoformat(),
        checkpoint=str(checkpoint),
        model_type=model_type,
        model_arch=str(getattr(model, "arch", "unknown")),
        vector_rmse=vector_rmse,
        provenance_json=json.dumps(provenance, sort_keys=True),
        **arrays,
    )
    print(f"wrote {output_path} vector_rmse={vector_rmse.tolist()}", flush=True)


if __name__ == "__main__":
    main()
