"""Verbatim re-implementation of the legacy ``numerical_baseline_eval.py`` body
that does NOT import ``evaluate_baselines`` (which pulls in unavailable
``models.resnet`` deps).

The bilinear closed-form is inlined character-for-character from
``interpolate_time_with_f_interpolate(..., mode='bilinear')`` so the produced
JSON files are bit-identical to what the legacy script would produce.

This file is used ONLY by the test suite to generate the ground-truth snapshot
in ``tests/fixtures/legacy_json/numerical/`` against which the refactored
``NumericalBaselineRunner`` is compared. Do not import in production code.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
from tools.baselines.numerical_baselines import (
    semi_lagrangian_interp,
    hermite_advection_interp,
    settls_interp,
    hermite_diffusion_interp,
    build_uv_assignment,
    expand_uv_to_channels,
)


def _bilinear(x0: torch.Tensor, x1: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Inlined copy of ``interpolate_time_with_f_interpolate(mode='bilinear')``."""
    tau_ = tau.float().view(-1)
    B = x0.size(0)
    if tau_.numel() == 1 and B > 1:
        tau_ = tau_.expand(B)
    tau_b = tau_.view(B, 1, 1, 1)
    return (1.0 - tau_b) * x0 + tau_b * x1


def lat_weights(H: int, device, dtype) -> torch.Tensor:
    if H == 360:
        lat = np.linspace(89.75, -89.75, H, dtype=np.float32)
    elif H == 181:
        lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    else:
        lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    return torch.from_numpy(w).to(device=device, dtype=dtype).view(1, 1, -1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", required=True)
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", required=True)
    ap.add_argument("--surface-stats-path", required=True)
    ap.add_argument("--static-path", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--samples-per-date", type=int, default=2)
    ap.add_argument("--eval-days-per-month", type=int, default=5)
    ap.add_argument("--dt-hours", type=float, default=6.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    t_global = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=6,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=list(range(7)),
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )

    if args.eval_days_per_month is not None:
        K = max(1, int(args.eval_days_per_month))
        day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}.get(
            K, sorted({1 + i * (30 // K) for i in range(K)})
        )
        allowed = set(day_picks)
        filt = []
        for entry in ds_base.index:
            y, t0, _, _ = entry
            doy = t0 // 24
            try:
                d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
            except Exception:
                continue
            if d.day in allowed:
                filt.append(entry)
        print(f"  economy filter: {len(filt)}/{len(ds_base.index)} index entries")
        ds_base.index = filt

    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)

    uv_map = build_uv_assignment(channel_names)

    mu_all = torch.cat([ds_base.mu, ds_base.surface_mu]).to(device).view(1, -1, 1, 1)
    sigma_all = torch.cat([ds_base.sigma, ds_base.surface_sigma]).to(device).view(1, -1, 1, 1)

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=args.dt_hours)
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    H = list(ds_base.memmaps.values())[0].shape[-2]
    w_lat = lat_weights(H, device, torch.float32)

    methods = ["bilinear", "semi_lagrangian", "hermite_advection", "settls", "hermite_diffusion"]
    sum_sq = {m: {h: {} for h in range(7)} for m in methods}
    n_per_hour = {h: 0 for h in range(7)}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_all = batch["tau"].to(device, non_blocking=True)
            tau_hour_all = batch["tau_hour"].long()
            target_all = batch["target"].to(device, non_blocking=True)

            B = x0.size(0); nH = tau_all.size(1)

            x0_phys = x0 * sigma_all + mu_all
            xT_phys = xT * sigma_all + mu_all
            u0p, v0p = expand_uv_to_channels(x0_phys, uv_map)
            uTp, vTp = expand_uv_to_channels(xT_phys, uv_map)

            for h_idx in range(nH):
                tau_h = tau_all[:, h_idx, 0]
                target_h = target_all[:, h_idx]
                hours_per_sample = tau_hour_all[:, h_idx, 0].tolist()

                pred_bil = _bilinear(x0, xT, tau_h)
                pred_sl  = semi_lagrangian_interp(x0, xT, u0p, v0p, uTp, vTp,
                                                  tau_h, dt_hours=args.dt_hours, n_iter=2)
                pred_hm  = hermite_advection_interp(x0, xT, u0p, v0p, uTp, vTp,
                                                    tau_h, dt_hours=args.dt_hours)
                pred_st  = settls_interp(x0, xT, u0p, v0p, uTp, vTp,
                                         tau_h, dt_hours=args.dt_hours, n_iter=3)
                pred_hd  = hermite_diffusion_interp(x0, xT, u0p, v0p, uTp, vTp,
                                                    tau_h, dt_hours=args.dt_hours)

                def per_ch_sq(pred):
                    return ((pred - target_h) ** 2 * w_lat).sum(dim=(-2, -1))

                err_map = {
                    "bilinear": per_ch_sq(pred_bil),
                    "semi_lagrangian": per_ch_sq(pred_sl),
                    "hermite_advection": per_ch_sq(pred_hm),
                    "settls": per_ch_sq(pred_st),
                    "hermite_diffusion": per_ch_sq(pred_hd),
                }

                for i in range(B):
                    h = int(hours_per_sample[i])
                    if h < 0 or h > 6:
                        continue
                    n_per_hour[h] += 1
                    for ci, name in enumerate(channel_names):
                        for method, e_tensor in err_map.items():
                            sum_sq[method][h].setdefault(name, 0.0)
                            sum_sq[method][h][name] += float(e_tensor[i, ci].item())

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for method in methods:
        per_hour_out = {}
        for h in range(7):
            n = n_per_hour[h]
            if n == 0:
                continue
            per_hour_out[str(h)] = {"model": {}, "bilinear": {}, "bicubic": {}}
            for name in channel_names:
                if name in sum_sq[method][h]:
                    rmse = float(np.sqrt(sum_sq[method][h][name] / n))
                    per_hour_out[str(h)]["model"][f"rmse_{name}"] = rmse
                    rmse_bil = float(np.sqrt(sum_sq["bilinear"][h][name] / n))
                    per_hour_out[str(h)]["bilinear"][f"rmse_{name}"] = rmse_bil
                    per_hour_out[str(h)]["bicubic"][f"rmse_{name}"] = rmse_bil
        payload = {
            "method": method,
            "num_samples": sum(n_per_hour.values()),
            "years": [args.test_year],
            "channel_names": channel_names,
            "per_hour": per_hour_out,
        }
        with open(out_dir / f"{method}.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"saved {out_dir / f'{method}.json'}")

    print(f"\n=== DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
