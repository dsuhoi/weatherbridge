#!/usr/bin/env python3
"""PySTEPS optical-flow + semi-Lagrangian extrapolation baseline.

Reference: Pulkkinen et al. (2019) "Pysteps: a community-driven open-source
library for precipitation nowcasting" Geosci. Model Dev. 12:4185–4219.

For each (x_0, x_T) pair compute per-channel dense Lucas-Kanade optical flow
V (in pixels per Δt), then extrapolate x_0 forward by τ·Δt and x_T backward by
(1-τ)·Δt and blend with weight (1-τ, τ).

CPU-bound; ~60-90 min for full 2020 (192 windows × 27 channels).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import ERA5WeatherHermiteDataset


def compute_motion_per_channel(x_0_np: np.ndarray, x_T_np: np.ndarray, valid_mask):
    """Dense Lucas-Kanade per channel. Returns list of motion fields (or None for skipped)."""
    from pysteps.motion.lucaskanade import dense_lucaskanade
    C = x_0_np.shape[0]
    motions = [None] * C
    for c in range(C):
        if not valid_mask[c]:
            continue
        try:
            arr = np.stack([x_0_np[c], x_T_np[c]], axis=0).astype(np.float32)
            V = dense_lucaskanade(arr, dense=True)
            motions[c] = V
        except Exception:
            motions[c] = None
    return motions


def extrap_per_channel(x_0_np, x_T_np, motions, tau_list):
    """For each tau in tau_list, extrapolate forward+backward+blend per channel.
    Returns array (n_tau, C, H, W).
    """
    from pysteps.extrapolation.semilagrangian import extrapolate
    C, H, W = x_0_np.shape
    out = np.zeros((len(tau_list), C, H, W), dtype=np.float32)
    # For channels without valid motion → straight linear blend per tau
    for c in range(C):
        V = motions[c]
        if V is None:
            for ti, tau in enumerate(tau_list):
                out[ti, c] = (1.0 - tau) * x_0_np[c] + tau * x_T_np[c]
            continue
        try:
            # forward from x_0 by tau (timesteps in units of Δt)
            x_fwd_all = extrapolate(x_0_np[c], V, timesteps=list(tau_list),
                                     allow_nonfinite_values=True)
            # backward from x_T by (1-tau): use −V and timesteps (1-tau)
            x_bwd_all = extrapolate(x_T_np[c], -V, timesteps=[1.0 - t for t in tau_list],
                                     allow_nonfinite_values=True)
            for ti, tau in enumerate(tau_list):
                fwd = x_fwd_all[ti]
                bwd = x_bwd_all[ti]
                # Replace NaN (out-of-frame regions) with the linear blend
                blend = (1.0 - tau) * x_0_np[c] + tau * x_T_np[c]
                fwd = np.where(np.isfinite(fwd), fwd, blend)
                bwd = np.where(np.isfinite(bwd), bwd, blend)
                out[ti, c] = (1.0 - tau) * fwd + tau * bwd
        except Exception:
            for ti, tau in enumerate(tau_list):
                out[ti, c] = (1.0 - tau) * x_0_np[c] + tau * x_T_np[c]
    return out


def lat_weights(H: int) -> np.ndarray:
    if H == 360:
        lat = np.linspace(89.75, -89.75, H, dtype=np.float32)
    elif H == 181:
        lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    else:
        lat = np.linspace(90.0, -90.0, H, dtype=np.float32)
    w = np.cos(np.deg2rad(lat)).astype(np.float32)
    w = w / w.sum()
    return w.reshape(1, -1, 1)  # (1, H, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="metrics/eval_0p5_2020_pysteps")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--dt-hours", type=float, default=6.0)
    # Optional: skip flow for SST (slow boundary field; LK fails on near-constant fields)
    ap.add_argument("--skip-channels", nargs="*", default=["sst"])
    args = ap.parse_args()

    t_global = time.time()
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[args.test_year], max_tau_hours=6,
        samples_per_date=args.samples_per_date, train=False,
        eval_hours=list(range(7)), static_path=args.static_path,
        stats_path=args.stats_path, surface_stats_path=args.surface_stats_path,
    )
    if args.eval_days_per_month is not None:
        import datetime as _dt
        K = max(1, int(args.eval_days_per_month))
        day_picks = {3:[1,11,21], 4:[1,8,15,22], 5:[1,7,14,21,28]}.get(K, sorted({1+i*(30//K) for i in range(K)}))
        allowed = set(day_picks); filt = []
        for entry in ds_base.index:
            y, t0, _, _ = entry
            doy = t0 // 24
            try:
                d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
            except Exception:
                continue
            if d.day in allowed:
                filt.append(entry)
        ds_base.index = filt
        print(f"  economy filter: {len(filt)} entries")

    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    C = len(channel_names)
    valid_mask = [name not in set(args.skip_channels) for name in channel_names]
    skipped = [n for n, v in zip(channel_names, valid_mask) if not v]
    print(f"  channels: {C}  (skip optical flow on: {skipped})")

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=args.dt_hours)
    print(f"  windows: {len(test_wrapped)}")
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=False, persistent_workers=True)

    H = list(ds_base.memmaps.values())[0].shape[-2]
    w_lat = lat_weights(H)  # (1, H, 1)

    sum_sq_pys = {h: {} for h in range(7)}
    sum_sq_bil = {h: {} for h in range(7)}
    n_per_hour = {h: 0 for h in range(7)}
    sample_t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        x0_b = batch["x0"].numpy()                                 # (B, C, H, W)
        xT_b = batch["xT"].numpy()
        tau_all_b = batch["tau"].numpy()                           # (B, nH, 1)
        tau_hour_all_b = batch["tau_hour"].long().numpy()          # (B, nH, 1)
        target_all_b = batch["target"].numpy()                     # (B, nH, C, H, W)
        B, nH = tau_all_b.shape[0], tau_all_b.shape[1]
        for i in range(B):
            tau_list = [float(tau_all_b[i, h_idx, 0]) for h_idx in range(nH)]
            motions = compute_motion_per_channel(x0_b[i], xT_b[i], valid_mask)
            preds = extrap_per_channel(x0_b[i], xT_b[i], motions, tau_list)   # (nH, C, H, W)
            for h_idx in range(nH):
                h = int(tau_hour_all_b[i, h_idx, 0])
                if h < 0 or h > 6:
                    continue
                tgt = target_all_b[i, h_idx]                                   # (C, H, W)
                pred = preds[h_idx]
                tau = tau_list[h_idx]
                bil = (1.0 - tau) * x0_b[i] + tau * xT_b[i]
                err_pys = ((pred - tgt) ** 2 * w_lat).sum(axis=(-2, -1))       # (C,)
                err_bil = ((bil  - tgt) ** 2 * w_lat).sum(axis=(-2, -1))
                n_per_hour[h] += 1
                for ci, name in enumerate(channel_names):
                    sum_sq_pys[h].setdefault(name, 0.0)
                    sum_sq_pys[h][name] += float(err_pys[ci])
                    sum_sq_bil[h].setdefault(name, 0.0)
                    sum_sq_bil[h][name] += float(err_bil[ci])
        elapsed = time.time() - sample_t0
        print(f"  batch {batch_idx+1}/{len(loader)}  elapsed={elapsed/60:.1f} min", flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for method, sums in (("pysteps_optical_flow", sum_sq_pys), ("bilinear_ref", sum_sq_bil)):
        per_hour_out = {}
        for h in range(7):
            n = n_per_hour[h]
            if n == 0:
                continue
            per_hour_out[str(h)] = {"model": {}, "bilinear": {}, "bicubic": {}}
            for name in channel_names:
                if name in sums[h]:
                    rmse = float(np.sqrt(sums[h][name] / n))
                    per_hour_out[str(h)]["model"][f"rmse_{name}"] = rmse
                    rmse_bil = float(np.sqrt(sum_sq_bil[h][name] / n))
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
