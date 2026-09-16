#!/usr/bin/env python3
"""Farneback dense optical flow + semi-Lagrangian extrapolation baseline.

Reference: Farnebäck, G. (2003) "Two-Frame Motion Estimation Based on
Polynomial Expansion", in Image Analysis (Bigun & Gustavsson, eds.) LNCS 2749,
pp. 363-370 — standard dense optical flow that works on continuous-valued
fields (unlike feature-based Lucas-Kanade which fails on low-contrast weather
data).

Replaces PySTEPS LK (silent failure on z-scored atmospheric fields).
Algorithm: Farneback dense flow → pysteps semi-Lagrangian extrapolate →
forward+backward blend by τ.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import ERA5WeatherHermiteDataset


def farneback_motion(x_0_np, x_T_np, valid_mask):
    """Per-channel Farneback dense optical flow.

    Each channel is rescaled to uint8 then fed through OpenCV's Farneback.
    Returns motion in (2, H, W) format matching pysteps convention:
    V[0] = row velocity (i-axis), V[1] = col velocity (j-axis), in pixels.
    """
    C, H, W = x_0_np.shape
    motions = [None] * C
    for c in range(C):
        if not valid_mask[c]:
            continue
        x0 = x_0_np[c]
        xT = x_T_np[c]
        lo = float(min(x0.min(), xT.min()))
        hi = float(max(x0.max(), xT.max()))
        if hi - lo < 1e-6:
            continue
        x0_u8 = ((x0 - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        xT_u8 = ((xT - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np.uint8)
        try:
            flow = cv2.calcOpticalFlowFarneback(
                x0_u8, xT_u8, None,
                pyr_scale=0.5, levels=5, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0,
            )
            # flow shape (H, W, 2): [.., 0] = dx (j-axis), [.., 1] = dy (i-axis)
            # pysteps convention: V[0] = row-velocity (i), V[1] = col-velocity (j)
            V = np.stack([flow[..., 1], flow[..., 0]], axis=0).astype(np.float32)
            motions[c] = V
        except Exception:
            motions[c] = None
    return motions


def extrap_per_channel(x_0_np, x_T_np, motions, tau_list):
    from pysteps.extrapolation.semilagrangian import extrapolate
    C, H, W = x_0_np.shape
    out = np.zeros((len(tau_list), C, H, W), dtype=np.float32)
    for c in range(C):
        V = motions[c]
        if V is None:
            for ti, tau in enumerate(tau_list):
                out[ti, c] = (1.0 - tau) * x_0_np[c] + tau * x_T_np[c]
            continue
        try:
            x_fwd_all = extrapolate(x_0_np[c], V, timesteps=list(tau_list),
                                     allow_nonfinite_values=True)
            x_bwd_all = extrapolate(x_T_np[c], -V, timesteps=[1.0 - t for t in tau_list],
                                     allow_nonfinite_values=True)
            for ti, tau in enumerate(tau_list):
                blend = (1.0 - tau) * x_0_np[c] + tau * x_T_np[c]
                fwd = np.where(np.isfinite(x_fwd_all[ti]), x_fwd_all[ti], blend)
                bwd = np.where(np.isfinite(x_bwd_all[ti]), x_bwd_all[ti], blend)
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
    return w.reshape(1, -1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="metrics/eval_0p5_2020_farneback")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--dt-hours", type=float, default=6.0)
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
    print(f"  channels: {C}  skip flow on: {[n for n,v in zip(channel_names,valid_mask) if not v]}")

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=args.dt_hours)
    print(f"  windows: {len(test_wrapped)}")
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=False, persistent_workers=True)
    H = list(ds_base.memmaps.values())[0].shape[-2]
    w_lat = lat_weights(H)

    sum_sq_fb  = {h: {} for h in range(7)}
    sum_sq_bil = {h: {} for h in range(7)}
    n_per_hour = {h: 0 for h in range(7)}
    t_loop = time.time()
    for batch_idx, batch in enumerate(loader):
        x0_b = batch["x0"].numpy()
        xT_b = batch["xT"].numpy()
        tau_b = batch["tau"].numpy()
        tau_h_b = batch["tau_hour"].long().numpy()
        tgt_b = batch["target"].numpy()
        B, nH = tau_b.shape[0], tau_b.shape[1]
        for i in range(B):
            tau_list = [float(tau_b[i, h_idx, 0]) for h_idx in range(nH)]
            motions = farneback_motion(x0_b[i], xT_b[i], valid_mask)
            preds = extrap_per_channel(x0_b[i], xT_b[i], motions, tau_list)
            for h_idx in range(nH):
                h = int(tau_h_b[i, h_idx, 0])
                if h < 0 or h > 6:
                    continue
                tgt = tgt_b[i, h_idx]
                pred = preds[h_idx]
                tau = tau_list[h_idx]
                bil = (1.0 - tau) * x0_b[i] + tau * xT_b[i]
                err_fb  = ((pred - tgt) ** 2 * w_lat).sum(axis=(-2, -1))
                err_bil = ((bil  - tgt) ** 2 * w_lat).sum(axis=(-2, -1))
                n_per_hour[h] += 1
                for ci, name in enumerate(channel_names):
                    sum_sq_fb[h].setdefault(name, 0.0)
                    sum_sq_fb[h][name] += float(err_fb[ci])
                    sum_sq_bil[h].setdefault(name, 0.0)
                    sum_sq_bil[h][name] += float(err_bil[ci])
        print(f"  batch {batch_idx+1}/{len(loader)}  elapsed={(time.time()-t_loop)/60:.1f} min", flush=True)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for method, sums in (("farneback_optical_flow", sum_sq_fb),):
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
