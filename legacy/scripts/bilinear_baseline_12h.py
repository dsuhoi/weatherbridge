#!/usr/bin/env python3
"""Bilinear baseline on 12h interpolation gap (test 2020).

Computes per-channel × per-hour RMSE for τ ∈ {1..11} using simple
torch.nn.functional.interpolate (bilinear) between x0 (t) and x1 (t+12).

No GPU required.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr

# ─── Config ─────────────────────────────────────────────────────────────
DATA_DIR = "/tmp/zarrs"
TEST_YEAR = 2020
PRESSURE_LEVELS = [1000, 925, 850, 700]
PL_VARS_ERA5 = ["t", "u", "v", "q", "z"]
PL_VARS_SHORT = ["temperature", "u_component_of_wind", "v_component_of_wind",
                 "specific_humidity", "geopotential"]
SURFACE_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc"]  # no tisr (analytic)
DELTA_T = 12
EVAL_HOURS = list(range(1, DELTA_T))  # 1..11
STATS_PATH = "data/json_stats.nc"
SURFACE_STATS_PATH = "data/surface_stats.json"
OUTPUT_JSON = "metrics/eval_2020/bilinear_12h.json"
SAMPLES_PER_DAY = 2  # 2 sample starts per day (t0, t0+12), each gives 11 (h=1..11) interpolation evals

# ─── Load stats ─────────────────────────────────────────────────────────
print("Loading stats...", flush=True)
ds_stats = xr.open_dataset(STATS_PATH)
pl_param_names = [f"{v.upper()}{lvl}" for v in ["T", "U", "V", "Q", "Z"] for lvl in PRESSURE_LEVELS]
pl_stats = ds_stats["climate_statistics"].sel(params=pl_param_names)
pl_mu = pl_stats.isel(stats=0).values  # (20,)
pl_sigma = pl_stats.isel(stats=1).values

with open(SURFACE_STATS_PATH) as f:
    surf_stats = json.load(f)
surf_mu = np.array([surf_stats[v]["mean"] for v in SURFACE_VARS], dtype=np.float32)
surf_sigma = np.array([surf_stats[v]["std"] for v in SURFACE_VARS], dtype=np.float32)

# ─── Load test PL + surface ─────────────────────────────────────────────
print(f"Loading {TEST_YEAR} PL + surface...", flush=True)
t0_load = time.time()
ds_pl = xr.open_zarr(f"{DATA_DIR}/zarr_{TEST_YEAR}.zarr", consolidated=True)
ds_surf = xr.open_zarr(f"{DATA_DIR}/surface_{TEST_YEAR}.zarr", consolidated=True)

# Stack PL: (T, V=5, L=4, H=181, W=360) → (T, 20, H, W)
pl_arrays = []
for v_short, v_full in zip(["t", "u", "v", "q", "z"], PL_VARS_ERA5):
    arr = ds_pl[v_full].sel(level=PRESSURE_LEVELS).values
    pl_arrays.append(arr)
pl_stack = np.concatenate([arr.astype(np.float32) for arr in pl_arrays], axis=1)  # (T, 5*4, H, W) = (T, 20, H, W)
print(f"  PL shape: {pl_stack.shape}, {time.time()-t0_load:.1f}s", flush=True)

t0_surf = time.time()
surf_arrays = [ds_surf[v].values.astype(np.float32) for v in SURFACE_VARS]
surf_stack = np.stack(surf_arrays, axis=1)  # (T, 6, H, W)
print(f"  Surface shape: {surf_stack.shape}, {time.time()-t0_surf:.1f}s", flush=True)

T_total = pl_stack.shape[0]
H, W = pl_stack.shape[2], pl_stack.shape[3]
print(f"  Total time samples: {T_total}", flush=True)

# Normalize per-channel
pl_mu_b = pl_mu.reshape(1, -1, 1, 1).astype(np.float32)
pl_sigma_b = np.clip(pl_sigma, 1e-6, None).reshape(1, -1, 1, 1).astype(np.float32)
surf_mu_b = surf_mu.reshape(1, -1, 1, 1)
surf_sigma_b = np.clip(surf_sigma, 1e-6, None).reshape(1, -1, 1, 1)

pl_norm = (pl_stack - pl_mu_b) / pl_sigma_b
surf_norm = (surf_stack - surf_mu_b) / surf_sigma_b
print(f"  Normalized.", flush=True)

# lat-cos weights (for RMSE)
lats = ds_pl.latitude.values.astype(np.float32)
lat_w = np.cos(np.deg2rad(lats))
lat_w /= lat_w.mean()
lat_w_2d = lat_w.reshape(-1, 1).astype(np.float32)  # (H, 1)

# ─── Evaluate ───────────────────────────────────────────────────────────
# For each valid t0 (with t0+12 < T_total), at each hour h ∈ {1..11}:
#  pred_bilinear(h) = (1 - h/12) * x[t0] + (h/12) * x[t0+12]
#  truth = x[t0 + h]
#  Accumulate sum_sq[h, ch] and count[h]

# Combined: PL (20) + surface (6) = 26 channels.
combined = np.concatenate([pl_norm, surf_norm], axis=1)  # (T, 26, H, W)
n_pl = pl_stack.shape[1]
print(f"  Combined channels: PL={n_pl} + surf={surf_stack.shape[1]} = {combined.shape[1]}", flush=True)

# Choose start indices: every 12 hours (no overlap of windows) for speed
start_indices = list(range(0, T_total - DELTA_T - 1, DELTA_T))[:1000]
print(f"  Sampling {len(start_indices)} 12h windows ({len(start_indices) * (DELTA_T-1)} total samples)", flush=True)

# Storage: sum_sq[h, ch] = mean squared error for hour h on channel ch (over all samples)
# count[h] = number of windows
# For per-pixel weighted RMSE: sum_sq is lat-weighted MSE per (h, ch)
sum_sq = np.zeros((DELTA_T - 1, combined.shape[1]), dtype=np.float64)
n_windows = 0

t0_eval = time.time()
for i, t0 in enumerate(start_indices):
    x0 = combined[t0]            # (C, H, W)
    xT = combined[t0 + DELTA_T]  # (C, H, W)
    for h_idx, h in enumerate(EVAL_HOURS):
        tau = h / DELTA_T
        pred = (1 - tau) * x0 + tau * xT
        truth = combined[t0 + h]
        sq = (pred - truth) ** 2                                # (C, H, W)
        weighted_sq = (sq * lat_w_2d).mean(axis=(1, 2))         # (C,)
        sum_sq[h_idx] += weighted_sq
    n_windows += 1
    if (i + 1) % 100 == 0:
        print(f"  [{i+1}/{len(start_indices)}] {time.time()-t0_eval:.1f}s", flush=True)

# Compute RMSE per (hour, channel)
rmse_grid = np.sqrt(sum_sq / n_windows)  # (11, 26)

# Channel labels
pl_chan_labels = [f"{v.upper()}{lvl}" for v in ["T", "U", "V", "Q", "Z"] for lvl in PRESSURE_LEVELS]
surf_chan_labels = SURFACE_VARS
all_labels = pl_chan_labels + surf_chan_labels

# Aggregate: per PL var (avg over 4 levels), and overall RMSE per hour
out = {
    "delta_t_hours": DELTA_T,
    "test_year": TEST_YEAR,
    "n_windows": n_windows,
    "eval_hours": EVAL_HOURS,
    "channel_labels": all_labels,
    "rmse_per_hour_per_channel": rmse_grid.tolist(),
    "rmse_overall_per_hour": rmse_grid.mean(axis=1).tolist(),
}

# Per-var aggregation
for short, vname in zip(["T", "U", "V", "Q", "Z"], PL_VARS_SHORT):
    # 4 levels: indices in combined where label.startswith(short)
    idxs = [i for i, lab in enumerate(all_labels) if lab.startswith(short) and lab not in surf_chan_labels]
    out[f"rmse_per_hour_{vname}"] = rmse_grid[:, idxs].mean(axis=1).tolist()
for sv in SURFACE_VARS:
    idx = all_labels.index(sv)
    out[f"rmse_per_hour_surface_{sv}"] = rmse_grid[:, idx].tolist()

Path("metrics/eval_2020").mkdir(parents=True, exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[ok] Saved {OUTPUT_JSON}", flush=True)

# Print summary
print("\n=== Bilinear baseline 12h on 2020 — overall + per-var RMSE per hour ===")
print(f"{'h':>3} | {'overall':>9} | {'T':>9} | {'U':>9} | {'V':>9} | {'Q':>9} | {'Z':>9}")
for h_idx, h in enumerate(EVAL_HOURS):
    print(f"{h:>3} | {out['rmse_overall_per_hour'][h_idx]:.5f} | "
          f"{out['rmse_per_hour_temperature'][h_idx]:.5f} | "
          f"{out['rmse_per_hour_u_component_of_wind'][h_idx]:.5f} | "
          f"{out['rmse_per_hour_v_component_of_wind'][h_idx]:.5f} | "
          f"{out['rmse_per_hour_specific_humidity'][h_idx]:.5f} | "
          f"{out['rmse_per_hour_geopotential'][h_idx]:.5f}")
