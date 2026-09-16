#!/usr/bin/env python3
"""Compute and save per-model × per-channel energy spectra on 2020 test.

Inputs:
  --ckpts PATH [PATH ...]   list of model checkpoints
  --names NAME [NAME ...]   display names corresponding to ckpts
  --hour H                  interpolation hour (default 3, middle)
  --n-samples N             number of test windows to average (default 64)
  --output PATH             output .npz with spectra arrays

Output schema:
  k:           (n_bins,)                 wavenumbers
  truth:       (n_channels, n_bins)      ground-truth E(k)
  bilinear:    (n_channels, n_bins)      bilinear-interp E(k)
  <name>:      (n_channels, n_bins)      model E(k)
  channel_labels: list[str]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from weather_time_interp.metrics.energy_spectra import radial_psd


def load_field_window(zarr_pl: xr.Dataset, zarr_surf: xr.Dataset,
                      t_idx: int, pl_levels: list[int]) -> np.ndarray:
    """Return concatenated PL+surface field at time index, shape (C, H, W)."""
    pl_arrays = []
    for v in ["t", "u", "v", "q", "z"]:
        pl_arrays.append(zarr_pl[v].sel(level=pl_levels).isel(time=t_idx).values.astype(np.float32))
    pl_stacked = np.concatenate(pl_arrays, axis=0)  # (5*L, H, W)
    surf = np.stack(
        [zarr_surf[v].isel(time=t_idx).values.astype(np.float32) for v in ["t2m","u10","v10","mslp","sst","tcc"]],
        axis=0,
    )
    return np.concatenate([pl_stacked, surf], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/tmp/zarrs")
    p.add_argument("--year", type=int, default=2020)
    p.add_argument("--pressure-levels", type=int, nargs="+", default=[1000, 925, 850, 700])
    p.add_argument("--hour", type=int, default=3, help="interpolation hour τ ∈ [1..delta-1]")
    p.add_argument("--delta-t", type=int, default=6)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--ckpts", nargs="*", default=[])
    p.add_argument("--names", nargs="*", default=[])
    p.add_argument("--output", default="metrics/energy_spectra_h3.npz")
    args = p.parse_args()

    if len(args.ckpts) != len(args.names):
        raise ValueError("--ckpts and --names must have same length")

    pl_path = f"{args.data_dir}/zarr_{args.year}.zarr"
    surf_path = f"{args.data_dir}/surface_{args.year}.zarr"
    ds_pl = xr.open_zarr(pl_path, consolidated=True)
    ds_surf = xr.open_zarr(surf_path, consolidated=True)
    T_total = ds_pl.sizes["time"]

    pl_chan_labels = [f"{v}{lvl}" for v in ["T","U","V","Q","Z"] for lvl in args.pressure_levels]
    surf_chan_labels = ["t2m", "u10", "v10", "mslp", "sst", "tcc"]
    all_labels = pl_chan_labels + surf_chan_labels
    n_channels = len(all_labels)

    # Pick samples_per_date sample positions evenly spread through the year
    start_idxs = np.linspace(0, T_total - args.delta_t - 1, args.n_samples).astype(int)

    print(f"[psd] computing spectra for hour={args.hour}, n_samples={args.n_samples}, "
          f"n_channels={n_channels}", flush=True)
    print(f"[psd] channels: {all_labels}", flush=True)

    # Load truth (t0+hour) and endpoints (t0, t0+delta) for bilinear baseline
    truth_specs = []
    bilin_specs = []

    # Lat-cos weights (for FFT input pre-weighting)
    lats = ds_pl.latitude.values.astype(np.float32)
    lat_w = torch.tensor(np.cos(np.deg2rad(lats)), dtype=torch.float32)
    lat_w /= lat_w.mean()

    for i, t0 in enumerate(start_idxs):
        x_true = load_field_window(ds_pl, ds_surf, t0 + args.hour, args.pressure_levels)
        x0 = load_field_window(ds_pl, ds_surf, t0, args.pressure_levels)
        xT = load_field_window(ds_pl, ds_surf, t0 + args.delta_t, args.pressure_levels)
        tau = args.hour / float(args.delta_t)
        x_bilin = (1 - tau) * x0 + tau * xT

        # PSD per channel
        x_true_t = torch.from_numpy(x_true)
        x_bilin_t = torch.from_numpy(x_bilin)

        for ch_idx in range(n_channels):
            if i == 0:
                truth_specs.append([])
                bilin_specs.append([])
            k, e_t = radial_psd(x_true_t[ch_idx], lat_weights=lat_w)
            _, e_b = radial_psd(x_bilin_t[ch_idx], lat_weights=lat_w)
            truth_specs[ch_idx].append(e_t.flatten())
            bilin_specs[ch_idx].append(e_b.flatten())

        if (i + 1) % 8 == 0:
            print(f"[psd] processed {i+1}/{args.n_samples}", flush=True)

    # Aggregate (mean over samples)
    n_bins = len(k)
    truth_arr = np.zeros((n_channels, n_bins))
    bilin_arr = np.zeros((n_channels, n_bins))
    for ch_idx in range(n_channels):
        truth_arr[ch_idx] = np.mean(np.stack(truth_specs[ch_idx]), axis=0)
        bilin_arr[ch_idx] = np.mean(np.stack(bilin_specs[ch_idx]), axis=0)

    output = {
        "k": k,
        "truth": truth_arr,
        "bilinear": bilin_arr,
        "channel_labels": all_labels,
        "hour": args.hour,
        "delta_t": args.delta_t,
        "n_samples": args.n_samples,
    }

    # Model spectra (lazy: only if ckpts provided)
    if args.ckpts:
        from trainer_weather_hermite import WeatherHermiteLightningModule
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for ckpt_path, name in zip(args.ckpts, args.names):
            if not Path(ckpt_path).exists():
                print(f"[psd] WARN: {ckpt_path} missing — skip {name}", flush=True)
                continue
            print(f"[psd] loading {name} from {ckpt_path}", flush=True)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            from evaluate_baselines import load_weather_hermite_model
            channel_groups = {"temperature": [0,1,2,3], "u_component_of_wind": [4,5,6,7],
                              "v_component_of_wind": [8,9,10,11], "specific_humidity": [12,13,14,15],
                              "geopotential": [16,17,18,19]}
            model = load_weather_hermite_model(Path(ckpt_path), device, channel_groups).eval()

            # We need normalized inputs to feed model; load stats
            stats = xr.open_dataset("data/json_stats.nc")
            pl_param_names = pl_chan_labels
            pl_stats_arr = stats["climate_statistics"].sel(params=pl_param_names).values  # (2, 20)
            pl_mu = pl_stats_arr[0]
            pl_sigma = np.clip(pl_stats_arr[1], 1e-6, None)
            with open("data/surface_stats.json") as f:
                surf_stats = json.load(f)
            surf_mu = np.array([surf_stats[v]["mean"] for v in surf_chan_labels], dtype=np.float32)
            surf_sigma = np.clip(np.array([surf_stats[v]["std"] for v in surf_chan_labels], dtype=np.float32), 1e-6, None)
            mu = np.concatenate([pl_mu, surf_mu]).reshape(-1, 1, 1)
            sigma = np.concatenate([pl_sigma, surf_sigma]).reshape(-1, 1, 1)

            # Static features
            static = torch.load("data/static_features.pt", weights_only=False)
            if isinstance(static, torch.Tensor):
                static_t = static.float().to(device)
            else:
                static_t = torch.tensor(static).float().to(device)

            model_specs = [[] for _ in range(n_channels)]
            with torch.no_grad():
                for i, t0 in enumerate(start_idxs):
                    x0 = load_field_window(ds_pl, ds_surf, t0, args.pressure_levels)
                    xT = load_field_window(ds_pl, ds_surf, t0 + args.delta_t, args.pressure_levels)
                    x0_norm = ((x0 - mu) / sigma).astype(np.float32)
                    xT_norm = ((xT - mu) / sigma).astype(np.float32)
                    x0_t = torch.from_numpy(x0_norm).unsqueeze(0).to(device)
                    xT_t = torch.from_numpy(xT_norm).unsqueeze(0).to(device)
                    tau_t = torch.tensor([args.hour / float(args.delta_t)], dtype=torch.float32, device=device)
                    cond_t = torch.tensor([float(args.delta_t)], dtype=torch.float32, device=device)
                    x_hat, _ = model(x0_t, xT_t, tau_t, cond_t, static=static_t)
                    # De-normalize
                    x_hat_denorm = (x_hat[0].cpu().numpy() * sigma + mu).astype(np.float32)
                    x_hat_t = torch.from_numpy(x_hat_denorm)
                    for ch_idx in range(n_channels):
                        _, e = radial_psd(x_hat_t[ch_idx], lat_weights=lat_w)
                        model_specs[ch_idx].append(e.flatten())
                    if (i + 1) % 8 == 0:
                        print(f"[psd][{name}] {i+1}/{args.n_samples}", flush=True)
            model_arr = np.zeros((n_channels, n_bins))
            for ch_idx in range(n_channels):
                model_arr[ch_idx] = np.mean(np.stack(model_specs[ch_idx]), axis=0)
            output[name] = model_arr

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output)
    print(f"[psd] saved {args.output} (keys: {list(output.keys())})", flush=True)


if __name__ == "__main__":
    main()
