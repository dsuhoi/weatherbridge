#!/usr/bin/env python3
"""Energy spectra evaluation: radial power spectrum E(k) per channel for
each method (ML models + numerical baselines + ground truth).

Computes spatial 2D FFT of predictions and targets, bins by radial wavenumber
|k|, averages over the test set, and writes:
  metrics/energy_spectra_0p5_2020/<method>.npz   (per-channel binned spectra)
  paper/energy_spectra/<channel>.png             (E(k) vs k log-log, selected channels)

This is the "spectral bias" diagnostic recommended for AAAI rebuttal:
deterministic regressors typically truncate the high-k tail relative to GT.
Our spectral consistency loss is designed to mitigate this; the eval here
makes the effect explicit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import (
    WeatherHermiteLightningModule,
    ERA5WeatherHermiteDataset,
)

PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]
PLOT_CHANNELS = ["U850", "V850", "T850", "Q850", "Z500", "t2m", "u10", "tcwv"]


def load_model(ckpt_path, device, channel_groups, envs=""):
    import os as _os
    for kv in envs.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            _os.environ[k] = v
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")
    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        hparams["block_out_channels"] = boc if len(boc) >= 3 else (128, 256, 512)
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        hparams["layers_per_block"] = lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**hparams)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, mt


def radial_bin_indices(H, W, n_bins, device):
    """Precompute per-bin index masks for radial averaging of rFFT2 output.

    Returns:
      k_centers: (n_bins,) wavenumber bin centers in cycles/grid (0..0.5).
      bin_idx:   (H, W//2+1) integer mask of bin assignment, -1 = drop.
    """
    ky = torch.fft.fftfreq(H, device=device).abs()
    kx = torch.fft.rfftfreq(W, device=device).abs()
    k_mag = (ky[:, None] ** 2 + kx[None, :] ** 2).sqrt()
    k_max = k_mag.max().item()
    edges = torch.linspace(0, k_max, n_bins + 1, device=device)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = torch.bucketize(k_mag, edges) - 1
    bin_idx = bin_idx.clamp(0, n_bins - 1)
    return centers.cpu().numpy(), bin_idx


def accumulate_spectrum(field, bin_idx, n_bins):
    """field: (B, C, H, W) — accumulate radially-binned |FFT|^2 per channel.

    Returns (C, n_bins) sum-of-squares and (n_bins,) bin counts.
    """
    spec = torch.fft.rfft2(field.float(), dim=(-2, -1)).abs()  # (B, C, H, W//2+1)
    power = spec.pow(2)
    B, C, H, Wh = power.shape
    flat = power.view(B, C, -1)              # (B, C, H*Wh)
    bidx = bin_idx.view(-1)                   # (H*Wh,)
    out = torch.zeros((C, n_bins), device=field.device, dtype=torch.float64)
    counts = torch.zeros(n_bins, device=field.device, dtype=torch.float64)
    out.index_add_(1, bidx, flat.sum(dim=0).double())   # sum over batch
    counts.index_add_(0, bidx, torch.ones_like(bidx, dtype=torch.float64))
    return out, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--method-name", required=True,
                    help="display name, e.g. 'dcae_skip_AC' or 'ground_truth'")
    ap.add_argument("--envs", default="LAT_CROP=-8")
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="metrics/energy_spectra_0p5_2020")
    ap.add_argument("--n-bins", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--include-ground-truth", action="store_true",
                    help="also save E(k) of the target field (run once per dataset)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  method: {args.method_name}")
    t_global = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[args.test_year], max_tau_hours=6,
        samples_per_date=args.samples_per_date, train=False,
        eval_hours=list(range(7)), static_path=args.static_path,
        stats_path=args.stats_path, surface_stats_path=args.surface_stats_path,
    )
    import datetime as _dt
    K = max(1, int(args.eval_days_per_month))
    day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22],
                 5: [1, 7, 14, 21, 28]}.get(
        K, sorted({1 + i * (30 // K) for i in range(K)}))
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
    ds_base.index = filt
    print(f"  windows={len(filt) // 5}")

    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    C = len(channel_names)

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    sample0 = next(iter(loader))
    H, W = sample0["x0"].shape[-2:]
    print(f"  HxW = {H}x{W},  C = {C}")
    k_centers, bin_idx = radial_bin_indices(H, W, args.n_bins, device)
    bin_idx = bin_idx.to(device)

    model, mt = load_model(args.ckpt, device, channel_groups, args.envs)
    print(f"  loaded {mt}, params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    pred_pow = torch.zeros((C, args.n_bins), device=device, dtype=torch.float64)
    gt_pow = torch.zeros((C, args.n_bins), device=device, dtype=torch.float64)
    counts = torch.zeros(args.n_bins, device=device, dtype=torch.float64)
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_all = batch["tau"].to(device, non_blocking=True)
            target_all = batch["target"].to(device, non_blocking=True)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)

            B, nH = x0.size(0), tau_all.size(1)
            cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)
            for h_idx in range(nH):
                tau_h = tau_all[:, h_idx, 0]
                target_h = target_all[:, h_idx]
                out = model(x0, xT, tau_h, cond, static=static)
                pred = out[0] if isinstance(out, tuple) else out
                p_pow, c = accumulate_spectrum(pred, bin_idx, args.n_bins)
                pred_pow += p_pow
                counts += c
                if args.include_ground_truth:
                    g_pow, _ = accumulate_spectrum(target_h, bin_idx, args.n_bins)
                    gt_pow += g_pow
                n_samples += B
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)}", flush=True)

    # Average: divide by sample count and bin pixel count
    counts_per_sample = counts.clamp(min=1.0)
    pred_Ek = (pred_pow / n_samples / counts_per_sample).cpu().numpy()
    gt_Ek = (gt_pow / n_samples / counts_per_sample).cpu().numpy() \
        if args.include_ground_truth else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method_name}.npz"
    payload = dict(k_centers=k_centers, pred_Ek=pred_Ek,
                   channel_names=np.array(channel_names),
                   n_samples=n_samples, H=H, W=W)
    if gt_Ek is not None:
        payload["gt_Ek"] = gt_Ek
    np.savez(out_path, **payload)
    print(f"\nwrote {out_path}")
    print(f"=== DONE in {(time.time() - t_global) / 60:.1f} min ===")


if __name__ == "__main__":
    main()
