#!/usr/bin/env python3
"""Quick ablation: zero out specified input channels of DC-AE Skip 6yr and
measure RMSE degradation per channel. Tests whether sst/tcc/tcwv are critical
for predictions on the other 24 channels (HRES/GFS missing-vars compatibility).

Outputs Markdown report comparing zero-filled vs full-input baseline.
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
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


def load_model(ckpt_path, device, channel_groups, envs=""):
    import os as _os
    for kv in envs.split():
        if "=" in kv:
            k, v = kv.split("=", 1); _os.environ[k] = v
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--envs", default="LAT_CROP=-8")
    ap.add_argument("--zero-channels", nargs="+", default=["sst", "tcc", "tcwv"])
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out", default="metrics/zero_channel_ablation/REPORT.md")
    ap.add_argument("--baseline-json", default="metrics/eval_6h_2020_paper_leaderboard/dcae_skip_0p5_6yr_pad.json",
                    help="JSON with full-input RMSE for the same model (for delta comparison)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  zero-channels: {args.zero_channels}")
    t_global = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[args.test_year], max_tau_hours=6,
        samples_per_date=args.samples_per_date, train=False,
        eval_hours=list(range(7)), static_path=args.static_path,
        stats_path=args.stats_path, surface_stats_path=args.surface_stats_path,
    )
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
    print(f"  windows={len(filt)//5}")

    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    C = len(channel_names)
    zero_idx = [channel_names.index(c) for c in args.zero_channels if c in channel_names]
    print(f"  zero indices: {zero_idx} for {args.zero_channels}")

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    H = list(ds_base.memmaps.values())[0].shape[-2]
    lat = np.linspace(89.75, -89.75, H, dtype=np.float32) if H == 360 else np.linspace(90.0, -90.0, H, dtype=np.float32)
    w_lat = torch.from_numpy(np.cos(np.deg2rad(lat))).to(device, torch.float32) / np.cos(np.deg2rad(lat)).sum()
    w_lat = w_lat.view(1, 1, -1, 1)

    mu = torch.cat([ds_base.mu, ds_base.surface_mu]).to(device).view(1, -1, 1, 1)
    sigma = torch.cat([ds_base.sigma, ds_base.surface_sigma]).to(device).view(1, -1, 1, 1)

    model, mt = load_model(args.ckpt, device, channel_groups, args.envs)
    print(f"  loaded {mt}, params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    sum_sq_norm = torch.zeros(C, device=device, dtype=torch.float64)
    sum_sq_phys = torch.zeros(C, device=device, dtype=torch.float64)
    n_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_all = batch["tau"].to(device, non_blocking=True)
            tau_hour_all = batch["tau_hour"].long()
            target_all = batch["target"].to(device, non_blocking=True)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)

            # Zero the specified input channels in both anchors
            x0z = x0.clone(); xTz = xT.clone()
            for ci in zero_idx:
                x0z[:, ci] = 0.0
                xTz[:, ci] = 0.0

            B, nH = x0.size(0), tau_all.size(1)
            cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)
            for h_idx in range(nH):
                tau_h = tau_all[:, h_idx, 0]
                target_h = target_all[:, h_idx]
                out = model(x0z, xTz, tau_h, cond, static=static)
                pred = out[0] if isinstance(out, tuple) else out
                err_norm = (pred - target_h) ** 2 * w_lat
                pred_phys = pred * sigma + mu
                tgt_phys = target_h * sigma + mu
                err_phys = (pred_phys - tgt_phys) ** 2 * w_lat
                sum_sq_norm += err_norm.sum(dim=(0, 2, 3)).double()
                sum_sq_phys += err_phys.sum(dim=(0, 2, 3)).double()
                n_samples += B
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)}", flush=True)

    rmse_norm = (sum_sq_norm / n_samples).sqrt().cpu().numpy()
    rmse_phys = (sum_sq_phys / n_samples).sqrt().cpu().numpy()

    # Load full-input baseline (computed earlier on the same model)
    baseline = None
    if Path(args.baseline_json).exists():
        d = json.loads(Path(args.baseline_json).read_text())
        baseline = {}
        for h in d['per_hour']:
            for cn, v in d['per_hour'][h]['model'].items():
                baseline.setdefault(cn, []).append(float(v))
        baseline = {k: float(np.mean(v)) for k, v in baseline.items()}   # mean over h=1..5
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        f"# Zero-channel ablation report",
        "",
        f"Model: `{args.ckpt}`",
        f"Zeroed input channels: **{args.zero_channels}**",
        f"Test year: {args.test_year}, windows: {n_samples // 5}",
        "",
        "## Per-channel RMSE (normalised, mean over h=1..5)",
        "| Channel | RMSE w/ zeroed input | Baseline (full input) | Δ vs baseline |",
        "|---|---|---|---|",
    ]
    for i, cn in enumerate(channel_names):
        v_zero = float(rmse_norm[i])
        base_str = "—"
        delta_str = "—"
        if baseline and f"rmse_{cn}" in baseline:
            b = baseline[f"rmse_{cn}"]
            base_str = f"{b:.4f}"
            delta_pct = (v_zero - b) / b * 100 if b > 0 else float("nan")
            delta_str = f"{delta_pct:+.1f}%"
        zeroed_mark = " 🚫" if cn in args.zero_channels else ""
        md.append(f"| {cn}{zeroed_mark} | {v_zero:.4f} | {base_str} | {delta_str} |")
    md.append("")
    md.append("## Per-channel RMSE (physical units)")
    md.append("| Channel | RMSE_phys |")
    md.append("|---|---|")
    for i, cn in enumerate(channel_names):
        md.append(f"| {cn} | {rmse_phys[i]:.4f} |")
    out_path.write_text("\n".join(md) + "\n")
    print(f"\nwrote {out_path}")
    print(f"=== DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
