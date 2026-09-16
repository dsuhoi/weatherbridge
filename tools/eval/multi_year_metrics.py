#!/usr/bin/env python3
"""Multi-year eval (2020+2021) computing RMSE / ACC / temporal correlation
per channel in absolute physical units, plus baselines (bilinear, Hermite-adv).

Emits:
  metrics/multi_year_2020_2021/{model}.json
  metrics/multi_year_2020_2021/REPORT.md   ← summary Markdown tables

Metrics (all lat-weighted, all in physical units after denormalization):
  RMSE  — sqrt(mean((pred - target)^2))
  ACC   — anomaly correlation (pred-clim, tgt-clim) per timestep, averaged
  TC    — pixel-wise temporal correlation Pearson(pred_t, target_t)
          aggregated as lat-weighted mean over (lat, lon).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset
from evaluate_baselines import interpolate_time_with_f_interpolate
from tools.eval.compute_acc import ClimatologyLookup, CLIM_PL_NAMES
from tools.baselines.numerical_baselines import (
    hermite_advection_interp, build_uv_assignment, expand_uv_to_channels,
)

CHANNEL_UNITS = {
    "T": "K", "U": "m/s", "V": "m/s", "Q": "kg/kg", "Z": "m²/s²",
    "t2m": "K", "u10": "m/s", "v10": "m/s",
    "mslp": "Pa", "sst": "K", "tcc": "-", "tcwv": "kg/m²",
}


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
    ap.add_argument("--test-years", type=int, nargs="+", default=[2020, 2021])
    ap.add_argument("--climatology", default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr")
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="metrics/multi_year_2020_2021")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--models", required=True,
                    help="Comma list NAME:CKPT[:ENVS]")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, years: {args.test_years}")
    t_global = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=args.test_years, max_tau_hours=6,
        samples_per_date=args.samples_per_date, train=False,
        eval_hours=list(range(7)), static_path=args.static_path,
        stats_path=args.stats_path, surface_stats_path=args.surface_stats_path,
    )
    # Economy days filter (same as batch_eval_memmap)
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
    print(f"  economy filter: {len(filt)} entries (~{len(filt)//5} windows per year)")

    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    n_pl = len(ds_base.channel_names)
    C = len(channel_names)
    print(f"  channels ({C}): {channel_names}")

    # Climatology for ACC
    _probe = __import__("xarray").open_zarr(str(args.climatology), consolidated=True)
    _clim_vars = set(_probe.data_vars); _probe.close()
    skip = {"tisr"}
    for c in channel_names:
        is_pl = len(c) > 1 and c[0] in CLIM_PL_NAMES and c[1:].isdigit()
        if is_pl:
            if CLIM_PL_NAMES[c[0]] not in _clim_vars:
                skip.add(c)
        else:
            if c not in _clim_vars:
                skip.add(c)
    acc_indices = [i for i, c in enumerate(channel_names) if c not in skip]
    acc_channel_names = [channel_names[i] for i in acc_indices]
    clim = ClimatologyLookup(Path(args.climatology), acc_channel_names, device)
    print(f"  ACC channels: {len(acc_channel_names)} (skip {sorted(skip)})")

    # Stats for denormalization
    mu = torch.cat([ds_base.mu, ds_base.surface_mu]).to(device).view(1, -1, 1, 1)
    sigma = torch.cat([ds_base.sigma, ds_base.surface_sigma]).to(device).view(1, -1, 1, 1)

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    print(f"  windows: {len(test_wrapped)}")
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    H = list(ds_base.memmaps.values())[0].shape[-2]
    W = list(ds_base.memmaps.values())[0].shape[-1]
    lat = np.linspace(89.75, -89.75, H, dtype=np.float32) if H == 360 else np.linspace(90.0, -90.0, H, dtype=np.float32)
    w_lat = torch.from_numpy(np.cos(np.deg2rad(lat))).to(device, torch.float32) / np.cos(np.deg2rad(lat)).sum()
    w_lat = w_lat.view(1, 1, -1, 1)

    uv_map = build_uv_assignment(channel_names)
    grouped = getattr(test_wrapped, "_grouped_indices", None)

    # Methods (ML models passed via --models + 2 numeric baselines)
    models_list = []
    for entry in args.models.split(","):
        parts = entry.split(":")
        if len(parts) >= 2:
            envs = ":".join(parts[2:]) if len(parts) > 2 else ""
            models_list.append((parts[0], parts[1], envs))
    methods = [m[0] for m in models_list] + ["bilinear", "hermite_advection"]

    # Accumulators per (method, channel):
    #   sum_sq_phys (for RMSE in physical units)
    #   sum_sq_norm (for completeness, normalized RMSE)
    #   ACC sums (sxy, sxx, syy) — only for channels in clim
    #   TC sums per (c, h, w): sx, sy, sxy, sx2, sy2, n   (heavy — H×W per channel)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    def new_accums():
        return {
            "sum_sq_phys": {c: torch.zeros(C, device=device, dtype=torch.float64) for c in [0]},  # actually one tensor per method, indexed by channel
            "n_samples":   0,
            "acc_sxy":     torch.zeros(len(acc_indices), device=device, dtype=torch.float64),
            "acc_sxx":     torch.zeros(len(acc_indices), device=device, dtype=torch.float64),
            "acc_syy":     torch.zeros(len(acc_indices), device=device, dtype=torch.float64),
            "acc_n":       0,
            "tc_sx":       torch.zeros(C, H, W, device=device, dtype=torch.float64),
            "tc_sy":       torch.zeros(C, H, W, device=device, dtype=torch.float64),
            "tc_sxy":      torch.zeros(C, H, W, device=device, dtype=torch.float64),
            "tc_sx2":      torch.zeros(C, H, W, device=device, dtype=torch.float64),
            "tc_sy2":      torch.zeros(C, H, W, device=device, dtype=torch.float64),
            "tc_n":        0,
        }
    # NOTE: tc_sx etc. are HUGE (C×H×W×8B = 27×360×720×8B = 56 MB per accumulator × 5 = 280 MB × 6 methods = 1.7 GB).
    #       Stored as fp64 for accuracy. Should fit on A100 80GB easily.
    accums = {m: new_accums() for m in methods}
    acc_idx_t = torch.tensor(acc_indices, device=device, dtype=torch.long)

    # Per-channel running sum for RMSE in physical units: separate accumulator (just one float per ch per method)
    rmse_phys_sum = {m: torch.zeros(C, device=device, dtype=torch.float64) for m in methods}
    rmse_norm_sum = {m: torch.zeros(C, device=device, dtype=torch.float64) for m in methods}
    n_samples = 0

    # Load ML models (one at a time would save GPU mem, but we need parallel eval per batch).
    # → keep loaded in CPU dict, move to GPU per-batch — too slow.
    # Compromise: keep all on GPU. Each ML model ~50-200 MB. Bilinear/Hermite are free.
    loaded = {}
    for name, ckpt, envs in models_list:
        if not Path(ckpt).exists():
            print(f"  [miss] {ckpt}"); continue
        m, mt = load_model(ckpt, device, channel_groups, envs)
        loaded[name] = (m, mt)
        print(f"  loaded {name}: type={mt}, params={sum(p.numel() for p in m.parameters())/1e6:.1f}M")

    print(f"\nsetup done in {time.time()-t_global:.1f}s\n")
    t_loop = time.time()
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
            B, nH = x0.size(0), tau_all.size(1)
            cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)

            x0_phys = x0 * sigma + mu; xT_phys = xT * sigma + mu
            u0p, v0p = expand_uv_to_channels(x0_phys, uv_map)
            uTp, vTp = expand_uv_to_channels(xT_phys, uv_map)

            for h_idx in range(nH):
                tau_h = tau_all[:, h_idx, 0]
                target_h = target_all[:, h_idx]
                target_h_phys = target_h * sigma + mu
                hours = tau_hour_all[:, h_idx, 0].tolist()

                preds = {}
                preds["bilinear"] = interpolate_time_with_f_interpolate(x0, xT, tau_h, mode="bilinear")
                preds["hermite_advection"] = hermite_advection_interp(x0, xT, u0p, v0p, uTp, vTp, tau_h, dt_hours=6.0)
                for name, (m, mt) in loaded.items():
                    out = m(x0, xT, tau_h, cond, static=static)
                    preds[name] = out[0] if isinstance(out, tuple) else out

                for method in methods:
                    if method not in preds:
                        continue
                    p_norm = preds[method]
                    p_phys = p_norm * sigma + mu
                    err_phys = (p_phys - target_h_phys) ** 2 * w_lat
                    err_norm = (p_norm - target_h) ** 2 * w_lat
                    # Per-channel sum across batch
                    rmse_phys_sum[method] += err_phys.sum(dim=(0, 2, 3)).double()
                    rmse_norm_sum[method] += err_norm.sum(dim=(0, 2, 3)).double()
                    # TC accumulators (sum over batch)
                    p_d = p_phys.double()
                    t_d = target_h_phys.double()
                    accums[method]["tc_sx"]  += p_d.sum(0)
                    accums[method]["tc_sy"]  += t_d.sum(0)
                    accums[method]["tc_sxy"] += (p_d * t_d).sum(0)
                    accums[method]["tc_sx2"] += (p_d * p_d).sum(0)
                    accums[method]["tc_sy2"] += (t_d * t_d).sum(0)
                    accums[method]["tc_n"]  += B
                    # ACC anomaly (per sample, restricted channels)
                    p_phys_a = p_phys.index_select(1, acc_idx_t)
                    t_phys_a = target_h_phys.index_select(1, acc_idx_t)
                    for i in range(B):
                        h_ = int(hours[i])
                        if h_ < 0 or h_ > 6:
                            continue
                        wi = batch_idx * loader.batch_size + i
                        if wi >= len(test_wrapped):
                            break
                        base_i = grouped[wi][0] if grouped is not None else wi
                        year, t0, _, _ = ds_base.index[base_i]
                        ts = ds_base.time_starts[int(year)] + timedelta(hours=int(t0 + h_))
                        doy = ts.timetuple().tm_yday
                        hf = ts.hour + ts.minute / 60.0
                        clim_t = clim.lookup(
                            doy,
                            hf,
                            ts.year,
                        ).unsqueeze(0).to(device)
                        t_an = t_phys_a[i:i+1] - clim_t
                        p_an = p_phys_a[i:i+1] - clim_t
                        accums[method]["acc_sxy"] += (w_lat * p_an * t_an).sum(dim=(0, 2, 3)).double()
                        accums[method]["acc_sxx"] += (w_lat * p_an * p_an).sum(dim=(0, 2, 3)).double()
                        accums[method]["acc_syy"] += (w_lat * t_an * t_an).sum(dim=(0, 2, 3)).double()
                        accums[method]["acc_n"]  += 1
                n_samples += B  # interior-hour batches per method

            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)}  elapsed={(time.time()-t_loop)/60:.1f} min", flush=True)

    # Finalize metrics per method
    final = {}
    for method in methods:
        rmse_phys = (rmse_phys_sum[method] / n_samples).sqrt()  # per-channel physical RMSE
        rmse_norm = (rmse_norm_sum[method] / n_samples).sqrt()
        # TC per pixel, then lat-weighted mean over (H, W)
        sx = accums[method]["tc_sx"]; sy = accums[method]["tc_sy"]
        sxy = accums[method]["tc_sxy"]; sx2 = accums[method]["tc_sx2"]; sy2 = accums[method]["tc_sy2"]
        n_tc = accums[method]["tc_n"]
        num = n_tc * sxy - sx * sy
        denom = ((n_tc * sx2 - sx * sx) * (n_tc * sy2 - sy * sy)).clamp_min(1e-12).sqrt()
        tc_pixel = num / denom                                 # (C, H, W)
        tc_perch = (tc_pixel * w_lat[0]).sum(dim=(1, 2))      # (C,)
        # ACC per channel (restricted to acc_channel_names)
        sxy_a = accums[method]["acc_sxy"]; sxx_a = accums[method]["acc_sxx"]; syy_a = accums[method]["acc_syy"]
        acc_perch_subset = sxy_a / (sxx_a * syy_a + 1e-12).sqrt()
        acc_perch = {acc_channel_names[i]: float(acc_perch_subset[i].item()) for i in range(len(acc_channel_names))}
        final[method] = {
            "rmse_phys_per_channel":  {channel_names[i]: float(rmse_phys[i].item()) for i in range(C)},
            "rmse_norm_per_channel":  {channel_names[i]: float(rmse_norm[i].item()) for i in range(C)},
            "tc_per_channel":         {channel_names[i]: float(tc_perch[i].item()) for i in range(C)},
            "acc_per_channel":        acc_perch,
            "n_samples": n_samples,
        }
        # Save JSON
        with open(out_dir / f"{method}.json", "w") as f:
            json.dump(final[method], f, indent=2)
        print(f"saved {out_dir / f'{method}.json'}")

    # Build Markdown report
    md = ["# Multi-year metrics on test set " + str(args.test_years),
          "",
          "Lat-weighted means. RMSE in physical units. ACC vs WB2 climatology 1990-2019.",
          "TC = pixel-wise temporal Pearson(pred_t, target_t) averaged with cos(lat) weight.",
          f"Test windows: {n_samples // 5} (sum over interior hours h=1..5).",
          ""]

    def unit_of(cn):
        if cn in CHANNEL_UNITS: return CHANNEL_UNITS[cn]
        return CHANNEL_UNITS.get(cn[0], "?")

    def build_table(metric_key, methods, channel_names, title, precision=4):
        md.append(f"## {title}\n")
        head = ["Channel", "Unit"] + methods
        md.append("| " + " | ".join(head) + " |")
        md.append("|" + "|".join(["---"] * len(head)) + "|")
        for cn in channel_names:
            row = [cn, unit_of(cn)]
            for mth in methods:
                v = final[mth][metric_key].get(cn)
                row.append("---" if v is None else f"{v:.{precision}f}")
            md.append("| " + " | ".join(row) + " |")
        md.append("")

    build_table("rmse_phys_per_channel", methods, channel_names, "RMSE (physical units, per channel)", precision=4)
    build_table("acc_per_channel",       methods, acc_channel_names,
                "ACC (anomaly correlation vs 1990-2019 climatology, per channel)", precision=4)
    build_table("tc_per_channel",        methods, channel_names,
                "Temporal correlation (pixel-wise Pearson, lat-weighted mean)", precision=4)

    with open(out_dir / "REPORT.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"saved {out_dir / 'REPORT.md'}")
    print(f"\n=== DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
