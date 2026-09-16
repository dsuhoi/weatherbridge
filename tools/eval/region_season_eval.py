#!/usr/bin/env python3
"""Per-region × per-season RMSE breakdown for all 4 ML models @ 0.5° 6yr.

Regions (lat bands):
  - Tropics      30°S - 30°N
  - Midlatitudes 30° - 60° (both hemispheres)
  - Polar        60° - 90° (both hemispheres)
Seasons (DJF / MAM / JJA / SON).

Emits metrics/region_season_2020/{model}.json with per-region per-season per-hour mean RMSE.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset
from evaluate_baselines import interpolate_time_with_f_interpolate
# Reuse the surgical loader in batch_eval_memmap which dispatches between
# WeatherHermite (4-stage DC-AE NoSkip / Skip / S-DYff / ModAFNO / FuXi) and
# the asymmetric 27→24 ATM-VFI v2 PixelAttentionVFI loader.
from tools.eval.batch_eval_memmap import load_model_safe as _load_model_safe


def load_model(ckpt_path, device, channel_groups):
    """Delegate to the canonical batch_eval_memmap loader.

    Returns ``(model, model_type_str)``. ``model_type_str`` is the original
    ``hparams['model_type']`` for WeatherHermite ckpts, or
    ``'atm_vfi_pixel_attn'`` for ATM-VFI v2.
    """
    return _load_model_safe(ckpt_path, device, channel_groups)


REGIONS = {
    "tropics":      (-30.0,  30.0),
    "midlatitudes_N": (30.0,  60.0),
    "midlatitudes_S": (-60.0, -30.0),
    "polar_N":        (60.0,  90.0),
    "polar_S":        (-90.0, -60.0),
}
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def build_region_masks(H: int, device, dtype):
    if H == 360:
        lat = torch.linspace(89.75, -89.75, H, device=device, dtype=dtype)
    else:
        lat = torch.linspace(90.0, -90.0, H, device=device, dtype=dtype)
    masks = {}
    for name, (lat_min, lat_max) in REGIONS.items():
        m = ((lat >= lat_min) & (lat <= lat_max)).to(dtype)  # (H,)
        masks[name] = m.view(1, 1, -1, 1)                    # (1, 1, H, 1)
    # Cosine-of-latitude weight for area-correctness
    cos_w = torch.cos(torch.deg2rad(lat)).to(dtype).view(1, 1, -1, 1)
    return masks, cos_w


def month_from_doy(year: int, t0: int, h: int) -> int:
    """Compute calendar month from year start + (t0 + h) hours."""
    from datetime import date, timedelta
    d = date(year, 1, 1) + timedelta(days=(t0 + h) // 24)
    return d.month


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--models", required=True,
                    help="Comma-separated NAME:CKPT[:ENVS]")
    ap.add_argument("--out-dir", default="metrics/region_season_2020")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument(
        "--keep-n-channels", type=int, default=0,
        help="If >0, slice x0/xT/target to first N channels before forward + "
             "accumulation. Required for ATM-VFI v2 (N=24, drops sst/tcc/tcwv) "
             "and any KEEP_24CH ckpts.",
    )
    args = ap.parse_args()
    keep_n = int(args.keep_n_channels) if int(args.keep_n_channels) > 0 else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

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
    channel_groups = ds_base.channel_groups
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    if keep_n is not None and keep_n < len(channel_names):
        channel_names = channel_names[:keep_n]
        print(f"  keep_n_channels={keep_n}: channel_names={channel_names}")

    test_wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    print(f"  windows: {len(test_wrapped)}")
    loader = DataLoader(test_wrapped, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True,
                        persistent_workers=args.num_workers > 0)

    H = list(ds_base.memmaps.values())[0].shape[-2]
    region_masks, cos_w = build_region_masks(H, device, torch.float32)
    region_normalizers = {}  # weight sum per region (for normalizing weighted mean)
    for rname, m in region_masks.items():
        w_norm = (m * cos_w).sum()
        region_normalizers[rname] = float(w_norm.item())
    print(f"  region cos-weight sums: " + ", ".join(f"{k}={v:.3f}" for k,v in region_normalizers.items()))

    # Models list
    models_list = []
    for entry in args.models.split(","):
        parts = entry.split(":")
        if len(parts) >= 2:
            envs = ":".join(parts[2:]) if len(parts) > 2 else ""
            models_list.append((parts[0], parts[1], envs))

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Group wrapped indices to base entries to extract calendar month
    grouped = getattr(test_wrapped, "_grouped_indices", None)

    for name, ckpt, envs in models_list:
        if not Path(ckpt).exists():
            print(f"  [skip] {ckpt}")
            continue
        import os as _os
        for kv in envs.split():
            if "=" in kv:
                k, v = kv.split("=", 1)
                _os.environ[k] = v
        t0_m = time.time()
        model, mt = load_model(ckpt, device, channel_groups)
        is_atmvfi = (mt == "atm_vfi_pixel_attn")
        print(f"  {name}: type={mt} (atm_vfi={is_atmvfi})")

        # Accumulators: per (region, season, hour) → per-channel sum_sq + count
        sum_sq = {r: {s: {h: {} for h in range(7)} for s in SEASONS} for r in REGIONS}
        bil_sum_sq = {r: {s: {h: {} for h in range(7)} for s in SEASONS} for r in REGIONS}
        counts = {r: {s: {h: 0 for h in range(7)} for s in SEASONS} for r in REGIONS}

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
                # Slice to keep_n channels (ATM-VFI v2 / KEEP_24CH ckpts).
                if keep_n is not None and x0.size(1) > keep_n:
                    x0 = x0[:, :keep_n].contiguous()
                    xT = xT[:, :keep_n].contiguous()
                    target_all = target_all[:, :, :keep_n].contiguous()
                B, nH = x0.size(0), tau_all.size(1)
                cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)

                for h_idx in range(nH):
                    tau_h = tau_all[:, h_idx, 0]
                    target_h = target_all[:, h_idx]
                    hours_per_sample = tau_hour_all[:, h_idx, 0].tolist()
                    pred_bil = interpolate_time_with_f_interpolate(x0, xT, tau_h, mode="bilinear")
                    if is_atmvfi:
                        # ATM-VFI's forward signature is model.net(x0, xT, tau)
                        # over the 24 prognostic channels. The asymmetric 27→24
                        # static concat is handled inside _atmvfi_forward_with_static.
                        pred_model = model.net(x0, xT, tau_h)
                    else:
                        out = model(x0, xT, tau_h, cond, static=static)
                        pred_model = out[0] if isinstance(out, tuple) else out
                    err_m = (pred_model - target_h) ** 2   # (B, C, H, W)
                    err_b = (pred_bil - target_h) ** 2
                    for i in range(B):
                        h = int(hours_per_sample[i])
                        if h < 0 or h > 6:
                            continue
                        wrapped_index = batch_idx * loader.batch_size + i
                        if wrapped_index >= len(test_wrapped):
                            break
                        base_idx = grouped[wrapped_index][0] if grouped is not None else wrapped_index
                        year, t0, _, _ = ds_base.index[base_idx]
                        mo = month_from_doy(int(year), int(t0), int(h))
                        season = next((s for s, mns in SEASONS.items() if mo in mns), None)
                        if season is None:
                            continue
                        em = err_m[i:i+1]                  # (1, C, H, W)
                        eb = err_b[i:i+1]
                        for rname, mask in region_masks.items():
                            w = mask * cos_w
                            wsum = float((w.sum()).item())
                            if wsum < 1e-9:
                                continue
                            # per-channel weighted mean SE in region
                            wmse_m = (em * w).sum(dim=(0, 2, 3)) / wsum   # (C,)
                            wmse_b = (eb * w).sum(dim=(0, 2, 3)) / wsum
                            for ci, cname in enumerate(channel_names):
                                sum_sq[rname][season][h].setdefault(cname, 0.0)
                                sum_sq[rname][season][h][cname] += float(wmse_m[ci].item())
                                bil_sum_sq[rname][season][h].setdefault(cname, 0.0)
                                bil_sum_sq[rname][season][h][cname] += float(wmse_b[ci].item())
                            counts[rname][season][h] += 1
                if batch_idx % 25 == 0:
                    print(f"    batch {batch_idx}/{len(loader)}", flush=True)

        # Save per-model JSON
        out_pay = {
            "name": name,
            "model_type": mt,
            "checkpoint": ckpt,
            "regions": list(REGIONS.keys()),
            "seasons": list(SEASONS.keys()),
            "hours": list(range(7)),
            "channel_names": channel_names,
            "per_region_season_hour": {},
        }
        for r in REGIONS:
            out_pay["per_region_season_hour"][r] = {}
            for s in SEASONS:
                out_pay["per_region_season_hour"][r][s] = {}
                for h in range(7):
                    n = counts[r][s][h]
                    if n == 0:
                        continue
                    rmse_m = {f"rmse_{cn}": float(np.sqrt(sum_sq[r][s][h][cn] / n)) for cn in channel_names if cn in sum_sq[r][s][h]}
                    rmse_b = {f"rmse_{cn}": float(np.sqrt(bil_sum_sq[r][s][h][cn] / n)) for cn in channel_names if cn in bil_sum_sq[r][s][h]}
                    out_pay["per_region_season_hour"][r][s][str(h)] = {
                        "n": n, "model": rmse_m, "bilinear": rmse_b,
                    }
        with open(Path(args.out_dir) / f"{name}.json", "w") as f:
            json.dump(out_pay, f, indent=2)
        print(f"  saved {name} in {(time.time()-t0_m)/60:.1f} min")
        del model
        torch.cuda.empty_cache()

    print(f"\n=== DONE in {(time.time()-t_global)/60:.1f} min ===")


if __name__ == "__main__":
    main()
