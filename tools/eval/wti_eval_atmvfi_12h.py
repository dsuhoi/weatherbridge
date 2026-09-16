#!/usr/bin/env python3
"""Per-τ 12h eval adapter for ATM-VFI checkpoints (custom LightningModule).

ATM-VFI lives in ``train_atm_vfi_12h_oddskip.py`` (a separate trainer outside
``trainer_weather_hermite``), so it can't be loaded by ``batch_eval_12h_memmap``.
This script loads the checkpoint, runs val 2020, and emits a JSON identical in
shape to ``batch_eval_12h_memmap.py`` for downstream plot scripts.

Usage::

    python tools/eval/wti_eval_atmvfi_12h.py \\
        --ckpt /path/to/atm_vfi_12h/last.ckpt \\
        --memmap-dir /tmp/wb2_0p5_cache \\
        --test-year 2020 \\
        --out metrics/eval_12h_2020_pre_ep10/ATM-VFI_3yr_12h_fibo.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


CH_24 = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]

SEEN_TAU_12H = [1, 2, 3, 5, 7, 9, 10, 11]
UNSEEN_TAU_12H = [4, 6, 8]


def _lat_weights(H: int, device: torch.device) -> torch.Tensor:
    """Cosine-latitude weights, shape (1, 1, H, 1), sums to 1."""
    lat = np.linspace(89.75, -89.75, H, dtype=np.float32)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    return torch.from_numpy(w).to(device=device, dtype=torch.float32).view(1, 1, -1, 1)


def _bilinear(x0: torch.Tensor, xT: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    tau_b = tau.view(-1, 1, 1, 1)
    return (1.0 - tau_b) * x0 + tau_b * xT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ATM-VFI_3yr_12h_fibo")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=2)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--max-tau-hours", type=int, default=12)
    ap.add_argument("--eval-hours", default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    eval_hours = sorted({int(x) for x in args.eval_hours.split(",") if x.strip()})

    # Load ATM-VFI from its own trainer file.
    train_mod_path = Path(__file__).resolve().parents[2] / "train_atm_vfi_12h_oddskip.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_atm_vfi", train_mod_path)
    atm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(atm_mod)
    PixelAttentionVFI = atm_mod.PixelAttentionVFI

    print(f"loading {args.ckpt}...")
    model = PixelAttentionVFI.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.1f}M")

    # Build memmap dataset (27 ch) — we'll slice to 24 in the loop.
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=args.max_tau_hours,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=eval_hours,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    # Economy filter (same as EvaluationRunner)
    day_picks = {2: [1, 15], 3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}
    K = max(1, int(args.eval_days_per_month))
    picks = day_picks.get(K, sorted({1 + i * (30 // K) for i in range(K)}))
    allowed = set(picks)
    import datetime as _dt
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
    print(f"  economy filter: {len(filt)}/{len(ds_base.index)}")
    ds_base.index = filt

    wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=float(args.max_tau_hours))
    loader = DataLoader(wrapped, batch_size=args.batch_size, num_workers=0, shuffle=False)
    print(f"  windows: {len(wrapped)}")

    H = 360
    w_lat = _lat_weights(H, device)

    mu_pl = ds_base.mu.to(device).view(1, -1, 1, 1)
    sigma_pl = ds_base.sigma.to(device).view(1, -1, 1, 1)
    mu_surf = ds_base.surface_mu.to(device).view(1, -1, 1, 1)
    sigma_surf = ds_base.surface_sigma.to(device).view(1, -1, 1, 1)
    # Concatenated mu/sigma over 27ch, slice to first 24
    mu_all = torch.cat([mu_pl, mu_surf], dim=1)[:, :24]
    sigma_all = torch.cat([sigma_pl, sigma_surf], dim=1)[:, :24]

    rmse_norm = {m: {tau: {} for tau in eval_hours} for m in ("model", "bilinear")}
    rmse_phys = {m: {tau: {} for tau in eval_hours} for m in ("model", "bilinear")}
    n_per_tau = {tau: 0 for tau in eval_hours}

    t0 = time.time()
    max_tau = float(args.max_tau_hours)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x0 = batch["x0"].to(device)
            xT = batch["xT"].to(device)
            tau_hour_all = batch["tau_hour"].long()  # (B, nH, 1)
            target_all = batch["target"].to(device)  # (B, nH, C, H, W)
            # Slice to 24 channels.
            x0 = x0[:, :24].contiguous()
            xT = xT[:, :24].contiguous()
            target_all = target_all[:, :, :24].contiguous()

            B = x0.size(0)
            nH = tau_hour_all.size(1)
            for h_idx in range(nH):
                tau_h_int = tau_hour_all[:, h_idx, 0]
                tau_norm = tau_h_int.float().to(device) / max_tau
                target_h = target_all[:, h_idx]

                # Build dict expected by ATM-VFI (uses .net directly).
                pred_model = model.net(x0, xT, tau_norm)
                pred_bil = _bilinear(x0, xT, tau_norm)

                # Norm RMSE
                err_m_n = ((pred_model - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                err_b_n = ((pred_bil - target_h) ** 2 * w_lat).sum(dim=(-2, -1))
                # Phys RMSE
                tgt_phys = target_h * sigma_all + mu_all
                m_phys = pred_model * sigma_all + mu_all
                b_phys = pred_bil * sigma_all + mu_all
                err_m_p = ((m_phys - tgt_phys) ** 2 * w_lat).sum(dim=(-2, -1))
                err_b_p = ((b_phys - tgt_phys) ** 2 * w_lat).sum(dim=(-2, -1))

                for i in range(B):
                    tau_i = int(tau_h_int[i].item())
                    if tau_i not in n_per_tau:
                        continue
                    n_per_tau[tau_i] += 1
                    for ci, name in enumerate(CH_24):
                        rmse_norm["model"][tau_i].setdefault(name, 0.0)
                        rmse_norm["model"][tau_i][name] += float(err_m_n[i, ci].item())
                        rmse_norm["bilinear"][tau_i].setdefault(name, 0.0)
                        rmse_norm["bilinear"][tau_i][name] += float(err_b_n[i, ci].item())
                        rmse_phys["model"][tau_i].setdefault(name, 0.0)
                        rmse_phys["model"][tau_i][name] += float(err_m_p[i, ci].item())
                        rmse_phys["bilinear"][tau_i].setdefault(name, 0.0)
                        rmse_phys["bilinear"][tau_i][name] += float(err_b_p[i, ci].item())
            if bi % 20 == 0:
                print(f"  batch {bi}/{len(loader)}")

    # Build JSON matching batch_eval_12h_memmap schema (no bicubic, no ACC).
    per_tau_out: Dict[str, Dict] = {}
    for tau in eval_hours:
        n = n_per_tau[tau]
        if n == 0:
            continue
        per_tau_out[str(tau)] = {"model": {}, "bilinear": {}, "bicubic": {}}
        for method in ("model", "bilinear"):
            for name in CH_24:
                per_tau_out[str(tau)][method][f"rmse_norm_{name}"] = float(
                    np.sqrt(rmse_norm[method][tau][name] / n)
                )
                per_tau_out[str(tau)][method][f"rmse_phys_{name}"] = float(
                    np.sqrt(rmse_phys[method][tau][name] / n)
                )
        # Bicubic placeholder = bilinear (atm-vfi standalone doesn't compute it)
        for k, v in per_tau_out[str(tau)]["bilinear"].items():
            per_tau_out[str(tau)]["bicubic"][k] = v

    payload = {
        "checkpoint": args.ckpt,
        "model_type": "atm_vfi_pixel_attn",
        "delta_t_hours": float(args.max_tau_hours),
        "num_samples": int(sum(n_per_tau.values())),
        "n_per_tau": {str(k): int(v) for k, v in n_per_tau.items()},
        "years": [args.test_year],
        "channel_names": CH_24,
        "acc_channel_names": [],
        "seen_tau": SEEN_TAU_12H,
        "unseen_tau": UNSEEN_TAU_12H,
        "per_tau": per_tau_out,
        "paper_tag": "12h_2020",
        "params_m": float(n_params),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved {out_path} in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
