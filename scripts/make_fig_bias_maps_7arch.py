#!/usr/bin/env python3
"""Per-architecture spatial bias maps for the paper appendix (7 architectures).

For each of {Bilinear, FuXi 24ch, ModAFNO 24ch, S-DYff 24ch, ATM-VFI v2,
WB-Mid 11M} computes the
time-averaged spatial bias (model - ERA5) at tau=3 h over the 2020
validation year, then renders a 6-row x 4-column figure (rows =
architectures, cols = {t2m, u10, v10, MSLP}). Per-column symmetric colour
scale (blue=under, red=over) for direct sign/magnitude comparison.

Inputs:
  - Memmap cache at /tmp/wb2_0p5_cache/ for 2020.
  - Checkpoints (env-vars or canonical defaults below).

Output:
  paper/figs/fig_bias_maps_7arch.pdf

Run on the cluster (GPU). It re-uses the proven loader machinery in
``tools/eval/batch_eval_memmap.py`` (ATM-VFI asymmetric I/O surgical
loader, DC-AE legacy compat, ModAFNO native res via env vars).

Usage on cloud.ru (canonical paths):
  cd /home/jovyan/dsuhoi/weather_time_interpolation
  CUDA_VISIBLE_DEVICES=0 \\
  CKPT_FUXI=logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt \\
  CKPT_MODAFNO=logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt \\
  CKPT_SDYFF=logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt \\
  CKPT_ATMVFI=logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt \\
  CKPT_WBMID=logs/wb_mid_11M_v2/last.ckpt \\
  MODAFNO_INP_H=360 MODAFNO_INP_W=720 \\
  MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720 \\
  SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0 \\
  python scripts/make_fig_bias_maps_7arch.py
"""
from __future__ import annotations
import os
import sys
import time
import importlib
from pathlib import Path

import numpy as np
import torch

# --- repo path ---
REPO = Path("/home/jovyan/dsuhoi/weather_time_interpolation")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import (
    WeatherHermiteLightningModule,
    ERA5WeatherHermiteDataset,
)

# Bring in proven loader + forward dispatcher.
from tools.eval.batch_eval_memmap import load_model_safe, _forward_hermite  # type: ignore


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

PANEL_CHANNELS = [
    ("t2m",  "2 m temperature",        "K"),
    ("u10",  "10 m zonal wind",        r"m s$^{-1}$"),
    ("v10",  "10 m meridional wind",   r"m s$^{-1}$"),
    ("mslp", "Mean sea level pressure", "Pa"),
]
CH_INDEX = {"t2m": 20, "u10": 21, "v10": 22, "mslp": 23}

ARCH_DISPLAY = [
    "Linear Interp.",
    "FuXi 24ch",
    "ModAFNO 24ch",
    "S-DYff 24ch",
    "ATM-VFI v2",
    "WB-Mid 11M",
]

# (display_name, env var, default canonical ckpt path)
ARCH_SPECS = [
    ("Linear Interp.",         None,           None),
    ("FuXi 24ch",        "CKPT_FUXI",
        "logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"),
    ("ModAFNO 24ch",     "CKPT_MODAFNO",
        "logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("S-DYff 24ch",      "CKPT_SDYFF",
        "logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("ATM-VFI v2",       "CKPT_ATMVFI",
        "logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt"),
    ("WB-Mid 11M",       "CKPT_WBMID",
        "logs/wb_mid_11M_v2/last.ckpt"),
]

TAU_HOURS = 3
MAX_SAMPLES = int(os.environ.get("BIAS_MAX_SAMPLES", "240"))
BATCH_LOG = int(os.environ.get("BIAS_LOG_EVERY", "20"))


# -----------------------------------------------------------------------
# Bias accumulation
# -----------------------------------------------------------------------

def compute_bias(model, model_type, ds, device, channel_indices):
    """Accumulate per-pixel mean(model_pred - target) at TAU_HOURS over 2020.

    Returns dict {channel_key -> (H, W) ndarray, "n" -> int}.
    Single dataset walk -> 4 channels per sample for efficiency.
    """
    base = getattr(ds, "base_dataset", ds)
    H = getattr(base, "lat_size", None) or getattr(base, "H", 360)
    W = getattr(base, "lon_size", None) or getattr(base, "W", 720)
    acc = {k: np.zeros((H, W), dtype=np.float64) for k in channel_indices}
    n = 0
    delta_t = 6.0
    is_atmvfi = (model_type == "atm_vfi_pixel_attn")

    if model is not None:
        model = model.to(device).eval()

    t_start = time.time()
    for i, sample in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        x0 = sample["x0"][:24, :, :].unsqueeze(0).to(device)
        xT = sample["xT"][:24, :, :].unsqueeze(0).to(device)
        target = sample["target"]
        # multi-tau (B?, n_tau, C, H, W) — pick tau index = TAU_HOURS - 1
        if target.ndim == 4:  # (n_tau, C, H, W) from per-sample dataset
            tgt = target[TAU_HOURS - 1, :24].unsqueeze(0).to(device)
        elif target.ndim == 5:
            tgt = target[:, TAU_HOURS - 1, :24].to(device)
        else:
            raise RuntimeError(f"unexpected target ndim {target.ndim}")

        tau_in = torch.tensor([[float(TAU_HOURS)]], device=device)
        cond = sample.get("cond", torch.zeros(1, 1, device=device))
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        static = sample.get("static")
        if static is not None:
            static = static.to(device)
            if static.ndim == 3:
                static = static.unsqueeze(0)

        with torch.no_grad():
            if model is None:
                w = float(TAU_HOURS) / delta_t
                pred = (1.0 - w) * x0 + w * xT
            else:
                # ATM-VFI: surgical 27-in / 24-out handled inside the loader
                # which monkey-patches net.forward to prepend static internally,
                # so we always pass 24ch here.
                pred_full = _forward_hermite(
                    model, x0, xT, tau_in, cond, static, atm_vfi=is_atmvfi,
                )
                # Slice to the first 24 channels (24-channel layout)
                pred = pred_full[:, :24, :, :]

        diff = (pred - tgt).squeeze(0).detach().float().cpu().numpy()
        for key, ci in channel_indices.items():
            acc[key] += diff[ci]
        n += 1
        if (i + 1) % BATCH_LOG == 0:
            dt = time.time() - t_start
            print(f"    [bias] {i+1}/{MAX_SAMPLES}  elapsed={dt:.1f}s  rate={(i+1)/dt:.2f}/s",
                  flush=True)

    if n == 0:
        return {k: np.full((H, W), np.nan, dtype=np.float32) for k in channel_indices}
    return {k: (acc[k] / n).astype(np.float32) for k in channel_indices}


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_grid(biases_by_arch_channel, out_pdf):
    n_rows = len(ARCH_DISPLAY)
    n_cols = len(PANEL_CHANNELS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(16, 1.8 * n_rows + 1.0),
                              sharex=True, sharey=True)

    # Column-wise symmetric vlim (95th pctl of |bias| across all rows).
    col_vlims = []
    for ci, (ch_key, _, _) in enumerate(PANEL_CHANNELS):
        vals = []
        for arch in ARCH_DISPLAY:
            arr = biases_by_arch_channel.get((arch, ch_key))
            if arr is not None and np.isfinite(arr).any():
                vals.append(np.abs(arr.ravel()))
        if not vals:
            col_vlims.append(1.0); continue
        flat = np.concatenate(vals)
        col_vlims.append(max(1e-6, float(np.percentile(flat[np.isfinite(flat)], 98))))

    for ri, arch in enumerate(ARCH_DISPLAY):
        for ci, (ch_key, ch_label, unit) in enumerate(PANEL_CHANNELS):
            ax = axes[ri, ci]
            arr = biases_by_arch_channel.get((arch, ch_key))
            if arr is None or not np.isfinite(arr).any():
                ax.set_facecolor("#eeeeee"); ax.text(0.5, 0.5, "n/a",
                                                      ha="center", va="center",
                                                      transform=ax.transAxes,
                                                      color="#888", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            vlim = col_vlims[ci]
            norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
            H, W = arr.shape
            lats = np.linspace(89.75, -89.75, H)
            lons = np.linspace(-179.75, 179.75, W)
            im = ax.pcolormesh(lons, lats, arr, cmap="RdBu_r",
                                norm=norm, shading="auto")
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.tick_params(labelsize=7)
            if ri == 0:
                ax.set_title(f"{ch_label}  [{unit}]", fontsize=11,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(arch, fontsize=10, fontweight="bold")
            if ri == n_rows - 1:
                cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                     pad=0.18, shrink=0.85, fraction=0.045)
                cbar.ax.tick_params(labelsize=7)
            else:
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02).ax.tick_params(labelsize=6)

    fig.suptitle(
        r"Time-averaged spatial bias (model $-$ ERA5) at $\tau=3$\,h on 2020",
        fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.01, 1, 0.985])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    out_pdf = REPO / "paper" / "figs" / "fig_bias_maps_7arch.pdf"
    cache_dir = REPO / "metrics" / "bias_maps_7arch"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading dataset (year=2020, max_tau=6, samples_per_date=4)...", flush=True)
    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=[2020], max_tau_hours=6, samples_per_date=4, train=False,
        static_path=str(REPO / "data" / "static_features_0p5.pt"),
        stats_path=str(REPO / "data" / "json_stats_0p5.nc"),
        surface_stats_path=str(REPO / "data" / "surface_stats_0p5.json"),
    )
    ds = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    channel_groups = ds_base.channel_groups

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    biases = {}
    for arch, env_var, default in ARCH_SPECS:
        cache_file = cache_dir / f"{arch.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        if cache_file.exists() and os.environ.get("BIAS_FORCE", "") != "1":
            print(f"[{arch}] CACHED -> {cache_file}", flush=True)
            data = np.load(cache_file)
            for ch_key in CH_INDEX:
                if ch_key in data.files:
                    biases[(arch, ch_key)] = data[ch_key]
            continue

        print(f"\n=== Processing {arch} ===", flush=True)
        if arch == "Linear Interp.":
            model, mt = None, "bilinear"
        else:
            ckpt = os.environ.get(env_var, "") or default
            ckpt_abs = ckpt if os.path.isabs(ckpt) else str(REPO / ckpt)
            if not Path(ckpt_abs).exists():
                print(f"[{arch}] MISSING ckpt {ckpt_abs} — skipping", flush=True)
                continue
            try:
                t0 = time.time()
                model, mt = load_model_safe(ckpt_abs, device, channel_groups)
                print(f"[{arch}] loaded ({mt})  {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[{arch}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
                continue
        try:
            result = compute_bias(model, mt, ds, device, CH_INDEX)
        except Exception as e:
            import traceback
            print(f"[{arch}] FORWARD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        # cache
        np.savez_compressed(cache_file, **result)
        print(f"[{arch}] cached -> {cache_file}", flush=True)
        for ch_key, arr in result.items():
            biases[(arch, ch_key)] = arr

    plot_grid(biases, out_pdf)


if __name__ == "__main__":
    main()
