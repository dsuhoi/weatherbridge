#!/usr/bin/env python3
"""Per-field τ-evolution bias maps for the AAAI 2027 paper appendix.

For each of 4 weather fields {t2m, u10, mslp, Q1000} produces a separate
4×5 bias-map grid:
  - rows: 4 models (Bilinear, FuXi, S-DYff, ATM-VFI v2)
  - cols: τ ∈ {1, 2, 3, 4, 5} h
  - cell: time-averaged spatial bias ``mean(prediction − ERA5)`` on 2020
  - shared symmetric colour scale per figure (one cbar per figure)

Reuses the loader + forward dispatcher from
``tools/eval/batch_eval_memmap.py`` (ATM-VFI 27→24 surgical I/O, etc.).

Outputs (4 PDFs):
  paper/figs/fig_bias_maps_t2m_5tau.pdf
  paper/figs/fig_bias_maps_u10_5tau.pdf
  paper/figs/fig_bias_maps_mslp_5tau.pdf
  paper/figs/fig_bias_maps_Q1000_5tau.pdf

Run on cloud.ru GPU node:

  cd /home/jovyan/dsuhoi/weather_time_interpolation
  CUDA_VISIBLE_DEVICES=0 BIAS_MAX_SAMPLES=20 BIAS_LOG_EVERY=2 \
  SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0 \
  python scripts/make_fig_bias_maps_5tau_per_field.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# --- repo path (cloudru canonical) ---
REPO = Path("/home/jovyan/dsuhoi/weather_time_interpolation")
if not REPO.exists():
    # local fallback: figure out from this file's location
    REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import ERA5WeatherHermiteDataset

from tools.eval.batch_eval_memmap import load_model_safe, _forward_hermite  # type: ignore


# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

# Channel layout: PL [T,U,V,Q,Z] × [1000,925,850,700] = 20 idx 0..19;
# surface [t2m,u10,v10,mslp,sst,tcc,tcwv] at 20..26.
# Q1000 → index 12 (Q starts at 4*3=12; 1000 hPa is the first level).
FIELDS = [
    # (key,  channel_idx, display_label,                   unit,      vlim)
    ("t2m",  20, "2 m temperature",            "K",                5.0),
    ("u10",  21, "10 m zonal wind",            r"m s$^{-1}$",      2.0),
    ("mslp", 23, "Mean sea level pressure",    "Pa",             200.0),
    ("Q1000", 12, "Specific humidity 1000 hPa", r"g kg$^{-1}$",    0.5),
]
FIELD_KEYS = [f[0] for f in FIELDS]

TAUS = [1, 2, 3, 4, 5]

MODELS = [
    # (display_name, env_var, default_ckpt)
    ("Linear Interp.",          None,           None),
    ("FuXi 24ch",        "CKPT_FUXI",
        "logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"),
    ("S-DYff 24ch",      "CKPT_SDYFF",
        "logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("ATM-VFI v2",       "CKPT_ATMVFI",
        "logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt"),
]
MODEL_NAMES = [m[0] for m in MODELS]

MAX_SAMPLES = int(os.environ.get("BIAS_MAX_SAMPLES", "20"))
LOG_EVERY = int(os.environ.get("BIAS_LOG_EVERY", "2"))


# -----------------------------------------------------------------------
# Bias accumulation
# -----------------------------------------------------------------------

def _denorm(arr_norm: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Convert a single-channel slice from normalised back to physical units."""
    return arr_norm * sigma + mu


def compute_bias_per_tau(model, model_type, ds, ds_base, device):
    """Return dict mapping (field_key, tau_h) -> (H, W) ndarray (physical units).

    Walks the dataset once; for each window does 5 forward passes (one per τ).
    Accumulates the per-pixel difference (pred − target) IN PHYSICAL UNITS
    for the 4 fields of interest.
    """
    # Pull (H, W) from the first memmap year (authoritative).
    first_year = next(iter(ds_base.memmaps))
    _, _, H, W = ds_base.memmaps[first_year].shape

    # Denorm constants per channel: surface fields use surface_mu/sigma;
    # pressure-level Q1000 uses the PL mu/sigma.
    n_pl = ds_base.in_channels  # 20
    surf_mu = ds_base.surface_mu.view(-1).numpy()  # (7,)
    surf_sigma = ds_base.surface_sigma.view(-1).numpy()
    pl_mu = ds_base.mu.view(-1).numpy()  # (20,)
    pl_sigma = ds_base.sigma.view(-1).numpy()

    def get_mu_sigma(ch_idx: int):
        if ch_idx < n_pl:
            return float(pl_mu[ch_idx]), float(pl_sigma[ch_idx])
        return float(surf_mu[ch_idx - n_pl]), float(surf_sigma[ch_idx - n_pl])

    acc = {(fk, tau): np.zeros((H, W), dtype=np.float64) for fk in FIELD_KEYS for tau in TAUS}
    n = 0
    is_atmvfi = (model_type == "atm_vfi_pixel_attn")
    if model is not None:
        model = model.to(device).eval()

    t_start = time.time()
    total = min(MAX_SAMPLES, len(ds))
    for i, sample in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        x0 = sample["x0"][:24, :, :].unsqueeze(0).to(device)
        xT = sample["xT"][:24, :, :].unsqueeze(0).to(device)
        target_all = sample["target"]  # shape (n_tau, 27, H, W) where n_tau=5
        if target_all.ndim != 4 or target_all.shape[0] < 5:
            raise RuntimeError(f"unexpected target shape {tuple(target_all.shape)}")

        cond = sample.get("cond", torch.zeros(1, 1, device=device))
        if cond.ndim == 1:
            cond = cond.unsqueeze(0)
        cond = cond.to(device)
        static = sample.get("static")
        if static is not None:
            static = static.to(device)
            if static.ndim == 3:
                static = static.unsqueeze(0)

        delta_t = 6.0
        for tau_h in TAUS:
            tgt_norm = target_all[tau_h - 1, :24].unsqueeze(0).to(device)  # (1,24,H,W) NORMALISED
            tau_in = torch.tensor([[float(tau_h)]], device=device)
            with torch.no_grad():
                if model is None:
                    w = float(tau_h) / delta_t
                    pred = (1.0 - w) * x0 + w * xT
                else:
                    pred_full = _forward_hermite(
                        model, x0, xT, tau_in, cond, static, atm_vfi=is_atmvfi,
                    )
                    pred = pred_full[:, :24, :, :]
            diff_norm = (pred - tgt_norm).squeeze(0).detach().float().cpu().numpy()
            # Convert each channel back to physical units before accumulating.
            for fk, ch_idx, *_ in FIELDS:
                mu, sigma = get_mu_sigma(ch_idx)
                # NOTE: bias is a *difference*; mu cancels out, only sigma matters.
                phys = diff_norm[ch_idx] * sigma
                if fk == "Q1000":
                    phys = phys * 1000.0  # kg/kg → g/kg
                acc[(fk, tau_h)] += phys
        n += 1
        if (i + 1) % LOG_EVERY == 0:
            dt = time.time() - t_start
            rate = (i + 1) / max(dt, 1e-6)
            print(f"    [{i+1}/{total}] elapsed={dt:.1f}s rate={rate:.2f}/s "
                  f"({rate*5:.2f} fwd/s)", flush=True)

    if n == 0:
        return {k: np.full((H, W), np.nan, dtype=np.float32) for k in acc}
    return {k: (acc[k] / n).astype(np.float32) for k in acc}


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_one_field(field_key, ch_label, unit, vlim_default,
                   biases_by_model_tau, out_pdf, H, W):
    """Render a single 5(model)×5(τ) figure with one shared colorbar."""
    n_rows = len(MODEL_NAMES)
    n_cols = len(TAUS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols + 0.8, 1.7 * n_rows + 0.6),
                             sharex=True, sharey=True)

    # vlim: take fixed default but soften if all data is much smaller
    vals = []
    for mn in MODEL_NAMES:
        for tau in TAUS:
            arr = biases_by_model_tau.get((mn, tau))
            if arr is not None and np.isfinite(arr).any():
                vals.append(np.abs(arr))
    if vals:
        empirical = float(np.percentile(np.concatenate([v.ravel() for v in vals]), 99))
        # Use the smaller of fixed default and 1.5×empirical, but not below empirical
        vlim = max(min(vlim_default, 1.5 * empirical), empirical)
    else:
        vlim = vlim_default
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vlim, vmax=vlim)

    lats = np.linspace(89.75, -89.75, H)
    lons = np.linspace(-179.75, 179.75, W)
    last_im = None

    for ri, mname in enumerate(MODEL_NAMES):
        for ci, tau in enumerate(TAUS):
            ax = axes[ri, ci]
            arr = biases_by_model_tau.get((mname, tau))
            if arr is None or not np.isfinite(arr).any():
                ax.set_facecolor("#eeeeee")
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="#888", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            im = ax.pcolormesh(lons, lats, arr, cmap="RdBu_r",
                               norm=norm, shading="auto", rasterized=True)
            last_im = im
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            if ri == n_rows - 1:
                ax.set_xticks([-180, -90, 0, 90, 180])
                ax.tick_params(axis="x", labelsize=7)
            else:
                ax.set_xticks([])
            if ci == 0:
                ax.set_yticks([-90, -45, 0, 45, 90])
                ax.tick_params(axis="y", labelsize=7)
            else:
                ax.set_yticks([])
            if ri == 0:
                ax.set_title(rf"$\tau={tau}$\,h", fontsize=10, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(mname, fontsize=10, fontweight="bold")

    # Shared horizontal colorbar across the bottom.
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    if last_im is not None:
        cbar_ax = fig.add_axes([0.15, 0.04, 0.7, 0.02])
        cbar = fig.colorbar(last_im, cax=cbar_ax, orientation="horizontal")
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label(rf"bias ({unit})", fontsize=9)

    fig.suptitle(
        rf"{ch_label}: time-averaged bias (model $-$ ERA5) on 2020, per $\tau$",
        fontsize=11, fontweight="bold", y=0.985,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    out_dir = REPO / "paper" / "figs"
    cache_dir = REPO / "metrics" / "bias_maps_5tau_per_field"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"REPO: {REPO}", flush=True)
    print(f"MAX_SAMPLES per model: {MAX_SAMPLES}; τ list: {TAUS}", flush=True)

    print("loading dataset (year=2020, max_tau_hours=6, samples_per_date=4)...", flush=True)
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
    print(f"device: {device}; len(ds)={len(ds)}", flush=True)

    # biases[(model_name, tau, field_key)] -> (H, W) ndarray
    biases = {}
    H = W = None

    for mname, env_var, default in MODELS:
        cache_file = cache_dir / f"{mname.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        if cache_file.exists() and os.environ.get("BIAS_FORCE", "") != "1":
            print(f"[{mname}] CACHED -> {cache_file}", flush=True)
            data = np.load(cache_file)
            for fk in FIELD_KEYS:
                for tau in TAUS:
                    key = f"{fk}__tau{tau}"
                    if key in data.files:
                        arr = data[key]
                        biases[(mname, tau, fk)] = arr
                        H, W = arr.shape
            continue

        print(f"\n=== Processing {mname} ===", flush=True)
        if mname == "Linear Interp.":
            model, mt = None, "bilinear"
        else:
            ckpt = os.environ.get(env_var, "") or default
            ckpt_abs = ckpt if os.path.isabs(ckpt) else str(REPO / ckpt)
            if not Path(ckpt_abs).exists():
                print(f"[{mname}] MISSING ckpt {ckpt_abs} — skipping", flush=True)
                continue
            try:
                t0 = time.time()
                model, mt = load_model_safe(ckpt_abs, device, channel_groups)
                print(f"[{mname}] loaded ({mt}) {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                import traceback
                print(f"[{mname}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                continue
        try:
            result = compute_bias_per_tau(model, mt, ds, ds_base, device)
        except Exception as e:
            import traceback
            print(f"[{mname}] FORWARD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        # Cache to disk
        save_dict = {}
        for (fk, tau), arr in result.items():
            save_dict[f"{fk}__tau{tau}"] = arr
            biases[(mname, tau, fk)] = arr
            H, W = arr.shape
        np.savez_compressed(cache_file, **save_dict)
        print(f"[{mname}] cached -> {cache_file}", flush=True)

    # Default size if no model ran
    if H is None:
        H, W = 360, 720

    # ---- Plot 4 PDFs ----
    for field_key, ch_idx, ch_label, unit, vlim_default in FIELDS:
        out_pdf = out_dir / f"fig_bias_maps_{field_key}_5tau.pdf"
        sub = {(mn, tau): biases.get((mn, tau, field_key))
               for mn in MODEL_NAMES for tau in TAUS}
        plot_one_field(field_key, ch_label, unit, vlim_default,
                       sub, out_pdf, H, W)


if __name__ == "__main__":
    main()
