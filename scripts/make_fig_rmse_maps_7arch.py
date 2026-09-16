#!/usr/bin/env python3
"""Per-architecture spatial *RMSE* maps for the paper appendix.

Companion to ``make_fig_bias_maps_7arch.py`` but accumulates per-pixel
*squared* differences and reports ``sqrt(mean((pred - ERA5)**2))`` instead
of the signed bias mean. Squared-error maps are visually honest about
which architecture errs less per pixel: bilinear, despite its near-zero
time-mean bias, has *large* per-step variance so its RMSE map is bright
everywhere; ML models concentrate their residual error around mountains
/ coasts and stay dim elsewhere.

For each of {Bilinear, SwinV2, ModAFNO 24ch, S-DYff 24ch,
PixelAttn-VFI}
computes the
time-averaged spatial RMSE at tau=3 h over the 2020 validation year,
then renders a row x 4-column figure (rows = architectures, cols =
{t2m, u10, v10, MSLP}). Sequential colour scale (viridis_r), shared per
column.

Outputs (normalised units inside accumulators, but the 7-arch headline
stays in normalised RMSE for parity with the original headline; the
5-tau follow-up reports physical units).

  paper/images/fig_rmse_maps_7arch.pdf
  metrics/rmse_maps_7arch/<arch>.npz

Run on cloud.ru GPU:
  cd /home/jovyan/dsuhoi/weather_time_interpolation
  CUDA_VISIBLE_DEVICES=0 \\
  CKPT_FUXI=logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt \\
  CKPT_MODAFNO=logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt \\
  CKPT_SDYFF=logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt \\
  CKPT_ATMVFI=logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt \\
  MODAFNO_INP_H=360 MODAFNO_INP_W=720 \\
  MODAFNO_NATIVE_H=360 MODAFNO_NATIVE_W=720 \\
  SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0 \\
  python scripts/make_fig_rmse_maps_7arch.py
"""
from __future__ import annotations
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/home/jovyan/dsuhoi/weather_time_interpolation")
if not REPO.exists():
    REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

try:
    import torch
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from trainer_weather_hermite import (
        WeatherHermiteLightningModule,
        ERA5WeatherHermiteDataset,
    )
    from tools.eval.batch_eval_memmap import load_model_safe, _forward_hermite  # type: ignore
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    ERA5MemmapDataset = None  # type: ignore[assignment]
    WeatherHermiteLightningModule = None  # type: ignore[assignment]
    ERA5WeatherHermiteDataset = None  # type: ignore[assignment]
    load_model_safe = None  # type: ignore[assignment]
    _forward_hermite = None  # type: ignore[assignment]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "SwinV2",
    "ModAFNO",
    "S-DYff",
    "PixelAttn-VFI",
]

ARCH_SPECS = [
    ("Linear Interp.",            None,           None),
    ("SwinV2",              "CKPT_FUXI",
        "logs/exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt"),
    ("ModAFNO",             "CKPT_MODAFNO",
        "logs/exp_modafno_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("S-DYff",              "CKPT_SDYFF",
        "logs/exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt"),
    ("PixelAttn-VFI",       "CKPT_ATMVFI",
        "logs/exp_atm_vfi_24ch_v2_static_3yr_135only_fibo/last.ckpt"),
]

CACHE_STEMS = {
    "Linear Interp.": "Bilinear",
    "PixelAttn-VFI": "ATM-VFI",
}


def _cache_stem(name: str) -> str:
    return CACHE_STEMS.get(name, name.replace(" ", "_").replace("(", "").replace(")", ""))

TAU_HOURS = 3
MAX_SAMPLES = int(os.environ.get("RMSE_MAX_SAMPLES", os.environ.get("BIAS_MAX_SAMPLES", "240")))
BATCH_LOG = int(os.environ.get("RMSE_LOG_EVERY", os.environ.get("BIAS_LOG_EVERY", "20")))


# Per-channel sigma cache (filled at runtime from ds_base). Used to convert
# normalised-space squared error back to physical units before sqrt.
SIGMA_BY_CH: dict[int, float] = {}


# -----------------------------------------------------------------------
# Squared-error accumulation
# -----------------------------------------------------------------------

def compute_rmse(model, model_type, ds, device, channel_indices):
    """Accumulate per-pixel mean((model_pred - target)**2) at TAU_HOURS over 2020.

    Returns dict {channel_key -> (H, W) float32 RMSE in physical units}.
    """
    base = getattr(ds, "base_dataset", ds)
    H = getattr(base, "lat_size", None) or getattr(base, "H", 360)
    W = getattr(base, "lon_size", None) or getattr(base, "W", 720)
    acc_sq = {k: np.zeros((H, W), dtype=np.float64) for k in channel_indices}
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
        if target.ndim == 4:
            tgt = target[TAU_HOURS - 1, :24].unsqueeze(0).to(device)
        elif target.ndim == 5:
            tgt = target[:, TAU_HOURS - 1, :24].to(device)
        else:
            raise RuntimeError(f"unexpected target ndim {target.ndim}")

        # Models expect tau as a fraction of delta_t (0..1). Bilinear weight w
        # uses the same fraction implicitly via tau_h / delta_t below.
        tau_in = torch.tensor([[float(TAU_HOURS) / delta_t]], device=device)
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
                pred_full = _forward_hermite(
                    model, x0, xT, tau_in, cond, static, atm_vfi=is_atmvfi,
                )
                pred = pred_full[:, :24, :, :]

        diff = (pred - tgt).squeeze(0).detach().float().cpu().numpy()  # normalised units
        for key, ci in channel_indices.items():
            sigma = SIGMA_BY_CH.get(ci, 1.0)
            phys_sq = (diff[ci] * sigma) ** 2
            acc_sq[key] += phys_sq
        n += 1
        if (i + 1) % BATCH_LOG == 0:
            dt = time.time() - t_start
            print(f"    [rmse] {i+1}/{MAX_SAMPLES}  elapsed={dt:.1f}s  rate={(i+1)/dt:.2f}/s",
                  flush=True)

    if n == 0:
        return {k: np.full((H, W), np.nan, dtype=np.float32) for k in channel_indices}
    return {k: np.sqrt(acc_sq[k] / n).astype(np.float32) for k in channel_indices}


# -----------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------

def plot_grid(rmse_by_arch_channel, out_pdf):
    n_rows = len(ARCH_DISPLAY)
    n_cols = len(PANEL_CHANNELS)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(16, 1.8 * n_rows + 1.0),
                              sharex=True, sharey=True)

    # Per-column vmax (98th pctl of RMSE across all rows, in physical units).
    col_vmax = []
    for ci, (ch_key, _, _) in enumerate(PANEL_CHANNELS):
        vals = []
        for arch in ARCH_DISPLAY:
            arr = rmse_by_arch_channel.get((arch, ch_key))
            if arr is not None and np.isfinite(arr).any():
                vals.append(arr.ravel())
        if not vals:
            col_vmax.append(1.0); continue
        flat = np.concatenate(vals)
        col_vmax.append(max(1e-6, float(np.percentile(flat[np.isfinite(flat)], 98))))

    for ri, arch in enumerate(ARCH_DISPLAY):
        for ci, (ch_key, ch_label, unit) in enumerate(PANEL_CHANNELS):
            ax = axes[ri, ci]
            arr = rmse_by_arch_channel.get((arch, ch_key))
            if arr is None or not np.isfinite(arr).any():
                ax.set_facecolor("#eeeeee"); ax.text(0.5, 0.5, "n/a",
                                                      ha="center", va="center",
                                                      transform=ax.transAxes,
                                                      color="#888", fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            vmax = col_vmax[ci]
            H, W = arr.shape
            lats = np.linspace(89.75, -89.75, H)
            lons = np.linspace(0.25, 359.75, W)
            im = ax.pcolormesh(lons, lats, arr, cmap="viridis_r",
                                vmin=0.0, vmax=vmax, shading="auto",
                                rasterized=True)
            ax.set_xlim(0, 360); ax.set_ylim(-90, 90)
            ax.set_xticks([0, 60, 120, 180, 240, 300, 360])
            ax.set_yticks([-90, -45, 0, 45, 90])
            ax.tick_params(labelsize=7, labelleft=True, labelbottom=True)
            ax.grid(True, color="white", alpha=0.35, lw=0.4, ls="--")
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
        r"Time-averaged spatial RMSE per pixel at $\tau=3$ h on 2020 (lower is better)",
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
    out_pdf = REPO / "paper" / "images" / "fig_rmse_maps_7arch.pdf"
    cache_dir = REPO / "metrics" / "rmse_maps_7arch"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if os.environ.get("RMSE_FORCE", "") != "1":
        rmses = {}
        missing = []
        for arch, _env_var, _default in ARCH_SPECS:
            cache_file = cache_dir / f"{_cache_stem(arch)}.npz"
            if not cache_file.exists():
                missing.append(arch)
                continue
            print(f"[{arch}] CACHED -> {cache_file}", flush=True)
            data = np.load(cache_file)
            for ch_key in CH_INDEX:
                if ch_key in data.files:
                    rmses[(arch, ch_key)] = data[ch_key]
        if not missing:
            plot_grid(rmses, out_pdf)
            return
        if torch is None:
            raise RuntimeError(
                "torch is required to recompute missing RMSE-map caches: "
                + ", ".join(missing)
            )

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

    # Fill sigma cache for selected channels (PL idx 0..19 -> ds_base.sigma;
    # surface idx 20..26 -> ds_base.surface_sigma).
    n_pl = ds_base.in_channels
    pl_sigma = ds_base.sigma.view(-1).numpy()
    surf_sigma = ds_base.surface_sigma.view(-1).numpy()
    for ch_key, ch_idx in CH_INDEX.items():
        if ch_idx < n_pl:
            SIGMA_BY_CH[ch_idx] = float(pl_sigma[ch_idx])
        else:
            SIGMA_BY_CH[ch_idx] = float(surf_sigma[ch_idx - n_pl])
    print(f"sigma cache: {SIGMA_BY_CH}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)

    rmses = {}
    for arch, env_var, default in ARCH_SPECS:
        cache_file = cache_dir / f"{_cache_stem(arch)}.npz"
        if cache_file.exists() and os.environ.get("RMSE_FORCE", "") != "1":
            print(f"[{arch}] CACHED -> {cache_file}", flush=True)
            data = np.load(cache_file)
            for ch_key in CH_INDEX:
                if ch_key in data.files:
                    rmses[(arch, ch_key)] = data[ch_key]
            continue

        print(f"\n=== Processing {arch} ===", flush=True)
        if arch == "Linear Interp.":
            model, mt = None, "bilinear"
        else:
            ckpt = os.environ.get(env_var, "") or default
            ckpt_abs = ckpt if os.path.isabs(ckpt) else str(REPO / ckpt)
            if not Path(ckpt_abs).exists():
                print(f"[{arch}] MISSING ckpt {ckpt_abs} - skipping", flush=True)
                continue
            try:
                t0 = time.time()
                model, mt = load_model_safe(ckpt_abs, device, channel_groups)
                print(f"[{arch}] loaded ({mt})  {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[{arch}] LOAD FAILED: {type(e).__name__}: {e}", flush=True)
                continue
        try:
            result = compute_rmse(model, mt, ds, device, CH_INDEX)
        except Exception as e:
            import traceback
            print(f"[{arch}] FORWARD FAILED: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            continue
        finally:
            if model is not None:
                del model
                torch.cuda.empty_cache()

        np.savez_compressed(cache_file, **result)
        print(f"[{arch}] cached -> {cache_file}", flush=True)
        # Per-channel min/max summary
        for ch_key, arr in result.items():
            rmses[(arch, ch_key)] = arr
            finite = arr[np.isfinite(arr)]
            if finite.size:
                print(f"    {ch_key}: min={finite.min():.4g}  max={finite.max():.4g}  mean={finite.mean():.4g}",
                      flush=True)

    plot_grid(rmses, out_pdf)


if __name__ == "__main__":
    main()
