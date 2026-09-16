#!/usr/bin/env python3
"""Per-architecture spatial bias maps for the paper appendix.

For each of {Bilinear, S-DYff, ATM-VFI, WeatherBridge} computes the
time-averaged spatial bias (model − ERA5) at τ=3 h over the 2020
validation year, then renders a 4-row × 4-column figure (4 architectures
× {t2m, u10, v10, MSLP}). Colour scale is symmetric and shared per
column so signs are comparable across architectures within each channel.

Inputs:
  - Memmap cache at /tmp/wb2_0p5_cache/ for 2020.
  - Checkpoints (env-vars or defaults below).

Output:
  paper/figs/fig_bias_maps_4arch.pdf

Run on the cluster (GPU) — it needs the trained checkpoints to do
forward passes. CPU fallback is implemented but slow. The script can be
launched once the active KD-v4 / NoKD-v4 jobs free a GPU.

Usage (on cloud.ru):
  CUDA_VISIBLE_DEVICES=0 \
  CKPT_BILIN=    \
  CKPT_SDYFF=logs/.../sdyff_ep8.ckpt \
  CKPT_ATMVFI=logs/.../atm_vfi_v2_135only_ep8.ckpt \
  CKPT_WBRIDGE=logs/.../weatherdcae_noskip_24ch_6yr_ep8.ckpt \
  python scripts/make_fig_bias_maps_4arch.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/jovyan/dsuhoi/weather_time_interpolation")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule, ERA5WeatherHermiteDataset


# Channels we visualise (surface — interpretable physical fields).
PANEL_CHANNELS = [
    ("t2m",  "2 m temperature",        "K"),
    ("u10",  "10 m zonal wind",         "m s$^{-1}$"),
    ("v10",  "10 m meridional wind",    "m s$^{-1}$"),
    ("mslp", "Mean sea level pressure", "Pa"),
]
# Channel indices in the standard 24-channel layout.
CH_INDEX = {"t2m": 20, "u10": 21, "v10": 22, "mslp": 23}
ARCHS = ["Bilinear", "S-DYff", "ATM-VFI", "WeatherBridge"]

TAU_HOURS = 3   # mid-window interior offset
MAX_SAMPLES = int(os.environ.get("BIAS_MAX_SAMPLES", "300"))  # cap for speed


def load_ckpt(env_var: str, arch: str):
    """Return loaded LightningModule (eval-mode) for the given architecture, or
    None for the closed-form Bilinear which we compute analytically."""
    if arch == "Bilinear":
        return None
    path = os.environ.get(env_var, "")
    if not path or not Path(path).exists():
        raise FileNotFoundError(f"checkpoint for {arch} missing — set {env_var}")
    return WeatherHermiteLightningModule.load_from_checkpoint(
        path, map_location="cpu"
    ).eval()


def compute_bias(model, ds, ch_idx: int, device: str = "cuda") -> np.ndarray:
    """Returns the time-averaged (model - target) bias map at τ=3.

    For Bilinear (model is None) we use (1-τ/Δ)·x0 + (τ/Δ)·xT.
    """
    H, W = ds.lat_size, ds.lon_size
    acc = np.zeros((H, W), dtype=np.float64)
    n = 0
    if model is not None:
        model = model.to(device)
    delta_t = 6.0
    for i, sample in enumerate(ds):
        if i >= MAX_SAMPLES:
            break
        x0 = sample["x0"][:24, :, :].unsqueeze(0).to(device)
        xT = sample["xT"][:24, :, :].unsqueeze(0).to(device)
        target = sample["target"][:24, :, :].unsqueeze(0)
        # τ index inside the multi-tau target tensor
        # `target` is shape (1, n_tau, C, H, W) for some datasets; handle both
        if target.ndim == 5:
            tau_axis = TAU_HOURS - 1  # τ=1 is index 0
            target = target[:, tau_axis]
        target = target.to(device)
        tau_in = torch.tensor([[float(TAU_HOURS)]], device=device)
        with torch.no_grad():
            if model is None:
                w = float(TAU_HOURS) / delta_t
                pred = (1.0 - w) * x0 + w * xT
            else:
                static = sample.get("static")
                cond = sample.get("cond", torch.zeros(1, 1, device=device))
                if static is not None:
                    static = static.to(device)
                pred_full, _ = model.model(x0, xT, tau_in, cond, static=static)
                pred = pred_full
        diff = (pred[0, ch_idx] - target[0, ch_idx]).cpu().numpy()
        acc += diff
        n += 1
    return (acc / max(1, n)).astype(np.float32)


def plot_grid(biases_by_arch_channel, out_pdf: Path) -> None:
    fig, axes = plt.subplots(len(ARCHS), len(PANEL_CHANNELS),
                              figsize=(16, 11), sharex=True, sharey=True)
    # Per-column symmetric vmin/vmax (max abs across architectures).
    col_extremes = []
    for ci, (ch_key, _, _) in enumerate(PANEL_CHANNELS):
        arr_list = [biases_by_arch_channel[(a, ch_key)]
                    for a in ARCHS if (a, ch_key) in biases_by_arch_channel]
        if not arr_list:
            col_extremes.append(1.0); continue
        col_extremes.append(max(np.nanmax(np.abs(a)) for a in arr_list))

    for ri, arch in enumerate(ARCHS):
        for ci, (ch_key, ch_label, unit) in enumerate(PANEL_CHANNELS):
            ax = axes[ri, ci]
            arr = biases_by_arch_channel.get((arch, ch_key))
            if arr is None:
                ax.set_visible(False); continue
            vlim = col_extremes[ci]
            norm = TwoSlopeNorm(vcenter=0, vmin=-vlim, vmax=vlim)
            H, W = arr.shape
            lats = np.linspace(89.75, -89.75, H)
            lons = np.linspace(-179.75, 179.75, W)
            im = ax.pcolormesh(lons, lats, arr, cmap="RdBu_r",
                                norm=norm, shading="auto")
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xticks([-180, -90, 0, 90, 180])
            ax.set_yticks([-90, -45, 0, 45, 90])
            if ri == 0:
                ax.set_title(f"{ch_label}  [{unit}]", fontsize=12,
                             fontweight="bold")
            if ci == 0:
                ax.set_ylabel(arch, fontsize=13, fontweight="bold")
            if ri == len(ARCHS) - 1:
                cbar = fig.colorbar(im, ax=ax, orientation="horizontal",
                                     pad=0.18, shrink=0.8, fraction=0.04)
                cbar.ax.tick_params(labelsize=8)
            else:
                fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.suptitle(
        f"Time-averaged spatial bias (model − ERA5) at " r"$\tau$"
        f" = {TAU_HOURS}\,h on 2020", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_pdf}", flush=True)


def main() -> None:
    out_pdf = (Path("/home/jovyan/dsuhoi/weather_time_interpolation/paper/figs")
                / "fig_bias_maps_4arch.pdf")

    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=[2020], max_tau_hours=6, samples_per_date=4, train=False,
        static_path="data/static_features_0p5.pt",
        stats_path="data/json_stats_0p5.nc",
        surface_stats_path="data/surface_stats_0p5.json",
    )
    ds = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)

    ckpts = {
        "Bilinear":     None,
        "S-DYff":       ("CKPT_SDYFF",  "S-DYff"),
        "ATM-VFI":      ("CKPT_ATMVFI", "ATM-VFI"),
        "WeatherBridge":("CKPT_WBRIDGE","WeatherBridge"),
    }

    models = {}
    for arch in ARCHS:
        if arch == "Bilinear":
            models[arch] = None
        else:
            env_var, _ = ckpts[arch]
            models[arch] = load_ckpt(env_var, arch)
            print(f"loaded {arch}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    biases = {}
    for arch in ARCHS:
        for ch_key, _, _ in PANEL_CHANNELS:
            print(f"computing bias  arch={arch}  channel={ch_key} …", flush=True)
            biases[(arch, ch_key)] = compute_bias(
                models[arch], ds, CH_INDEX[ch_key], device=device,
            )

    plot_grid(biases, out_pdf)


if __name__ == "__main__":
    main()
