#!/usr/bin/env python3
"""Plot per-hour RMSE for each channel × each method (6-hour interpolation).

Loads JSONs from:
  metrics/eval_6h_2020_paper_leaderboard/     (4 ML models)
  metrics/eval_0p5_2020_numeric/  (numerical baselines)
Output: paper/per_hour_curves/ — 6 group PNGs (T, U, V, Q, Z, surface) and 1 mosaic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ML_MODELS = [
    ("dcae_skip_0p5_6yr_pad",    "DC-AE Skip 6yr",     "#1b9e77", "o", 2.0),
    ("modafno_0p5_6yr",          "ModAFNO 6yr",         "#d95f02", "s", 1.2),
    ("sdyff_dyffusion_0p5_6yr",  "S-DYff DYffusion 6yr","#7570b3", "d", 1.2),
    ("fuxi_0p5_6yr",             "FuXi SwinV2 6yr",     "#e7298a", "^", 1.2),
]
NUM_BASELINES = [
    ("bilinear",          "Bilinear",          "#666666", "x", 1.2),
    ("hermite_advection", "Hermite-advection", "#1f77b4", "v", 1.2),
]

PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS = ["t2m", "u10", "v10", "mslp"]
HOURS = [1, 2, 3, 4, 5]

UNITS = {
    "T": "(K)", "U": "(m/s)", "V": "(m/s)", "Q": "(kg/kg)", "Z": "(m²/s²)",
    "t2m": "(K)", "u10": "(m/s)", "v10": "(m/s)",
    "mslp": "(Pa)", "sst": "(K)", "tcc": "(-)", "tcwv": "(kg/m²)",
}


def load_jsons(fast_dir: Path, num_dir: Path):
    data = {}
    for stem, disp, _, _, _ in ML_MODELS:
        p = fast_dir / f"{stem}.json"
        if p.exists():
            data[disp] = json.loads(p.read_text())
    for stem, disp, _, _, _ in NUM_BASELINES:
        p = num_dir / f"{stem}.json"
        if p.exists():
            data[disp] = json.loads(p.read_text())
    return data


def get_per_hour_rmse(d: dict, channel: str):
    vals = []
    for h in HOURS:
        v = d["per_hour"].get(str(h), {}).get("model", {}).get(f"rmse_{channel}")
        vals.append(float(v) if v is not None else float("nan"))
    return np.array(vals)


def make_group_plot(data, channels, group_title, out_path: Path, log_y=False):
    """Plot one panel per channel in group, lines = methods."""
    n = len(channels)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 2.9), sharex=True)
    axes = np.array(axes).reshape(rows, cols)
    all_methods = [(d, c, m, w) for stem, d, c, m, w in ML_MODELS + NUM_BASELINES]
    for ax_idx, ch in enumerate(channels):
        r, c = divmod(ax_idx, cols)
        ax = axes[r, c]
        for disp, color, marker, lw in all_methods:
            if disp not in data:
                continue
            y = get_per_hour_rmse(data[disp], ch)
            ax.plot(HOURS, y, marker=marker, color=color, linewidth=lw,
                    markersize=5, label=disp)
        unit = UNITS.get(ch[0] if ch[0] in UNITS else ch, "")
        ax.set_title(f"{ch} {unit}", fontsize=10)
        ax.set_xticks(HOURS)
        ax.grid(True, alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        if c == 0:
            ax.set_ylabel("RMSE (normalised)")
        if r == rows - 1:
            ax.set_xlabel("Interior hour h")
    # Hide unused axes
    for ax_idx in range(n, rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r, c].set_visible(False)
    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
               bbox_to_anchor=(0.5, 1.02), fontsize=9, frameon=False)
    fig.suptitle(group_title, y=1.06, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--num-dir",  default="metrics/eval_0p5_2020_numeric")
    ap.add_argument("--out-dir",  default="paper/per_hour_curves")
    args = ap.parse_args()
    data = load_jsons(Path(args.fast_dir), Path(args.num_dir))
    print(f"loaded {len(data)} methods")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Per pressure-level group: 4 levels × group
    for var in PL_VARS:
        channels = [f"{var}{lvl}" for lvl in PL_LEVELS]
        title = {
            "T": "Temperature (T)",
            "U": "Zonal wind (U)",
            "V": "Meridional wind (V)",
            "Q": "Specific humidity (Q)",
            "Z": "Geopotential (Z)",
        }[var]
        make_group_plot(data, channels, f"{title} — per-hour RMSE",
                        out_root / f"per_hour_{var}.png")

    # Surface group
    make_group_plot(data, SURF_VARS, "Surface variables — per-hour RMSE",
                    out_root / "per_hour_surface.png")

    # Mosaic with the complete 24-field paper protocol.
    all_channels = [f"{v}{l}" for v in PL_VARS for l in PL_LEVELS] + SURF_VARS
    cols = 4
    rows = (len(all_channels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.4), sharex=True)
    axes = np.array(axes).reshape(rows, cols)
    all_methods = [(d, c, m, w) for stem, d, c, m, w in ML_MODELS + NUM_BASELINES]
    for ax_idx, ch in enumerate(all_channels):
        r, c = divmod(ax_idx, cols)
        ax = axes[r, c]
        for disp, color, marker, lw in all_methods:
            if disp not in data:
                continue
            y = get_per_hour_rmse(data[disp], ch)
            ax.plot(HOURS, y, marker=marker, color=color, linewidth=lw, markersize=4)
        ax.set_title(ch, fontsize=9)
        ax.set_xticks(HOURS)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if c == 0:
            ax.set_ylabel("RMSE", fontsize=8)
        if r == rows - 1:
            ax.set_xlabel("h", fontsize=8)
    for ax_idx in range(len(all_channels), rows * cols):
        r, c = divmod(ax_idx, cols)
        axes[r, c].set_visible(False)
    legend_handles = [
        plt.Line2D([], [], color=clr, marker=mk, linewidth=lw, label=disp)
        for disp, clr, mk, lw in
        [(d, c, m, w) for _, d, c, m, w in ML_MODELS + NUM_BASELINES]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), fontsize=10, frameon=False)
    fig.suptitle("Per-hour RMSE for all 24 fields — 0.5° models + numerical baselines",
                 y=1.005, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_root / "per_hour_all_24ch_mosaic.png",
                dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_root / 'per_hour_all_24ch_mosaic.png'}")


if __name__ == "__main__":
    main()
