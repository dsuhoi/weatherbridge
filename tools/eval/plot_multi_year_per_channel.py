#!/usr/bin/env python3
"""Per-channel grouped bar plots of RMSE on the 2020+2021 multi-year test set
for all evaluated methods (DC-AE Skip, DC-AE No-skip, ModAFNO, S-DYff, SwinV2,
bilinear, hermite-advection).

Reads JSONs from metrics/multi_year_2020_2021/ produced by
tools/eval/multi_year_metrics.py.

Output: paper/multi_year_per_channel/{pl_<var>,surface,mosaic}.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (json stem -> (display name, color, hatch))
METHODS = [
    ("dcae_skip_0p5_6yr_pad",   "DC-AE Skip (full)",      "#1b9e77", ""),
    ("dcae_noskip",             "DC-AE No-skip (ours)",   "#d62728", ""),
    ("modafno_0p5_6yr",         "ModAFNO",                "#d95f02", ""),
    ("sdyff_dyffusion_0p5_6yr", "S-DYff DYffusion",       "#7570b3", ""),
    ("fuxi_0p5_6yr",            "SwinV2",                 "#e7298a", ""),
    ("bilinear",                "Bilinear",               "#888888", "//"),
    ("hermite_advection",       "Hermite-advection",      "#1f77b4", "//"),
]

PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS = ["t2m", "u10", "v10", "mslp"]

UNIT = {"T": "K", "U": "m/s", "V": "m/s", "Q": "kg/kg", "Z": "m²/s²",
        "t2m": "K", "u10": "m/s", "v10": "m/s",
        "mslp": "Pa", "sst": "K", "tcc": "-", "tcwv": "kg/m²"}


def load_methods(npz_dir: Path):
    data = {}
    for stem, disp, color, hatch in METHODS:
        p = npz_dir / f"{stem}.json"
        if not p.exists():
            print(f"  warn: missing {p.name}")
            continue
        d = json.loads(p.read_text())
        rmse = d.get("rmse_phys_per_channel", {})
        data[disp] = dict(rmse=rmse, color=color, hatch=hatch)
    return data


def group_plot(data, channels, group_title, ylabel, out_path, log_y=False):
    methods_present = [(m, d) for m, d in data.items()
                       if any(c in d["rmse"] for c in channels)]
    n_methods = len(methods_present)
    n_chs = len(channels)
    bar_w = 0.85 / n_methods
    x = np.arange(n_chs)
    fig, ax = plt.subplots(figsize=(max(6, n_chs * 1.5), 4.5))
    for i, (m, d) in enumerate(methods_present):
        vals = [d["rmse"].get(c, np.nan) for c in channels]
        bars = ax.bar(x + i * bar_w - 0.425, vals, bar_w,
                      color=d["color"], edgecolor="black", linewidth=0.5,
                      hatch=d["hatch"], label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=0)
    ax.set_ylabel(ylabel)
    ax.set_title(group_title, fontsize=12, fontweight="bold")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def relative_plot(data, channels, ref_method, group_title, out_path):
    """Plot RMSE relative to ref_method (%)."""
    methods_present = [(m, d) for m, d in data.items()
                       if any(c in d["rmse"] for c in channels) and m != ref_method]
    ref = data[ref_method]["rmse"]
    n_methods = len(methods_present)
    n_chs = len(channels)
    bar_w = 0.85 / n_methods
    x = np.arange(n_chs)
    fig, ax = plt.subplots(figsize=(max(6, n_chs * 1.5), 4.5))
    for i, (m, d) in enumerate(methods_present):
        vals = [(d["rmse"].get(c, np.nan) - ref.get(c, np.nan)) / ref.get(c, np.nan) * 100
                for c in channels]
        ax.bar(x + i * bar_w - 0.425, vals, bar_w,
               color=d["color"], edgecolor="black", linewidth=0.5,
               hatch=d["hatch"], label=m)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=0)
    ax.set_ylabel(f"Δ RMSE vs {ref_method} (%)")
    ax.set_title(group_title + f" — relative to {ref_method}",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="upper left", ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="metrics/multi_year_2020_2021")
    ap.add_argument("--out-dir", default="paper/multi_year_per_channel")
    ap.add_argument("--ref", default="DC-AE Skip (full)",
                    help="reference method for relative plots")
    args = ap.parse_args()
    data = load_methods(Path(args.in_dir))
    if not data:
        raise SystemExit("no JSONs found")
    out = Path(args.out_dir)

    # PL groups
    for v in PL_VARS:
        chs = [f"{v}{lvl}" for lvl in PL_LEVELS]
        group_plot(data, chs,
                   {"T": "Temperature", "U": "Zonal wind", "V": "Meridional wind",
                    "Q": "Specific humidity", "Z": "Geopotential"}[v]
                   + f" — RMSE (2020+2021)",
                   ylabel=f"RMSE ({UNIT[v]})",
                   out_path=out / f"pl_{v}.png",
                   log_y=(v == "Z"))
        relative_plot(data, chs,
                      args.ref,
                      f"{v} per-level",
                      out / f"pl_{v}_relative.png")

    # Surface group
    group_plot(data, SURF_VARS,
               "Surface variables — RMSE (2020+2021)",
               ylabel="RMSE (physical)",
               out_path=out / "surface.png",
               log_y=True)
    relative_plot(data, SURF_VARS,
                  args.ref,
                  "Surface variables",
                  out / "surface_relative.png")

    # Mosaic: complete 24-field paper protocol.
    all_chs = [f"{v}{lvl}" for v in PL_VARS for lvl in PL_LEVELS] + SURF_VARS
    # 6 rows x 4 cols mosaic of bar charts
    rows = 6
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 2.8))
    axes = np.array(axes).reshape(rows, cols)
    methods_present = list(data.items())
    for idx, ch in enumerate(all_chs):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        vals = [(m, data[m]["rmse"].get(ch, np.nan)) for m, d in methods_present]
        names = [m for m, _ in vals]
        ys = [v for _, v in vals]
        colors = [data[m]["color"] for m in names]
        ax.bar(range(len(ys)), ys, color=colors, edgecolor="black", linewidth=0.4)
        unit = UNIT.get(ch[0] if ch[0] in "TUVQZ" else ch, "")
        ax.set_title(f"{ch} ({unit})", fontsize=9)
        ax.set_xticks([])
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25, axis="y")
    for idx in range(len(all_chs), rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].set_visible(False)
    # Legend at the bottom
    handles = [plt.Rectangle((0, 0), 1, 1, color=data[m]["color"],
                             ec="black", lw=0.4) for m in methods_present]
    fig.legend(handles, [m for m, _ in methods_present], loc="lower center",
               ncol=4, fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-field RMSE on 2020+2021 test set, all 24 fields",
                 y=1.005, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path = out / "mosaic_all_24.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
