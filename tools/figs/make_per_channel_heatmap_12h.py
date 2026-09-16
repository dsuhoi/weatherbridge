#!/usr/bin/env python3
"""Surrogate for Fig 7 bias maps using JSON-only data.

The true Fig 7 (per-pixel bias maps) needs model+memmap on the cluster
(see scripts/run_fig7_bias_maps_12h.sh). In the meantime this script
produces a complementary figure that IS computable offline:

  figs/fig_per_channel_heatmap_12h.{png,pdf}
    - Rows: 7 models
    - Cols: 24 channels
    - Cells: log10(rmse_norm_<channel>) averaged across τ ∈ {1..11}
    - Color: viridis (low = blue, high = yellow)
    - Annotation: leader-row outlined, ModAFNO catastrophic cells boxed

The heatmap exposes per-channel weaknesses (e.g. ATM-VFI's wind-PL
dominance, ModAFNO's catastrophic V_PL failure) in a single panel
without needing pixel-level data.

Output: figs/fig_per_channel_heatmap_12h.{png, pdf}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]

MODEL_DISPLAY_ORDER = [
    ("ATM-VFI_3yr_12h_fibo",          "ATM-VFI"),
    ("DC-AE_NoSkip_3yr_12h_fibo",     "DC-AE NoSkip 3yr"),
    ("WeatherDCAE_NoSkip_6yr_12h",    "WeatherDCAE 6yr*"),
    ("DC-AE_Skip_3yr_12h_fibo",       "DC-AE Skip 3yr"),
    ("FuXi_3yr_12h_fibo",             "FuXi"),
    ("SDyff_3yr_12h_fibo",            "S-DYff"),
    ("ModAFNO_3yr_12h_fibo",          "ModAFNO"),
]


def _avg_per_channel(d: dict) -> List[float]:
    out = []
    for c in CHANNELS_ORDER:
        vals = []
        for tau_s, by_m in d["per_tau"].items():
            v = by_m.get("model", {}).get(f"rmse_norm_{c}")
            if v is not None:
                vals.append(float(v))
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="metrics/eval_12h_2020_ep10")
    ap.add_argument("--out-base", default="figs/fig_per_channel_heatmap_12h")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    grid = []
    row_labels = []
    for stem, disp in MODEL_DISPLAY_ORDER:
        p = metrics_dir / f"{stem}.json"
        if not p.exists():
            continue
        d = json.load(open(p))
        grid.append(_avg_per_channel(d))
        row_labels.append(disp)

    arr = np.array(grid, dtype=np.float64)
    # log10 to compress dynamic range (ModAFNO V_PL row would otherwise
    # saturate the colormap)
    log_arr = np.log10(arr + 1e-3)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    im = ax.imshow(log_arr, cmap="viridis", aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(len(CHANNELS_ORDER)))
    ax.set_xticklabels(CHANNELS_ORDER, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    cbar = plt.colorbar(im, ax=ax, pad=0.012)
    cbar.set_label(r"$\log_{10}$ RMSE$_{\mathrm{norm}}$ (avg over $\tau \in \{1..11\}$)",
                   fontsize=9)
    # Cell text
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isnan(v):
                continue
            tc = "w" if log_arr[i, j] < log_arr.mean() else "k"
            ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                    fontsize=6.5, color=tc)
    # Class boundary lines
    for x in (3.5, 7.5, 11.5, 15.5, 19.5):
        ax.axvline(x, color="white", linewidth=0.8, alpha=0.55)

    ax.set_title("Per-channel normalised RMSE — 12 h interpolation leaderboard "
                 "(strict ep10, 2020 ERA5 0.5°)", fontsize=10)
    plt.tight_layout()

    out_png = Path(args.out_base + ".png")
    out_pdf = Path(args.out_base + ".pdf")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.savefig(out_pdf)
    plt.close()
    print(f"saved {out_png}")
    print(f"saved {out_pdf}")


if __name__ == "__main__":
    main()
