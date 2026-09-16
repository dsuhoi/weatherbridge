#!/usr/bin/env python3
"""Plot E(k) vs k for selected channels, all methods + ground truth on one
log-log axis. Output: paper/energy_spectra/<channel>.png plus a mosaic.

Reads .npz files from metrics/energy_spectra_0p5_2020/ (output of
energy_spectra.py) and overlays them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METHOD_STYLE = {
    "dcae_skip_AC":          ("DC-AE Skip +A+C (ours)", "#d62728", "-",  2.4),
    "dcae_skip":             ("DC-AE Skip (no novelties)", "#1b9e77", "-", 1.6),
    "dcae_skip_no_skip":     ("DC-AE no-skip ablation",   "#999999", "--", 1.2),
    "modafno":               ("ModAFNO",                  "#d95f02", ":",  1.2),
    "sdyff_dyffusion":       ("S-DYff DYffusion",         "#7570b3", ":",  1.2),
    "fuxi_swinv2":           ("SwinV2",                   "#e7298a", ":",  1.2),
    "bilinear":              ("Bilinear",                 "#bbbbbb", "-.",  1.0),
    "hermite_advection":     ("Hermite-advection",        "#1f77b4", "-.",  1.0),
    "ground_truth":          ("Ground truth (ERA5)",      "#000000", "-",  2.0),
}

PLOT_CHANNELS = ["T850", "U850", "V850", "Q850", "Z500",
                 "t2m", "u10", "tcwv"]


def load_all(npz_dir: Path):
    out = {}
    for p in sorted(npz_dir.glob("*.npz")):
        stem = p.stem
        if stem not in METHOD_STYLE:
            continue
        d = np.load(p, allow_pickle=True)
        out[stem] = dict(
            k=d["k_centers"], Ek=d["pred_Ek"],
            channels=[str(c) for c in d["channel_names"]],
            gt_Ek=d["gt_Ek"] if "gt_Ek" in d.files else None,
        )
    return out


def plot_channel(data, channel, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    # Plot ground truth if present in any record
    gt_done = False
    for method, d in data.items():
        if channel not in d["channels"]:
            continue
        ci = d["channels"].index(channel)
        disp, color, ls, lw = METHOD_STYLE[method]
        # mask DC near zero to avoid log(0)
        ek = d["Ek"][ci]
        mask = (d["k"] > 0) & (ek > 0)
        ax.loglog(d["k"][mask], ek[mask], color=color, ls=ls, lw=lw, label=disp)
        if d.get("gt_Ek") is not None and not gt_done:
            gt = d["gt_Ek"][ci]
            m2 = (d["k"] > 0) & (gt > 0)
            ax.loglog(d["k"][m2], gt[m2], color="#000000", ls="-", lw=2.0,
                      label="Ground truth (ERA5)")
            gt_done = True
    # k^-3 reference slope
    k_ref = np.logspace(-2, -0.5, 50)
    e_ref = 1e-2 * (k_ref / k_ref[0]) ** (-3)
    ax.loglog(k_ref, e_ref, color="#cccccc", ls=":", lw=1.2)
    ax.text(k_ref[-1], e_ref[-1], "  $k^{-3}$", color="#888888", fontsize=9)
    ax.set_xlabel("Wavenumber $k$ (cycles / grid)")
    ax.set_ylabel(r"Power spectral density $E(k)$")
    ax.set_title(f"Energy spectrum — {channel}", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="metrics/energy_spectra_0p5_2020")
    ap.add_argument("--out-dir", default="paper/energy_spectra")
    args = ap.parse_args()
    data = load_all(Path(args.npz_dir))
    if not data:
        raise SystemExit(f"no .npz under {args.npz_dir}")
    out_dir = Path(args.out_dir)
    for ch in PLOT_CHANNELS:
        plot_channel(data, ch, out_dir / f"{ch}.png")
    # Mosaic 2x4
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    gt_legend_added = False
    for i, ch in enumerate(PLOT_CHANNELS):
        r, c = divmod(i, 4)
        ax = axes[r, c]
        gt_done = False
        for method, d in data.items():
            if ch not in d["channels"]:
                continue
            ci = d["channels"].index(ch)
            disp, color, ls, lw = METHOD_STYLE[method]
            ek = d["Ek"][ci]
            mask = (d["k"] > 0) & (ek > 0)
            ax.loglog(d["k"][mask], ek[mask], color=color, ls=ls, lw=lw,
                      label=disp if not gt_legend_added else None)
            if d.get("gt_Ek") is not None and not gt_done:
                gt = d["gt_Ek"][ci]
                m2 = (d["k"] > 0) & (gt > 0)
                ax.loglog(d["k"][m2], gt[m2], color="#000000", ls="-", lw=1.6,
                          label="GT (ERA5)" if not gt_legend_added else None)
                gt_done = True
        ax.set_title(ch, fontsize=10)
        ax.grid(True, which="both", alpha=0.2)
        if r == 1:
            ax.set_xlabel("k")
        if c == 0:
            ax.set_ylabel("E(k)")
        gt_legend_added = True
    fig.suptitle("Energy spectra — interior states ($\\tau \\in 1..5$h), 2020 test",
                 y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "mosaic.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_dir / 'mosaic.png'}")


if __name__ == "__main__":
    main()
