#!/usr/bin/env python3
"""Plot SH-energy-spectra comparison for the S-DYff ensemble experiment.

Reads ``metrics/sh_spectra_ens/<prefix>_<ch>.npz`` (one NPZ per channel) and
produces a 2 x 4 panel figure: each channel shows log10 E(l) vs l for three
variants -- ground truth, single MC-dropout sample, ensemble mean -- with a
dashed vertical line at l = ell_lo (HF cutoff). Saves both ``.pdf`` and
``.png``.

Companion to ``sh_spectra_ens_sdyff.py``. Run AFTER the eval finishes.

Usage::

    python tools/eval/plot_sh_ens_blur.py \\
        --npz-dir metrics/sh_spectra_ens \\
        --prefix sdyff_24ch_6yr_ens16_tau3 \\
        --channels t2m,u10,v10,mslp,T850,U850,Q850,Z700 \\
        --out figs/fig_sh_ens_blur
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", default="metrics/sh_spectra_ens")
    ap.add_argument("--prefix", default="sdyff_24ch_6yr_ens16_tau3")
    ap.add_argument("--channels",
                    default="t2m,u10,v10,mslp,T850,U850,Q850,Z700")
    ap.add_argument("--out", default="figs/fig_sh_ens_blur",
                    help="Path stem (no extension) -- writes .pdf and .png.")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    assert len(channels) <= 8, "layout is 2x4; pass <=8 channels"

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    axes = axes.flatten()
    handles = None
    for i, ch in enumerate(channels):
        ax = axes[i]
        p = npz_dir / f"{args.prefix}_{ch}.npz"
        if not p.exists():
            ax.text(0.5, 0.5, f"missing\n{p.name}", ha="center", va="center")
            ax.set_title(ch)
            continue
        d = np.load(p, allow_pickle=True)
        ell = d["ell"]
        gt = d["gt_spectrum"]
        sg = d["single_spectrum"]
        en = d["ens_mean_spectrum"]
        ell_lo = int(d["ell_lo"])
        r_gt = float(d["R_HF_gt"])
        r_sg = float(d["R_HF_single"])
        r_en = float(d["R_HF_ens"])

        # Skip l=0 in log plot to avoid log(0); start at 1.
        m = ell >= 1
        h1, = ax.plot(ell[m], np.log10(np.maximum(gt[m], 1e-30)),
                      color="k", lw=1.6, label="ground truth")
        h2, = ax.plot(ell[m], np.log10(np.maximum(sg[m], 1e-30)),
                      color="tab:blue", lw=1.3, label="single MC sample")
        h3, = ax.plot(ell[m], np.log10(np.maximum(en[m], 1e-30)),
                      color="tab:red", lw=1.3, label=f"ens-mean (N={int(d['n_ensemble'])})")
        ax.axvline(ell_lo, color="grey", ls="--", lw=1.0, alpha=0.8)
        ax.set_title(
            f"{ch}  |  $R_{{HF}}$: gt={r_gt:.3f}, single={r_sg:.3f}, ens={r_en:.3f}",
            fontsize=10,
        )
        ax.grid(True, ls=":", alpha=0.4)
        if i // 4 == 1:
            ax.set_xlabel("spherical degree $\\ell$")
        if i % 4 == 0:
            ax.set_ylabel("$\\log_{10} E(\\ell)$")
        if handles is None:
            handles = [h1, h2, h3]

    # Hide unused axes (if <8 channels)
    for j in range(len(channels), 8):
        axes[j].axis("off")

    fig.suptitle(
        f"SH angular power spectra: ground truth vs single MC sample vs ensemble mean "
        f"(S-DYff, $\\tau=3$h)",
        fontsize=11,
    )
    if handles is not None:
        fig.legend(handles, [h.get_label() for h in handles],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02),
                   frameon=False)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out.with_suffix(".pdf")
    png_path = out.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")


if __name__ == "__main__":
    main()
