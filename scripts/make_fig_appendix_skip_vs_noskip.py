"""WeatherDCAE Skip vs NoSkip spectral ablation, unified 6 h horizon.

The pair uses the same 2017--2019 data, 24-field protocol, optimiser,
backbone widths (64/128/256), depth (3/3/3), and latent width (256).
The Skip arm adds only two zero-initialised scalar lateral gates.

Within this controlled single-realisation comparison, the architectural
difference is the U-Net-style gated zero-init skip path. Both use the same
evaluation windows and spherical-harmonic implementation.

X-axis: angular wavelength λ = 360°/ℓ from 4.5° (synoptic) to 2.0° (near grid).
Y-axis: power-spectrum ratio model/ERA5 (1.0 = perfect, <1 over-smoothing,
>1 over-generation).

Output: paper/figs/fig_appendix_skip_vs_noskip.pdf
"""
from __future__ import annotations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator, NullFormatter
import numpy as np

from paper_plot_style import MODEL_COLORS

ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = ROOT / "metrics" / "journal_spectra" / "6h_2020"
OUT_PDF = ROOT / "paper" / "figs" / "fig_appendix_skip_vs_noskip.pdf"

CHANNELS = [("u10", "U10 (zonal wind)"), ("V850", "V850 (meridional wind)")]
TAUS = [2, 3]
ELL_LO, ELL_HI = 80, 180
Y_LO, Y_HI = 0.6, 1.05

MODELS = {
    "weatherdcae_skip": (
        "WeatherDCAE-Skip",
        MODEL_COLORS["WeatherDCAE-Skip"],
        "-",
        1.8,
    ),
    "weatherdcae_14m": (
        "WeatherDCAE-14M",
        MODEL_COLORS["WeatherDCAE-14M"],
        "-",
        1.8,
    ),
}


def _load(stem: str, tau: int):
    p = NPZ_DIR / f"{stem}_tau{tau}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    chs = [str(c) for c in d["channel_names"]]
    return d["ell"], d["pred_El"], d["gt_El"], chs


def _ratio(stem: str, tau: int, ch: str):
    out = _load(stem, tau)
    if out is None:
        return None, None
    ell, pred, gt, chs = out
    if ch not in chs:
        return None, None
    j = chs.index(ch)
    mask = (ell >= ELL_LO) & (ell <= ELL_HI)
    return ell[mask], pred[j, mask] / np.clip(gt[j, mask], 1e-30, None)


def main() -> None:
    fig, axes = plt.subplots(len(CHANNELS), len(TAUS),
                             figsize=(10.5, 4.6), sharex=True, sharey=True)
    xticks_ell = [80, 100, 130, 180]
    xticks_deg = [f"{360.0 / v:.1f}°" for v in xticks_ell]

    for ri, (ch_key, ch_label) in enumerate(CHANNELS):
        for ci, tau in enumerate(TAUS):
            ax = axes[ri, ci]
            ax.axvspan(ELL_LO, 120, color="#dde7ff", alpha=0.35, lw=0)
            ax.axvspan(120, 150, color="#fff2cc", alpha=0.35, lw=0)
            ax.axvspan(150, ELL_HI, color="#ffd9d9", alpha=0.40, lw=0)
            ax.axhline(1.0, color="black", lw=1.5)
            for stem, (disp, color, ls, lw) in MODELS.items():
                ell, ratio = _ratio(stem, tau, ch_key)
                if ratio is None:
                    continue
                ax.plot(ell, ratio, color=color, ls=ls, lw=lw, label=disp)
            ax.set_xscale("log")
            ax.set_xlim(ELL_LO, ELL_HI)
            ax.set_ylim(Y_LO, Y_HI)
            ax.yaxis.set_major_locator(MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(MultipleLocator(0.025))
            ax.xaxis.set_major_locator(FixedLocator(xticks_ell))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xticklabels(xticks_deg)
            ax.grid(True, which="major", alpha=0.35, linestyle="-", lw=0.5)
            ax.grid(True, which="minor", alpha=0.18, linestyle=":", lw=0.4)
            ax.set_title(f"{ch_label}, " + r"$\tau$" + f"={tau} h",
                         fontsize=10, fontweight="bold")
            if ri == len(CHANNELS) - 1:
                ax.set_xlabel(
                    r"$\lambda = 360^\circ/\ell$  (synoptic $\to$ meso)",
                    fontsize=9,
                )
            if ci == 0:
                ax.set_ylabel("Power ratio model/ERA5", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "WeatherDCAE Skip vs NoSkip (matched backbone, 3 yr, 6 h horizon)",
        fontsize=11, fontweight="bold", y=0.99,
    )
    fig.legend(handles, labels,
               loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.02), fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.90, hspace=0.34, wspace=0.07,
                        left=0.07, right=0.99, bottom=0.18)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
