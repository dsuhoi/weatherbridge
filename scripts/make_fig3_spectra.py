"""Regenerate the 12 h spherical-harmonic comparison used in the paper.

Plots the spherical-harmonic power ratio model/ERA5 for lower-tropospheric
specific humidity (Q850) and meridional wind (V850) at a seen hour (τ=5)
and an unseen hour (τ=6). A ratio of
1.0 (black solid) indicates spectral fidelity with ERA5; values below 1
indicate over-smoothing, above 1 indicate over-generation.

X-axis: angular wavelength in degrees of arc (λ_deg = 360°/ℓ), running
from ~4.5° down to 2° — i.e. synoptic-to-mesoscale on a 0.5° lat-lon
grid. The underlying spherical-harmonic degree ℓ ranges over [80, 180]
under latitude-strip area quadrature on the source 2x2 block-average grid.

Inputs (read-only):
  metrics/journal_spectra/12h_2020/{model}_tau{5,6}.npz
      produced by the canonical journal spectral queue on the actual
      WeatherBench2 block-average latitude grid.

Output:
  paper/images/fig_spectra_ratio_hard.pdf
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, MultipleLocator, NullFormatter, NullLocator
from paper_plot_style import MODEL_COLORS, MODEL_LINESTYLES
from weather_time_interp.grid import WB2_BLOCK_GRID_NAME

ROOT = Path(__file__).resolve().parents[1]
NPZ_DIR = Path(
    os.environ.get(
        "WTI_JOURNAL_SPECTRA_ROOT",
        str(ROOT / "metrics" / "journal_spectra_v6"),
    )
) / "12h_2020"
OUT_PDF = ROOT / "paper" / "images" / "fig_spectra_ratio_hard.pdf"

CHANNELS = [("Q850", "Q850"), ("V850", "V850")]
TAUS = [5, 6]
PANEL_CONFIGS = [
    (channel_key, channel_label, tau)
    for channel_key, channel_label in CHANNELS
    for tau in TAUS
]

MODELS = {
    "linear": ("Linear Interp.", 0.9),
    "fuxi": ("SwinV2", 0.9),
    "modafno": ("ModAFNO", 0.9),
    "sdyff": ("S-DYff", 1.0),
    "pixelattn_vfi": ("PixelAttn-VFI", 1.0),
    "weatherdcae_14m": ("WeatherDCAE-14M", 1.1),
    "flow_spectral": ("WeatherBridge", 1.4),
}

ELL_LO, ELL_HI = 80, 180
Y_LO, Y_HI = 0.35, 1.05


def _ell_to_deg(ell):
    """Angular wavelength in degrees of arc: λ_deg = 360°/ℓ."""
    return 360.0 / np.asarray(ell, dtype=float)


def _load(stem: str, tau: int):
    p = NPZ_DIR / f"{stem}_tau{tau}.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"missing canonical spectral metric {p}; sync the completed queue"
        )
    with np.load(p, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata_json"].item()))
        if (
            metadata.get("schema_version") != 6
            or metadata.get("sht_grid")
            != WB2_BLOCK_GRID_NAME + "_latitude_strip_area_sht"
            or metadata.get("sample_strategy") != "all_valid_anchor_windows"
        ):
            raise ValueError(f"{p} is not a dense corrected spectral artifact")
        return (
            data["ell"].copy(),
            data["pred_El"].copy(),
            data["gt_El"].copy(),
            [str(channel) for channel in data["channel_names"]],
        )


def _ratio(stem: str, tau: int, ch: str):
    ell, pred, gt, chs = _load(stem, tau)
    if ch not in chs:
        raise KeyError(f"{ch} missing from {stem}_tau{tau}.npz")
    j = chs.index(ch)
    p = pred[j]
    g = gt[j]
    mask = (ell >= ELL_LO) & (ell <= ELL_HI)
    eps = 1e-30
    return ell[mask], (p[mask] / np.clip(g[mask], eps, None))


def main() -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 6.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # X-axis ticks placed on ℓ but LABELLED in degrees (λ_deg = 360°/ℓ).
    # Sparse spacing — log-scale crowding made adjacent labels collide.
    xticks_ell = [80, 100, 130, 180]
    xticks_deg = [f"{360.0 / value:.1f}°" for value in xticks_ell]

    for ci, (chan_key, chan_label, tau) in enumerate(PANEL_CONFIGS):
        row, column = divmod(ci, 2)
        ax = axes[row, column]

        # Synoptic, meso-alpha, and near-grid spectral bands.
        ax.axvspan(ELL_LO, 120, color="#dde7ff", alpha=0.35, lw=0)
        ax.axvspan(120, 150, color="#fff2cc", alpha=0.35, lw=0)
        ax.axvspan(150, ELL_HI, color="#ffd9d9", alpha=0.40, lw=0)

        ax.axhline(1.0, color="black", lw=1.6)

        for stem, (disp, lw) in MODELS.items():
            ell, values = _ratio(stem, tau, chan_key)
            ax.plot(
                ell,
                values,
                color=MODEL_COLORS[disp],
                ls=MODEL_LINESTYLES[disp],
                lw=lw,
                label=disp,
            )

        ax.set_xscale("log")
        ax.set_xlim(ELL_LO, ELL_HI)
        ax.set_ylim(Y_LO, Y_HI)

        # Finer Y-grid: major 0.1, minor 0.025.
        ax.yaxis.set_major_locator(MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))

        # X positions are spherical degree, labels are angular wavelength.
        ax.xaxis.set_major_locator(FixedLocator(xticks_ell))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xticklabels(xticks_deg)

        ax.grid(True, which="major", alpha=0.35, linestyle="-", lw=0.5)
        ax.grid(True, which="minor", alpha=0.18, linestyle=":", lw=0.4)

        split = "trained" if tau == 5 else "held out"
        ax.set_title(
            f"{chan_label}, {split} " + r"$\tau$" + f"={tau} h",
            fontsize=9,
        )
        panel = chr(ord("a") + ci)
        ax.text(
            -0.08,
            1.08,
            f"{panel})",
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        if row == 1:
            ax.set_xlabel(r"Angular wavelength $\lambda = 360^\circ/\ell$", fontsize=8)
        if column == 0:
            ax.set_ylabel("Power ratio model/ERA5", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.subplots_adjust(
        top=0.94,
        hspace=0.30,
        wspace=0.16,
        left=0.10,
        right=0.985,
        bottom=0.18,
    )
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.015),
        fontsize=8,
        frameon=False,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
