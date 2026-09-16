"""Generate paper/images/fig_case2_t2m.pdf — companion to the Case 2
wind-speed study, but showing the 2-m air temperature field over the
Hurricane Laura window (Gulf of Mexico, 26 Aug 2020).

Same layout as scripts/make_fig_case2_wind.py and the Haishen template
(3 rows × 5 cols, ERA5 + Bilinear + S-DYff, τ ∈ {1..5}),
so the two figures can be compared side-by-side to choose whichever
field is more visually informative for the paper.

Reads `demo/precomputed/hurricane_laura/North_America__t2m.npz`
(panel order: ERA5, Bilinear, WeatherDCAE, FuXi, S-DYff). FuXi row is
dropped to keep the layout consistent with the body case studies.

Output:
  paper/images/fig_case2_t2m.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "demo" / "precomputed" / "hurricane_laura" / "North_America__t2m.npz"
OUT_PDF = ROOT / "paper" / "images" / "fig_case2_t2m.pdf"

DISPLAY_ROWS = ["ERA5", "Linear Interp.", "S-DYff"]
ROW_LABELS = {
    "ERA5":         "ERA5 (truth)",
    "Linear Interp.":     "Linear Interp.",
    "S-DYff":       "S-DYff",
}

# Highlight box over the Hurricane Laura eyewall (central Gulf of Mexico),
# matching the box used in the |U10| figure for direct comparison.
HIGHLIGHT_BOXES = [
    dict(lon_w=266.5, lon_e=272.0, lat_s=24.5, lat_n=29.5,
         color="#ff2020", lw=2.0),
]


def main() -> None:
    d = np.load(SRC, allow_pickle=True)
    methods = list(d["methods"])
    taus = list(d["taus"])
    lat = d["lat"]
    lon = d["lon"]
    panels = d["panels"]    # (M, T, H, W) — Kelvin

    SRC_KEY = {"Linear Interp.": "Bilinear"}
    rows_idx = [methods.index(SRC_KEY.get(m, m)) for m in DISPLAY_ROWS]

    # Recompute RMSE in t2m space (K) so the per-panel caption is
    # self-consistent with the field we are actually plotting.
    truth_idx = methods.index("ERA5")
    diff = panels - panels[truth_idx][None]
    rmse = np.sqrt((diff ** 2).mean(axis=(-2, -1)))

    n_rows = len(DISPLAY_ROWS)
    n_cols = len(taus)

    stack = panels[rows_idx, :, :, :].ravel()
    vmin = float(np.percentile(stack, 1))
    vmax = float(np.percentile(stack, 99))

    fig = plt.figure(figsize=(n_cols * 4.0, n_rows * 2.6))
    gs = GridSpec(n_rows, n_cols, wspace=0.06, hspace=0.14, figure=fig)

    extent = [lon[0], lon[-1], lat[-1], lat[0]]
    im_for_cbar = None
    contour_levels = np.linspace(vmin, vmax, 9)

    for ri, m_idx in enumerate(rows_idx):
        method = DISPLAY_ROWS[ri]
        for ci, tau in enumerate(taus):
            ax = fig.add_subplot(gs[ri, ci])
            data = panels[m_idx, ci]
            im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax,
                           extent=extent, origin="upper", aspect="auto")
            try:
                ax.contour(lon, lat, data, levels=contour_levels,
                           colors="black", linewidths=0.4, alpha=0.7)
            except Exception:
                pass
            im_for_cbar = im

            if ri == 0:
                ax.set_title(f"τ = {int(tau)} h", fontsize=11)
            if ci == 0:
                ax.text(-0.12, 0.5, ROW_LABELS[method],
                        transform=ax.transAxes, rotation=90,
                        va="center", ha="right", fontsize=12, fontweight="bold")

            if method != "ERA5":
                ax.text(0.98, 0.95, f"RMSE={rmse[m_idx, ci]:.3g}",
                        transform=ax.transAxes, fontsize=8,
                        ha="right", va="top",
                        bbox=dict(facecolor="white", alpha=0.75, linewidth=0,
                                  boxstyle="round,pad=0.18"))

            if int(tau) in (2, 3):
                for box in HIGHLIGHT_BOXES:
                    rect = plt.Rectangle(
                        (box["lon_w"], box["lat_s"]),
                        box["lon_e"] - box["lon_w"],
                        box["lat_n"] - box["lat_s"],
                        fill=False, edgecolor=box["color"],
                        linewidth=box["lw"], linestyle="-", zorder=10)
                    ax.add_patch(rect)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(
        "Hurricane Laura, 26 August 2020 — 2-m air temperature t2m",
        fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(right=0.93, top=0.94, left=0.04, bottom=0.04)
    if im_for_cbar is not None:
        cax = fig.add_axes([0.94, 0.10, 0.012, 0.80])
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.ax.tick_params(labelsize=9)
        cb.set_label("K", fontsize=10)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
