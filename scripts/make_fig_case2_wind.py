"""Regenerate paper/images/fig_case2_wind.pdf in English using the
precomputed Hurricane Laura panels shipped with the demo.

Mirrors the structure of scripts/make_fig5_haishen_local.py 1-for-1:
the same 3-row × 5-column layout, the same row labels, identical
highlight-box convention on τ=2 and τ=3 columns, and the same
viridis colour ramp / per-panel RMSE caption format. The only
content difference is that the field shown is the 10-m wind speed
magnitude |U10| = sqrt(u10^2 + v10^2) (m s^-1) on Hurricane Laura
over the Gulf of Mexico, 26 August 2020.

Reads `demo/precomputed/hurricane_laura/North_America__{u10,v10}.npz`
(panel order: ERA5, Bilinear, WeatherDCAE, FuXi, S-DYff). The figure
shows the rows used by the paper text: ERA5 (truth), Bilinear, and
S-DYff. Five columns for τ ∈ {1..5} h. The τ=2 and
τ=3 columns are annotated with one highlight box over the storm
eyewall in the central Gulf.

No cluster access is required; this is a pure-matplotlib reader.

Output:
  paper/images/fig_case2_wind.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_U = ROOT / "demo" / "precomputed" / "hurricane_laura" / "North_America__u10.npz"
SRC_V = ROOT / "demo" / "precomputed" / "hurricane_laura" / "North_America__v10.npz"
OUT_PDF = ROOT / "paper" / "images" / "fig_case2_wind.pdf"

DISPLAY_ROWS = ["ERA5", "Linear Interp.", "S-DYff"]
ROW_LABELS = {
    "ERA5":         "ERA5 (truth)",
    "Linear Interp.":     "Linear Interp.",
    "S-DYff":       "S-DYff",
}

# Highlight box drawn on the τ=2 AND τ=3 columns (every row) over the
# Hurricane Laura eyewall in the central Gulf of Mexico. Longitudes use
# the 0-360 convention that matches the precomputed NPZ grid.
HIGHLIGHT_BOXES = [
    dict(lon_w=266.5, lon_e=272.0, lat_s=24.5, lat_n=29.5,
         color="#ff2020", lw=2.0),
]


def _load_speed():
    du = np.load(SRC_U, allow_pickle=True)
    dv = np.load(SRC_V, allow_pickle=True)
    assert list(du["methods"]) == list(dv["methods"])
    assert list(du["taus"]) == list(dv["taus"])
    methods = list(du["methods"])
    taus = list(du["taus"])
    lat = du["lat"]
    lon = du["lon"]
    u = du["panels"]
    v = dv["panels"]
    speed = np.sqrt(u * u + v * v)
    # Per-(method, tau) RMSE recomputed in wind-speed space (m s^-1).
    truth_idx = methods.index("ERA5")
    diff = speed - speed[truth_idx][None]
    rmse = np.sqrt((diff ** 2).mean(axis=(-2, -1)))
    return methods, taus, lat, lon, speed, rmse


def main() -> None:
    methods, taus, lat, lon, panels, rmse = _load_speed()

    SRC_KEY = {"Linear Interp.": "Bilinear"}
    rows_idx = [methods.index(SRC_KEY.get(m, m)) for m in DISPLAY_ROWS]
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
        "Hurricane Laura, 26 August 2020 — surface wind speed |U10|",
        fontsize=14, fontweight="bold", y=0.995)
    fig.subplots_adjust(right=0.93, top=0.94, left=0.04, bottom=0.04)
    if im_for_cbar is not None:
        cax = fig.add_axes([0.94, 0.10, 0.012, 0.80])
        cb = fig.colorbar(im_for_cbar, cax=cax)
        cb.ax.tick_params(labelsize=9)
        cb.set_label(r"m s$^{-1}$", fontsize=10)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
