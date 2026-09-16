"""Canonical model colors and markers for paper figures."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl

# Springer Nature expects embedded vector fonts. Matplotlib's PDF default is
# Type 3, so all paper figures use TrueType outlines instead.
mpl.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

MODEL_COLORS = {
    "ERA5": "#111111",
    "Linear Interp.": "#6B7280",
    "SwinV2": "#8C564B",
    "FuXi": "#8C564B",  # Legacy artifact label.
    "ModAFNO": "#17BECF",
    "S-DYff": "#9467BD",
    "S-DYff-ENS": "#1F77B4",
    "PixelAttn-VFI": "#2CA02C",
    "ATM-VFI": "#2CA02C",
    "WeatherDCAE-14M": "#FF7F0E",
    "WeatherDCAE-Skip": "#E377C2",
    "Flow": "#56B4E9",
    "WeatherBridge": "#D62728",
    "Frozen coefficient adapter": "#A50F15",
    "Refine": "#7F8C8D",
    "Detail": "#1F77B4",
}

# Springer Nature requires text to stay legible at final size, with 5 pt as the
# hard floor. The manuscript column is 372 pt wide, so a figure drawn at this
# width is placed at scale one and its point sizes survive into the PDF.
COLUMN_WIDTH_IN = 372.0 / 72.0
# Two-column figures are drawn at the manuscript text width so LaTeX does not
# enlarge labels or turn a nominally compact plot into an over-height float.
TEXT_WIDTH_IN = 468.0 / 72.0


def use_paper_rc(base: float = 7.0) -> None:
    """Set the shared rc block for figures drawn at final column width."""
    mpl.rcParams.update(
        {
            "font.size": base,
            "axes.titlesize": base + 0.5,
            "axes.labelsize": base,
            "xtick.labelsize": base - 0.5,
            "ytick.labelsize": base - 0.5,
            "legend.fontsize": base - 0.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.2,
            "grid.linewidth": 0.4,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

MODEL_MARKERS = {
    "Linear Interp.": "o",
    "SwinV2": "s",
    "FuXi": "s",
    "ModAFNO": "^",
    "S-DYff": "D",
    "S-DYff-ENS": "d",
    "PixelAttn-VFI": "v",
    "ATM-VFI": "v",
    "WeatherDCAE-14M": "P",
    "WeatherDCAE-Skip": "h",
    "Flow": "<",
    "WeatherBridge": "X",
    "Frozen coefficient adapter": "*",
    "Refine": ">",
}

# Distinct line patterns keep the figures legible in greyscale and avoid
# relying only on the red/green distinction between WeatherBridge and
# PixelAttn-VFI.
MODEL_LINESTYLES = {
    "Linear Interp.": (0, (4, 2)),
    "SwinV2": (0, (1, 1)),
    "FuXi": (0, (1, 1)),
    "ModAFNO": (0, (6, 2)),
    "S-DYff": (0, (3, 1, 1, 1)),
    "S-DYff-ENS": (0, (2, 1)),
    "PixelAttn-VFI": (0, (5, 1, 1, 1)),
    "ATM-VFI": (0, (5, 1, 1, 1)),
    "WeatherDCAE-14M": (0, (8, 2, 2, 2)),
    "WeatherDCAE-Skip": (0, (2, 1)),
    "Flow": (0, (2, 2)),
    "WeatherBridge": "-",
    # The adapted checkpoint differs from its parent only by a darker red, which
    # is not a safe distinction at final size or in greyscale, so it also gets
    # its own dash pattern.
    "Frozen coefficient adapter": (0, (3, 1.3)),
    "Refine": (0, (7, 2, 1, 2)),
}


def add_panel_labels(
    axes: Iterable,
    *,
    inside: bool = False,
    fontsize: float | None = None,
) -> None:
    """Add npj-style lower-case panel labels in reading order."""
    for index, ax in enumerate(axes):
        label = f"{chr(ord('a') + index)})"
        if inside:
            ax.text(
                0.02,
                0.97,
                label,
                transform=ax.transAxes,
                fontsize=9 if fontsize is None else fontsize,
                fontweight="bold",
                va="top",
                ha="left",
                zorder=20,
                bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1},
            )
        else:
            ax.text(
                -0.09,
                1.08,
                label,
                transform=ax.transAxes,
                fontsize=11 if fontsize is None else fontsize,
                fontweight="bold",
                va="bottom",
                ha="left",
            )
