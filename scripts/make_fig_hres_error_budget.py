#!/usr/bin/env python3
"""Figure: how the error of forecast-anchored interpolation splits by lead.

Both anchors are IFS HRES forecast states and the target is the ERA5 analysis,
so the achievable error is bounded below by the error already present in the
anchors. This figure separates that floor from the part an interpolator can
actually address.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLUMN_WIDTH_IN, MODEL_COLORS, use_paper_rc

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(
    os.environ.get(
        "WTI_HRES_ERROR_BUDGET_JSON",
        ROOT / "metrics" / "hres_degradation_aware_2021_dev_v1" / "error_budget_by_lead.json",
    )
)
OUT = ROOT / "paper" / "images" / "fig_hres_error_budget.pdf"

LINEAR = MODEL_COLORS["Linear Interp."]
BRIDGE = MODEL_COLORS["WeatherBridge"]


def main() -> None:
    use_paper_rc(7.0)

    rows = json.loads(SRC.read_text())
    lead = np.array([r["lead_hours"] for r in rows])
    anchor = np.array([r["anchor"] for r in rows])
    linear = np.array([r["linear"] for r in rows])
    model = np.array([r["model"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(COLUMN_WIDTH_IN, 2.15))

    left = axes[0]
    left.fill_between(lead, 0, anchor, color="0.87", lw=0, label="anchor forecast error")
    left.fill_between(
        lead, anchor, linear, color="#4C72B0", alpha=0.35, lw=0, label="added by interpolation"
    )
    left.plot(lead, linear, color="#2E5C8A", lw=1.1, label="Linear interp.")
    left.plot(
        lead, model, color=BRIDGE, lw=1.3, ls="-", marker="X", ms=3.2, label="WeatherBridge"
    )
    left.plot(lead, anchor, color="0.30", lw=0.9, ls=(0, (4, 2)), label="anchor floor")
    left.set_xlabel("left-anchor forecast lead (h)")
    left.set_ylabel("normalised RMSE")
    left.set_title("a)  Error budget", loc="left", fontweight="bold")
    left.set_xlim(lead.min(), lead.max())
    left.set_ylim(0, linear.max() * 1.32)
    left.legend(fontsize=6.5, loc="upper left", handlelength=1.4, handletextpad=0.5,
                borderaxespad=0.2, labelspacing=0.3)
    left.grid(alpha=0.3, lw=0.4)
    left.set_axisbelow(True)

    right = axes[1]
    share = 100.0 * (linear - anchor) / linear
    right.plot(lead, share, color="#2E5C8A", lw=1.3, marker="o", ms=2.8)
    right.set_xlabel("left-anchor forecast lead (h)")
    right.set_ylabel("interpolation share of\ntotal error (%)")
    right.set_title("b)  What interpolation can address", loc="left", fontweight="bold")
    right.set_xlim(lead.min(), lead.max())
    right.set_ylim(0, share.max() * 1.15)
    right.grid(alpha=0.3, lw=0.4)
    right.set_axisbelow(True)

    fig.tight_layout(pad=0.3, w_pad=1.4)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)
    print(
        f"interpolation share: {share[0]:.1f}% at lead {lead[0]:.0f} h"
        f" -> {share[-1]:.1f}% at {lead[-1]:.0f} h"
    )
    print(
        "model below anchor floor by "
        f"{100 * (anchor.mean() - model.mean()) / anchor.mean():.2f}% on average"
    )


if __name__ == "__main__":
    main()
