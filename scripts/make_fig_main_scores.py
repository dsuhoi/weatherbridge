#!/usr/bin/env python3
"""Combine the canonical 2020 RMSE and ACC curves in one paper figure."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scripts.make_fig1_rmse_per_tau import (
        METRICS_DIR,
        MODELS,
        ROOT,
        TAUS,
        _curve as rmse_curve,
        _load_model_payloads,
    )
    from scripts.make_fig2_acc_per_tau import _curve as acc_curve
    from scripts.paper_plot_style import (
        COLUMN_WIDTH_IN,
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        use_paper_rc,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    from make_fig1_rmse_per_tau import (
        METRICS_DIR,
        MODELS,
        ROOT,
        TAUS,
        _curve as rmse_curve,
        _load_model_payloads,
    )
    from make_fig2_acc_per_tau import _curve as acc_curve
    from paper_plot_style import (
        COLUMN_WIDTH_IN,
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        use_paper_rc,
    )


OUT_PDF = ROOT / "paper" / "images" / "fig_main_scores.pdf"


def _curves(curve_fn):
    payloads = _load_model_payloads()
    reference = payloads["WeatherBridge"]
    values = {"Linear Interp.": curve_fn(reference, "bilinear")}
    values.update(
        {
            name: curve_fn(payload, "model")
            for name, payload in payloads.items()
        }
    )
    return values


def _draw(ax, curves, ylabel: str, title: str) -> None:
    for held_tau in (2, 4):
        ax.axvspan(
            held_tau - 0.16,
            held_tau + 0.16,
            color="#E5E7EB",
            zorder=0,
        )
    for name, values in curves.items():
        is_bridge = name == "WeatherBridge"
        ax.plot(
            TAUS,
            values,
            label={"S-DYff": "S-DYff (N=1)",
                   "S-DYff-ENS": "S-DYff-ENS (N=21)"}.get(name, name),
            color=MODEL_COLORS[name],
            linestyle=MODEL_LINESTYLES[name],
            marker=MODEL_MARKERS[name],
            linewidth=2.4 if is_bridge else 1.4,
            markersize=5.5 if is_bridge else 4.2,
            markerfacecolor=(
                "none" if name == "Linear Interp." else MODEL_COLORS[name]
            ),
        )
    ax.set_xticks(TAUS)
    ax.set_xlim(0.75, 5.25)
    ax.set_xlabel(r"Interpolation hour $\tau$ (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle=":")


def main() -> None:
    # Drawn at the manuscript column width so the figure is placed at scale one
    # and its text keeps the point sizes chosen here.
    use_paper_rc(7.5)
    fig, axes = plt.subplots(
        1, 2, figsize=(COLUMN_WIDTH_IN, 2.85)
    )
    _draw(axes[0], _curves(rmse_curve), "Normalised RMSE", "a)  Error")
    _draw(axes[1], _curves(acc_curve), "ACC", "b)  Anomaly correlation")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=6.8,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.subplots_adjust(left=0.11, right=0.99, top=0.90, bottom=0.34, wspace=0.36)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
