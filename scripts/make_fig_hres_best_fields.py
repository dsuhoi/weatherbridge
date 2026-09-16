#!/usr/bin/env python3
"""Plot matched HRES lead-bin RMSE for representative WeatherBridge wins."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import (
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
    TEXT_WIDTH_IN,
    use_paper_rc,
)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "metrics" / "hres_same_trajectory_12h_v1"
OUT = ROOT / "paper" / "images" / "fig_hres_best_fields.pdf"

MODELS = [
    ("Linear interpolation", "linear"),
    ("PixelAttn-VFI", "pixelattn_vfi"),
    ("WeatherDCAE-14M", "weatherdcae_14m"),
    ("WeatherBridge", "flow_spectral"),
]

FIELDS = [
    ("Q850", r"Specific humidity, 850 hPa"),
    ("T700", r"Temperature, 700 hPa"),
    ("U850", r"Zonal wind, 850 hPa"),
    ("mslp", "Mean sea-level pressure"),
]


def _load(stem: str) -> dict:
    return json.loads((SRC / f"{stem}.json").read_text())


def _protocol_signature(payload: dict) -> tuple:
    protocol = payload["protocol"]
    anchors = payload["provenance"]["forecast_anchors"]
    return (
        protocol["delta_t_hours"],
        tuple(protocol["taus"]),
        protocol["anchor_pairing"],
        protocol["target_source"],
        protocol["max_inits"],
        anchors["manifest_sha256"],
        anchors["archive_manifest"]["sha256"],
    )


def load_matched_scores() -> tuple[list[dict], dict[str, dict[str, list[float]]]]:
    payloads = {label: _load(stem) for label, stem in MODELS}
    signatures = {_protocol_signature(payload) for payload in payloads.values()}
    if len(signatures) != 1:
        raise ValueError("HRES artifacts do not share one evaluation protocol")

    lead_bins = payloads["Linear interpolation"]["lead_bins"]
    scores: dict[str, dict[str, list[float]]] = {}
    for label, payload in payloads.items():
        model_name = payload["model_name"]
        model_scores: dict[str, list[float]] = {field: [] for field, _ in FIELDS}
        for lead_bin in lead_bins:
            body = payload["per_lead_bin"][lead_bin["name"]]["6"][model_name]
            for field, _ in FIELDS:
                model_scores[field].append(float(body[f"rmse_norm_{field}"]))
        scores[label] = model_scores
    return lead_bins, scores


def main() -> None:
    use_paper_rc(8.0)
    lead_bins, scores = load_matched_scores()
    x = np.arange(len(lead_bins))
    labels = [f"+{item['lo']}-{item['hi']} h" for item in lead_bins]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(TEXT_WIDTH_IN, 0.61 * TEXT_WIDTH_IN),
        sharex=True,
    )
    panel_letters = "abcd"
    for ax, (field, title), letter in zip(axes.flat, FIELDS, panel_letters):
        for label, _stem in MODELS:
            is_bridge = label == "WeatherBridge"
            style_label = "Linear Interp." if label == "Linear interpolation" else label
            ax.plot(
                x,
                scores[label][field],
                label=label,
                color=MODEL_COLORS[style_label],
                linestyle=MODEL_LINESTYLES[style_label],
                marker=MODEL_MARKERS[style_label],
                linewidth=2.5 if is_bridge else 1.5,
                markersize=5.8 if is_bridge else 4.6,
                markerfacecolor=(
                    "white" if label == "Linear interpolation" else MODEL_COLORS[style_label]
                ),
                markeredgewidth=0.9,
            )
        ax.set_title(f"{letter})  {title}", loc="left", fontweight="bold")
        ax.set_xticks(x, labels)
        ax.grid(True, alpha=0.28, linestyle=":")
        ax.margins(x=0.07)

    for ax in axes[:, 0]:
        ax.set_ylabel("Normalised RMSE")
    for ax in axes[-1, :]:
        ax.set_xlabel("Left-anchor forecast lead")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.94, bottom=0.19, wspace=0.22, hspace=0.34)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
