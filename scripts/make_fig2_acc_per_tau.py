#!/usr/bin/env python3
"""Plot full-year 2020 ACC from the canonical journal evaluator."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.paper_plot_style import (
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    from paper_plot_style import MODEL_COLORS, MODEL_LINESTYLES, MODEL_MARKERS

ROOT = Path(__file__).resolve().parents[1]
METRICS_ROOT = Path(
    os.environ.get(
        "WTI_JOURNAL_METRICS_ROOT",
        ROOT / "metrics" / "journal_unified",
    )
)
METRICS_DIR = METRICS_ROOT / "6h_2020"
OUT_PDF = ROOT / "paper" / "images" / "fig2_acc_per_tau.pdf"
TAUS = [1, 2, 3, 4, 5]

MODELS = [
    ("SwinV2", "fuxi_24ch_6yr_ep8.json"),
    ("ModAFNO", "modafno_24ch_6yr_ep8.json"),
    ("S-DYff", "sdyff_24ch_6yr_ep8.json"),
    ("PixelAttn-VFI", "atm_vfi_6yr_ep8_matched.json"),
    ("WeatherDCAE-14M", "weatherdcae_14m_6yr_ep8_matched.json"),
    ("WeatherBridge", "weatherbridge_pp3_14m_6yr_ep8.json"),
]


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing canonical metric {path}; sync the completed journal evaluator"
        )
    return json.loads(path.read_text())


def _load_model_payloads() -> dict[str, dict]:
    payloads = {
        name: _load(METRICS_DIR / filename)
        for name, filename in MODELS
    }
    indices = {
        payload.get("evaluation_protocol", {}).get("index_sha256")
        for payload in payloads.values()
    }
    if None in indices or len(indices) != 1:
        raise ValueError("figure models use different or missing window indices")
    return payloads


def _curve(payload: dict, method: str) -> np.ndarray:
    channels = [str(value) for value in payload["acc_channel_names"]]
    return np.asarray(
        [
            np.mean(
                [
                    float(payload["per_tau"][str(tau)][method][f"acc_{channel}"])
                    for channel in channels
                ]
            )
            for tau in TAUS
        ],
        dtype=np.float64,
    )


def main() -> None:
    payloads = _load_model_payloads()
    reference = payloads["WeatherBridge"]
    curves = {"Linear Interp.": _curve(reference, "bilinear")}
    curves.update(
        {
            name: _curve(payload, "model")
            for name, payload in payloads.items()
        }
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for name, values in curves.items():
        is_bridge = name == "WeatherBridge"
        ax.plot(
            TAUS,
            values,
            label=name,
            color=MODEL_COLORS[name],
            linestyle=MODEL_LINESTYLES[name],
            marker=MODEL_MARKERS[name],
            linewidth=2.6 if is_bridge else 1.6,
            markersize=7 if is_bridge else 5.5,
            markerfacecolor=(
                "none" if name == "Linear Interp." else MODEL_COLORS[name]
            ),
        )

    ax.set_xticks(TAUS)
    ax.set_xlim(0.75, 5.25)
    ax.set_xlabel(r"Interpolation hour $\tau$ (h)")
    ax.set_ylabel("Anomaly correlation coefficient")
    ax.set_title("Full-year 2020 anomaly correlation")
    ax.grid(True, alpha=0.3, linestyle=":")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(bottom=0.25)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
