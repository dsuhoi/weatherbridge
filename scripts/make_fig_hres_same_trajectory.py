#!/usr/bin/env python3
"""Figure: densifying a numerical forecast trajectory (HRES anchors, HRES target).

Both anchors and the target come from one IFS HRES run, so there is no
forecast-analysis mismatch and the task is pure temporal reconstruction. This
is the regime where a learned interpolator has real headroom, in contrast to
the forecast-to-analysis setting where most of the error is forecast error.
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
        "WTI_HRES_SAME_TRAJECTORY_DIR",
        ROOT / "metrics" / "hres_same_trajectory_12h_v1",
    )
)
OUT = ROOT / "paper" / "images" / "fig_hres_same_trajectory.pdf"

# display name -> artifact stem, ordered as they are ranked in the table
MODELS = [
    ("Linear interpolation", "linear"),
    ("PixelAttn-VFI", "pixelattn_vfi"),
    ("WeatherDCAE-14M", "weatherdcae_14m"),
    ("WeatherBridge", "flow_spectral"),
]

MODEL_HATCHES = {
    "PixelAttn-VFI": "////",
    "WeatherDCAE-14M": "....",
    "WeatherBridge": "xxxx",
}

GROUPS = [
    ("T", ["T1000", "T925", "T850", "T700"]),
    ("U", ["U1000", "U925", "U850", "U700"]),
    ("V", ["V1000", "V925", "V850", "V700"]),
    ("Q", ["Q1000", "Q925", "Q850", "Q700"]),
    ("Z", ["Z1000", "Z925", "Z850", "Z700"]),
    ("surface", ["t2m", "u10", "v10", "mslp"]),
]


def per_channel(stem: str) -> dict[str, float] | None:
    path = SRC / f"{stem}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    name = payload["model_name"]
    out: dict[str, float] = {}
    for _tau, body in payload["per_tau"].items():
        inner = body.get(name) or next(iter(body.values()))
        for key, value in inner.items():
            if key.startswith("rmse_norm_") and value is not None:
                out[key[len("rmse_norm_") :]] = value
    return out


def gain(reference: dict[str, float], scores: dict[str, float], keys) -> float:
    base = np.mean([reference[k] for k in keys])
    return 100.0 * (base - np.mean([scores[k] for k in keys])) / base


def main() -> None:
    use_paper_rc(7.0)

    scores: dict[str, dict[str, float]] = {}
    for label, stem in MODELS:
        got = per_channel(stem)
        if got is None:
            print(f"[skip] {label}: {stem}.json not found")
            continue
        scores[label] = got

    linear = scores["Linear interpolation"]
    learned = [label for label, _ in MODELS if label in scores and label != "Linear interpolation"]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(COLUMN_WIDTH_IN, 4.05),
        gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.55},
    )

    # a) aggregate reduction relative to linear interpolation
    top = axes[0]
    shared = sorted(set(linear).intersection(*(set(scores[m]) for m in learned)))
    overall = {m: gain(linear, scores[m], shared) for m in learned}
    order = sorted(learned, key=lambda m: overall[m])
    top.barh(
        range(len(order)),
        [overall[m] for m in order],
        color=[MODEL_COLORS[m] for m in order],
        height=0.66,
        edgecolor="0.2",
        linewidth=0.45,
        hatch=[MODEL_HATCHES[m] for m in order],
    )
    for y, model in enumerate(order):
        top.text(overall[model] + 0.7, y, f"{overall[model]:.1f}", va="center", fontsize=6.5)
    top.set_yticks(range(len(order)))
    top.set_yticklabels(order)
    top.set_xlim(0, max(overall.values()) * 1.16)
    top.set_xlabel("RMSE reduction vs linear interpolation (%)")
    top.set_title("a)  Densifying an IFS HRES trajectory", loc="left", fontweight="bold")
    top.grid(axis="x", alpha=0.3, lw=0.4)
    top.set_axisbelow(True)

    # b) the same reduction resolved by variable group
    bottom = axes[1]
    width = 0.78 / len(learned)
    xs = np.arange(len(GROUPS))
    for index, model in enumerate(learned):
        values = [
            gain(linear, scores[model], [c for c in chans if c in scores[model] and c in linear])
            for _name, chans in GROUPS
        ]
        bottom.bar(
            xs + index * width - 0.39 + width / 2,
            values,
            width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="0.2",
            linewidth=0.45,
            hatch=MODEL_HATCHES[model],
        )
    bottom.axhline(0.0, color="0.25", lw=0.6)
    bottom.set_xticks(xs)
    bottom.set_xticklabels([name for name, _ in GROUPS])
    bottom.set_ylabel("RMSE reduction vs linear (%)")
    bottom.set_ylim(-8, 56)
    bottom.set_xlim(-0.55, len(GROUPS) - 0.45)
    bottom.set_title(
        "b)  Resolved by variable group (colours as in a)", loc="left", fontweight="bold"
    )
    bottom.grid(axis="y", alpha=0.3, lw=0.4)
    bottom.set_axisbelow(True)

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)
    for model in sorted(learned, key=lambda m: -overall[m]):
        print(f"  {model:<26}{overall[model]:6.2f}%")


if __name__ == "__main__":
    main()
