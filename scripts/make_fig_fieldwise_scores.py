#!/usr/bin/env python3
"""Plot per-field normalized RMSE and ACC error for every paper model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scripts.paper_plot_style import (
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        TEXT_WIDTH_IN,
        use_paper_rc,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    from paper_plot_style import (
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        TEXT_WIDTH_IN,
        use_paper_rc,
    )


ROOT = Path(__file__).resolve().parent.parent
CHANNELS = (
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
)
MODEL_FILES = {
    6: (
        ("SwinV2", "fuxi_24ch_6yr_ep8.json"),
        ("ModAFNO", "modafno_24ch_6yr_ep8.json"),
        ("S-DYff", "sdyff_24ch_6yr_ep8.json"),
        ("PixelAttn-VFI", "atm_vfi_6yr_ep8_matched.json"),
        ("WeatherDCAE-14M", "weatherdcae_14m_6yr_ep8_matched.json"),
        ("WeatherBridge", "weatherbridge_pp3_14m_6yr_ep8.json"),
    ),
    12: (
        ("SwinV2", "fuxi_3yr_ep10.json"),
        ("ModAFNO", "modafno_3yr_ep10.json"),
        ("S-DYff", "sdyff_3yr_ep10.json"),
        ("PixelAttn-VFI", "atm_vfi_3yr_ep10_matched.json"),
        ("WeatherDCAE-14M", "weatherdcae_14m_3yr_ep10_matched.json"),
        ("WeatherBridge", "weatherbridge_14m_3yr_ep10.json"),
    ),
}
CORRECTED_BASELINE_MODELS = {
    "SwinV2",
    "ModAFNO",
    "S-DYff",
    "PixelAttn-VFI",
}
GROUP_BOUNDS = (0, 4, 8, 12, 16, 20, 24)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract(payload: dict[str, Any], method: str, hours: range) -> tuple[np.ndarray, np.ndarray]:
    rmse = np.empty((len(hours), len(CHANNELS)), dtype=np.float64)
    acc = np.empty_like(rmse)
    for hour_index, tau in enumerate(hours):
        entry = payload["per_tau"][str(tau)][method]
        for channel_index, channel in enumerate(CHANNELS):
            rmse[hour_index, channel_index] = entry[f"rmse_norm_{channel}"]
            acc[hour_index, channel_index] = entry[f"acc_{channel}"]
    if not np.isfinite(rmse).all() or not np.isfinite(acc).all():
        raise ValueError("fieldwise score input contains non-finite values")
    return rmse, acc


def load_horizon(metrics_root: Path, horizon: int) -> dict[str, Any]:
    hours = range(1, horizon)
    horizon_root = metrics_root / f"{horizon}h_2020"
    files = MODEL_FILES[horizon]
    missing = [horizon_root / filename for _, filename in files if not (horizon_root / filename).is_file()]
    if missing:
        raise FileNotFoundError("missing fieldwise inputs: " + ", ".join(map(str, missing)))

    payloads = {
        name: json.loads((horizon_root / filename).read_text())
        for name, filename in files
    }
    source_paths = {name: horizon_root / filename for name, filename in files}
    reference = payloads["WeatherBridge"]
    linear_rmse, linear_acc = _extract(reference, "bilinear", hours)

    names = ["Linear Interp.", *(name for name, _ in files)]
    rmse_by_model = [linear_rmse.mean(axis=0)]
    acc_error_by_model = [(1.0 - linear_acc).mean(axis=0)]
    bilinear_deltas: dict[str, float] = {}
    for name, _ in files:
        rmse, acc = _extract(payloads[name], "model", hours)
        rmse_by_model.append(rmse.mean(axis=0))
        acc_error_by_model.append((1.0 - acc).mean(axis=0))
        own_linear_rmse, _ = _extract(payloads[name], "bilinear", hours)
        bilinear_deltas[name] = float(np.max(np.abs(own_linear_rmse - linear_rmse)))

    rmse_values = np.stack(rmse_by_model)
    acc_error_values = np.stack(acc_error_by_model)
    bridge_index = names.index("WeatherBridge")
    return {
        "horizon": horizon,
        "hours": list(hours),
        "names": names,
        "rmse": rmse_values,
        "acc_error": acc_error_values,
        "rmse_wins": int(np.sum(np.argmin(rmse_values, axis=0) == bridge_index)),
        "acc_wins": int(np.sum(np.argmin(acc_error_values, axis=0) == bridge_index)),
        "source_paths": source_paths,
        "linear_source": source_paths["WeatherBridge"],
        "bilinear_max_abs_delta": bilinear_deltas,
    }


def _style_field_axis(ax: plt.Axes) -> None:
    for group_index, (start, stop) in enumerate(zip(GROUP_BOUNDS[:-1], GROUP_BOUNDS[1:], strict=True)):
        if group_index % 2 == 1:
            ax.axvspan(start - 0.5, stop - 0.5, color="#F3F4F6", zorder=0)
    for boundary in GROUP_BOUNDS[1:-1]:
        ax.axvline(boundary - 0.5, color="#9CA3AF", linewidth=0.45, zorder=1)
    ax.set_xlim(-0.5, len(CHANNELS) - 0.5)
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.45, linestyle=":")
    ax.set_axisbelow(True)


def _draw_panel(
    ax: plt.Axes,
    scores: dict[str, Any],
    metric: str,
    panel: str,
) -> None:
    values = scores[metric]
    x = np.arange(len(CHANNELS))
    for model_index, name in enumerate(scores["names"]):
        is_bridge = name == "WeatherBridge"
        ax.plot(
            x,
            values[model_index],
            label=name,
            color=MODEL_COLORS[name],
            linestyle=MODEL_LINESTYLES[name],
            marker=MODEL_MARKERS[name],
            linewidth=1.8 if is_bridge else 0.9,
            markersize=3.4 if is_bridge else 2.2,
            markeredgewidth=0.3,
            alpha=1.0 if is_bridge else 0.88,
            zorder=5 if is_bridge else 3,
        )
    _style_field_axis(ax)
    ax.set_yscale("log")
    if metric == "rmse":
        wins = scores["rmse_wins"]
        ylabel = "Mean normalised RMSE"
    else:
        wins = scores["acc_wins"]
        ylabel = r"Mean $1-\mathrm{ACC}$"
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{panel})  {scores['horizon']} h: WeatherBridge best on {wins}/24 fields",
        loc="left",
        fontweight="bold",
    )


def make_figure(six: dict[str, Any], twelve: dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    use_paper_rc(6.8)
    fig, grid = plt.subplots(
        2,
        2,
        figsize=(TEXT_WIDTH_IN, 4.65),
        sharex=True,
        constrained_layout=False,
    )
    axes = grid.ravel()
    _draw_panel(axes[0], six, "rmse", "a")
    _draw_panel(axes[1], six, "acc_error", "b")
    _draw_panel(axes[2], twelve, "rmse", "c")
    _draw_panel(axes[3], twelve, "acc_error", "d")

    for ax in axes[2:]:
        ax.set_xticks(np.arange(len(CHANNELS)))
        ax.set_xticklabels(CHANNELS, rotation=55, ha="right", fontsize=5.5)
        ax.set_xlabel("Weather field")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.997),
        columnspacing=1.05,
        handlelength=1.7,
        fontsize=6.0,
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.995,
        top=0.905,
        bottom=0.155,
        hspace=0.46,
        wspace=0.27,
    )
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=ROOT / "metrics" / "journal_unified",
    )
    parser.add_argument(
        "--out-pdf",
        type=Path,
        default=ROOT / "paper" / "images" / "fig_fieldwise_scores.pdf",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "paper" / "images" / "fig_fieldwise_scores.png",
    )
    parser.add_argument(
        "--out-provenance",
        type=Path,
        default=ROOT / "metrics" / "fieldwise_scores_2020" / "figure_provenance.json",
    )
    args = parser.parse_args()

    six = load_horizon(args.metrics_root, 6)
    twelve = load_horizon(args.metrics_root, 12)
    make_figure(six, twelve, args.out_pdf, args.out_png)

    sources = {
        f"{scores['horizon']}h:{name}": {
            "path": str(path),
            "sha256": sha256_file(path),
        }
        for scores in (six, twelve)
        for name, path in scores["source_paths"].items()
    }
    provenance = {
        "schema_version": 1,
        "aggregation": "arithmetic mean over all interior query hours for each field",
        "metrics": ["normalized_rmse", "one_minus_acc"],
        "channels": list(CHANNELS),
        "weatherbridge_field_wins": {
            "6h": {"normalized_rmse": six["rmse_wins"], "acc": six["acc_wins"]},
            "12h": {"normalized_rmse": twelve["rmse_wins"], "acc": twelve["acc_wins"]},
        },
        "corrected_baseline_models": sorted(CORRECTED_BASELINE_MODELS),
        "bilinear_max_abs_delta_from_canonical": {
            "6h": six["bilinear_max_abs_delta"],
            "12h": twelve["bilinear_max_abs_delta"],
        },
        "sources": sources,
        "outputs": {
            "pdf": {"path": str(args.out_pdf), "sha256": sha256_file(args.out_pdf)},
            "png": {"path": str(args.out_png), "sha256": sha256_file(args.out_png)},
        },
    }
    args.out_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.out_provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")
    print(f"wrote {args.out_provenance}")


if __name__ == "__main__":
    main()
