#!/usr/bin/env python3
"""Plot the frozen 2022 confirmation of the low-cost NWP adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.paper_plot_style import (
    COLUMN_WIDTH_IN,
    MODEL_COLORS,
    MODEL_LINESTYLES,
    MODEL_MARKERS,
)
from weather_time_interp.normalization import PAPER_CHANNELS_24

MODEL_ORDER = (
    "Linear Interp.",
    "WeatherDCAE-14M",
    "WeatherBridge",
    "Frozen coefficient adapter",
)
CONTROL_KEYS = {
    "WeatherDCAE-14M": "weatherdcae_14m",
    "WeatherBridge": "flow_spectral",
}

DISPLAY_NAMES = {
    "T1000": "T 1000 hPa",
    "T925": "T 925 hPa",
    "T850": "T 850 hPa",
    "T700": "T 700 hPa",
    "U1000": "U 1000 hPa",
    "U925": "U 925 hPa",
    "U850": "U 850 hPa",
    "U700": "U 700 hPa",
    "V1000": "V 1000 hPa",
    "V925": "V 925 hPa",
    "V850": "V 850 hPa",
    "V700": "V 700 hPa",
    "Q1000": "Q 1000 hPa",
    "Q925": "Q 925 hPa",
    "Q850": "Q 850 hPa",
    "Q700": "Q 700 hPa",
    "Z1000": "Z 1000 hPa",
    "Z925": "Z 925 hPa",
    "Z850": "Z 850 hPa",
    "Z700": "Z 700 hPa",
    "t2m": "T2m",
    "u10": "U10",
    "v10": "V10",
    "mslp": "MSLP",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _load_scores(
    path: Path,
    *,
    model_key: str,
) -> tuple[dict[str, Any], np.ndarray]:
    payload = _load_json(path)
    protocol = payload.get("protocol", {})
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected forecast schema 2")
    if payload.get("channels") != list(PAPER_CHANNELS_24):
        raise ValueError(f"{path}: non-canonical channel order")
    if protocol.get("delta_t_hours") != 6 or protocol.get("taus") != [1, 2, 3, 4, 5]:
        raise ValueError(f"{path}: expected the complete six-hour protocol")
    if protocol.get("max_inits") != 16:
        raise ValueError(f"{path}: expected 16 frozen 2022 initialisations")
    if payload.get("model_name") != model_key:
        raise ValueError(f"{path}: expected model key {model_key!r}")

    rows: list[list[float]] = []
    for tau in range(1, 6):
        record = payload.get("per_tau", {}).get(str(tau), {}).get(model_key)
        if record is None:
            raise ValueError(f"{path}: missing tau={tau} record")
        if not isinstance(record, dict):
            raise TypeError(f"{path}: invalid tau={tau} record")
        row = []
        for channel in PAPER_CHANNELS_24:
            score = record.get(f"rmse_norm_{channel}")
            count = record.get(f"n_pairs_{channel}")
            if score is None or not np.isfinite(score) or int(count or 0) <= 0:
                raise ValueError(f"{path}: invalid {channel} score at tau={tau}")
            row.append(float(score))
        rows.append(row)
    return payload, np.asarray(rows, dtype=np.float64)


def load_confirmation(
    linear_path: Path,
    adapted_path: Path,
    summary_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load the paired artifacts and verify the frozen test contract."""
    linear_payload, linear = _load_scores(linear_path, model_key="linear")
    adapted_payload, adapted = _load_scores(
        adapted_path,
        model_key="flow_adapted",
    )
    summary = _load_json(summary_path)
    if not summary.get("selection_independent"):
        raise ValueError("summary is not selection-independent")
    if summary.get("test_split") != "2022_independent":
        raise ValueError("summary is not the frozen 2022 confirmation")
    if summary.get("winner") != "flow_adapted":
        raise ValueError("summary winner is not the adapted model")

    linear_index = linear_payload.get("paired_artifact", {}).get(
        "window_index_sha256"
    )
    adapted_index = adapted_payload.get("paired_artifact", {}).get(
        "window_index_sha256"
    )
    if not linear_index or linear_index != adapted_index:
        raise ValueError("paired window indices differ")
    linear_protocol = dict(linear_payload.get("protocol", {}))
    adapted_protocol = dict(adapted_payload.get("protocol", {}))
    linear_protocol.pop("inference_tau_batch_size", None)
    adapted_protocol.pop("inference_tau_batch_size", None)
    if linear_protocol != adapted_protocol:
        raise ValueError("evaluation protocols differ")

    summary_artifacts = summary.get("provenance", {}).get("artifacts", {})
    expected_hashes = {
        "linear": _sha256(linear_path),
        "flow_adapted": _sha256(adapted_path),
    }
    for name, digest in expected_hashes.items():
        recorded = summary_artifacts.get(name, {}).get("json", {}).get("sha256")
        if recorded != digest:
            raise ValueError(f"{name}: summary artifact hash differs")

    return linear, adapted, summary


def load_controls(
    control_paths: dict[str, Path],
    complete_path: Path,
    *,
    linear_path: Path,
) -> dict[str, np.ndarray]:
    """Load descriptive frozen-checkpoint controls on the same 2022 index."""
    if set(control_paths) != set(CONTROL_KEYS):
        raise ValueError("the complete principal architecture set is required")
    complete = _load_json(complete_path)
    if (
        complete.get("status") != "complete"
        or complete.get("test_split") != "2022_independent_descriptive_controls"
        or complete.get("selection_independent") is not False
        or complete.get("checkpoint_weights_frozen_before_2022") is not True
    ):
        raise ValueError("invalid 2022 architecture completion marker")

    linear_payload = _load_json(linear_path)
    reference_index = linear_payload["paired_artifact"]["window_index_sha256"]
    reference_protocol = dict(linear_payload["protocol"])
    reference_protocol.pop("inference_tau_batch_size", None)
    if complete.get("window_index_sha256") != reference_index:
        raise ValueError("architecture controls use a different paired index")

    values: dict[str, np.ndarray] = {}
    for label, model_key in CONTROL_KEYS.items():
        path = control_paths[label]
        payload, scores = _load_scores(path, model_key=model_key)
        if payload["paired_artifact"]["window_index_sha256"] != reference_index:
            raise ValueError(f"{label}: paired window index differs")
        protocol = dict(payload["protocol"])
        protocol.pop("inference_tau_batch_size", None)
        if protocol != reference_protocol:
            raise ValueError(f"{label}: evaluation protocol differs")
        recorded = complete.get("artifacts", {}).get(model_key, {}).get("sha256")
        if recorded != _sha256(path):
            raise ValueError(f"{label}: completion-marker hash differs")
        values[label] = scores
    return values


def make_figure(
    linear_path: Path,
    adapted_path: Path,
    summary_path: Path,
    control_paths: dict[str, Path],
    controls_complete_path: Path,
    *,
    output: Path,
    manifest: Path,
) -> dict[str, Any]:
    linear, adapted, summary = load_confirmation(
        linear_path,
        adapted_path,
        summary_path,
    )
    values = load_controls(
        control_paths,
        controls_complete_path,
        linear_path=linear_path,
    )
    values["Linear Interp."] = linear
    values["Frozen coefficient adapter"] = adapted
    families = summary["families"]
    taus = np.arange(1, 6)
    improvement = 100.0 * (linear - adapted) / linear
    tolerance = 1.0e-12
    wins = int(np.count_nonzero(improvement > tolerance))
    ties = int(np.count_nonzero(np.abs(improvement) <= tolerance))
    losses = int(np.count_nonzero(improvement < -tolerance))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.55,
        }
    )
    # Drawn at the manuscript column width, so the figure is placed at
    # essentially scale one and the 6.2 pt tick labels stay above the 5 pt
    # legibility floor of the submitted PDF.
    figure, axes = plt.subplots(
        6,
        4,
        figsize=(COLUMN_WIDTH_IN, 9.1 * COLUMN_WIDTH_IN / 7.2),
        sharex=True,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()
    for channel_index, (axis, channel) in enumerate(
        zip(axes_flat, PAPER_CHANNELS_24, strict=True)
    ):
        for held_tau in (2, 4):
            axis.axvspan(
                held_tau - 0.42,
                held_tau + 0.42,
                color="#E5E7EB",
                alpha=0.48,
                linewidth=0,
                zorder=0,
            )
        for label in MODEL_ORDER:
            emphasized = label in ("WeatherBridge", "Frozen coefficient adapter")
            axis.plot(
                taus,
                values[label][:, channel_index],
                color=MODEL_COLORS[label],
                linestyle=MODEL_LINESTYLES[label],
                marker=MODEL_MARKERS[label],
                linewidth=1.35 if emphasized else 0.9,
                markersize=2.8 if emphasized else 2.2,
                markeredgewidth=0.4,
                label=label,
                zorder=(
                    5
                    if label == "Frozen coefficient adapter"
                    else 4 if emphasized else 3
                ),
            )
        minimum = min(
            float(values[label][:, channel_index].min()) for label in MODEL_ORDER
        )
        maximum = max(
            float(values[label][:, channel_index].max()) for label in MODEL_ORDER
        )
        span = max(maximum - minimum, maximum * 0.025, 1.0e-6)
        axis.set_ylim(max(0.0, minimum - 0.12 * span), maximum + 0.12 * span)
        axis.set_xlim(0.55, 5.45)
        axis.set_xticks(taus)
        axis.set_title(DISPLAY_NAMES[channel], pad=2.0)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.65)
        axis.tick_params(length=2.0, width=0.5, pad=1.5)
        axis.text(
            0.02,
            0.97,
            f"{chr(ord('a') + channel_index)})",
            transform=axis.transAxes,
            fontsize=6.2,
            fontweight="bold",
            va="top",
            ha="left",
            zorder=20,
            bbox={
                "facecolor": "white",
                "alpha": 0.72,
                "edgecolor": "none",
                "pad": 0.6,
            },
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(MODEL_ORDER),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.15,
        handlelength=2.2,
    )
    figure.text(
        0.016,
        0.50,
        "Normalised RMSE (panel-specific scale)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.2,
    )
    figure.text(
        0.53,
        0.012,
        r"Interior forecast hour $\tau$ (h)",
        va="bottom",
        ha="center",
        fontsize=7.2,
    )
    figure.subplots_adjust(
        left=0.094,
        right=0.992,
        bottom=0.050,
        top=0.930,
        wspace=0.34,
        hspace=0.47,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.pdf")
    figure.savefig(temporary_output, format="pdf", dpi=300)
    plt.close(figure)
    os.replace(temporary_output, output)

    source = Path(__file__).resolve()
    result = {
        "schema_version": 1,
        "status": "complete",
        "test_split": summary["test_split"],
        "selection_independent": True,
        "comparison_roles": {
            "confirmatory": ["Linear Interp.", "Frozen coefficient adapter"],
            "descriptive_post_hoc": [
                "WeatherDCAE-14M",
                "WeatherBridge",
            ],
        },
        "horizon_hours": 6,
        "figure_kind": "per_field_normalised_rmse_trajectories",
        "taus": taus.tolist(),
        "channels": list(PAPER_CHANNELS_24),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "overall_rmse_reduction_percent": families["all"][
            "comparisons_to_winner"
        ]["linear"]["delta_rmse_percent_challenger_minus_winner"],
        "source_sha256": _sha256(source),
        "artifact_sha256": {
            "linear": _sha256(linear_path),
            "flow_adapted": _sha256(adapted_path),
            "paired_summary": _sha256(summary_path),
            "controls_complete": _sha256(controls_complete_path),
            **{
                CONTROL_KEYS[label]: _sha256(control_paths[label])
                for label in CONTROL_KEYS
            },
        },
        "models": list(MODEL_ORDER),
        "figure": {"path": str(output.resolve()), "sha256": _sha256(output)},
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--weatherdcae", type=Path, required=True)
    parser.add_argument("--weatherbridge", type=Path, required=True)
    parser.add_argument("--controls-complete", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    make_figure(
        args.linear,
        args.adapted,
        args.summary,
        {
            "WeatherDCAE-14M": args.weatherdcae,
            "WeatherBridge": args.weatherbridge,
        },
        args.controls_complete,
        output=args.output,
        manifest=args.manifest,
    )
    print(f"[write] {args.output}")
    print(f"[write] {args.manifest}")


if __name__ == "__main__":
    main()
