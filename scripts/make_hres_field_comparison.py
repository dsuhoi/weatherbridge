#!/usr/bin/env python3
"""Plot per-field IFS HRES forecast-anchor interpolation errors."""

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
)
HELD_TAUS = {6: (2, 4), 12: (4, 6, 8)}
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


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_artifact_specs(specs: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid artifact specification: {spec!r}")
        label, raw_path = spec.split("=", 1)
        if not label or label in artifacts:
            raise ValueError(f"duplicate or empty model label: {label!r}")
        artifacts[label] = Path(raw_path)
    if set(artifacts) != set(MODEL_ORDER):
        missing = sorted(set(MODEL_ORDER) - set(artifacts))
        extra = sorted(set(artifacts) - set(MODEL_ORDER))
        raise ValueError(
            f"HRES figure requires the canonical model set; "
            f"missing={missing}, extra={extra}"
        )
    return artifacts


def _load_payload(path: Path, horizon: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    expected_taus = list(range(1, horizon))
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected forecast schema 2")
    if payload.get("channels") != list(PAPER_CHANNELS_24):
        raise ValueError(f"{path}: non-canonical channel order")
    protocol = payload.get("protocol", {})
    if protocol.get("delta_t_hours") != horizon:
        raise ValueError(f"{path}: wrong anchor horizon")
    if protocol.get("taus") != expected_taus:
        raise ValueError(f"{path}: incomplete query-hour protocol")
    if protocol.get("anchor_pairing") != "same_initialization_forecast_leads":
        raise ValueError(f"{path}: wrong HRES anchor pairing")
    if protocol.get("target_source") != "ERA5_at_intermediate_valid_time":
        raise ValueError(f"{path}: wrong verification target")
    if protocol.get("future_analysis_as_input") is not False:
        raise ValueError(f"{path}: future analysis leakage is not excluded")
    if protocol.get("max_inits") != 16:
        raise ValueError(f"{path}: expected the frozen 16-init archive")
    if not payload.get("paired_artifact", {}).get("window_index_sha256"):
        raise ValueError(f"{path}: missing paired-window fingerprint")
    return payload


def load_comparison(
    artifact_paths: dict[str, Path],
    horizon: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load and cross-check the canonical HRES comparison artifacts."""
    if horizon not in HELD_TAUS:
        raise ValueError("horizon must be 6 or 12 hours")

    payloads = {
        label: _load_payload(path, horizon)
        for label, path in artifact_paths.items()
    }
    reference = payloads["WeatherBridge"]
    reference_index = reference["paired_artifact"]["window_index_sha256"]
    data_record = {
        "forecast_anchors": reference["provenance"]["forecast_anchors"],
        "era5_truth": reference["provenance"]["era5_truth"],
        "pressure_stats": reference["provenance"]["pressure_stats"],
        "surface_stats": reference["provenance"]["surface_stats"],
    }
    data_hash = _canonical_hash(data_record)

    values: dict[str, np.ndarray] = {}
    for label in MODEL_ORDER:
        payload = payloads[label]
        current_record = {
            "forecast_anchors": payload["provenance"]["forecast_anchors"],
            "era5_truth": payload["provenance"]["era5_truth"],
            "pressure_stats": payload["provenance"]["pressure_stats"],
            "surface_stats": payload["provenance"]["surface_stats"],
        }
        if _canonical_hash(current_record) != data_hash:
            raise ValueError(f"{label}: HRES/ERA5 provenance differs")
        if payload["paired_artifact"]["window_index_sha256"] != reference_index:
            raise ValueError(f"{label}: paired HRES window index differs")

        model_key = payload.get("model_name")
        rows: list[list[float]] = []
        for tau in range(1, horizon):
            tau_record = payload["per_tau"].get(str(tau), {})
            if model_key not in tau_record:
                raise ValueError(f"{label}: missing tau={tau} model record")
            record = tau_record[model_key]
            row: list[float] = []
            for channel in PAPER_CHANNELS_24:
                value = record.get(f"rmse_norm_{channel}")
                count = record.get(f"n_pairs_{channel}")
                if value is None or not np.isfinite(value) or not count:
                    raise ValueError(
                        f"{label}: invalid {channel} score at tau={tau}"
                    )
                row.append(float(value))
            rows.append(row)
        values[label] = np.asarray(rows, dtype=np.float64)

    metadata = {
        "window_index_sha256": reference_index,
        "data_provenance_sha256": data_hash,
        "forecast_init_count": reference["provenance"]["forecast_anchors"][
            "init_count"
        ],
    }
    if metadata["forecast_init_count"] != 16:
        raise ValueError("canonical HRES archive must contain 16 initialisations")
    return values, metadata


def make_figure(
    artifact_paths: dict[str, Path],
    *,
    horizon: int,
    output: Path,
    manifest: Path,
) -> dict[str, Any]:
    values, metadata = load_comparison(artifact_paths, horizon)
    taus = np.arange(1, horizon)
    held_taus = HELD_TAUS[horizon]

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
    figure, axes = plt.subplots(
        6,
        4,
        # Drawn at the manuscript column width so the figure is placed at
        # scale one and its 6.2 pt labels clear the 5 pt legibility floor.
        figsize=(COLUMN_WIDTH_IN, 9.1 * COLUMN_WIDTH_IN / 7.2),
        sharex=True,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()
    for channel_index, (axis, channel) in enumerate(
        zip(axes_flat, PAPER_CHANNELS_24, strict=True)
    ):
        for tau in held_taus:
            axis.axvspan(
                tau - 0.42,
                tau + 0.42,
                color="#E5E7EB",
                alpha=0.42,
                linewidth=0,
                zorder=0,
            )
        for label in MODEL_ORDER:
            axis.plot(
                taus,
                values[label][:, channel_index],
                color=MODEL_COLORS[label],
                linestyle=MODEL_LINESTYLES[label],
                marker=MODEL_MARKERS[label],
                linewidth=1.35 if label == "WeatherBridge" else 1.0,
                markersize=3.0 if label == "WeatherBridge" else 2.4,
                markeredgewidth=0.45,
                label=label,
                zorder=5 if label == "WeatherBridge" else 3,
            )
        minimum = min(
            float(values[label][:, channel_index].min())
            for label in MODEL_ORDER
        )
        maximum = max(
            float(values[label][:, channel_index].max())
            for label in MODEL_ORDER
        )
        span = max(maximum - minimum, maximum * 0.025, 1.0e-6)
        axis.set_ylim(max(0.0, minimum - 0.12 * span), maximum + 0.12 * span)
        axis.set_xlim(0.55, horizon - 0.55)
        axis.set_title(DISPLAY_NAMES[channel], pad=2.0)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.4, alpha=0.65)
        axis.tick_params(length=2.0, width=0.5, pad=1.5)
        if horizon == 6:
            axis.set_xticks(taus)
        else:
            axis.set_xticks((1, 3, 5, 7, 9, 11))
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
        columnspacing=1.5,
        handlelength=2.5,
    )
    figure.text(
        0.012,
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
        left=0.070,
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
    style_source = source.with_name("paper_plot_style.py")
    result = {
        "schema_version": 1,
        "status": "complete",
        "horizon_hours": horizon,
        "taus": taus.tolist(),
        "held_taus": list(held_taus),
        "models": list(MODEL_ORDER),
        "channels": list(PAPER_CHANNELS_24),
        **metadata,
        "source_sha256": {
            str(source): _sha256(source),
            str(style_source): _sha256(style_source),
        },
        "artifact_sha256": {
            label: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for label, path in artifact_paths.items()
        },
        "figure": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
        },
    }
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(result, sort_keys=True) + "\n")
    os.replace(temporary_manifest, manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", action="append", required=True)
    parser.add_argument("--horizon", type=int, choices=(6, 12), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    artifacts = _parse_artifact_specs(args.artifact)
    make_figure(
        artifacts,
        horizon=args.horizon,
        output=args.output,
        manifest=args.manifest,
    )
    print(f"[write] {args.output}")
    print(f"[write] {args.manifest}")


if __name__ == "__main__":
    main()
