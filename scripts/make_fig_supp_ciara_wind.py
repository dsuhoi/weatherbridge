#!/usr/bin/env python3
"""Render a spatially explicit Storm Ciara wind comparison."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

try:
    from scripts.paper_plot_style import (
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        TEXT_WIDTH_IN,
    )
except ModuleNotFoundError:  # Direct execution: python scripts/<file>.py
    from paper_plot_style import (
        MODEL_COLORS,
        MODEL_LINESTYLES,
        MODEL_MARKERS,
        TEXT_WIDTH_IN,
    )


ROOT = Path(__file__).resolve().parents[1]
CASE_DIRS = {
    "WeatherDCAE-14M": ROOT / "metrics" / "case_studies_weatherdcae_14m",
    "PixelAttn-VFI": ROOT / "metrics" / "case_studies_pixelattn_vfi",
    "WeatherBridge": ROOT / "metrics" / "case_studies_weatherbridge",
}
LEARNED_METHODS = ("WeatherDCAE-14M", "PixelAttn-VFI", "WeatherBridge")
DISPLAY_METHODS = ("Linear Interp.",) + LEARNED_METHODS
SHORT_LABELS = {
    "Linear Interp.": "Linear",
    "WeatherDCAE-14M": "DCAE-14M",
    "PixelAttn-VFI": "PixelAttn",
    "WeatherBridge": "WeatherBridge",
}
EXPECTED_ARCH = {
    "WeatherDCAE-14M": "dcae_14m",
    "PixelAttn-VFI": "atmvfi",
    "WeatherBridge": "flow_pp3",
}
EXPECTED_INIT_TIME = "2020-02-09T09:00:00"
EVENT_NAME = "Storm Ciara"
EVENT_DIR = "storm_ciara_2020"
DISPLAY_TAU = 3
DISPLAY_TITLE = r"Storm Ciara, 9 February 2020, 12 UTC ($\tau=3$ h)"
MAP_EXTENT = (-25.0, 20.0, 45.0, 65.0)
X_TICKS = (-20.0, 0.0, 20.0)
Y_TICKS = (45.0, 55.0, 65.0)
X_TICK_LABELS = ("20°W", "0°", "20°E")
Y_TICK_LABELS = ("45°N", "55°N", "65°N")
SELECTION_NOTE = (
    "Event and 09-15 UTC window fixed from meteorological impact before "
    "model-error inspection."
)
OUTPUT = ROOT / "paper" / "images" / "fig_ciara_wind10.pdf"
SUMMARY = (
    ROOT
    / "metrics"
    / "case_studies_weatherbridge"
    / EVENT_DIR
    / "wind10_summary.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_case(path: Path, label: str) -> dict:
    with np.load(path, allow_pickle=False) as case:
        if case["methods"].tolist() != [label]:
            raise ValueError(f"{path}: model label mismatch")
        if case["init_time"].item() != EXPECTED_INIT_TIME:
            raise ValueError(f"{path}: event time mismatch")
        if case["model_arch"].item() != EXPECTED_ARCH[label]:
            raise ValueError(f"{path}: checkpoint architecture mismatch")
        provenance = json.loads(case["provenance_json"].item())
        if (
            provenance.get("schema_version") != 1
            or provenance.get("event") != EVENT_NAME
            or provenance.get("model_label") != label
        ):
            raise ValueError(f"{path}: provenance mismatch")
        return {
            key: np.array(case[key], copy=True)
            for key in (
                "taus",
                "lat",
                "lon",
                "prediction_u",
                "prediction_v",
                "truth_u",
                "truth_v",
                "linear_u",
                "linear_v",
            )
        } | {"provenance": provenance}


def _weighted_vector_rmse(
    pred_u: np.ndarray,
    pred_v: np.ndarray,
    truth_u: np.ndarray,
    truth_v: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    squared_error = (pred_u - truth_u) ** 2 + (pred_v - truth_v) ** 2
    weights = np.broadcast_to(
        np.cos(np.deg2rad(lat))[None, :, None], squared_error.shape
    )
    return np.sqrt(
        np.sum(squared_error * weights, axis=(-2, -1))
        / np.sum(weights, axis=(-2, -1))
    )


def _format_map_axis(axis, *, left_labels: bool, bottom_labels: bool) -> None:
    projection = ccrs.PlateCarree()
    axis.set_extent(MAP_EXTENT, crs=projection)
    axis.coastlines(resolution="50m", linewidth=0.48, color="#202020")
    axis.set_xticks(X_TICKS, crs=projection)
    axis.set_yticks(Y_TICKS, crs=projection)
    axis.set_xticklabels(X_TICK_LABELS if bottom_labels else ())
    axis.set_yticklabels(Y_TICK_LABELS if left_labels else ())
    axis.tick_params(labelsize=5.8, length=2.0, pad=1.0)


def main() -> None:
    source_paths = {
        label: directory / EVENT_DIR / "wind10.npz"
        for label, directory in CASE_DIRS.items()
    }
    cases = {label: _load_case(path, label) for label, path in source_paths.items()}
    reference = cases["WeatherBridge"]
    for label, case in cases.items():
        for key in ("taus", "lat", "lon"):
            if not np.array_equal(case[key], reference[key]):
                raise ValueError(f"{label}: {key} does not match WeatherBridge")
        for key in ("truth_u", "truth_v", "linear_u", "linear_v"):
            if not np.allclose(case[key], reference[key], rtol=0.0, atol=1e-6):
                raise ValueError(f"{label}: {key} does not match WeatherBridge")

    taus = reference["taus"]
    lat = reference["lat"]
    lon = np.where(reference["lon"] > 180.0, reference["lon"] - 360.0, reference["lon"])
    components: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "ERA5": (reference["truth_u"], reference["truth_v"]),
        "Linear Interp.": (reference["linear_u"], reference["linear_v"]),
    }
    components.update(
        {
            label: (case["prediction_u"], case["prediction_v"])
            for label, case in cases.items()
        }
    )
    truth_u, truth_v = components["ERA5"]
    rmse = {
        label: _weighted_vector_rmse(u, v, truth_u, truth_v, lat)
        for label, (u, v) in components.items()
        if label != "ERA5"
    }
    tau_index = int(np.flatnonzero(taus == DISPLAY_TAU)[0])
    error_fields = {
        label: np.hypot(u[tau_index] - truth_u[tau_index], v[tau_index] - truth_v[tau_index])
        for label, (u, v) in components.items()
        if label != "ERA5"
    }
    error_vmax = float(
        np.percentile(np.concatenate([field.ravel() for field in error_fields.values()]), 99.0)
    )
    speed = np.hypot(truth_u[tau_index], truth_v[tau_index])
    speed_vmax = float(np.percentile(speed, 99.5))

    figure = plt.figure(figsize=(TEXT_WIDTH_IN, 3.95))
    grid = GridSpec(
        2,
        3,
        figure=figure,
        left=0.065,
        right=0.89,
        bottom=0.11,
        top=0.90,
        wspace=0.27,
        hspace=0.24,
    )
    projection = ccrs.PlateCarree()
    map_specs = (
        ("ERA5", (0, 0), "ERA5 wind speed"),
        ("Linear Interp.", (0, 1), "Linear error"),
        ("WeatherDCAE-14M", (0, 2), "DCAE-14M error"),
        ("PixelAttn-VFI", (1, 1), "PixelAttn error"),
        ("WeatherBridge", (1, 2), "WeatherBridge error"),
    )
    error_image = None
    speed_image = None
    map_axes = {}
    lon_step = max(1, len(lon) // 14)
    lat_step = max(1, len(lat) // 8)
    for panel_index, (label, position, title) in enumerate(map_specs):
        axis = figure.add_subplot(grid[position], projection=projection)
        map_axes[label] = axis
        if label == "ERA5":
            image = axis.pcolormesh(
                lon, lat, speed, cmap="magma", vmin=0.0, vmax=speed_vmax,
                shading="nearest", transform=projection,
            )
            arrow_u = truth_u[tau_index]
            arrow_v = truth_v[tau_index]
            arrow_color = "white"
            speed_image = image
        else:
            pred_u, pred_v = components[label]
            image = axis.pcolormesh(
                lon, lat, error_fields[label], cmap="YlOrRd", vmin=0.0,
                vmax=error_vmax, shading="nearest", transform=projection,
            )
            error_image = image
            arrow_u = pred_u[tau_index] - truth_u[tau_index]
            arrow_v = pred_v[tau_index] - truth_v[tau_index]
            arrow_color = "#202020"
            axis.text(
                0.98, 0.04, f"vRMSE={rmse[label][tau_index]:.3f}",
                transform=axis.transAxes, fontsize=6.2, ha="right", va="bottom",
                bbox={"facecolor": "white", "alpha": 0.84, "linewidth": 0,
                      "boxstyle": "round,pad=0.16"},
            )
        axis.quiver(
            lon[::lon_step], lat[::lat_step],
            arrow_u[::lat_step, ::lon_step], arrow_v[::lat_step, ::lon_step],
            transform=projection, color=arrow_color,
            edgecolor="black" if label == "ERA5" else "none", linewidth=0.15,
            alpha=0.86, pivot="mid", scale=220 if label == "ERA5" else 45,
            width=0.004, headwidth=3.2,
        )
        _format_map_axis(
            axis, left_labels=position[1] == 0, bottom_labels=position[0] == 1
        )
        axis.set_title(
            f"{chr(ord('a') + panel_index)})  {title}", fontsize=7.3,
            fontweight="bold", color=MODEL_COLORS.get(label, "#111111"), pad=2.0,
        )
        if label == "WeatherBridge":
            for spine in axis.spines.values():
                spine.set_color(MODEL_COLORS[label])
                spine.set_linewidth(1.4)
    line_axis = figure.add_subplot(grid[1, 0])
    for held_tau in (2, 4):
        line_axis.axvspan(
            held_tau - 0.18, held_tau + 0.18, color="#E5E7EB", alpha=0.85, lw=0
        )
    for label in DISPLAY_METHODS:
        line_axis.plot(
            taus, rmse[label], label=SHORT_LABELS[label],
            color=MODEL_COLORS[label], linestyle=MODEL_LINESTYLES[label],
            marker=MODEL_MARKERS[label],
            linewidth=1.35 if label == "WeatherBridge" else 1.0, markersize=4.0,
        )
    line_axis.set_xlabel(r"Query hour $\tau$", fontsize=6.8)
    line_axis.set_ylabel(r"vRMSE (m s$^{-1}$)", fontsize=6.8, labelpad=1.5)
    line_axis.set_xticks(taus)
    line_axis.tick_params(labelsize=5.8, length=2.2)
    line_axis.grid(alpha=0.28, linewidth=0.5)
    line_handles, line_labels = line_axis.get_legend_handles_labels()

    # Cartopy preserves geographic aspect, whereas a regular Matplotlib axis
    # fills its GridSpec cell. Match the line panel to the map row explicitly.
    figure.canvas.draw()
    lower_map_position = map_axes["WeatherBridge"].get_position()
    line_position = line_axis.get_position()
    line_axis.set_position(
        [
            line_position.x0,
            lower_map_position.y0,
            line_position.width,
            lower_map_position.height,
        ]
    )

    # Keep the comparison curves unobstructed: the panel heading and legend
    # occupy the whitespace between the lower plot and the ERA5 colour bar.
    line_panel_position = line_axis.get_position()
    figure.text(
        line_panel_position.x0,
        line_panel_position.y1 + 0.078,
        "f)  Vector RMSE by query hour",
        fontsize=7.3,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    legend_axis = figure.add_axes(
        [
            line_panel_position.x0,
            line_panel_position.y1 + 0.012,
            line_panel_position.width,
            0.058,
        ]
    )
    legend_axis.axis("off")
    legend_axis.legend(
        line_handles,
        line_labels,
        fontsize=5.7,
        frameon=False,
        ncol=2,
        mode="expand",
        loc="center",
        borderaxespad=0.0,
        handlelength=1.6,
        handletextpad=0.35,
        columnspacing=0.8,
    )

    upper_map_position = map_axes["ERA5"].get_position()
    if speed_image is not None:
        speed_axis = figure.add_axes(
            [
                upper_map_position.x0,
                upper_map_position.y0 - 0.075,
                upper_map_position.width,
                0.018,
            ]
        )
        speed_cbar = figure.colorbar(
            speed_image, cax=speed_axis, orientation="horizontal"
        )
        speed_cbar.set_label(r"Wind speed (m s$^{-1}$)", fontsize=6.2, labelpad=1.5)
        speed_cbar.ax.tick_params(labelsize=5.6, length=2, pad=1.0)

    if error_image is not None:
        error_axis = figure.add_axes(
            [
                0.915,
                lower_map_position.y0,
                0.014,
                upper_map_position.y1 - lower_map_position.y0,
            ]
        )
        error_cbar = figure.colorbar(error_image, cax=error_axis)
        error_cbar.set_label(r"Vector-error magnitude (m s$^{-1}$)", fontsize=6.4)
        error_cbar.ax.tick_params(labelsize=5.7, length=2)
    figure.suptitle(DISPLAY_TITLE, fontsize=9.2, fontweight="bold", y=0.985)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(figure)

    reported = {
        label: {
            "per_tau_vector_rmse_ms-1": {
                str(int(tau)): float(value) for tau, value in zip(taus, values)
            },
            "mean_all_tau_vector_rmse_ms-1": float(np.mean(values)),
        }
        for label, values in rmse.items()
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(
        json.dumps(
            {
                "event": EVENT_NAME,
                "init_time": EXPECTED_INIT_TIME,
                "display_tau_hours": DISPLAY_TAU,
                "field": "10-m wind vector",
                "evaluation_bbox": {
                    "lat_min": MAP_EXTENT[2], "lat_max": MAP_EXTENT[3],
                    "lon_min": MAP_EXTENT[0], "lon_max": MAP_EXTENT[1],
                },
                "taus_hours": [int(tau) for tau in taus],
                "metrics": reported,
                "selection_note": SELECTION_NOTE,
                "source_provenance": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in source_paths.values()
                },
                "model_provenance": {
                    label: case["provenance"] for label, case in cases.items()
                },
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUTPUT}")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    main()
