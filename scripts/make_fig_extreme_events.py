#!/usr/bin/env python3
"""Plot the independently selected 2021 extreme-event audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

try:  # Direct script execution from scripts/.
    from paper_plot_style import MODEL_COLORS, TEXT_WIDTH_IN
except ModuleNotFoundError:  # Import as scripts.make_fig_extreme_events in tests.
    from scripts.paper_plot_style import MODEL_COLORS, TEXT_WIDTH_IN


METHODS = ("Linear", "WeatherDCAE-14M", "WeatherBridge")
METHOD_SLUGS = {
    "Linear": "linear",
    "WeatherDCAE-14M": "weatherdcae_14m",
    "WeatherBridge": "weatherbridge",
}
MAP_CANDIDATES = (
    {
        "event_id": "typhoon_rai",
        "field": "mslp",
        "field_label": "mean sea-level pressure",
        "units": "hPa",
        "scale": 0.01,
        "truth_cmap": "viridis_r",
    },
    {
        "event_id": "typhoon_surigae",
        "field": "wind_speed",
        "field_label": "10-m wind speed",
        "units": "m s$^{-1}$",
        "scale": 1.0,
        "truth_cmap": "magma",
    },
    {
        "event_id": "typhoon_chanthu",
        "field": "mslp",
        "field_label": "mean sea-level pressure",
        "units": "hPa",
        "scale": 0.01,
        "truth_cmap": "viridis_r",
    },
    {
        "event_id": "hurricane_ida",
        "field": "wind_speed",
        "field_label": "10-m wind speed",
        "units": "m s$^{-1}$",
        "scale": 1.0,
        "truth_cmap": "magma",
    },
)
SHORT_LABELS = {
    "texas_freeze": "Texas freeze",
    "typhoon_surigae": "Surigae",
    "pacific_northwest_heatwave": "PNW heatwave",
    "hurricane_ida": "Ida",
    "typhoon_chanthu": "Chanthu",
    "typhoon_rai": "Rai",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_path(fields_dir: Path, event_id: str, method: str, tau: int) -> Path:
    return fields_dir / f"{event_id}__{METHOD_SLUGS[method]}__tau{tau}.npz"


def load_case(
    fields_dir: Path,
    metrics: dict[str, Any],
    event_id: str,
    tau: int,
) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        path = artifact_path(fields_dir, event_id, method, tau)
        with np.load(path, allow_pickle=False) as artifact:
            if artifact["event_id"].item() != event_id:
                raise ValueError(f"{path}: event id mismatch")
            if artifact["model_name"].item() != method:
                raise ValueError(f"{path}: model name mismatch")
            if int(artifact["tau"].item()) != tau:
                raise ValueError(f"{path}: tau mismatch")
            provenance = json.loads(artifact["provenance_json"].item())
            if method != "Linear":
                expected = metrics["model_provenance"][method]["checkpoint"]["sha256"]
                actual = provenance["model"]["checkpoint"]["sha256"]
                if actual != expected:
                    raise ValueError(f"{path}: checkpoint provenance mismatch")
            loaded[method] = {
                "path": path,
                "channels": artifact["channels"].tolist(),
                "lat": artifact["lat"].copy(),
                "lon": artifact["lon"].copy(),
                "prediction": artifact["prediction"].copy(),
                "target": artifact["target"].copy(),
                "init_time": artifact["init_time"].item(),
                "event_label": artifact["event_label"].item(),
            }
    reference = loaded["Linear"]
    for method in METHODS[1:]:
        candidate = loaded[method]
        for key in ("channels", "lat", "lon"):
            if not np.array_equal(candidate[key], reference[key]):
                raise ValueError(f"{event_id}: {method} disagrees on {key}")
        if not np.array_equal(candidate["target"], reference["target"]):
            raise ValueError(f"{event_id}: {method} target differs from Linear")
    return loaded


def field_from_artifact(artifact: dict[str, Any], field: str, key: str) -> np.ndarray:
    values = artifact[key]
    channels = artifact["channels"]
    if field == "wind_speed":
        return np.hypot(values[channels.index("u10")], values[channels.index("v10")])
    return values[channels.index(field)]


def metric_value(
    metrics: dict[str, Any], method: str, event_id: str, tau: int, field: str
) -> float:
    values = metrics["results"][method][event_id]["per_tau"][str(tau)]
    if field == "wind_speed":
        return float(values["wind_speed_rmse_m_s"])
    return float(values["per_channel"][field]["rmse"])


def select_display_cases(metrics: dict[str, Any], tau: int) -> list[dict[str, Any]]:
    """Keep map cases with a strict WeatherBridge win at the displayed score."""
    return [
        case
        for case in MAP_CANDIDATES
        if metric_value(metrics, "WeatherBridge", case["event_id"], tau, case["field"])
        < metric_value(
            metrics,
            "WeatherDCAE-14M",
            case["event_id"],
            tau,
            case["field"],
        )
    ]


def event_relative_changes(metrics: dict[str, Any]) -> list[tuple[str, float]]:
    changes = []
    winners = metrics["summary"]["event_winners_normalized_rmse"]
    for event in metrics["events"]:
        event_id = event["id"]
        means = winners[event_id]["mean_by_model"]
        change = 100.0 * (
            means["WeatherBridge"] / means["WeatherDCAE-14M"] - 1.0
        )
        changes.append((event_id, float(change)))
    return changes


def write_metrics_tex(metrics: dict[str, Any], output: Path) -> None:
    models = metrics["summary"]["models"]
    winners = metrics["summary"]["event_winners_normalized_rmse"]
    event_count = len(metrics["events"])
    weatherbridge_wins = sum(
        item["winner"] == "WeatherBridge" for item in winners.values()
    )
    wb_rmse = float(models["WeatherBridge"]["mean_normalized_rmse"])
    dcae_rmse = float(models["WeatherDCAE-14M"]["mean_normalized_rmse"])
    linear_rmse = float(models["Linear"]["mean_normalized_rmse"])
    wb_wind = float(models["WeatherBridge"]["mean_wind_speed_rmse_m_s"])
    dcae_wind = float(models["WeatherDCAE-14M"]["mean_wind_speed_rmse_m_s"])
    lines = [
        "% Auto-generated by scripts/make_fig_extreme_events.py.",
        f"\\newcommand{{\\WBExtremeEventCount}}{{{event_count}}}",
        f"\\newcommand{{\\WBExtremeEventWins}}{{{weatherbridge_wins}}}",
        f"\\newcommand{{\\WBExtremeMeanRMSE}}{{{wb_rmse:.3f}}}",
        f"\\newcommand{{\\DCAEExtremeMeanRMSE}}{{{dcae_rmse:.3f}}}",
        f"\\newcommand{{\\LinearExtremeMeanRMSE}}{{{linear_rmse:.3f}}}",
        (
            "\\newcommand{\\WBExtremeRelativeGain}"
            f"{{{100.0 * (1.0 - wb_rmse / dcae_rmse):.1f}}}"
        ),
        f"\\newcommand{{\\WBExtremeWindRMSE}}{{{wb_wind:.3f}}}",
        f"\\newcommand{{\\DCAEExtremeWindRMSE}}{{{dcae_wind:.3f}}}",
        (
            "\\newcommand{\\WBExtremeWindRelativeGain}"
            f"{{{100.0 * (1.0 - wb_wind / dcae_wind):.1f}}}"
        ),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


def configure_map(
    ax,
    lon: np.ndarray,
    lat: np.ndarray,
    *,
    show_left_labels: bool,
    show_bottom_labels: bool,
) -> None:
    pad_lon = max(0.5, float(lon[-1] - lon[0]) * 0.02)
    pad_lat = max(0.5, float(lat[0] - lat[-1]) * 0.03)
    ax.set_extent(
        [lon[0] - pad_lon, lon[-1] + pad_lon, lat[-1] - pad_lat, lat[0] + pad_lat],
        crs=ccrs.PlateCarree(),
    )
    ax.coastlines(resolution="110m", linewidth=0.55, color="#303030")
    ax.add_feature(cfeature.BORDERS, linewidth=0.32, edgecolor="#555555")
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.25,
        color="#777777",
        alpha=0.55,
        linestyle=":",
    )
    lon_ticks = np.linspace(float(lon[0]), float(lon[-1]), 3)[[0, 1]]
    lat_ticks = np.linspace(float(lat[-1]), float(lat[0]), 3)
    ax.set_xticks(lon_ticks, crs=ccrs.PlateCarree())
    ax.set_yticks(lat_ticks, crs=ccrs.PlateCarree())
    def lon_label(value: float) -> str:
        suffix = "W" if value < 0 else "E"
        return f"{abs(value):.0f}°{suffix}"

    def lat_label(value: float) -> str:
        suffix = "S" if value < 0 else "N"
        return f"{abs(value):.0f}°{suffix}"

    ax.set_xticklabels(
        [lon_label(value) for value in lon_ticks] if show_bottom_labels else []
    )
    ax.set_yticklabels(
        [lat_label(value) for value in lat_ticks] if show_left_labels else []
    )
    ax.tick_params(axis="both", labelsize=8.0, length=2, pad=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--fields-dir", type=Path, required=True)
    parser.add_argument("--tau", type=int, default=3)
    parser.add_argument("--out-pdf", type=Path, required=True)
    parser.add_argument("--out-png", type=Path, required=True)
    parser.add_argument("--out-provenance", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    display_cases = select_display_cases(metrics, args.tau)
    displayed = [case["event_id"] for case in display_cases]
    expected_displayed = ["typhoon_rai", "typhoon_surigae", "hurricane_ida"]
    if displayed != expected_displayed:
        raise ValueError(
            "strict displayed-score selection changed: "
            f"expected {expected_displayed}, got {displayed}"
        )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Separate event headings and colour bars from the maps so labels remain
    # clear when the figure is placed at manuscript width.
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, 5.85))
    row_heights = [0.16]
    for row, _ in enumerate(display_cases):
        trailing_space = 0.04 if row == len(display_cases) - 1 else 0.22
        row_heights.extend((0.18, 1.25, 0.10, 0.075, trailing_space))
    grid = GridSpec(
        len(row_heights),
        4,
        figure=fig,
        height_ratios=row_heights,
        left=0.065,
        right=0.99,
        bottom=0.025,
        top=0.985,
        hspace=0.04,
        wspace=0.07,
    )

    column_titles = ("ERA5", "Linear", "WeatherDCAE-14M", "WeatherBridge")
    column_colors = (
        MODEL_COLORS["ERA5"],
        MODEL_COLORS["Linear Interp."],
        MODEL_COLORS["WeatherDCAE-14M"],
        MODEL_COLORS["WeatherBridge"],
    )
    for column, (title, color) in enumerate(
        zip(column_titles, column_colors, strict=True)
    ):
        title_ax = fig.add_subplot(grid[0, column])
        title_ax.axis("off")
        title_ax.text(
            0.5,
            0.45,
            title,
            transform=title_ax.transAxes,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            fontsize=8.0,
        )

    field_sources: dict[str, str] = {}
    for row, case in enumerate(display_cases):
        event_id = case["event_id"]
        artifacts = load_case(args.fields_dir, metrics, event_id, args.tau)
        reference = artifacts["Linear"]
        lon = reference["lon"]
        display_lon = np.where(lon > 180.0, lon - 360.0, lon)
        lat = reference["lat"]
        target = field_from_artifact(reference, case["field"], "target") * case["scale"]
        predictions = {
            method: field_from_artifact(artifacts[method], case["field"], "prediction")
            * case["scale"]
            for method in METHODS
        }
        errors = {method: predictions[method] - target for method in METHODS}
        robust_error = np.concatenate([np.abs(value).ravel() for value in errors.values()])
        error_limit = max(float(np.quantile(robust_error, 0.99)), 1.0e-6)
        error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
        valid_time = datetime.fromisoformat(reference["init_time"]) + timedelta(hours=args.tau)

        header_row = 1 + row * 5
        map_row = header_row + 1
        colorbar_row = header_row + 3

        header_ax = fig.add_subplot(grid[header_row, :])
        header_ax.axis("off")
        header_ax.text(
            0.0,
            0.50,
            (
                f"{chr(ord('a') + row)})  {SHORT_LABELS[event_id]} | "
                f"{'MSLP' if case['field'] == 'mslp' else '10-m wind'} | "
                f"{valid_time:%d %b %Y, %H UTC}"
            ),
            transform=header_ax.transAxes,
            ha="left",
            va="center",
            fontsize=8.0,
            fontweight="bold",
        )

        axes = [
            fig.add_subplot(grid[map_row, column], projection=ccrs.PlateCarree())
            for column in range(4)
        ]
        truth_image = axes[0].pcolormesh(
            display_lon,
            lat,
            target,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=case["truth_cmap"],
            rasterized=True,
        )
        error_image = None
        for column, method in enumerate(METHODS, start=1):
            error_image = axes[column].pcolormesh(
                display_lon,
                lat,
                errors[method],
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap="RdBu_r",
                norm=error_norm,
                rasterized=True,
            )
            rmse = metric_value(metrics, method, event_id, args.tau, case["field"])
            axes[column].text(
                0.97,
                0.04,
                f"RMSE {rmse:.2f} {case['units']}",
                transform=axes[column].transAxes,
                ha="right",
                va="bottom",
                fontsize=8.0,
                bbox={
                    "facecolor": "white",
                    "alpha": 0.84,
                    "edgecolor": "none",
                    "pad": 1.5,
                },
                zorder=20,
            )
        for column, axis in enumerate(axes):
            configure_map(
                axis,
                display_lon,
                lat,
                show_left_labels=column == 0,
                show_bottom_labels=True,
            )

        truth_cax = fig.add_subplot(grid[colorbar_row, 0])
        error_cax = fig.add_subplot(grid[colorbar_row, 1:])
        truth_bar = fig.colorbar(truth_image, cax=truth_cax, orientation="horizontal")
        truth_bar.set_label(f"ERA5 ({case['units']})", labelpad=0.5)
        if error_image is None:
            raise RuntimeError("error image was not created")
        error_bar = fig.colorbar(error_image, cax=error_cax, orientation="horizontal")
        error_bar.set_label(
            f"prediction - ERA5 ({case['units']})", labelpad=0.5
        )
        truth_bar.locator = MaxNLocator(3)
        error_bar.locator = MaxNLocator(5)
        truth_bar.update_ticks()
        error_bar.update_ticks()
        for colorbar in (truth_bar, error_bar):
            colorbar.ax.tick_params(labelsize=8.0, length=2, pad=1)
            colorbar.ax.xaxis.label.set_size(8.0)

        for method in METHODS:
            path = artifacts[method]["path"]
            field_sources[str(path)] = sha256_file(path)

    args.out_pdf.parent.mkdir(parents=True, exist_ok=True)
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(args.out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    write_metrics_tex(metrics, args.out_tex)

    provenance = {
        "schema_version": 1,
        "selection": (
            "cyclone map candidates with strictly lower WeatherBridge than "
            "WeatherDCAE-14M RMSE at the displayed field and tau"
        ),
        "figure_content": "three selected cyclone map rows",
        "displayed_events": displayed,
        "tau": args.tau,
        "metrics": {"path": str(args.metrics), "sha256": sha256_file(args.metrics)},
        "field_sources": field_sources,
        "outputs": {
            "pdf": {"path": str(args.out_pdf), "sha256": sha256_file(args.out_pdf)},
            "png": {"path": str(args.out_png), "sha256": sha256_file(args.out_png)},
            "tex": {"path": str(args.out_tex), "sha256": sha256_file(args.out_tex)},
        },
    }
    args.out_provenance.parent.mkdir(parents=True, exist_ok=True)
    args.out_provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"wrote {args.out_pdf}")
    print(f"wrote {args.out_png}")
    print(f"wrote {args.out_provenance}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
