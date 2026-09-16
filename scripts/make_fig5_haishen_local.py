"""Render the matched-model Haishen pressure case study.

The figure compares ERA5, linear interpolation, WeatherDCAE-14M, PixelAttn-VFI
and WeatherBridge at the five interior hours. Model panels come
from checkpoint-backed exports on the same grid and are accepted only when
their ERA5 targets match the historical case bundle.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from paper_plot_style import COLUMN_WIDTH_IN
from paper_plot_style import MODEL_COLORS, add_panel_labels


ROOT = Path(__file__).resolve().parents[1]
CASE_DIRS = {
    "WeatherDCAE-14M": ROOT / "metrics" / "case_studies_weatherdcae_14m",
    "PixelAttn-VFI": ROOT / "metrics" / "case_studies_pixelattn_vfi",
    "WeatherBridge": ROOT / "metrics" / "case_studies_weatherbridge",
}
FIELD_SPECS = {
    "mslp": {
        "base": ROOT / "demo" / "precomputed" / "typhoon_haishen" / "East_Asia__mslp.npz",
        "output": ROOT / "paper" / "images" / "fig_haishen_mslp.pdf",
        "summary": CASE_DIRS["WeatherBridge"] / "typhoon_haishen" / "mslp_summary.json",
        "title": "Typhoon Haishen, 7 September 2020 - mean sea-level pressure",
        "units": "hPa",
        "scale": 1.0,
        "decimals": 3,
    },
}
EVAL_BBOX = {
    "lat_min": 25.0,
    "lat_max": 48.0,
    "lon_min": 115.0,
    "lon_max": 145.0,
}
EXPECTED_INIT_TIME = "2020-09-07T00:00:00"
LON_TICKS = (105, 120, 135)
LAT_TICKS = (25, 40, 55)
DISPLAY_ROWS = [
    "ERA5",
    "Linear Interp.",
    "WeatherDCAE-14M",
    "PixelAttn-VFI",
    "WeatherBridge",
]
ROW_LABELS = {
    "ERA5": "ERA5",
    "Linear Interp.": "Linear",
    "WeatherDCAE-14M": "DCAE-14M",
    "PixelAttn-VFI": "PixelAttn",
    "WeatherBridge": "WeatherBridge",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model_case(
    path: Path,
    *,
    label: str,
    taus: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    truth: np.ndarray,
) -> tuple[np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as case:
        if case["methods"].tolist() != [label]:
            raise ValueError(f"{path}: expected only {label}")
        if case["init_time"].item() != EXPECTED_INIT_TIME:
            raise ValueError(f"{path}: unexpected Haishen initial time")
        if not np.array_equal(case["taus"], taus):
            raise ValueError(f"{path}: tau axis does not match base")
        if not np.allclose(case["lat"], lat) or not np.allclose(case["lon"], lon):
            raise ValueError(f"{path}: spatial grid does not match base")
        if not np.allclose(case["truth"], truth, rtol=0.0, atol=1e-4):
            raise ValueError(f"{path}: ERA5 target does not match base")
        provenance = json.loads(case["provenance_json"].item())
        if (
            provenance.get("model_label") != label
            or provenance.get("schema_version") != 3
            or provenance.get("grid_convention")
            != "wb2_0p25_pair_average_cell_centres_v1"
        ):
            raise ValueError(f"{path}: model or grid provenance mismatch")
        return np.array(case["panels"], copy=True), provenance


def _render_field(field: str, spec: dict) -> None:
    base_path = spec["base"]
    with np.load(base_path, allow_pickle=True) as base:
        methods = list(base["methods"])
        taus = np.array(base["taus"], copy=True)
        # Historical labels are 0.125 degrees west and south of the actual
        # 2x2 block-average cell centres.
        lat = np.array(base["lat"], copy=True) + np.float32(0.125)
        lon = np.array(base["lon"], copy=True) + np.float32(0.125)
        panels = np.array(base["panels"], copy=True)

    truth_idx = methods.index("ERA5")
    truth = panels[truth_idx]
    model_provenance = {}
    source_paths = [base_path]
    for label in DISPLAY_ROWS[2:]:
        source = CASE_DIRS[label] / "typhoon_haishen" / f"{field}.npz"
        model_panels, provenance = _load_model_case(
            source,
            label=label,
            taus=taus,
            lat=lat,
            lon=lon,
            truth=truth,
        )
        methods.append(label)
        panels = np.concatenate([panels, model_panels], axis=0)
        model_provenance[label] = provenance
        source_paths.append(source)

    panels *= float(spec["scale"])
    lat_mask = (lat >= EVAL_BBOX["lat_min"]) & (lat <= EVAL_BBOX["lat_max"])
    lon_mask = (lon >= EVAL_BBOX["lon_min"]) & (lon <= EVAL_BBOX["lon_max"])
    crop = panels[:, :, lat_mask, :][:, :, :, lon_mask]
    truth_crop = crop[truth_idx]
    rmse = np.sqrt(np.mean((crop - truth_crop[None]) ** 2, axis=(-2, -1)))

    source_key = {"Linear Interp.": "Bilinear"}
    model_indices = [
        methods.index(source_key.get(label, label)) for label in DISPLAY_ROWS
    ]
    displayed_taus = (2, 3, 4)
    tau_indices = [int(np.flatnonzero(taus == tau)[0]) for tau in displayed_taus]
    n_rows = len(displayed_taus)
    n_cols = len(DISPLAY_ROWS)
    display_stack = panels[model_indices][:, tau_indices].ravel()
    vmin = float(np.percentile(display_stack, 1))
    vmax = float(np.percentile(display_stack, 99))

    panel_w = COLUMN_WIDTH_IN / n_cols
    fig = plt.figure(figsize=(COLUMN_WIDTH_IN, n_rows * panel_w * 0.93))
    grid = GridSpec(n_rows, n_cols, wspace=0.055, hspace=0.115, figure=fig)
    extent = [lon[0], lon[-1], lat[-1], lat[0]]
    panel_axes = []
    image = None
    for row, (tau, tau_index) in enumerate(zip(displayed_taus, tau_indices)):
        for column, (method, model_index) in enumerate(
            zip(DISPLAY_ROWS, model_indices)
        ):
            axis = fig.add_subplot(grid[row, column])
            panel_axes.append(axis)
            data = panels[model_index, tau_index]
            image = axis.imshow(
                data,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                extent=extent,
                origin="upper",
                aspect="auto",
            )
            axis.contour(
                lon,
                lat,
                data,
                levels=np.linspace(vmin, vmax, 8),
                colors="black",
                linewidths=1.0,
                alpha=0.42,
            )
            if row == 0:
                axis.set_title(
                    ROW_LABELS[method],
                    fontsize=7.2,
                    fontweight="bold",
                    color=MODEL_COLORS[method],
                )
            if column == 0:
                axis.set_ylabel(
                    rf"$\tau={tau}\,\mathrm{{h}}$",
                    fontsize=7.2,
                    fontweight="bold",
                    labelpad=15.0,
                )
            if method == "WeatherBridge":
                for spine in axis.spines.values():
                    spine.set_color(MODEL_COLORS[method])
                    spine.set_linewidth(1.4)
            if method != "ERA5":
                value = rmse[model_index, tau_index]
                axis.text(
                    0.98,
                    0.04,
                    f"RMSE={value:.{spec['decimals']}f}",
                    transform=axis.transAxes,
                    fontsize=5.8,
                    ha="right",
                    va="bottom",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.78,
                        "linewidth": 0,
                        "boxstyle": "round,pad=0.16",
                    },
                )
            if row == n_rows - 1:
                axis.set_xticks(LON_TICKS)
                axis.set_xticklabels([f"{value}°E" for value in LON_TICKS])
            else:
                axis.set_xticks([])
            if column == 0:
                axis.set_yticks(LAT_TICKS)
                axis.set_yticklabels([f"{value}°N" for value in LAT_TICKS])
            else:
                axis.set_yticks([])
            axis.tick_params(axis="x", labelsize=5.6, length=2.0, pad=1.2)
            axis.tick_params(axis="y", labelsize=6.1, length=2.0, pad=1.2)

    add_panel_labels(panel_axes, inside=True, fontsize=7.4)
    fig.suptitle(spec["title"], fontsize=9.2, fontweight="bold", y=0.998)
    fig.subplots_adjust(right=0.923, top=0.885, left=0.125, bottom=0.075)
    if image is not None:
        color_axis = fig.add_axes([0.938, 0.12, 0.014, 0.68])
        colorbar = fig.colorbar(image, cax=color_axis)
        colorbar.ax.tick_params(labelsize=6.6)
        colorbar.set_label(spec["units"], fontsize=7.6)

    output = spec["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=450, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    reported = {}
    for label in DISPLAY_ROWS[1:]:
        values = rmse[methods.index(source_key.get(label, label))]
        record = {
            "per_tau": {
                str(int(tau)): float(value) for tau, value in zip(taus, values)
            },
            "mean_all_tau": float(np.mean(values)),
        }
        if field == "mslp":
            record["per_tau_hpa"] = dict(record["per_tau"])
            record["mean_all_tau_hpa"] = record["mean_all_tau"]
        reported[label] = record
    summary = spec["summary"]
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            {
                "field": field,
                "units": spec["units"],
                "event": "Typhoon Haishen",
                "date": "2020-09-07",
                "evaluation_bbox": EVAL_BBOX,
                "taus_hours": [int(tau) for tau in taus],
                "metrics": reported,
                "source_provenance": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in source_paths
                },
                "model_provenance": model_provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {output}")
    print(f"wrote {summary}")


def main() -> None:
    for field, spec in FIELD_SPECS.items():
        _render_field(field, spec)


if __name__ == "__main__":
    main()
