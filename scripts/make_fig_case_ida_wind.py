"""Generate the Hurricane Ida case-study figure.

The evidence-dense layout retains the full-domain ERA5 context, reports the
fixed-crop error at every interior hour, and zooms the Gulf at tau=3 for a
direct structural comparison. The 2021 anchor window is excluded from model
training but was inspected during paper development.

Field shown: 10-m wind-speed magnitude
|U10| = sqrt(u10^2 + v10^2) (m s^-1).

Reads `demo/precomputed_2021/hurricane_ida_2021/North_America__{u10,v10}.npz`
and the exported WeatherBridge panels under
`metrics/case_studies_weatherbridge/hurricane_ida_2021/`.
The NPZ tiles were produced by `demo/demo_precompute.py --year 2021`
on cloudru against the wb2_2021 memmap.

Output:
  paper/images/fig_case_ida_wind.pdf
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from paper_plot_style import MODEL_COLORS
from paper_plot_style import MODEL_LINESTYLES, MODEL_MARKERS, add_panel_labels

ROOT = Path(__file__).resolve().parents[1]
SRC_U = ROOT / "demo" / "precomputed_2021" / "hurricane_ida_2021" / "North_America__u10.npz"
SRC_V = ROOT / "demo" / "precomputed_2021" / "hurricane_ida_2021" / "North_America__v10.npz"
WEATHERBRIDGE_ROOT = (
    ROOT
    / "metrics"
    / "case_studies_weatherbridge"
    / "hurricane_ida_2021"
)
WEATHERBRIDGE_U = WEATHERBRIDGE_ROOT / "u10.npz"
WEATHERBRIDGE_V = WEATHERBRIDGE_ROOT / "v10.npz"
OUT_PDF = ROOT / "paper" / "images" / "fig_case_ida_wind.pdf"
OUT_METRICS = WEATHERBRIDGE_ROOT / "wind_speed_summary.json"
EVAL_BBOX = {
    "lat_min": 20.0,
    "lat_max": 32.0,
    "lon_min": 260.0,
    "lon_max": 285.0,
}
LON_TICKS = (240, 265, 290)
LAT_TICKS = (25, 40, 55)

CURVE_METHODS = (
    "Linear Interp.",
    "FuXi",
    "S-DYff",
    "WeatherDCAE-14M",
    "WeatherBridge",
)
MAP_METHODS = (
    "ERA5",
    "WeatherDCAE-14M",
    "WeatherBridge",
)
SOURCE_KEY = {
    "Linear Interp.": "Bilinear",
    "WeatherDCAE-14M": "WeatherDCAE",
}
METHOD_COLORS = {
    name: MODEL_COLORS[name] for name in ("ERA5", *CURVE_METHODS)
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_transport_component(
    path: Path,
    *,
    base: np.lib.npyio.NpzFile,
    arch: str,
) -> tuple[np.ndarray, dict]:
    detail = np.load(path, allow_pickle=False)
    artifact_methods = detail["methods"].tolist()
    if len(artifact_methods) != 1:
        raise ValueError(f"{path}: expected one model panel")
    if detail["model_arch"].item() != arch:
        raise ValueError(f"{path}: expected {arch} architecture")
    if detail["init_time"].item() != "2021-08-29T12:00:00":
        raise ValueError(f"{path}: unexpected Ida initial time")
    if not np.array_equal(detail["taus"], base["taus"]):
        raise ValueError(f"{path}: taus do not match base case")
    for key in ("lat", "lon"):
        expected = base[key] + np.float32(0.125)
        if not np.allclose(detail[key], expected, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{path}: corrected {key} does not match base case")
    truth = base["panels"][list(base["methods"]).index("ERA5")]
    if not np.allclose(detail["truth"], truth, rtol=0.0, atol=1.0e-4):
        raise ValueError(f"{path}: ERA5 target does not match base case")
    provenance = json.loads(detail["provenance_json"].item())
    if (
        provenance.get("model_label") != artifact_methods[0]
        or provenance.get("model_arch") != arch
        or provenance.get("schema_version") != 3
        or provenance.get("grid_convention")
        != "wb2_0p25_pair_average_cell_centres_v1"
    ):
        raise ValueError(f"{path}: model provenance mismatch")
    return detail["panels"], provenance


def _load_speed():
    du = np.load(SRC_U, allow_pickle=True)
    dv = np.load(SRC_V, allow_pickle=True)
    for key in ("methods", "taus", "lat", "lon"):
        if not np.array_equal(du[key], dv[key]):
            raise ValueError(f"Ida U/V baseline bundles disagree on {key}")
    if du["panels"].shape != dv["panels"].shape:
        raise ValueError("Ida U/V baseline panel shapes differ")
    methods = list(du["methods"])
    taus = list(du["taus"])
    # Correct the legacy labels in the baseline bundle to the actual 2x2
    # block-average cell centres used by the memmap fields.
    lat = du["lat"] + np.float32(0.125)
    lon = du["lon"] + np.float32(0.125)
    u = du["panels"]
    v = dv["panels"]
    model_provenance = {}
    model_u, provenance_u = _load_transport_component(
        WEATHERBRIDGE_U,
        base=du,
        arch="flow_pp3",
    )
    model_v, provenance_v = _load_transport_component(
        WEATHERBRIDGE_V,
        base=dv,
        arch="flow_pp3",
    )
    methods.append("WeatherBridge")
    u = np.concatenate([u, model_u], axis=0)
    v = np.concatenate([v, model_v], axis=0)
    model_provenance["WeatherBridge"] = {
        "u10": provenance_u,
        "v10": provenance_v,
    }
    speed = np.sqrt(u * u + v * v)
    truth_idx = methods.index("ERA5")
    lat_mask = (lat >= EVAL_BBOX["lat_min"]) & (lat <= EVAL_BBOX["lat_max"])
    lon_mask = (lon >= EVAL_BBOX["lon_min"]) & (lon <= EVAL_BBOX["lon_max"])
    crop = speed[:, :, lat_mask][:, :, :, lon_mask]
    diff = crop - crop[truth_idx][None]
    rmse = np.sqrt((diff ** 2).mean(axis=(-2, -1)))
    return methods, taus, lat, lon, speed, rmse, model_provenance


def main() -> None:
    methods, taus, lat, lon, panels, rmse, model_provenance = _load_speed()

    map_indices = [methods.index(SOURCE_KEY.get(name, name)) for name in MAP_METHODS]
    tau_values = np.asarray(taus, dtype=int)
    tau3_index = tau_values.tolist().index(3)
    lat_mask = (lat >= EVAL_BBOX["lat_min"]) & (lat <= EVAL_BBOX["lat_max"])
    lon_mask = (lon >= EVAL_BBOX["lon_min"]) & (lon <= EVAL_BBOX["lon_max"])
    lat_crop = lat[lat_mask]
    lon_crop = lon[lon_mask]
    truth = panels[methods.index("ERA5"), tau3_index]
    truth_crop = truth[lat_mask][:, lon_mask]
    crop_max_index = np.unravel_index(int(np.argmax(truth_crop)), truth_crop.shape)
    lat_index = np.flatnonzero(lat_mask)[crop_max_index[0]]
    lon_index = np.flatnonzero(lon_mask)[crop_max_index[1]]
    reference_max = float(truth[lat_index, lon_index])

    reported = {}
    for label in CURVE_METHODS:
        source_label = SOURCE_KEY.get(label, label)
        method_index = methods.index(source_label)
        values = rmse[method_index]
        reported[label] = {
            "per_tau_m_s": {
                str(int(tau)): float(value)
                for tau, value in zip(taus, values)
            },
            "mean_all_tau_m_s": float(np.mean(values)),
            "tau3_error_at_truth_max_m_s": float(
                panels[method_index, tau3_index, lat_index, lon_index]
                - reference_max
            ),
        }

    crop_stack = panels[map_indices, tau3_index][:, lat_mask][:, :, lon_mask]
    vmin = 0.0
    vmax = max(20.0, 5.0 * float(np.ceil(np.nanmax(crop_stack) / 5.0)))
    contour_levels = np.linspace(vmin, vmax, 8)
    full_extent = [lon[0], lon[-1], lat[-1], lat[0]]
    crop_extent = [lon_crop[0], lon_crop[-1], lat_crop[-1], lat_crop[0]]

    fig = plt.figure(figsize=(12.8, 7.0), constrained_layout=True)
    grid = GridSpec(
        2,
        7,
        figure=fig,
        width_ratios=[1, 1, 1, 1, 1, 1, 0.08],
        height_ratios=[0.93, 1.0],
    )

    context_ax = fig.add_subplot(grid[0, :3])
    context_im = context_ax.imshow(
        truth,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        extent=full_extent,
        origin="upper",
        aspect="auto",
    )
    context_ax.contour(
        lon,
        lat,
        truth,
        levels=contour_levels,
        colors="black",
        linewidths=1.0,
        alpha=0.35,
    )
    context_ax.add_patch(
        plt.Rectangle(
            (EVAL_BBOX["lon_min"], EVAL_BBOX["lat_min"]),
            EVAL_BBOX["lon_max"] - EVAL_BBOX["lon_min"],
            EVAL_BBOX["lat_max"] - EVAL_BBOX["lat_min"],
            fill=False,
            edgecolor=MODEL_COLORS["WeatherBridge"],
            linewidth=1.8,
            zorder=10,
        )
    )
    context_ax.plot(
        lon[lon_index],
        lat[lat_index],
        marker="x",
        color="white",
        markeredgewidth=1.7,
        markersize=7,
        zorder=11,
    )
    context_ax.set_title(r"ERA5 context at $\tau=3$ h", fontsize=11)
    context_ax.set_xticks(LON_TICKS)
    context_ax.set_xticklabels([f"{360 - value}°W" for value in LON_TICKS])
    context_ax.set_yticks(LAT_TICKS)
    context_ax.set_yticklabels([f"{value}°N" for value in LAT_TICKS])
    context_ax.tick_params(labelsize=8)

    curve_ax = fig.add_subplot(grid[0, 3:6])
    for held_tau in (2, 4):
        curve_ax.axvspan(held_tau - 0.16, held_tau + 0.16, color="#E5E7EB", zorder=0)
    for label in CURVE_METHODS:
        method_index = methods.index(SOURCE_KEY.get(label, label))
        curve_ax.plot(
            tau_values,
            rmse[method_index],
            color=METHOD_COLORS[label],
            linestyle=MODEL_LINESTYLES[label],
            marker=MODEL_MARKERS[label],
            linewidth=2.1 if label == "WeatherBridge" else 1.25,
            markersize=5.2 if label == "WeatherBridge" else 4.2,
            label=label,
            zorder=4 if label == "WeatherBridge" else 2,
        )
    curve_ax.set_title("Gulf crop error across interior hours", fontsize=11)
    curve_ax.set_xlabel(r"Interior hour $\tau$ (h)")
    curve_ax.set_ylabel(r"RMSE (m s$^{-1}$)")
    curve_ax.set_xticks(tau_values)
    curve_ax.grid(axis="y", color="#D1D5DB", linewidth=0.7)
    curve_ax.legend(frameon=False, fontsize=7.3, loc="upper center", ncol=3)
    curve_ax.text(
        0.99,
        0.03,
        "grey bands: held-out hours",
        transform=curve_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#4B5563",
    )

    crop_axes = []
    for column, (label, method_index) in enumerate(zip(MAP_METHODS, map_indices)):
        ax = fig.add_subplot(grid[1, 2 * column : 2 * column + 2])
        crop_axes.append(ax)
        crop_data = panels[method_index, tau3_index][lat_mask][:, lon_mask]
        ax.imshow(
            crop_data,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            extent=crop_extent,
            origin="upper",
            aspect="auto",
        )
        ax.contour(
            lon_crop,
            lat_crop,
            crop_data,
            levels=contour_levels,
            colors="black",
            linewidths=0.65,
            alpha=0.38,
        )
        ax.plot(
            lon[lon_index],
            lat[lat_index],
            marker="x",
            color="white",
            markeredgewidth=1.4,
            markersize=6,
        )
        ax.set_title(
            "ERA5 target" if label == "ERA5" else label,
            color=METHOD_COLORS[label],
        )
        if label == "ERA5":
            annotation = f"ERA5 max={reference_max:.1f} m s$^{{-1}}$"
        else:
            values = reported[label]
            annotation = (
                f"RMSE={values['per_tau_m_s']['3']:.3f}\n"
                f"error at ERA5 max={values['tau3_error_at_truth_max_m_s']:+.1f} m s$^{{-1}}$"
            )
        ax.text(
            0.03,
            0.04,
            annotation,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.5,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2},
        )
        ax.set_xticks((260, 270, 280))
        ax.set_xticklabels(("100°W", "90°W", "80°W"))
        if column == 0:
            ax.set_yticks((20, 26, 32))
            ax.set_yticklabels(("20°N", "26°N", "32°N"))
        else:
            ax.set_yticks([])
        ax.tick_params(labelsize=7.5)
        if label == "WeatherBridge":
            for spine in ax.spines.values():
                spine.set_color(METHOD_COLORS[label])
                spine.set_linewidth(1.5)

    colorbar_ax = fig.add_subplot(grid[:, 6])
    colorbar = fig.colorbar(context_im, cax=colorbar_ax)
    colorbar.set_label(r"Wind speed (m s$^{-1}$)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)

    add_panel_labels([context_ax, curve_ax, *crop_axes], inside=True)
    fig.suptitle(
        "Hurricane Ida, 29 August 2021 — 10-m wind-speed interpolation",
        fontsize=13,
        fontweight="bold",
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=450, bbox_inches="tight")
    plt.close(fig)

    OUT_METRICS.write_text(
        json.dumps(
            {
                "field": "10m_wind_speed",
                "units": "m s-1",
                "event": "Hurricane Ida",
                "date": "2021-08-29",
                "anchor_hours_utc": [12, 18],
                "evaluation_bbox": EVAL_BBOX,
                "taus_hours": [int(tau) for tau in taus],
                "truth_max_tau3": {
                    "latitude": float(lat[lat_index]),
                    "longitude": float(lon[lon_index]),
                    "speed_m_s": reference_max,
                },
                "metrics": reported,
                "source_provenance": {
                    str(path.relative_to(ROOT)): _sha256(path)
                    for path in (
                        SRC_U,
                        SRC_V,
                        WEATHERBRIDGE_U,
                        WEATHERBRIDGE_V,
                    )
                },
                "model_provenance": model_provenance,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
