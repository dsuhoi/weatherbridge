#!/usr/bin/env python3
"""Render real-data assets for the WeatherBridge graphical overview."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "pnw_temperature_overview.npz"
ASSETS = HERE / "assets"
EXTENT = (210.125, 275.125, 29.875, 61.875)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_axis(axis) -> None:
    axis.set_extent(EXTENT, crs=ccrs.PlateCarree())
    axis.coastlines(
        resolution="50m",
        linewidth=1.25,
        color="#202124",
        alpha=0.92,
        zorder=5,
    )
    axis.set_axis_off()


def render_field(
    field: np.ndarray,
    *,
    latitude: np.ndarray,
    longitude: np.ndarray,
    output: Path,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    projection = ccrs.PlateCarree()
    figure = plt.figure(figsize=(7.2, 3.6), dpi=220, facecolor="white")
    axis = figure.add_axes([0.0, 0.0, 1.0, 1.0], projection=projection)
    axis.pcolormesh(
        longitude,
        latitude,
        field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
        transform=projection,
        rasterized=True,
    )
    configure_axis(axis)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white", pad_inches=0)
    plt.close(figure)


def local_highpass(field: np.ndarray) -> np.ndarray:
    """Match the objective's periodic-longitude, replicated-latitude filter."""
    padded_lon = np.pad(field, ((0, 0), (1, 1)), mode="wrap")
    padded = np.pad(padded_lon, ((1, 1), (0, 0)), mode="edge")
    local_mean = sum(
        padded[row : row + field.shape[0], col : col + field.shape[1]]
        for row in range(3)
        for col in range(3)
    ) / 9.0
    return field - local_mean


def radial_fft_magnitude(
    field: np.ndarray,
    bins: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    magnitude = np.abs(np.fft.rfft2(field - np.mean(field), norm="ortho"))
    fy = np.fft.fftfreq(field.shape[0])[:, None]
    fx = np.fft.rfftfreq(field.shape[1])[None, :]
    radius = np.hypot(fy, fx)
    edges = np.linspace(0.0, float(radius.max()), bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    values = np.empty(bins, dtype=np.float64)
    for index in range(bins):
        mask = (radius >= edges[index]) & (radius < edges[index + 1])
        values[index] = float(np.mean(magnitude[mask])) if np.any(mask) else np.nan
    valid = np.isfinite(values)
    return centres[valid], values[valid]


def render_fft_summary(
    target: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    frequency, target_magnitude = radial_fft_magnitude(target)
    pred_frequency, pred_magnitude = radial_fft_magnitude(prediction)
    if not np.allclose(frequency, pred_frequency):
        raise ValueError("target and prediction FFT bins differ")

    figure, axis = plt.subplots(figsize=(6.8, 3.2), dpi=220, facecolor="white")
    figure.subplots_adjust(left=0.13, right=0.97, bottom=0.2, top=0.93)
    axis.fill_between(
        frequency[1:],
        target_magnitude[1:],
        pred_magnitude[1:],
        color="#e6d8e8",
        alpha=0.7,
        linewidth=0,
    )
    axis.plot(
        frequency[1:],
        target_magnitude[1:],
        color="#202124",
        linewidth=3.0,
        label="ERA5 target",
    )
    axis.plot(
        frequency[1:],
        pred_magnitude[1:],
        color="#d62728",
        linewidth=3.0,
        label="WeatherBridge",
    )
    axis.set_yscale("log")
    axis.set_xlabel("spatial frequency", fontsize=14)
    axis.set_ylabel("magnitude", fontsize=14)
    axis.tick_params(axis="both", labelsize=11, colors="#656a71")
    axis.grid(color="#e4e6e8", linewidth=0.8, alpha=0.85)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#8c9197")
    axis.legend(frameon=False, fontsize=12, loc="upper right")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor="white")
    plt.close(figure)


def color_samples(cmap_name: str) -> list[str]:
    cmap = matplotlib.colormaps[cmap_name]
    return [
        matplotlib.colors.to_hex(cmap(value))
        for value in np.linspace(0.0, 1.0, 11)
    ]


def main() -> None:
    with np.load(DATA, allow_pickle=False) as data:
        event_id = data["event_id"].item()
        event_label = data["event_label"].item()
        init_time = data["init_time"].item()
        fields = data["fields"].tolist()
        latitude = np.asarray(data["latitude"], dtype=np.float64)
        longitude = np.asarray(data["longitude"], dtype=np.float64)
        hours = np.asarray(data["hours"], dtype=np.int64)
        era5 = np.asarray(data["era5"], dtype=np.float32)
        query_hours = np.asarray(data["query_hours"], dtype=np.int64)
        interior_predictions = np.asarray(
            data["interior_predictions"],
            dtype=np.float32,
        )
        interior_rmse = np.asarray(data["interior_rmse"], dtype=np.float64)
        prediction = np.asarray(data["midpoint_prediction"], dtype=np.float32)
        midpoint_rmse = np.asarray(data["midpoint_rmse"], dtype=np.float64)
        provenance = json.loads(data["provenance_json"].item())

    if event_id != "pacific_northwest_heatwave":
        raise ValueError(f"unexpected event: {event_id}")
    if fields != ["T850", "t2m"] or hours.tolist() != list(range(7)):
        raise ValueError("unexpected fields or hour sequence")
    if era5.shape != (7, 2, latitude.size, longitude.size):
        raise ValueError("invalid ERA5 trajectory shape")
    if query_hours.tolist() != [1, 2, 3, 4, 5]:
        raise ValueError("unexpected graphical-overview query hours")
    if interior_predictions.shape != (5, 2, latitude.size, longitude.size):
        raise ValueError("invalid WeatherBridge trajectory shape")
    if interior_rmse.shape != (5, 2):
        raise ValueError("invalid WeatherBridge trajectory RMSE shape")
    if prediction.shape != (2, latitude.size, longitude.size):
        raise ValueError("invalid WeatherBridge prediction shape")
    if (
        not np.all(np.isfinite(era5))
        or not np.all(np.isfinite(interior_predictions))
        or not np.all(np.isfinite(prediction))
    ):
        raise ValueError("temperature assets contain non-finite values")

    ASSETS.mkdir(parents=True, exist_ok=True)
    for stale in ASSETS.glob("*.png"):
        stale.unlink()

    target = era5[3, 0]
    predicted = prediction[0]
    display_fields = np.concatenate(
        [era5[:, 0].reshape(-1), interior_predictions[:, 0].reshape(-1)]
    )
    temperature_min = float(np.floor(np.quantile(display_fields, 0.005)))
    temperature_max = float(np.ceil(np.quantile(display_fields, 0.995)))
    states = {
        "temperature_anchor0.png": era5[0, 0],
        "temperature_prediction.png": predicted,
        "temperature_target.png": target,
        "temperature_anchor6.png": era5[6, 0],
    }
    for name, field in states.items():
        render_field(
            field,
            latitude=latitude,
            longitude=longitude,
            output=ASSETS / name,
            cmap="coolwarm",
            vmin=temperature_min,
            vmax=temperature_max,
        )

    for index, query_hour in enumerate(query_hours):
        render_field(
            interior_predictions[index, 0],
            latitude=latitude,
            longitude=longitude,
            output=ASSETS / f"temperature_prediction_h{query_hour}.png",
            cmap="coolwarm",
            vmin=temperature_min,
            vmax=temperature_max,
        )

    anchor_difference = era5[6, 0] - era5[0, 0]
    difference_limit = float(
        np.quantile(np.abs(anchor_difference), 0.995)
    )
    render_field(
        anchor_difference,
        latitude=latitude,
        longitude=longitude,
        output=ASSETS / "temperature_anchor_difference.png",
        cmap="RdBu_r",
        vmin=-difference_limit,
        vmax=difference_limit,
    )

    signed_error = predicted - target
    error_limit = float(np.quantile(np.abs(signed_error), 0.995))
    render_field(
        signed_error,
        latitude=latitude,
        longitude=longitude,
        output=ASSETS / "temperature_error.png",
        cmap="RdBu_r",
        vmin=-error_limit,
        vmax=error_limit,
    )
    highpass_error = np.abs(
        local_highpass(predicted) - local_highpass(target)
    )
    highpass_limit = float(np.quantile(highpass_error, 0.995))
    render_field(
        highpass_error,
        latitude=latitude,
        longitude=longitude,
        output=ASSETS / "temperature_highpass_error.png",
        cmap="magma",
        vmin=0.0,
        vmax=highpass_limit,
    )
    render_fft_summary(target, predicted, ASSETS / "temperature_fft.png")

    browser_data = {
        "schemaVersion": 6,
        "event": {
            "id": event_id,
            "label": event_label,
            "initialTime": init_time,
            "field": "Temperature at 850 hPa",
            "queryHour": 3,
            "queryHours": query_hours.tolist(),
            "midpointRmseK": float(midpoint_rmse[0]),
            "rmseKByQueryHour": interior_rmse[:, 0].tolist(),
        },
        "lossWeights": {"pixel": 1.0, "highpass": 0.05, "fft": 0.02},
        "scales": {
            "temperature": {
                "vmin": temperature_min,
                "vmax": temperature_max,
                "colors": color_samples("coolwarm"),
            },
            "error": {
                "vmin": -error_limit,
                "vmax": error_limit,
                "colors": color_samples("RdBu_r"),
            },
            "highpass": {
                "vmin": 0.0,
                "vmax": highpass_limit,
                "colors": color_samples("magma"),
            },
            "anchorDifference": {
                "vmin": -difference_limit,
                "vmax": difference_limit,
                "colors": color_samples("RdBu_r"),
            },
        },
        "source": {
            "dataSha256": sha256_file(DATA),
            "checkpointSha256": provenance["checkpoint"]["sha256"],
        },
    }
    data_js = HERE / "overview_data.js"
    data_js.write_text(
        "window.OVERVIEW_DATA = "
        + json.dumps(browser_data, separators=(",", ":"))
        + ";\n"
    )

    generated = [*sorted(ASSETS.glob("*.png")), data_js]
    records = {}
    for path in generated:
        record: dict[str, object] = {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".png":
            with Image.open(path) as image:
                record["pixel_size"] = list(image.size)
        records[str(path.relative_to(HERE))] = record
    manifest = {
        "schema_version": 6,
        "source_data": {
            "path": str(DATA.relative_to(ROOT)),
            "sha256": sha256_file(DATA),
        },
        "assets": records,
    }
    (HERE / "assets_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {len(generated)} Pacific Northwest temperature assets")


if __name__ == "__main__":
    main()
