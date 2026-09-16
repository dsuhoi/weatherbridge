#!/usr/bin/env python3
"""Publish a clearly labelled synthetic bundle for UI and deployment smoke tests."""

from __future__ import annotations

import argparse
import datetime as dt
import tempfile
from pathlib import Path

import numpy as np

from production.weatherbridge_app.render import (
    PALETTES,
    colorize,
    palette_for,
    save_webp,
)
from production.weatherbridge_app.store import BundleStore
from weather_time_interp.normalization import PAPER_CHANNELS_24


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", default="production/runtime/store")
    args = parser.parse_args()
    store = BundleStore(args.store)
    store.initialize()
    now = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    run_id = f"demo-{now.strftime('%Y%m%dT%H%M%SZ')}"
    if (store.runs / run_id).exists():
        BundleStore._atomic_json(store.root / "latest.json", {"run_id": run_id})
        return 0
    staged = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=store.runs))
    lat = np.linspace(np.pi / 2, -np.pi / 2, 360)[:, None]
    lon = np.linspace(0, 2 * np.pi, 720, endpoint=False)[None, :]
    fields = {}
    taus = [1, 2, 3, 4, 5]
    for channel, field in enumerate(PAPER_CHANNELS_24):
        family = field if field in {"t2m", "u10", "v10", "mslp"} else field[0]
        base = np.sin((1 + channel % 4) * lon + channel * 0.17) * np.cos(lat) ** 2
        base += 0.45 * np.cos(3 * lat - channel * 0.11)
        if family in {"T", "t2m"}:
            offset, amplitude, unit = 273.0, 18.0, "K"
        elif family in {"U", "V", "u10", "v10"}:
            offset, amplitude, unit = 0.0, 22.0, "m s-1"
        elif family == "Q":
            offset, amplitude, unit = 4.0, 3.0, "g kg-1"
        elif family == "Z":
            offset, amplitude, unit = 35000.0, 3000.0, "m2 s-2"
        else:
            offset, amplitude, unit = 1010.0, 24.0, "hPa"
        series = np.stack(
            [offset + amplitude * np.roll(base, tau * 5, axis=1) for tau in taus]
        )
        vmin, vmax = np.percentile(series, [1, 99]).astype(float)
        palette = palette_for(field)
        images, hours = {}, {}
        for index, tau in enumerate(taus):
            relative = Path("images") / field / f"tau_{tau}.webp"
            save_webp(colorize(series[index], vmin, vmax, palette), staged / relative)
            images[str(tau)] = relative.as_posix()
            hours[str(tau)] = {
                "valid_time": (now + dt.timedelta(hours=tau))
                .isoformat()
                .replace("+00:00", "Z"),
                "min": float(series[index].min()),
                "max": float(series[index].max()),
                "mean": float(series[index].mean()),
            }
        fields[field] = {
            "unit": unit,
            "palette": palette,
            "palette_stops": list(PALETTES[palette]),
            "vmin": vmin,
            "vmax": vmax,
            "images": images,
            "hours": hours,
        }
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "model": "Synthetic deployment smoke test",
        "source": "synthetic-demo-not-for-science",
        "generated_at": dt.datetime.now(dt.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "anchor_start": now.isoformat().replace("+00:00", "Z"),
        "anchor_end": (now + dt.timedelta(hours=6)).isoformat().replace("+00:00", "Z"),
        "anchor_hours": 6.0,
        "interpolation_hours": taus,
        "grid": {"height": 360, "width": 720, "resolution_degrees": 0.5},
        "fields": fields,
    }
    BundleStore._atomic_json(staged / "manifest.json", manifest)
    store.publish(staged, run_id)
    print(f"published synthetic smoke bundle {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
