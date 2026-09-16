#!/usr/bin/env python3
"""Watch for global anchor pairs, run WeatherBridge, and publish bundles."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from production.weatherbridge_app.render import (
    PALETTES,
    colorize,
    palette_for,
    save_webp,
)
from production.weatherbridge_app.store import BundleStore
from weather_time_interp.normalization import (
    PAPER_CHANNELS_24,
    file_provenance,
    load_channel_stats,
)

UNITS = {
    "T": "K",
    "U": "m s-1",
    "V": "m s-1",
    "Q": "g kg-1",
    "Z": "m2 s-2",
    "t2m": "K",
    "u10": "m s-1",
    "v10": "m s-1",
    "mslp": "hPa",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        raise ValueError(f"anchor bundle is missing {key}")
    return str(np.asarray(data[key]).item())


def _physical_field(field: str, values: np.ndarray) -> np.ndarray:
    if field.startswith("Q"):
        return values * 1000.0
    if field == "mslp":
        return values / 100.0
    return values


def _run_id(end_time: dt.datetime, anchor_path: Path) -> str:
    digest = hashlib.sha256(anchor_path.read_bytes()).hexdigest()[:10]
    return f"{end_time.strftime('%Y%m%dT%H%M%SZ')}-{digest}"


def process_anchor(args: argparse.Namespace, anchor_path: Path) -> Path:
    import torch

    from examples._bare_loader import load_bare

    model_provenance = file_provenance(args.model_path)
    if (
        args.expected_model_sha256
        and model_provenance["sha256"] != args.expected_model_sha256
    ):
        raise ValueError(
            "model SHA-256 mismatch: "
            f"expected {args.expected_model_sha256}, got {model_provenance['sha256']}"
        )
    stats = load_channel_stats(args.stats_path, args.surface_stats_path)
    static = torch.load(
        args.static_path, map_location="cpu", weights_only=False
    ).float()
    if static.ndim == 3:
        static = static.unsqueeze(0)
    with np.load(anchor_path, allow_pickle=False) as data:
        x0_np = np.asarray(data["x0"], dtype=np.float32)
        xT_np = np.asarray(data["xT"], dtype=np.float32)
        start = dt.datetime.fromisoformat(
            _scalar(data, "valid_time_start").replace("Z", "+00:00")
        )
        end = dt.datetime.fromisoformat(
            _scalar(data, "valid_time_end").replace("Z", "+00:00")
        )
        source = _scalar(data, "source") if "source" in data.files else "upstream"
        normalized = (
            bool(np.asarray(data["normalized"]).item())
            if "normalized" in data.files
            else False
        )
    if x0_np.shape != (24, 360, 720) or xT_np.shape != x0_np.shape:
        raise ValueError(
            f"expected x0/xT shape (24, 360, 720), got {x0_np.shape}/{xT_np.shape}"
        )
    delta_hours = (end - start).total_seconds() / 3600.0
    if delta_hours not in {6.0, 12.0}:
        raise ValueError(f"unsupported anchor spacing: {delta_hours} h")
    if not np.isfinite(x0_np).all() or not np.isfinite(xT_np).all():
        raise ValueError("anchor fields contain NaN or infinity")
    if not normalized:
        x0_np = stats.normalize(x0_np)
        xT_np = stats.normalize(xT_np)

    device = torch.device(
        args.device
        if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_bare(args.model_path, device)
    model.eval()
    x0 = torch.from_numpy(x0_np).unsqueeze(0).to(device)
    xT = torch.from_numpy(xT_np).unsqueeze(0).to(device)
    static = static[:, : int(model.n_static_features)].to(device)
    taus = list(range(1, int(delta_hours)))
    predictions = []
    with torch.inference_mode():
        for tau in taus:
            output = model(
                x0,
                xT,
                torch.tensor([tau / delta_hours], device=device),
                torch.tensor([delta_hours], device=device),
                static=static,
            )
            if isinstance(output, tuple):
                output = output[0]
            predictions.append(output[0].float().cpu().numpy())
    normalized_prediction = np.stack(predictions)
    physical = stats.denormalize(normalized_prediction)
    if not np.isfinite(physical).all():
        raise RuntimeError("model output contains NaN or infinity")

    store = BundleStore(args.store)
    store.initialize()
    run_id = _run_id(end.astimezone(dt.timezone.utc), anchor_path)
    staged = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=store.runs))
    try:
        land_mask = static[0, 0].detach().cpu().numpy() if static.shape[1] else None
        fields: dict[str, object] = {}
        raw = {}
        for channel, field in enumerate(PAPER_CHANNELS_24):
            values = _physical_field(field, physical[:, channel])
            raw[field] = values.astype(np.float16)
            vmin, vmax = np.percentile(values, [1.0, 99.0]).astype(float)
            palette = palette_for(field)
            images = {}
            per_tau = {}
            for index, tau in enumerate(taus):
                relative = Path("images") / field / f"tau_{tau}.webp"
                save_webp(
                    colorize(values[index], vmin, vmax, palette, land_mask),
                    staged / relative,
                )
                images[str(tau)] = relative.as_posix()
                per_tau[str(tau)] = {
                    "valid_time": (start + dt.timedelta(hours=tau))
                    .astimezone(dt.timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "min": float(values[index].min()),
                    "max": float(values[index].max()),
                    "mean": float(values[index].mean()),
                }
            family = field if field in UNITS else field[0]
            fields[field] = {
                "unit": UNITS[family],
                "palette": palette,
                "palette_stops": list(PALETTES[palette]),
                "vmin": vmin,
                "vmax": vmax,
                "images": images,
                "hours": per_tau,
            }
        np.savez_compressed(staged / "forecast_float16.npz", **raw)
        manifest = {
            "schema_version": 1,
            "run_id": run_id,
            "model": "WeatherBridge",
            "model_provenance": model_provenance,
            "source": source,
            "generated_at": dt.datetime.now(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "anchor_start": start.astimezone(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "anchor_end": end.astimezone(dt.timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "anchor_hours": delta_hours,
            "interpolation_hours": taus,
            "grid": {"height": 360, "width": 720, "resolution_degrees": 0.5},
            "fields": fields,
        }
        BundleStore._atomic_json(staged / "manifest.json", manifest)
        return store.publish(staged, run_id)
    except Exception:
        shutil.rmtree(staged, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inbox", default=os.getenv("WB_INBOX", "/var/lib/weatherbridge/inbox")
    )
    parser.add_argument(
        "--archive", default=os.getenv("WB_ARCHIVE", "/var/lib/weatherbridge/archive")
    )
    parser.add_argument(
        "--store",
        default=os.getenv("WB_FORECAST_STORE", "/var/lib/weatherbridge/store"),
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv(
            "WB_MODEL_PATH",
            "/weights/weatherbridge_14m_6h_bare.pt",
        ),
    )
    parser.add_argument(
        "--expected-model-sha256",
        default=os.getenv("WB_MODEL_SHA256", ""),
    )
    parser.add_argument(
        "--static-path",
        default=os.getenv("WB_STATIC_PATH", "/data/static_features_0p5.pt"),
    )
    parser.add_argument(
        "--stats-path", default=os.getenv("WB_STATS_PATH", "/data/json_stats_0p5.nc")
    )
    parser.add_argument(
        "--surface-stats-path",
        default=os.getenv("WB_SURFACE_STATS_PATH", "/data/surface_stats_0p5.json"),
    )
    parser.add_argument("--device", default=os.getenv("WB_DEVICE", "auto"))
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    inbox, archive = Path(args.inbox), Path(args.archive)
    inbox.mkdir(parents=True, exist_ok=True)
    archive.mkdir(parents=True, exist_ok=True)
    while True:
        candidates = sorted(inbox.glob("*.npz"), key=lambda path: path.stat().st_mtime)
        if candidates:
            anchor = candidates[0]
            published = process_anchor(args, anchor)
            os.replace(anchor, archive / anchor.name)
            print(f"published {published}", flush=True)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
