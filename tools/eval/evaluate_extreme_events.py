#!/usr/bin/env python3
"""Evaluate interpolation models on independently selected extreme events."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.eval.capmatched_loader import load_capmatched_checkpoint
from tools.eval.export_weatherbridge_cases import (
    FULL_LAT,
    FULL_LON,
    _array_sha256,
    _memmap_identity,
    _normalise_24,
    _physical_24,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.normalization import PAPER_CHANNELS_24, file_provenance


TAUS = tuple(range(1, 6))


def parse_model(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME:CHECKPOINT")
    return name, Path(path)


def parse_taus(value: str) -> tuple[int, ...]:
    taus = tuple(int(item) for item in value.split(",") if item)
    if len(set(taus)) != len(taus) or any(tau not in TAUS for tau in taus):
        raise argparse.ArgumentTypeError("field taus must be unique values in 1..5")
    return taus


def field_artifact_path(
    output_dir: Path, event_id: str, model_name: str, tau: int
) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    if not slug:
        raise ValueError(f"cannot form artifact name from model {model_name!r}")
    return output_dir / f"{event_id}__{slug}__tau{tau}.npz"


def write_field_artifact(
    output_dir: Path,
    *,
    event: dict[str, Any],
    model_name: str,
    tau: int,
    channels: list[str],
    channel_names: list[str],
    lat: np.ndarray,
    lon: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    provenance: dict[str, Any],
) -> Path:
    indices = [channel_names.index(channel) for channel in channels]
    selected_prediction = np.asarray(prediction[indices], dtype=np.float32)
    selected_target = np.asarray(target[indices], dtype=np.float32)
    if selected_prediction.shape != selected_target.shape:
        raise ValueError("prediction and target field shapes differ")
    if not np.all(np.isfinite(selected_prediction)) or not np.all(
        np.isfinite(selected_target)
    ):
        raise ValueError("field artifact contains non-finite values")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = field_artifact_path(output_dir, event["id"], model_name, tau)
    np.savez_compressed(
        output,
        event_id=np.asarray(event["id"]),
        event_label=np.asarray(event["label"]),
        init_time=np.asarray(event["init_time"]),
        bbox=np.asarray(event["bbox"], dtype=np.float32),
        model_name=np.asarray(model_name),
        tau=np.asarray(tau, dtype=np.int8),
        channels=np.asarray(channels),
        lat=np.asarray(lat, dtype=np.float32),
        lon=np.asarray(lon, dtype=np.float32),
        prediction=selected_prediction,
        target=selected_target,
        provenance_json=np.asarray(json.dumps(provenance, sort_keys=True)),
    )
    return output


def spatial_indices(
    bbox: list[float],
    *,
    lat: np.ndarray = FULL_LAT,
    lon: np.ndarray = FULL_LON,
) -> tuple[np.ndarray, np.ndarray]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain lat_min, lat_max, lon_min, lon_max")
    lat_min, lat_max, lon_min, lon_max = map(float, bbox)
    if not -90.0 <= lat_min < lat_max <= 90.0:
        raise ValueError(f"invalid latitude bounds: {bbox}")
    if not 0.0 <= lon_min < 360.0 or not 0.0 < lon_max <= 360.0:
        raise ValueError(f"invalid longitude bounds: {bbox}")
    lat_idx = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
    if lon_min <= lon_max:
        lon_idx = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
    else:
        lon_idx = np.flatnonzero((lon >= lon_min) | (lon <= lon_max))
    if not lat_idx.size or not lon_idx.size:
        raise ValueError(f"bbox selects no grid cells: {bbox}")
    return lat_idx, lon_idx


def latitude_weighted_mean(values: np.ndarray, lat: np.ndarray) -> float:
    """Average arrays ending in (latitude, longitude) on the sphere."""
    if values.shape[-2] != lat.size:
        raise ValueError("latitude length does not match values")
    weights = np.cos(np.deg2rad(lat, dtype=np.float64))
    reshape = (1,) * (values.ndim - 2) + (lat.size, 1)
    broadcast = np.broadcast_to(weights.reshape(reshape), values.shape)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("metric array has no finite values")
    return float(np.sum(values[finite] * broadcast[finite]) / np.sum(broadcast[finite]))


def validate_manifest(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schema_version") != 1:
        raise ValueError("expected extreme-event manifest schema_version=1")
    events = payload.get("events")
    if not isinstance(events, list) or len(events) < 2:
        raise ValueError("manifest must contain at least two events")
    ids: set[str] = set()
    for event in events:
        event_id = event.get("id")
        if not isinstance(event_id, str) or not event_id or event_id in ids:
            raise ValueError(f"invalid or duplicate event id: {event_id!r}")
        ids.add(event_id)
        init_time = datetime.fromisoformat(event["init_time"])
        if init_time.year != 2021 or init_time.minute or init_time.second:
            raise ValueError(f"{event_id}: expected an hourly 2021 init_time")
        if init_time.hour % 6:
            raise ValueError(f"{event_id}: init_time must use a six-hour anchor")
        spatial_indices(event["bbox"])
        if not event.get("channels") or not event.get("source", {}).get("url"):
            raise ValueError(f"{event_id}: channels and source URL are required")
    return events


def crop(array: np.ndarray, lat_idx: np.ndarray, lon_idx: np.ndarray) -> np.ndarray:
    return array[..., lat_idx[:, None], lon_idx]


def field_metrics(
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    *,
    channels: list[str],
    channel_names: list[str],
    lat: np.ndarray,
) -> dict[str, Any]:
    indices = [channel_names.index(channel) for channel in channels]
    normalized_mse = (pred_norm[indices] - target_norm[indices]) ** 2
    result: dict[str, Any] = {
        "normalized_rmse": latitude_weighted_mean(normalized_mse, lat) ** 0.5,
        "per_channel": {},
    }
    for channel, index in zip(channels, indices, strict=True):
        error = pred_phys[index] - target_phys[index]
        scale = 0.01 if channel == "mslp" else 1.0
        result["per_channel"][channel] = {
            "rmse": latitude_weighted_mean(error**2, lat) ** 0.5 * scale,
            "bias": latitude_weighted_mean(error, lat) * scale,
            "units": "hPa" if channel == "mslp" else (
                "K" if channel.startswith("T") or channel == "t2m" else "m s-1"
            ),
        }
    if {"u10", "v10"}.issubset(channels):
        u_index = channel_names.index("u10")
        v_index = channel_names.index("v10")
        pred_speed = np.hypot(pred_phys[u_index], pred_phys[v_index])
        target_speed = np.hypot(target_phys[u_index], target_phys[v_index])
        result["wind_speed_rmse_m_s"] = latitude_weighted_mean(
            (pred_speed - target_speed) ** 2, lat
        ) ** 0.5
        result["vector_wind_rmse_m_s"] = latitude_weighted_mean(
            (pred_phys[u_index] - target_phys[u_index]) ** 2
            + (pred_phys[v_index] - target_phys[v_index]) ** 2,
            lat,
        ) ** 0.5
    return result


def extreme_metrics(
    definitions: list[dict[str, Any]],
    pred_phys: np.ndarray,
    target_phys: np.ndarray,
    channel_names: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for definition in definitions:
        if definition.get("derived") == "wind_speed":
            u_index = channel_names.index("u10")
            v_index = channel_names.index("v10")
            pred = np.hypot(pred_phys[u_index], pred_phys[v_index])
            target = np.hypot(target_phys[u_index], target_phys[v_index])
        else:
            index = channel_names.index(definition["channel"])
            pred = pred_phys[index]
            target = target_phys[index]
        reducer = np.nanmin if definition["kind"] == "min" else np.nanmax
        scale = float(definition.get("scale", 1.0))
        pred_value = float(reducer(pred)) * scale
        target_value = float(reducer(target)) * scale
        result[definition["name"]] = {
            "prediction": pred_value,
            "target": target_value,
            "error": pred_value - target_value,
            "units": definition["units"],
        }
    return result


def summarize(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    models = list(results)
    event_ids = list(next(iter(results.values())))
    summary: dict[str, Any] = {"models": {}}
    for model in models:
        event_rmse = []
        wind_rmse = []
        for event in results[model].values():
            for metrics in event["per_tau"].values():
                event_rmse.append(metrics["normalized_rmse"])
                if "wind_speed_rmse_m_s" in metrics:
                    wind_rmse.append(metrics["wind_speed_rmse_m_s"])
        summary["models"][model] = {
            "mean_normalized_rmse": float(np.mean(event_rmse)),
            "mean_wind_speed_rmse_m_s": (
                float(np.mean(wind_rmse)) if wind_rmse else None
            ),
        }
    summary["event_winners_normalized_rmse"] = {}
    for event_id in event_ids:
        means = {
            model: float(
                np.mean(
                    [
                        item["normalized_rmse"]
                        for item in results[model][event_id]["per_tau"].values()
                    ]
                )
            )
            for model in models
        }
        summary["event_winners_normalized_rmse"][event_id] = {
            "winner": min(means, key=means.get),
            "mean_by_model": means,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=parse_model, action="append", required=True)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--field-output-dir", type=Path)
    parser.add_argument("--field-taus", type=parse_taus, default=())
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    events = validate_manifest(manifest)
    model_names = [name for name, _ in args.model]
    if len(set(model_names)) != len(model_names) or "Linear" in model_names:
        raise ValueError("model names must be unique and cannot use reserved name Linear")
    if args.field_taus and args.field_output_dir is None:
        raise ValueError("--field-taus requires --field-output-dir")

    dataset = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[2021],
        max_tau_hours=6,
        samples_per_date=1,
        train=False,
        eval_hours=list(TAUS),
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    channel_names = list(dataset.channel_names) + list(
        dataset.surface_variables[: 24 - dataset.in_channels]
    )
    if tuple(channel_names) != PAPER_CHANNELS_24:
        raise ValueError(
            "evaluation channel order does not match the paper's 24-field contract"
        )
    raw = dataset.memmaps[2021]
    static = torch.load(
        args.static_path, map_location=args.device, weights_only=False
    )[:3].unsqueeze(0)
    device = torch.device(args.device)
    year_start = datetime(2021, 1, 1)
    prepared: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {"Linear": {}}

    for event in events:
        init_time = datetime.fromisoformat(event["init_time"])
        t0 = int((init_time - year_start).total_seconds() // 3600)
        if t0 < 0 or t0 + 6 >= raw.shape[0]:
            raise ValueError(f"{event['id']}: window falls outside the memmap")
        lat_idx, lon_idx = spatial_indices(event["bbox"])
        raw_x0 = np.array(raw[t0, :24], copy=True)
        raw_xT = np.array(raw[t0 + 6, :24], copy=True)
        x0 = _normalise_24(dataset, raw_x0)
        xT = _normalise_24(dataset, raw_xT)
        prepared[event["id"]] = {
            "event": event,
            "t0": t0,
            "lat_idx": lat_idx,
            "lon_idx": lon_idx,
            "x0": x0,
            "xT": xT,
            "input_sha256": {
                "x0": _array_sha256(raw_x0),
                "xT": _array_sha256(raw_xT),
            },
        }
        results["Linear"][event["id"]] = {"per_tau": {}}
        for tau in TAUS:
            target_phys_full = np.array(raw[t0 + tau, :24], copy=True)
            target_norm_full = _normalise_24(dataset, target_phys_full).numpy()
            pred_norm_full = ((1.0 - tau / 6.0) * x0 + (tau / 6.0) * xT).numpy()
            pred_phys_full = _physical_24(
                dataset, torch.from_numpy(pred_norm_full)
            ).numpy()
            pred_norm = crop(pred_norm_full, lat_idx, lon_idx)
            target_norm = crop(target_norm_full, lat_idx, lon_idx)
            pred_phys = crop(pred_phys_full, lat_idx, lon_idx)
            target_phys = crop(target_phys_full, lat_idx, lon_idx)
            metrics = field_metrics(
                pred_norm,
                target_norm,
                pred_phys,
                target_phys,
                channels=event["channels"],
                channel_names=channel_names,
                lat=FULL_LAT[lat_idx],
            )
            metrics["extremes"] = extreme_metrics(
                event.get("extremes", []), pred_phys, target_phys, channel_names
            )
            results["Linear"][event["id"]]["per_tau"][str(tau)] = metrics
            if tau in args.field_taus:
                write_field_artifact(
                    args.field_output_dir,
                    event=event,
                    model_name="Linear",
                    tau=tau,
                    channels=event["channels"],
                    channel_names=channel_names,
                    lat=FULL_LAT[lat_idx],
                    lon=FULL_LON[lon_idx],
                    prediction=pred_phys,
                    target=target_phys,
                    provenance={
                        "schema_version": 1,
                        "method": "linear_interpolation",
                        "input_sha256": prepared[event["id"]]["input_sha256"],
                    },
                )

    model_provenance: dict[str, Any] = {}
    for model_name, checkpoint in args.model:
        model, model_type = load_capmatched_checkpoint(
            str(checkpoint), device, static_path=args.static_path
        )
        model.eval()
        model_provenance[model_name] = {
            "model_type": model_type,
            "arch": str(getattr(model, "arch", "unknown")),
            "checkpoint": file_provenance(checkpoint),
        }
        results[model_name] = {}
        for event_id, item in prepared.items():
            event = item["event"]
            lat_idx = item["lat_idx"]
            lon_idx = item["lon_idx"]
            x0 = item["x0"].unsqueeze(0).to(device)
            xT = item["xT"].unsqueeze(0).to(device)
            results[model_name][event_id] = {"per_tau": {}}
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                for tau in TAUS:
                    tau_norm = torch.tensor([tau / 6.0], device=device)
                    delta_t = torch.tensor([6.0], device=device)
                    output = model(x0, xT, tau_norm, delta_t, static=static)
                    pred_norm_tensor = output[0] if isinstance(output, tuple) else output
                    pred_norm_full = pred_norm_tensor[0].float().cpu().numpy()
                    target_phys_full = np.array(raw[item["t0"] + tau, :24], copy=True)
                    target_norm_full = _normalise_24(dataset, target_phys_full).numpy()
                    pred_phys_full = _physical_24(
                        dataset, torch.from_numpy(pred_norm_full)
                    ).numpy()
                    pred_norm = crop(pred_norm_full, lat_idx, lon_idx)
                    target_norm = crop(target_norm_full, lat_idx, lon_idx)
                    pred_phys = crop(pred_phys_full, lat_idx, lon_idx)
                    target_phys = crop(target_phys_full, lat_idx, lon_idx)
                    metrics = field_metrics(
                        pred_norm,
                        target_norm,
                        pred_phys,
                        target_phys,
                        channels=event["channels"],
                        channel_names=channel_names,
                        lat=FULL_LAT[lat_idx],
                    )
                    metrics["extremes"] = extreme_metrics(
                        event.get("extremes", []),
                        pred_phys,
                        target_phys,
                        channel_names,
                    )
                    results[model_name][event_id]["per_tau"][str(tau)] = metrics
                    if tau in args.field_taus:
                        write_field_artifact(
                            args.field_output_dir,
                            event=event,
                            model_name=model_name,
                            tau=tau,
                            channels=event["channels"],
                            channel_names=channel_names,
                            lat=FULL_LAT[lat_idx],
                            lon=FULL_LON[lon_idx],
                            prediction=pred_phys,
                            target=target_phys,
                            provenance={
                                "schema_version": 1,
                                "model": model_provenance[model_name],
                                "input_sha256": item["input_sha256"],
                            },
                        )
            print(f"evaluated {model_name}: {event_id}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "manifest": file_provenance(args.manifest),
        "selection_policy": manifest["selection_policy"],
        "events": events,
        "evaluation": {
            "year": 2021,
            "taus": list(TAUS),
            "latitude_weighting": "cosine_cell_centre",
            "normalization": "training_channel_mean_std",
            "memmap": _memmap_identity(raw),
            "static": file_provenance(args.static_path),
            "stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(args.surface_stats_path),
            "evaluator": file_provenance(__file__),
            "index_sha256": hashlib.sha256(
                json.dumps(
                    [
                        [event["id"], event["init_time"], event["bbox"]]
                        for event in events
                    ],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "input_sha256": {
                event_id: item["input_sha256"] for event_id, item in prepared.items()
            },
        },
        "model_provenance": model_provenance,
        "results": results,
        "summary": summarize(results),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
