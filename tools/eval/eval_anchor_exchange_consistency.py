#!/usr/bin/env python3
"""Evaluate anchor-exchange equivariance and exact endpoint behavior."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.eval.batch_eval_12h_memmap import _load_model_safe
from tools.eval.region_season_12h_eval import (
    _economy_filter,
    region_season_evaluation_source_paths,
)
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.normalization import file_provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def anchor_exchange_evaluation_source_paths() -> dict[str, Path]:
    """Return every repository source that can affect this diagnostic."""
    repo_root = Path(__file__).resolve().parents[2]
    sources = dict(region_season_evaluation_source_paths())
    sources.update(
        {
            "tools/eval/eval_anchor_exchange_consistency.py": (
                Path(__file__).resolve()
            ),
            "tools/eval/batch_eval_12h_memmap.py": (
                repo_root / "tools" / "eval" / "batch_eval_12h_memmap.py"
            ),
        }
    )
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing anchor-exchange evaluation sources: "
            + ", ".join(sorted(missing))
        )
    return sources


def _index_sha256(*columns: np.ndarray) -> str:
    if (
        not columns
        or columns[0].size == 0
        or any(column.ndim != 1 for column in columns)
    ):
        raise ValueError("index columns must be non-empty 1-D arrays")
    if len({column.size for column in columns}) != 1:
        raise ValueError("index columns must have equal length")
    return hashlib.sha256(
        json.dumps(
            [
                tuple(int(value) for value in row)
                for row in zip(*columns)
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _parse_models(value: str) -> list[tuple[str, str]]:
    models = []
    for entry in value.split(","):
        name, separator, checkpoint = entry.partition(":")
        if not separator or not name or not checkpoint:
            raise ValueError(f"invalid NAME:CHECKPOINT entry: {entry}")
        models.append((name, checkpoint))
    if len({name for name, _ in models}) != len(models):
        raise ValueError("model names must be unique")
    return models


def _predict(
    model: torch.nn.Module,
    *,
    is_atmvfi: bool,
    x0: torch.Tensor,
    x1: torch.Tensor,
    tau: torch.Tensor,
    cond: torch.Tensor,
    static: torch.Tensor | None,
) -> torch.Tensor:
    if is_atmvfi:
        return model.net(x0, x1, tau)
    output = model(x0, x1, tau, cond, static=static)
    return output[0] if isinstance(output, tuple) else output


def _weighted_mse(
    left: torch.Tensor,
    right: torch.Tensor,
    latitude_weight: torch.Tensor,
) -> torch.Tensor:
    return (
        (left.float() - right.float()).square() * latitude_weight
    ).mean(dim=(-2, -1))


def _summary(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("expected non-empty [samples, channels] MSE array")
    per_channel = np.sqrt(np.maximum(values.mean(axis=0), 0.0))
    return {
        "pooled_rmse": float(
            np.sqrt(np.maximum(values.mean(), 0.0))
        ),
        "channel_macro_rmse": float(per_channel.mean()),
        "per_channel_rmse": per_channel.tolist(),
        "n_samples": int(values.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--test-year", type=int, required=True)
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument(
        "--static-path",
        default="data/static_features_0p5.pt",
    )
    parser.add_argument("--models", required=True)
    parser.add_argument("--max-tau-hours", type=int, default=6)
    parser.add_argument("--eval-hours")
    parser.add_argument("--samples-per-date", type=int, default=2)
    parser.add_argument("--eval-days-per-month", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--keep-n-channels", type=int, default=24)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    eval_hours = (
        tuple(range(1, args.max_tau_hours))
        if args.eval_hours is None
        else tuple(
            int(value) for value in args.eval_hours.split(",") if value
        )
    )
    if not eval_hours:
        raise ValueError("eval-hours must be non-empty")
    if any(hour <= 0 or hour >= args.max_tau_hours for hour in eval_hours):
        raise ValueError("eval-hours must be strictly inside the anchor window")
    if (
        eval_hours != tuple(range(1, args.max_tau_hours))
        or args.samples_per_date != 2
        or args.eval_days_per_month != 2
        or args.keep_n_channels != 24
    ):
        raise ValueError(
            "anchor-exchange protocol requires every interior hour, "
            "samples_per_date=2, eval_days_per_month=2, and "
            "keep_n_channels=24"
        )
    models = _parse_models(args.models)
    device = torch.device(args.device)
    evaluation_source_paths = anchor_exchange_evaluation_source_paths()
    evaluation_code_provenance = {
        name: _sha256(path)
        for name, path in evaluation_source_paths.items()
    }
    evaluation_dataset_provenance = memmap_dataset_provenance(
        args.memmap_dir,
        [args.test_year],
    )
    evaluation_input_provenance = {
        "static_features": file_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }

    base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.test_year],
        max_tau_hours=args.max_tau_hours,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=list(eval_hours),
        release_memmap_pages=True,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    _economy_filter(base, args.eval_days_per_month)
    wrapped = ERA5WeatherHermiteDataset(
        base,
        delta_t_hours=float(args.max_tau_hours),
    )
    loader = DataLoader(
        wrapped,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    channel_names = (
        list(base.channel_names) + list(base.surface_variables)
    )[: args.keep_n_channels]
    height = next(iter(base.memmaps.values())).shape[-2]
    latitude = torch.linspace(
        89.75 if height == 360 else 90.0,
        -89.75 if height == 360 else -90.0,
        height,
        device=device,
    )
    latitude_weight = torch.cos(torch.deg2rad(latitude))
    latitude_weight = (
        latitude_weight / latitude_weight.mean()
    ).view(1, 1, height, 1)
    grouped = getattr(wrapped, "_grouped_indices", None)

    shared_window_keys: list[tuple[int, int, int]] | None = None
    shared_endpoint_keys: list[tuple[int, int]] | None = None
    arrays: dict[str, np.ndarray] = {}
    report_models: dict[str, Any] = {}
    for name, checkpoint in models:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint_provenance = file_provenance(checkpoint_path)
        model, model_type = _load_model_safe(
            str(checkpoint_path),
            device,
            base.channel_groups,
            static_path=args.static_path,
        )
        is_atmvfi = model_type == "atm_vfi_pixel_attn"
        exchange_values: list[np.ndarray] = []
        forward_values: list[np.ndarray] = []
        endpoint0_values: list[np.ndarray] = []
        endpoint1_values: list[np.ndarray] = []
        window_keys: list[tuple[int, int, int]] = []
        endpoint_keys: list[tuple[int, int]] = []

        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                x0 = batch["x0"][:, : args.keep_n_channels].to(
                    device,
                    non_blocking=True,
                )
                x1 = batch["xT"][:, : args.keep_n_channels].to(
                    device,
                    non_blocking=True,
                )
                targets = batch["target"][
                    :, :, : args.keep_n_channels
                ].to(device, non_blocking=True)
                tau_hours = batch["tau_hour"].long()
                static = batch.get("static")
                if static is not None:
                    static = static.to(device, non_blocking=True)
                batch_size = x0.size(0)
                cond = torch.full(
                    (batch_size,),
                    float(args.max_tau_hours),
                    device=device,
                )
                zeros = torch.zeros(batch_size, device=device)
                ones = torch.ones(batch_size, device=device)
                endpoint0 = _predict(
                    model,
                    is_atmvfi=is_atmvfi,
                    x0=x0,
                    x1=x1,
                    tau=zeros,
                    cond=cond,
                    static=static,
                )
                endpoint1 = _predict(
                    model,
                    is_atmvfi=is_atmvfi,
                    x0=x0,
                    x1=x1,
                    tau=ones,
                    cond=cond,
                    static=static,
                )
                endpoint0_values.append(
                    _weighted_mse(endpoint0, x0, latitude_weight).cpu().numpy()
                )
                endpoint1_values.append(
                    _weighted_mse(endpoint1, x1, latitude_weight).cpu().numpy()
                )
                for sample in range(batch_size):
                    wrapped_index = batch_index * loader.batch_size + sample
                    base_index = (
                        grouped[wrapped_index][0]
                        if grouped is not None
                        else wrapped_index
                    )
                    year, t0, _, _ = base.index[base_index]
                    endpoint_keys.append((int(year), int(t0)))

                for hour_index in range(tau_hours.size(1)):
                    hour = tau_hours[:, hour_index, 0]
                    tau = hour.float().to(device) / float(
                        args.max_tau_hours
                    )
                    forward = _predict(
                        model,
                        is_atmvfi=is_atmvfi,
                        x0=x0,
                        x1=x1,
                        tau=tau,
                        cond=cond,
                        static=static,
                    )
                    reverse = _predict(
                        model,
                        is_atmvfi=is_atmvfi,
                        x0=x1,
                        x1=x0,
                        tau=1.0 - tau,
                        cond=cond,
                        static=static,
                    )
                    exchange_values.append(
                        _weighted_mse(
                            forward,
                            reverse,
                            latitude_weight,
                        ).cpu().numpy()
                    )
                    forward_values.append(
                        _weighted_mse(
                            forward,
                            targets[:, hour_index],
                            latitude_weight,
                        ).cpu().numpy()
                    )
                    for sample in range(batch_size):
                        wrapped_index = (
                            batch_index * loader.batch_size + sample
                        )
                        base_index = (
                            grouped[wrapped_index][0]
                            if grouped is not None
                            else wrapped_index
                        )
                        year, t0, _, _ = base.index[base_index]
                        window_keys.append(
                            (int(year), int(t0), int(hour[sample]))
                        )

        exchange = np.concatenate(exchange_values, axis=0)
        forward = np.concatenate(forward_values, axis=0)
        endpoint0 = np.concatenate(endpoint0_values, axis=0)
        endpoint1 = np.concatenate(endpoint1_values, axis=0)
        if shared_window_keys is None:
            shared_window_keys = window_keys
            shared_endpoint_keys = endpoint_keys
        elif (
            window_keys != shared_window_keys
            or endpoint_keys != shared_endpoint_keys
        ):
            raise RuntimeError("paired evaluation index changed across models")
        arrays[f"{name}_exchange_mse"] = exchange
        arrays[f"{name}_forward_mse"] = forward
        arrays[f"{name}_endpoint0_mse"] = endpoint0
        arrays[f"{name}_endpoint1_mse"] = endpoint1
        exchange_summary = _summary(exchange)
        forward_summary = _summary(forward)
        if file_provenance(checkpoint_path) != checkpoint_provenance:
            raise RuntimeError(
                f"{checkpoint_path}: checkpoint changed during evaluation"
            )
        report_models[name] = {
            "model_type": model_type,
            "checkpoint_provenance": checkpoint_provenance,
            "exchange": exchange_summary,
            "forward_error": forward_summary,
            "exchange_to_forward_rmse_ratio": (
                exchange_summary["pooled_rmse"]
                / max(forward_summary["pooled_rmse"], 1e-12)
            ),
            "endpoint0": _summary(endpoint0),
            "endpoint1": _summary(endpoint1),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if shared_window_keys is None or shared_endpoint_keys is None:
        raise RuntimeError("evaluation produced no windows")
    window_array = np.asarray(shared_window_keys, dtype=np.int64)
    endpoint_array = np.asarray(shared_endpoint_keys, dtype=np.int64)
    arrays.update(
        {
            "window_year": window_array[:, 0],
            "window_t0": window_array[:, 1],
            "window_tau": window_array[:, 2],
            "endpoint_year": endpoint_array[:, 0],
            "endpoint_t0": endpoint_array[:, 1],
            "channel_names": np.asarray(channel_names),
        }
    )
    window_index_sha256 = _index_sha256(
        arrays["window_year"],
        arrays["window_t0"],
        arrays["window_tau"],
    )
    endpoint_index_sha256 = _index_sha256(
        arrays["endpoint_year"],
        arrays["endpoint_t0"],
    )
    current_code_provenance = {
        name: _sha256(path)
        for name, path in evaluation_source_paths.items()
    }
    current_dataset_provenance = memmap_dataset_provenance(
        args.memmap_dir,
        [args.test_year],
    )
    current_input_provenance = {
        "static_features": file_provenance(args.static_path),
        "pressure_level_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
    }
    if current_code_provenance != evaluation_code_provenance:
        raise RuntimeError("evaluation source changed during evaluation")
    if current_dataset_provenance != evaluation_dataset_provenance:
        raise RuntimeError("evaluation dataset changed during evaluation")
    if current_input_provenance != evaluation_input_provenance:
        raise RuntimeError("evaluation inputs changed during evaluation")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    npz_path = args.out_json.with_suffix(".npz")
    npz_temporary = npz_path.with_suffix(npz_path.suffix + ".tmp")
    with npz_temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(npz_temporary, npz_path)
    report = {
        "schema_version": 2,
        "diagnostic_only": True,
        "evaluation_code_provenance": evaluation_code_provenance,
        "test_year": args.test_year,
        "delta_t_hours": args.max_tau_hours,
        "eval_hours": list(eval_hours),
        "channel_names": channel_names,
        "evaluation_protocol": {
            "samples_per_date": args.samples_per_date,
            "eval_days_per_month": args.eval_days_per_month,
            "window_index_sha256": window_index_sha256,
            "endpoint_index_sha256": endpoint_index_sha256,
            "n_windows": int(window_array.shape[0]),
            "n_endpoints": int(endpoint_array.shape[0]),
        },
        "evaluation_dataset_provenance": evaluation_dataset_provenance,
        "evaluation_input_provenance": evaluation_input_provenance,
        "paired_windows_file": npz_path.name,
        "paired_windows_size_bytes": npz_path.stat().st_size,
        "paired_windows_sha256": _sha256(npz_path),
        "models": report_models,
        "interpretation": (
            "Lower exchange error means better equivariance under "
            "(x0,xT,tau)->(xT,x0,1-tau). This is a diagnostic and is not "
            "part of the 2020 winner rule."
        ),
    }
    temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.out_json)
    print(json.dumps({name: row["exchange"] for name, row in report_models.items()}))


if __name__ == "__main__":
    main()
