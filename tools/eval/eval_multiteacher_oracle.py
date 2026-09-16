"""Fail-closed validation gate for band-aware multi-teacher distillation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from tools.eval.capmatched_loader import load_capmatched_checkpoint
from tools.train.train_capacity_matched_6h import (
    TauRescaleAnd24chWrapper,
    compose_multiteacher_target,
    distillation_channel_mask,
    weather_field_pole_parity,
)
from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from weather_time_interp.normalization import file_provenance


MODEL_NAMES = ("low_teacher", "high_teacher", "composite")


def _extract_prediction(output: object) -> torch.Tensor:
    prediction = output[0] if isinstance(output, tuple) else output
    if not isinstance(prediction, torch.Tensor):
        raise TypeError("model output must be a tensor or a tensor tuple")
    return prediction


def _aggregate_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def assess_oracle_metrics(
    rmse: dict[str, np.ndarray],
    tau_hours: list[int],
    high_channel_mask: np.ndarray,
    *,
    aggregate_tolerance: float,
    held_tolerance: float,
) -> dict[str, object]:
    """Apply the pre-training oracle gate to per-tau, per-field RMSE."""
    low = np.asarray(rmse["low_teacher"], dtype=np.float64)
    composite = np.asarray(rmse["composite"], dtype=np.float64)
    if low.shape != composite.shape or low.shape != (
        len(tau_hours),
        high_channel_mask.size,
    ):
        raise ValueError("oracle RMSE arrays do not match tau/channel protocol")
    active = high_channel_mask.astype(bool)
    excluded = ~active
    checks: dict[str, bool] = {
        "aggregate_noninferior": (
            _aggregate_rmse(composite)
            <= _aggregate_rmse(low) * (1.0 + aggregate_tolerance)
        ),
        "advected_improves": (
            _aggregate_rmse(composite[:, active])
            < _aggregate_rmse(low[:, active])
        ),
        "excluded_fields_unchanged": bool(
            np.allclose(
                composite[:, excluded],
                low[:, excluded],
                rtol=0.0,
                atol=1.0e-7,
            )
        ),
    }
    for held_hour in (2, 4):
        if held_hour not in tau_hours:
            continue
        index = tau_hours.index(held_hour)
        checks[f"held_tau_{held_hour}_noninferior"] = (
            _aggregate_rmse(composite[index])
            <= _aggregate_rmse(low[index]) * (1.0 + held_tolerance)
        )
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "aggregate_relative_tolerance": aggregate_tolerance,
            "held_tau_relative_tolerance": held_tolerance,
        },
        "summary": {
            name: {
                "aggregate_rmse": _aggregate_rmse(values),
                "advected_rmse": _aggregate_rmse(values[:, active]),
                "held_rmse": {
                    str(hour): _aggregate_rmse(values[tau_hours.index(hour)])
                    for hour in (2, 4)
                    if hour in tau_hours
                },
            }
            for name, values in rmse.items()
        },
    }


def _stratified_indices(
    index: list[tuple[int, int, int, float]],
    tau_hours: list[int],
    max_samples: int,
) -> list[int]:
    if max_samples < len(tau_hours):
        raise ValueError("max_samples must cover every requested tau")
    selected: list[int] = []
    base_quota, remainder = divmod(max_samples, len(tau_hours))
    for position, hour in enumerate(tau_hours):
        candidates = [
            item_index
            for item_index, item in enumerate(index)
            if int(item[2]) == hour
        ]
        quota = min(len(candidates), base_quota + (position < remainder))
        if quota <= 0:
            raise ValueError(f"no validation examples for tau={hour}")
        chosen = np.linspace(0, len(candidates) - 1, quota, dtype=int)
        selected.extend(candidates[int(item)] for item in chosen)
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low_checkpoint", required=True)
    parser.add_argument("--high_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memmap_dir", required=True)
    parser.add_argument("--static_path", required=True)
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--surface_stats_path", required=True)
    parser.add_argument("--year", type=int, default=2020)
    parser.add_argument("--window_hours", type=int, default=6)
    parser.add_argument("--tau_hours", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--max_samples", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--high_mask_profile",
        choices=("all", "advected"),
        default="advected",
    )
    parser.add_argument("--aggregate_tolerance", type=float, default=0.001)
    parser.add_argument("--held_tolerance", type=float, default=0.002)
    args = parser.parse_args()

    if args.max_samples <= 0 or args.batch_size <= 0 or args.workers < 0:
        parser.error("sample, batch, and worker counts must be valid")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in (args.aggregate_tolerance, args.held_tolerance)
    ):
        parser.error("oracle tolerances must be finite and non-negative")
    if not torch.cuda.is_available():
        parser.error("oracle evaluation requires CUDA")
    device = torch.device(f"cuda:{args.gpu}")

    base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.year],
        max_tau_hours=args.window_hours,
        samples_per_date=1,
        train=False,
        eval_hours=args.tau_hours,
        release_memmap_pages=True,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    dataset = TauRescaleAnd24chWrapper(base, delta_t=args.window_hours)
    indices = _stratified_indices(base.index, args.tau_hours, args.max_samples)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    low_model, low_type = load_capmatched_checkpoint(
        args.low_checkpoint,
        device,
        static_path=args.static_path,
    )
    high_model, high_type = load_capmatched_checkpoint(
        args.high_checkpoint,
        device,
        static_path=args.static_path,
    )
    mask = distillation_channel_mask(args.high_mask_profile).to(device)
    parity = weather_field_pole_parity().to(device)
    tau_to_index = {hour: index for index, hour in enumerate(args.tau_hours)}
    sums = {
        name: torch.zeros(
            len(args.tau_hours),
            24,
            dtype=torch.float64,
            device=device,
        )
        for name in MODEL_NAMES
    }
    counts = torch.zeros(
        len(args.tau_hours),
        dtype=torch.float64,
        device=device,
    )
    height = width = 0
    with torch.inference_mode():
        for batch in loader:
            x0 = batch["x0"].to(device, non_blocking=True)
            x1 = batch["x1"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            tau = batch["tau"].reshape(-1).to(device, non_blocking=True)
            tau_hour = batch["tau_hour"].reshape(-1).long().to(device)
            height, width = target.shape[-2:]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                low = _extract_prediction(low_model(x0, x1, tau))
                high = _extract_prediction(high_model(x0, x1, tau))
                composite = compose_multiteacher_target(
                    low,
                    high,
                    mask,
                    parity,
                )
            latitude = torch.cos(
                torch.deg2rad(
                    torch.linspace(
                        89.75,
                        -89.75,
                        height,
                        device=device,
                    )
                )
            )
            latitude = latitude / latitude.sum() * height
            latitude = latitude.view(1, 1, height, 1)
            for name, prediction in zip(
                MODEL_NAMES,
                (low, high, composite),
                strict=True,
            ):
                squared = (
                    (prediction.float() - target.float()).square()
                    * latitude
                ).sum(dim=(-2, -1))
                for hour, tau_index in tau_to_index.items():
                    active = tau_hour == hour
                    if active.any():
                        sums[name][tau_index].add_(
                            squared[active].sum(dim=0).double()
                        )
            for hour, tau_index in tau_to_index.items():
                counts[tau_index].add_((tau_hour == hour).sum())

    if height <= 0 or (counts <= 0).any():
        raise RuntimeError("oracle evaluation produced incomplete metrics")
    rmse = {
        name: (
            sums[name]
            / (counts.view(-1, 1) * height * width)
        ).sqrt().cpu().numpy()
        for name in MODEL_NAMES
    }
    gate = assess_oracle_metrics(
        rmse,
        args.tau_hours,
        mask.cpu().numpy(),
        aggregate_tolerance=args.aggregate_tolerance,
        held_tolerance=args.held_tolerance,
    )
    channel_names = [*base.channel_names, *base.surface_variables[:4]]
    payload = {
        "schema_version": 1,
        "status": "pass" if gate["pass"] else "fail",
        "gate": gate,
        "protocol": {
            "selection_role": "validation_only_pretraining_gate",
            "year": args.year,
            "window_hours": args.window_hours,
            "tau_hours": args.tau_hours,
            "sample_count": int(counts.sum().item()),
            "sample_count_per_tau": {
                str(hour): int(counts[index].item())
                for index, hour in enumerate(args.tau_hours)
            },
            "sample_indices": indices,
            "high_mask_profile": args.high_mask_profile,
            "high_mask_active_indices": mask.nonzero().reshape(-1).cpu().tolist(),
            "decomposition": "spherical_local_3x3_phase_preserving",
            "precision": "bf16_inference_fp32_accumulation",
        },
        "models": {
            "low_teacher": {
                "type": low_type,
                "checkpoint": file_provenance(args.low_checkpoint),
            },
            "high_teacher": {
                "type": high_type,
                "checkpoint": file_provenance(args.high_checkpoint),
            },
        },
        "channel_names": channel_names,
        "rmse_by_tau_channel": {
            name: values.tolist()
            for name, values in rmse.items()
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": payload["status"], "gate": gate}, indent=2))
    if not gate["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
