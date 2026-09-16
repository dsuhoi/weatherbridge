#!/usr/bin/env python3
"""Build a validation-only field-by-tau route for response distillation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_rmse(path: Path) -> tuple[list[int], list[str], dict[int, dict[str, float]]]:
    payload = json.loads(path.read_text())
    channels = list(payload["channel_names"])
    values = {
        int(tau): {
            key.removeprefix("rmse_norm_"): float(value)
            for key, value in item["model"].items()
            if key.startswith("rmse_norm_")
        }
        for tau, item in payload["per_tau"].items()
    }
    tau_hours = sorted(values)
    if any(list(values[tau]) != channels for tau in tau_hours):
        raise ValueError("RMSE channel order does not match channel_names")
    return tau_hours, channels, values


def build_route(
    control_path: Path,
    teacher_path: Path,
    *,
    minimum_improvement_pct: float,
    routing_granularity: str = "cell",
) -> dict[str, object]:
    control_taus, control_channels, control = _load_rmse(control_path)
    teacher_taus, teacher_channels, teacher = _load_rmse(teacher_path)
    if control_taus != teacher_taus or control_channels != teacher_channels:
        raise ValueError("control and teacher evaluation protocols differ")
    relative_delta_pct = [
        [
            100.0 * (teacher[tau][channel] / control[tau][channel] - 1.0)
            for channel in control_channels
        ]
        for tau in control_taus
    ]
    if routing_granularity == "cell":
        field_relative_delta_pct = None
        route = [
            [delta <= -minimum_improvement_pct for delta in row]
            for row in relative_delta_pct
        ]
    elif routing_granularity == "field":
        field_relative_delta_pct = []
        for channel in control_channels:
            control_rmse = math.sqrt(
                sum(control[tau][channel] ** 2 for tau in control_taus)
                / len(control_taus)
            )
            teacher_rmse = math.sqrt(
                sum(teacher[tau][channel] ** 2 for tau in teacher_taus)
                / len(teacher_taus)
            )
            field_relative_delta_pct.append(
                100.0 * (teacher_rmse / control_rmse - 1.0)
            )
        active_fields = [
            delta <= -minimum_improvement_pct
            for delta in field_relative_delta_pct
        ]
        route = [list(active_fields) for _ in control_taus]
    else:
        raise ValueError("routing granularity must be cell or field")
    return {
        "schema_version": 1,
        "selection_role": "era5_2020_validation_only",
        "selection_year": 2020,
        "minimum_teacher_improvement_pct": minimum_improvement_pct,
        "routing_granularity": routing_granularity,
        "tau_hours": control_taus,
        "channel_names": control_channels,
        "teacher_route": route,
        "relative_delta_pct": relative_delta_pct,
        "field_relative_delta_pct": field_relative_delta_pct,
        "active_routes": sum(value for row in route for value in row),
        "control": {
            "path": str(control_path.resolve()),
            "sha256": _sha256(control_path),
        },
        "teacher": {
            "path": str(teacher_path.resolve()),
            "sha256": _sha256(teacher_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--minimum-improvement-pct", type=float, default=0.5)
    parser.add_argument(
        "--routing-granularity",
        choices=("cell", "field"),
        default="cell",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 <= args.minimum_improvement_pct <= 100.0:
        parser.error("minimum improvement must lie in [0, 100]")
    payload = build_route(
        args.control,
        args.teacher,
        minimum_improvement_pct=args.minimum_improvement_pct,
        routing_granularity=args.routing_granularity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "active_routes": payload["active_routes"]}))


if __name__ == "__main__":
    main()
