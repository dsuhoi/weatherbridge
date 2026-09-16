#!/usr/bin/env python3
"""Select the 6 h architecture champion from frozen 2020 evidence only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tools.eval.select_universal_pyramid import summarize_metrics


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME:PATH")
    return name, Path(raw_path)


def _relative(left: float, right: float) -> float:
    if right <= 0.0:
        raise ValueError("reference metric must be positive")
    return left / right - 1.0


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    objectives = (
        "rmse_mean",
        "held_rmse_mean",
        "latency_ms",
        "params_m",
    )
    no_worse = all(float(left[key]) <= float(right[key]) for key in objectives)
    strictly_better = any(
        float(left[key]) < float(right[key]) for key in objectives
    )
    return no_worse and strictly_better


def select_champion(
    candidates: list[tuple[str, Path]],
    flow_path: Path,
    quality_path: Path,
    inference_path: Path,
    *,
    quality_gap_limit: float,
    latency_ratio_limit: float,
    maximum_parameters_m: float,
    cell_regression_limit: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one candidate is required")
    names = [name for name, _ in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate names must be unique")
    flow = summarize_metrics(flow_path)
    quality = summarize_metrics(quality_path)
    inference_payload = json.loads(inference_path.read_text())
    inference = inference_payload.get("models", {})
    if not isinstance(inference, dict) or "flow_spectral" not in inference:
        raise ValueError("inference JSON lacks flow_spectral")
    flow_latency = float(inference["flow_spectral"]["latency_ms"])
    if not math.isfinite(flow_latency) or flow_latency <= 0.0:
        raise ValueError("Flow latency must be finite and positive")

    rows: dict[str, dict[str, Any]] = {}
    index_hashes = {
        flow["evaluation_index_sha256"],
        quality["evaluation_index_sha256"],
    }
    for name, path in candidates:
        metrics = summarize_metrics(path)
        index_hashes.add(metrics["evaluation_index_sha256"])
        if name not in inference:
            raise ValueError(f"inference JSON lacks candidate {name}")
        cost = inference[name]
        latency_ms = float(cost["latency_ms"])
        params_m = float(cost["params_m"])
        if (
            not math.isfinite(latency_ms)
            or latency_ms <= 0.0
            or not math.isfinite(params_m)
            or params_m <= 0.0
        ):
            raise ValueError(f"invalid compute measurements for {name}")
        relative_cells = {
            cell: _relative(float(value), float(flow["cells"][cell]))
            for cell, value in metrics["cells"].items()
        }
        overall_vs_flow = _relative(
            float(metrics["rmse_mean"]),
            float(flow["rmse_mean"]),
        )
        held_vs_flow = _relative(
            float(metrics["held_rmse_mean"]),
            float(flow["held_rmse_mean"]),
        )
        overall_vs_quality = _relative(
            float(metrics["rmse_mean"]),
            float(quality["rmse_mean"]),
        )
        latency_ratio = latency_ms / flow_latency
        aggregate_quality_pass = overall_vs_flow <= 0.0 and held_vs_flow <= 0.0
        near_quality_frontier = overall_vs_quality <= quality_gap_limit
        compute_pass = (
            latency_ratio <= latency_ratio_limit
            and params_m <= maximum_parameters_m
        )
        eligible = aggregate_quality_pass and near_quality_frontier and compute_pass
        rows[name] = {
            **metrics,
            "latency_ms": latency_ms,
            "params_m": params_m,
            "latency_ratio_to_flow": latency_ratio,
            "overall_relative_to_flow": overall_vs_flow,
            "held_relative_to_flow": held_vs_flow,
            "overall_relative_to_quality": overall_vs_quality,
            "selection_rmse": (
                0.7 * float(metrics["held_rmse_mean"])
                + 0.3 * float(metrics["seen_rmse_mean"])
            ),
            "cell_wins_vs_flow": sum(
                value <= cell_regression_limit
                for value in relative_cells.values()
            ),
            "cell_count": len(relative_cells),
            "worst_cell_relative_to_flow": max(relative_cells.values()),
            "per_cell_relative_to_flow": relative_cells,
            "aggregate_quality_pass": aggregate_quality_pass,
            "near_quality_frontier": near_quality_frontier,
            "compute_pass": compute_pass,
            "eligible": eligible,
            "strict_uniform_pass": bool(
                eligible
                and max(relative_cells.values()) <= cell_regression_limit
            ),
        }
    if None in index_hashes or len(index_hashes) != 1:
        raise ValueError("candidate and references use different evaluation indices")

    eligible_names = [name for name in names if rows[name]["eligible"]]
    pareto = [
        name
        for name in eligible_names
        if not any(
            other != name and _dominates(rows[other], rows[name])
            for other in eligible_names
        )
    ]
    quality_winner = (
        min(
            eligible_names,
            key=lambda name: (
                float(rows[name]["selection_rmse"]),
                float(rows[name]["rmse_mean"]),
                float(rows[name]["latency_ms"]),
            ),
        )
        if eligible_names
        else None
    )
    efficiency_winner = (
        min(
            eligible_names,
            key=lambda name: (
                float(rows[name]["latency_ms"]),
                float(rows[name]["params_m"]),
                float(rows[name]["rmse_mean"]),
            ),
        )
        if eligible_names
        else None
    )
    uniform_names = [
        name for name in eligible_names if rows[name]["strict_uniform_pass"]
    ]
    uniform_winner = (
        min(uniform_names, key=lambda name: float(rows[name]["selection_rmse"]))
        if uniform_names
        else None
    )
    primary = uniform_winner or quality_winner
    return {
        "schema_version": 1,
        "selection_rule": {
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
            "held_hours": [2, 4],
            "seen_hours": [1, 3, 5],
            "held_weight": 0.7,
            "quality_gap_limit_vs_upr": quality_gap_limit,
            "latency_ratio_limit_vs_flow": latency_ratio_limit,
            "maximum_parameters_m": maximum_parameters_m,
            "maximum_cell_regression_vs_flow": cell_regression_limit,
            "primary_rule": (
                "uniform winner when available; otherwise minimum held-weighted "
                "RMSE among quality/compute eligible candidates"
            ),
        },
        "candidates": rows,
        "flow_reference": flow,
        "quality_reference": quality,
        "pareto_candidates": pareto,
        "quality_winner": quality_winner,
        "efficiency_winner": efficiency_winner,
        "uniform_winner": uniform_winner,
        "primary": primary,
        "winner": primary,
        "eligible_for_ood_confirmation": primary is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--flow-reference", type=Path, required=True)
    parser.add_argument("--quality-reference", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--quality-gap-limit", type=float, default=0.05)
    parser.add_argument("--latency-ratio-limit", type=float, default=1.25)
    parser.add_argument("--maximum-parameters-m", type=float, default=15.0)
    parser.add_argument("--cell-regression-limit", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "quality_gap_limit",
        "latency_ratio_limit",
        "maximum_parameters_m",
        "cell_regression_limit",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    result = select_champion(
        args.candidate,
        args.flow_reference,
        args.quality_reference,
        args.inference,
        quality_gap_limit=args.quality_gap_limit,
        latency_ratio_limit=args.latency_ratio_limit,
        maximum_parameters_m=args.maximum_parameters_m,
        cell_regression_limit=args.cell_regression_limit,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "primary": result["primary"],
                "quality_winner": result["quality_winner"],
                "efficiency_winner": result["efficiency_winner"],
                "uniform_winner": result["uniform_winner"],
                "pareto_candidates": result["pareto_candidates"],
            }
        )
    )


if __name__ == "__main__":
    main()
