#!/usr/bin/env python3
"""Freeze the 2020 quality/efficiency decision for Universal-Pyramid."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def summarize_metrics(path: Path) -> dict[str, object]:
    data = _load(path)
    years = [int(year) for year in data.get("years", [])]
    if years != [2020]:
        raise ValueError(f"{path}: selection metrics must contain only 2020")
    channels = [str(name) for name in data.get("channel_names", [])]
    if len(channels) != 24:
        raise ValueError(f"{path}: expected 24 canonical channels")
    per_tau = data.get("per_tau", {})
    cells: dict[str, float] = {}
    tau_rmse: dict[str, float] = {}
    tau_acc: dict[str, float] = {}
    for hour in range(1, 6):
        key = str(hour)
        try:
            model = per_tau[key]["model"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"{path}: missing model metrics at hour {hour}") from error
        values = []
        for channel in channels:
            metric = f"rmse_norm_{channel}"
            value = float(model[metric])
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{path}: invalid {key}/{metric}={value}")
            cells[f"h{hour}/{channel}"] = value
            values.append(value)
        tau_rmse[key] = sum(values) / len(values)
        acc = float(model["acc_mean"])
        if not math.isfinite(acc):
            raise ValueError(f"{path}: invalid h{hour}/acc_mean={acc}")
        tau_acc[key] = acc
    held = 0.5 * (tau_rmse["2"] + tau_rmse["4"])
    seen = sum(tau_rmse[str(hour)] for hour in (1, 3, 5)) / 3.0
    return {
        "path": str(path.resolve()),
        "checkpoint": data.get("checkpoint"),
        "evaluation_index_sha256": data.get("evaluation_protocol", {}).get(
            "index_sha256"
        ),
        "rmse_mean": sum(tau_rmse.values()) / len(tau_rmse),
        "held_rmse_mean": held,
        "seen_rmse_mean": seen,
        "acc_mean": sum(tau_acc.values()) / len(tau_acc),
        "per_tau_rmse": tau_rmse,
        "cells": cells,
    }


def _relative(left: float, right: float) -> float:
    if right <= 0.0:
        raise ValueError("reference metric must be positive")
    return left / right - 1.0


def select(
    candidate_path: Path,
    flow_path: Path,
    quality_path: Path,
    inference_path: Path,
    *,
    quality_gap_limit: float,
    latency_ratio_limit: float,
    cell_regression_limit: float,
) -> dict[str, object]:
    candidate = summarize_metrics(candidate_path)
    flow = summarize_metrics(flow_path)
    quality = summarize_metrics(quality_path)
    indices = {
        candidate["evaluation_index_sha256"],
        flow["evaluation_index_sha256"],
        quality["evaluation_index_sha256"],
    }
    if None in indices or len(indices) != 1:
        raise ValueError("candidate and references use different evaluation indices")

    inference = _load(inference_path).get("models", {})
    try:
        candidate_cost = inference["universal_pyramid"]
        flow_cost = inference["flow_spectral"]
    except (KeyError, TypeError) as error:
        raise ValueError("inference JSON lacks universal_pyramid/flow_spectral") from error
    parameters = float(candidate_cost["params_m"])
    latency_ratio = float(candidate_cost["latency_ms"]) / float(
        flow_cost["latency_ms"]
    )

    relative_cells = {
        key: _relative(float(value), float(flow["cells"][key]))
        for key, value in candidate["cells"].items()
    }
    worst_cell = max(relative_cells, key=relative_cells.get)
    cell_wins = sum(value <= 0.0 for value in relative_cells.values())
    overall_vs_flow = _relative(
        float(candidate["rmse_mean"]),
        float(flow["rmse_mean"]),
    )
    held_vs_flow = _relative(
        float(candidate["held_rmse_mean"]),
        float(flow["held_rmse_mean"]),
    )
    overall_vs_quality = _relative(
        float(candidate["rmse_mean"]),
        float(quality["rmse_mean"]),
    )
    aggregate_quality_pass = overall_vs_flow <= 0.0 and held_vs_flow <= 0.0
    compute_pass = parameters <= 10.0 and latency_ratio <= latency_ratio_limit
    pareto_pass = (
        aggregate_quality_pass
        and overall_vs_quality <= quality_gap_limit
        and compute_pass
    )
    strict_uniform_pass = (
        pareto_pass
        and max(relative_cells.values()) <= cell_regression_limit
    )
    return {
        "schema_version": 1,
        "selection_rule": {
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
            "held_hours": [2, 4],
            "seen_hours": [1, 3, 5],
            "maximum_parameters_m": 10.0,
            "quality_gap_limit_vs_upr": quality_gap_limit,
            "latency_ratio_limit_vs_flow": latency_ratio_limit,
            "maximum_cell_regression_vs_flow": cell_regression_limit,
        },
        "candidate": candidate,
        "flow_reference": flow,
        "quality_reference": quality,
        "comparison": {
            "overall_relative_to_flow": overall_vs_flow,
            "held_relative_to_flow": held_vs_flow,
            "overall_relative_to_upr": overall_vs_quality,
            "parameters_m": parameters,
            "latency_ratio_to_flow": latency_ratio,
            "cell_wins_vs_flow": cell_wins,
            "cell_count": len(relative_cells),
            "worst_cell": worst_cell,
            "worst_cell_relative_to_flow": relative_cells[worst_cell],
            "per_cell_relative_to_flow": relative_cells,
        },
        "aggregate_quality_pass": aggregate_quality_pass,
        "compute_pass": compute_pass,
        "pareto_pass": pareto_pass,
        "strict_uniform_pass": strict_uniform_pass,
        "eligible_for_ood_evaluation": pareto_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--flow-reference", type=Path, required=True)
    parser.add_argument("--quality-reference", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--quality-gap-limit", type=float, default=0.05)
    parser.add_argument("--latency-ratio-limit", type=float, default=0.80)
    parser.add_argument("--cell-regression-limit", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "quality_gap_limit",
        "latency_ratio_limit",
        "cell_regression_limit",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    result = select(
        args.candidate,
        args.flow_reference,
        args.quality_reference,
        args.inference,
        quality_gap_limit=args.quality_gap_limit,
        latency_ratio_limit=args.latency_ratio_limit,
        cell_regression_limit=args.cell_regression_limit,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "pareto_pass": result["pareto_pass"],
                "strict_uniform_pass": result["strict_uniform_pass"],
                "eligible_for_ood_evaluation": result[
                    "eligible_for_ood_evaluation"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
