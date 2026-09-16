#!/usr/bin/env python3
"""Gate a WeatherDCAE-FM pilot against its matched direct control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rmse_cells(report: dict, tau_hours: tuple[int, ...]) -> dict[str, float]:
    cells: dict[str, float] = {}
    for tau_hour in tau_hours:
        metrics = report["per_tau"][str(tau_hour)]["model"]
        for name, value in metrics.items():
            if name.startswith("rmse_norm_"):
                cells[f"tau{tau_hour}/{name.removeprefix('rmse_norm_')}"] = float(
                    value
                )
    return cells


def _comparison(candidate: dict, control: dict) -> dict:
    candidate_cells = _rmse_cells(candidate, (1, 2, 3, 4, 5))
    control_cells = _rmse_cells(control, (1, 2, 3, 4, 5))
    if candidate_cells.keys() != control_cells.keys():
        raise ValueError("candidate/control RMSE cells do not match")
    relative = {
        name: 100.0 * (candidate_cells[name] / control_cells[name] - 1.0)
        for name in candidate_cells
    }
    held = [value for name, value in relative.items() if name.startswith(("tau2/", "tau4/"))]
    return {
        "mean_rmse_change_pct": sum(relative.values()) / len(relative),
        "held_rmse_change_pct": sum(held) / len(held),
        "worst_cell_regression_pct": max(relative.values()),
        "wins": sum(value < 0.0 for value in relative.values()),
        "cells": len(relative),
        "strict_cellwise_dominance": all(value < 0.0 for value in relative.values()),
        "relative_change_pct_by_cell": relative,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--base", type=Path)
    parser.add_argument(
        "--reference",
        action="append",
        default=[],
        metavar="NAME:JSON",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = json.loads(args.candidate.read_text())
    control = json.loads(args.control.read_text())
    report = {
        "schema_version": 1,
        "selection_role": "era5_2020_screen_only",
        "candidate_vs_matched_control": _comparison(candidate, control),
    }
    if args.base is not None:
        report["candidate_vs_starting_base"] = _comparison(
            candidate,
            json.loads(args.base.read_text()),
        )
    for reference in args.reference:
        name, separator, raw_path = reference.partition(":")
        if not separator or not name or not raw_path:
            parser.error("--reference must be NAME:JSON")
        report[f"candidate_vs_{name}"] = _comparison(
            candidate,
            json.loads(Path(raw_path).read_text()),
        )

    comparison = report["candidate_vs_matched_control"]
    report["continue_seed_ensemble"] = bool(
        comparison["mean_rmse_change_pct"] < -0.02
        and comparison["held_rmse_change_pct"] < 0.02
        and comparison["worst_cell_regression_pct"] < 1.0
    )
    report["gate"] = {
        "mean_rmse_change_pct_lt": -0.02,
        "held_rmse_change_pct_lt": 0.02,
        "worst_cell_regression_pct_lt": 1.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["continue_seed_ensemble"] else 2)


if __name__ == "__main__":
    main()
