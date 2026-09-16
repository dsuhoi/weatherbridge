#!/usr/bin/env python3
"""Select a Refine candidate using a frozen sparse-tau validation rule."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_final_validation(path: Path) -> dict[str, float | int | str]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "epoch",
            "step",
            "val/rmse_h1",
            "val/rmse_h2",
            "val/rmse_h3",
            "val/rmse_h4",
            "val/rmse_h5",
            "val/rmse_mean",
        }
        if not required.issubset(reader.fieldnames or ()):
            missing = sorted(required.difference(reader.fieldnames or ()))
            raise ValueError(f"{path}: missing columns {missing}")
        for row in reader:
            try:
                parsed: dict[str, float | int | str] = {
                    "epoch": int(row["epoch"]),
                    "step": int(row["step"]),
                    "source": str(path.resolve()),
                }
                for hour in range(1, 6):
                    parsed[f"rmse_h{hour}"] = float(row[f"val/rmse_h{hour}"])
                parsed["rmse_mean"] = float(row["val/rmse_mean"])
            except (TypeError, ValueError):
                continue
            if all(
                math.isfinite(float(value))
                for key, value in parsed.items()
                if key.startswith("rmse_")
            ):
                rows.append(parsed)
    if not rows:
        raise ValueError(f"{path}: no complete validation rows")
    return max(rows, key=lambda row: (int(row["epoch"]), int(row["step"])))


def score_validation(
    metrics: dict[str, float | int | str],
    held_weight: float,
) -> dict[str, float | int | str]:
    held = 0.5 * (float(metrics["rmse_h2"]) + float(metrics["rmse_h4"]))
    seen = (
        float(metrics["rmse_h1"])
        + float(metrics["rmse_h3"])
        + float(metrics["rmse_h5"])
    ) / 3.0
    return {
        **metrics,
        "held_rmse_mean": held,
        "seen_rmse_mean": seen,
        "selection_score": held_weight * held + (1.0 - held_weight) * seen,
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition(":")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("expected NAME:PATH")
    return name, Path(raw_path)


def select_candidates(
    candidates: list[tuple[str, Path]],
    reference: tuple[str, Path],
    *,
    held_weight: float,
    overall_regression_limit: float,
) -> dict[str, object]:
    if not 0.0 <= held_weight <= 1.0:
        raise ValueError("held_weight must lie in [0, 1]")
    if overall_regression_limit < 0.0:
        raise ValueError("overall_regression_limit must be non-negative")
    if len(candidates) < 2:
        raise ValueError("at least two candidates are required")

    reference_name, reference_path = reference
    reference_metrics = score_validation(
        read_final_validation(reference_path),
        held_weight,
    )
    scored: dict[str, dict[str, object]] = {}
    for name, path in candidates:
        metrics = score_validation(read_final_validation(path), held_weight)
        relative_overall = (
            float(metrics["rmse_mean"])
            / float(reference_metrics["rmse_mean"])
            - 1.0
        )
        relative_held = (
            float(metrics["held_rmse_mean"])
            / float(reference_metrics["held_rmse_mean"])
            - 1.0
        )
        scored[name] = {
            **metrics,
            "relative_overall_vs_reference": relative_overall,
            "relative_held_vs_reference": relative_held,
            "screen_pass": relative_overall <= overall_regression_limit,
        }

    ranking = sorted(
        scored,
        key=lambda name: (
            float(scored[name]["selection_score"]),
            float(scored[name]["rmse_mean"]),
            name,
        ),
    )
    winner = ranking[0]
    return {
        "schema_version": 1,
        "selection_rule": {
            "held_hours": [2, 4],
            "seen_hours": [1, 3, 5],
            "held_weight": held_weight,
            "seen_weight": 1.0 - held_weight,
            "overall_regression_limit_vs_reference": overall_regression_limit,
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
        },
        "reference": {
            "name": reference_name,
            **reference_metrics,
        },
        "candidates": scored,
        "ranking": ranking,
        "winner": winner,
        "winner_screen_pass": bool(scored[winner]["screen_pass"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", type=parse_named_path, required=True)
    parser.add_argument("--reference", type=parse_named_path, required=True)
    parser.add_argument("--held-weight", type=float, default=0.7)
    parser.add_argument("--overall-regression-limit", type=float, default=0.005)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    result = select_candidates(
        args.candidate,
        args.reference,
        held_weight=args.held_weight,
        overall_regression_limit=args.overall_regression_limit,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"winner": result["winner"], "screen_pass": result["winner_screen_pass"]}))


if __name__ == "__main__":
    main()
