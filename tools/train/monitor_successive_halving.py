#!/usr/bin/env python3
"""Prune a training arm that persistently trails a matched reference curve."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import signal
import time
from pathlib import Path
from typing import Any


def load_validation_curve(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        return {}
    curve: dict[int, dict[str, float]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("val/rmse_mean"):
                continue
            epoch = int(row["epoch"])
            curve[epoch] = {
                "mean": float(row["val/rmse_mean"]),
                "held": 0.5
                * (
                    float(row["val/rmse_h2"])
                    + float(row["val/rmse_h4"])
                ),
            }
    return curve


def pruning_decision(
    candidate: dict[int, dict[str, float]],
    reference: dict[int, dict[str, float]],
    *,
    minimum_epoch: int,
    regression_threshold: float,
    hard_regression_threshold: float,
    patience: int,
) -> dict[str, Any] | None:
    consecutive = 0
    for epoch in sorted(set(candidate) & set(reference)):
        if epoch < minimum_epoch:
            continue
        candidate_row = candidate[epoch]
        reference_row = reference[epoch]
        mean_regression = candidate_row["mean"] / reference_row["mean"] - 1.0
        held_regression = candidate_row["held"] / reference_row["held"] - 1.0
        both_regress = (
            mean_regression > regression_threshold
            and held_regression > regression_threshold
        )
        consecutive = consecutive + 1 if both_regress else 0
        hard_failure = (
            mean_regression > hard_regression_threshold
            and held_regression > hard_regression_threshold
        )
        if hard_failure or consecutive >= patience:
            return {
                "decision_epoch": epoch,
                "candidate_rmse_mean": candidate_row["mean"],
                "reference_rmse_mean": reference_row["mean"],
                "relative_mean_regression": mean_regression,
                "candidate_held_rmse": candidate_row["held"],
                "reference_held_rmse": reference_row["held"],
                "relative_held_regression": held_regression,
                "hard_failure": hard_failure,
                "consecutive_failures": consecutive,
            }
    return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--target-pid", type=int, required=True)
    parser.add_argument("--minimum-epoch", type=int, default=1)
    parser.add_argument("--regression-threshold", type=float, default=0.03)
    parser.add_argument("--hard-regression-threshold", type=float, default=0.08)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--final-epoch", type=int, default=7)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    while _pid_alive(args.target_pid):
        candidate = load_validation_curve(args.candidate)
        reference = load_validation_curve(args.reference)
        decision = pruning_decision(
            candidate,
            reference,
            minimum_epoch=args.minimum_epoch,
            regression_threshold=args.regression_threshold,
            hard_regression_threshold=args.hard_regression_threshold,
            patience=args.patience,
        )
        if decision is not None:
            payload = {
                "schema_version": 1,
                "status": "pruned",
                "decision_time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "candidate": args.candidate_name,
                "reference": args.reference_name,
                "target_pid": args.target_pid,
                "rule": {
                    "minimum_epoch": args.minimum_epoch,
                    "regression_threshold": args.regression_threshold,
                    "hard_regression_threshold": args.hard_regression_threshold,
                    "patience": args.patience,
                    "metrics": ["val/rmse_mean", "mean(val/rmse_h2,h4)"],
                },
                **decision,
            }
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n")
            temporary.replace(args.out_json)
            os.kill(args.target_pid, signal.SIGTERM)
            print(json.dumps(payload))
            return
        if candidate and max(candidate) >= args.final_epoch:
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
