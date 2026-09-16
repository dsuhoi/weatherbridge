#!/usr/bin/env python3
"""Gate the quarter-resolution UPR efficiency follow-up after two epochs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tools.eval.select_upr_lite_two_epoch_screen import read_two_epoch_row


DEFAULT_REFERENCE = (
    "exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707"
)
DEFAULT_CANDIDATE = (
    "exp_upr_lite_implicit_global_q4_24ch_6h_2014_19_"
    "sparse135_lr1e4_b8_s202707"
)


def assess(
    log_root: Path,
    *,
    reference_experiment: str = DEFAULT_REFERENCE,
    candidate_experiment: str = DEFAULT_CANDIDATE,
    maximum_relative_rmse: float = 1.02,
) -> dict[str, Any]:
    if maximum_relative_rmse < 1.0:
        raise ValueError("maximum_relative_rmse must be at least 1")
    reference = read_two_epoch_row(log_root / reference_experiment)
    candidate = read_two_epoch_row(log_root / candidate_experiment)
    ratio = candidate["held_rmse_mean"] / reference["held_rmse_mean"]
    return {
        "schema_version": 1,
        "followup": "quarter-resolution motion pyramid",
        "reference_arch": "upr_lite_implicit_global",
        "candidate_arch": "upr_lite_implicit_global_q4",
        "reference_experiment": reference_experiment,
        "candidate_experiment": candidate_experiment,
        "budget_epochs": 2,
        "scheduler_epochs": 8,
        "selection_year": 2020,
        "held_selection_tau": [2, 4],
        "maximum_relative_rmse": maximum_relative_rmse,
        "quality_ratio": ratio,
        "relative_delta_pct": 100.0 * (ratio - 1.0),
        "promoted": ratio <= maximum_relative_rmse,
        "reference": reference,
        "candidate": candidate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path(
            "/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs"
        ),
    )
    parser.add_argument("--reference-experiment", default=DEFAULT_REFERENCE)
    parser.add_argument("--candidate-experiment", default=DEFAULT_CANDIDATE)
    parser.add_argument("--maximum-relative-rmse", type=float, default=1.02)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/upr_lite_screen/q4_followup.json"),
    )
    args = parser.parse_args()

    report = assess(
        args.log_root,
        reference_experiment=args.reference_experiment,
        candidate_experiment=args.candidate_experiment,
        maximum_relative_rmse=args.maximum_relative_rmse,
    )
    report["assessment_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
