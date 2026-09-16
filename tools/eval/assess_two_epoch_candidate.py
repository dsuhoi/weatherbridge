#!/usr/bin/env python3
"""Apply a prespecified two-epoch compute-allocation gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tools.eval.summarize_matched_training_curves import build_report


def assess(
    candidate_csv: Path,
    reference_csv: Path,
    *,
    candidate_name: str,
    reference_name: str,
    held_relative_limit: float,
    all_hour_relative_limit: float,
    per_hour_relative_limit: float,
) -> dict:
    comparison = build_report(
        candidate_csv,
        reference_csv,
        left_name=candidate_name,
        right_name=reference_name,
        expand_version_siblings=False,
    )
    epoch = comparison["matched_epochs"].get("1")
    if epoch is None:
        raise ValueError(
            "candidate and reference both require two completed epochs"
        )
    per_hour_max = max(epoch["per_hour_relative_delta"].values())
    checks = {
        "held_relative_delta": (
            epoch["held_relative_delta"] <= held_relative_limit
        ),
        "all_hour_relative_delta": (
            epoch["all_hour_relative_delta"] <= all_hour_relative_limit
        ),
        "per_hour_relative_delta_max": (
            per_hour_max <= per_hour_relative_limit
        ),
    }
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "candidate": candidate_name,
        "reference": reference_name,
        "promoted": all(checks.values()),
        "checks": checks,
        "limits": {
            "held_relative_delta": held_relative_limit,
            "all_hour_relative_delta": all_hour_relative_limit,
            "per_hour_relative_delta_max": per_hour_relative_limit,
        },
        "observed": {
            "held_relative_delta": epoch["held_relative_delta"],
            "all_hour_relative_delta": epoch["all_hour_relative_delta"],
            "per_hour_relative_delta_max": per_hour_max,
        },
        "comparison": comparison,
        "note": (
            "This gate allocates training compute only. Full-year 2020 "
            "metrics select the model; 2021 remains temporal OOD."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--held-relative-limit", type=float, required=True)
    parser.add_argument("--all-hour-relative-limit", type=float, required=True)
    parser.add_argument("--per-hour-relative-limit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = assess(
        args.pilot_csv,
        args.reference_csv,
        candidate_name=args.candidate_name,
        reference_name=args.reference_name,
        held_relative_limit=args.held_relative_limit,
        all_hour_relative_limit=args.all_hour_relative_limit,
        per_hour_relative_limit=args.per_hour_relative_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"promoted": report["promoted"]}))


if __name__ == "__main__":
    main()
