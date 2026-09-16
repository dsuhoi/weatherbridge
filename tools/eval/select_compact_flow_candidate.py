#!/usr/bin/env python3
"""Rank short-budget compact models against the matched Flow reference."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from tools.eval.summarize_matched_training_curves import build_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_candidate(
    candidates: dict[str, Path],
    reference_csv: Path,
    *,
    reference_name: str,
    epoch: int,
    held_relative_limit: float,
    all_hour_relative_limit: float,
    per_hour_relative_limit: float,
) -> dict:
    if not candidates:
        raise ValueError("at least one candidate is required")

    results: dict[str, dict] = {}
    for name, path in candidates.items():
        comparison = build_report(
            path,
            reference_csv,
            left_name=name,
            right_name=reference_name,
            expand_version_siblings=False,
        )
        row = comparison["matched_epochs"].get(str(epoch))
        if row is None:
            raise ValueError(
                f"{name} and {reference_name} require completed epoch {epoch}"
            )
        per_hour_max = max(row["per_hour_relative_delta"].values())
        checks = {
            "held_relative_delta": (
                row["held_relative_delta"] <= held_relative_limit
            ),
            "all_hour_relative_delta": (
                row["all_hour_relative_delta"] <= all_hour_relative_limit
            ),
            "per_hour_relative_delta_max": (
                per_hour_max <= per_hour_relative_limit
            ),
        }
        results[name] = {
            "eligible": all(checks.values()),
            "checks": checks,
            "held_rmse": row[f"{name}_held_rmse"],
            "all_hour_rmse": row[f"{name}_all_hour_mean"],
            "held_relative_delta": row["held_relative_delta"],
            "all_hour_relative_delta": row["all_hour_relative_delta"],
            "per_hour_relative_delta": row["per_hour_relative_delta"],
            "per_hour_relative_delta_max": per_hour_max,
            "metrics_csv": str(path.resolve()),
            "metrics_sha256": _sha256(path),
        }

    eligible = [
        name for name, result in results.items() if result["eligible"]
    ]
    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["held_rmse"],
            results[name]["all_hour_rmse"],
            name,
        ),
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "epoch": epoch,
        "reference": reference_name,
        "reference_metrics_csv": str(reference_csv.resolve()),
        "reference_metrics_sha256": _sha256(reference_csv),
        "limits": {
            "held_relative_delta": held_relative_limit,
            "all_hour_relative_delta": all_hour_relative_limit,
            "per_hour_relative_delta_max": per_hour_relative_limit,
        },
        "ranking": ranking,
        "eligible": eligible,
        "selected": next((name for name in ranking if name in eligible), None),
        "candidates": results,
        "note": (
            "This short-budget validation gate allocates compute only. "
            "Full-year selection and untouched-year evaluation are required "
            "before making a generalization claim."
        ),
    }


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=METRICS_CSV")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        required=True,
        metavar="NAME=METRICS_CSV",
    )
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--reference-name", default="weatherbridge_ref")
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--held-relative-limit", type=float, required=True)
    parser.add_argument("--all-hour-relative-limit", type=float, required=True)
    parser.add_argument("--per-hour-relative-limit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate names must be unique")
    report = select_candidate(
        candidates,
        args.reference_csv,
        reference_name=args.reference_name,
        epoch=args.epoch,
        held_relative_limit=args.held_relative_limit,
        all_hour_relative_limit=args.all_hour_relative_limit,
        per_hour_relative_limit=args.per_hour_relative_limit,
    )
    report["generator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "selected": report["selected"],
                "ranking": report["ranking"],
            }
        )
    )


if __name__ == "__main__":
    main()
