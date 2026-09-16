#!/usr/bin/env python3
"""Compare matched validation trajectories without treating them as test skill."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence


PathInput = Path | Sequence[Path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_sort_key(path: Path) -> tuple[int, str]:
    name = path.parent.name
    if name.startswith("version_"):
        suffix = name.removeprefix("version_")
        if suffix.isdigit():
            return int(suffix), str(path)
    return 2**31 - 1, str(path)


def _resolve_csv_paths(
    value: PathInput,
    *,
    expand_version_siblings: bool = True,
) -> tuple[Path, ...]:
    requested = (value,) if isinstance(value, Path) else tuple(value)
    if not requested:
        raise ValueError("at least one metrics CSV is required")
    if expand_version_siblings and len(requested) == 1:
        path = requested[0]
        if (
            path.name == "metrics.csv"
            and path.parent.name.startswith("version_")
            and path.parent.parent.name == "lightning_logs"
        ):
            siblings = tuple(
                sorted(
                    path.parent.parent.glob("version_*/metrics.csv"),
                    key=_version_sort_key,
                )
            )
            if siblings:
                requested = siblings
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in requested:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return tuple(unique)


def load_validation_curve(
    paths: PathInput,
    *,
    hours: tuple[int, ...],
    steps_per_epoch: int,
    expand_version_siblings: bool = True,
) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    sources: dict[int, Path] = {}
    resolved_paths = _resolve_csv_paths(
        paths,
        expand_version_siblings=expand_version_siblings,
    )
    for path in resolved_paths:
        with path.open(newline="") as handle:
            for raw in csv.DictReader(handle):
                if not raw.get("val/recon_l1"):
                    continue
                epoch = int(raw["epoch"])
                if epoch in rows:
                    raise ValueError(
                        f"{path}: duplicate validation epoch {epoch}; "
                        f"already present in {sources[epoch]}"
                    )
                expected_step = (epoch + 1) * steps_per_epoch - 1
                step = int(raw["step"])
                if step != expected_step:
                    raise ValueError(
                        f"{path}: epoch {epoch} step {step} != {expected_step}"
                    )
                per_hour = {
                    hour: float(raw[f"val/rmse_h{hour}"])
                    for hour in hours
                }
                if not all(math.isfinite(value) for value in per_hour.values()):
                    raise ValueError(f"{path}: non-finite validation metric")
                rows[epoch] = {
                    "step": step,
                    "per_hour_rmse": {
                        str(hour): per_hour[hour]
                        for hour in hours
                    },
                    "all_hour_mean": sum(per_hour.values()) / len(per_hour),
                }
                sources[epoch] = path
    if not rows:
        joined = ", ".join(str(path) for path in resolved_paths)
        raise ValueError(f"{joined}: no completed validation epochs")
    return rows


def build_report(
    left_path: PathInput,
    right_path: PathInput,
    *,
    left_name: str,
    right_name: str,
    hours: tuple[int, ...] = (1, 2, 3, 4, 5),
    held_hours: tuple[int, ...] = (2, 4),
    steps_per_epoch: int = 1642,
    expand_version_siblings: bool = True,
) -> dict[str, Any]:
    if not set(held_hours).issubset(hours):
        raise ValueError("held hours must be included in all hours")
    left = load_validation_curve(
        left_path,
        hours=hours,
        steps_per_epoch=steps_per_epoch,
        expand_version_siblings=expand_version_siblings,
    )
    right = load_validation_curve(
        right_path,
        hours=hours,
        steps_per_epoch=steps_per_epoch,
        expand_version_siblings=expand_version_siblings,
    )

    def add_held(curve: dict[int, dict[str, Any]]) -> None:
        for row in curve.values():
            row["held_rmse"] = sum(
                row["per_hour_rmse"][str(hour)]
                for hour in held_hours
            ) / len(held_hours)

    add_held(left)
    add_held(right)
    matched: dict[str, dict[str, Any]] = {}
    for epoch in sorted(set(left) & set(right)):
        left_row = left[epoch]
        right_row = right[epoch]
        matched[str(epoch)] = {
            "step": left_row["step"],
            f"{left_name}_held_rmse": left_row["held_rmse"],
            f"{right_name}_held_rmse": right_row["held_rmse"],
            "held_relative_delta": (
                left_row["held_rmse"] / right_row["held_rmse"] - 1.0
            ),
            f"{left_name}_all_hour_mean": left_row["all_hour_mean"],
            f"{right_name}_all_hour_mean": right_row["all_hour_mean"],
            "all_hour_relative_delta": (
                left_row["all_hour_mean"] / right_row["all_hour_mean"] - 1.0
            ),
            "per_hour_relative_delta": {
                str(hour): (
                    left_row["per_hour_rmse"][str(hour)]
                    / right_row["per_hour_rmse"][str(hour)]
                    - 1.0
                )
                for hour in hours
            },
        }
    reference_final_epoch = max(right)
    reference_final_held = right[reference_final_epoch]["held_rmse"]
    first_crossing = next(
        (
            epoch
            for epoch in sorted(left)
            if left[epoch]["held_rmse"] < reference_final_held
        ),
        None,
    )
    left_paths = _resolve_csv_paths(
        left_path,
        expand_version_siblings=expand_version_siblings,
    )
    right_paths = _resolve_csv_paths(
        right_path,
        expand_version_siblings=expand_version_siblings,
    )
    return {
        "schema_version": 2,
        "diagnostic_only": True,
        "left": left_name,
        "right": right_name,
        "hours": list(hours),
        "held_hours": list(held_hours),
        "steps_per_epoch": steps_per_epoch,
        "input_sha256": {
            left_name: [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in left_paths
            ],
            right_name: [
                {
                    "path": str(path.resolve()),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in right_paths
            ],
        },
        "completed_epochs": {
            left_name: sorted(left),
            right_name: sorted(right),
        },
        "matched_epochs": matched,
        "reference_final": {
            "epoch": reference_final_epoch,
            "held_rmse": reference_final_held,
        },
        "left_first_epoch_better_than_reference_final": first_crossing,
        "note": (
            "Training-validation trajectories are optimization diagnostics; "
            "full-year 2020 selection and frozen 2021 confirmation remain "
            "the evidence for model quality and generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--right-csv", type=Path, nargs="+", required=True)
    parser.add_argument("--left-name", required=True)
    parser.add_argument("--right-name", required=True)
    parser.add_argument("--hours", default="1,2,3,4,5")
    parser.add_argument("--held-hours", default="2,4")
    parser.add_argument("--steps-per-epoch", type=int, default=1642)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        args.left_csv,
        args.right_csv,
        left_name=args.left_name,
        right_name=args.right_name,
        hours=tuple(
            int(value) for value in args.hours.split(",") if value.strip()
        ),
        held_hours=tuple(
            int(value)
            for value in args.held_hours.split(",")
            if value.strip()
        ),
        steps_per_epoch=args.steps_per_epoch,
    )
    report["generator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "matched_epochs": list(report["matched_epochs"]),
                "first_crossing": report[
                    "left_first_epoch_better_than_reference_final"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
