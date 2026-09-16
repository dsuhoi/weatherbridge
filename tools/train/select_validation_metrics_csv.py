#!/usr/bin/env python3
"""Select the Lightning CSV that contains a completed validation epoch."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


def _version(path: Path) -> int:
    match = re.fullmatch(r"version_(\d+)", path.parent.name)
    return int(match.group(1)) if match else -1


def _contains_validation(
    path: Path,
    *,
    required_epoch: int,
    required_step: int | None,
    metric: str,
) -> bool:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {"epoch", "step", metric}
        if not required_columns.issubset(reader.fieldnames or ()):
            return False
        for row in reader:
            try:
                epoch = int(row["epoch"])
                step = int(row["step"])
                value = float(row[metric])
            except (TypeError, ValueError):
                continue
            if (
                epoch == required_epoch
                and (required_step is None or step == required_step)
                and math.isfinite(value)
            ):
                return True
    return False


def select_validation_metrics_csv(
    run_dir: Path,
    *,
    required_epoch: int,
    required_step: int | None = None,
    metric: str = "val/rmse_mean",
) -> Path:
    candidates = [
        path
        for path in run_dir.glob("lightning_logs/version_*/metrics.csv")
        if _contains_validation(
            path,
            required_epoch=required_epoch,
            required_step=required_step,
            metric=metric,
        )
    ]
    if not candidates:
        step_note = (
            ""
            if required_step is None
            else f", step {required_step}"
        )
        raise FileNotFoundError(
            f"no metrics.csv contains validation epoch {required_epoch}"
            f"{step_note} with finite {metric}: {run_dir}"
        )
    return max(
        candidates,
        key=lambda path: (_version(path), path.stat().st_mtime_ns),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--required-epoch", type=int, required=True)
    parser.add_argument("--required-step", type=int)
    parser.add_argument("--metric", default="val/rmse_mean")
    args = parser.parse_args()
    print(
        select_validation_metrics_csv(
            args.run_dir,
            required_epoch=args.required_epoch,
            required_step=args.required_step,
            metric=args.metric,
        )
    )


if __name__ == "__main__":
    main()
