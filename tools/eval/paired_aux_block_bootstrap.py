#!/usr/bin/env python3
"""Paired weekly-block tests for ACC, temporal, and physical diagnostics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.eval.paired_block_bootstrap import (
    load_acc_scores,
    load_temporal_metrics,
    paired_block_bootstrap,
    paired_cellwise_score_tests,
    sha256_file,
)


def parse_entry(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected NAME:NPZ")
    return name, Path(path)


def json_default(value: object) -> object:
    """Convert NumPy values emitted by metric helpers to JSON-native types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=parse_entry)
    parser.add_argument("--right", action="append", required=True, type=parse_entry)
    parser.add_argument(
        "--metric",
        choices=("acc", "temporal_curvature", "physical"),
        required=True,
    )
    parser.add_argument("--taus", default="")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    left_name, left_path = args.left
    comparisons = {}
    right_sha256 = {}
    if args.metric == "acc":
        left = load_acc_scores(left_path)
        taus = (
            np.asarray([int(value) for value in args.taus.split(",") if value])
            if args.taus
            else np.unique(left.tau)
        )
        for right_name, right_path in args.right:
            comparisons[right_name] = paired_cellwise_score_tests(
                left,
                load_acc_scores(right_path),
                taus=taus,
                block_days=args.block_days,
                draws=args.draws,
                seed=args.seed,
                better="higher",
            )
            right_sha256[right_name] = sha256_file(right_path)
    elif args.metric == "temporal_curvature":
        left = load_temporal_metrics(left_path)
        for right_name, right_path in args.right:
            comparisons[right_name] = paired_block_bootstrap(
                left,
                load_temporal_metrics(right_path),
                taus=np.asarray([0], dtype=np.int8),
                block_days=args.block_days,
                draws=args.draws,
                seed=args.seed,
                include_cellwise=True,
            )
            right_sha256[right_name] = sha256_file(right_path)
    else:
        from tools.eval.physical_block_bootstrap import (
            compare_physical,
            load_physical_windows,
        )

        left = load_physical_windows(left_path)
        taus = (
            np.asarray([int(value) for value in args.taus.split(",") if value])
            if args.taus
            else np.unique(left.tau)
        )
        for right_name, right_path in args.right:
            comparisons[right_name] = compare_physical(
                left,
                load_physical_windows(right_path),
                taus=taus,
                block_days=args.block_days,
                draws=args.draws,
                seed=args.seed,
            )
            right_sha256[right_name] = sha256_file(right_path)

    output = {
        "schema_version": 2,
        "metric": args.metric,
        "left": left_name,
        "left_path": str(left_path),
        "left_sha256": sha256_file(left_path),
        "right_sha256": right_sha256,
        "index_sha256": left.index_sha256,
        "comparisons": comparisons,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2, default=json_default) + "\n")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
