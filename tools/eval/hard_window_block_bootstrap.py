#!/usr/bin/env python3
"""Paired weekly-block tests on a model-independent hard-window subset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    assert_paired,
    load_metrics,
    paired_block_bootstrap,
    sha256_file,
)


def parse_entry(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator:
        raise argparse.ArgumentTypeError("expected NAME:NPZ")
    return name, Path(path)


def subset_metrics(metrics: WindowMetrics, mask: np.ndarray) -> WindowMetrics:
    if mask.shape != metrics.t0.shape or not mask.any():
        raise ValueError("hard-window mask must select at least one window")
    return WindowMetrics(
        year=metrics.year[mask],
        t0=metrics.t0[mask],
        tau=metrics.tau[mask],
        mse=metrics.mse[mask],
        channels=metrics.channels,
    )


def hard_window_comparisons(
    left_path: Path,
    rights: list[tuple[str, Path]],
    *,
    quantile: float,
    taus: np.ndarray | None,
    block_days: int,
    draws: int,
    seed: int,
) -> dict:
    """Compare models on windows selected only by bilinear interpolation error."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("quantile must lie strictly between 0.5 and 1")

    left = load_metrics(left_path)
    baseline = load_metrics(left_path, method="bilinear")
    assert_paired(left, baseline)
    tau_values = np.unique(left.tau) if taus is None else np.asarray(taus)
    score = np.sqrt(np.maximum(baseline.mse.mean(axis=1), 0.0))
    mask = np.zeros(score.shape, dtype=bool)
    thresholds: dict[str, float] = {}
    selected_by_tau: dict[str, int] = {}
    for tau in tau_values:
        tau_mask = baseline.tau == tau
        if not tau_mask.any():
            raise ValueError(f"no bilinear windows for tau={int(tau)}")
        threshold = float(np.quantile(score[tau_mask], quantile))
        selected = tau_mask & (score >= threshold)
        mask |= selected
        thresholds[str(int(tau))] = threshold
        selected_by_tau[str(int(tau))] = int(selected.sum())

    left_subset = subset_metrics(left, mask)
    comparisons = {}
    for right_name, right_path in rights:
        right = load_metrics(right_path)
        assert_paired(left, right)
        comparisons[right_name] = paired_block_bootstrap(
            left_subset,
            subset_metrics(right, mask),
            taus=tau_values,
            block_days=block_days,
            draws=draws,
            seed=seed,
            include_cellwise=True,
        )
    return {
        "metric": "hard_window_rmse",
        "subset_definition": (
            "per-tau upper tail of normalized bilinear aggregate window RMSE"
        ),
        "selection_is_model_independent": True,
        "quantile": quantile,
        "threshold_by_tau": thresholds,
        "selected_windows_by_tau": selected_by_tau,
        "n_windows": int(mask.sum()),
        "index_sha256": left.index_sha256,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, type=parse_entry)
    parser.add_argument("--right", action="append", required=True, type=parse_entry)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--taus", default="")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    left_name, left_path = args.left
    taus = (
        np.asarray([int(value) for value in args.taus.split(",") if value])
        if args.taus
        else None
    )
    result = hard_window_comparisons(
        left_path,
        args.right,
        quantile=args.quantile,
        taus=taus,
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
    )
    output = {
        "schema_version": 1,
        "left": left_name,
        "left_path": str(left_path),
        "left_sha256": sha256_file(left_path),
        "right_sha256": {
            name: sha256_file(path) for name, path in args.right
        },
        **result,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
