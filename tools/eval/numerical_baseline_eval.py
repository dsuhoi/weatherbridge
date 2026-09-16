#!/usr/bin/env python3
"""Evaluate semi-Lagrangian and Hermite-advection numerical baselines on 2020.

Phase 1 of the EvaluationRunner refactor: this script is now a thin argparse
wrapper that delegates to ``NumericalBaselineRunner`` from
``weather_time_interp.eval_runner``. Output JSON is bit-identical to the
pre-refactor implementation (covered by ``tests/test_eval_runner.py``).

Reads same memmap as batch_eval_memmap.py, produces per-channel × per-hour RMSE
in the same JSON format. No model loading; pure numerical methods.

Usage:
    python tools/eval/numerical_baseline_eval.py \
        --memmap-dir /tmp/wb2_0p5_cache --test-year 2020 \
        --out-dir metrics/eval_0p5_2020_numeric
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.eval_runner import (  # noqa: E402
    Config,
    NumericalBaselineRunner,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="metrics/eval_0p5_2020_numeric")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument(
        "--full-year",
        action="store_true",
        help="Evaluate every valid anchor window instead of the economy subset.",
    )
    ap.add_argument("--dt-hours", type=float, default=6.0)
    ap.add_argument("--max-tau-hours", type=int, default=6)
    ap.add_argument(
        "--eval-hours",
        type=str,
        default="0,1,2,3,4,5,6",
        help="Comma-separated τ values to evaluate.",
    )
    ap.add_argument(
        "--legacy-summed-longitude-rmse",
        action="store_true",
        help="Reproduce historical RMSE that omitted the longitude mean.",
    )
    args = ap.parse_args()

    eval_hours = sorted({int(x) for x in args.eval_hours.split(",") if x.strip()})

    cfg = Config(
        memmap_dir=args.memmap_dir,
        test_years=[args.test_year],
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
        static_path=args.static_path,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        samples_per_date=args.samples_per_date,
        eval_days_per_month=None if args.full_year else args.eval_days_per_month,
        dt_hours=args.dt_hours,
        max_tau_hours=args.max_tau_hours,
        eval_hours=tuple(eval_hours),
        normalize_longitude=not args.legacy_summed_longitude_rmse,
    )
    NumericalBaselineRunner(cfg).run()


if __name__ == "__main__":
    main()
