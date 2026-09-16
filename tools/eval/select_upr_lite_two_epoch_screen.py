#!/usr/bin/env python3
"""Select UPR-Lite arms after the fixed two-epoch 6h screen."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


EXPERIMENTS = {
    "upr_lite": (
        "exp_upr_lite_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707",
        0.709,
    ),
    "upr_lite_lap": (
        "exp_upr_lite_lap_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707",
        0.737,
    ),
    "upr_lite_column": (
        "exp_upr_lite_column_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707",
        0.747,
    ),
    "upr_lite_continuous": (
        "exp_upr_lite_continuous_24ch_6h_2014_19_sparse135_lr1e4_gwarp_b8_s202707",
        0.762,
    ),
    "upr_lite_continuous_m": (
        "exp_upr_lite_continuous_m_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707",
        1.590,
    ),
    "upr_lite_implicit_global": (
        "exp_upr_lite_implicit_global_24ch_6h_2014_19_sparse135_lr1e4_b8_s202707",
        1.660,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(path: Path) -> int:
    for part in path.parts:
        if part.startswith("version_"):
            try:
                return int(part.removeprefix("version_"))
            except ValueError:
                pass
    return -1


def read_two_epoch_row(experiment_dir: Path) -> dict[str, Any]:
    metric_files = sorted(
        experiment_dir.rglob("metrics.csv"),
        key=lambda path: (_version(path), path.stat().st_mtime_ns),
    )
    matches: list[tuple[Path, dict[str, str]]] = []
    for path in metric_files:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("epoch") != "1" or not row.get("val/rmse_h2"):
                    continue
                matches.append((path, row))
    if not matches:
        raise FileNotFoundError(
            f"{experiment_dir}: no completed validation row for epoch 1"
        )

    path, row = matches[-1]
    rmse_h2 = float(row["val/rmse_h2"])
    rmse_h4 = float(row["val/rmse_h4"])
    if not math.isfinite(rmse_h2) or not math.isfinite(rmse_h4):
        raise ValueError(f"{path}: non-finite held-hour RMSE")
    return {
        "rmse_h2": rmse_h2,
        "rmse_h4": rmse_h4,
        "held_rmse_mean": (rmse_h2 + rmse_h4) / 2.0,
        "epoch": 1,
        "step": int(row["step"]),
        "metrics_file": str(path),
        "metrics_sha256": _sha256(path),
    }


def build_screen_report(
    log_root: Path,
    experiments: dict[str, tuple[str, float]] = EXPERIMENTS,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for arch, (experiment, params_m) in experiments.items():
        try:
            metrics = read_two_epoch_row(log_root / experiment)
        except FileNotFoundError:
            missing.append(arch)
            continue
        rows[arch] = {
            "experiment": experiment,
            "params_m": params_m,
            **metrics,
        }

    ranking = sorted(
        rows,
        key=lambda arch: (
            rows[arch]["held_rmse_mean"],
            rows[arch]["params_m"],
            arch,
        ),
    )
    return {
        "schema_version": 1,
        "screen": {
            "selection_year": 2020,
            "budget_epochs": 2,
            "scheduler_epochs": 8,
            "train_years": [2014, 2015, 2016, 2017, 2018, 2019],
            "validation_years": [2020],
            "learning_rate": 1e-4,
            "effective_batch_size": 16,
            "precision": "bf16-mixed",
            "seed": 202707,
            "supervised_train_tau": [1, 3, 5],
            "held_selection_tau": [2, 4],
            "score": "mean(val/rmse_h2, val/rmse_h4)",
            "ranking_tie_breakers": ["params_m", "architecture_name"],
            "objective": {
                "common": "cosine-latitude-weighted L1",
                "plain_upr_lite_control": "no high-pass term",
                "detail_arms": "common + 0.05 * local high-pass L1",
            },
            "temporal_ood_2021_used": False,
            "retain_count": 3,
        },
        "complete": not missing,
        "missing": missing,
        "ranking": ranking,
        "selected": ranking[:3] if not missing else [],
        "models": rows,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/upr_lite_screen/two_epoch_screen.json"),
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()

    report = build_screen_report(args.log_root)
    report["selection_code_sha256"] = _sha256(Path(__file__))
    if not report["complete"] and not args.allow_incomplete:
        raise SystemExit(
            "two-epoch screen is incomplete: " + ", ".join(report["missing"])
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(
        f".{args.output.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
