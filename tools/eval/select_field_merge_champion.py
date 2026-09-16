#!/usr/bin/env python3
"""Select a field-head merge only when it strictly improves both controls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


TARGET_FIELDS = ("Z1000", "Z925", "Z850", "Z700", "mslp")


def _load_rmse(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text())
    return {
        int(tau): {
            key.removeprefix("rmse_norm_"): float(value)
            for key, value in item["model"].items()
            if key.startswith("rmse_norm_")
        }
        for tau, item in payload["per_tau"].items()
    }


def _aggregate(
    metrics: dict[int, dict[str, float]],
    *,
    taus: tuple[int, ...] | None = None,
    channels: tuple[str, ...] | None = None,
) -> float:
    selected_taus = taus or tuple(sorted(metrics))
    selected_channels = channels or tuple(next(iter(metrics.values())))
    values = [metrics[tau][channel] for tau in selected_taus for channel in selected_channels]
    return math.sqrt(sum(value * value for value in values) / len(values))


def _summary(metrics: dict[int, dict[str, float]]) -> dict[str, float]:
    channels = tuple(next(iter(metrics.values())))
    excluded = tuple(channel for channel in channels if channel not in TARGET_FIELDS)
    return {
        "aggregate_rmse": _aggregate(metrics),
        "target_rmse": _aggregate(metrics, channels=TARGET_FIELDS),
        "excluded_rmse": _aggregate(metrics, channels=excluded),
        "held_rmse": _aggregate(metrics, taus=(2, 4)),
    }


def _relative(candidate: float, control: float) -> float:
    return 100.0 * (candidate / control - 1.0)


def assess_field_merges(
    starting: dict[int, dict[str, float]],
    matched: dict[int, dict[str, float]],
    candidates: dict[str, dict[int, dict[str, float]]],
) -> dict[str, object]:
    controls = {"starting": _summary(starting), "matched": _summary(matched)}
    assessments = {}
    for name, metrics in candidates.items():
        summary = _summary(metrics)
        comparisons = {}
        passed = True
        for control_name, control in controls.items():
            deltas = {
                key.removesuffix("_rmse") + "_delta_pct": _relative(value, control[key])
                for key, value in summary.items()
            }
            checks = {
                "aggregate_improves": deltas["aggregate_delta_pct"] < 0.0,
                "target_improves": deltas["target_delta_pct"] < 0.0,
                "excluded_noninferior": deltas["excluded_delta_pct"] <= 0.0,
                "held_noninferior": deltas["held_delta_pct"] <= 0.0,
            }
            comparison_pass = all(checks.values())
            passed = passed and comparison_pass
            comparisons[control_name] = {
                "pass": comparison_pass,
                "checks": checks,
                "deltas_pct": deltas,
            }
        assessments[name] = {
            "pass": passed,
            "metrics": summary,
            "comparisons": comparisons,
        }
    passing = [name for name, item in assessments.items() if item["pass"]]
    selected = min(
        passing,
        key=lambda name: assessments[name]["comparisons"]["matched"]["deltas_pct"]["aggregate_delta_pct"],
        default=None,
    )
    return {
        "schema_version": 1,
        "status": "pass" if selected is not None else "fail",
        "selected": selected,
        "selection_role": "era5_2020_validation_only",
        "target_fields": list(TARGET_FIELDS),
        "controls": controls,
        "candidates": assessments,
    }


def _named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME:PATH")
    return name, Path(path)


def _provenance(path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting-control", type=Path, required=True)
    parser.add_argument("--matched-control", type=Path, required=True)
    parser.add_argument("--candidate", action="append", type=_named_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = assess_field_merges(
        _load_rmse(args.starting_control),
        _load_rmse(args.matched_control),
        {name: _load_rmse(path) for name, path in args.candidate},
    )
    payload["provenance"] = {
        "starting_control": _provenance(args.starting_control),
        "matched_control": _provenance(args.matched_control),
        "candidates": {name: _provenance(path) for name, path in args.candidate},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "selected": payload["selected"]}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
