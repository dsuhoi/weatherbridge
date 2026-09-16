"""Select a specialist-distillation arm on ERA5 validation evidence only."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def _load_rmse(path: Path) -> dict[int, dict[str, float]]:
    payload = json.loads(path.read_text())
    result: dict[int, dict[str, float]] = {}
    for tau_text, tau_payload in payload["per_tau"].items():
        model = tau_payload["model"]
        result[int(tau_text)] = {
            key.removeprefix("rmse_norm_"): float(value)
            for key, value in model.items()
            if key.startswith("rmse_norm_")
        }
    return result


def _aggregate(
    metrics: dict[int, dict[str, float]],
    *,
    taus: tuple[int, ...] | None = None,
    channels: tuple[str, ...] | None = None,
) -> float:
    selected_taus = taus or tuple(sorted(metrics))
    selected_channels = channels or tuple(sorted(next(iter(metrics.values()))))
    values = [
        metrics[tau][channel]
        for tau in selected_taus
        for channel in selected_channels
    ]
    return math.sqrt(sum(value * value for value in values) / len(values))


def assess_candidates(
    control: dict[int, dict[str, float]],
    candidates: dict[str, dict[int, dict[str, float]]],
    *,
    aggregate_tolerance_pct: float,
    held_tolerance_pct: float,
) -> dict[str, object]:
    """Apply aggregate guards, then rank candidates by moisture transfer."""
    channels = tuple(sorted(next(iter(control.values()))))
    moisture = tuple(channel for channel in channels if channel.startswith("Q"))
    non_moisture = tuple(channel for channel in channels if channel not in moisture)
    held_taus = tuple(tau for tau in (2, 4) if tau in control)

    def relative(candidate: float, reference: float) -> float:
        return 100.0 * (candidate / reference - 1.0)

    control_summary = {
        "aggregate_rmse": _aggregate(control),
        "moisture_rmse": _aggregate(control, channels=moisture),
        "non_moisture_rmse": _aggregate(control, channels=non_moisture),
        "held_rmse": _aggregate(control, taus=held_taus),
    }
    assessments: dict[str, dict[str, object]] = {}
    for name, metrics in candidates.items():
        summary = {
            "aggregate_rmse": _aggregate(metrics),
            "moisture_rmse": _aggregate(metrics, channels=moisture),
            "non_moisture_rmse": _aggregate(metrics, channels=non_moisture),
            "held_rmse": _aggregate(metrics, taus=held_taus),
        }
        deltas = {
            key.removesuffix("_rmse") + "_delta_pct": relative(
                value,
                control_summary[key],
            )
            for key, value in summary.items()
        }
        checks = {
            "aggregate_noninferior": (
                deltas["aggregate_delta_pct"] <= aggregate_tolerance_pct
            ),
            "held_noninferior": (
                deltas["held_delta_pct"] <= held_tolerance_pct
            ),
            "moisture_improves": deltas["moisture_delta_pct"] < 0.0,
        }
        assessments[name] = {
            "pass": all(checks.values()),
            "checks": checks,
            "metrics": summary,
            "deltas_vs_control_pct": deltas,
        }
    passing = [name for name, item in assessments.items() if item["pass"]]
    selected = min(
        passing,
        key=lambda name: (
            assessments[name]["deltas_vs_control_pct"]["moisture_delta_pct"],
            assessments[name]["deltas_vs_control_pct"]["aggregate_delta_pct"],
        ),
        default=None,
    )
    return {
        "status": "pass" if selected is not None else "fail",
        "selected": selected,
        "selection_role": "era5_2020_validation_only",
        "thresholds": {
            "aggregate_tolerance_pct": aggregate_tolerance_pct,
            "held_tolerance_pct": held_tolerance_pct,
        },
        "control": control_summary,
        "candidates": assessments,
    }


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition(":")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME:PATH")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True, type=Path)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=_parse_named_path,
    )
    parser.add_argument("--aggregate-tolerance-pct", type=float, default=0.02)
    parser.add_argument("--held-tolerance-pct", type=float, default=0.02)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    for value in (args.aggregate_tolerance_pct, args.held_tolerance_pct):
        if not math.isfinite(value) or value < 0.0:
            parser.error("selection tolerances must be finite and non-negative")
    result = assess_candidates(
        _load_rmse(args.control),
        {name: _load_rmse(path) for name, path in args.candidate},
        aggregate_tolerance_pct=args.aggregate_tolerance_pct,
        held_tolerance_pct=args.held_tolerance_pct,
    )
    result["provenance"] = {
        "control": {
            "path": str(args.control.resolve()),
            "sha256": hashlib.sha256(args.control.read_bytes()).hexdigest(),
        },
        "candidates": {
            name: {
                "path": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in args.candidate
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out)
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
