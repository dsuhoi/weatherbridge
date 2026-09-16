#!/usr/bin/env python3
"""Select a compact fine-tune using non-moisture fieldwise RMSE."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SEEN_TAUS = ("1", "3", "5")
HELD_TAUS = ("2", "4")
SURFACE_FIELDS = ("t2m", "u10", "v10", "mslp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_deltas(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    fields: list[str],
    taus: tuple[str, ...],
) -> list[float]:
    deltas: list[float] = []
    for tau in taus:
        candidate_metrics = candidate["per_tau"][tau]["model"]
        reference_metrics = reference["per_tau"][tau]["model"]
        for field in fields:
            key = f"rmse_norm_{field}"
            deltas.append(
                candidate_metrics[key] / reference_metrics[key] - 1.0
            )
    return deltas


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty metric collection")
    return sum(values) / len(values)


def build_selection(
    candidates: dict[str, Path],
    reference_path: Path,
    *,
    base_mean_limit: float,
    seen_mean_limit: float,
    held_mean_limit: float,
    per_tau_limit: float,
    surface_mean_limit: float,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text())
    fields = list(reference["channel_names"])
    base_fields = [field for field in fields if not field.startswith("Q")]
    if len(fields) != 24 or len(base_fields) != 20:
        raise ValueError("expected 24 fields with four Q pressure levels")

    results: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        candidate = json.loads(path.read_text())
        for key in ("years", "n_per_tau", "evaluation_protocol"):
            if candidate[key] != reference[key]:
                raise ValueError(f"{name}: evaluation protocol mismatch at {key}")
        if candidate["channel_names"] != fields:
            raise ValueError(f"{name}: channel order mismatch")

        per_tau = {
            tau: _mean(
                _relative_deltas(
                    candidate,
                    reference,
                    base_fields,
                    (tau,),
                )
            )
            for tau in (*SEEN_TAUS, *HELD_TAUS)
        }
        base_deltas = _relative_deltas(
            candidate,
            reference,
            base_fields,
            (*SEEN_TAUS, *HELD_TAUS),
        )
        seen_deltas = _relative_deltas(
            candidate,
            reference,
            base_fields,
            SEEN_TAUS,
        )
        held_deltas = _relative_deltas(
            candidate,
            reference,
            base_fields,
            HELD_TAUS,
        )
        surface_deltas = _relative_deltas(
            candidate,
            reference,
            list(SURFACE_FIELDS),
            (*SEEN_TAUS, *HELD_TAUS),
        )
        checks = {
            "base_mean": _mean(base_deltas) <= base_mean_limit,
            "seen_mean": _mean(seen_deltas) <= seen_mean_limit,
            "held_mean": _mean(held_deltas) <= held_mean_limit,
            "per_tau_max": max(per_tau.values()) <= per_tau_limit,
            "surface_mean": (
                _mean(surface_deltas) <= surface_mean_limit
            ),
        }
        results[name] = {
            "eligible": all(checks.values()),
            "checks": checks,
            "base_mean_relative_delta": _mean(base_deltas),
            "seen_mean_relative_delta": _mean(seen_deltas),
            "held_mean_relative_delta": _mean(held_deltas),
            "surface_mean_relative_delta": _mean(surface_deltas),
            "per_tau_base_mean_relative_delta": per_tau,
            "base_field_tau_wins": sum(
                delta < 0.0 for delta in base_deltas
            ),
            "base_field_tau_total": len(base_deltas),
            "metrics_path": str(path.resolve()),
            "metrics_sha256": _sha256(path),
        }

    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["base_mean_relative_delta"],
            results[name]["surface_mean_relative_delta"],
            results[name]["seen_mean_relative_delta"],
            name,
        ),
    )
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "objective_fields": base_fields,
        "excluded_fields": [
            field for field in fields if field not in base_fields
        ],
        "seen_taus": [int(tau) for tau in SEEN_TAUS],
        "held_taus": [int(tau) for tau in HELD_TAUS],
        "limits": {
            "base_mean_relative_delta": base_mean_limit,
            "seen_mean_relative_delta": seen_mean_limit,
            "held_mean_relative_delta": held_mean_limit,
            "per_tau_base_mean_relative_delta_max": per_tau_limit,
            "surface_mean_relative_delta": surface_mean_limit,
        },
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _sha256(reference_path),
        "ranking": ranking,
        "eligible": [
            name for name in ranking if results[name]["eligible"]
        ],
        "selected": next(
            (name for name in ranking if results[name]["eligible"]),
            None,
        ),
        "candidates": results,
        "note": (
            "This economy-day screen allocates one additional fine-tuning "
            "epoch only. Full-year evaluation is required for claims."
        ),
    }


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=METRICS_JSON")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        required=True,
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--base-mean-limit", type=float, required=True)
    parser.add_argument("--seen-mean-limit", type=float, required=True)
    parser.add_argument("--held-mean-limit", type=float, required=True)
    parser.add_argument("--per-tau-limit", type=float, required=True)
    parser.add_argument("--surface-mean-limit", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate names must be unique")
    report = build_selection(
        candidates,
        args.reference,
        base_mean_limit=args.base_mean_limit,
        seen_mean_limit=args.seen_mean_limit,
        held_mean_limit=args.held_mean_limit,
        per_tau_limit=args.per_tau_limit,
        surface_mean_limit=args.surface_mean_limit,
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
