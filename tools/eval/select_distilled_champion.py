#!/usr/bin/env python3
"""Promote a specialist-distillation arm only when it is broadly better."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _strict_checks(deltas: dict[str, Any]) -> dict[str, bool]:
    required = (
        "aggregate_delta_pct",
        "moisture_delta_pct",
        "non_moisture_delta_pct",
        "held_delta_pct",
    )
    if not all(isinstance(deltas.get(key), (int, float)) for key in required):
        raise ValueError("missing strict champion deltas")
    return {
        "aggregate_improves": deltas["aggregate_delta_pct"] < 0.0,
        "moisture_improves": deltas["moisture_delta_pct"] < 0.0,
        "non_moisture_noninferior": deltas["non_moisture_delta_pct"] <= 0.0,
        "held_noninferior": deltas["held_delta_pct"] <= 0.0,
    }


def select_strict_champion(
    screen: dict[str, Any],
    matched_screen: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        screen.get("selection_role") != "era5_2020_validation_only"
        or not isinstance(screen.get("candidates"), dict)
    ):
        raise ValueError("expected a validation-only distillation screen")
    if matched_screen is not None and (
        matched_screen.get("selection_role") != "era5_2020_validation_only"
        or not isinstance(matched_screen.get("candidates"), dict)
    ):
        raise ValueError("expected a validation-only matched-control screen")
    matched_candidates = (
        matched_screen["candidates"] if matched_screen is not None else None
    )
    assessments: dict[str, dict[str, Any]] = {}
    for name, candidate in screen["candidates"].items():
        deltas = candidate.get("deltas_vs_control_pct", {})
        try:
            checks = _strict_checks(deltas)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error
        matched_assessment: dict[str, Any] | None = None
        matched_pass = True
        if matched_candidates is not None:
            matched_candidate = matched_candidates.get(name)
            if not isinstance(matched_candidate, dict):
                matched_assessment = {
                    "present": False,
                    "pass": False,
                    "strict_checks": {},
                }
                matched_pass = False
            else:
                matched_deltas = matched_candidate.get(
                    "deltas_vs_control_pct", {}
                )
                try:
                    matched_checks = _strict_checks(matched_deltas)
                except ValueError as error:
                    raise ValueError(f"{name} matched control: {error}") from error
                matched_pass = all(matched_checks.values())
                matched_assessment = {
                    "present": True,
                    "pass": matched_pass,
                    "strict_checks": matched_checks,
                    "deltas_vs_matched_control_pct": matched_deltas,
                }
        assessments[name] = {
            **candidate,
            "pass": all(checks.values()) and matched_pass,
            "strict_checks": checks,
        }
        if matched_assessment is not None:
            assessments[name]["matched_control_assessment"] = matched_assessment
    passing = [name for name, value in assessments.items() if value["pass"]]
    selected = min(
        passing,
        key=lambda name: (
            assessments[name]["deltas_vs_control_pct"]["aggregate_delta_pct"],
            assessments[name]["deltas_vs_control_pct"]["moisture_delta_pct"],
        ),
        default=None,
    )
    return {
        "schema_version": 1,
        "status": "pass" if selected is not None else "fail",
        "selected": selected,
        "selection_role": "era5_2020_validation_only",
        "selection_rule": {
            "name": "strict_distilled_champion_v1",
            "selection_year": 2020,
            "ood_year_excluded_from_selection": 2021,
            "aggregate_must_improve": True,
            "moisture_must_improve": True,
            "non_moisture_tolerance_pct": 0.0,
            "held_tau_tolerance_pct": 0.0,
            "matched_control_required": matched_screen is not None,
        },
        "control": screen.get("control"),
        "matched_control": (
            matched_screen.get("control") if matched_screen is not None else None
        ),
        "candidates": assessments,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--matched-screen", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matched_screen = (
        json.loads(args.matched_screen.read_text())
        if args.matched_screen is not None
        else None
    )
    payload = select_strict_champion(
        json.loads(args.screen.read_text()),
        matched_screen,
    )
    payload["screen_source"] = {
        "path": str(args.screen.resolve()),
        "sha256": hashlib.sha256(args.screen.read_bytes()).hexdigest(),
    }
    if args.matched_screen is not None:
        payload["matched_screen_source"] = {
            "path": str(args.matched_screen.resolve()),
            "sha256": hashlib.sha256(
                args.matched_screen.read_bytes()
            ).hexdigest(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": payload["status"], "selected": payload["selected"]}))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
