#!/usr/bin/env python3
"""Require a candidate to beat the reference envelope in every metric cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TAUS = ("1", "2", "3", "4", "5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("candidate must be NAME=METRICS_JSON")
    return name, Path(raw_path)


def _reference(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if separator:
        if not name or not raw_path:
            raise argparse.ArgumentTypeError(
                "reference must be NAME=METRICS_JSON or METRICS_JSON"
            )
        return name, Path(raw_path)
    path = Path(value)
    return path.stem, path


def _rmse_protocol_signature(payload: dict[str, Any]) -> dict[str, Any]:
    protocol = payload["evaluation_protocol"]
    return {
        "years": payload["years"],
        "n_per_tau": payload["n_per_tau"],
        "rmse_reduction": protocol.get("rmse_reduction"),
        "samples_per_date": protocol.get("samples_per_date"),
        "eval_days_per_month": protocol.get("eval_days_per_month"),
        "full_year": protocol.get("full_year"),
        "eval_hours": protocol.get("eval_hours"),
        "index_sha256": protocol.get("index_sha256"),
    }


def build_selection(
    candidates: dict[str, Path],
    reference_path: Path,
    *,
    cell_limit: float,
    include_q: bool = False,
    additional_references: dict[str, Path] | None = None,
) -> dict[str, Any]:
    reference = json.loads(reference_path.read_text())
    reference_paths = {"primary": reference_path}
    if additional_references:
        overlap = set(reference_paths).intersection(additional_references)
        if overlap:
            raise ValueError(
                f"duplicate reference names: {sorted(overlap)}"
            )
        reference_paths.update(additional_references)
    references = {"primary": reference}
    fields = list(reference["channel_names"])
    base_fields = [field for field in fields if not field.startswith("Q")]
    if len(fields) != 24 or len(base_fields) != 20:
        raise ValueError("expected 24 fields with four Q pressure levels")
    objective_fields = fields if include_q else base_fields
    reference_protocol = _rmse_protocol_signature(reference)
    for name, path in reference_paths.items():
        if name == "primary":
            continue
        payload = json.loads(path.read_text())
        if _rmse_protocol_signature(payload) != reference_protocol:
            raise ValueError(
                f"reference {name}: evaluation protocol mismatch"
            )
        if payload["channel_names"] != fields:
            raise ValueError(f"reference {name}: channel order mismatch")
        references[name] = payload

    envelope: dict[str, dict[str, float]] = {}
    envelope_source: dict[str, dict[str, str]] = {}
    for tau in TAUS:
        envelope[tau] = {}
        envelope_source[tau] = {}
        for field in objective_fields:
            key = f"rmse_norm_{field}"
            value, source = min(
                (
                    payload["per_tau"][tau]["model"][key],
                    name,
                )
                for name, payload in references.items()
            )
            envelope[tau][field] = value
            envelope_source[tau][field] = source

    results: dict[str, dict[str, Any]] = {}
    for name, path in candidates.items():
        candidate = json.loads(path.read_text())
        if _rmse_protocol_signature(candidate) != reference_protocol:
            raise ValueError(f"{name}: evaluation protocol mismatch")
        if candidate["channel_names"] != fields:
            raise ValueError(f"{name}: channel order mismatch")

        cells: dict[str, dict[str, float]] = {}
        deltas: list[float] = []
        for tau in TAUS:
            candidate_metrics = candidate["per_tau"][tau]["model"]
            cells[tau] = {}
            for field in objective_fields:
                key = f"rmse_norm_{field}"
                delta = (
                    candidate_metrics[key] / envelope[tau][field] - 1.0
                )
                cells[tau][field] = delta
                deltas.append(delta)

        failing = [
            {
                "tau_hour": int(tau),
                "field": field,
                "relative_delta": cells[tau][field],
            }
            for tau in TAUS
            for field in objective_fields
            if cells[tau][field] >= cell_limit
        ]
        failing.sort(
            key=lambda item: item["relative_delta"],
            reverse=True,
        )
        per_tau_worst = {
            tau: max(cells[tau].values())
            for tau in TAUS
        }
        per_field_worst = {
            field: max(cells[tau][field] for tau in TAUS)
            for field in objective_fields
        }
        results[name] = {
            "eligible": not failing,
            "strict_cell_limit": cell_limit,
            "mean_relative_delta": sum(deltas) / len(deltas),
            "worst_relative_delta": max(deltas),
            "best_relative_delta": min(deltas),
            "winning_cells": sum(delta < cell_limit for delta in deltas),
            "total_cells": len(deltas),
            "per_tau_worst_relative_delta": per_tau_worst,
            "per_field_worst_relative_delta": per_field_worst,
            "cell_relative_delta": cells,
            "failing_cells": failing,
            "metrics_path": str(path.resolve()),
            "metrics_sha256": _sha256(path),
        }

    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["worst_relative_delta"],
            results[name]["mean_relative_delta"],
            name,
        ),
    )
    eligible = [name for name in ranking if results[name]["eligible"]]
    return {
        "schema_version": 1,
        "criterion": (
            f"Every one of the {len(objective_fields)} objective fields must "
            "have lower normalized RMSE than the best supplied reference "
            "at each hour 1-5."
        ),
        "strict_cell_limit": cell_limit,
        "include_q": include_q,
        "objective_fields": objective_fields,
        "taus": [int(tau) for tau in TAUS],
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": _sha256(reference_path),
        "references": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for name, path in reference_paths.items()
        },
        "reference_envelope_source": envelope_source,
        "ranking": ranking,
        "eligible": eligible,
        "selected": eligible[0] if eligible else None,
        "candidates": results,
        "note": (
            "This RMSE gate is necessary but not sufficient for final "
            "selection. A promoted candidate still requires full-year, "
            "untouched-year, ACC, spectral, and compute checks."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_candidate,
        required=True,
    )
    parser.add_argument(
        "--reference",
        action="append",
        type=_reference,
        required=True,
        help=(
            "Reference artifact as NAME=METRICS_JSON. Repeat to require "
            "dominance over the per-cell best baseline."
        ),
    )
    parser.add_argument("--cell-limit", type=float, default=0.0)
    parser.add_argument("--include-q", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = dict(args.candidate)
    if len(candidates) != len(args.candidate):
        parser.error("candidate names must be unique")
    references = dict(args.reference)
    if len(references) != len(args.reference):
        parser.error("reference names must be unique")
    primary_name = next(iter(references))
    reference_path = references.pop(primary_name)
    report = build_selection(
        candidates,
        reference_path,
        cell_limit=args.cell_limit,
        include_q=args.include_q,
        additional_references=references,
    )
    report["primary_reference_name"] = primary_name
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
                "winning_cells": {
                    name: report["candidates"][name]["winning_cells"]
                    for name in report["ranking"]
                },
            }
        )
    )


if __name__ == "__main__":
    main()
