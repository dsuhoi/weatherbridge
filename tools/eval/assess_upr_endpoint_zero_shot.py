#!/usr/bin/env python3
"""Assess the state-identical smooth-endpoint UPR control on fixed 2020 data."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.eval.paired_block_bootstrap import (
    load_metrics,
    paired_block_bootstrap,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"{path}: missing evaluation protocol")
    if protocol.get("full_year") is not False:
        raise ValueError(f"{path}: zero-shot gate must be economy-only")
    if protocol.get("rmse_reduction") != (
        "spherical_strip_area_weighted_spatial_mean"
    ):
        raise ValueError(f"{path}: improper RMSE reduction")
    if set(int(value) for value in protocol.get("eval_hours", [])) != {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(f"{path}: incomplete 6h tau schedule")
    if payload.get("years") != [2020]:
        raise ValueError(f"{path}: gate must use 2020 only")
    checkpoint = payload.get("checkpoint_provenance")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"{path}: missing checkpoint provenance")
    checkpoint_path = Path(str(checkpoint.get("path", "")))
    recorded_sha = checkpoint.get("sha256")
    if not checkpoint_path.is_file() or not isinstance(recorded_sha, str):
        raise ValueError(f"{path}: invalid checkpoint provenance")
    if _sha256(checkpoint_path) != recorded_sha:
        raise ValueError(f"{path}: stale checkpoint provenance")
    return payload


def assess(
    reference_json: Path,
    candidate_json: Path,
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    reference = _validate_json(reference_json)
    candidate = _validate_json(candidate_json)
    reference_index = reference["evaluation_protocol"]["index_sha256"]
    candidate_index = candidate["evaluation_protocol"]["index_sha256"]
    if reference_index != candidate_index:
        raise ValueError("candidate/reference window indices differ")
    reference_checkpoint = reference.get("checkpoint_provenance")
    candidate_checkpoint = candidate.get("checkpoint_provenance")
    if not isinstance(reference_checkpoint, dict) or not isinstance(
        candidate_checkpoint,
        dict,
    ):
        raise ValueError("missing checkpoint provenance")
    candidate_checkpoint_path = Path(
        str(candidate_checkpoint.get("path", ""))
    )
    converted = torch.load(
        candidate_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    conversion = (
        converted.get("architecture_conversion")
        if isinstance(converted, dict)
        else None
    )
    if (
        not isinstance(conversion, dict)
        or conversion.get("kind")
        != "zero_shot_state_compatible_control"
        or conversion.get("source_arch") != "upr_implicit_global_14m"
        or conversion.get("target_arch")
        != "upr_endpoint_implicit_global_14m"
        or conversion.get("source_checkpoint_sha256")
        != reference_checkpoint.get("sha256")
        or conversion.get("weights_unchanged") is not True
    ):
        raise ValueError("candidate is not bound to the reference weights")

    reference_window = (
        reference_json.parent / reference["window_metrics_file"]
    )
    candidate_window = (
        candidate_json.parent / candidate["window_metrics_file"]
    )
    left = load_metrics(candidate_window)
    right = load_metrics(reference_window)
    groups = {
        "all": np.asarray([1, 2, 3, 4, 5], dtype=np.int16),
        "edge": np.asarray([1, 5], dtype=np.int16),
        "held": np.asarray([2, 4], dtype=np.int16),
    }
    comparisons = {
        group: paired_block_bootstrap(
            left,
            right,
            taus=taus,
            block_days=7,
            draws=draws,
            seed=seed,
        )
        for group, taus in groups.items()
    }
    per_hour_regression = max(
        comparison["relative_delta_pct"]
        for comparison in comparisons["all"]["per_tau"].values()
    )
    promote_retrain = (
        comparisons["edge"]["delta_left_minus_right"] < 0.0
        and comparisons["edge"]["p_paired_block_permutation"] < 0.10
        and comparisons["held"]["relative_delta_pct"] <= 1.0
        and comparisons["all"]["relative_delta_pct"] <= 0.5
        and per_hour_regression <= 1.0
    )
    return {
        "schema_version": 1,
        "evidence_level": "economy_2020_zero_shot_allocation_only",
        "reference": "upr_implicit_global_14m",
        "candidate": "upr_endpoint_implicit_global_14m",
        "state_identical_weights": True,
        "architecture_conversion": conversion,
        "reference_checkpoint_sha256": reference_checkpoint["sha256"],
        "candidate_checkpoint_sha256": candidate_checkpoint["sha256"],
        "window_index_sha256": reference_index,
        "reference_json_sha256": _sha256(reference_json),
        "candidate_json_sha256": _sha256(candidate_json),
        "comparisons": comparisons,
        "worst_per_hour_relative_delta_pct": per_hour_regression,
        "promotion_rule": {
            "edge_improves_p_lt": 0.10,
            "held_relative_delta_pct_max": 1.0,
            "all_relative_delta_pct_max": 0.5,
            "per_hour_relative_delta_pct_max": 1.0,
        },
        "promote_matched_retrain": promote_retrain,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=202707)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess(
        args.reference_json,
        args.candidate_json,
        draws=args.draws,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
