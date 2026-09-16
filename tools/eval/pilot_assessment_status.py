#!/usr/bin/env python3
"""Recompute and validate a two-epoch pilot allocation decision."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from tools.eval.assess_flow_spherical_ep_pilot import (
    assess as assess_spherical,
)
from tools.eval.assess_two_epoch_candidate import assess as assess_generic


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_paths(
    records: Any,
    *,
    label: str,
    errors: list[str],
) -> tuple[Path, ...]:
    if not isinstance(records, list) or not records:
        errors.append(f"{label}: missing input records")
        return ()
    paths: list[Path] = []
    seen: set[Path] = set()
    for index, record in enumerate(records):
        item = f"{label}[{index}]"
        if not isinstance(record, dict):
            errors.append(f"{item}: record is not an object")
            continue
        raw_path = record.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"{item}: invalid path")
            continue
        path = Path(raw_path)
        try:
            resolved = path.resolve(strict=True)
        except OSError as error:
            errors.append(f"{item}: missing input: {error}")
            continue
        if resolved in seen:
            errors.append(f"{item}: duplicate input path")
            continue
        seen.add(resolved)
        stat = resolved.stat()
        try:
            expected_size = int(record.get("size_bytes"))
        except (TypeError, ValueError, OverflowError):
            errors.append(f"{item}: invalid size")
            continue
        if stat.st_size != expected_size:
            errors.append(f"{item}: size changed")
            continue
        expected_hash = record.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or _sha256(resolved) != expected_hash
        ):
            errors.append(f"{item}: content hash changed")
            continue
        paths.append(resolved)
    return tuple(paths)


def _geometry_path(
    records: dict[str, Any],
    name: str,
    errors: list[str],
) -> Path | None:
    paths = _validated_paths(
        [records.get(name)],
        label=f"geometry_rescue_inputs.{name}",
        errors=errors,
    )
    return paths[0] if len(paths) == 1 else None


def _compare_fields(
    payload: dict[str, Any],
    recomputed: dict[str, Any],
    fields: tuple[str, ...],
    errors: list[str],
) -> None:
    for field in fields:
        if payload.get(field) != recomputed.get(field):
            errors.append(f"{field}: stored decision differs from recomputation")


def validate_pilot_assessment(
    artifact_path: Path,
    *,
    expected_candidate: str,
    expected_reference: str,
    allow_geometry_rescue: bool = False,
    held_relative_limit: float = 0.02,
    all_hour_relative_limit: float = 0.02,
    per_hour_relative_limit: float = 0.05,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "promoted": False,
            "errors": [f"cannot read assessment: {error}"],
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "promoted": False,
            "errors": ["assessment root is not an object"],
        }

    spherical = "pilot_name" in payload
    candidate_key = "pilot_name" if spherical else "candidate"
    reference_key = "reference_name" if spherical else "reference"
    if payload.get(candidate_key) != expected_candidate:
        errors.append("candidate name mismatch")
    if payload.get(reference_key) != expected_reference:
        errors.append("reference name mismatch")

    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        errors.append("missing comparison")
        comparison = {}
    if comparison.get("left") != expected_candidate:
        errors.append("comparison candidate mismatch")
    if comparison.get("right") != expected_reference:
        errors.append("comparison reference mismatch")
    if comparison.get("hours") != [1, 2, 3, 4, 5]:
        errors.append("comparison hour set mismatch")
    if comparison.get("held_hours") != [2, 4]:
        errors.append("comparison held-hour set mismatch")
    if comparison.get("steps_per_epoch") != 1642:
        errors.append("comparison optimizer-step schedule mismatch")
    inputs = comparison.get("input_sha256")
    if not isinstance(inputs, dict):
        errors.append("missing comparison input hashes")
        inputs = {}
    candidate_paths = _validated_paths(
        inputs.get(expected_candidate),
        label=f"comparison.{expected_candidate}",
        errors=errors,
    )
    reference_paths = _validated_paths(
        inputs.get(expected_reference),
        label=f"comparison.{expected_reference}",
        errors=errors,
    )
    if errors:
        return {"valid": False, "promoted": False, "errors": errors}

    try:
        if spherical:
            geometry = payload.get("geometry_rescue_inputs")
            polar_kwargs: dict[str, Path] = {}
            checks = payload.get("checks")
            global_gate = (
                payload.get("promotion_path") == "global_gate"
                and isinstance(checks, dict)
                and bool(checks)
                and all(value is True for value in checks.values())
            )
            if geometry is not None:
                if not allow_geometry_rescue:
                    errors.append(
                        "geometry rescue is forbidden for this candidate"
                    )
                elif not isinstance(geometry, dict):
                    errors.append("invalid geometry rescue inputs")
                elif not global_gate:
                    mapping = {
                        "pilot_metrics": "polar_pilot_json",
                        "reference_metrics": "polar_reference_json",
                        "pilot_checkpoint": "polar_pilot_checkpoint",
                        "reference_checkpoint": (
                            "polar_reference_checkpoint"
                        ),
                    }
                    for record_name, argument_name in mapping.items():
                        path = _geometry_path(
                            geometry,
                            record_name,
                            errors,
                        )
                        if path is not None:
                            polar_kwargs[argument_name] = path
            if errors:
                return {
                    "valid": False,
                    "promoted": False,
                    "errors": errors,
                }
            recomputed = assess_spherical(
                candidate_paths,
                reference_paths,
                pilot_name=expected_candidate,
                reference_name=expected_reference,
                held_relative_limit=held_relative_limit,
                all_hour_relative_limit=all_hour_relative_limit,
                per_hour_relative_limit=per_hour_relative_limit,
                **polar_kwargs,
            )
            if global_gate:
                fields = (
                    "pilot_name",
                    "reference_name",
                    "promoted",
                    "promotion_path",
                    "checks",
                    "limits",
                    "comparison",
                )
                for field in (
                    "held_relative_delta",
                    "all_hour_relative_delta",
                    "per_hour_relative_delta_max",
                ):
                    if payload.get("observed", {}).get(field) != (
                        recomputed.get("observed", {}).get(field)
                    ):
                        errors.append(
                            f"observed.{field}: stored decision differs "
                            "from recomputation"
                        )
            else:
                fields = (
                    "pilot_name",
                    "reference_name",
                    "promoted",
                    "promotion_path",
                    "checks",
                    "geometry_rescue_checks",
                    "geometry_rescue_inputs",
                    "limits",
                    "observed",
                    "comparison",
                )
        else:
            recomputed = assess_generic(
                candidate_paths,
                reference_paths,
                candidate_name=expected_candidate,
                reference_name=expected_reference,
                held_relative_limit=held_relative_limit,
                all_hour_relative_limit=all_hour_relative_limit,
                per_hour_relative_limit=per_hour_relative_limit,
            )
            fields = (
                "candidate",
                "reference",
                "promoted",
                "checks",
                "limits",
                "observed",
                "comparison",
            )
    except (KeyError, OSError, TypeError, ValueError) as error:
        errors.append(f"cannot recompute assessment: {error}")
        return {"valid": False, "promoted": False, "errors": errors}

    _compare_fields(payload, recomputed, fields, errors)
    _validated_paths(
        inputs.get(expected_candidate),
        label=f"comparison.{expected_candidate}.postcheck",
        errors=errors,
    )
    _validated_paths(
        inputs.get(expected_reference),
        label=f"comparison.{expected_reference}.postcheck",
        errors=errors,
    )
    return {
        "valid": not errors,
        "promoted": bool(recomputed["promoted"]) if not errors else False,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--expected-candidate", required=True)
    parser.add_argument("--expected-reference", required=True)
    parser.add_argument("--allow-geometry-rescue", action="store_true")
    parser.add_argument("--held-relative-limit", type=float, default=0.02)
    parser.add_argument("--all-hour-relative-limit", type=float, default=0.02)
    parser.add_argument("--per-hour-relative-limit", type=float, default=0.05)
    parser.add_argument("--print-promoted", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = validate_pilot_assessment(
        args.artifact,
        expected_candidate=args.expected_candidate,
        expected_reference=args.expected_reference,
        allow_geometry_rescue=args.allow_geometry_rescue,
        held_relative_limit=args.held_relative_limit,
        all_hour_relative_limit=args.all_hour_relative_limit,
        per_hour_relative_limit=args.per_hour_relative_limit,
    )
    if not result["valid"]:
        if not args.quiet:
            print(json.dumps(result, indent=2), file=sys.stderr)
        raise SystemExit(1)
    if args.print_promoted:
        print(int(result["promoted"]))
    elif not args.quiet:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
