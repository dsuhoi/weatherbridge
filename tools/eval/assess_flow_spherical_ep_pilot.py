#!/usr/bin/env python3
"""Apply the prespecified two-epoch gate to a spherical candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from tools.eval.summarize_region_season_generalization import load_windows
from tools.eval.summarize_matched_training_curves import build_report


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _polar_score(
    path: Path,
    expected_model: str,
    expected_checkpoint: Path,
) -> float:
    payload = json.loads(path.read_text())
    checkpoint_sha256 = _sha256_file(expected_checkpoint)
    if (
        payload.get("schema_version") != 6
        or payload.get("model_name") != expected_model
        or payload.get("test_year") != 2020
        or float(payload.get("delta_t_hours", -1)) != 6.0
        or sorted(payload.get("eval_hours", [])) != [2, 4]
        or payload.get("samples_per_date") != 2
        or payload.get("eval_days_per_month") != 8
        or payload.get("eval_day_picks")
        != [1, 5, 9, 13, 16, 20, 24, 28]
    ):
        raise ValueError(f"{path}: polar pilot protocol mismatch")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if (
        checkpoint.resolve() != expected_checkpoint.resolve()
        or payload.get("checkpoint_sha256")
        != checkpoint_sha256
    ):
        raise ValueError(f"{path}: polar pilot checkpoint mismatch")
    windows = load_windows(
        path,
        expected_model,
        {
            "models": {
                expected_model: {
                    "checkpoint_sha256": checkpoint_sha256,
                }
            }
        },
        2020,
    )
    polar_indices = [
        windows.regions.index(region)
        for region in ("Arctic", "Antarctic")
    ]
    polar_mse = windows.mse[:, polar_indices, :].mean(axis=(0, 1))
    if (
        polar_mse.shape != (len(windows.channels),)
        or not np.all(np.isfinite(polar_mse))
        or np.any(polar_mse < 0.0)
    ):
        raise ValueError(f"{path}: invalid polar paired metrics")
    return float(np.sqrt(polar_mse).mean())


def assess(
    pilot_csv: Path,
    reference_csv: Path,
    *,
    pilot_name: str = "flow_spherical_ep",
    reference_name: str = "upr_implicit_global_14m",
    held_relative_limit: float = 0.02,
    all_hour_relative_limit: float = 0.02,
    per_hour_relative_limit: float = 0.05,
    polar_pilot_json: Path | None = None,
    polar_reference_json: Path | None = None,
    polar_pilot_checkpoint: Path | None = None,
    polar_reference_checkpoint: Path | None = None,
    rescue_global_relative_limit: float = 0.05,
    rescue_per_hour_relative_limit: float = 0.10,
    rescue_polar_relative_limit: float = -0.02,
) -> dict:
    polar_inputs = (
        polar_pilot_json,
        polar_reference_json,
        polar_pilot_checkpoint,
        polar_reference_checkpoint,
    )
    if any(value is not None for value in polar_inputs) and not all(
        value is not None for value in polar_inputs
    ):
        raise ValueError(
            "polar JSONs and their exact checkpoints must be provided together"
        )
    comparison = build_report(
        pilot_csv,
        reference_csv,
        left_name=pilot_name,
        right_name=reference_name,
        expand_version_siblings=False,
    )
    epoch = comparison["matched_epochs"].get("1")
    if epoch is None:
        raise ValueError("both pilot and reference require two completed epochs")
    per_hour_max = max(epoch["per_hour_relative_delta"].values())
    checks = {
        "held_relative_delta": (
            epoch["held_relative_delta"] <= held_relative_limit
        ),
        "all_hour_relative_delta": (
            epoch["all_hour_relative_delta"] <= all_hour_relative_limit
        ),
        "per_hour_relative_delta_max": (
            per_hour_max <= per_hour_relative_limit
        ),
    }
    pilot_polar = None
    reference_polar = None
    polar_relative_delta = None
    geometry_rescue_inputs = None
    rescue_checks = {
        "polar_metrics_available": False,
        "held_relative_delta": False,
        "all_hour_relative_delta": False,
        "per_hour_relative_delta_max": False,
        "polar_relative_delta": False,
    }
    if polar_pilot_json is not None and polar_reference_json is not None:
        pilot_polar = _polar_score(
            polar_pilot_json,
            pilot_name,
            polar_pilot_checkpoint,
        )
        reference_polar = _polar_score(
            polar_reference_json,
            reference_name,
            polar_reference_checkpoint,
        )
        if reference_polar <= 0.0:
            raise ValueError("reference polar RMSE must be positive")
        polar_relative_delta = pilot_polar / reference_polar - 1.0
        geometry_rescue_inputs = {
            "pilot_metrics": _artifact_record(polar_pilot_json),
            "reference_metrics": _artifact_record(polar_reference_json),
            "pilot_checkpoint": _artifact_record(
                polar_pilot_checkpoint
            ),
            "reference_checkpoint": _artifact_record(
                polar_reference_checkpoint
            ),
        }
        rescue_checks = {
            "polar_metrics_available": True,
            "held_relative_delta": (
                epoch["held_relative_delta"]
                <= rescue_global_relative_limit
            ),
            "all_hour_relative_delta": (
                epoch["all_hour_relative_delta"]
                <= rescue_global_relative_limit
            ),
            "per_hour_relative_delta_max": (
                per_hour_max <= rescue_per_hour_relative_limit
            ),
            "polar_relative_delta": (
                polar_relative_delta <= rescue_polar_relative_limit
            ),
        }
    global_gate = all(checks.values())
    geometry_rescue = all(rescue_checks.values())
    return {
        "schema_version": 2,
        "diagnostic_only": True,
        "pilot_name": pilot_name,
        "reference_name": reference_name,
        "promoted": global_gate or geometry_rescue,
        "promotion_path": (
            "global_gate"
            if global_gate
            else "polar_geometry_rescue"
            if geometry_rescue
            else "excluded"
        ),
        "checks": checks,
        "geometry_rescue_checks": rescue_checks,
        "geometry_rescue_inputs": geometry_rescue_inputs,
        "limits": {
            "held_relative_delta": held_relative_limit,
            "all_hour_relative_delta": all_hour_relative_limit,
            "per_hour_relative_delta_max": per_hour_relative_limit,
            "rescue_global_relative_delta": rescue_global_relative_limit,
            "rescue_per_hour_relative_delta_max": (
                rescue_per_hour_relative_limit
            ),
            "rescue_polar_relative_delta": rescue_polar_relative_limit,
        },
        "observed": {
            "held_relative_delta": epoch["held_relative_delta"],
            "all_hour_relative_delta": epoch["all_hour_relative_delta"],
            "per_hour_relative_delta_max": per_hour_max,
            "pilot_polar_rmse": pilot_polar,
            "reference_polar_rmse": reference_polar,
            "polar_relative_delta": polar_relative_delta,
        },
        "comparison": comparison,
        "note": (
            "This gate only allocates further training compute. The polar "
            "rescue uses fixed 2020 Arctic/Antarctic windows and bounded "
            "global regression. Full-year "
            "2020 metrics select the model; 2021 remains temporal OOD."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-csv", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument(
        "--pilot-name",
        default="flow_spherical_ep",
    )
    parser.add_argument(
        "--reference-name",
        default="upr_implicit_global_14m",
    )
    parser.add_argument("--polar-pilot-json", type=Path)
    parser.add_argument("--polar-reference-json", type=Path)
    parser.add_argument("--polar-pilot-checkpoint", type=Path)
    parser.add_argument("--polar-reference-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = assess(
        args.pilot_csv,
        args.reference_csv,
        pilot_name=args.pilot_name,
        reference_name=args.reference_name,
        polar_pilot_json=args.polar_pilot_json,
        polar_reference_json=args.polar_reference_json,
        polar_pilot_checkpoint=args.polar_pilot_checkpoint,
        polar_reference_checkpoint=args.polar_reference_checkpoint,
    )
    report["generator_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"promoted": report["promoted"]}))


if __name__ == "__main__":
    main()
