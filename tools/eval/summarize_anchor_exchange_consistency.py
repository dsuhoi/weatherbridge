#!/usr/bin/env python3
"""Summarize paired anchor-exchange diagnostics with weekly block tests."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.eval_anchor_exchange_consistency import (
    _index_sha256,
    _summary,
    anchor_exchange_evaluation_source_paths,
)
from tools.eval.paired_block_bootstrap import (
    WindowScores,
    paired_block_score_test,
)
from tools.eval.validate_upr_lite_selection import _annotate_holm
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


EXCHANGE_NONINFERIORITY_MARGIN = 0.02


DEFAULT_TAU_GROUPS = {
    "all": np.asarray([1, 2, 3, 4, 5], dtype=np.int16),
    "seen": np.asarray([1, 3, 5], dtype=np.int16),
    "unseen": np.asarray([2, 4], dtype=np.int16),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_summary(
    reported: Any,
    expected: dict[str, Any],
    *,
    label: str,
) -> None:
    if not isinstance(reported, dict) or set(reported) != set(expected):
        raise ValueError(f"{label}: summary structure mismatch")
    if reported.get("n_samples") != expected["n_samples"]:
        raise ValueError(f"{label}: summary sample count mismatch")
    for key in ("pooled_rmse", "channel_macro_rmse"):
        if not np.isclose(
            reported.get(key),
            expected[key],
            rtol=1e-7,
            atol=1e-10,
        ):
            raise ValueError(f"{label}: {key} mismatch")
    values = np.asarray(reported.get("per_channel_rmse"))
    expected_values = np.asarray(expected["per_channel_rmse"])
    if (
        values.shape != expected_values.shape
        or not np.allclose(
            values,
            expected_values,
            rtol=1e-7,
            atol=1e-10,
        )
    ):
        raise ValueError(f"{label}: per-channel summary mismatch")


def load_scores(
    artifact_path: Path,
    selection: dict[str, Any],
) -> tuple[dict[str, WindowScores], dict[str, Any]]:
    artifact = json.loads(artifact_path.read_text())
    recorded_code = artifact.get("evaluation_code_provenance")
    current_sources = anchor_exchange_evaluation_source_paths()
    if (
        not isinstance(recorded_code, dict)
        or set(recorded_code) != set(current_sources)
    ):
        raise ValueError(
            f"{artifact_path}: evaluation source manifest mismatch"
        )
    for name, source_path in current_sources.items():
        if recorded_code.get(name) != _sha256(source_path):
            raise ValueError(
                f"{artifact_path}: evaluation source hash mismatch: {name}"
            )
    delta_t_hours = artifact.get("delta_t_hours")
    test_year = artifact.get("test_year")
    eval_hours = artifact.get("eval_hours")
    evaluation_protocol = artifact.get("evaluation_protocol")
    dataset_provenance = artifact.get("evaluation_dataset_provenance")
    input_provenance = artifact.get("evaluation_input_provenance")
    if (
        artifact.get("schema_version") != 2
        or artifact.get("diagnostic_only") is not True
        or isinstance(delta_t_hours, bool)
        or delta_t_hours not in {6, 12}
        or eval_hours != list(range(1, delta_t_hours))
        or isinstance(test_year, bool)
        or test_year not in {2020, 2021}
        or artifact.get("channel_names")
        != list(CANONICAL_24_CHANNELS)
        or not isinstance(evaluation_protocol, dict)
        or evaluation_protocol.get("samples_per_date") != 2
        or evaluation_protocol.get("eval_days_per_month") != 2
        or not isinstance(
            evaluation_protocol.get("window_index_sha256"),
            str,
        )
        or len(evaluation_protocol["window_index_sha256"]) != 64
        or not isinstance(
            evaluation_protocol.get("endpoint_index_sha256"),
            str,
        )
        or len(evaluation_protocol["endpoint_index_sha256"]) != 64
        or not isinstance(dataset_provenance, dict)
        or dataset_provenance.get("years") != [test_year]
        or not isinstance(input_provenance, dict)
        or set(input_provenance)
        != {
            "static_features",
            "pressure_level_stats",
            "surface_stats",
        }
    ):
        raise ValueError(f"{artifact_path}: evaluation protocol mismatch")
    selected_models = selection.get("models")
    artifact_models = artifact.get("models")
    if (
        not isinstance(selected_models, dict)
        or not isinstance(artifact_models, dict)
        or set(selected_models) != set(artifact_models)
    ):
        raise ValueError(f"{artifact_path}: model set mismatch")
    npz_path = artifact_path.parent / artifact["paired_windows_file"]
    if (
        not npz_path.is_file()
        or npz_path.stat().st_size
        != artifact.get("paired_windows_size_bytes")
        or _sha256(npz_path) != artifact.get("paired_windows_sha256")
    ):
        raise ValueError(f"{artifact_path}: paired NPZ hash mismatch")

    scores: dict[str, WindowScores] = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        expected_array_names = {
            "window_year",
            "window_t0",
            "window_tau",
            "endpoint_year",
            "endpoint_t0",
            "channel_names",
        }
        for model in selected_models:
            expected_array_names.update(
                {
                    f"{model}_exchange_mse",
                    f"{model}_forward_mse",
                    f"{model}_endpoint0_mse",
                    f"{model}_endpoint1_mse",
                }
            )
        if set(arrays.files) != expected_array_names:
            raise ValueError(
                f"{artifact_path}: paired NPZ array manifest mismatch"
            )
        year = np.asarray(arrays["window_year"])
        t0 = np.asarray(arrays["window_t0"])
        tau = np.asarray(arrays["window_tau"])
        endpoint_year = np.asarray(arrays["endpoint_year"])
        endpoint_t0 = np.asarray(arrays["endpoint_t0"])
        channels = tuple(str(value) for value in arrays["channel_names"])
        if (
            channels != CANONICAL_24_CHANNELS
            or set(int(value) for value in year) != {test_year}
            or set(int(value) for value in tau) != set(eval_hours)
            or set(int(value) for value in endpoint_year) != {test_year}
            or t0.shape != year.shape
            or tau.shape != year.shape
            or endpoint_t0.shape != endpoint_year.shape
            or len(set(zip(year, t0, tau))) != year.size
            or len(set(zip(endpoint_year, endpoint_t0)))
            != endpoint_year.size
            or evaluation_protocol.get("n_windows") != year.size
            or evaluation_protocol.get("n_endpoints")
            != endpoint_year.size
            or evaluation_protocol.get("window_index_sha256")
            != _index_sha256(year, t0, tau)
            or evaluation_protocol.get("endpoint_index_sha256")
            != _index_sha256(endpoint_year, endpoint_t0)
        ):
            raise ValueError(
                f"{artifact_path}: paired protocol coverage mismatch"
            )
        for model, selected in selected_models.items():
            provenance = artifact_models[model].get(
                "checkpoint_provenance"
            )
            if (
                not isinstance(provenance, dict)
                or provenance.get("sha256")
                != selected.get("checkpoint_sha256")
            ):
                raise ValueError(
                    f"{artifact_path}: {model} checkpoint hash mismatch"
                )
            exchange_raw = np.asarray(arrays[f"{model}_exchange_mse"])
            forward_raw = np.asarray(arrays[f"{model}_forward_mse"])
            endpoint0_raw = np.asarray(arrays[f"{model}_endpoint0_mse"])
            endpoint1_raw = np.asarray(arrays[f"{model}_endpoint1_mse"])
            if (
                exchange_raw.shape != (year.size, len(channels))
                or forward_raw.shape != exchange_raw.shape
                or endpoint0_raw.shape
                != (endpoint_year.size, len(channels))
                or endpoint1_raw.shape != endpoint0_raw.shape
                or any(
                    not np.all(np.isfinite(values))
                    or np.any(values < 0.0)
                    for values in (
                        exchange_raw,
                        forward_raw,
                        endpoint0_raw,
                        endpoint1_raw,
                    )
                )
            ):
                raise ValueError(
                    f"{artifact_path}: invalid paired arrays for {model}"
                )
            reported = artifact_models[model]
            expected_summaries = {
                "exchange": _summary(exchange_raw),
                "forward_error": _summary(forward_raw),
                "endpoint0": _summary(endpoint0_raw),
                "endpoint1": _summary(endpoint1_raw),
            }
            for label, expected_summary in expected_summaries.items():
                _validate_summary(
                    reported.get(label),
                    expected_summary,
                    label=f"{artifact_path}: {model} {label}",
                )
            expected_ratio = (
                expected_summaries["exchange"]["pooled_rmse"]
                / max(
                    expected_summaries["forward_error"]["pooled_rmse"],
                    1e-12,
                )
            )
            if not np.isclose(
                reported.get("exchange_to_forward_rmse_ratio"),
                expected_ratio,
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError(
                    f"{artifact_path}: {model} exchange ratio mismatch"
                )
            values = np.asarray(exchange_raw, dtype=np.float64)
            scores[model] = WindowScores(
                year=year,
                t0=t0,
                tau=tau,
                values=values,
                channels=channels,
            )
    return scores, artifact


def compare(
    selection: dict[str, Any],
    artifacts: dict[str, Path],
    *,
    block_days: int,
    draws: int,
    seed: int,
    tau_groups: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    selection_models = selection.get("models")
    if (
        isinstance(selection.get("schema_version"), bool)
        or not isinstance(selection.get("schema_version"), int)
        or selection["schema_version"] < 12
        or selection.get("ood_attached_at_selection_time") is not False
        or not isinstance(selection_models, dict)
        or selection.get("winner") not in selection_models
    ):
        raise ValueError("invalid frozen selection")
    if set(artifacts) != {"2020", "2021"}:
        raise ValueError(
            "anchor-exchange comparison requires 2020 and 2021 artifacts"
        )
    tau_groups = tau_groups or DEFAULT_TAU_GROUPS
    required_groups = {"all", "seen", "unseen"}
    if set(tau_groups) != required_groups:
        raise ValueError(f"tau_groups must contain exactly {required_groups}")
    normalized_groups = {
        name: np.asarray(values, dtype=np.int16)
        for name, values in tau_groups.items()
    }
    all_taus = set(int(value) for value in normalized_groups["all"])
    seen_taus = set(int(value) for value in normalized_groups["seen"])
    unseen_taus = set(int(value) for value in normalized_groups["unseen"])
    if (
        not all_taus
        or not seen_taus
        or not unseen_taus
        or seen_taus & unseen_taus
        or seen_taus | unseen_taus != all_taus
    ):
        raise ValueError(
            "seen and unseen taus must be a non-empty partition of all taus"
        )
    winner = str(selection["winner"])
    result: dict[str, Any] = {}
    source_artifacts: dict[str, Any] = {}
    validated_artifacts: dict[str, dict[str, Any]] = {}
    shared_protocol: tuple[int, tuple[int, ...]] | None = None
    input_provenance_sha256: str | None = None
    code_provenance_sha256: str | None = None
    dataset_provenance_sha256: dict[str, str] = {}
    for year, path in artifacts.items():
        scores, artifact = load_scores(path, selection)
        validated_artifacts[year] = artifact
        if artifact["test_year"] != int(year):
            raise ValueError(f"{path}: test year/path mismatch")
        artifact_protocol = (
            int(artifact.get("delta_t_hours", -1)),
            tuple(int(value) for value in artifact.get("eval_hours", [])),
        )
        if (
            artifact_protocol[0] < 2
            or set(artifact_protocol[1]) != all_taus
        ):
            raise ValueError(f"{path}: artifact tau protocol mismatch")
        if shared_protocol is None:
            shared_protocol = artifact_protocol
        elif artifact_protocol != shared_protocol:
            raise ValueError("anchor-exchange protocol differs across years")
        current_input_sha256 = _canonical_sha256(
            artifact["evaluation_input_provenance"]
        )
        if input_provenance_sha256 is None:
            input_provenance_sha256 = current_input_sha256
        elif current_input_sha256 != input_provenance_sha256:
            raise ValueError(
                "anchor-exchange input provenance differs across years"
            )
        current_code_sha256 = _canonical_sha256(
            artifact["evaluation_code_provenance"]
        )
        if code_provenance_sha256 is None:
            code_provenance_sha256 = current_code_sha256
        elif current_code_sha256 != code_provenance_sha256:
            raise ValueError(
                "anchor-exchange code provenance differs across years"
            )
        dataset_provenance_sha256[year] = _canonical_sha256(
            artifact["evaluation_dataset_provenance"]
        )
        source_artifacts[year] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "paired_windows_sha256": artifact["paired_windows_sha256"],
        }
        year_result: dict[str, Any] = {}
        for group, taus in normalized_groups.items():
            comparisons = {
                model: paired_block_score_test(
                    scores[winner],
                    model_scores,
                    taus=taus,
                    block_days=block_days,
                    draws=draws,
                    seed=seed,
                    better="lower",
                )
                for model, model_scores in scores.items()
                if model != winner
            }
            _annotate_holm(
                list(comparisons.values()),
                output_key="p_holm_exchange_family",
            )
            for comparison in comparisons.values():
                margin = (
                    EXCHANGE_NONINFERIORITY_MARGIN
                    * abs(float(comparison["right"]))
                )
                comparison["winner_noninferiority_margin"] = margin
                comparison["winner_noninferior"] = bool(
                    float(comparison["delta_ci95"][2]) <= margin
                    and float(comparison["left"])
                    <= (
                        (1.0 + EXCHANGE_NONINFERIORITY_MARGIN)
                        * float(comparison["right"])
                    )
                )
            year_result[group] = comparisons
        result[year] = year_result

    ood = result.get("2021", {}).get("unseen", {})
    significant_regressions = [
        model
        for model, comparison in ood.items()
        if comparison["delta_left_minus_right"] > 0.0
        and comparison["p_holm_exchange_family"] < 0.05
    ]
    significant_dominance = [
        model
        for model, comparison in ood.items()
        if comparison["delta_left_minus_right"] < 0.0
        and comparison["p_holm_exchange_family"] < 0.05
    ]
    noninferiority_failures = [
        model
        for model, comparison in ood.items()
        if not comparison["winner_noninferior"]
    ]
    winner_endpoints = {
        year: {
            endpoint: source["models"][winner][endpoint][
                "pooled_rmse"
            ]
            for endpoint in ("endpoint0", "endpoint1")
        }
        for year, source in {
            key: artifact
            for key, artifact in validated_artifacts.items()
        }.items()
    }
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "winner": winner,
        "source_artifacts": source_artifacts,
        "protocol": {
            "delta_t_hours": shared_protocol[0],
            "all_taus": sorted(all_taus),
            "seen_taus": sorted(seen_taus),
            "unseen_taus": sorted(unseen_taus),
            "channels": list(CANONICAL_24_CHANNELS),
            "samples_per_date": 2,
            "eval_days_per_month": 2,
            "block_days": block_days,
            "draws": draws,
            "seed": seed,
            "noninferiority_margin": (
                EXCHANGE_NONINFERIORITY_MARGIN
            ),
        },
        "evaluation_input_provenance_sha256": input_provenance_sha256,
        "evaluation_code_provenance_sha256": code_provenance_sha256,
        "evaluation_dataset_provenance_sha256": (
            dataset_provenance_sha256
        ),
        "comparisons": result,
        "winner_endpoint_rmse": winner_endpoints,
        "no_significant_2021_unseen_exchange_regression": (
            not significant_regressions
        ),
        "significant_2021_unseen_exchange_regressions": (
            significant_regressions
        ),
        "significant_2021_unseen_exchange_dominance": (
            significant_dominance
        ),
        "winner_2021_unseen_exchange_noninferior_2pct": (
            not noninferiority_failures
        ),
        "winner_2021_unseen_exchange_noninferiority_failures": (
            noninferiority_failures
        ),
        "note": (
            "Exchange equivariance is a post-selection diagnostic and does "
            "not change the frozen 2020 winner."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--artifact-2020", type=Path, required=True)
    parser.add_argument("--artifact-2021", type=Path, required=True)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--all-taus", default="1,2,3,4,5")
    parser.add_argument("--seen-taus", default="1,3,5")
    parser.add_argument("--unseen-taus", default="2,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    tau_groups = {
        name: np.asarray(
            [int(value) for value in raw.split(",") if value],
            dtype=np.int16,
        )
        for name, raw in (
            ("all", args.all_taus),
            ("seen", args.seen_taus),
            ("unseen", args.unseen_taus),
        )
    }
    report = compare(
        selection,
        {
            "2020": args.artifact_2020,
            "2021": args.artifact_2021,
        },
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
        tau_groups=tau_groups,
    )
    report["selection_manifest"] = {
        "path": str(args.selection.resolve()),
        "sha256": _sha256(args.selection),
    }
    report["generator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "no_ood_regression": report[
                    "no_significant_2021_unseen_exchange_regression"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
