#!/usr/bin/env python3
"""Separate architecture gains from the local high-pass auxiliary loss."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.paired_block_bootstrap import (
    load_acc_scores,
    load_metrics,
    paired_block_bootstrap,
    paired_block_score_test,
)
from tools.eval.select_upr_lite_candidate import (
    load_field_metrics,
    load_spectral_metrics,
)
from tools.eval.spectral_block_bootstrap import (
    compare as compare_spectra,
)
from tools.eval.spectral_block_bootstrap import load_windows
from tools.eval.summarize_upr_lite_seeds import choose_transfer_candidate
from tools.eval.validate_upr_lite_selection import _annotate_holm


COMPARISON_MODELS = {
    "architecture_only": ("nohf", "reference"),
    "auxiliary_effect": ("default", "nohf"),
    "full_method": ("default", "reference"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _field_artifact(root: Path, model: str) -> tuple[dict, Path]:
    metrics_path = root / f"{model}.json"
    payload = json.loads(metrics_path.read_text())
    window_file = payload.get("window_metrics_file")
    if not isinstance(window_file, str) or not window_file:
        raise ValueError(f"{metrics_path}: missing window metrics")
    window_path = root / window_file
    if not window_path.is_file():
        raise FileNotFoundError(window_path)
    return load_field_metrics(metrics_path), window_path


def _spectral_path(root: Path, model: str, tau: int) -> Path:
    path = root / f"{model}_tau{tau}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _annotate_family(
    tests: list[tuple[str, str, dict, str]],
) -> list[dict[str, Any]]:
    _annotate_holm(
        [payload for _, _, payload, _ in tests],
        output_key="p_holm_ablation_family",
    )
    output = []
    for scope, metric, payload, better in tests:
        delta = float(payload["delta_left_minus_right"])
        p_holm = float(payload["p_holm_ablation_family"])
        improvement = delta < 0.0 if better == "lower" else delta > 0.0
        regression = delta > 0.0 if better == "lower" else delta < 0.0
        output.append(
            {
                "scope": scope,
                "metric": metric,
                "better": better,
                "delta_left_minus_right": delta,
                "p_raw": float(
                    payload["p_paired_block_permutation"]
                ),
                "p_holm": p_holm,
                "significant_improvement": improvement and p_holm < 0.05,
                "significant_regression": regression and p_holm < 0.05,
            }
        )
    return output


def build_report(
    selection: dict[str, Any],
    *,
    primary_root_2020: Path,
    primary_root_2021: Path,
    nohf_root_2020: Path,
    nohf_root_2021: Path,
    primary_spectra_root: Path,
    nohf_spectra_root: Path,
    unseen_taus: tuple[int, ...] = (2, 4),
    block_days: int = 7,
    draws: int = 5000,
    seed: int = 2027,
    hf_ell_min: int = 180,
) -> dict[str, Any]:
    if selection.get("ood_attached_at_selection_time") is not False:
        raise ValueError("high-pass ablation requires a frozen 2020 selection")
    candidate = choose_transfer_candidate(selection, "quality")
    reference = "weatherbridge_ref"
    nohf = f"{candidate}_nohf"
    selection_models = selection.get("models")
    if (
        not isinstance(selection_models, dict)
        or candidate not in selection_models
        or reference not in selection_models
    ):
        raise ValueError("selection lacks candidate or PP3 reference")

    field_roots = {
        "2020": {
            "default": (primary_root_2020, candidate),
            "nohf": (nohf_root_2020, nohf),
            "reference": (primary_root_2020, reference),
        },
        "2021": {
            "default": (primary_root_2021, candidate),
            "nohf": (nohf_root_2021, nohf),
            "reference": (primary_root_2021, reference),
        },
    }
    fields: dict[str, dict[str, dict[str, Any]]] = {}
    field_artifacts: dict[str, dict[str, Any]] = {}
    checkpoint_hashes: dict[str, set[str]] = {
        role: set() for role in ("default", "nohf", "reference")
    }
    field_input_provenance: dict[str, dict[str, Any]] = {}
    field_dataset_provenance: dict[str, dict[str, Any]] = {}
    for year, role_specs in field_roots.items():
        fields[year] = {}
        field_artifacts[year] = {}
        indices: set[str] = set()
        input_hashes: set[str] = set()
        dataset_hashes: set[str] = set()
        for role, (root, model) in role_specs.items():
            metrics, window_path = _field_artifact(root, model)
            if not metrics["evaluation_full_year"]:
                raise ValueError(f"{model}: {year} metrics are not full-year")
            fields[year][role] = {
                "metrics": metrics,
                "window_path": window_path,
            }
            if metrics["evaluation_dataset_provenance"].get("years") != [
                int(year)
            ]:
                raise ValueError(f"{model}: wrong field dataset year")
            checkpoint_hashes[role].add(metrics["checkpoint_sha256"])
            indices.add(str(metrics["window_index_sha256"]))
            input_hashes.add(
                _canonical_sha256(metrics["evaluation_input_provenance"])
            )
            dataset_hashes.add(
                _canonical_sha256(metrics["evaluation_dataset_provenance"])
            )
            field_artifacts[year][role] = {
                "metrics": _artifact(root / f"{model}.json"),
                "windows": _artifact(window_path),
            }
        if len(indices) != 1:
            raise ValueError(f"{year} field window-index mismatch")
        if len(input_hashes) != 1 or len(dataset_hashes) != 1:
            raise ValueError(f"{year} field provenance mismatch")
        field_input_provenance[year] = fields[year]["default"][
            "metrics"
        ]["evaluation_input_provenance"]
        field_dataset_provenance[year] = fields[year]["default"][
            "metrics"
        ]["evaluation_dataset_provenance"]

    if field_input_provenance["2020"] != field_input_provenance["2021"]:
        raise ValueError("field input provenance changed across years")

    if any(len(values) != 1 for values in checkpoint_hashes.values()):
        raise ValueError("field checkpoint changed between 2020 and 2021")
    if (
        next(iter(checkpoint_hashes["default"]))
        != selection_models[candidate]["checkpoint_sha256"]
        or next(iter(checkpoint_hashes["reference"]))
        != selection_models[reference]["checkpoint_sha256"]
    ):
        raise ValueError("field checkpoint differs from frozen selection")

    taus = np.asarray(unseen_taus, dtype=np.int16)
    comparisons: dict[str, Any] = {"field": {}, "spectral": {}}
    family_tests: dict[str, list[tuple[str, str, dict, str]]] = {
        name: [] for name in COMPARISON_MODELS
    }
    for year in ("2020", "2021"):
        comparisons["field"][year] = {}
        for comparison, (left_role, right_role) in COMPARISON_MODELS.items():
            left_path = fields[year][left_role]["window_path"]
            right_path = fields[year][right_role]["window_path"]
            rmse = paired_block_bootstrap(
                load_metrics(left_path),
                load_metrics(right_path),
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            acc = paired_block_score_test(
                load_acc_scores(left_path),
                load_acc_scores(right_path),
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
                better="higher",
            )
            comparisons["field"][year][comparison] = {
                "rmse": rmse,
                "acc": acc,
            }
            family_tests[comparison].extend(
                (
                    (year, "rmse", rmse, "lower"),
                    (year, "acc", acc, "higher"),
                )
            )

    spectral_specs = {
        "default": (primary_spectra_root, candidate),
        "nohf": (nohf_spectra_root, nohf),
        "reference": (primary_spectra_root, reference),
    }
    spectral_artifacts: dict[str, dict[str, Any]] = {}
    spectral_indices: set[str] = set()
    spectral_inputs: set[str] = set()
    spectral_datasets: set[str] = set()
    spectral_input_reference: dict[str, Any] | None = None
    spectral_dataset_reference: dict[str, Any] | None = None
    for role, (root, model) in spectral_specs.items():
        metrics = load_spectral_metrics(
            root,
            model,
            hf_ell_min,
            taus=set(unseen_taus),
            required_lmax=359,
        )
        if metrics["checkpoint_sha256"] not in checkpoint_hashes[role]:
            raise ValueError(f"{role}: field/spectral checkpoint mismatch")
        spectral_indices.add(str(metrics["window_index_sha256"]))
        spectral_inputs.add(
            _canonical_sha256(metrics["evaluation_input_provenance"])
        )
        spectral_datasets.add(
            _canonical_sha256(metrics["evaluation_dataset_provenance"])
        )
        if spectral_input_reference is None:
            spectral_input_reference = metrics[
                "evaluation_input_provenance"
            ]
            spectral_dataset_reference = metrics[
                "evaluation_dataset_provenance"
            ]
        spectral_artifacts[role] = {
            str(tau): _artifact(_spectral_path(root, model, tau))
            for tau in unseen_taus
        }
    if len(spectral_indices) != 1:
        raise ValueError("spectral window-index mismatch")
    if len(spectral_inputs) != 1 or len(spectral_datasets) != 1:
        raise ValueError("spectral provenance mismatch")
    assert spectral_input_reference is not None
    assert spectral_dataset_reference is not None
    if spectral_dataset_reference != field_dataset_provenance["2020"]:
        raise ValueError("field/spectral 2020 dataset provenance mismatch")
    for key in (
        "static_features",
        "pressure_level_stats",
        "surface_stats",
    ):
        if (
            spectral_input_reference.get(key)
            != field_input_provenance["2020"].get(key)
        ):
            raise ValueError(
                f"field/spectral input provenance mismatch for {key}"
            )

    for tau in unseen_taus:
        comparisons["spectral"][str(tau)] = {}
        windows = {
            role: load_windows(_spectral_path(root, model, tau))
            for role, (root, model) in spectral_specs.items()
        }
        channel_indices = np.arange(len(windows["default"].channels))
        for comparison, (left_role, right_role) in COMPARISON_MODELS.items():
            result = compare_spectra(
                windows[left_role],
                windows[right_role],
                channel_indices=channel_indices,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            comparisons["spectral"][str(tau)][comparison] = result
            family_tests[comparison].extend(
                (
                    (
                        f"tau{tau}",
                        "spectral_energy",
                        result["energy_log_error"],
                        "lower",
                    ),
                    (
                        f"tau{tau}",
                        "spectral_shape",
                        result["shape_log_error"],
                        "lower",
                    ),
                    (
                        f"tau{tau}",
                        "spectral_coherence",
                        result["coherence"],
                        "higher",
                    ),
                )
            )

    significance = {
        name: _annotate_family(tests)
        for name, tests in family_tests.items()
    }
    architecture_tests = significance["architecture_only"]
    architecture_rmse = [
        result
        for result in architecture_tests
        if result["metric"] == "rmse"
    ]
    architecture_confirmed = (
        len(architecture_rmse) == 2
        and all(
            result["significant_improvement"]
            for result in architecture_rmse
        )
        and not any(
            result["significant_regression"]
            for result in architecture_tests
        )
    )
    auxiliary_tests = significance["auxiliary_effect"]
    auxiliary_supported = (
        any(
            result["significant_improvement"]
            for result in auxiliary_tests
        )
        and not any(
            result["significant_regression"]
            for result in auxiliary_tests
        )
    )
    return {
        "schema_version": 2,
        "candidate": candidate,
        "nohf_model": nohf,
        "reference": reference,
        "protocol": {
            "unseen_taus": list(unseen_taus),
            "block_days": block_days,
            "draws": draws,
            "seed": seed,
            "hf_ell_min": hf_ell_min,
            "holm_family_size_per_comparison": len(
                family_tests["architecture_only"]
            ),
        },
        "comparisons": comparisons,
        "significance": significance,
        "training_seed_count": 1,
        "inference_scope": "single_seed_conditional_on_checkpoint",
        "architecture_only_single_seed_promotion_passed": (
            architecture_confirmed
        ),
        "architecture_superiority_seed_consistent": None,
        "auxiliary_effect_single_seed_supported": auxiliary_supported,
        "checkpoint_sha256": {
            role: next(iter(values))
            for role, values in checkpoint_hashes.items()
        },
        "artifacts": {
            "field": field_artifacts,
            "spectral": spectral_artifacts,
        },
        "note": (
            "architecture_only compares the selected architecture retrained "
            "with lambda_hf=0 against PP3, which also uses lambda_hf=0. "
            "auxiliary_effect compares the default and no-HF checkpoints of "
            "the same architecture. All primary RMSE, ACC, and spectral tests "
            "within each comparison share one Holm family. This report is a "
            "single-seed promotion gate and does not establish robustness "
            "across training seeds."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--primary-root-2020", type=Path, required=True)
    parser.add_argument("--primary-root-2021", type=Path, required=True)
    parser.add_argument("--nohf-root-2020", type=Path, required=True)
    parser.add_argument("--nohf-root-2021", type=Path, required=True)
    parser.add_argument("--primary-spectra-root", type=Path, required=True)
    parser.add_argument("--nohf-spectra-root", type=Path, required=True)
    parser.add_argument("--unseen-taus", default="2,4")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--hf-ell-min", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.selection.read_text()),
        primary_root_2020=args.primary_root_2020,
        primary_root_2021=args.primary_root_2021,
        nohf_root_2020=args.nohf_root_2020,
        nohf_root_2021=args.nohf_root_2021,
        primary_spectra_root=args.primary_spectra_root,
        nohf_spectra_root=args.nohf_spectra_root,
        unseen_taus=tuple(
            int(value)
            for value in args.unseen_taus.split(",")
            if value.strip()
        ),
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
        hf_ell_min=args.hf_ell_min,
    )
    report["selection_sha256"] = _sha256(args.selection)
    report["generator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "architecture_only_single_seed_promotion_passed": report[
                    "architecture_only_single_seed_promotion_passed"
                ],
                "auxiliary_effect_single_seed_supported": report[
                    "auxiliary_effect_single_seed_supported"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
