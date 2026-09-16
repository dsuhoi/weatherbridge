#!/usr/bin/env python3
"""Compare a selected UPR model with PP3 across matched training seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.select_upr_lite_candidate import (
    load_field_metrics,
    load_spectral_metrics,
)
from tools.eval.summarize_upr_lite_seeds import (
    choose_transfer_candidate,
    load_temporal_summary,
)


DEFAULT_SEEDS = (202707, 202708, 202709)
FIELD_METRICS = {
    "unseen_rmse": "lower",
    "unseen_skill": "higher",
    "unseen_acc": "higher",
    "unseen_physical_ratio": "lower",
    "unseen_physical_ratio_max": "lower",
    "unseen_cvar95_rmse": "lower",
    "unseen_worst_season_rmse": "lower",
}
SPECTRAL_METRICS = {
    "hf_log_energy_error": "lower",
    "hf_log_shape_error": "lower",
    "hf_coherence": "higher",
}


def _provenance_sha256(provenance: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _metric_comparison(
    candidate: float,
    reference: float,
    *,
    better: str,
) -> dict[str, float]:
    improvement = (
        reference - candidate
        if better == "lower"
        else candidate - reference
    )
    return {
        "candidate": float(candidate),
        "reference": float(reference),
        "candidate_improvement": float(improvement),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    improvements = np.asarray(
        [row["candidate_improvement"] for row in rows],
        dtype=np.float64,
    )
    if improvements.size < 2 or not np.all(np.isfinite(improvements)):
        raise ValueError("seed deltas must contain at least two finite values")
    return {
        "candidate_mean": float(
            np.mean([row["candidate"] for row in rows])
        ),
        "reference_mean": float(
            np.mean([row["reference"] for row in rows])
        ),
        "improvement_mean": float(improvements.mean()),
        "improvement_std": float(improvements.std(ddof=1)),
        "improvement_min": float(improvements.min()),
        "improvement_max": float(improvements.max()),
        "candidate_better_all_seeds": bool(np.all(improvements > 0.0)),
    }


def build_report(
    selection: dict[str, Any],
    *,
    primary_root_2020: Path,
    primary_root_2021: Path,
    candidate_replicate_root_2020: Path,
    candidate_replicate_root_2021: Path,
    reference_replicate_root_2020: Path,
    reference_replicate_root_2021: Path,
    primary_spectra_root_2020: Path,
    candidate_replicate_spectra_root_2020: Path,
    reference_replicate_spectra_root_2020: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    hf_ell_min: int = 180,
    spectral_taus: tuple[int, ...] = (2, 4),
    candidate_primary_seed_from_replicate: bool = False,
    candidate_lambda_hf: float = 0.05,
    reference_lambda_hf: float = 0.0,
    candidate_artifact_name: str | None = None,
    candidate_artifact_mode: str | None = None,
    reference_primary_root_2020: Path | None = None,
    reference_primary_root_2021: Path | None = None,
    reference_primary_spectra_root_2020: Path | None = None,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    if (
        not np.isfinite(candidate_lambda_hf)
        or candidate_lambda_hf < 0.0
        or not np.isfinite(reference_lambda_hf)
        or reference_lambda_hf < 0.0
    ):
        raise ValueError("high-pass loss weights must be finite and non-negative")
    training_objective_matched = bool(
        np.isclose(
            candidate_lambda_hf,
            reference_lambda_hf,
            rtol=0.0,
            atol=1e-12,
        )
    )
    selected_architecture = choose_transfer_candidate(selection, "quality")
    candidate = candidate_artifact_name or selected_architecture
    if (
        candidate_artifact_name is not None
        and candidate_artifact_name
        != f"{selected_architecture}_nohf"
    ):
        raise ValueError(
            "candidate artifact override must be the selected architecture "
            "with the '_nohf' suffix"
        )
    if candidate_artifact_mode is None:
        candidate_artifact_mode = (
            "fresh_postselection_replicates"
            if candidate_primary_seed_from_replicate
            else "selection_seed_plus_replicates"
        )
    if candidate_artifact_mode not in {
        "fresh_postselection_replicates",
        "selection_seed_plus_replicates",
    }:
        raise ValueError("unsupported candidate artifact mode")
    reference_primary_roots = (
        reference_primary_root_2020 or primary_root_2020,
        reference_primary_root_2021 or primary_root_2021,
    )
    reference_primary_spectra_root = (
        reference_primary_spectra_root_2020 or primary_spectra_root_2020
    )
    per_seed: dict[str, Any] = {}
    paired_indices: dict[str, dict[str, str]] = {}
    paired_provenance: dict[str, dict[str, str]] = {}
    paired_dataset_provenance: dict[str, dict[str, str]] = {}
    field_input_provenance: dict[str, Any] | None = None
    field_dataset_provenance: dict[str, dict[str, Any]] = {}
    spectral_input_provenance: dict[str, Any] | None = None
    spectral_dataset_provenance: dict[str, Any] | None = None

    for index, seed in enumerate(seeds):
        candidate_uses_replicate = (
            candidate_primary_seed_from_replicate or index > 0
        )
        candidate_name = (
            f"{candidate}_s{seed}"
            if candidate_uses_replicate
            else candidate
        )
        reference_name = (
            "weatherbridge_ref"
            if index == 0
            else f"weatherbridge_ref_s{seed}"
        )
        candidate_roots = (
            (
                candidate_replicate_root_2020,
                candidate_replicate_root_2021,
            )
            if candidate_uses_replicate
            else (primary_root_2020, primary_root_2021)
        )
        reference_roots = (
            reference_primary_roots
            if index == 0
            else (
                reference_replicate_root_2020,
                reference_replicate_root_2021,
            )
        )
        row: dict[str, Any] = {"field": {}}
        candidate_field_checkpoints: set[str] = set()
        reference_field_checkpoints: set[str] = set()
        paired_indices[str(seed)] = {}
        paired_provenance[str(seed)] = {}
        paired_dataset_provenance[str(seed)] = {}
        for year_index, year in enumerate((2020, 2021)):
            candidate_metrics = load_field_metrics(
                candidate_roots[year_index] / f"{candidate_name}.json"
            )
            reference_metrics = load_field_metrics(
                reference_roots[year_index] / f"{reference_name}.json"
            )
            if not candidate_metrics["evaluation_full_year"] or not (
                reference_metrics["evaluation_full_year"]
            ):
                raise ValueError(f"seed {seed}: {year} metrics are not full-year")
            candidate_field_checkpoints.add(
                str(candidate_metrics["checkpoint_sha256"])
            )
            reference_field_checkpoints.add(
                str(reference_metrics["checkpoint_sha256"])
            )
            candidate_index = str(candidate_metrics["window_index_sha256"])
            reference_index = str(reference_metrics["window_index_sha256"])
            if candidate_index != reference_index:
                raise ValueError(
                    f"seed {seed}: {year} window-index mismatch"
                )
            paired_indices[str(seed)][str(year)] = candidate_index
            candidate_provenance = candidate_metrics[
                "evaluation_input_provenance"
            ]
            reference_provenance = reference_metrics[
                "evaluation_input_provenance"
            ]
            if candidate_provenance != reference_provenance:
                raise ValueError(
                    f"seed {seed}: {year} input-provenance mismatch"
                )
            if field_input_provenance is None:
                field_input_provenance = candidate_provenance
            elif field_input_provenance != candidate_provenance:
                raise ValueError(
                    "field input-provenance mismatch across seeds or years"
                )
            paired_provenance[str(seed)][str(year)] = (
                _provenance_sha256(candidate_provenance)
            )
            candidate_dataset = candidate_metrics[
                "evaluation_dataset_provenance"
            ]
            reference_dataset = reference_metrics[
                "evaluation_dataset_provenance"
            ]
            if candidate_dataset != reference_dataset:
                raise ValueError(
                    f"seed {seed}: {year} dataset-provenance mismatch"
                )
            year_key = str(year)
            if year_key not in field_dataset_provenance:
                field_dataset_provenance[year_key] = candidate_dataset
            elif field_dataset_provenance[year_key] != candidate_dataset:
                raise ValueError(
                    f"{year} dataset provenance differs across seeds"
                )
            paired_dataset_provenance[str(seed)][year_key] = (
                _provenance_sha256(candidate_dataset)
            )
            row["field"][str(year)] = {
                metric: _metric_comparison(
                    candidate_metrics[metric],
                    reference_metrics[metric],
                    better=better,
                )
                for metric, better in FIELD_METRICS.items()
            }
            candidate_temporal = load_temporal_summary(
                candidate_roots[year_index] / f"{candidate_name}.json"
            )
            reference_temporal = load_temporal_summary(
                reference_roots[year_index] / f"{reference_name}.json"
            )
            if (
                candidate_temporal["window_index_sha256"]
                != reference_temporal["window_index_sha256"]
                or candidate_temporal["centers"]
                != reference_temporal["centers"]
            ):
                raise ValueError(
                    f"seed {seed}: {year} temporal-index mismatch"
                )
            paired_indices[str(seed)][
                f"temporal_{year}"
            ] = candidate_temporal["window_index_sha256"]
            row.setdefault("temporal", {})[str(year)] = {
                "curvature_rmse": _metric_comparison(
                    candidate_temporal["curvature_rmse"],
                    reference_temporal["curvature_rmse"],
                    better="lower",
                )
            }

        if (
            len(candidate_field_checkpoints) != 1
            or len(reference_field_checkpoints) != 1
        ):
            raise ValueError(
                f"seed {seed}: field checkpoint changed across years"
            )
        candidate_field_checkpoint = next(iter(candidate_field_checkpoints))
        reference_field_checkpoint = next(iter(reference_field_checkpoints))
        candidate_spectra_root = (
            candidate_replicate_spectra_root_2020
            if candidate_uses_replicate
            else primary_spectra_root_2020
        )
        reference_spectra_root = (
            reference_primary_spectra_root
            if index == 0
            else reference_replicate_spectra_root_2020
        )
        candidate_spectra = load_spectral_metrics(
            candidate_spectra_root,
            candidate_name,
            hf_ell_min,
            taus=set(spectral_taus),
        )
        reference_spectra = load_spectral_metrics(
            reference_spectra_root,
            reference_name,
            hf_ell_min,
            taus=set(spectral_taus),
        )
        if (
            candidate_spectra["checkpoint_sha256"]
            != candidate_field_checkpoint
            or reference_spectra["checkpoint_sha256"]
            != reference_field_checkpoint
        ):
            raise ValueError(
                f"seed {seed}: field/spectral checkpoint mismatch"
            )
        row["checkpoint_sha256"] = {
            "candidate": candidate_field_checkpoint,
            "reference": reference_field_checkpoint,
        }
        candidate_index = str(candidate_spectra["window_index_sha256"])
        reference_index = str(reference_spectra["window_index_sha256"])
        if candidate_index != reference_index:
            raise ValueError(f"seed {seed}: spectral window-index mismatch")
        paired_indices[str(seed)]["spectral_2020"] = candidate_index
        candidate_provenance = candidate_spectra[
            "evaluation_input_provenance"
        ]
        reference_provenance = reference_spectra[
            "evaluation_input_provenance"
        ]
        if candidate_provenance != reference_provenance:
            raise ValueError(
                f"seed {seed}: spectral input-provenance mismatch"
            )
        if spectral_input_provenance is None:
            spectral_input_provenance = candidate_provenance
        elif spectral_input_provenance != candidate_provenance:
            raise ValueError(
                "spectral input-provenance mismatch across seeds"
            )
        paired_provenance[str(seed)]["spectral_2020"] = (
            _provenance_sha256(candidate_provenance)
        )
        candidate_dataset = candidate_spectra[
            "evaluation_dataset_provenance"
        ]
        reference_dataset = reference_spectra[
            "evaluation_dataset_provenance"
        ]
        if candidate_dataset != reference_dataset:
            raise ValueError(
                f"seed {seed}: spectral dataset-provenance mismatch"
            )
        if spectral_dataset_provenance is None:
            spectral_dataset_provenance = candidate_dataset
        elif spectral_dataset_provenance != candidate_dataset:
            raise ValueError(
                "spectral dataset-provenance mismatch across seeds"
            )
        paired_dataset_provenance[str(seed)]["spectral_2020"] = (
            _provenance_sha256(candidate_dataset)
        )
        row["spectral_2020"] = {
            metric: _metric_comparison(
                candidate_spectra[metric],
                reference_spectra[metric],
                better=better,
            )
            for metric, better in SPECTRAL_METRICS.items()
        }
        per_seed[str(seed)] = row

    assert field_input_provenance is not None
    assert spectral_input_provenance is not None
    assert spectral_dataset_provenance is not None
    for key in (
        "static_features",
        "pressure_level_stats",
        "surface_stats",
    ):
        if spectral_input_provenance[key] != field_input_provenance[key]:
            raise ValueError(
                f"field/spectral input-provenance mismatch for {key}"
            )
    if field_dataset_provenance["2020"] != spectral_dataset_provenance:
        raise ValueError("field/spectral 2020 dataset-provenance mismatch")

    aggregate: dict[str, Any] = {"field": {}}
    for year in ("2020", "2021"):
        aggregate["field"][year] = {
            metric: _aggregate(
                [per_seed[str(seed)]["field"][year][metric] for seed in seeds]
            )
            for metric in FIELD_METRICS
        }
    aggregate["spectral_2020"] = {
        metric: _aggregate(
            [per_seed[str(seed)]["spectral_2020"][metric] for seed in seeds]
        )
        for metric in SPECTRAL_METRICS
    }
    aggregate["temporal"] = {
        year: {
            "curvature_rmse": _aggregate(
                [
                    per_seed[str(seed)]["temporal"][year][
                        "curvature_rmse"
                    ]
                    for seed in seeds
                ]
            )
        }
        for year in ("2020", "2021")
    }
    temporal_no_regression_gates = {
        f"candidate_temporal_no_regression_all_seeds_{year}": all(
            per_seed[str(seed)]["temporal"][year]["curvature_rmse"][
                "candidate"
            ]
            <= 1.05
            * per_seed[str(seed)]["temporal"][year]["curvature_rmse"][
                "reference"
            ]
            for seed in seeds
        )
        for year in ("2020", "2021")
    }
    primary_quality_gates = {
        "candidate_lower_unseen_rmse_all_seeds_2020": aggregate["field"][
            "2020"
        ]["unseen_rmse"]["candidate_better_all_seeds"],
        "candidate_lower_unseen_rmse_all_seeds_2021": aggregate["field"][
            "2021"
        ]["unseen_rmse"]["candidate_better_all_seeds"],
        "candidate_better_hf_energy_all_seeds": aggregate["spectral_2020"][
            "hf_log_energy_error"
        ]["candidate_better_all_seeds"],
        "candidate_better_hf_shape_all_seeds": aggregate["spectral_2020"][
            "hf_log_shape_error"
        ]["candidate_better_all_seeds"],
        "candidate_better_hf_coherence_all_seeds": aggregate[
            "spectral_2020"
        ]["hf_coherence"]["candidate_better_all_seeds"],
        **temporal_no_regression_gates,
    }
    cross_metric_gates = {
        f"candidate_better_{metric}_all_seeds_{year}": aggregate["field"][
            year
        ][metric]["candidate_better_all_seeds"]
        for year in ("2020", "2021")
        for metric in FIELD_METRICS
    }
    spectral_gates = {
        f"candidate_better_{metric}_all_seeds_2020": aggregate[
            "spectral_2020"
        ][metric]["candidate_better_all_seeds"]
        for metric in SPECTRAL_METRICS
    }
    strict_gates = {
        **cross_metric_gates,
        **spectral_gates,
        **temporal_no_regression_gates,
    }
    return {
        "schema_version": 6,
        "candidate": candidate,
        "selected_architecture": selected_architecture,
        "reference": "weatherbridge_ref",
        "seeds": list(seeds),
        "candidate_seed_artifact_mode": candidate_artifact_mode,
        "training_objective": {
            "candidate_lambda_hf": float(candidate_lambda_hf),
            "reference_lambda_hf": float(reference_lambda_hf),
            "matched": training_objective_matched,
        },
        "spectral_protocol": {
            "year": 2020,
            "taus": list(spectral_taus),
            "hf_ell_min": hf_ell_min,
            "lmax": 359,
        },
        "temporal_protocol": {
            "years": [2020, 2021],
            "metric": "cosine_latitude_weighted_second_difference_rmse",
            "max_relative_regression": 0.05,
        },
        "paired_window_index_sha256": paired_indices,
        "paired_evaluation_input_provenance_sha256": paired_provenance,
        "paired_evaluation_dataset_provenance_sha256": (
            paired_dataset_provenance
        ),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "primary_quality_superiority_gates": primary_quality_gates,
        "primary_quality_superiority_seed_consistent": all(
            primary_quality_gates.values()
        ),
        "strict_cross_metric_superiority_gates": strict_gates,
        "system_superiority_seed_consistent": all(
            strict_gates.values()
        ),
        "architecture_superiority_seed_consistent": (
            all(strict_gates.values())
            if training_objective_matched
            else None
        ),
        "note": (
            "These are paired descriptive deltas across three matched training "
            "seeds. Window-level block tests are reported separately; no "
            "training-seed hypothesis test is claimed at n=3. Broad "
            "system superiority requires seed-consistent improvement on every "
            "reported field and spectral metric and no greater than 5% "
            "temporal-curvature regression on any seed. Architecture "
            "superiority is defined only when loss weights are matched."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# UPR versus WeatherBridge-PP3 Seed Comparison",
        "",
        f"Candidate: **{report['candidate']}**",
        "",
        "| Seed | RMSE improvement 2020 | RMSE improvement 2021 | "
        "HF energy-error improvement | HF shape-error improvement | "
        "HF coherence improvement | Temporal improvement 2021 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        row = report["per_seed"][str(seed)]
        lines.append(
            f"| {seed} | "
            f"{row['field']['2020']['unseen_rmse']['candidate_improvement']:+.6f} | "
            f"{row['field']['2021']['unseen_rmse']['candidate_improvement']:+.6f} | "
            f"{row['spectral_2020']['hf_log_energy_error']['candidate_improvement']:+.6f} | "
            f"{row['spectral_2020']['hf_log_shape_error']['candidate_improvement']:+.6f} | "
            f"{row['spectral_2020']['hf_coherence']['candidate_improvement']:+.6f} | "
            f"{row['temporal']['2021']['curvature_rmse']['candidate_improvement']:+.6f} |"
        )
    lines.extend(
        (
            "",
            "RMSE + spectrum seed consistency: "
            f"**{report['primary_quality_superiority_seed_consistent']}**",
            "",
            "Strict system cross-metric seed consistency: "
            f"**{report['system_superiority_seed_consistent']}**",
            "",
            "Matched-objective architecture consistency: "
            f"**{report['architecture_superiority_seed_consistent']}**",
            "",
            report["note"],
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("metrics/upr_lite_candidate_selection.json"),
    )
    parser.add_argument("--primary-root-2020", type=Path, required=True)
    parser.add_argument("--primary-root-2021", type=Path, required=True)
    parser.add_argument("--reference-primary-root-2020", type=Path)
    parser.add_argument("--reference-primary-root-2021", type=Path)
    parser.add_argument("--candidate-replicate-root-2020", type=Path, required=True)
    parser.add_argument("--candidate-replicate-root-2021", type=Path, required=True)
    parser.add_argument("--reference-replicate-root-2020", type=Path, required=True)
    parser.add_argument("--reference-replicate-root-2021", type=Path, required=True)
    parser.add_argument("--primary-spectra-root-2020", type=Path, required=True)
    parser.add_argument("--reference-primary-spectra-root-2020", type=Path)
    parser.add_argument(
        "--candidate-replicate-spectra-root-2020",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reference-replicate-spectra-root-2020",
        type=Path,
        required=True,
    )
    parser.add_argument("--seeds", default="202707,202708,202709")
    parser.add_argument("--hf-ell-min", type=int, default=180)
    parser.add_argument("--spectral-taus", default="2,4")
    parser.add_argument("--candidate-lambda-hf", type=float, default=0.05)
    parser.add_argument("--reference-lambda-hf", type=float, default=0.0)
    parser.add_argument(
        "--candidate-artifact-name",
        help=(
            "Artifact basename override; only '<selected>_nohf' is accepted."
        ),
    )
    parser.add_argument(
        "--candidate-artifact-mode",
        choices=(
            "fresh_postselection_replicates",
            "selection_seed_plus_replicates",
        ),
    )
    parser.add_argument(
        "--candidate-primary-seed-from-replicate",
        action="store_true",
        help="Load the candidate's first seed from replicate roots.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_vs_pp3_seed_comparison.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("metrics/upr_vs_pp3_seed_comparison.md"),
    )
    args = parser.parse_args()

    report = build_report(
        json.loads(args.selection.read_text()),
        primary_root_2020=args.primary_root_2020,
        primary_root_2021=args.primary_root_2021,
        candidate_replicate_root_2020=args.candidate_replicate_root_2020,
        candidate_replicate_root_2021=args.candidate_replicate_root_2021,
        reference_replicate_root_2020=args.reference_replicate_root_2020,
        reference_replicate_root_2021=args.reference_replicate_root_2021,
        primary_spectra_root_2020=args.primary_spectra_root_2020,
        candidate_replicate_spectra_root_2020=(
            args.candidate_replicate_spectra_root_2020
        ),
        reference_replicate_spectra_root_2020=(
            args.reference_replicate_spectra_root_2020
        ),
        seeds=tuple(int(value) for value in args.seeds.split(",") if value),
        hf_ell_min=args.hf_ell_min,
        spectral_taus=tuple(
            int(value)
            for value in args.spectral_taus.split(",")
            if value
        ),
        candidate_primary_seed_from_replicate=(
            args.candidate_primary_seed_from_replicate
        ),
        candidate_lambda_hf=args.candidate_lambda_hf,
        reference_lambda_hf=args.reference_lambda_hf,
        candidate_artifact_name=args.candidate_artifact_name,
        candidate_artifact_mode=args.candidate_artifact_mode,
        reference_primary_root_2020=args.reference_primary_root_2020,
        reference_primary_root_2021=args.reference_primary_root_2021,
        reference_primary_spectra_root_2020=(
            args.reference_primary_spectra_root_2020
        ),
    )
    report["summary_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["selection_sha256"] = hashlib.sha256(
        args.selection.read_bytes()
    ).hexdigest()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = args.out_json.with_name(
        f".{args.out_json.name}.{os.getpid()}.tmp"
    )
    md_tmp = args.out_md.with_name(f".{args.out_md.name}.{os.getpid()}.tmp")
    json_tmp.write_text(json.dumps(report, indent=2) + "\n")
    md_tmp.write_text(markdown_report(report))
    json_tmp.replace(args.out_json)
    md_tmp.replace(args.out_md)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "seed_consistent": report[
                    "system_superiority_seed_consistent"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
