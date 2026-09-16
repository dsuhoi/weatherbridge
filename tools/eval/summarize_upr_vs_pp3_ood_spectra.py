#!/usr/bin/env python3
"""Summarize independent 2021 spectral confirmation across matched seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from tools.eval.select_upr_lite_candidate import load_spectral_metrics
from tools.eval.summarize_upr_lite_seeds import choose_transfer_candidate
from tools.eval.summarize_upr_vs_pp3_seeds import (
    SPECTRAL_METRICS,
    _aggregate,
    _metric_comparison,
)


DEFAULT_SEEDS = (202707, 202708, 202709)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def build_report(
    selection: dict[str, Any],
    spectra_root: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    hf_ell_min: int = 180,
    spectral_taus: tuple[int, ...] = (2, 4),
    candidate_primary_seed_from_replicate: bool = False,
    reference_spectra_root: Path | None = None,
    candidate_artifact_name: str | None = None,
    candidate_artifact_mode: str | None = None,
    candidate_lambda_hf: float = 0.05,
    reference_lambda_hf: float = 0.0,
    checkpoint_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    if (
        not math.isfinite(candidate_lambda_hf)
        or candidate_lambda_hf < 0.0
        or not math.isfinite(reference_lambda_hf)
        or reference_lambda_hf < 0.0
    ):
        raise ValueError("high-pass loss weights must be finite and non-negative")
    training_objective_matched = (
        abs(candidate_lambda_hf - reference_lambda_hf) <= 1e-12
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
    if reference_spectra_root is None:
        reference_spectra_root = spectra_root
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
    if checkpoint_report is not None and (
        checkpoint_report.get("schema_version") != 6
        or checkpoint_report.get("candidate") != candidate
        or checkpoint_report.get("reference") != "weatherbridge_ref"
        or checkpoint_report.get("seeds") != list(seeds)
        or checkpoint_report.get("candidate_seed_artifact_mode")
        != candidate_artifact_mode
    ):
        raise ValueError("seed checkpoint report protocol mismatch")
    per_seed: dict[str, Any] = {}
    index_hashes: set[str] = set()
    input_provenance_hashes: set[str] = set()
    dataset_provenance_hashes: set[str] = set()

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
        candidate_metrics = load_spectral_metrics(
            spectra_root,
            candidate_name,
            hf_ell_min,
            taus=set(spectral_taus),
            required_lmax=359,
            expected_channels=24,
        )
        reference_metrics = load_spectral_metrics(
            reference_spectra_root,
            reference_name,
            hf_ell_min,
            taus=set(spectral_taus),
            required_lmax=359,
            expected_channels=24,
        )
        if checkpoint_report is not None:
            expected = checkpoint_report.get("per_seed", {}).get(
                str(seed), {}
            ).get("checkpoint_sha256")
            if not isinstance(expected, dict) or (
                candidate_metrics["checkpoint_sha256"]
                != expected.get("candidate")
                or reference_metrics["checkpoint_sha256"]
                != expected.get("reference")
            ):
                raise ValueError(
                    f"seed {seed}: OOD spectral checkpoint mismatch"
                )
        candidate_index = str(candidate_metrics["window_index_sha256"])
        reference_index = str(reference_metrics["window_index_sha256"])
        if candidate_index != reference_index:
            raise ValueError(f"seed {seed}: spectral window-index mismatch")
        candidate_input = candidate_metrics["evaluation_input_provenance"]
        reference_input = reference_metrics["evaluation_input_provenance"]
        if candidate_input != reference_input:
            raise ValueError(f"seed {seed}: spectral input-provenance mismatch")
        candidate_dataset = candidate_metrics[
            "evaluation_dataset_provenance"
        ]
        reference_dataset = reference_metrics[
            "evaluation_dataset_provenance"
        ]
        if candidate_dataset != reference_dataset:
            raise ValueError(
                f"seed {seed}: spectral dataset-provenance mismatch"
            )
        if candidate_dataset.get("years") != [2021]:
            raise ValueError(f"seed {seed}: expected 2021 spectral dataset")
        index_hashes.add(candidate_index)
        input_provenance_hashes.add(_canonical_sha256(candidate_input))
        dataset_provenance_hashes.add(_canonical_sha256(candidate_dataset))
        per_seed[str(seed)] = {
            "candidate_checkpoint_sha256": candidate_metrics[
                "checkpoint_sha256"
            ],
            "reference_checkpoint_sha256": reference_metrics[
                "checkpoint_sha256"
            ],
            "metrics": {
                metric: _metric_comparison(
                    candidate_metrics[metric],
                    reference_metrics[metric],
                    better=better,
                )
                for metric, better in SPECTRAL_METRICS.items()
            },
        }

    if len(index_hashes) != 1:
        raise ValueError("spectral window index differs across seeds")
    if len(input_provenance_hashes) != 1:
        raise ValueError("spectral input provenance differs across seeds")
    if len(dataset_provenance_hashes) != 1:
        raise ValueError("spectral dataset provenance differs across seeds")

    aggregate = {
        metric: _aggregate(
            [
                per_seed[str(seed)]["metrics"][metric]
                for seed in seeds
            ]
        )
        for metric in SPECTRAL_METRICS
    }
    gates = {
        f"candidate_better_{metric}_all_seeds_2021": aggregate[metric][
            "candidate_better_all_seeds"
        ]
        for metric in SPECTRAL_METRICS
    }
    return {
        "schema_version": 2,
        "candidate": candidate,
        "selected_architecture": selected_architecture,
        "reference": "weatherbridge_ref",
        "seeds": list(seeds),
        "candidate_seed_artifact_mode": candidate_artifact_mode,
        "checkpoint_linkage_verified": checkpoint_report is not None,
        "training_objective": {
            "candidate_lambda_hf": float(candidate_lambda_hf),
            "reference_lambda_hf": float(reference_lambda_hf),
            "matched": training_objective_matched,
        },
        "spectral_protocol": {
            "confirmation_year": 2021,
            "used_for_model_selection": False,
            "taus": list(spectral_taus),
            "hf_ell_min": hf_ell_min,
            "lmax": 359,
            "channels": 24,
        },
        "paired_window_index_sha256": next(iter(index_hashes)),
        "evaluation_input_provenance_sha256": next(
            iter(input_provenance_hashes)
        ),
        "evaluation_dataset_provenance_sha256": next(
            iter(dataset_provenance_hashes)
        ),
        "per_seed": per_seed,
        "aggregate": aggregate,
        "gates": gates,
        "ood_spectral_superiority_seed_consistent": all(gates.values()),
        "architecture_ood_spectral_superiority_seed_consistent": (
            all(gates.values()) if training_objective_matched else None
        ),
        "note": (
            "This frozen 2021 spectrum is an independent confirmation and "
            "does not alter the 2020 model selection."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# UPR versus WeatherBridge-PP3 OOD Spectrum",
        "",
        f"Candidate: **{report['candidate']}**",
        "",
        "| Seed | HF energy-error improvement | HF shape-error improvement | "
        "HF coherence improvement |",
        "|---:|---:|---:|---:|",
    ]
    for seed in report["seeds"]:
        metrics = report["per_seed"][str(seed)]["metrics"]
        lines.append(
            f"| {seed} | "
            f"{metrics['hf_log_energy_error']['candidate_improvement']:+.6f} | "
            f"{metrics['hf_log_shape_error']['candidate_improvement']:+.6f} | "
            f"{metrics['hf_coherence']['candidate_improvement']:+.6f} |"
        )
    lines.extend(
        (
            "",
            "OOD spectral superiority across all seeds: "
            f"**{report['ood_spectral_superiority_seed_consistent']}**",
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
    parser.add_argument("--spectra-root", type=Path, required=True)
    parser.add_argument("--reference-spectra-root", type=Path)
    parser.add_argument("--seed-comparison", type=Path)
    parser.add_argument("--seeds", default="202707,202708,202709")
    parser.add_argument("--hf-ell-min", type=int, default=180)
    parser.add_argument("--spectral-taus", default="2,4")
    parser.add_argument("--candidate-artifact-name")
    parser.add_argument(
        "--candidate-artifact-mode",
        choices=(
            "fresh_postselection_replicates",
            "selection_seed_plus_replicates",
        ),
    )
    parser.add_argument("--candidate-lambda-hf", type=float, default=0.05)
    parser.add_argument("--reference-lambda-hf", type=float, default=0.0)
    parser.add_argument(
        "--candidate-primary-seed-from-replicate",
        action="store_true",
        help="Load the candidate's first seed as a fresh replicate.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_vs_pp3_ood_spectra_2021.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("metrics/upr_vs_pp3_ood_spectra_2021.md"),
    )
    args = parser.parse_args()
    selection_bytes = args.selection.read_bytes()
    checkpoint_report = (
        json.loads(args.seed_comparison.read_text())
        if args.seed_comparison is not None
        else None
    )
    selection_sha256 = hashlib.sha256(selection_bytes).hexdigest()
    if checkpoint_report is not None and (
        checkpoint_report.get("selection_sha256") != selection_sha256
    ):
        raise ValueError("seed checkpoint report selection mismatch")
    report = build_report(
        json.loads(selection_bytes),
        args.spectra_root,
        seeds=tuple(
            int(value) for value in args.seeds.split(",") if value
        ),
        hf_ell_min=args.hf_ell_min,
        spectral_taus=tuple(
            int(value)
            for value in args.spectral_taus.split(",")
            if value
        ),
        candidate_primary_seed_from_replicate=(
            args.candidate_primary_seed_from_replicate
        ),
        reference_spectra_root=args.reference_spectra_root,
        candidate_artifact_name=args.candidate_artifact_name,
        candidate_artifact_mode=args.candidate_artifact_mode,
        candidate_lambda_hf=args.candidate_lambda_hf,
        reference_lambda_hf=args.reference_lambda_hf,
        checkpoint_report=checkpoint_report,
    )
    report["summary_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["selection_sha256"] = selection_sha256
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
                "ood_spectral_superiority_seed_consistent": report[
                    "ood_spectral_superiority_seed_consistent"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
