#!/usr/bin/env python3
"""Summarize three-seed robustness for the selected UPR-Lite candidate."""
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


DEFAULT_SEEDS = (202707, 202708, 202709)
METRICS = (
    "seen_rmse",
    "unseen_rmse",
    "seen_skill",
    "unseen_skill",
    "seen_acc",
    "unseen_acc",
    "seen_physical_ratio",
    "unseen_physical_ratio",
    "seen_physical_ratio_max",
    "unseen_physical_ratio_max",
    "unseen_cvar95_rmse",
    "unseen_cvar95_skill",
    "unseen_worst_season_rmse",
    "unseen_worst_season_skill",
    "skill_gap",
)
SPECTRAL_METRICS = (
    "hf_energy_ratio",
    "hf_log_energy_error",
    "hf_log_shape_error",
    "hf_coherence",
)
TEMPORAL_METRICS = (
    "curvature_rmse",
    "curvature_ratio_to_bilinear",
)
DERIVED_CHECKPOINT_SUFFIXES = ("_avg3",)
TRANSFERABLE_EXPERIMENTAL_MODELS = frozenset(
    {"flow_spherical_ep", "amt", "amt_residual"}
)


def base_upr_architecture(name: str) -> str:
    """Map an evaluation-only checkpoint label to its trainable architecture."""
    for suffix in DERIVED_CHECKPOINT_SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def choose_upr_candidate(selection: dict[str, Any]) -> str:
    winner = str(selection["winner"])
    if winner.startswith("upr_"):
        return base_upr_architecture(winner)
    rows = selection["models"]
    upr_models = [name for name in rows if name.startswith("upr_")]
    if not upr_models:
        raise ValueError("selection contains no UPR-Lite candidate")
    eligible = set(selection.get("eligible", ()))
    eligible_upr_models = [
        name for name in upr_models if name in eligible
    ]
    candidate_pool = eligible_upr_models or upr_models
    selected = min(
        candidate_pool,
        key=lambda name: (float(rows[name]["mean_rank"]), name),
    )
    return base_upr_architecture(selected)


def _is_transferable_experimental_model(name: str) -> bool:
    base_name = base_upr_architecture(name)
    return (
        base_name.startswith("upr_")
        or base_name in TRANSFERABLE_EXPERIMENTAL_MODELS
    )


def validate_frozen_selection_for_followup(
    selection: dict[str, Any],
    *,
    expected_field_hour_cells: int = 120,
) -> None:
    """Require the current fail-closed 2020 selection before follow-ups."""
    schema_version = selection.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 13
    ):
        raise ValueError("follow-up requires frozen selection schema 13")
    if selection.get("ood_attached_at_selection_time") is not False:
        raise ValueError("follow-up selection must be frozen before 2021 OOD")
    models = selection.get("models")
    eligible = selection.get("eligible")
    if not isinstance(models, dict) or not models:
        raise ValueError("follow-up selection has no model rows")
    if not isinstance(eligible, list) or not eligible:
        raise ValueError("follow-up selection has no eligible models")
    for key in ("winner", "efficiency_winner"):
        selected = selection.get(key)
        if not isinstance(selected, str) or selected not in models:
            raise ValueError(f"follow-up selection has invalid {key}")
    if any(
        not isinstance(row, dict)
        or not isinstance(row.get("checkpoint_sha256"), str)
        or not row["checkpoint_sha256"]
        for row in models.values()
    ):
        raise ValueError("follow-up selection has unbound model checkpoint")
    rule = selection.get("selection_rule")
    diagnostics = (
        rule.get("gate_diagnostics")
        if isinstance(rule, dict)
        else None
    )
    dominance = (
        diagnostics.get("field_hour_dominance")
        if isinstance(diagnostics, dict)
        else None
    )
    if (
        not isinstance(dominance, dict)
        or dominance.get("active") is not True
        or dominance.get("total_cells") != expected_field_hour_cells
    ):
        raise ValueError(
            "follow-up selection lacks complete field-by-hour dominance"
        )
    frozen = selection.get("paired_window_index_sha256")
    required_frozen = {
        "field_2020",
        "spectral_2020",
        "field_dataset_2020",
        "spectral_dataset_2020",
    }
    if (
        not isinstance(frozen, dict)
        or not required_frozen.issubset(frozen)
        or any(not frozen[key] for key in required_frozen)
    ):
        raise ValueError("follow-up selection lacks frozen 2020 artifacts")


def choose_transfer_candidate(
    selection: dict[str, Any],
    objective: str,
) -> str:
    """Choose a trainable non-reference arm for matched transfer.

    The overall quality or efficiency winner may be an existing reference,
    which is retrained by its own matched queue. In that case, retain the
    strongest eligible experimental architecture for the requested objective.
    """
    if objective not in {"quality", "efficiency"}:
        raise ValueError(f"unsupported transfer objective: {objective}")
    rows = selection["models"]
    experimental = [
        name
        for name in rows
        if _is_transferable_experimental_model(name)
    ]
    if not experimental:
        raise ValueError(
            "selection contains no transferable experimental candidate"
        )
    eligible = set(selection.get("eligible", ()))
    eligible_experimental = [
        name for name in experimental if name in eligible
    ]
    candidate_pool = eligible_experimental or experimental
    selected_key = (
        "winner" if objective == "quality" else "efficiency_winner"
    )
    selected = str(selection.get(selected_key, ""))
    if selected in candidate_pool:
        return base_upr_architecture(selected)
    rank_key = (
        "selection_quality_mean_rank"
        if objective == "quality"
        else "selection_efficiency_mean_rank"
    )
    fallback_rank_key = (
        "quality_mean_rank"
        if objective == "quality"
        else "efficiency_mean_rank"
    )

    def row_rank(
        name: str,
        key: str,
        fallback_key: str,
    ) -> float:
        value = rows[name].get(key)
        if value is None:
            value = rows[name].get(
                fallback_key,
                rows[name].get("mean_rank", float("inf")),
            )
        return float(value)

    selected = min(
        candidate_pool,
        key=lambda name: (
            row_rank(
                name,
                rank_key,
                fallback_rank_key,
            ),
            row_rank(
                name,
                "selection_quality_mean_rank",
                "quality_mean_rank",
            ),
            name,
        ),
    )
    return base_upr_architecture(selected)


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("seed metrics must be non-empty and finite")
    mean = float(np.mean(array))
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "cv": abs(std / mean) if mean != 0.0 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _provenance_sha256(provenance: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            provenance,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def load_temporal_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol")
    summary = payload.get("temporal_curvature_rmse_norm")
    if not isinstance(protocol, dict) or not isinstance(summary, dict):
        raise ValueError(f"{path}: missing temporal metrics")
    if protocol.get("save_temporal_metrics") is not True:
        raise ValueError(f"{path}: temporal metrics were not enabled")
    index_sha = protocol.get("temporal_index_sha256")
    centers = protocol.get("temporal_centers")
    if not isinstance(index_sha, str) or not index_sha:
        raise ValueError(f"{path}: missing temporal index hash")
    if not isinstance(centers, list) or not centers:
        raise ValueError(f"{path}: missing temporal centres")
    model = float(summary.get("model", float("nan")))
    bilinear = float(summary.get("bilinear", float("nan")))
    if (
        not np.isfinite(model)
        or not np.isfinite(bilinear)
        or model < 0.0
        or bilinear <= 0.0
    ):
        raise ValueError(f"{path}: invalid temporal curvature summary")
    return {
        "curvature_rmse": model,
        "bilinear_curvature_rmse": bilinear,
        "curvature_ratio_to_bilinear": model / bilinear,
        "window_index_sha256": index_sha,
        "centers": tuple(int(value) for value in centers),
    }


def build_seed_report(
    selection: dict[str, Any],
    root_2020: Path,
    root_2021: Path,
    replicate_root_2020: Path,
    replicate_root_2021: Path,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    *,
    spectra_root_2020: Path | None = None,
    replicate_spectra_root_2020: Path | None = None,
    hf_ell_min: int = 180,
    spectral_taus: tuple[int, ...] = (2, 4),
    require_temporal: bool = False,
    all_seeds_from_replicates: bool = False,
) -> dict[str, Any]:
    if len(seeds) < 2:
        raise ValueError("at least two seeds are required")
    include_spectra = (
        spectra_root_2020 is not None
        or replicate_spectra_root_2020 is not None
    )
    if include_spectra and (
        spectra_root_2020 is None
        or replicate_spectra_root_2020 is None
    ):
        raise ValueError(
            "base and replicate spectral roots must be supplied together"
        )
    candidate = choose_transfer_candidate(selection, "quality")
    per_seed: dict[str, dict[str, Any]] = {}
    for index, seed in enumerate(seeds):
        use_replicate = all_seeds_from_replicates or index > 0
        if not use_replicate:
            path_2020 = root_2020 / f"{candidate}.json"
            path_2021 = root_2021 / f"{candidate}.json"
        else:
            name = f"{candidate}_s{seed}"
            path_2020 = replicate_root_2020 / f"{name}.json"
            path_2021 = replicate_root_2021 / f"{name}.json"
        metrics_2020 = load_field_metrics(path_2020)
        metrics_2021 = load_field_metrics(path_2021)
        for year, metrics in (
            (2020, metrics_2020),
            (2021, metrics_2021),
        ):
            if not metrics["evaluation_full_year"]:
                raise ValueError(
                    f"seed {seed}: {year} field metrics are not full-year"
                )
        per_seed[str(seed)] = {
            "2020": metrics_2020,
            "2021": metrics_2021,
        }
        if require_temporal:
            per_seed[str(seed)]["temporal_2020"] = (
                load_temporal_summary(path_2020)
            )
            per_seed[str(seed)]["temporal_2021"] = (
                load_temporal_summary(path_2021)
            )
        if include_spectra:
            spectral_name = (
                candidate
                if not use_replicate
                else f"{candidate}_s{seed}"
            )
            spectral_root = (
                spectra_root_2020
                if not use_replicate
                else replicate_spectra_root_2020
            )
            per_seed[str(seed)]["spectral_2020"] = load_spectral_metrics(
                spectral_root,
                spectral_name,
                hf_ell_min,
                taus=set(spectral_taus),
            )

    paired_indices = {}
    input_provenance_hashes: dict[str, str] = {}
    dataset_provenance_hashes: dict[str, str] = {}
    for year in ("2020", "2021"):
        hashes = {
            str(seed): str(
                per_seed[str(seed)][year]["window_index_sha256"]
            )
            for seed in seeds
        }
        if len(set(hashes.values())) != 1:
            raise ValueError(f"{year} seed window-index mismatch: {hashes}")
        paired_indices[year] = next(iter(hashes.values()))
        provenance_hashes = {
            str(seed): _provenance_sha256(
                per_seed[str(seed)][year][
                    "evaluation_input_provenance"
                ]
            )
            for seed in seeds
        }
        if len(set(provenance_hashes.values())) != 1:
            raise ValueError(
                f"{year} seed input-provenance mismatch: "
                f"{provenance_hashes}"
            )
        input_provenance_hashes[f"field_{year}"] = next(
            iter(provenance_hashes.values())
        )
        dataset_hashes = {
            str(seed): _provenance_sha256(
                per_seed[str(seed)][year][
                    "evaluation_dataset_provenance"
                ]
            )
            for seed in seeds
        }
        if len(set(dataset_hashes.values())) != 1:
            raise ValueError(
                f"{year} seed dataset-provenance mismatch: "
                f"{dataset_hashes}"
            )
        dataset_provenance_hashes[f"field_{year}"] = next(
            iter(dataset_hashes.values())
        )
    if (
        input_provenance_hashes["field_2020"]
        != input_provenance_hashes["field_2021"]
    ):
        raise ValueError("field input-provenance mismatch across years")
    if include_spectra:
        spectral_hashes = {
            str(seed): str(
                per_seed[str(seed)]["spectral_2020"][
                    "window_index_sha256"
                ]
            )
            for seed in seeds
        }
        if len(set(spectral_hashes.values())) != 1:
            raise ValueError(
                "spectral seed window-index mismatch: "
                f"{spectral_hashes}"
            )
        paired_indices["spectral_2020"] = next(
            iter(spectral_hashes.values())
        )
        spectral_provenance_hashes = {
            str(seed): _provenance_sha256(
                per_seed[str(seed)]["spectral_2020"][
                    "evaluation_input_provenance"
                ]
            )
            for seed in seeds
        }
        if len(set(spectral_provenance_hashes.values())) != 1:
            raise ValueError(
                "spectral seed input-provenance mismatch: "
                f"{spectral_provenance_hashes}"
            )
        input_provenance_hashes["spectral_2020"] = next(
            iter(spectral_provenance_hashes.values())
        )
        spectral_dataset_hashes = {
            str(seed): _provenance_sha256(
                per_seed[str(seed)]["spectral_2020"][
                    "evaluation_dataset_provenance"
                ]
            )
            for seed in seeds
        }
        if len(set(spectral_dataset_hashes.values())) != 1:
            raise ValueError(
                "spectral seed dataset-provenance mismatch: "
                f"{spectral_dataset_hashes}"
            )
        dataset_provenance_hashes["spectral_2020"] = next(
            iter(spectral_dataset_hashes.values())
        )
        field_provenance = per_seed[str(seeds[0])]["2020"][
            "evaluation_input_provenance"
        ]
        spectral_provenance = per_seed[str(seeds[0])]["spectral_2020"][
            "evaluation_input_provenance"
        ]
        for key in (
            "static_features",
            "pressure_level_stats",
            "surface_stats",
        ):
            if spectral_provenance[key] != field_provenance[key]:
                raise ValueError(
                    "field/spectral input-provenance mismatch "
                    f"for {key}"
                )
        if (
            per_seed[str(seeds[0])]["2020"][
                "evaluation_dataset_provenance"
            ]
            != per_seed[str(seeds[0])]["spectral_2020"][
                "evaluation_dataset_provenance"
            ]
        ):
            raise ValueError(
                "field/spectral 2020 dataset-provenance mismatch"
            )
    if require_temporal:
        for year in ("2020", "2021"):
            temporal_hashes = {
                str(seed): str(
                    per_seed[str(seed)][f"temporal_{year}"][
                        "window_index_sha256"
                    ]
                )
                for seed in seeds
            }
            if len(set(temporal_hashes.values())) != 1:
                raise ValueError(
                    f"{year} seed temporal-index mismatch: "
                    f"{temporal_hashes}"
                )
            paired_indices[f"temporal_{year}"] = next(
                iter(temporal_hashes.values())
            )
            center_schedules = {
                per_seed[str(seed)][f"temporal_{year}"]["centers"]
                for seed in seeds
            }
            if len(center_schedules) != 1:
                raise ValueError(
                    f"{year} seed temporal-centre mismatch"
                )

    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for year in ("2020", "2021"):
        aggregate[year] = {
            metric: _aggregate(
                [per_seed[str(seed)][year][metric] for seed in seeds]
            )
            for metric in METRICS
        }
    if include_spectra:
        aggregate["spectral_2020"] = {
            metric: _aggregate(
                [
                    per_seed[str(seed)]["spectral_2020"][metric]
                    for seed in seeds
                ]
            )
            for metric in SPECTRAL_METRICS
        }
    if require_temporal:
        for year in ("2020", "2021"):
            aggregate[f"temporal_{year}"] = {
                metric: _aggregate(
                    [
                        per_seed[str(seed)][f"temporal_{year}"][metric]
                        for seed in seeds
                    ]
                )
                for metric in TEMPORAL_METRICS
            }

    gates = {
        "all_2021_unseen_skill_positive": all(
            per_seed[str(seed)]["2021"]["unseen_skill"] > 0.0
            for seed in seeds
        ),
        "all_2020_skill_gaps_within_0p05": all(
            abs(per_seed[str(seed)]["2020"]["skill_gap"]) <= 0.05
            for seed in seeds
        ),
        "unseen_rmse_cv_2020_le_0p05": (
            aggregate["2020"]["unseen_rmse"]["cv"] <= 0.05
        ),
        "unseen_rmse_cv_2021_le_0p05": (
            aggregate["2021"]["unseen_rmse"]["cv"] <= 0.05
        ),
        "all_2021_physical_max_ratios_le_1p05": all(
            per_seed[str(seed)]["2021"][
                "unseen_physical_ratio_max"
            ] <= 1.05
            for seed in seeds
        ),
        "tail_rmse_cv_2021_le_0p10": (
            aggregate["2021"]["unseen_cvar95_rmse"]["cv"] <= 0.10
        ),
        "worst_season_rmse_cv_2021_le_0p10": (
            aggregate["2021"]["unseen_worst_season_rmse"]["cv"] <= 0.10
        ),
    }
    if include_spectra:
        gates.update(
            {
                "hf_energy_ratio_cv_2020_le_0p10": (
                    aggregate["spectral_2020"]["hf_energy_ratio"]["cv"]
                    <= 0.10
                ),
                "hf_log_shape_error_std_2020_le_0p02": (
                    aggregate["spectral_2020"]["hf_log_shape_error"]["std"]
                    <= 0.02
                ),
                "hf_coherence_cv_2020_le_0p10": (
                    aggregate["spectral_2020"]["hf_coherence"]["cv"]
                    <= 0.10
                ),
            }
        )
    if require_temporal:
        gates.update(
            {
                "all_2021_temporal_ratios_le_1p05": all(
                    per_seed[str(seed)]["temporal_2021"][
                        "curvature_ratio_to_bilinear"
                    ] <= 1.05
                    for seed in seeds
                ),
                "temporal_curvature_rmse_cv_2020_le_0p10": (
                    aggregate["temporal_2020"]["curvature_rmse"]["cv"]
                    <= 0.10
                ),
                "temporal_curvature_rmse_cv_2021_le_0p10": (
                    aggregate["temporal_2021"]["curvature_rmse"]["cv"]
                    <= 0.10
                ),
            }
        )
    return {
        "schema_version": (
            5
            if include_spectra and require_temporal
            else 4 if include_spectra else 3 if require_temporal else 2
        ),
        "candidate": candidate,
        "seeds": list(seeds),
        "seed_artifact_mode": (
            "fresh_postselection_replicates"
            if all_seeds_from_replicates
            else "selection_seed_plus_replicates"
        ),
        "spectral_protocol": (
            {
                "year": 2020,
                "taus": list(spectral_taus),
                "hf_ell_min": hf_ell_min,
                "lmax": 359,
            }
            if include_spectra
            else None
        ),
        "temporal_protocol": (
            {
                "years": [2020, 2021],
                "metric": "cosine_latitude_weighted_second_difference_rmse",
                "centers": list(
                    per_seed[str(seeds[0])]["temporal_2020"]["centers"]
                ),
            }
            if require_temporal
            else None
        ),
        "per_seed": per_seed,
        "paired_window_index_sha256": paired_indices,
        "evaluation_input_provenance_sha256": input_provenance_hashes,
        "evaluation_dataset_provenance_sha256": (
            dataset_provenance_hashes
        ),
        "aggregate": aggregate,
        "descriptive_stability_gates": gates,
        "three_seed_consistent": all(gates.values()),
        "note": (
            "Mean, sample standard deviation, and coefficient of variation are "
            "descriptive across three training seeds; they are not a hypothesis test."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    has_spectra = report.get("spectral_protocol") is not None
    has_temporal = report.get("temporal_protocol") is not None
    lines = [
        "# UPR-Lite Seed Robustness",
        "",
        f"Candidate: **{report['candidate']}**",
        "",
        "| Seed | 2020 unseen RMSE | 2020 unseen skill | "
        "2021 unseen RMSE | 2021 unseen skill | 2021 unseen ACC | "
        "Physics/Bilinear"
        + (
            " | HF ratio | HF shape error | HF coherence |"
            if has_spectra
            else " |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|"
        + ("---:|---:|---:|" if has_spectra else ""),
    ]
    for seed in report["seeds"]:
        row_2020 = report["per_seed"][str(seed)]["2020"]
        row_2021 = report["per_seed"][str(seed)]["2021"]
        line = (
            f"| {seed} | {row_2020['unseen_rmse']:.6f} | "
            f"{row_2020['unseen_skill']:.4f} | "
            f"{row_2021['unseen_rmse']:.6f} | "
            f"{row_2021['unseen_skill']:.4f} | "
            f"{row_2021['unseen_acc']:.6f} | "
            f"{row_2021['unseen_physical_ratio']:.4f}"
        )
        if has_spectra:
            spectral = report["per_seed"][str(seed)]["spectral_2020"]
            line += (
                f" | {spectral['hf_energy_ratio']:.4f} | "
                f"{spectral['hf_log_shape_error']:.4f} | "
                f"{spectral['hf_coherence']:.4f}"
            )
        lines.append(line + " |")
    agg_2020 = report["aggregate"]["2020"]
    agg_2021 = report["aggregate"]["2021"]
    lines.extend(
        (
            "",
            "| Split | RMSE mean | RMSE std | RMSE CV | "
            "Skill mean | Skill std |",
            "|---|---:|---:|---:|---:|---:|",
            "| 2020 unseen | "
            f"{agg_2020['unseen_rmse']['mean']:.6f} | "
            f"{agg_2020['unseen_rmse']['std']:.6f} | "
            f"{agg_2020['unseen_rmse']['cv']:.4f} | "
            f"{agg_2020['unseen_skill']['mean']:.4f} | "
            f"{agg_2020['unseen_skill']['std']:.4f} |",
            "| 2021 unseen | "
            f"{agg_2021['unseen_rmse']['mean']:.6f} | "
            f"{agg_2021['unseen_rmse']['std']:.6f} | "
            f"{agg_2021['unseen_rmse']['cv']:.4f} | "
            f"{agg_2021['unseen_skill']['mean']:.4f} | "
            f"{agg_2021['unseen_skill']['std']:.4f} |",
            "",
        )
    )
    if has_spectra:
        spectral = report["aggregate"]["spectral_2020"]
        lines.extend(
            (
                "| Spectral metric | Mean | Std | CV |",
                "|---|---:|---:|---:|",
                "| HF energy ratio | "
                f"{spectral['hf_energy_ratio']['mean']:.4f} | "
                f"{spectral['hf_energy_ratio']['std']:.4f} | "
                f"{spectral['hf_energy_ratio']['cv']:.4f} |",
                "| HF log-shape error | "
                f"{spectral['hf_log_shape_error']['mean']:.4f} | "
                f"{spectral['hf_log_shape_error']['std']:.4f} | "
                f"{spectral['hf_log_shape_error']['cv']:.4f} |",
                "| HF coherence | "
                f"{spectral['hf_coherence']['mean']:.4f} | "
                f"{spectral['hf_coherence']['std']:.4f} | "
                f"{spectral['hf_coherence']['cv']:.4f} |",
                "",
            )
        )
    if has_temporal:
        temporal_2020 = report["aggregate"]["temporal_2020"]
        temporal_2021 = report["aggregate"]["temporal_2021"]
        lines.extend(
            (
                "| Temporal metric | Mean | Std | CV |",
                "|---|---:|---:|---:|",
                "| 2020 curvature RMSE | "
                f"{temporal_2020['curvature_rmse']['mean']:.6f} | "
                f"{temporal_2020['curvature_rmse']['std']:.6f} | "
                f"{temporal_2020['curvature_rmse']['cv']:.4f} |",
                "| 2021 curvature RMSE | "
                f"{temporal_2021['curvature_rmse']['mean']:.6f} | "
                f"{temporal_2021['curvature_rmse']['std']:.6f} | "
                f"{temporal_2021['curvature_rmse']['cv']:.4f} |",
                "| 2021 ratio to bilinear | "
                f"{temporal_2021['curvature_ratio_to_bilinear']['mean']:.4f} | "
                f"{temporal_2021['curvature_ratio_to_bilinear']['std']:.4f} | "
                f"{temporal_2021['curvature_ratio_to_bilinear']['cv']:.4f} |",
                "",
            )
        )
    lines.extend(
        (
            f"Descriptively consistent: **{report['three_seed_consistent']}**",
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
    parser.add_argument(
        "--root-2020",
        type=Path,
        default=Path("metrics/upr_lite_screen_6h_2020"),
    )
    parser.add_argument(
        "--root-2021",
        type=Path,
        default=Path("metrics/upr_lite_screen_6h_2021"),
    )
    parser.add_argument(
        "--replicate-root-2020",
        type=Path,
        default=Path("metrics/upr_lite_seed_replicates_6h_2020"),
    )
    parser.add_argument(
        "--replicate-root-2021",
        type=Path,
        default=Path("metrics/upr_lite_seed_replicates_6h_2021"),
    )
    parser.add_argument(
        "--spectra-root-2020",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--replicate-spectra-root-2020",
        type=Path,
        default=None,
    )
    parser.add_argument("--hf-ell-min", type=int, default=180)
    parser.add_argument("--spectral-taus", default="2,4")
    parser.add_argument("--require-temporal", action="store_true")
    parser.add_argument(
        "--all-seeds-from-replicates",
        action="store_true",
        help="Load every seed, including the first, from replicate roots.",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_lite_seed_robustness.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("metrics/upr_lite_seed_robustness.md"),
    )
    args = parser.parse_args()

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    selection = json.loads(args.selection.read_text())
    report = build_seed_report(
        selection,
        args.root_2020,
        args.root_2021,
        args.replicate_root_2020,
        args.replicate_root_2021,
        seeds,
        spectra_root_2020=args.spectra_root_2020,
        replicate_spectra_root_2020=args.replicate_spectra_root_2020,
        hf_ell_min=args.hf_ell_min,
        spectral_taus=tuple(
            int(value)
            for value in args.spectral_taus.split(",")
            if value.strip()
        ),
        require_temporal=args.require_temporal,
        all_seeds_from_replicates=args.all_seeds_from_replicates,
    )
    report["summary_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    json_temporary = args.out_json.with_name(
        f".{args.out_json.name}.{os.getpid()}.tmp"
    )
    md_temporary = args.out_md.with_name(
        f".{args.out_md.name}.{os.getpid()}.tmp"
    )
    json_temporary.write_text(json.dumps(report, indent=2) + "\n")
    md_temporary.write_text(markdown_report(report))
    json_temporary.replace(args.out_json)
    md_temporary.replace(args.out_md)
    print(
        json.dumps(
            {
                "candidate": report["candidate"],
                "three_seed_consistent": report["three_seed_consistent"],
            }
        )
    )


if __name__ == "__main__":
    main()
