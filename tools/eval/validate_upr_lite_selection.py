#!/usr/bin/env python3
"""Validate the selected UPR-Lite candidate with paired block tests."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    WindowScores,
    load_acc_scores,
    load_metrics,
    load_temporal_metrics,
    paired_block_bootstrap,
    paired_block_score_test,
)
from tools.eval.physical_block_bootstrap import (
    compare_physical,
    load_physical_windows,
)
from tools.eval.spectral_block_bootstrap import (
    compare as compare_spectra,
)
from tools.eval.spectral_block_bootstrap import (
    load_windows,
)
from tools.eval.select_upr_lite_candidate import (
    load_field_metrics as load_selection_field_metrics,
)
from tools.eval.select_upr_lite_candidate import (
    load_spectral_metrics as load_selection_spectral_metrics,
)


DEFAULT_MODELS = (
    "upr_lite",
    "upr_lite_lap",
    "upr_lite_column",
    "upr_lite_continuous",
    "upr_lite_continuous_m",
    "upr_lite_implicit_global",
    "weatherbridge_ref",
    "atmvfi_ref",
)
TAU_GROUPS = {
    "all": np.asarray([1, 2, 3, 4, 5], dtype=np.int16),
    "seen": np.asarray([1, 3, 5], dtype=np.int16),
    "unseen": np.asarray([2, 4], dtype=np.int16),
}
SEASONS = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spectral_artifact_paths(
    root: Path,
    model: str,
    taus: set[int],
) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(root.glob(f"{model}_tau*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            tau = int(payload["tau"])
        if tau not in taus:
            continue
        if tau in paths:
            raise ValueError(f"{model}: duplicate spectral tau={tau}")
        paths[tau] = path
    if set(paths) != taus:
        raise FileNotFoundError(
            f"{model}: spectral artifact set mismatch; "
            f"expected={sorted(taus)}, found={sorted(paths)}"
        )
    return paths


def verify_selection_artifacts(
    selection: dict[str, Any],
    models: tuple[str, ...],
    root_2020: Path,
    root_2021: Path,
    spectra_root: Path,
    spectral_taus: set[int],
) -> dict[str, Any]:
    """Verify that bootstrap inputs still match the frozen selection inputs."""
    schema_version = selection.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 12
    ):
        raise ValueError("unsupported frozen selection schema")
    frozen = selection.get("paired_window_index_sha256")
    rule = selection.get("selection_rule")
    selection_models = selection.get("models")
    if not isinstance(frozen, dict) or not isinstance(rule, dict):
        raise ValueError("selection report lacks frozen artifact metadata")
    if not isinstance(selection_models, dict):
        raise ValueError("selection report lacks the selected model set")
    if set(models) != set(selection_models):
        raise ValueError(
            "validation model set differs from selection model set: "
            f"validation={sorted(models)}, "
            f"selection={sorted(selection_models)}"
        )
    required_frozen = {
        "field_2020",
        "spectral_2020",
        "spectral_grid",
        "spectral_channels",
        "field_input_provenance",
        "spectral_input_provenance",
        "field_dataset_2020",
        "spectral_dataset_2020",
    }
    if not required_frozen.issubset(frozen):
        raise ValueError("selection report has incomplete frozen hashes")
    cost_artifact = selection.get("cost_artifact")
    if not isinstance(cost_artifact, dict):
        raise ValueError("selection report lacks frozen cost artifact")
    cost_path = Path(str(cost_artifact.get("path", "")))
    cost_size = cost_artifact.get("size_bytes")
    if (
        isinstance(cost_size, bool)
        or not isinstance(cost_size, int)
        or cost_size < 0
        or not cost_path.is_file()
        or cost_path.stat().st_size != cost_size
        or _file_sha256(cost_path) != cost_artifact.get("sha256")
    ):
        raise ValueError("selection cost artifact changed after selection")
    cost_payload = json.loads(cost_path.read_text())
    cost_models = cost_payload.get("models")
    if not isinstance(cost_models, dict) or not set(models).issubset(
        cost_models
    ):
        raise ValueError("selection cost model set changed")
    frozen_cost_checkpoints = cost_artifact.get(
        "checkpoint_sha256_by_model"
    )
    if not isinstance(frozen_cost_checkpoints, dict):
        raise ValueError("selection lacks frozen cost checkpoint hashes")
    for model in models:
        expected_checkpoint = selection_models[model].get(
            "checkpoint_sha256"
        )
        current_checkpoint = cost_models[model].get("checkpoint_sha256")
        if (
            not expected_checkpoint
            or current_checkpoint != expected_checkpoint
            or frozen_cost_checkpoints.get(model) != expected_checkpoint
        ):
            raise ValueError(
                f"{model}: cost/selection checkpoint hash mismatch"
            )

    rule_taus = rule.get("spectral_taus")
    if isinstance(rule_taus, list) and spectral_taus != {
        int(value) for value in rule_taus
    }:
        raise ValueError(
            "validation spectral taus differ from selection spectral taus"
        )
    hf_ell_min = int(rule["hf_ell_min"])
    required_lmax = rule.get("required_lmax")
    expected_channels = rule.get("expected_spectral_channels")

    field_artifacts: dict[str, dict[str, dict[str, str]]] = {}
    field_provenance_hashes: set[str] = set()
    field_input_reference: dict[str, Any] | None = None
    for year, root in (("2020", root_2020), ("2021", root_2021)):
        expected_index = frozen.get(f"field_{year}")
        expected_dataset_hash = frozen.get(f"field_dataset_{year}")
        observed_indices: set[str] = set()
        observed_dataset_hashes: set[str] = set()
        field_artifacts[year] = {}
        for model in models:
            metrics_path = root / f"{model}.json"
            metrics = load_selection_field_metrics(metrics_path)
            if not metrics["evaluation_full_year"]:
                raise ValueError(f"{model}: {year} field metrics are not full-year")
            current_index = str(metrics["window_index_sha256"])
            observed_indices.add(current_index)
            if expected_index is not None and current_index != expected_index:
                raise ValueError(
                    f"{model}: {year} field window index changed after selection"
                )
            if (
                metrics["checkpoint_sha256"]
                != selection_models[model]["checkpoint_sha256"]
            ):
                raise ValueError(
                    f"{model}: {year} field checkpoint changed after selection"
                )
            input_provenance = metrics["evaluation_input_provenance"]
            provenance_hash = _canonical_sha256(input_provenance)
            if provenance_hash != frozen["field_input_provenance"]:
                raise ValueError(
                    f"{model}: {year} field input provenance changed after selection"
                )
            field_provenance_hashes.add(provenance_hash)
            if field_input_reference is None:
                field_input_reference = input_provenance
            dataset_provenance = metrics["evaluation_dataset_provenance"]
            if dataset_provenance.get("years") != [int(year)]:
                raise ValueError(
                    f"{model}: {year} field dataset provenance has wrong year"
                )
            dataset_hash = _canonical_sha256(
                dataset_provenance
            )
            observed_dataset_hashes.add(dataset_hash)
            if (
                expected_dataset_hash is not None
                and dataset_hash != expected_dataset_hash
            ):
                raise ValueError(
                    f"{model}: {year} field dataset provenance "
                    "changed after selection"
                )
            payload = json.loads(metrics_path.read_text())
            window_file = payload.get("window_metrics_file")
            if not window_file:
                raise ValueError(f"{metrics_path}: missing window_metrics_file")
            window_path = metrics_path.parent / str(window_file)
            field_artifacts[year][model] = {
                "metrics_json_sha256": _file_sha256(metrics_path),
                "window_npz_sha256": _file_sha256(window_path),
                "window_index_sha256": current_index,
                "input_provenance_sha256": provenance_hash,
                "dataset_provenance_sha256": dataset_hash,
            }
        if len(observed_indices) != 1:
            raise ValueError(
                f"{year} field window indices differ across models"
            )
        if len(observed_dataset_hashes) != 1:
            raise ValueError(
                f"{year} field dataset provenance differs across models"
            )
    if len(field_provenance_hashes) != 1:
        raise ValueError("field input provenance differs across validation inputs")
    if field_input_reference is None:
        raise ValueError("field input provenance is unavailable")
    normalization = selection.get("normalization_provenance")
    if (
        not isinstance(normalization, dict)
        or normalization.get("verified") is not True
    ):
        raise ValueError("selection lacks verified normalization provenance")
    manifest_path = Path(str(normalization.get("manifest_path", "")))
    if (
        not manifest_path.is_file()
        or _file_sha256(manifest_path)
        != normalization.get("manifest_sha256")
    ):
        raise ValueError("normalization manifest changed after selection")
    period = normalization.get("declared_period")
    if (
        not isinstance(period, list)
        or len(period) != 2
        or not all(isinstance(value, int) for value in period)
        or period[0] > period[1]
        or any(period[0] <= year <= period[1] for year in (2020, 2021))
    ):
        raise ValueError("normalization period overlaps evaluation")
    normalization_artifacts = normalization.get("artifacts")
    if not isinstance(normalization_artifacts, dict):
        raise ValueError("normalization artifact provenance is missing")
    for manifest_key, field_key in (
        ("pressure_level_stats", "pressure_level_stats"),
        ("surface_stats", "surface_stats"),
    ):
        manifest_artifact = normalization_artifacts.get(manifest_key)
        field_artifact = field_input_reference.get(field_key)
        if (
            not isinstance(manifest_artifact, dict)
            or not isinstance(field_artifact, dict)
            or manifest_artifact.get("sha256")
            != field_artifact.get("sha256")
        ):
            raise ValueError(
                f"{manifest_key}: normalization/evaluation hash mismatch"
            )

    spectral_artifacts: dict[str, dict[str, Any]] = {}
    spectral_provenance_hashes: set[str] = set()
    for model in models:
        metrics = load_selection_spectral_metrics(
            spectra_root,
            model,
            hf_ell_min,
            taus=spectral_taus,
            required_lmax=(
                int(required_lmax) if required_lmax is not None else None
            ),
            expected_channels=(
                int(expected_channels)
                if expected_channels is not None
                else None
            ),
        )
        checks = {
            "window_index_sha256": "spectral_2020",
            "spectral_grid_sha256": "spectral_grid",
            "spectral_channel_names_sha256": "spectral_channels",
        }
        for current_key, frozen_key in checks.items():
            if metrics[current_key] != frozen[frozen_key]:
                raise ValueError(
                    f"{model}: {current_key} changed after selection"
                )
        if (
            metrics["checkpoint_sha256"]
            != selection_models[model]["checkpoint_sha256"]
        ):
            raise ValueError(
                f"{model}: spectral checkpoint changed after selection"
            )
        provenance_hash = _canonical_sha256(
            metrics["evaluation_input_provenance"]
        )
        if provenance_hash != frozen["spectral_input_provenance"]:
            raise ValueError(
                f"{model}: spectral input provenance changed after selection"
            )
        spectral_provenance_hashes.add(provenance_hash)
        dataset_hash = _canonical_sha256(
            metrics["evaluation_dataset_provenance"]
        )
        if dataset_hash != frozen["spectral_dataset_2020"]:
            raise ValueError(
                f"{model}: spectral dataset provenance changed after selection"
            )
        paths = _spectral_artifact_paths(
            spectra_root,
            model,
            spectral_taus,
        )
        spectral_artifacts[model] = {
            "npz_sha256_by_tau": {
                str(tau): _file_sha256(path)
                for tau, path in sorted(paths.items())
            },
            "window_index_sha256": metrics["window_index_sha256"],
            "spectral_grid_sha256": metrics["spectral_grid_sha256"],
            "spectral_channel_names_sha256": (
                metrics["spectral_channel_names_sha256"]
            ),
            "input_provenance_sha256": provenance_hash,
            "dataset_provenance_sha256": dataset_hash,
        }
    if len(spectral_provenance_hashes) != 1:
        raise ValueError(
            "spectral input provenance differs across validation inputs"
        )
    return {
        "model_set_sha256": _canonical_sha256(sorted(models)),
        "cost_artifact_sha256": cost_artifact["sha256"],
        "field": field_artifacts,
        "spectral": spectral_artifacts,
    }


def field_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
    tau_groups: dict[str, np.ndarray] = TAU_GROUPS,
) -> dict:
    window_root = root / "window_metrics"
    left = load_metrics(window_root / f"{winner}.npz")
    competitors = {
        model: load_metrics(window_root / f"{model}.npz")
        for model in models
        if model != winner
    }
    competitors["bilinear"] = load_metrics(
        window_root / f"{winner}.npz",
        method="bilinear",
    )
    return {
        group: {
            name: paired_block_bootstrap(
                left,
                right,
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            for name, right in competitors.items()
        }
        for group, taus in tau_groups.items()
    }


def _window_months(metrics: WindowMetrics) -> np.ndarray:
    months = np.empty(metrics.t0.shape, dtype=np.int8)
    for year in np.unique(metrics.year):
        mask = metrics.year == year
        timestamps = (
            np.datetime64(f"{int(year):04d}-01-01T00", "h")
            + metrics.t0[mask].astype("timedelta64[h]")
        )
        months[mask] = (
            timestamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
        )
    return months


def _subset_metrics(
    metrics: WindowMetrics,
    mask: np.ndarray,
) -> WindowMetrics:
    if mask.shape != metrics.t0.shape or not mask.any():
        raise ValueError("seasonal subset must select at least one window")
    return WindowMetrics(
        year=metrics.year[mask],
        t0=metrics.t0[mask],
        tau=metrics.tau[mask],
        mse=metrics.mse[mask],
        channels=metrics.channels,
    )


def seasonal_field_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
) -> dict[str, dict[str, dict]]:
    """Compare unseen-hour RMSE inside each meteorological season."""
    window_root = root / "window_metrics"
    left_full = load_metrics(window_root / f"{winner}.npz")
    competitors = {
        model: load_metrics(window_root / f"{model}.npz")
        for model in models
        if model != winner
    }
    competitors["bilinear"] = load_metrics(
        window_root / f"{winner}.npz",
        method="bilinear",
    )
    months = _window_months(left_full)
    output: dict[str, dict[str, dict]] = {}
    for season, season_months in SEASONS.items():
        mask = np.isin(left_full.tau, taus) & np.isin(
            months,
            season_months,
        )
        left = _subset_metrics(left_full, mask)
        output[season] = {
            model: paired_block_bootstrap(
                left,
                _subset_metrics(right, mask),
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            for model, right in competitors.items()
        }
    return output


def _window_rmse_scores(metrics: WindowMetrics) -> WindowScores:
    return WindowScores(
        year=metrics.year,
        t0=metrics.t0,
        tau=metrics.tau,
        values=np.sqrt(
            np.maximum(metrics.mse.mean(axis=1, keepdims=True), 0.0)
        ),
        channels=("channel_rms_window",),
    )


def _subset_scores(
    scores: WindowScores,
    mask: np.ndarray,
) -> WindowScores:
    if mask.shape != scores.t0.shape or not mask.any():
        raise ValueError("extreme subset must select at least one window")
    return WindowScores(
        year=scores.year[mask],
        t0=scores.t0[mask],
        tau=scores.tau[mask],
        values=scores.values[mask],
        channels=scores.channels,
    )


def extreme_field_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    taus: np.ndarray,
    quantile: float,
    block_days: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Compare models on a common bilinear-defined hard-window subset."""
    if not 0.5 < quantile < 1.0:
        raise ValueError("extreme quantile must lie strictly between 0.5 and 1")
    window_root = root / "window_metrics"
    baseline = _window_rmse_scores(
        load_metrics(
            window_root / f"{winner}.npz",
            method="bilinear",
        )
    )
    baseline_scalar = baseline.values[:, 0]
    mask = np.zeros(baseline_scalar.shape, dtype=bool)
    thresholds: dict[str, float] = {}
    for tau in taus:
        tau_mask = baseline.tau == tau
        if not tau_mask.any():
            raise ValueError(f"no baseline windows for tau={int(tau)}")
        threshold = float(np.quantile(baseline_scalar[tau_mask], quantile))
        thresholds[str(int(tau))] = threshold
        mask |= tau_mask & (baseline_scalar >= threshold)

    left = _subset_scores(
        _window_rmse_scores(
            load_metrics(window_root / f"{winner}.npz")
        ),
        mask,
    )
    competitors = {
        model: _subset_scores(
            _window_rmse_scores(
                load_metrics(window_root / f"{model}.npz")
            ),
            mask,
        )
        for model in models
        if model != winner
    }
    competitors["bilinear"] = _subset_scores(baseline, mask)
    return {
        "subset_definition": "per-tau upper tail of bilinear window RMSE",
        "quantile": quantile,
        "threshold_by_tau": thresholds,
        "n_windows": int(mask.sum()),
        "comparisons": {
            model: paired_block_score_test(
                left,
                right,
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
                better="lower",
            )
            for model, right in competitors.items()
        },
    }


def acc_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
    tau_groups: dict[str, np.ndarray] = TAU_GROUPS,
) -> dict:
    window_root = root / "window_metrics"
    left = load_acc_scores(window_root / f"{winner}.npz")
    competitors = {
        model: load_acc_scores(window_root / f"{model}.npz")
        for model in models
        if model != winner
    }
    competitors["bilinear"] = load_acc_scores(
        window_root / f"{winner}.npz",
        method="bilinear",
    )
    return {
        group: {
            name: paired_block_score_test(
                left,
                right,
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
                better="higher",
            )
            for name, right in competitors.items()
        }
        for group, taus in tau_groups.items()
    }


def spectral_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
    taus: np.ndarray = TAU_GROUPS["all"],
) -> dict:
    output = {}
    for tau in taus:
        left = load_windows(root / f"{winner}_tau{tau}.npz")
        channel_indices = np.arange(len(left.channels))
        output[str(int(tau))] = {
            model: compare_spectra(
                left,
                load_windows(root / f"{model}_tau{tau}.npz"),
                channel_indices=channel_indices,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            for model in models
            if model != winner
        }
    return output


def physical_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
    tau_groups: dict[str, np.ndarray] = TAU_GROUPS,
) -> dict:
    window_root = root / "window_metrics"
    left = load_physical_windows(window_root / f"{winner}.npz")
    competitors = {
        model: load_physical_windows(window_root / f"{model}.npz")
        for model in models
        if model != winner
    }
    competitors["bilinear"] = load_physical_windows(
        window_root / f"{winner}.npz",
        method="bilinear",
    )
    return {
        group: {
            name: compare_physical(
                left,
                right,
                taus=taus,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            for name, right in competitors.items()
        }
        for group, taus in tau_groups.items()
    }


def temporal_comparisons(
    winner: str,
    models: tuple[str, ...],
    root: Path,
    *,
    block_days: int,
    draws: int,
    seed: int,
) -> dict:
    """Compare complete anchor-window temporal-curvature reconstruction."""
    window_root = root / "window_metrics"
    left = load_temporal_metrics(window_root / f"{winner}.npz")
    competitors = {
        model: load_temporal_metrics(window_root / f"{model}.npz")
        for model in models
        if model != winner
    }
    competitors["bilinear"] = load_temporal_metrics(
        window_root / f"{winner}.npz",
        method="bilinear",
    )
    return {
        name: paired_block_bootstrap(
            left,
            right,
            taus=np.asarray([0], dtype=np.int16),
            block_days=block_days,
            draws=draws,
            seed=seed,
        )
        for name, right in competitors.items()
    }


def _annotate_holm(
    results: list[dict],
    *,
    output_key: str,
) -> None:
    """Attach Holm-adjusted p-values to one pre-specified test family."""
    if not results:
        return
    p_values = np.asarray(
        [result["p_paired_block_permutation"] for result in results],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(p_values))
        or np.any(p_values < 0.0)
        or np.any(p_values > 1.0)
    ):
        raise ValueError("Holm family contains an invalid p-value")
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    n_tests = len(results)
    for rank, index in enumerate(order):
        running_max = max(
            running_max,
            (n_tests - rank) * float(p_values[index]),
        )
        adjusted[index] = min(1.0, running_max)
    for result, value in zip(results, adjusted):
        result[output_key] = float(value)


def _physical_dominates(result: dict, *, p_key: str) -> bool:
    significant_wins = 0
    significant_losses = 0
    for diagnostic in result["diagnostics"].values():
        delta = diagnostic["delta_left_minus_right"]
        significant = diagnostic[p_key] < 0.05
        significant_wins += int(significant and delta > 0.0)
        significant_losses += int(significant and delta < 0.0)
    return significant_wins >= 2 and significant_losses == 0


def summarize(
    field: dict,
    spectral: dict,
    physical: dict,
    acc: dict | None = None,
    seasonal: dict | None = None,
    extreme: dict | None = None,
    temporal: dict | None = None,
    ood_robust_skill: dict[str, float] | None = None,
    *,
    selection_quality_gate_passed: bool = True,
) -> dict:
    robust_skill_values: dict[str, float] | None = None
    absolute_ood_robust_skill_passed = True
    if ood_robust_skill is not None:
        required = {
            "tail_skill_2021_unseen",
            "worst_season_skill_2021_unseen",
        }
        if set(ood_robust_skill) != required:
            raise ValueError("OOD robust-skill keys do not match protocol")
        robust_skill_values = {
            key: float(ood_robust_skill[key])
            for key in sorted(required)
        }
        if not all(
            np.isfinite(value)
            for value in robust_skill_values.values()
        ):
            raise ValueError("OOD robust skills must be finite")
        absolute_ood_robust_skill_passed = all(
            value > 0.0 for value in robust_skill_values.values()
        )

    critical_scopes = (
        field["2020"]["unseen"],
        field["2021"]["unseen"],
    )
    bilinear_results = [scope["bilinear"] for scope in critical_scopes]
    _annotate_holm(
        bilinear_results,
        output_key="p_holm_unseen_bilinear_year_family",
    )
    beats_bilinear_both_years = all(
        result["delta_left_minus_right"] < 0.0
        and result["p_holm_unseen_bilinear_year_family"] < 0.05
        for result in bilinear_results
    )
    ood_bilinear = field["2021"]["unseen"]["bilinear"]
    beats_bilinear_ood = (
        ood_bilinear["delta_left_minus_right"] < 0.0
        and ood_bilinear["p_holm_unseen_bilinear_year_family"] < 0.05
    )

    ood_field_results = [
        result
        for model, result in field["2021"]["unseen"].items()
        if model != "bilinear"
    ]
    _annotate_holm(
        ood_field_results,
        output_key="p_holm_2021_unseen_field_family",
    )
    field_challenger_wins = []
    for model, result in field["2021"]["unseen"].items():
        if model == "bilinear":
            continue
        if (
            result["delta_left_minus_right"] > 0.0
            and result["p_holm_2021_unseen_field_family"] < 0.05
        ):
            field_challenger_wins.append(
                {
                    "year": "2021",
                    "tau_group": "unseen",
                    "model": model,
                    "relative_delta_pct": result["relative_delta_pct"],
                    "p_raw": result["p_paired_block_permutation"],
                    "p_holm": result[
                        "p_holm_2021_unseen_field_family"
                    ],
                }
            )

    individual_hour_results = []
    for group, comparisons in field["2021"].items():
        if not group.startswith("h"):
            continue
        individual_hour_results.extend(
            result
            for model, result in comparisons.items()
            if model != "bilinear"
        )
    _annotate_holm(
        individual_hour_results,
        output_key="p_holm_2021_individual_hour_field_family",
    )
    individual_hour_regressions = []
    for group, comparisons in field["2021"].items():
        if not group.startswith("h"):
            continue
        for model, result in comparisons.items():
            if model == "bilinear":
                continue
            p_holm = result[
                "p_holm_2021_individual_hour_field_family"
            ]
            if result["delta_left_minus_right"] > 0.0 and p_holm < 0.05:
                individual_hour_regressions.append(
                    {
                        "year": "2021",
                        "tau_group": group,
                        "tau": int(group[1:]),
                        "model": model,
                        "relative_delta_pct": result[
                            "relative_delta_pct"
                        ],
                        "p_raw": result[
                            "p_paired_block_permutation"
                        ],
                        "p_holm": p_holm,
                    }
                )

    spectral_energy_results = []
    spectral_shape_results = []
    spectral_coherence_results = []
    for comparisons in spectral.values():
        for result in comparisons.values():
            spectral_energy_results.append(result["energy_log_error"])
            spectral_shape_results.append(result["shape_log_error"])
            spectral_coherence_results.append(result["coherence"])
    _annotate_holm(
        spectral_energy_results,
        output_key="p_holm_selection_spectral_energy_family",
    )
    _annotate_holm(
        spectral_shape_results,
        output_key="p_holm_selection_spectral_shape_family",
    )
    _annotate_holm(
        spectral_coherence_results,
        output_key="p_holm_selection_spectral_coherence_family",
    )
    spectral_dominance = []
    spectral_regressions = []
    for tau, comparisons in spectral.items():
        for model, result in comparisons.items():
            energy = result["energy_log_error"]
            shape = result["shape_log_error"]
            coherence = result["coherence"]
            challenger_wins = {
                "energy_log_error": (
                    energy["delta_left_minus_right"] > 0.0
                    and energy[
                        "p_holm_selection_spectral_energy_family"
                    ] < 0.05
                ),
                "shape_log_error": (
                    shape["delta_left_minus_right"] > 0.0
                    and shape[
                        "p_holm_selection_spectral_shape_family"
                    ] < 0.05
                ),
                "coherence": (
                    coherence["delta_left_minus_right"] < 0.0
                    and coherence[
                        "p_holm_selection_spectral_coherence_family"
                    ] < 0.05
                ),
            }
            challenger_metric_wins = sum(challenger_wins.values())
            for metric, significant in challenger_wins.items():
                if significant:
                    spectral_regressions.append(
                        {
                            "tau": int(tau),
                            "model": model,
                            "metric": metric,
                        }
                    )
            winner_metric_wins = sum(
                (
                    energy["delta_left_minus_right"] < 0.0
                    and energy[
                        "p_holm_selection_spectral_energy_family"
                    ] < 0.05,
                    shape["delta_left_minus_right"] < 0.0
                    and shape[
                        "p_holm_selection_spectral_shape_family"
                    ] < 0.05,
                    coherence["delta_left_minus_right"] > 0.0
                    and coherence[
                        "p_holm_selection_spectral_coherence_family"
                    ] < 0.05,
                )
            )
            if challenger_metric_wins >= 2 and winner_metric_wins == 0:
                spectral_dominance.append(
                    {
                        "tau": int(tau),
                        "model": model,
                        "p_energy_holm": energy[
                            "p_holm_selection_spectral_energy_family"
                        ],
                        "p_shape_holm": shape[
                            "p_holm_selection_spectral_shape_family"
                        ],
                        "p_coherence_holm": coherence[
                            "p_holm_selection_spectral_coherence_family"
                        ],
                    }
                )

    ood_physical_results = []
    for model, result in physical["2021"]["unseen"].items():
        if model != "bilinear":
            ood_physical_results.extend(result["diagnostics"].values())
    _annotate_holm(
        ood_physical_results,
        output_key="p_holm_2021_unseen_physical_family",
    )
    physical_dominance = []
    physical_regressions = []
    for model, result in physical["2021"]["unseen"].items():
        if model == "bilinear":
            continue
        if _physical_dominates(
            result,
            p_key="p_holm_2021_unseen_physical_family",
        ):
            physical_dominance.append(
                {
                    "year": "2021",
                    "tau_group": "unseen",
                    "model": model,
                }
            )
        for diagnostic_name, diagnostic in result["diagnostics"].items():
            p_holm = diagnostic[
                "p_holm_2021_unseen_physical_family"
            ]
            if (
                diagnostic["delta_left_minus_right"] > 0.0
                and p_holm < 0.05
            ):
                physical_regressions.append(
                    {
                        "year": "2021",
                        "tau_group": "unseen",
                        "model": model,
                        "diagnostic": diagnostic_name,
                        "relative_delta_pct": diagnostic[
                            "relative_delta_pct"
                        ],
                        "p_raw": diagnostic[
                            "p_paired_block_permutation"
                        ],
                        "p_holm": p_holm,
                    }
                )
    acc_challenger_wins = []
    if acc is not None:
        ood_acc_results = [
            result
            for model, result in acc["2021"]["unseen"].items()
            if model != "bilinear"
        ]
        _annotate_holm(
            ood_acc_results,
            output_key="p_holm_2021_unseen_acc_family",
        )
        for model, result in acc["2021"]["unseen"].items():
            if model == "bilinear":
                continue
            if (
                result["delta_left_minus_right"] < 0.0
                and result["p_holm_2021_unseen_acc_family"] < 0.05
            ):
                acc_challenger_wins.append(
                    {
                        "year": "2021",
                        "tau_group": "unseen",
                        "model": model,
                        "relative_delta_pct": result["relative_delta_pct"],
                        "p_raw": result["p_paired_block_permutation"],
                        "p_holm": result[
                            "p_holm_2021_unseen_acc_family"
                        ],
                    }
                )
    seasonal_regressions = []
    if seasonal is not None:
        ood_seasonal_results = [
            result
            for comparisons in seasonal["2021"].values()
            for result in comparisons.values()
        ]
        _annotate_holm(
            ood_seasonal_results,
            output_key="p_holm_2021_unseen_seasonal_family",
        )
        for season, comparisons in seasonal["2021"].items():
            for model, result in comparisons.items():
                p_holm = result[
                    "p_holm_2021_unseen_seasonal_family"
                ]
                if (
                    result["delta_left_minus_right"] > 0.0
                    and p_holm < 0.05
                ):
                    seasonal_regressions.append(
                        {
                            "year": "2021",
                            "tau_group": "unseen",
                            "season": season,
                            "model": model,
                            "relative_delta_pct": result[
                                "relative_delta_pct"
                            ],
                            "p_raw": result[
                                "p_paired_block_permutation"
                            ],
                            "p_holm": p_holm,
                        }
                    )
    extreme_regressions = []
    if extreme is not None:
        ood_extreme_results = list(
            extreme["2021"]["comparisons"].values()
        )
        _annotate_holm(
            ood_extreme_results,
            output_key="p_holm_2021_unseen_extreme_family",
        )
        for model, result in extreme["2021"]["comparisons"].items():
            p_holm = result["p_holm_2021_unseen_extreme_family"]
            if (
                result["delta_left_minus_right"] > 0.0
                and p_holm < 0.05
            ):
                extreme_regressions.append(
                    {
                        "year": "2021",
                        "tau_group": "unseen",
                        "subset": "bilinear_hardest_5pct_per_tau",
                        "model": model,
                        "relative_delta_pct": result[
                            "relative_delta_pct"
                        ],
                        "p_raw": result[
                            "p_paired_block_permutation"
                        ],
                        "p_holm": p_holm,
                    }
                )
    temporal_regressions = []
    if temporal is not None:
        ood_temporal_results = list(temporal["2021"].values())
        _annotate_holm(
            ood_temporal_results,
            output_key="p_holm_2021_temporal_curvature_family",
        )
        for model, result in temporal["2021"].items():
            p_holm = result[
                "p_holm_2021_temporal_curvature_family"
            ]
            if (
                result["delta_left_minus_right"] > 0.0
                and p_holm < 0.05
            ):
                temporal_regressions.append(
                    {
                        "year": "2021",
                        "model": model,
                        "metric": "temporal_curvature_rmse",
                        "relative_delta_pct": result[
                            "relative_delta_pct"
                        ],
                        "p_raw": result[
                            "p_paired_block_permutation"
                        ],
                        "p_holm": p_holm,
                    }
                )
    selection_confirmed = (
        selection_quality_gate_passed
        and beats_bilinear_both_years
        and not field_challenger_wins
        and not physical_dominance
    )
    cross_metric_generalization_confirmed = (
        selection_confirmed
        and not individual_hour_regressions
        and not spectral_regressions
        and not acc_challenger_wins
        and not physical_regressions
        and not seasonal_regressions
        and not extreme_regressions
        and not temporal_regressions
        and absolute_ood_robust_skill_passed
    )
    return {
        "multiplicity_correction": "Holm within each pre-specified family",
        "unseen_bilinear_year_family": {
            year: {
                "delta_left_minus_right": result[
                    "delta_left_minus_right"
                ],
                "p_raw": result["p_paired_block_permutation"],
                "p_holm": result[
                    "p_holm_unseen_bilinear_year_family"
                ],
            }
            for year, result in zip(("2020", "2021"), bilinear_results)
        },
        "beats_bilinear_on_2020_and_2021_unseen": (
            beats_bilinear_both_years
        ),
        "beats_bilinear_on_2021_unseen": beats_bilinear_ood,
        "selection_quality_gate_passed": selection_quality_gate_passed,
        "significant_field_challenger_wins": field_challenger_wins,
        "significant_individual_hour_field_regressions": (
            individual_hour_regressions
        ),
        "significant_spectral_dominance": spectral_dominance,
        "selection_set_significant_spectral_dominance": spectral_dominance,
        "significant_spectral_regressions": spectral_regressions,
        "significant_physical_dominance": physical_dominance,
        "significant_physical_regressions": physical_regressions,
        "significant_acc_challenger_wins": acc_challenger_wins,
        "significant_seasonal_regressions": seasonal_regressions,
        "significant_extreme_regressions": extreme_regressions,
        "significant_temporal_curvature_regressions": (
            temporal_regressions
        ),
        "ood_robust_skill": robust_skill_values,
        "absolute_ood_robust_skill_passed": (
            absolute_ood_robust_skill_passed
        ),
        "selection_confirmed": selection_confirmed,
        "cross_metric_generalization_confirmed": (
            cross_metric_generalization_confirmed
        ),
        "confirmation_policy": (
            "Significant unseen-hour improvement over bilinear in both 2020 "
            "and frozen 2021 after Holm correction across the two years, plus "
            "frozen 2021 field and physical dominance "
            "tests and a passed 2020 selection quality gate determine "
            "selection_confirmed. Frozen 2021 unseen "
            "mean-window ACC and 2020 HF-energy, HF-shape, and coherence "
            "tests are confirmation "
            "diagnostics. The stricter "
            "cross_metric_generalization_confirmed flag additionally "
            "requires no Holm-significant individual-hour field regression, "
            "physical regression, ACC challenger win, seasonal unseen-hour "
            "regression, temporal-curvature regression, or individual "
            "spectral regression, and no "
            "regression on the common "
            "bilinear-defined hardest 5% of OOD unseen-hour windows. "
            "Model-specific OOD CVaR95 and worst-season skill must also both "
            "remain positive relative to paired linear interpolation. It "
            "does not alter the prespecified winner."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("metrics/upr_lite_candidate_selection.json"),
    )
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
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
        "--spectra-root",
        type=Path,
        default=Path("metrics/upr_lite_screen_spectra_6h_2020"),
    )
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--all-tau", default="1,2,3,4,5")
    parser.add_argument("--seen-tau", default="1,3,5")
    parser.add_argument("--unseen-tau", default="2,4")
    parser.add_argument(
        "--spectral-tau",
        default="",
        help="Comma-separated spectral taus; defaults to --all-tau.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_lite_candidate_validation.json"),
    )
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    winner = str(selection["winner"])
    gate_relaxed = selection.get("selection_rule", {}).get(
        "gate_relaxed_because_no_model_passed"
    )
    if not isinstance(gate_relaxed, bool):
        raise ValueError(
            "selection report lacks a boolean 2020 quality-gate result"
        )
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    parse_taus = lambda value: np.asarray(  # noqa: E731
        [int(item) for item in value.split(",") if item.strip()],
        dtype=np.int16,
    )
    tau_groups = {
        "all": parse_taus(args.all_tau),
        "seen": parse_taus(args.seen_tau),
        "unseen": parse_taus(args.unseen_tau),
    }
    tau_groups.update(
        {
            f"h{int(tau)}": np.asarray([tau], dtype=np.int16)
            for tau in tau_groups["all"]
        }
    )
    spectral_taus = (
        parse_taus(args.spectral_tau)
        if args.spectral_tau
        else tau_groups["all"]
    )
    artifact_preflight = verify_selection_artifacts(
        selection,
        models,
        args.root_2020,
        args.root_2021,
        args.spectra_root,
        {int(value) for value in spectral_taus},
    )
    field = {
        "2020": field_comparisons(
            winner,
            models,
            args.root_2020,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
        "2021": field_comparisons(
            winner,
            models,
            args.root_2021,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
    }
    acc = {
        "2020": acc_comparisons(
            winner,
            models,
            args.root_2020,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
        "2021": acc_comparisons(
            winner,
            models,
            args.root_2021,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
    }
    spectral = spectral_comparisons(
        winner,
        models,
        args.spectra_root,
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
        taus=spectral_taus,
    )
    physical = {
        "2020": physical_comparisons(
            winner,
            models,
            args.root_2020,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
        "2021": physical_comparisons(
            winner,
            models,
            args.root_2021,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
            tau_groups=tau_groups,
        ),
    }
    seasonal = {
        "2020": seasonal_field_comparisons(
            winner,
            models,
            args.root_2020,
            taus=tau_groups["unseen"],
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
        "2021": seasonal_field_comparisons(
            winner,
            models,
            args.root_2021,
            taus=tau_groups["unseen"],
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
    }
    extreme = {
        "2020": extreme_field_comparisons(
            winner,
            models,
            args.root_2020,
            taus=tau_groups["unseen"],
            quantile=0.95,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
        "2021": extreme_field_comparisons(
            winner,
            models,
            args.root_2021,
            taus=tau_groups["unseen"],
            quantile=0.95,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
    }
    temporal = {
        "2020": temporal_comparisons(
            winner,
            models,
            args.root_2020,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
        "2021": temporal_comparisons(
            winner,
            models,
            args.root_2021,
            block_days=args.block_days,
            draws=args.draws,
            seed=args.seed,
        ),
    }
    winner_ood_metrics = load_selection_field_metrics(
        args.root_2021 / f"{winner}.json"
    )
    ood_robust_skill = {
        "tail_skill_2021_unseen": winner_ood_metrics[
            "unseen_cvar95_skill"
        ],
        "worst_season_skill_2021_unseen": winner_ood_metrics[
            "unseen_worst_season_skill"
        ],
    }
    output = {
        "schema_version": 11,
        "validation_code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "selection_report_sha256": _file_sha256(args.selection),
        "validated_artifacts": artifact_preflight,
        "validation_dependency_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "paired_block_bootstrap.py",
                "spectral_block_bootstrap.py",
                "physical_block_bootstrap.py",
            )
        },
        "winner": winner,
        "field": field,
        "acc": acc,
        "spectral": spectral,
        "physical": physical,
        "seasonal": seasonal,
        "extreme": extreme,
        "temporal": temporal,
        "summary": summarize(
            field,
            spectral,
            physical,
            acc,
            seasonal,
            extreme,
            temporal,
            ood_robust_skill,
            selection_quality_gate_passed=not gate_relaxed,
        ),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.out_json.with_name(f".{args.out_json.name}.tmp")
    temporary_path.write_text(json.dumps(output, indent=2) + "\n")
    temporary_path.replace(args.out_json)
    print(json.dumps(output["summary"]))


if __name__ == "__main__":
    main()
