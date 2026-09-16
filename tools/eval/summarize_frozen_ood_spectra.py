#!/usr/bin/env python3
"""Validate a frozen winner on independent OOD spectral windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.select_upr_lite_candidate import (
    load_spectral_metrics as load_selection_spectral_metrics,
)
from tools.eval.spectral_block_bootstrap import (
    compare as compare_spectra,
)
from tools.eval.spectral_block_bootstrap import load_windows
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


METRICS = {
    "energy_log_error": "lower",
    "shape_log_error": "lower",
    "coherence": "higher",
    "signed_cospectrum": "higher",
}
CANONICAL_CHANNEL_HASH = hashlib.sha256(
    "\n".join(CANONICAL_24_CHANNELS).encode("utf-8")
).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _annotate_holm(results: list[dict[str, Any]]) -> None:
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
        raise ValueError("OOD spectral family contains an invalid p-value")
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(
            running_max,
            (len(results) - rank) * float(p_values[index]),
        )
        adjusted[index] = min(1.0, running_max)
    for result, value in zip(results, adjusted):
        result["p_holm_ood_spectral_family"] = float(value)


def build_report(
    selection: dict[str, Any],
    spectra_root: Path,
    models: tuple[str, ...],
    *,
    delta_t_hours: int,
    spectral_taus: tuple[int, ...],
    hf_ell_min: int = 180,
    block_days: int = 7,
    draws: int = 5000,
    seed: int = 2027,
) -> dict[str, Any]:
    schema_version = selection.get("schema_version")
    selection_models = selection.get("models")
    winner = selection.get("winner")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 12
        or selection.get("ood_attached_at_selection_time") is not False
        or not isinstance(selection_models, dict)
        or not isinstance(winner, str)
        or not winner
    ):
        raise ValueError("invalid frozen selection")
    if (
        len(models) < 2
        or len(set(models)) != len(models)
        or set(models) != set(selection_models)
        or winner not in models
    ):
        raise ValueError("OOD spectral model set differs from selection")
    if (
        delta_t_hours <= 0
        or not spectral_taus
        or len(set(spectral_taus)) != len(spectral_taus)
        or any(tau <= 0 or tau >= delta_t_hours for tau in spectral_taus)
    ):
        raise ValueError("invalid OOD spectral horizon or tau set")

    expected_taus = set(spectral_taus)
    index_hashes: set[str] = set()
    grid_hashes: set[str] = set()
    channel_hashes: set[str] = set()
    input_hashes: set[str] = set()
    dataset_hashes: set[str] = set()
    artifact_hashes: dict[str, dict[str, str]] = {}
    for model in models:
        metrics = load_selection_spectral_metrics(
            spectra_root,
            model,
            hf_ell_min,
            taus=expected_taus,
            required_lmax=359,
            expected_channels=24,
        )
        if (
            metrics["checkpoint_sha256"]
            != selection_models[model].get("checkpoint_sha256")
        ):
            raise ValueError(f"{model}: checkpoint differs from selection")
        if (
            metrics["spectral_channel_names_sha256"]
            != CANONICAL_CHANNEL_HASH
        ):
            raise ValueError(f"{model}: non-canonical spectral fields")
        dataset = metrics["evaluation_dataset_provenance"]
        if dataset.get("years") != [2021]:
            raise ValueError(f"{model}: expected independent 2021 spectrum")
        index_hashes.add(str(metrics["window_index_sha256"]))
        grid_hashes.add(str(metrics["spectral_grid_sha256"]))
        channel_hashes.add(str(metrics["spectral_channel_names_sha256"]))
        input_hashes.add(
            _canonical_sha256(metrics["evaluation_input_provenance"])
        )
        dataset_hashes.add(_canonical_sha256(dataset))
        artifact_hashes[model] = {
            str(tau): _file_sha256(
                spectra_root / f"{model}_tau{tau}.npz"
            )
            for tau in spectral_taus
        }
    for label, values in (
        ("window index", index_hashes),
        ("spectral grid", grid_hashes),
        ("channel order", channel_hashes),
        ("input provenance", input_hashes),
        ("dataset provenance", dataset_hashes),
    ):
        if len(values) != 1:
            raise ValueError(f"OOD spectral {label} differs across models")

    comparisons: dict[str, dict[str, Any]] = {}
    for tau in spectral_taus:
        left = load_windows(spectra_root / f"{winner}_tau{tau}.npz")
        if left.channels != CANONICAL_24_CHANNELS:
            raise ValueError("OOD spectrum does not contain canonical fields")
        channel_indices = np.arange(len(left.channels))
        comparisons[str(tau)] = {
            model: compare_spectra(
                left,
                load_windows(spectra_root / f"{model}_tau{tau}.npz"),
                channel_indices=channel_indices,
                block_days=block_days,
                draws=draws,
                seed=seed,
            )
            for model in models
            if model != winner
        }

    for metric in METRICS:
        _annotate_holm(
            [
                comparison[metric]
                for tau_results in comparisons.values()
                for comparison in tau_results.values()
            ]
        )
    regressions = []
    for tau, tau_results in comparisons.items():
        for challenger, comparison in tau_results.items():
            for metric, better in METRICS.items():
                result = comparison[metric]
                delta = float(result["delta_left_minus_right"])
                worse = delta > 0.0 if better == "lower" else delta < 0.0
                if (
                    worse
                    and result["p_holm_ood_spectral_family"] < 0.05
                ):
                    regressions.append(
                        {
                            "tau": int(tau),
                            "challenger": challenger,
                            "metric": metric,
                            "delta_winner_minus_challenger": delta,
                            "p_holm": result[
                                "p_holm_ood_spectral_family"
                            ],
                        }
                    )
    return {
        "schema_version": 1,
        "winner": winner,
        "models": list(models),
        "protocol": {
            "delta_t_hours": delta_t_hours,
            "confirmation_year": 2021,
            "used_for_model_selection": False,
            "spectral_taus": list(spectral_taus),
            "hf_ell_min": hf_ell_min,
            "lmax": 359,
            "channels": 24,
            "block_days": block_days,
            "draws": draws,
            "seed": seed,
            "multiple_testing": (
                "Holm within each spectral metric across all "
                "tau-by-challenger comparisons"
            ),
        },
        "paired_window_index_sha256": next(iter(index_hashes)),
        "spectral_grid_sha256": next(iter(grid_hashes)),
        "spectral_channel_names_sha256": next(iter(channel_hashes)),
        "evaluation_input_provenance_sha256": next(iter(input_hashes)),
        "evaluation_dataset_provenance_sha256": next(
            iter(dataset_hashes)
        ),
        "artifact_sha256": artifact_hashes,
        "comparisons": comparisons,
        "significant_ood_spectral_regressions": regressions,
        "no_significant_ood_spectral_regression": not regressions,
        "note": (
            "The 2021 spectrum is a post-selection confirmation and cannot "
            "change the frozen 2020 winner."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--spectra-root", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--delta-t-hours", type=int, required=True)
    parser.add_argument("--spectral-taus", required=True)
    parser.add_argument("--hf-ell-min", type=int, default=180)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(
        json.loads(args.selection.read_text()),
        args.spectra_root,
        tuple(value for value in args.models.split(",") if value),
        delta_t_hours=args.delta_t_hours,
        spectral_taus=tuple(
            int(value)
            for value in args.spectral_taus.split(",")
            if value
        ),
        hf_ell_min=args.hf_ell_min,
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
    )
    report["selection_sha256"] = _file_sha256(args.selection)
    report["summary_code_sha256"] = _file_sha256(Path(__file__))
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_name(
        f".{args.out_json.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.out_json)
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "no_significant_ood_spectral_regression": report[
                    "no_significant_ood_spectral_regression"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
