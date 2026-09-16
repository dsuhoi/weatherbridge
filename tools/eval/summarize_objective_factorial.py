#!/usr/bin/env python3
"""Summarize the paired UPR/Flow x high-pass objective factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    aggregate_rmse,
    assert_paired,
    load_metrics,
)
from tools.eval.select_upr_lite_candidate import load_field_metrics


CELL_ORDER = ("upr_hf", "upr_nohf", "flow_hf", "flow_nohf")
DEFAULT_MODELS = {
    "upr_hf": "upr_implicit_global_14m",
    "upr_nohf": "upr_implicit_global_14m_nohf",
    "flow_hf": "flow_pp3_hf",
    "flow_nohf": "weatherbridge_ref",
}


def _validate_run_contract(
    selection: dict[str, Any],
    groups: dict[str, np.ndarray],
) -> int:
    schema_version = selection.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 12
        or selection.get("ood_attached_at_selection_time") is not False
    ):
        raise ValueError("unsupported frozen selection schema")
    if (
        set(int(value) for value in groups.get("all", ()))
        != {1, 2, 3, 4, 5}
        or set(int(value) for value in groups.get("seen", ()))
        != {1, 3, 5}
        or set(int(value) for value in groups.get("unseen", ()))
        != {2, 4}
    ):
        raise ValueError(
            "objective factorial requires all/seen/unseen tau sets "
            "{1,2,3,4,5}/{1,3,5}/{2,4}"
        )
    return schema_version


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_taus(value: str) -> np.ndarray:
    taus = np.asarray(
        sorted({int(item) for item in value.split(",") if item}),
        dtype=np.int16,
    )
    if taus.size == 0 or np.any(taus <= 0):
        raise argparse.ArgumentTypeError("taus must be positive integers")
    return taus


def _effect_coefficients(log_scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    architecture = 0.5 * (
        log_scores["upr_hf"]
        - log_scores["flow_hf"]
        + log_scores["upr_nohf"]
        - log_scores["flow_nohf"]
    )
    highpass = 0.5 * (
        log_scores["upr_hf"]
        - log_scores["upr_nohf"]
        + log_scores["flow_hf"]
        - log_scores["flow_nohf"]
    )
    interaction = (
        log_scores["upr_hf"]
        - log_scores["upr_nohf"]
        - log_scores["flow_hf"]
        + log_scores["flow_nohf"]
    )
    return {
        "architecture": architecture,
        "highpass": highpass,
        "interaction": interaction,
    }


def _effect_payload(point: float, draws: np.ndarray) -> dict[str, Any]:
    ci_log = np.percentile(draws, [2.5, 50.0, 97.5])
    return {
        "log_effect": float(point),
        "rmse_multiplier": float(np.exp(point)),
        "relative_effect_pct": float(100.0 * np.expm1(point)),
        "log_effect_ci95": ci_log.tolist(),
        "relative_effect_pct_ci95": (
            100.0 * np.expm1(ci_log)
        ).tolist(),
        "bootstrap_probability_below_zero": float(np.mean(draws < 0.0)),
    }


def _channel_rmse(
    metrics: WindowMetrics,
    taus: np.ndarray,
) -> np.ndarray:
    values = []
    for tau in taus:
        mask = metrics.tau == tau
        if not mask.any():
            raise ValueError(f"tau={int(tau)} has no windows")
        values.append(np.sqrt(metrics.mse[mask].mean(axis=0)))
    result = np.mean(values, axis=0)
    if np.any(result <= 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("factorial RMSE values must be finite and positive")
    return result


def factorial_block_bootstrap(
    cells: dict[str, WindowMetrics],
    *,
    taus: np.ndarray,
    block_days: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate paired log-RMSE main effects and their interaction."""
    if set(cells) != set(CELL_ORDER):
        raise ValueError(f"factorial cells must be exactly {CELL_ORDER}")
    if block_days <= 0 or draws <= 0:
        raise ValueError("block_days and draws must be positive")

    reference = cells[CELL_ORDER[0]]
    for cell in CELL_ORDER[1:]:
        assert_paired(reference, cells[cell])
    tau_list = np.asarray(sorted({int(tau) for tau in taus}), dtype=np.int16)
    missing = [
        int(tau)
        for tau in tau_list
        if not np.any(reference.tau == tau)
    ]
    if tau_list.size == 0 or missing:
        raise ValueError(f"requested taus have no windows: {missing}")

    selected = np.isin(reference.tau, tau_list)
    year = reference.year[selected].astype(np.int64)
    t0 = reference.t0[selected].astype(np.int64)
    tau_values = reference.tau[selected]
    block_id = year * 100_000 + t0 // (24 * block_days)
    blocks = np.unique(block_id)
    if len(blocks) < 2:
        raise ValueError("factorial bootstrap requires at least two blocks")

    n_channels = len(reference.channels)
    counts = np.zeros((len(tau_list), len(blocks)), dtype=np.float64)
    sums = {
        cell: np.zeros(
            (len(tau_list), len(blocks), n_channels),
            dtype=np.float64,
        )
        for cell in CELL_ORDER
    }
    for tau_index, tau in enumerate(tau_list):
        for block_index, block in enumerate(blocks):
            mask = (tau_values == tau) & (block_id == block)
            if not mask.any():
                continue
            counts[tau_index, block_index] = int(mask.sum())
            for cell in CELL_ORDER:
                sums[cell][tau_index, block_index] = (
                    cells[cell].mse[selected][mask].sum(axis=0)
                )

    rng = np.random.default_rng(seed)
    block_weights = rng.multinomial(
        len(blocks),
        np.full(len(blocks), 1.0 / len(blocks)),
        size=draws,
    ).astype(np.float64)
    score_draws = {
        cell: np.empty((draws, len(tau_list)), dtype=np.float64)
        for cell in CELL_ORDER
    }
    for tau_index, tau in enumerate(tau_list):
        sample_count = block_weights @ counts[tau_index]
        if np.any(sample_count <= 0):
            raise ValueError(
                f"bootstrap draw contains no windows for tau={int(tau)}"
            )
        for cell in CELL_ORDER:
            mean_mse = (
                block_weights @ sums[cell][tau_index]
            ) / sample_count[:, None]
            score_draws[cell][:, tau_index] = np.sqrt(mean_mse).mean(axis=1)

    point_scores = {
        cell: aggregate_rmse(metrics, tau_list)
        for cell, metrics in cells.items()
    }
    if any(
        not np.isfinite(score) or score <= 0.0
        for score in point_scores.values()
    ):
        raise ValueError("factorial RMSE values must be finite and positive")
    point_effects = _effect_coefficients(
        {cell: np.log(score) for cell, score in point_scores.items()}
    )
    draw_effects = _effect_coefficients(
        {
            cell: np.log(values.mean(axis=1))
            for cell, values in score_draws.items()
        }
    )

    effects = {
        name: _effect_payload(float(point_effects[name]), draw_effects[name])
        for name in point_effects
    }
    effects["architecture"]["contrast"] = "UPR minus Flow"
    effects["highpass"]["contrast"] = "HF minus no-HF"
    effects["interaction"]["contrast"] = (
        "(UPR HF effect) minus (Flow HF effect)"
    )

    per_tau: dict[str, Any] = {}
    for tau_index, tau in enumerate(tau_list):
        tau_points = {
            cell: aggregate_rmse(metrics, np.asarray([tau]))
            for cell, metrics in cells.items()
        }
        tau_point_effects = _effect_coefficients(
            {cell: np.log(score) for cell, score in tau_points.items()}
        )
        tau_draw_effects = _effect_coefficients(
            {
                cell: np.log(values[:, tau_index])
                for cell, values in score_draws.items()
            }
        )
        per_tau[str(int(tau))] = {
            "cell_rmse": tau_points,
            "effects": {
                name: _effect_payload(
                    float(tau_point_effects[name]),
                    tau_draw_effects[name],
                )
                for name in tau_point_effects
            },
        }

    channel_scores = {
        cell: _channel_rmse(metrics, tau_list)
        for cell, metrics in cells.items()
    }
    channel_effects = _effect_coefficients(
        {cell: np.log(values) for cell, values in channel_scores.items()}
    )
    per_field = {}
    for index, channel in enumerate(reference.channels):
        per_field[channel] = {
            "cell_rmse": {
                cell: float(values[index])
                for cell, values in channel_scores.items()
            },
            "architecture_relative_effect_pct": float(
                100.0 * np.expm1(channel_effects["architecture"][index])
            ),
            "highpass_relative_effect_pct": float(
                100.0 * np.expm1(channel_effects["highpass"][index])
            ),
            "interaction_relative_effect_pct": float(
                100.0 * np.expm1(channel_effects["interaction"][index])
            ),
        }
    worst_field = max(
        per_field,
        key=lambda channel: per_field[channel][
            "architecture_relative_effect_pct"
        ],
    )

    return {
        "taus": tau_list.tolist(),
        "block_days": block_days,
        "n_blocks": int(len(blocks)),
        "n_windows": int(selected.sum()),
        "bootstrap_draws": draws,
        "seed": seed,
        "cell_rmse": point_scores,
        "cell_rmse_ci95": {
            cell: np.percentile(
                values.mean(axis=1),
                [2.5, 50.0, 97.5],
            ).tolist()
            for cell, values in score_draws.items()
        },
        "effects": effects,
        "per_tau": per_tau,
        "per_field": per_field,
        "worst_architecture_field": {
            "channel": worst_field,
            "relative_effect_pct": per_field[worst_field][
                "architecture_relative_effect_pct"
            ],
        },
    }


def _load_year(
    *,
    selection: dict[str, Any],
    root: Path,
    year: int,
    models: dict[str, str],
) -> tuple[dict[str, WindowMetrics], dict[str, Any]]:
    selection_models = selection.get("models")
    if not isinstance(selection_models, dict):
        raise ValueError("selection report lacks model metadata")

    cells: dict[str, WindowMetrics] = {}
    artifacts: dict[str, Any] = {}
    input_provenance_hashes: set[str] = set()
    dataset_provenance_hashes: set[str] = set()
    for cell in CELL_ORDER:
        model = models[cell]
        if model not in selection_models:
            raise ValueError(f"{model}: absent from frozen selection")
        metrics_path = root / f"{model}.json"
        summary = load_field_metrics(metrics_path)
        if not summary["evaluation_full_year"]:
            raise ValueError(f"{metrics_path}: evaluation is not full-year")
        dataset = summary["evaluation_dataset_provenance"]
        if dataset.get("years") != [year]:
            raise ValueError(f"{metrics_path}: expected dataset year {year}")
        expected_checkpoint = selection_models[model].get("checkpoint_sha256")
        if summary["checkpoint_sha256"] != expected_checkpoint:
            raise ValueError(
                f"{model}: checkpoint differs from frozen selection"
            )

        payload = json.loads(metrics_path.read_text())
        window_value = payload.get("window_metrics_file")
        if not window_value:
            raise ValueError(f"{metrics_path}: missing window_metrics_file")
        window_path = Path(str(window_value))
        if not window_path.is_absolute():
            window_path = metrics_path.parent / window_path
        metrics = load_metrics(window_path)
        if set(int(value) for value in np.unique(metrics.year)) != {year}:
            raise ValueError(f"{window_path}: contains the wrong year")
        cells[cell] = metrics

        input_hash = hashlib.sha256(
            json.dumps(
                summary["evaluation_input_provenance"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        dataset_hash = hashlib.sha256(
            json.dumps(
                dataset,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        input_provenance_hashes.add(input_hash)
        dataset_provenance_hashes.add(dataset_hash)
        artifacts[cell] = {
            "model": model,
            "checkpoint_sha256": expected_checkpoint,
            "metrics_json": str(metrics_path),
            "metrics_json_sha256": _file_sha256(metrics_path),
            "window_metrics": str(window_path),
            "window_metrics_sha256": _file_sha256(window_path),
            "window_index_sha256": metrics.index_sha256,
            "input_provenance_sha256": input_hash,
            "dataset_provenance_sha256": dataset_hash,
        }

    reference = cells[CELL_ORDER[0]]
    for cell in CELL_ORDER[1:]:
        assert_paired(reference, cells[cell])
    if len(input_provenance_hashes) != 1:
        raise ValueError(f"{year}: evaluation input provenance differs by cell")
    if len(dataset_provenance_hashes) != 1:
        raise ValueError(f"{year}: dataset provenance differs by cell")
    return cells, artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--root-2020", required=True, type=Path)
    parser.add_argument("--root-2021", required=True, type=Path)
    parser.add_argument("--upr-hf-model", default=DEFAULT_MODELS["upr_hf"])
    parser.add_argument(
        "--upr-nohf-model",
        default=DEFAULT_MODELS["upr_nohf"],
    )
    parser.add_argument("--flow-hf-model", default=DEFAULT_MODELS["flow_hf"])
    parser.add_argument(
        "--flow-nohf-model",
        default=DEFAULT_MODELS["flow_nohf"],
    )
    parser.add_argument("--all-taus", type=_parse_taus, default="1,2,3,4,5")
    parser.add_argument("--seen-taus", type=_parse_taus, default="1,3,5")
    parser.add_argument("--unseen-taus", type=_parse_taus, default="2,4")
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text())
    models = {
        "upr_hf": args.upr_hf_model,
        "upr_nohf": args.upr_nohf_model,
        "flow_hf": args.flow_hf_model,
        "flow_nohf": args.flow_nohf_model,
    }
    groups = {
        "all": args.all_taus,
        "seen": args.seen_taus,
        "unseen": args.unseen_taus,
    }
    schema_version = _validate_run_contract(selection, groups)

    years = {}
    for year, root in ((2020, args.root_2020), (2021, args.root_2021)):
        cells, artifacts = _load_year(
            selection=selection,
            root=root,
            year=year,
            models=models,
        )
        years[str(year)] = {
            "artifacts": artifacts,
            "analyses": {
                name: factorial_block_bootstrap(
                    cells,
                    taus=taus,
                    block_days=args.block_days,
                    draws=args.draws,
                    seed=args.seed,
                )
                for name, taus in groups.items()
            },
        }

    primary_effects = [
        years[str(year)]["analyses"]["unseen"]["effects"]["architecture"]
        for year in (2020, 2021)
    ]
    single_seed_upr_win = all(
        effect["log_effect_ci95"][2] < 0.0 for effect in primary_effects
    )
    single_seed_flow_win = all(
        effect["log_effect_ci95"][0] > 0.0 for effect in primary_effects
    )
    output = {
        "schema_version": 2,
        "design": "paired 2x2 UPR/Flow x HF/no-HF",
        "training_seed_count": 1,
        "uncertainty_scope": (
            "weekly weather-window sampling conditional on one matched "
            "training seed"
        ),
        "training_seed_generalization_claimed": False,
        "primary_analysis": "unseen",
        "effect_scale": "log macro-normalized-RMSE",
        "negative_effect_is_better": True,
        "models": models,
        "cell_factors": {
            "upr_hf": {"architecture": "UPR", "lambda_hf": 0.05},
            "upr_nohf": {"architecture": "UPR", "lambda_hf": 0.0},
            "flow_hf": {"architecture": "Flow-PP3", "lambda_hf": 0.05},
            "flow_nohf": {"architecture": "Flow-PP3", "lambda_hf": 0.0},
        },
        "selection": {
            "path": str(args.selection),
            "sha256": _file_sha256(args.selection),
            "schema_version": schema_version,
        },
        "years": years,
        "single_seed_architecture_effect_direction": (
            "UPR"
            if single_seed_upr_win
            else "Flow"
            if single_seed_flow_win
            else "inconclusive"
        ),
        "single_seed_architecture_effect_rule": (
            "same nonzero 95% weekly-block-bootstrap architecture-effect "
            "direction on unseen tau={2,4} in both 2020 and 2021; this does "
            "not establish robustness across training seeds"
        ),
        "generator_sha256": _file_sha256(Path(__file__)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(output, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.out_json)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
