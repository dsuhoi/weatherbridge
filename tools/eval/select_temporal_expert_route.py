"""Freeze a target-hour expert route from paired full-year 2020 metrics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    paired_block_bootstrap,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def parse_expert(value: str) -> tuple[str, Path, Path]:
    name, separator, remainder = value.partition(":")
    metrics, separator, checkpoint = remainder.partition(":")
    if (
        not separator
        or not name.isidentifier()
        or not metrics
        or not checkpoint
    ):
        raise argparse.ArgumentTypeError(
            "expert must be NAME:FIELD_JSON:CHECKPOINT"
        )
    return name, Path(metrics), Path(checkpoint)


def load_expert(
    name: str,
    metrics_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    payload = json.loads(metrics_path.read_text())
    checkpoint_hash = file_sha256(checkpoint_path)
    if payload.get("checkpoint_provenance", {}).get("sha256") != checkpoint_hash:
        raise ValueError(f"{name}: field/checkpoint hash mismatch")
    protocol = payload.get("evaluation_protocol", {})
    dataset = payload.get("evaluation_dataset_provenance", {})
    if protocol.get("full_year") is not True:
        raise ValueError(f"{name}: route selection requires full-year metrics")
    if (
        protocol.get("rmse_reduction")
        != "spherical_strip_area_weighted_spatial_mean"
    ):
        raise ValueError(f"{name}: route selection requires proper RMSE")
    if dataset.get("years") != [2020]:
        raise ValueError(f"{name}: route selection is restricted to 2020")

    channels = tuple(str(value) for value in payload["channel_names"])
    per_tau: dict[int, float] = {}
    for tau_text, methods in payload["per_tau"].items():
        model = methods["model"]
        channel_rmse = [
            float(model[f"rmse_norm_{channel}"])
            for channel in channels
        ]
        if any(not math.isfinite(value) or value < 0 for value in channel_rmse):
            raise ValueError(f"{name}: invalid RMSE at tau={tau_text}")
        per_tau[int(tau_text)] = float(np.mean(channel_rmse))

    window_path = metrics_path.parent / payload["window_metrics_file"]
    window_provenance = payload.get("window_metrics_provenance")
    if (
        not isinstance(window_provenance, dict)
        or window_provenance.get("size_bytes") != window_path.stat().st_size
        or window_provenance.get("sha256") != file_sha256(window_path)
        or window_provenance.get("index_sha256")
        != protocol.get("index_sha256")
    ):
        raise ValueError(f"{name}: stale paired window provenance")
    with np.load(window_path, allow_pickle=False) as windows:
        window_year = np.asarray(windows["year"])
        window_t0 = np.asarray(windows["t0"])
        window_tau = np.asarray(windows["tau"])
        window_channels = tuple(
            str(value) for value in windows["channel_names"]
        )
        model_mse = np.asarray(windows["mse_norm_model"], dtype=np.float64)
        bilinear_mse = np.asarray(
            windows["mse_norm_bilinear"],
            dtype=np.float64,
        )
    if (
        model_mse.ndim != 2
        or model_mse.shape != bilinear_mse.shape
        or window_year.ndim != 1
        or window_t0.ndim != 1
        or window_tau.ndim != 1
        or window_year.size != model_mse.shape[0]
        or window_t0.size != model_mse.shape[0]
        or model_mse.shape[0] != window_tau.size
        or model_mse.shape[1] != len(channels)
        or window_channels != channels
        or not np.issubdtype(window_year.dtype, np.integer)
        or not np.issubdtype(window_t0.dtype, np.integer)
        or not np.issubdtype(window_tau.dtype, np.integer)
        or not np.all(window_year == 2020)
        or not np.all(np.isfinite(model_mse))
        or not np.all(np.isfinite(bilinear_mse))
        or np.any(model_mse < 0)
        or np.any(bilinear_mse < 0)
    ):
        raise ValueError(f"{name}: invalid paired window metrics")
    window_metrics = WindowMetrics(
        year=window_year,
        t0=window_t0,
        tau=window_tau,
        mse=model_mse,
        channels=channels,
    )
    if window_metrics.index_sha256 != protocol.get("index_sha256"):
        raise ValueError(f"{name}: paired window index hash mismatch")
    if set(per_tau) != set(int(value) for value in np.unique(window_tau)):
        raise ValueError(f"{name}: JSON/NPZ target-hour mismatch")
    derived_per_tau: dict[int, float] = {}
    for tau in sorted(per_tau):
        mask = window_tau == tau
        derived = float(
            np.sqrt(model_mse[mask].mean(axis=0)).mean()
        )
        if not math.isclose(
            derived,
            per_tau[tau],
            rel_tol=1e-5,
            abs_tol=1e-8,
        ):
            raise ValueError(f"{name}: JSON/NPZ RMSE mismatch at tau={tau}")
        derived_per_tau[tau] = derived
    return {
        "name": name,
        "metrics_path": str(metrics_path.resolve()),
        "metrics_sha256": file_sha256(metrics_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_arch": payload["checkpoint_provenance"].get("arch"),
        "channels": channels,
        "seen_tau": tuple(int(value) for value in payload["seen_tau"]),
        "unseen_tau": tuple(int(value) for value in payload["unseen_tau"]),
        "per_tau_rmse": derived_per_tau,
        "window_year": window_year,
        "window_t0": window_t0,
        "window_tau": window_tau,
        "model_mse": model_mse,
        "bilinear_mse": bilinear_mse,
        "evaluation_index_sha256": protocol.get("index_sha256"),
        "input_provenance_sha256": canonical_sha256(
            payload.get("evaluation_input_provenance")
        ),
        "dataset_provenance_sha256": canonical_sha256(dataset),
    }


def _validate_pairing(experts: dict[str, dict[str, Any]]) -> None:
    first = next(iter(experts.values()))
    scalar_keys = (
        "channels",
        "seen_tau",
        "unseen_tau",
        "evaluation_index_sha256",
        "input_provenance_sha256",
        "dataset_provenance_sha256",
    )
    array_keys = (
        "window_year",
        "window_t0",
        "window_tau",
        "bilinear_mse",
    )
    for name, expert in experts.items():
        for key in scalar_keys:
            if expert[key] != first[key]:
                raise ValueError(f"{name}: paired artifact mismatch for {key}")
        for key in array_keys:
            if not np.array_equal(expert[key], first[key]):
                raise ValueError(f"{name}: paired artifact mismatch for {key}")
        if set(expert["per_tau_rmse"]) != set(first["per_tau_rmse"]):
            raise ValueError(f"{name}: target-hour set mismatch")
        if expert["model_mse"].shape != first["model_mse"].shape:
            raise ValueError(f"{name}: paired model-MSE shape mismatch")


def _comparison_seed(seed: int, tau: int, expert_name: str) -> int:
    digest = hashlib.sha256(
        f"{seed}:{tau}:{expert_name}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def _paired_block_test(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    tau: int,
    block_days: int,
    bootstrap_draws: int,
    seed: int,
) -> dict[str, Any]:
    candidate_metrics = WindowMetrics(
        year=candidate["window_year"],
        t0=candidate["window_t0"],
        tau=candidate["window_tau"],
        mse=candidate["model_mse"],
        channels=candidate["channels"],
    )
    reference_metrics = WindowMetrics(
        year=reference["window_year"],
        t0=reference["window_t0"],
        tau=reference["window_tau"],
        mse=reference["model_mse"],
        channels=reference["channels"],
    )
    result = paired_block_bootstrap(
        candidate_metrics,
        reference_metrics,
        taus=np.asarray([tau], dtype=np.int16),
        block_days=block_days,
        draws=bootstrap_draws,
        seed=seed,
    )
    return {
        "candidate_rmse": result["left_rmse"],
        "default_rmse": result["right_rmse"],
        "delta_candidate_minus_default": (
            result["delta_left_minus_right"]
        ),
        "delta_ci95": result["delta_ci95"],
        "p_one_sided": result["p_left_better_one_sided"],
        "p_holm": 1.0,
        "num_time_blocks": result["n_blocks"],
        "num_paired_windows": result["n_windows"],
        "seed": seed,
    }


def select_route(
    experts: dict[str, dict[str, Any]],
    *,
    default_expert: str,
    min_relative_gain: float,
    alpha: float,
    bootstrap_draws: int,
    seed: int,
    block_days: int,
) -> tuple[dict[int, str], dict[int, dict[str, Any]]]:
    if default_expert not in experts:
        raise ValueError("default expert is not present")
    if min_relative_gain < 0:
        raise ValueError("min_relative_gain must be non-negative")
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    if bootstrap_draws < 100:
        raise ValueError("bootstrap_draws must be at least 100")
    if block_days < 1:
        raise ValueError("block_days must be positive")
    _validate_pairing(experts)

    route: dict[int, str] = {}
    diagnostics: dict[int, dict[str, Any]] = {}
    switch_candidates: list[tuple[int, str]] = []
    taus = sorted(next(iter(experts.values()))["per_tau_rmse"])
    for tau in taus:
        rmse = {
            name: float(expert["per_tau_rmse"][tau])
            for name, expert in experts.items()
        }
        best = min(rmse, key=lambda name: (rmse[name], name))
        relative_gain = (
            (rmse[default_expert] - rmse[best]) / rmse[default_expert]
            if rmse[default_expert] > 0
            else 0.0
        )
        candidate_tests: dict[str, dict[str, Any]] = {}
        for name in sorted(experts):
            if name == default_expert:
                continue
            candidate_gain = (
                (rmse[default_expert] - rmse[name])
                / rmse[default_expert]
                if rmse[default_expert] > 0
                else 0.0
            )
            comparison_seed = _comparison_seed(seed, tau, name)
            candidate_tests[name] = _paired_block_test(
                experts[name],
                experts[default_expert],
                tau=tau,
                block_days=block_days,
                bootstrap_draws=bootstrap_draws,
                seed=comparison_seed,
            )
            candidate_tests[name]["relative_gain_vs_default"] = (
                candidate_gain
            )
            switch_candidates.append((tau, name))
        route[tau] = default_expert
        best_test = candidate_tests.get(best, {})
        diagnostics[tau] = {
            "rmse_by_expert": rmse,
            "empirical_best": best,
            "selected": default_expert,
            "relative_gain_vs_default": relative_gain,
            "paired_bootstrap_p_one_sided": best_test.get(
                "p_one_sided",
                1.0,
            ),
            "paired_bootstrap_p_holm": 1.0,
            "paired_bootstrap_delta_upper": (
                best_test.get("delta_ci95", [0.0, 0.0, 0.0])[2]
            ),
            "num_paired_windows": best_test.get(
                "num_paired_windows",
                0,
            ),
            "num_time_blocks": best_test.get("num_time_blocks", 0),
            "candidate_tests": candidate_tests,
        }
    previous_adjusted = 0.0
    family_size = len(switch_candidates)
    for rank, (tau, name) in enumerate(
        sorted(
            switch_candidates,
            key=lambda item: (
                diagnostics[item[0]]["candidate_tests"][item[1]][
                    "p_one_sided"
                ],
                item[0],
                item[1],
            ),
        ),
    ):
        candidate_test = diagnostics[tau]["candidate_tests"][name]
        raw = candidate_test["p_one_sided"]
        adjusted = min(
            1.0,
            max(previous_adjusted, raw * (family_size - rank)),
        )
        previous_adjusted = adjusted
        candidate_test["p_holm"] = adjusted

    for tau in taus:
        passing = [
            name
            for name, candidate_test in diagnostics[tau][
                "candidate_tests"
            ].items()
            if (
                candidate_test["relative_gain_vs_default"]
                >= min_relative_gain
                and candidate_test["p_holm"] <= alpha
                and candidate_test["delta_ci95"][2] < 0.0
            )
        ]
        if passing:
            selected = min(
                passing,
                key=lambda name: (
                    diagnostics[tau]["rmse_by_expert"][name],
                    name,
                ),
            )
            route[tau] = selected
            diagnostics[tau]["selected"] = selected
        best = diagnostics[tau]["empirical_best"]
        if best in diagnostics[tau]["candidate_tests"]:
            best_test = diagnostics[tau]["candidate_tests"][best]
            diagnostics[tau]["paired_bootstrap_p_holm"] = best_test[
                "p_holm"
            ]
            diagnostics[tau]["paired_bootstrap_delta_upper"] = best_test[
                "delta_ci95"
            ][2]
            diagnostics[tau]["num_paired_windows"] = best_test[
                "num_paired_windows"
            ]
            diagnostics[tau]["num_time_blocks"] = best_test[
                "num_time_blocks"
            ]
    return route, diagnostics


def select_default_expert(
    experts: dict[str, dict[str, Any]],
) -> str:
    return min(
        experts,
        key=lambda name: (
            float(np.mean(list(experts[name]["per_tau_rmse"].values()))),
            name,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expert",
        action="append",
        type=parse_expert,
        required=True,
        help="NAME:FIELD_JSON:CHECKPOINT; repeat for each expert.",
    )
    parser.add_argument(
        "--default-expert",
        required=True,
        help="Expert name or 'auto' for the best 2020 all-hour mean RMSE.",
    )
    parser.add_argument("--min-relative-gain", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=202707)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    experts = {
        name: load_expert(name, metrics, checkpoint)
        for name, metrics, checkpoint in args.expert
    }
    if len(experts) != len(args.expert):
        raise ValueError("expert names must be unique")
    default_expert = (
        select_default_expert(experts)
        if args.default_expert == "auto"
        else args.default_expert
    )
    route, diagnostics = select_route(
        experts,
        default_expert=default_expert,
        min_relative_gain=args.min_relative_gain,
        alpha=args.alpha,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        block_days=args.block_days,
    )
    report = {
        "schema_version": 2,
        "evidence_level": "frozen_full_year_2020_tau_router",
        "selection_year": 2020,
        "ood_year_loaded": False,
        "default_expert": default_expert,
        "default_expert_rule": (
            "lowest_full_year_2020_all_hour_channel_macro_rmse"
            if args.default_expert == "auto"
            else "explicit"
        ),
        "route_by_tau": {
            str(hour): name for hour, name in sorted(route.items())
        },
        "selection_rule": {
            "metric": "channel_macro_normalized_rmse",
            "paired_statistic": (
                "per_tau_mean_of_per_channel_rmse"
            ),
            "min_relative_gain": args.min_relative_gain,
            "alpha_one_sided": args.alpha,
            "multiplicity_correction": (
                "Holm across all expert-by-target-hour switches"
            ),
            "null_test": "paired complete-time-block permutation",
            "bootstrap_draws": args.bootstrap_draws,
            "block_days": args.block_days,
            "seed": args.seed,
        },
        "per_tau": {
            str(hour): value
            for hour, value in sorted(diagnostics.items())
        },
        "experts": {
            name: {
                key: expert[key]
                for key in (
                    "metrics_path",
                    "metrics_sha256",
                    "checkpoint_path",
                    "checkpoint_sha256",
                    "checkpoint_arch",
                )
            }
            for name, expert in experts.items()
        },
        "paired_artifacts": {
            "evaluation_index_sha256": next(iter(experts.values()))[
                "evaluation_index_sha256"
            ],
            "input_provenance_sha256": next(iter(experts.values()))[
                "input_provenance_sha256"
            ],
            "dataset_provenance_sha256": next(iter(experts.values()))[
                "dataset_provenance_sha256"
            ],
        },
        "selection_code_sha256": file_sha256(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(report["route_by_tau"], sort_keys=True))


if __name__ == "__main__":
    main()
