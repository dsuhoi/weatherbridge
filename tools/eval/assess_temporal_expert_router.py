"""Assess a 2020-frozen temporal router on paired 2021 evidence."""
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
from weather_time_interp.metrics.physical_consistency import (
    SELECTION_DIAGNOSTICS,
)


def parse_mapping(value: str) -> tuple[str, str]:
    name, separator, path = value.partition("=")
    if not separator or not name.isidentifier() or not path:
        raise argparse.ArgumentTypeError("mapping must be NAME=PATH_OR_PREFIX")
    return name, path


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


def load_field(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol", {})
    dataset = payload.get("evaluation_dataset_provenance", {})
    input_provenance = payload.get("evaluation_input_provenance")
    if (
        protocol.get("full_year") is not True
        or protocol.get("rmse_reduction")
        != "spherical_strip_area_weighted_spatial_mean"
        or not isinstance(input_provenance, dict)
        or not isinstance(dataset, dict)
        or dataset.get("years") != [2021]
    ):
        raise ValueError(f"{path}: invalid 2021 field protocol")
    window_path = path.parent / payload["window_metrics_file"]
    window_provenance = payload.get("window_metrics_provenance")
    if (
        not isinstance(window_provenance, dict)
        or window_provenance.get("size_bytes") != window_path.stat().st_size
        or window_provenance.get("sha256") != file_sha256(window_path)
        or window_provenance.get("index_sha256")
        != protocol.get("index_sha256")
    ):
        raise ValueError(f"{path}: stale paired window provenance")
    with np.load(window_path, allow_pickle=False) as windows:
        year = np.asarray(windows["year"])
        t0 = np.asarray(windows["t0"])
        tau = np.asarray(windows["tau"])
        window_channels = tuple(
            str(value) for value in windows["channel_names"]
        )
        mse = np.asarray(windows["mse_norm_model"], dtype=np.float64)
        bilinear_mse = np.asarray(
            windows["mse_norm_bilinear"],
            dtype=np.float64,
        )
    channels = tuple(str(value) for value in payload["channel_names"])
    if (
        mse.ndim != 2
        or bilinear_mse.shape != mse.shape
        or year.ndim != 1
        or t0.ndim != 1
        or tau.ndim != 1
        or year.size != mse.shape[0]
        or t0.size != mse.shape[0]
        or mse.shape[0] != tau.size
        or mse.shape[1] != len(channels)
        or window_channels != channels
        or not np.issubdtype(year.dtype, np.integer)
        or not np.issubdtype(t0.dtype, np.integer)
        or not np.issubdtype(tau.dtype, np.integer)
        or not np.all(year == 2021)
        or not np.all(np.isfinite(mse))
        or not np.all(np.isfinite(bilinear_mse))
        or np.any(mse < 0)
        or np.any(bilinear_mse < 0)
    ):
        raise ValueError(f"{path}: invalid window metrics")
    metrics = WindowMetrics(year, t0, tau, mse, channels)
    if metrics.index_sha256 != protocol.get("index_sha256"):
        raise ValueError(f"{path}: paired window index hash mismatch")
    checkpoint_provenance = payload.get("checkpoint_provenance", {})
    checkpoint_sha256 = checkpoint_provenance.get("sha256")
    checkpoint_path = Path(str(checkpoint_provenance.get("path", "")))
    if (
        not isinstance(checkpoint_sha256, str)
        or len(checkpoint_sha256) != 64
        or not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != checkpoint_sha256
    ):
        raise ValueError(f"{path}: stale checkpoint provenance")
    return {
        "payload": payload,
        "year": year,
        "t0": t0,
        "tau": tau,
        "mse": mse,
        "bilinear_mse": bilinear_mse,
        "channels": channels,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "input_provenance_sha256": canonical_sha256(
            input_provenance
        ),
        "dataset_provenance_sha256": canonical_sha256(dataset),
    }


def _assert_paired(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    label: str,
) -> None:
    for key in ("year", "t0", "tau"):
        if not np.array_equal(reference[key], candidate[key]):
            raise ValueError(f"{label}: paired window-index mismatch")
    if reference["mse"].shape != candidate["mse"].shape:
        raise ValueError(f"{label}: paired channel shape mismatch")
    if reference["channels"] != candidate["channels"]:
        raise ValueError(f"{label}: paired channel-order mismatch")
    for key in (
        "input_provenance_sha256",
        "dataset_provenance_sha256",
    ):
        if reference[key] != candidate[key]:
            raise ValueError(f"{label}: paired {key} mismatch")
    if not np.array_equal(
        reference["bilinear_mse"],
        candidate["bilinear_mse"],
    ):
        raise ValueError(f"{label}: paired bilinear baseline mismatch")


def block_bootstrap_gain(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    draws: int,
    seed: int,
    block_hours: int,
) -> dict[str, float]:
    _assert_paired(reference, candidate, "bootstrap")
    if block_hours <= 0 or block_hours % 24:
        raise ValueError("block_hours must be a positive whole number of days")
    candidate_metrics = WindowMetrics(
        year=candidate["year"],
        t0=candidate["t0"],
        tau=candidate["tau"],
        mse=candidate["mse"],
        channels=candidate["channels"],
    )
    reference_metrics = WindowMetrics(
        year=reference["year"],
        t0=reference["t0"],
        tau=reference["tau"],
        mse=reference["mse"],
        channels=reference["channels"],
    )
    result = paired_block_bootstrap(
        candidate_metrics,
        reference_metrics,
        taus=np.asarray(sorted(set(candidate["tau"])), dtype=np.int16),
        block_days=block_hours // 24,
        draws=draws,
        seed=seed,
    )
    candidate_mean = float(result["left_rmse"])
    reference_mean = float(result["right_rmse"])
    return {
        "candidate_mean": candidate_mean,
        "reference_mean": reference_mean,
        "relative_gain": (
            (reference_mean - candidate_mean) / reference_mean
        ),
        "delta_mean": float(result["delta_left_minus_right"]),
        "delta_ci_low": float(result["delta_ci95"][0]),
        "delta_ci_high": float(result["delta_ci95"][2]),
        "p_one_sided": float(result["p_left_better_one_sided"]),
        "p_regression_one_sided": float(
            result["p_left_worse_one_sided"]
        ),
        "p_two_sided": float(result["p_paired_block_permutation"]),
        "num_blocks": int(result["n_blocks"]),
    }


def extreme_subset(
    value: dict[str, Any],
    *,
    quantile: float,
) -> dict[str, Any]:
    if not 0.5 < quantile < 1.0:
        raise ValueError("extreme quantile must lie in (0.5, 1)")
    baseline = np.sqrt(
        np.maximum(value["bilinear_mse"], 0.0)
    ).mean(axis=1)
    keep = np.zeros(value["tau"].shape, dtype=bool)
    thresholds: dict[int, float] = {}
    for tau in sorted({int(item) for item in value["tau"]}):
        tau_mask = value["tau"] == tau
        threshold = float(np.quantile(baseline[tau_mask], quantile))
        thresholds[tau] = threshold
        keep |= tau_mask & (baseline >= threshold)
    if np.count_nonzero(keep) < 2:
        raise ValueError("extreme subset contains too few paired windows")
    return {
        **value,
        "year": value["year"][keep],
        "t0": value["t0"][keep],
        "tau": value["tau"][keep],
        "mse": value["mse"][keep],
        "bilinear_mse": value["bilinear_mse"][keep],
        "extreme_threshold_by_tau": thresholds,
        "extreme_num_windows": int(np.count_nonzero(keep)),
    }


def noninferiority_gate(
    result: dict[str, float],
    *,
    margin_relative: float,
    alpha: float,
) -> dict[str, float | bool]:
    if not 0.0 <= margin_relative < 1.0:
        raise ValueError("non-inferiority margin must lie in [0, 1)")
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must lie in (0, 0.5)")
    reference_mean = float(result["reference_mean"])
    if reference_mean <= 0.0:
        raise ValueError("non-inferiority reference must be positive")
    margin_absolute = margin_relative * reference_mean
    passed = (
        float(result["relative_gain"]) >= -margin_relative
        and float(result["delta_ci_high"]) <= margin_absolute
        and not (
            float(result["delta_mean"]) > 0.0
            and float(result["p_regression_one_sided"]) <= alpha
        )
    )
    return {
        "margin_relative": margin_relative,
        "margin_absolute": margin_absolute,
        "pass": passed,
    }


def field_gates(
    router: dict[str, Any],
    experts: dict[str, dict[str, Any]],
    route_by_tau: dict[int, str],
    default_expert: str,
) -> dict[str, Any]:
    default = experts[default_expert]
    _assert_paired(default, router, "router/default")
    channels = tuple(router["payload"]["channel_names"])
    max_default_ratio = 0.0
    max_selected_relative_error = 0.0
    max_physical_ratio = 0.0
    max_acc_drop_vs_default = 0.0
    max_selected_acc_error = 0.0
    unseen = {int(value) for value in router["payload"]["unseen_tau"]}
    unseen_router: list[float] = []
    unseen_default: list[float] = []
    for tau, selected_name in route_by_tau.items():
        selected = experts[selected_name]
        router_cell = router["payload"]["per_tau"][str(tau)]["model"]
        selected_cell = selected["payload"]["per_tau"][str(tau)]["model"]
        default_cell = default["payload"]["per_tau"][str(tau)]["model"]
        linear_cell = router["payload"]["per_tau"][str(tau)]["bilinear"]
        for channel in channels:
            key = f"rmse_norm_{channel}"
            router_value = float(router_cell[key])
            selected_value = float(selected_cell[key])
            default_value = float(default_cell[key])
            if selected_value > 0:
                max_selected_relative_error = max(
                    max_selected_relative_error,
                    abs(router_value - selected_value) / selected_value,
                )
            if default_value > 0:
                max_default_ratio = max(
                    max_default_ratio,
                    router_value / default_value,
                )
            if tau in unseen:
                unseen_router.append(router_value)
                unseen_default.append(default_value)
        router_acc = float(router_cell["acc_mean"])
        selected_acc = float(selected_cell["acc_mean"])
        default_acc = float(default_cell["acc_mean"])
        max_acc_drop_vs_default = max(
            max_acc_drop_vs_default,
            default_acc - router_acc,
        )
        max_selected_acc_error = max(
            max_selected_acc_error,
            abs(router_acc - selected_acc),
        )
        for diagnostic in SELECTION_DIAGNOSTICS:
            key = f"physical_{diagnostic}"
            baseline = float(linear_cell[key])
            max_physical_ratio = max(
                max_physical_ratio,
                float(router_cell[key]) / baseline,
            )
    unseen_ratio = float(np.mean(unseen_router) / np.mean(unseen_default))
    router_temporal = float(
        router["payload"]["temporal_curvature_rmse_norm"]["model"]
    )
    default_temporal = float(
        default["payload"]["temporal_curvature_rmse_norm"]["model"]
    )
    bilinear_temporal = float(
        router["payload"]["temporal_curvature_rmse_norm"]["bilinear"]
    )
    if (
        not all(
            math.isfinite(value)
            for value in (
                router_temporal,
                default_temporal,
                bilinear_temporal,
            )
        )
        or router_temporal < 0.0
        or default_temporal <= 0.0
        or bilinear_temporal <= 0.0
    ):
        raise ValueError("invalid temporal-curvature diagnostics")
    temporal_default_ratio = router_temporal / default_temporal
    temporal_bilinear_ratio = router_temporal / bilinear_temporal
    return {
        "max_field_ratio_to_default": max_default_ratio,
        "max_relative_error_to_selected_expert": (
            max_selected_relative_error
        ),
        "unseen_channel_macro_ratio_to_default": unseen_ratio,
        "max_physical_ratio_to_bilinear": max_physical_ratio,
        "max_acc_drop_vs_default": max_acc_drop_vs_default,
        "max_acc_error_to_selected_expert": max_selected_acc_error,
        "temporal_curvature_ratio_to_default": temporal_default_ratio,
        "temporal_curvature_ratio_to_bilinear": temporal_bilinear_ratio,
        "pass": (
            max_default_ratio <= 1.01
            and max_selected_relative_error <= 1e-5
            and unseen_ratio <= 1.001
            and max_physical_ratio <= 1.05
            and max_acc_drop_vs_default <= 0.001
            and max_selected_acc_error <= 1e-6
            and temporal_default_ratio <= 1.05
            and temporal_bilinear_ratio <= 1.05
        ),
    }


def spectral_consistency(
    router_root: Path,
    router_name: str,
    expert_root: Path,
    expert_prefixes: dict[str, str],
    route_by_tau: dict[int, str],
    taus: set[int] | None = None,
) -> dict[str, Any]:
    requested_taus = set(route_by_tau) if taus is None else set(taus)
    if not requested_taus or not requested_taus.issubset(route_by_tau):
        raise ValueError("spectral taus are empty or absent from the route")
    max_relative_error = 0.0
    checked: list[int] = []
    for tau, expert_name in sorted(route_by_tau.items()):
        if tau not in requested_taus:
            continue
        router_path = router_root / f"{router_name}_tau{tau}.npz"
        expert_path = (
            expert_root / f"{expert_prefixes[expert_name]}_tau{tau}.npz"
        )
        with np.load(router_path, allow_pickle=False) as router, np.load(
            expert_path,
            allow_pickle=False,
        ) as expert:
            for key in ("ell", "gt_El", "window_year", "window_t0"):
                if not np.array_equal(router[key], expert[key]):
                    raise ValueError(
                        f"tau={tau}: routed/source spectral index mismatch"
                    )
            routed_energy = np.asarray(router["pred_El"], dtype=np.float64)
            expert_energy = np.asarray(expert["pred_El"], dtype=np.float64)
            denominator = np.maximum(np.abs(expert_energy), 1e-12)
            max_relative_error = max(
                max_relative_error,
                float(np.max(np.abs(routed_energy - expert_energy) / denominator)),
            )
        checked.append(tau)
    return {
        "taus": checked,
        "max_pred_energy_relative_error": max_relative_error,
        "pass": max_relative_error <= 1e-5,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path)
    parser.add_argument("--router-field-2021", type=Path, required=True)
    parser.add_argument(
        "--expert-field-2021",
        type=parse_mapping,
        action="append",
        required=True,
    )
    parser.add_argument("--router-spectra-2020", type=Path, required=True)
    parser.add_argument("--router-name", default="upr_flow_tau_router")
    parser.add_argument("--expert-spectra-2020", type=Path, required=True)
    parser.add_argument(
        "--spectral-taus",
        default="",
        help="Comma-separated spectral taus; empty checks every routed hour.",
    )
    parser.add_argument(
        "--expert-spectrum-prefix",
        type=parse_mapping,
        action="append",
        required=True,
    )
    parser.add_argument("--min-relative-gain", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=202708)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--extreme-quantile", type=float, default=0.95)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    route = json.loads(args.route.read_text())
    if (
        route.get("selection_year") != 2020
        or route.get("ood_year_loaded") is not False
    ):
        raise ValueError("router route was not frozen before OOD")
    route_by_tau = {
        int(hour): str(name)
        for hour, name in route["route_by_tau"].items()
    }
    experts = {
        name: load_field(Path(path))
        for name, path in args.expert_field_2021
    }
    route_experts = route.get("experts", {})
    if (
        not isinstance(route_experts, dict)
        or not all(
            isinstance(metadata, dict)
            for metadata in route_experts.values()
        )
        or set(experts) != set(route_experts)
        or set(route_by_tau.values()) - set(experts)
    ):
        raise ValueError("route/expert 2021 field artifacts mismatch")
    for name, expert in experts.items():
        if expert["checkpoint_sha256"] != route_experts[name].get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{name}: route/2021 checkpoint mismatch")
    default_expert = str(route["default_expert"])
    router = load_field(args.router_field_2021)
    router_checkpoint = (
        args.router_checkpoint
        if args.router_checkpoint is not None
        else Path(router["checkpoint_path"])
    )
    if router["checkpoint_sha256"] != file_sha256(router_checkpoint):
        raise ValueError("router field/checkpoint hash mismatch")
    gain = block_bootstrap_gain(
        router,
        experts[default_expert],
        draws=args.draws,
        seed=args.seed,
        block_hours=args.block_days * 24,
    )
    extreme_router = extreme_subset(
        router,
        quantile=args.extreme_quantile,
    )
    extreme_default = extreme_subset(
        experts[default_expert],
        quantile=args.extreme_quantile,
    )
    extreme_gain = block_bootstrap_gain(
        extreme_router,
        extreme_default,
        draws=args.draws,
        seed=args.seed + 1,
        block_hours=args.block_days * 24,
    )
    extreme_gain["quantile"] = args.extreme_quantile
    extreme_gain["num_windows"] = extreme_router["extreme_num_windows"]
    extreme_noninferiority = noninferiority_gate(
        extreme_gain,
        margin_relative=0.005,
        alpha=args.alpha,
    )
    extreme_gain["noninferiority_margin_absolute"] = float(
        extreme_noninferiority["margin_absolute"]
    )
    extreme_pass = bool(extreme_noninferiority["pass"])
    fields = field_gates(
        router,
        experts,
        route_by_tau,
        default_expert,
    )
    spectra = spectral_consistency(
        args.router_spectra_2020,
        args.router_name,
        args.expert_spectra_2020,
        dict(args.expert_spectrum_prefix),
        route_by_tau,
        {
            int(value)
            for value in args.spectral_taus.split(",")
            if value.strip()
        }
        or None,
    )
    gain_pass = (
        gain["relative_gain"] >= args.min_relative_gain
        and gain["p_one_sided"] <= args.alpha
        and gain["delta_ci_high"] < 0.0
    )
    report = {
        "schema_version": 1,
        "evidence_level": "frozen_2020_route_confirmed_on_2021",
        "route_by_tau": {
            str(hour): name for hour, name in sorted(route_by_tau.items())
        },
        "default_expert": default_expert,
        "all_hour_2021_gain": gain,
        "extreme_2021_gain": {
            **extreme_gain,
            "pass": extreme_pass,
        },
        "field_and_physics_gates": fields,
        "spectral_dispatch_consistency_2020": spectra,
        "thresholds": {
            "min_relative_gain": args.min_relative_gain,
            "alpha_one_sided": args.alpha,
            "max_field_ratio_to_default": 1.01,
            "max_unseen_ratio_to_default": 1.001,
            "max_physical_ratio_to_bilinear": 1.05,
            "max_dispatch_relative_error": 1e-5,
            "max_acc_drop_vs_default": 0.001,
            "max_temporal_curvature_ratio": 1.05,
            "extreme_quantile": args.extreme_quantile,
            "extreme_noninferiority_margin": 0.005,
        },
        "promoted": bool(
            gain_pass
            and extreme_pass
            and fields["pass"]
            and spectra["pass"]
        ),
    }
    if not all(math.isfinite(value) for value in gain.values()):
        raise ValueError("non-finite temporal-router bootstrap result")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"promoted": report["promoted"]}))


if __name__ == "__main__":
    main()
