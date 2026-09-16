#!/usr/bin/env python3
"""Rank UPR-Lite candidates by interpolation skill and spectral fidelity."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from tools.data.verify_normalization_provenance import (
    verify as verify_normalization_provenance,
)
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.metrics.physical_consistency import (
    SELECTION_DIAGNOSTICS,
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


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_cost_artifact(
    path: Path,
    models: tuple[str, ...],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(payload, dict)
        or isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version < 2
    ):
        raise ValueError(f"{path}: unsupported cost artifact schema")
    costs = payload.get("models")
    if not isinstance(costs, dict) or not set(models).issubset(costs):
        raise ValueError(f"{path}: cost model set mismatch")
    benchmark_source = Path(__file__).with_name(
        "benchmark_capmatched_inference.py"
    )
    if payload.get("evaluation_script_sha256") != _file_sha256(
        benchmark_source
    ):
        raise ValueError(f"{path}: benchmark code hash mismatch")
    repo_root = Path(__file__).resolve().parents[2]
    supporting = payload.get("supporting_code_sha256")
    required_supporting = {
        "tools/eval/capmatched_loader.py",
        "tools/train/train_capacity_matched_6h.py",
        "weather_time_interp/model/weatherbridge_upr_lite_model.py",
        "weather_time_interp/model/weatherbridge_upr_scaled_model.py",
        "weather_time_interp/model/weatherbridge_upr_spherical_model.py",
        "weather_time_interp/model/temporal_expert_router.py",
        "weather_time_interp/model/weatherbridge_flow_model.py",
        "weather_time_interp/model/weather_amt_model.py",
        "weather_time_interp/model/amt_upstream/feat_enc.py",
        "weather_time_interp/model/amt_upstream/flow_utils.py",
        "weather_time_interp/model/amt_upstream/ifrnet.py",
        "weather_time_interp/model/amt_upstream/multi_flow.py",
        "weather_time_interp/model/amt_upstream/raft.py",
        "weather_time_interp/model/dcae_adaln_model.py",
        "weather_time_interp/model/dcae_adaln_skip_model.py",
        "legacy/scripts/train_atm_vfi_12h_oddskip.py",
    }
    if (
        not isinstance(supporting, dict)
        or not required_supporting.issubset(supporting)
    ):
        raise ValueError(f"{path}: missing benchmark supporting-code hashes")
    for relative, expected_hash in supporting.items():
        source = repo_root / str(relative)
        if not source.is_file() or _file_sha256(source) != expected_hash:
            raise ValueError(
                f"{path}: benchmark supporting code changed: {relative}"
            )
    static = payload.get("static_features")
    if not isinstance(static, dict):
        raise ValueError(f"{path}: missing benchmark static provenance")
    static_path = Path(str(static.get("path", "")))
    static_size = static.get("size_bytes")
    if (
        isinstance(static_size, bool)
        or not isinstance(static_size, int)
        or static_size < 0
        or not static_path.is_file()
        or static_path.stat().st_size != static_size
        or _file_sha256(static_path) != static.get("sha256")
    ):
        raise ValueError(f"{path}: benchmark static provenance mismatch")

    device = payload.get("device")
    capability = payload.get("device_capability")
    input_seed = payload.get("input_seed")
    if not isinstance(device, str) or "A100" not in device:
        raise ValueError(f"{path}: benchmark device is not an A100")
    if capability != [8, 0]:
        raise ValueError(f"{path}: benchmark device capability mismatch")
    if isinstance(input_seed, bool) or not isinstance(input_seed, int):
        raise TypeError(f"{path}: invalid benchmark input seed")

    artifact_tau_values = payload.get("tau_values")
    if artifact_tau_values != [0.5]:
        raise ValueError(f"{path}: selection benchmark must use tau=0.5")

    common_protocol: tuple[Any, ...] | None = None
    for name in models:
        result = costs[name]
        if not isinstance(result, dict):
            raise TypeError(f"{path}: invalid benchmark row for {name}")

        protocol_values: list[int] = []
        for key in (
            "batch_size",
            "height",
            "width",
            "warmup",
            "iterations",
            "repeats",
            "input_seed",
        ):
            value = result.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{path}: {name} has invalid benchmark {key}"
                )
            protocol_values.append(value)
        protocol = tuple(protocol_values)
        tau_values = result.get("tau_values")
        tau_mode = result.get("tau_mode")
        if tau_values != artifact_tau_values or tau_mode != "constant":
            raise ValueError(
                f"{path}: {name} benchmark target-time mismatch"
            )
        protocol = (*protocol, tuple(tau_values), tau_mode)
        if common_protocol is None:
            common_protocol = protocol
        elif protocol != common_protocol:
            raise ValueError(f"{path}: benchmark protocols differ by model")
        if (
            result["batch_size"] != 1
            or result["height"] != 360
            or result["width"] != 720
            or result["warmup"] < 3
            or result["iterations"] < 10
            or result["repeats"] < 5
            or result["input_seed"] != input_seed
        ):
            raise ValueError(f"{path}: {name} benchmark protocol mismatch")

        repeat_values = result.get("repeat_latency_ms")
        try:
            repeat_array = np.asarray(repeat_values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: {name} has invalid latency repeats"
            ) from exc
        if (
            repeat_array.ndim != 1
            or repeat_array.size != result["repeats"]
            or not np.all(np.isfinite(repeat_array))
            or np.any(repeat_array <= 0.0)
        ):
            raise ValueError(f"{path}: {name} has invalid latency repeats")

        expected_latency = float(np.median(repeat_array))
        expected_mean = float(np.mean(repeat_array))
        expected_std = float(np.std(repeat_array))
        expected_p95 = float(np.quantile(repeat_array, 0.95))
        expected_throughput = 1000.0 / expected_latency
        expected_values = {
            "latency_ms": expected_latency,
            "latency_mean_ms": expected_mean,
            "latency_std_ms": expected_std,
            "latency_p95_ms": expected_p95,
            "samples_per_second": expected_throughput,
        }
        for key, expected in expected_values.items():
            try:
                observed = float(result[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: {name} has invalid benchmark {key}"
                ) from exc
            if not math.isfinite(observed) or not math.isclose(
                observed,
                expected,
                rel_tol=1e-6,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"{path}: {name} benchmark statistic mismatch: {key}"
                )
        if expected_std / expected_mean > 0.10:
            raise ValueError(f"{path}: {name} has unstable benchmark latency")
        for key in ("params_m", "latency_ms", "peak_memory_mib"):
            try:
                value = float(result[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: {name} has invalid benchmark {key}"
                ) from exc
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{path}: {name} has invalid benchmark {key}"
                )

        checkpoint = Path(str(result.get("checkpoint", "")))
        checkpoint_hash = result.get("checkpoint_sha256")
        if (
            not checkpoint.is_file()
            or not isinstance(checkpoint_hash, str)
            or len(checkpoint_hash) != 64
            or _file_sha256(checkpoint) != checkpoint_hash
        ):
            raise ValueError(f"{path}: {name} checkpoint provenance mismatch")
    return payload


def _mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("metric values must be non-empty and finite")
    return float(np.mean(array))


def _validated_dataset_provenance(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label}: missing evaluation dataset provenance")
    try:
        current = memmap_dataset_provenance(
            value["root"],
            [int(year) for year in value["years"]],
            sampled_bytes_per_file=int(
                value["sampled_bytes_per_file_limit"]
            ),
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: invalid evaluation dataset provenance: {error}"
        ) from error
    if current != value:
        raise ValueError(f"{label}: evaluation dataset provenance is stale")
    return value


def load_field_metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    evaluation_protocol = dict(payload.get("evaluation_protocol", {}))
    input_provenance = payload.get("evaluation_input_provenance")
    required_inputs = {
        "static_features",
        "pressure_level_stats",
        "surface_stats",
        "climatology",
    }
    if (
        not isinstance(input_provenance, dict)
        or not required_inputs.issubset(input_provenance)
        or not isinstance(input_provenance["climatology"], dict)
    ):
        raise ValueError(
            f"{path}: missing complete evaluation input provenance"
        )
    dataset_provenance = _validated_dataset_provenance(
        payload.get("evaluation_dataset_provenance"),
        label=str(path),
    )
    checkpoint_provenance = payload.get("checkpoint_provenance")
    if (
        not isinstance(checkpoint_provenance, dict)
        or not isinstance(checkpoint_provenance.get("sha256"), str)
        or not checkpoint_provenance["sha256"]
    ):
        raise ValueError(f"{path}: missing checkpoint content hash")
    channels = payload["channel_names"]
    seen = [int(value) for value in payload["seen_tau"]]
    unseen = [int(value) for value in payload["unseen_tau"]]

    per_tau: dict[int, dict[str, float]] = {}
    for tau_text, methods in payload["per_tau"].items():
        tau = int(tau_text)
        model = methods["model"]
        linear = methods["bilinear"]
        model_channel_rmse = {
            name: float(model[f"rmse_norm_{name}"])
            for name in channels
        }
        linear_channel_rmse = {
            name: float(linear[f"rmse_norm_{name}"])
            for name in channels
        }
        if any(
            not math.isfinite(value) or value < 0.0
            for value in model_channel_rmse.values()
        ):
            raise ValueError(f"{path}: invalid model RMSE at tau={tau}")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in linear_channel_rmse.values()
        ):
            raise ValueError(f"{path}: invalid bilinear RMSE at tau={tau}")
        rmse = _mean(list(model_channel_rmse.values()))
        linear_rmse = _mean(list(linear_channel_rmse.values()))
        channel_bilinear_ratios = [
            model_channel_rmse[name] / linear_channel_rmse[name]
            for name in channels
        ]
        physical_ratios = []
        for diagnostic in SELECTION_DIAGNOSTICS:
            key = f"physical_{diagnostic}"
            baseline_value = float(linear[key])
            if not math.isfinite(baseline_value) or baseline_value <= 0.0:
                raise ValueError(
                    f"{path}: invalid bilinear {key} at tau={tau}"
                )
            physical_ratios.append(float(model[key]) / baseline_value)
        per_tau[tau] = {
            "rmse": rmse,
            "linear_rmse": linear_rmse,
            "skill": 1.0 - rmse / linear_rmse,
            "acc": float(model["acc_mean"]),
            "physical_ratio": _mean(physical_ratios),
            "physical_ratio_max": max(physical_ratios),
            "field_bilinear_ratio_max": max(channel_bilinear_ratios),
        }

    def group(hours: list[int], key: str) -> float:
        return _mean([per_tau[hour][key] for hour in hours])

    window_file = payload.get("window_metrics_file")
    if not window_file:
        raise ValueError(f"{path}: missing paired window_metrics_file")
    with np.load(path.parent / window_file, allow_pickle=False) as windows:
        window_tau = np.asarray(windows["tau"])
        window_year = np.asarray(windows["year"])
        window_t0 = np.asarray(windows["t0"])
        model_mse = np.asarray(windows["mse_norm_model"], dtype=np.float64)
        linear_mse = np.asarray(windows["mse_norm_bilinear"], dtype=np.float64)
    index_text = "\n".join(
        f"{int(year)},{int(t0)},{int(tau)}"
        for year, t0, tau in zip(window_year, window_t0, window_tau)
    )
    window_index_sha256 = hashlib.sha256(
        index_text.encode("utf-8")
    ).hexdigest()
    unseen_mask = np.isin(window_tau, unseen)
    if not unseen_mask.any():
        raise ValueError(f"{path}: no unseen windows")

    def robust_scores(mse: np.ndarray) -> tuple[float, float]:
        values = np.sqrt(np.maximum(mse.mean(axis=1), 0.0))
        unseen_values = values[unseen_mask]
        threshold = np.quantile(unseen_values, 0.95)
        cvar95 = float(unseen_values[unseen_values >= threshold].mean())
        season_values: dict[str, list[float]] = {
            season: []
            for season in ("DJF", "MAM", "JJA", "SON")
        }
        for year, t0, value, keep in zip(
            window_year,
            window_t0,
            values,
            unseen_mask,
        ):
            if not keep:
                continue
            month = (datetime(int(year), 1, 1) + timedelta(hours=int(t0))).month
            season = (
                "DJF"
                if month in (12, 1, 2)
                else "MAM"
                if month in (3, 4, 5)
                else "JJA"
                if month in (6, 7, 8)
                else "SON"
            )
            season_values[season].append(float(value))
        non_empty = [values for values in season_values.values() if values]
        if not non_empty:
            raise ValueError(f"{path}: no seasonal unseen windows")
        worst_season = max(_mean(values) for values in non_empty)
        return cvar95, worst_season

    model_cvar95, model_worst_season = robust_scores(model_mse)
    linear_cvar95, linear_worst_season = robust_scores(linear_mse)
    return {
        "seen_rmse": group(seen, "rmse"),
        "unseen_rmse": group(unseen, "rmse"),
        "rmse_per_tau": {
            hour: per_tau[hour]["rmse"]
            for hour in sorted(per_tau)
        },
        "rmse_per_tau_per_channel": {
            tau: {
                name: float(
                    payload["per_tau"][str(tau)]["model"][
                        f"rmse_norm_{name}"
                    ]
                )
                for name in channels
            }
            for tau in sorted(per_tau)
        },
        "seen_skill": group(seen, "skill"),
        "unseen_skill": group(unseen, "skill"),
        "seen_acc": group(seen, "acc"),
        "unseen_acc": group(unseen, "acc"),
        "seen_physical_ratio": group(seen, "physical_ratio"),
        "unseen_physical_ratio": group(unseen, "physical_ratio"),
        "seen_physical_ratio_max": max(
            per_tau[hour]["physical_ratio_max"] for hour in seen
        ),
        "unseen_physical_ratio_max": max(
            per_tau[hour]["physical_ratio_max"] for hour in unseen
        ),
        "all_physical_ratio_max": max(
            cell["physical_ratio_max"] for cell in per_tau.values()
        ),
        "all_field_bilinear_ratio_max": max(
            cell["field_bilinear_ratio_max"] for cell in per_tau.values()
        ),
        "unseen_cvar95_rmse": model_cvar95,
        "unseen_cvar95_skill": 1.0 - model_cvar95 / linear_cvar95,
        "unseen_worst_season_rmse": model_worst_season,
        "unseen_worst_season_skill": (
            1.0 - model_worst_season / linear_worst_season
        ),
        "skill_gap": group(seen, "skill") - group(unseen, "skill"),
        "window_index_sha256": window_index_sha256,
        "evaluation_full_year": bool(
            evaluation_protocol.get("full_year", False)
        ),
        "evaluation_index_sha256": evaluation_protocol.get("index_sha256"),
        "evaluation_num_samples": int(payload.get("num_samples", 0)),
        "evaluation_input_provenance": input_provenance,
        "evaluation_dataset_provenance": dataset_provenance,
        "checkpoint_sha256": checkpoint_provenance["sha256"],
    }


def load_spectral_metrics(
    root: Path,
    model: str,
    hf_ell_min: int,
    taus: set[int] | None = None,
    *,
    required_lmax: int | None = None,
    expected_channels: int | None = None,
) -> dict[str, Any]:
    log_energy_errors: list[float] = []
    log_shape_errors: list[float] = []
    aggregate_log_shape_errors: list[float] = []
    ratios: list[float] = []
    coherences: list[float] = []
    sample_count = 0
    window_indices: list[tuple[int, int, int]] = []
    found_taus: set[int] = set()
    spectral_grid: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None
    input_provenance: dict[str, Any] | None = None
    dataset_provenance: dict[str, Any] | None = None
    checkpoint_sha256: str | None = None
    for path in sorted(root.glob(f"{model}_tau*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            tau = int(payload["tau"])
            if taus is not None and tau not in taus:
                continue
            if tau in found_taus:
                raise ValueError(f"{model}: duplicate spectral tau={tau}")
            found_taus.add(tau)
            ell = np.asarray(payload["ell"])
            pred = np.asarray(payload["pred_El"], dtype=np.float64)
            truth = np.asarray(payload["gt_El"], dtype=np.float64)
            n_samples = int(payload["n_samples"])
            window_year = np.asarray(payload["window_year"])
            window_t0 = np.asarray(payload["window_t0"])
            window_coherence = (
                np.asarray(payload["window_hf_coherence"], dtype=np.float64)
                if "window_hf_coherence" in payload
                else None
            )
            window_shape_error = (
                np.asarray(
                    payload["window_hf_log_shape_error"],
                    dtype=np.float64,
                )
                if "window_hf_log_shape_error" in payload
                else None
            )
            coherence_l = (
                np.asarray(payload["coherence_l"], dtype=np.float64)
                if "coherence_l" in payload
                else None
            )
            if "metadata_json" not in payload:
                raise ValueError(f"{path}: missing metadata_json")
            metadata = json.loads(str(payload["metadata_json"].item()))
            current_input_provenance = metadata.get(
                "evaluation_input_provenance"
            )
            current_dataset_provenance = _validated_dataset_provenance(
                metadata.get("evaluation_dataset_provenance"),
                label=str(path),
            )
            current_checkpoint = metadata.get("checkpoint_provenance")
            current_checkpoint_sha256 = (
                current_checkpoint.get("sha256")
                if isinstance(current_checkpoint, dict)
                else None
            )
            if (
                not isinstance(current_checkpoint_sha256, str)
                or not current_checkpoint_sha256
            ):
                raise ValueError(f"{path}: missing checkpoint content hash")
            required_inputs = {
                "static_features",
                "pressure_level_stats",
                "surface_stats",
            }
            if (
                not isinstance(current_input_provenance, dict)
                or not required_inputs.issubset(current_input_provenance)
            ):
                raise ValueError(
                    f"{path}: missing evaluation input provenance"
                )
            if input_provenance is None:
                input_provenance = current_input_provenance
            elif input_provenance != current_input_provenance:
                raise ValueError(
                    f"{model}: spectral input provenance mismatch across taus"
                )
            if dataset_provenance is None:
                dataset_provenance = current_dataset_provenance
            elif dataset_provenance != current_dataset_provenance:
                raise ValueError(
                    f"{model}: spectral dataset provenance mismatch across taus"
                )
            if checkpoint_sha256 is None:
                checkpoint_sha256 = current_checkpoint_sha256
            elif checkpoint_sha256 != current_checkpoint_sha256:
                raise ValueError(
                    f"{model}: spectral checkpoint mismatch across taus"
                )
            if ell.ndim != 1 or ell.size == 0:
                raise ValueError(f"{path}: ell must be a non-empty vector")
            if pred.ndim != 2 or pred.shape != truth.shape:
                raise ValueError(
                    f"{path}: pred_El and gt_El must have matching 2-D shapes"
                )
            if pred.shape[1] != ell.size:
                raise ValueError(f"{path}: spectral degree dimension mismatch")
            if not np.all(np.diff(ell) > 0):
                raise ValueError(f"{path}: ell must be strictly increasing")
            if required_lmax is not None:
                required_grid = np.arange(required_lmax + 1)
                if (
                    ell.size < required_grid.size
                    or not np.array_equal(ell[: required_grid.size], required_grid)
                ):
                    raise ValueError(
                        f"{path}: spectrum does not cover every degree "
                        f"0..{required_lmax}"
                    )
            if spectral_grid is None:
                spectral_grid = ell.copy()
            elif not np.array_equal(spectral_grid, ell):
                raise ValueError(f"{model}: spectral ell-grid mismatch across taus")
            if n_samples <= 0:
                raise ValueError(f"{path}: n_samples must be positive")
            if window_year.size != n_samples or window_t0.size != n_samples:
                raise ValueError(
                    f"{path}: window index length does not match n_samples"
                )
            expected_window_shape = (n_samples, pred.shape[0])
            if (
                window_coherence is not None
                and window_coherence.shape != expected_window_shape
            ):
                raise ValueError(
                    f"{path}: window coherence must have shape "
                    f"{expected_window_shape}"
                )
            if window_coherence is None:
                raise ValueError(f"{path}: missing per-window HF coherence")
            if window_coherence is not None and (
                not np.all(np.isfinite(window_coherence))
                or np.any(window_coherence < 0.0)
                or np.any(window_coherence > 1.0)
            ):
                raise ValueError(f"{path}: invalid per-window HF coherence")
            if coherence_l is not None and (
                not np.all(np.isfinite(coherence_l))
                or np.any(coherence_l < 0.0)
                or np.any(coherence_l > 1.0)
            ):
                raise ValueError(f"{path}: invalid spectral coherence")
            if window_shape_error is None:
                raise ValueError(f"{path}: missing per-window HF shape error")
            if (
                window_shape_error.shape != expected_window_shape
                or not np.all(np.isfinite(window_shape_error))
                or np.any(window_shape_error < 0.0)
            ):
                raise ValueError(f"{path}: invalid per-window HF shape error")
            current_channel_names = (
                tuple(str(value) for value in payload["channel_names"])
                if "channel_names" in payload
                else None
            )
            if expected_channels is not None:
                if current_channel_names is None:
                    raise ValueError(f"{path}: missing channel_names")
                if len(current_channel_names) != expected_channels:
                    raise ValueError(
                        f"{path}: expected {expected_channels} channels, "
                        f"found {len(current_channel_names)}"
                    )
            if current_channel_names is not None:
                if len(current_channel_names) != pred.shape[0]:
                    raise ValueError(
                        f"{path}: channel_names length does not match spectra"
                    )
                if channel_names is None:
                    channel_names = current_channel_names
                elif channel_names != current_channel_names:
                    raise ValueError(
                        f"{model}: spectral channel order mismatch across taus"
                    )
            sample_count += n_samples
            log_shape_errors.extend(window_shape_error.reshape(-1).tolist())
            window_indices.extend(
                (tau, int(year), int(t0))
                for year, t0 in zip(window_year, window_t0)
            )
        keep = ell >= hf_ell_min
        if not np.any(keep):
            raise ValueError(f"{path}: no ell >= {hf_ell_min}")
        pred_hf = np.maximum(pred[:, keep], 0.0)
        truth_hf = np.maximum(truth[:, keep], 0.0)
        eps = max(float(np.nanmedian(truth_hf)) * 1e-12, 1e-30)
        ratio = (pred_hf.sum(axis=1) + eps) / (truth_hf.sum(axis=1) + eps)
        ratios.extend(ratio.tolist())
        log_energy_errors.extend(np.abs(np.log(ratio)).tolist())
        aggregate_log_shape_errors.extend(
            np.mean(
                np.abs(np.log(pred_hf + eps) - np.log(truth_hf + eps)),
                axis=1,
            ).tolist()
        )
        if window_coherence.size:
            coherences.extend(window_coherence.reshape(-1).tolist())
        else:
            raise ValueError(f"{path}: empty per-window HF coherence")
    if taus is not None and found_taus != taus:
        missing = sorted(taus - found_taus)
        unexpected = sorted(found_taus - taus)
        raise FileNotFoundError(
            f"{model}: incomplete spectral taus; "
            f"missing={missing}, unexpected={unexpected}"
        )
    if not ratios:
        raise FileNotFoundError(f"no spectra found for {model} under {root}")
    assert spectral_grid is not None
    assert input_provenance is not None
    assert dataset_provenance is not None
    assert checkpoint_sha256 is not None
    return {
        "hf_energy_ratio": _mean(ratios),
        "hf_log_energy_error": _mean(log_energy_errors),
        "hf_log_energy_error_max": max(log_energy_errors),
        "hf_log_shape_error": _mean(log_shape_errors),
        "hf_log_shape_error_p95": float(
            np.quantile(np.asarray(log_shape_errors), 0.95)
        ),
        "hf_aggregate_log_shape_error": _mean(
            aggregate_log_shape_errors
        ),
        "hf_coherence": _mean(coherences),
        "hf_coherence_p05": float(
            np.quantile(np.asarray(coherences), 0.05)
        ),
        "spectral_sample_count": int(sample_count),
        "spectral_taus": sorted(found_taus),
        "spectral_grid_sha256": hashlib.sha256(
            ",".join(str(int(value)) for value in spectral_grid).encode("utf-8")
        ).hexdigest(),
        "spectral_channel_names_sha256": (
            hashlib.sha256("\n".join(channel_names).encode("utf-8")).hexdigest()
            if channel_names is not None
            else None
        ),
        "window_index_sha256": hashlib.sha256(
            "\n".join(
                f"{tau},{year},{t0}"
                for tau, year, t0 in window_indices
            ).encode("utf-8")
        ).hexdigest(),
        "evaluation_input_provenance": input_provenance,
        "evaluation_dataset_provenance": dataset_provenance,
        "checkpoint_sha256": checkpoint_sha256,
    }


def _rank(values: dict[str, float], reverse: bool = False) -> dict[str, float]:
    """Return average ranks so exact ties do not depend on model order."""
    ordered = sorted(
        values.items(),
        key=lambda item: item[1],
        reverse=reverse,
    )
    ranks: dict[str, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for name, _ in ordered[start:end]:
            ranks[name] = average_rank
        start = end
    return ranks


def pareto_front(rows: dict[str, dict[str, Any]]) -> list[str]:
    objectives = (
        ("rmse_2020_unseen", False),
        ("acc_2020_unseen", True),
        ("hf_log_energy_error", False),
        ("hf_log_shape_error", False),
        ("hf_coherence", True),
        ("physical_2020_all_max", False),
        ("field_2020_all_bilinear_ratio_max", False),
        ("worst_hour_rmse_ratio_to_best_2020", False),
        ("tail_rmse_2020_unseen", False),
        ("worst_season_rmse_2020_unseen", False),
        ("params_m", False),
        ("latency_ms", False),
        ("peak_memory_mib", False),
    )
    front: list[str] = []
    for name, row in rows.items():
        dominated = False
        for other_name, other in rows.items():
            if other_name == name:
                continue
            no_worse = True
            strictly_better = False
            for key, maximize in objectives:
                left = other[key]
                right = row[key]
                if maximize:
                    no_worse &= left >= right
                    strictly_better |= left > right
                else:
                    no_worse &= left <= right
                    strictly_better |= left < right
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(name)
    return sorted(front)


def select(
    rows: dict[str, dict[str, Any]],
) -> tuple[str, list[str], dict[str, Any]]:
    def mean_ranks(
        names: list[str],
    ) -> tuple[dict[str, float], dict[str, float]]:
        quality_specs = (
            ("rmse_2020_unseen", False),
            ("acc_2020_unseen", True),
            ("hf_log_energy_error", False),
            ("hf_log_shape_error", False),
            ("hf_coherence", True),
            ("physical_2020_all_max", False),
            ("field_2020_all_bilinear_ratio_max", False),
            ("tail_rmse_2020_unseen", False),
            ("worst_season_rmse_2020_unseen", False),
        )
        efficiency_specs = (
            ("params_m", False),
            ("latency_ms", False),
            ("peak_memory_mib", False),
        )
        quality_ranks = [
            _rank(
                {name: rows[name][key] for name in names},
                reverse=reverse,
            )
            for key, reverse in quality_specs
        ]
        efficiency_ranks = [
            _rank(
                {name: rows[name][key] for name in names},
                reverse=reverse,
            )
            for key, reverse in efficiency_specs
        ]
        return (
            {
                name: _mean([ranking[name] for ranking in quality_ranks])
                for name in names
            },
            {
                name: _mean(
                    [ranking[name] for ranking in efficiency_ranks]
                )
                for name in names
            },
        )

    all_names = list(rows)
    global_quality_rank, global_efficiency_rank = mean_ranks(all_names)
    for name, row in rows.items():
        row["quality_mean_rank"] = global_quality_rank[name]
        row["efficiency_mean_rank"] = global_efficiency_rank[name]
        # Backward-compatible key used by delayed UPR seed/transfer queues.
        row["mean_rank"] = row["quality_mean_rank"]

    tau_sets = {
        tuple(sorted(int(tau) for tau in row["rmse_2020_per_tau"]))
        for row in rows.values()
    }
    if len(tau_sets) != 1:
        raise ValueError("models do not cover the same 2020 tau schedule")
    all_taus = next(iter(tau_sets))
    if not all_taus:
        raise ValueError("2020 tau schedule is empty")
    safety_eligible = [
        name
        for name, row in rows.items()
        if abs(row["skill_gap_2020"]) <= 0.05
        and row["skill_2020_unseen"] > 0.0
        and row["tail_skill_2020_unseen"] > 0.0
        and row["worst_season_skill_2020_unseen"] > 0.0
        and row["physical_2020_all_max"] <= 1.05
        and row["field_2020_all_bilinear_ratio_max"] <= 1.05
    ]
    for name, row in rows.items():
        row["absolute_safety_gate_passed"] = name in safety_eligible
    if not safety_eligible:
        raise ValueError(
            "no model passed the all-hour RMSE/physical safety gate "
            "(absolute stage)"
        )

    raw_field_hour_maps = {
        name: row.get("rmse_2020_per_tau_per_channel")
        for name, row in rows.items()
    }
    maps_present = {
        name: isinstance(value, dict)
        for name, value in raw_field_hour_maps.items()
    }
    if any(maps_present.values()) and not all(maps_present.values()):
        raise ValueError(
            "field-by-hour RMSE is present for only part of the model set"
        )
    field_hour_diagnostics: dict[str, Any] = {
        "active": all(maps_present.values()),
        "benchmark_population": "absolute_safety_eligible_only",
    }
    if all(maps_present.values()):
        field_hour_maps: dict[str, dict[int, dict[str, float]]] = {}
        common_cells: set[tuple[int, str]] | None = None
        for name, raw_map in raw_field_hour_maps.items():
            assert isinstance(raw_map, dict)
            normalized: dict[int, dict[str, float]] = {}
            cells: set[tuple[int, str]] = set()
            for tau_raw, channel_map_raw in raw_map.items():
                tau = int(tau_raw)
                if not isinstance(channel_map_raw, dict):
                    raise TypeError(
                        f"{name}: invalid field-by-hour RMSE at tau={tau}"
                    )
                channel_map = {
                    str(channel): float(value)
                    for channel, value in channel_map_raw.items()
                }
                if not channel_map or any(
                    not math.isfinite(value) or value <= 0.0
                    for value in channel_map.values()
                ):
                    raise ValueError(
                        f"{name}: invalid field-by-hour RMSE at tau={tau}"
                    )
                normalized[tau] = channel_map
                cells.update((tau, channel) for channel in channel_map)
            if set(normalized) != set(all_taus):
                raise ValueError(
                    f"{name}: field-by-hour tau schedule mismatch"
                )
            if common_cells is None:
                common_cells = cells
            elif cells != common_cells:
                raise ValueError(
                    "models do not cover the same field-by-hour cells"
                )
            field_hour_maps[name] = normalized
        if not common_cells:
            raise ValueError("field-by-hour RMSE cell set is empty")

        best_by_cell: dict[tuple[int, str], float] = {}
        winners_by_cell: dict[tuple[int, str], list[str]] = {}
        for tau, channel in sorted(common_cells):
            values = {
                name: field_hour_maps[name][tau][channel]
                for name in safety_eligible
            }
            best = min(values.values())
            best_by_cell[(tau, channel)] = best
            winners_by_cell[(tau, channel)] = sorted(
                name
                for name, value in values.items()
                if math.isclose(value, best, rel_tol=1.0e-12, abs_tol=0.0)
            )

        total_cells = len(common_cells)
        for name, row in rows.items():
            ratios = [
                field_hour_maps[name][tau][channel]
                / best_by_cell[(tau, channel)]
                for tau, channel in sorted(common_cells)
            ]
            wins = sum(
                name in winners_by_cell[(tau, channel)]
                for tau, channel in common_cells
            )
            row["field_hour_win_count_2020"] = wins
            row["field_hour_total_2020"] = total_cells
            row["field_hour_win_fraction_2020"] = wins / total_cells
            row["field_hour_mean_ratio_to_best_2020"] = _mean(ratios)
            row["field_hour_worst_ratio_to_best_2020"] = max(ratios)
            row["field_hour_uniform_winner_2020"] = wins == total_cells
        field_hour_diagnostics.update(
            {
                "total_cells": total_cells,
                "best_rmse": {
                    str(tau): {
                        channel: best_by_cell[(tau, channel)]
                        for candidate_tau, channel in sorted(common_cells)
                        if candidate_tau == tau
                    }
                    for tau in all_taus
                },
                "winners": {
                    str(tau): {
                        channel: winners_by_cell[(tau, channel)]
                        for candidate_tau, channel in sorted(common_cells)
                        if candidate_tau == tau
                    }
                    for tau in all_taus
                },
                "uniform_winners": sorted(
                    name
                    for name in safety_eligible
                    if rows[name]["field_hour_uniform_winner_2020"]
                ),
            }
        )
    else:
        for row in rows.values():
            row["field_hour_win_count_2020"] = None
            row["field_hour_total_2020"] = None
            row["field_hour_win_fraction_2020"] = None
            row["field_hour_mean_ratio_to_best_2020"] = None
            row["field_hour_worst_ratio_to_best_2020"] = None
            row["field_hour_uniform_winner_2020"] = None

    best_rmse_per_tau: dict[int, float] = {}
    for tau in all_taus:
        values = [
            float(rows[name]["rmse_2020_per_tau"][tau])
            for name in safety_eligible
        ]
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"invalid 2020 RMSE at tau={tau}")
        best_rmse_per_tau[tau] = min(values)
    for row in rows.values():
        row["worst_hour_rmse_ratio_to_best_2020"] = max(
            float(row["rmse_2020_per_tau"][tau])
            / best_rmse_per_tau[tau]
            for tau in all_taus
        )

    best_rmse_2020 = min(
        rows[name]["rmse_2020_unseen"]
        for name in safety_eligible
    )
    field_eligible = [
        name
        for name in safety_eligible
        if (
            rows[name]["worst_hour_rmse_ratio_to_best_2020"] <= 1.05
            and rows[name]["rmse_2020_unseen"] <= 1.03 * best_rmse_2020
        )
    ]
    for name, row in rows.items():
        row["field_gate_passed"] = name in field_eligible
    if not field_eligible:
        raise ValueError(
            "no model passed the all-hour RMSE/physical safety gate"
        )

    best_energy_max = min(
        rows[name]["hf_log_energy_error_max"]
        for name in field_eligible
    )
    best_shape_p95 = min(
        rows[name]["hf_log_shape_error_p95"]
        for name in field_eligible
    )
    best_coherence_p05 = max(
        rows[name]["hf_coherence_p05"]
        for name in field_eligible
    )
    spectral_limits = {
        "hf_log_energy_error_max": best_energy_max * 1.25 + 0.02,
        "hf_log_shape_error_p95": best_shape_p95 * 1.25 + 0.02,
        "hf_coherence_p05": max(0.0, best_coherence_p05 - 0.05),
    }
    eligible = [
        name
        for name in field_eligible
        if (
            rows[name]["hf_log_energy_error_max"]
            <= spectral_limits["hf_log_energy_error_max"]
            and rows[name]["hf_log_shape_error_p95"]
            <= spectral_limits["hf_log_shape_error_p95"]
            and rows[name]["hf_coherence_p05"]
            >= spectral_limits["hf_coherence_p05"]
        )
    ]
    for name, row in rows.items():
        row["spectral_gate_passed"] = name in eligible
    if not eligible:
        raise ValueError(
            "no field-safe model passed the spectral non-inferiority gate"
        )
    selection_quality_rank, selection_efficiency_rank = mean_ranks(eligible)
    for name, row in rows.items():
        row["selection_quality_mean_rank"] = (
            selection_quality_rank.get(name)
        )
        row["selection_efficiency_mean_rank"] = (
            selection_efficiency_rank.get(name)
        )
        if name in selection_quality_rank:
            row["mean_rank"] = selection_quality_rank[name]
    primary_rmse_margin = 0.005
    best_primary_rmse = min(
        rows[name]["rmse_2020_unseen"]
        for name in eligible
    )
    quality_shortlist = [
        name
        for name in eligible
        if rows[name]["rmse_2020_unseen"]
        <= (1.0 + primary_rmse_margin) * best_primary_rmse
    ]
    shortlist_quality_rank, shortlist_efficiency_rank = mean_ranks(
        quality_shortlist
    )
    for name, row in rows.items():
        row["winner_shortlist_quality_mean_rank"] = (
            shortlist_quality_rank.get(name)
        )
        row["winner_shortlist_efficiency_mean_rank"] = (
            shortlist_efficiency_rank.get(name)
        )
    winner = min(
        quality_shortlist,
        key=lambda name: (
            rows[name]["winner_shortlist_quality_mean_rank"],
            rows[name]["winner_shortlist_efficiency_mean_rank"],
            name,
        ),
    )
    return winner, eligible, {
        "best_rmse_2020_unseen": best_rmse_2020,
        "best_rmse_2020_per_tau": best_rmse_per_tau,
        "absolute_safety_eligible": sorted(safety_eligible),
        "rmse_benchmark_population": "absolute_safety_eligible_only",
        "robust_skill_relative_to_linear_min": 0.0,
        "worst_hour_rmse_relative_margin": 0.05,
        "field_bilinear_ratio_limit": 1.05,
        "field_eligible": sorted(field_eligible),
        "field_hour_dominance": field_hour_diagnostics,
        "spectral_limits": spectral_limits,
        "spectral_error_relative_margin": 0.25,
        "spectral_error_absolute_margin": 0.02,
        "spectral_coherence_absolute_margin": 0.05,
        "winner_ranking_population": "final_eligible_models_only",
        "primary_quality_metric": "rmse_2020_unseen",
        "primary_rmse_practical_equivalence_margin": (
            primary_rmse_margin
        ),
        "best_primary_rmse": best_primary_rmse,
        "quality_shortlist": sorted(quality_shortlist),
    }


def build_report(
    models: tuple[str, ...],
    root_2020: Path,
    root_2021: Path | None,
    spectra_root: Path,
    hf_ell_min: int,
    model_params_m: dict[str, float] | None = None,
    model_costs: dict[str, dict[str, float]] | None = None,
    spectral_taus: set[int] | None = None,
    required_spectral_lmax: int | None = None,
    expected_spectral_channels: int | None = None,
) -> dict[str, Any]:
    if model_params_m is None:
        model_params_m = {model: 1.0 for model in models}
    if model_costs is None:
        model_costs = {
            model: {
                "latency_ms": 1.0,
                "peak_memory_mib": 1.0,
            }
            for model in models
        }
    rows: dict[str, dict[str, Any]] = {}
    field_indices_2020: dict[str, str] = {}
    field_indices_2021: dict[str, str] = {}
    spectral_indices: dict[str, str] = {}
    spectral_grid_hashes: dict[str, str] = {}
    spectral_channel_hashes: dict[str, str | None] = {}
    field_provenance_2020: dict[str, dict[str, Any]] = {}
    field_provenance_2021: dict[str, dict[str, Any]] = {}
    spectral_provenance: dict[str, dict[str, Any]] = {}
    field_dataset_2020: dict[str, dict[str, Any]] = {}
    field_dataset_2021: dict[str, dict[str, Any]] = {}
    spectral_dataset_2020: dict[str, dict[str, Any]] = {}
    for model in models:
        try:
            params_m = float(model_params_m[model])
            latency_ms = float(model_costs[model]["latency_ms"])
            peak_memory_mib = float(
                model_costs[model]["peak_memory_mib"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{model}: incomplete cost metrics") from error
        if (
            not math.isfinite(params_m)
            or params_m <= 0.0
            or not math.isfinite(latency_ms)
            or latency_ms <= 0.0
            or not math.isfinite(peak_memory_mib)
            or peak_memory_mib <= 0.0
        ):
            raise ValueError(f"{model}: invalid cost metrics")
        metrics_2020 = load_field_metrics(root_2020 / f"{model}.json")
        metrics_2021 = (
            load_field_metrics(root_2021 / f"{model}.json")
            if root_2021 is not None
            else None
        )
        year_metrics = [(2020, metrics_2020)]
        if metrics_2021 is not None:
            year_metrics.append((2021, metrics_2021))
        for year, metrics in year_metrics:
            if not metrics["evaluation_full_year"]:
                raise ValueError(
                    f"{model}: {year} field metrics are not full-year"
                )
        spectra = load_spectral_metrics(
            spectra_root,
            model,
            hf_ell_min,
            taus=spectral_taus,
            required_lmax=required_spectral_lmax,
            expected_channels=expected_spectral_channels,
        )
        field_indices_2020[model] = str(
            metrics_2020.pop("window_index_sha256")
        )
        if metrics_2021 is not None:
            field_indices_2021[model] = str(
                metrics_2021.pop("window_index_sha256")
            )
        spectral_indices[model] = str(
            spectra.pop("window_index_sha256")
        )
        spectral_grid_hashes[model] = str(
            spectra.pop("spectral_grid_sha256")
        )
        spectral_channel_hashes[model] = spectra.pop(
            "spectral_channel_names_sha256"
        )
        field_provenance_2020[model] = metrics_2020.pop(
            "evaluation_input_provenance"
        )
        if metrics_2021 is not None:
            field_provenance_2021[model] = metrics_2021.pop(
                "evaluation_input_provenance"
            )
        spectral_provenance[model] = spectra.pop(
            "evaluation_input_provenance"
        )
        field_dataset_2020[model] = metrics_2020.pop(
            "evaluation_dataset_provenance"
        )
        if metrics_2021 is not None:
            field_dataset_2021[model] = metrics_2021.pop(
                "evaluation_dataset_provenance"
            )
        spectral_dataset_2020[model] = spectra.pop(
            "evaluation_dataset_provenance"
        )
        checkpoint_sha256_2020 = str(
            metrics_2020.pop("checkpoint_sha256")
        )
        spectral_checkpoint_sha256 = str(
            spectra.pop("checkpoint_sha256")
        )
        checkpoint_hashes = {
            checkpoint_sha256_2020,
            spectral_checkpoint_sha256,
        }
        if metrics_2021 is not None:
            checkpoint_hashes.add(
                str(metrics_2021.pop("checkpoint_sha256"))
            )
        if len(checkpoint_hashes) != 1:
            raise ValueError(
                f"{model}: field/spectral checkpoint hash mismatch"
            )
        cost_checkpoint_sha256 = model_costs[model].get(
            "checkpoint_sha256"
        )
        if (
            cost_checkpoint_sha256 is not None
            and str(cost_checkpoint_sha256) != checkpoint_sha256_2020
        ):
            raise ValueError(
                f"{model}: cost/quality checkpoint hash mismatch"
            )
        spectra.pop("spectral_taus")
        row = {
            "checkpoint_sha256": checkpoint_sha256_2020,
            "params_m": params_m,
            "latency_ms": latency_ms,
            "peak_memory_mib": peak_memory_mib,
            "rmse_2020_seen": metrics_2020["seen_rmse"],
            "rmse_2020_unseen": metrics_2020["unseen_rmse"],
            "rmse_2020_per_tau": metrics_2020["rmse_per_tau"],
            "rmse_2020_per_tau_per_channel": (
                metrics_2020["rmse_per_tau_per_channel"]
            ),
            "skill_2020_seen": metrics_2020["seen_skill"],
            "skill_2020_unseen": metrics_2020["unseen_skill"],
            "skill_gap_2020": metrics_2020["skill_gap"],
            "acc_2020_unseen": metrics_2020["unseen_acc"],
            "physical_2020_unseen": metrics_2020["unseen_physical_ratio"],
            "physical_2020_unseen_max": (
                metrics_2020["unseen_physical_ratio_max"]
            ),
            "physical_2020_all_max": (
                metrics_2020["all_physical_ratio_max"]
            ),
            "field_2020_all_bilinear_ratio_max": (
                metrics_2020["all_field_bilinear_ratio_max"]
            ),
            "tail_rmse_2020_unseen": metrics_2020["unseen_cvar95_rmse"],
            "tail_skill_2020_unseen": (
                metrics_2020["unseen_cvar95_skill"]
            ),
            "worst_season_rmse_2020_unseen": (
                metrics_2020["unseen_worst_season_rmse"]
            ),
            "worst_season_skill_2020_unseen": (
                metrics_2020["unseen_worst_season_skill"]
            ),
            **spectra,
        }
        if metrics_2021 is not None:
            row.update(
                {
                    "rmse_2021_seen": metrics_2021["seen_rmse"],
                    "rmse_2021_unseen": metrics_2021["unseen_rmse"],
                    "rmse_2021_per_tau": metrics_2021["rmse_per_tau"],
                    "rmse_2021_per_tau_per_channel": (
                        metrics_2021["rmse_per_tau_per_channel"]
                    ),
                    "skill_2021_seen": metrics_2021["seen_skill"],
                    "skill_2021_unseen": metrics_2021["unseen_skill"],
                    "skill_gap_2021": metrics_2021["skill_gap"],
                    "acc_2021_unseen": metrics_2021["unseen_acc"],
                    "physical_2021_unseen": (
                        metrics_2021["unseen_physical_ratio"]
                    ),
                    "physical_2021_unseen_max": (
                        metrics_2021["unseen_physical_ratio_max"]
                    ),
                    "physical_2021_all_max": (
                        metrics_2021["all_physical_ratio_max"]
                    ),
                    "field_2021_all_bilinear_ratio_max": (
                        metrics_2021["all_field_bilinear_ratio_max"]
                    ),
                    "tail_rmse_2021_unseen": (
                        metrics_2021["unseen_cvar95_rmse"]
                    ),
                    "tail_skill_2021_unseen": (
                        metrics_2021["unseen_cvar95_skill"]
                    ),
                    "worst_season_rmse_2021_unseen": (
                        metrics_2021["unseen_worst_season_rmse"]
                    ),
                    "worst_season_skill_2021_unseen": (
                        metrics_2021["unseen_worst_season_skill"]
                    ),
                }
            )
        rows[model] = row
    indexed_artifacts = [
        ("2020 field", field_indices_2020),
        ("spectral", spectral_indices),
    ]
    if root_2021 is not None:
        indexed_artifacts.append(("2021 field", field_indices_2021))
    for label, indices in indexed_artifacts:
        if len(set(indices.values())) != 1:
            raise ValueError(f"{label} window-index mismatch: {indices}")
    if len(set(spectral_grid_hashes.values())) != 1:
        raise ValueError(
            f"spectral ell-grid mismatch: {spectral_grid_hashes}"
        )
    if len(set(spectral_channel_hashes.values())) != 1:
        raise ValueError(
            f"spectral channel-order mismatch: {spectral_channel_hashes}"
        )
    provenance_sets = [
        ("2020 field", field_provenance_2020),
        ("spectral", spectral_provenance),
        ("2020 field dataset", field_dataset_2020),
        ("2020 spectral dataset", spectral_dataset_2020),
    ]
    if root_2021 is not None:
        provenance_sets.extend(
            (
                ("2021 field", field_provenance_2021),
                ("2021 field dataset", field_dataset_2021),
            )
        )
    for label, provenances in provenance_sets:
        hashes = {
            model: _canonical_sha256(provenance)
            for model, provenance in provenances.items()
        }
        if len(set(hashes.values())) != 1:
            raise ValueError(f"{label} input-provenance mismatch: {hashes}")
    field_reference = next(iter(field_provenance_2020.values()))
    if root_2021 is not None and any(
        provenance != field_reference
        for provenance in field_provenance_2021.values()
    ):
        raise ValueError("field input-provenance mismatch across years")
    spectral_reference = next(iter(spectral_provenance.values()))
    for key in (
        "static_features",
        "pressure_level_stats",
        "surface_stats",
    ):
        if spectral_reference[key] != field_reference[key]:
            raise ValueError(
                f"field/spectral input-provenance mismatch for {key}"
            )
    field_dataset_reference_2020 = next(iter(field_dataset_2020.values()))
    spectral_dataset_reference = next(iter(spectral_dataset_2020.values()))
    if field_dataset_reference_2020 != spectral_dataset_reference:
        raise ValueError("field/spectral 2020 dataset-provenance mismatch")
    if field_dataset_reference_2020.get("years") != [2020]:
        raise ValueError("2020 field dataset provenance has wrong year")
    field_dataset_reference_2021 = None
    if root_2021 is not None:
        field_dataset_reference_2021 = next(
            iter(field_dataset_2021.values())
        )
        if field_dataset_reference_2021.get("years") != [2021]:
            raise ValueError("2021 field dataset provenance has wrong year")
        if (
            field_dataset_reference_2020.get("root")
            != field_dataset_reference_2021.get("root")
        ):
            raise ValueError("field dataset roots differ across years")
    winner, eligible, gate_diagnostics = select(rows)
    efficiency_winner = min(
        eligible,
        key=lambda name: (
            rows[name]["selection_efficiency_mean_rank"],
            rows[name]["selection_quality_mean_rank"],
            name,
        ),
    )
    frozen_artifacts = {
        "field_2020": next(iter(field_indices_2020.values())),
        "spectral_2020": next(iter(spectral_indices.values())),
        "spectral_grid": next(iter(spectral_grid_hashes.values())),
        "spectral_channels": next(iter(spectral_channel_hashes.values())),
        "field_input_provenance": _canonical_sha256(field_reference),
        "spectral_input_provenance": _canonical_sha256(
            spectral_reference
        ),
        "field_dataset_2020": _canonical_sha256(
            field_dataset_reference_2020
        ),
        "spectral_dataset_2020": _canonical_sha256(
            spectral_dataset_reference
        ),
    }
    if field_dataset_reference_2021 is not None:
        frozen_artifacts.update(
            {
                "field_2021": next(iter(field_indices_2021.values())),
                "field_dataset_2021": _canonical_sha256(
                    field_dataset_reference_2021
                ),
            }
        )
    return {
        "schema_version": 13,
        "ood_attached_at_selection_time": root_2021 is not None,
        "paired_window_index_sha256": frozen_artifacts,
        "selection_rule": {
            "selection_year": 2020,
            "temporal_ood_confirmation_year": 2021,
            "temporal_ood_used_for_selection": False,
            "gate": (
                "abs(skill_gap_2020) <= 0.05, skill_2020_unseen > 0, "
                "2020 unseen CVaR95 skill > 0 and worst-season skill > 0 "
                "relative to paired linear interpolation, "
                "every physical diagnostic/tau ratio on every 2020 hour "
                "<= 1.05, every channel/tau field RMSE relative to bilinear "
                "<= 1.05, every individual-hour channel-macro RMSE within 5% "
                "of the best model at that hour, and 2020 unseen RMSE within "
                "3% of the best model; "
                "field-by-hour win counts and worst ratios to the best "
                "absolute-safe model are reported as dominance diagnostics; "
                "among field-safe models, worst-channel HF log-energy error "
                "and p95 per-window/channel HF log-shape error must be within "
                "25% plus 0.02 of their best values, and p05 HF coherence "
                "must be within 0.05 of its best value"
            ),
            "gate_policy": "fail_closed",
            "gate_diagnostics": gate_diagnostics,
            "ranking": (
                "models within 0.5% of the best eligible 2020 unseen RMSE "
                "are ranked by mean ordinal rank over 2020 unseen RMSE, "
                "2020 unseen ACC, HF log-energy error, HF log-shape error, "
                "HF coherence, 2020 all-hour worst physical-error ratio, "
                "2020 worst channel/tau RMSE ratio to bilinear, CVaR95, and "
                "worst-season RMSE; parameter count, measured A100 latency, "
                "and peak inference memory break quality ties. Ranks used "
                "for winner selection are recomputed over final eligible "
                "models only"
            ),
            "temporal_ood_policy": (
                (
                    "the selection process did not load 2021 artifacts "
                    "before freezing the winner"
                    if root_2021 is None
                    else
                    "2021 metrics are reported but are not used by the gate, "
                    "ranking, Pareto front, or winner selection"
                )
            ),
            "hf_ell_min": hf_ell_min,
            "required_lmax": required_spectral_lmax,
            "expected_spectral_channels": expected_spectral_channels,
            "spectral_taus": (
                sorted(spectral_taus)
                if spectral_taus is not None
                else "all_available"
            ),
            "gate_relaxed_because_no_model_passed": False,
        },
        "winner": winner,
        "efficiency_winner": efficiency_winner,
        "eligible": sorted(eligible),
        "pareto_population": "final_eligible_models_only",
        "pareto_front": pareto_front(
            {name: rows[name] for name in eligible}
        ),
        "models": rows,
    }


def markdown_report(report: dict[str, Any]) -> str:
    ood_attached = bool(report.get("ood_attached_at_selection_time", True))
    lines = ["# UPR-Lite Candidate Selection", ""]
    if ood_attached:
        lines.extend(
            (
                "| Model | Params M | Latency ms | Peak MiB | "
                "RMSE 2020 unseen | "
                "Worst h/best | Skill 2020 unseen | Gap | RMSE 2021 unseen | "
                "Skill 2021 unseen | ACC20 | ACC21 | HF energy error | "
                "HF shape error | HF coherence | Physics20 all max/Bilinear | "
                "Field20 all max/Bilinear | "
                "CVaR95-21 | Worst season-21 | Quality rank | Cost rank |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            )
        )
    else:
        lines.extend(
            (
                "| Model | Params M | Latency ms | Peak MiB | "
                "RMSE 2020 unseen | "
                "Worst h/best | Skill 2020 unseen | Gap | ACC20 | "
                "HF energy error | HF shape error | HF coherence | "
                "Physics20 all max/Bilinear | "
                "Field20 all max/Bilinear | "
                "CVaR95-20 | Worst season-20 | Quality rank | Cost rank |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                "---:|---:|---:|---:|---:|---:|---:|",
            )
        )
    for name, row in sorted(
        report["models"].items(),
        key=lambda item: item[1]["mean_rank"],
    ):
        prefix = (
            f"| {name} | {row['params_m']:.3f} | "
            f"{row['latency_ms']:.2f} | "
            f"{row['peak_memory_mib']:.0f} | "
            f"{row['rmse_2020_unseen']:.6f} | "
            f"{row['worst_hour_rmse_ratio_to_best_2020']:.4f} | "
            f"{row['skill_2020_unseen']:.4f} | "
            f"{row['skill_gap_2020']:+.4f} | "
        )
        if ood_attached:
            body = (
                f"{row['rmse_2021_unseen']:.6f} | "
                f"{row['skill_2021_unseen']:.4f} | "
                f"{row['acc_2020_unseen']:.6f} | "
                f"{row['acc_2021_unseen']:.6f} | "
                f"{row['hf_log_energy_error']:.4f} | "
                f"{row['hf_log_shape_error']:.4f} | "
                f"{row['hf_coherence']:.4f} | "
                f"{row['physical_2020_all_max']:.4f} | "
                f"{row['field_2020_all_bilinear_ratio_max']:.4f} | "
                f"{row['tail_rmse_2021_unseen']:.6f} | "
                f"{row['worst_season_rmse_2021_unseen']:.6f} | "
            )
        else:
            body = (
                f"{row['acc_2020_unseen']:.6f} | "
                f"{row['hf_log_energy_error']:.4f} | "
                f"{row['hf_log_shape_error']:.4f} | "
                f"{row['hf_coherence']:.4f} | "
                f"{row['physical_2020_all_max']:.4f} | "
                f"{row['field_2020_all_bilinear_ratio_max']:.4f} | "
                f"{row['tail_rmse_2020_unseen']:.6f} | "
                f"{row['worst_season_rmse_2020_unseen']:.6f} | "
            )
        lines.append(
            prefix
            + body
            + f"{row['quality_mean_rank']:.2f} | "
            + f"{row['efficiency_mean_rank']:.2f} |"
        )
    lines.extend(
        (
            "",
            f"Quality winner: **{report['winner']}**",
            "",
            f"Efficiency winner: **{report['efficiency_winner']}**",
            "",
            (
                "The 2021 columns are temporal-OOD diagnostics and did not "
                "affect the winner."
                if ood_attached
                else
                "The winner was frozen before the selection process loaded "
                "any 2021 artifact."
            ),
            "",
            "Pareto front: " + ", ".join(report["pareto_front"]),
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODELS),
        help="Comma-separated eval/model-name prefixes.",
    )
    parser.add_argument(
        "--root-2020",
        type=Path,
        default=Path("metrics/upr_lite_screen_6h_2020"),
    )
    parser.add_argument(
        "--root-2021",
        type=Path,
        default=None,
        help=(
            "Optional compatibility mode that attaches 2021 diagnostics. "
            "Omit this when freezing the winner."
        ),
    )
    parser.add_argument(
        "--spectra-root",
        type=Path,
        default=Path("metrics/upr_lite_screen_spectra_6h_2020"),
    )
    parser.add_argument("--hf-ell-min", type=int, default=90)
    parser.add_argument("--required-lmax", type=int, default=359)
    parser.add_argument("--expected-spectral-channels", type=int, default=24)
    parser.add_argument(
        "--spectral-taus",
        default="",
        help="Comma-separated spectral taus used for ranking; empty uses all.",
    )
    parser.add_argument(
        "--cost-json",
        type=Path,
        default=Path("metrics/upr_lite_inference_cost.json"),
    )
    parser.add_argument(
        "--normalization-manifest",
        type=Path,
        default=Path("data/normalization_provenance.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("metrics/upr_lite_candidate_selection.json"),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("metrics/upr_lite_candidate_selection.md"),
    )
    parser.add_argument(
        "--freeze-output",
        action="store_true",
        help=(
            "Refuse to replace an existing selection manifest unless the "
            "new JSON and Markdown are byte-identical."
        ),
    )
    args = parser.parse_args()

    models = tuple(value.strip() for value in args.models.split(",") if value.strip())
    cost_payload = _validate_cost_artifact(args.cost_json, models)
    model_costs = cost_payload["models"]
    report = build_report(
        models,
        args.root_2020,
        args.root_2021,
        args.spectra_root,
        args.hf_ell_min,
        {
            model: float(model_costs[model]["params_m"])
            for model in models
        },
        model_costs,
        spectral_taus=(
            {
                int(value)
                for value in args.spectral_taus.split(",")
                if value.strip()
            }
            or None
        ),
        required_spectral_lmax=args.required_lmax,
        expected_spectral_channels=args.expected_spectral_channels,
    )
    report["selection_code_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    report["normalization_provenance"] = (
        verify_normalization_provenance(
            args.normalization_manifest,
            Path(__file__).resolve().parents[2],
            (2020, 2021),
        )
    )
    report["cost_artifact"] = {
        "path": str(args.cost_json.resolve()),
        "size_bytes": args.cost_json.stat().st_size,
        "sha256": _file_sha256(args.cost_json),
        "benchmark_code_sha256": cost_payload[
            "evaluation_script_sha256"
        ],
        "static_features_sha256": cost_payload["static_features"][
            "sha256"
        ],
        "checkpoint_sha256_by_model": {
            model: str(model_costs[model]["checkpoint_sha256"])
            for model in models
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(report, indent=2) + "\n"
    markdown_text = markdown_report(report)
    if args.freeze_output and args.out_json.exists():
        if args.out_json.read_text() != json_text:
            raise RuntimeError(
                f"refusing to replace frozen selection {args.out_json}"
            )
        if (
            not args.out_md.exists()
            or args.out_md.read_text() != markdown_text
        ):
            raise RuntimeError(
                f"frozen selection Markdown mismatch: {args.out_md}"
            )
        print(
            json.dumps(
                {
                    "winner": report["winner"],
                    "pareto": report["pareto_front"],
                    "frozen_output_unchanged": True,
                }
            )
        )
        return
    json_temporary = args.out_json.with_suffix(args.out_json.suffix + ".tmp")
    md_temporary = args.out_md.with_suffix(args.out_md.suffix + ".tmp")
    json_temporary.write_text(json_text)
    md_temporary.write_text(markdown_text)
    json_temporary.replace(args.out_json)
    md_temporary.replace(args.out_md)
    print(json.dumps({"winner": report["winner"], "pareto": report["pareto_front"]}))


if __name__ == "__main__":
    main()
