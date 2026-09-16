#!/usr/bin/env python3
"""Validate that a field-evaluation artifact matches current inputs and code."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.metrics.physical_consistency import (
    DIAGNOSTIC_COMPONENTS,
)
from weather_time_interp.normalization import (
    STATIC_COSINE_LATITUDE_GRID,
    STATIC_FEATURES_3,
)


PHYSICAL_DIAGNOSTICS = (
    "wind_divergence_nmse",
    "wind_vorticity_nmse",
    "kinetic_energy_nmse",
    "hydrostatic_balance_mse",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evaluation_source_paths() -> dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    model_root = repo_root / "weather_time_interp" / "model"
    return {
        "batch_eval_12h_memmap.py": Path(__file__).with_name(
            "batch_eval_12h_memmap.py"
        ),
        "batch_eval_12h_memmap_fast.py": Path(__file__).with_name(
            "batch_eval_12h_memmap_fast.py"
        ),
        "climatology.py": Path(__file__).with_name("climatology.py"),
        "physical_consistency.py": (
            repo_root
            / "weather_time_interp"
            / "metrics"
            / "physical_consistency.py"
        ),
        "grid.py": repo_root / "weather_time_interp" / "grid.py",
        "eval_runner.py": (
            repo_root / "weather_time_interp" / "eval_runner.py"
        ),
        "normalization.py": (
            repo_root / "weather_time_interp" / "normalization.py"
        ),
        "capmatched_loader.py": Path(__file__).with_name(
            "capmatched_loader.py"
        ),
        "train_capacity_matched_6h.py": (
            repo_root / "tools" / "train" / "train_capacity_matched_6h.py"
        ),
        "training_protocol.py": (
            repo_root / "tools" / "train" / "training_protocol.py"
        ),
        "weatherbridge_upr_lite_model.py": (
            model_root / "weatherbridge_upr_lite_model.py"
        ),
        "weatherbridge_upr_scaled_model.py": (
            model_root / "weatherbridge_upr_scaled_model.py"
        ),
        "weatherbridge_upr_spherical_model.py": (
            model_root / "weatherbridge_upr_spherical_model.py"
        ),
        "weatherbridge_flow_model.py": (
            model_root / "weatherbridge_flow_model.py"
        ),
        "weather_amt_model.py": model_root / "weather_amt_model.py",
        "amt_feat_enc.py": model_root / "amt_upstream" / "feat_enc.py",
        "amt_flow_utils.py": (
            model_root / "amt_upstream" / "flow_utils.py"
        ),
        "amt_ifrnet.py": model_root / "amt_upstream" / "ifrnet.py",
        "amt_multi_flow.py": (
            model_root / "amt_upstream" / "multi_flow.py"
        ),
        "amt_raft.py": model_root / "amt_upstream" / "raft.py",
        "dcae_adaln_model.py": model_root / "dcae_adaln_model.py",
        "dcae_adaln_skip_model.py": (
            model_root / "dcae_adaln_skip_model.py"
        ),
        "train_atm_vfi_12h_oddskip.py": (
            repo_root / "legacy" / "scripts" / "train_atm_vfi_12h_oddskip.py"
        ),
    }


def _int_set(value: Any, *, label: str, errors: list[str]) -> set[int]:
    if isinstance(value, (str, bytes, dict)) or value is None:
        errors.append(f"{label} is not an integer sequence")
        return set()
    try:
        return {int(item) for item in value}
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label} contains a non-integer value")
        return set()


def _matches_int(
    value: Any,
    expected: int,
    *,
    label: str,
    errors: list[str],
) -> None:
    try:
        current = int(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{label} is not an integer")
        return
    if current != expected:
        errors.append(f"{label} mismatch")


def _parse_year_window(
    value: Any,
    *,
    label: str,
    errors: list[str],
) -> tuple[int, int] | None:
    if not isinstance(value, str):
        errors.append(f"{label} is not a YYYY-YYYY string")
        return None
    parts = value.split("-")
    if (
        len(parts) != 2
        or any(len(part) != 4 or not part.isdigit() for part in parts)
    ):
        errors.append(f"{label} is not a YYYY-YYYY string")
        return None
    start_year, end_year = map(int, parts)
    if start_year > end_year:
        errors.append(f"{label} starts after it ends")
        return None
    return start_year, end_year


def validate_eval_artifact(
    artifact_path: Path,
    *,
    checkpoint_path: Path,
    required_taus: tuple[int, ...],
    acc_mode: str,
    require_full_year: bool = True,
    require_window_metrics: bool = True,
    require_physical_metrics: bool = False,
    require_temporal_metrics: bool = False,
    require_proper_rmse: bool = True,
    expected_climatology_window: str = "1990-2019",
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return {
            "valid": False,
            "artifact": str(artifact_path),
            "errors": [f"cannot read artifact: {error}"],
        }
    if not isinstance(payload, dict):
        return {
            "valid": False,
            "artifact": str(artifact_path),
            "errors": ["artifact root must be a JSON object"],
        }

    protocol = payload.get("evaluation_protocol")
    if not isinstance(protocol, dict):
        errors.append("missing evaluation_protocol")
        protocol = {}
    if require_full_year and protocol.get("full_year") is not True:
        errors.append("evaluation is not full-year")
    if _int_set(
        protocol.get("eval_hours"),
        label="evaluation eval_hours",
        errors=errors,
    ) != set(required_taus):
        errors.append("evaluation tau set mismatch")
    if not protocol.get("index_sha256"):
        errors.append("missing evaluation index hash")
    if (
        require_proper_rmse
        and protocol.get("rmse_reduction")
        != "spherical_strip_area_weighted_spatial_mean"
    ):
        errors.append("RMSE reduction is not the proper spatial mean")
    try:
        num_samples = int(payload.get("num_samples", 0))
    except (TypeError, ValueError, OverflowError):
        num_samples = 0
        errors.append("num_samples is not an integer")
    if num_samples <= 0:
        errors.append("num_samples must be positive")
    per_tau = payload.get("per_tau")
    if not isinstance(per_tau, dict):
        errors.append("per_tau must be an object")
        per_tau = {}
    if _int_set(
        per_tau.keys(),
        label="per_tau keys",
        errors=errors,
    ) != set(required_taus):
        errors.append("per_tau keys mismatch")
    channel_names = payload.get("channel_names")
    if (
        not isinstance(channel_names, list)
        or not channel_names
        or not all(isinstance(name, str) and name for name in channel_names)
        or len(channel_names) != len(set(channel_names))
    ):
        errors.append("invalid channel_names")
        channel_names = []
    acc_channel_names = payload.get("acc_channel_names")
    if not isinstance(acc_channel_names, list) or not all(
        isinstance(name, str) and name for name in acc_channel_names
    ):
        errors.append("invalid acc_channel_names")
        acc_channel_names = []
    n_per_tau = payload.get("n_per_tau")
    if not isinstance(n_per_tau, dict):
        errors.append("missing n_per_tau")
        n_per_tau = {}
    else:
        counts: list[int] = []
        for tau in required_taus:
            try:
                count = int(n_per_tau[str(tau)])
            except (KeyError, TypeError, ValueError, OverflowError):
                errors.append(f"tau {tau}: invalid sample count")
                continue
            if count <= 0:
                errors.append(f"tau {tau}: sample count must be positive")
            counts.append(count)
        if counts and sum(counts) != num_samples:
            errors.append("n_per_tau does not sum to num_samples")
    for tau in required_taus:
        methods = per_tau.get(str(tau), {})
        if not isinstance(methods, dict):
            methods = {}
        for method in ("model", "bilinear"):
            values = methods.get(method, {})
            if not isinstance(values, dict):
                errors.append(f"tau {tau}: missing {method} metrics")
                continue
            for channel in channel_names:
                for prefix in ("rmse_norm", "rmse_phys"):
                    key = f"{prefix}_{channel}"
                    try:
                        value = float(values[key])
                    except (KeyError, TypeError, ValueError):
                        errors.append(
                            f"tau {tau}: missing {method} {key}"
                        )
                        continue
                    if not math.isfinite(value) or value < 0.0:
                        errors.append(
                            f"tau {tau}: invalid {method} {key}"
                        )

    checkpoint = checkpoint_path.resolve()
    if not checkpoint.is_file():
        errors.append("checkpoint is missing")
    else:
        current_stat = checkpoint.stat()
        provenance = payload.get("checkpoint_provenance")
        if not isinstance(provenance, dict):
            errors.append("missing checkpoint provenance")
        else:
            if provenance.get("path") != str(checkpoint):
                errors.append("checkpoint path mismatch")
            _matches_int(
                provenance.get("size_bytes"),
                current_stat.st_size,
                label="checkpoint size",
                errors=errors,
            )
            _matches_int(
                provenance.get("mtime_ns"),
                current_stat.st_mtime_ns,
                label="checkpoint mtime",
                errors=errors,
            )
            checkpoint_hash = provenance.get("sha256")
            if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
                errors.append("missing checkpoint content hash")
            elif checkpoint_hash != _sha256(checkpoint):
                errors.append("checkpoint content hash mismatch")

    inputs = payload.get("evaluation_input_provenance")
    required_inputs = {
        "static_features",
        "pressure_level_stats",
        "surface_stats",
        "climatology",
    }
    if not isinstance(inputs, dict) or not required_inputs.issubset(inputs):
        errors.append("missing complete evaluation input provenance")
    else:
        for name in (
            "static_features",
            "pressure_level_stats",
            "surface_stats",
        ):
            value = inputs[name]
            if not isinstance(value, dict) or not value.get("sha256"):
                errors.append(f"invalid {name} provenance")
        static_semantic = (
            inputs.get("static_features", {}).get("semantic", {})
            if isinstance(inputs.get("static_features"), dict)
            else {}
        )
        if protocol.get("latitude_grid") == "wb2_0p25_2x2_block_average_v1":
            if (
                static_semantic.get("cosine_latitude_grid")
                != STATIC_COSINE_LATITUDE_GRID
            ):
                errors.append("missing verified static cosine-latitude geometry")
            if (
                static_semantic.get("offset_from_wb2_block_centres_degrees")
                != -0.125
            ):
                errors.append("static cosine-latitude offset mismatch")
        climatology = inputs["climatology"]
        if acc_mode == "enabled":
            if (
                not isinstance(climatology, dict)
                or not climatology.get("cache_identity_sha256")
            ):
                errors.append("missing climatology fingerprint")
            if isinstance(climatology, dict):
                semantic = climatology.get("semantic")
                if not isinstance(semantic, dict):
                    errors.append(
                        "missing climatology semantic provenance"
                    )
                else:
                    expected = _parse_year_window(
                        expected_climatology_window,
                        label="expected climatology window",
                        errors=errors,
                    )
                    declared = _parse_year_window(
                        semantic.get("declared_window"),
                        label="declared climatology window",
                        errors=errors,
                    )
                    if (
                        expected is not None
                        and declared is not None
                        and declared != expected
                    ):
                        errors.append("climatology window mismatch")
                    evaluation_years = _int_set(
                        payload.get("years"),
                        label="artifact years",
                        errors=errors,
                    )
                    semantic_years = _int_set(
                        semantic.get("evaluation_years"),
                        label="climatology evaluation years",
                        errors=errors,
                    )
                    if semantic_years != evaluation_years:
                        errors.append(
                            "climatology evaluation years mismatch"
                        )
                    if semantic.get("precedes_evaluation") is not True:
                        errors.append(
                            "climatology is not marked as preceding evaluation"
                        )
                    if (
                        declared is not None
                        and evaluation_years
                        and declared[1] >= min(evaluation_years)
                    ):
                        errors.append(
                            "climatology overlaps evaluation period"
                        )
        elif climatology is not None:
            errors.append("climatology must be null when ACC is disabled")

    memmap = payload.get("evaluation_dataset_provenance")
    if not isinstance(memmap, dict):
        errors.append("invalid memmap dataset provenance")
    else:
        root = memmap.get("root")
        years = memmap.get("years")
        sample_limit = memmap.get("sampled_bytes_per_file_limit")
        try:
            current_memmap = memmap_dataset_provenance(
                root,
                [int(year) for year in years],
                sampled_bytes_per_file=int(sample_limit),
            )
        except (
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            errors.append(f"cannot verify memmap dataset: {error}")
        else:
            if current_memmap != memmap:
                errors.append("memmap dataset provenance mismatch")
            if current_memmap.get("years") != payload.get("years"):
                errors.append("memmap years mismatch")

    if payload.get("static_feature_names") != list(STATIC_FEATURES_3):
        errors.append("static feature schema mismatch")
    if acc_mode == "enabled":
        if not payload.get("acc_channel_names"):
            errors.append("ACC channels are missing")
        for tau in required_taus:
            methods = per_tau.get(str(tau), {})
            model = (
                methods.get("model", {})
                if isinstance(methods, dict)
                else {}
            )
            try:
                acc = float(model["acc_mean"])
            except (KeyError, TypeError, ValueError):
                errors.append(f"tau {tau}: missing ACC")
                continue
            if not math.isfinite(acc):
                errors.append(f"tau {tau}: non-finite ACC")
            elif abs(acc) > 1.000001:
                errors.append(f"tau {tau}: ACC outside [-1, 1]")
    if require_physical_metrics:
        if protocol.get("save_physical_metrics") is not True:
            errors.append("physical metrics were not requested")
        for tau in required_taus:
            methods = per_tau.get(str(tau), {})
            if not isinstance(methods, dict):
                methods = {}
            for method in ("model", "bilinear"):
                values = methods.get(method, {})
                if not isinstance(values, dict):
                    values = {}
                for diagnostic in PHYSICAL_DIAGNOSTICS:
                    key = f"physical_{diagnostic}"
                    try:
                        value = float(values[key])
                    except (KeyError, TypeError, ValueError):
                        errors.append(f"tau {tau}: missing {method} {key}")
                        continue
                    if not math.isfinite(value):
                        errors.append(
                            f"tau {tau}: non-finite {method} {key}"
                        )
    if require_temporal_metrics:
        if protocol.get("save_temporal_metrics") is not True:
            errors.append("temporal metrics were not requested")
        temporal_summary = payload.get("temporal_curvature_rmse_norm")
        if not isinstance(temporal_summary, dict):
            errors.append("missing temporal curvature summary")
        else:
            for method in ("model", "bilinear"):
                try:
                    value = float(temporal_summary[method])
                except (KeyError, TypeError, ValueError):
                    errors.append(
                        f"missing temporal curvature summary for {method}"
                    )
                    continue
                if not math.isfinite(value) or value < 0.0:
                    errors.append(
                        f"invalid temporal curvature summary for {method}"
                    )

    evaluation_code = payload.get("evaluation_code_provenance")
    source_paths = _evaluation_source_paths()
    if not isinstance(evaluation_code, dict):
        errors.append("evaluation code hash mismatch")
    else:
        for name, expected_hash in evaluation_code.items():
            source_path = source_paths.get(str(name))
            if (
                source_path is None
                or not source_path.is_file()
                or not isinstance(expected_hash, str)
                or _sha256(source_path) != expected_hash
            ):
                errors.append(f"evaluation code hash mismatch: {name}")
        for required_source in (
            "batch_eval_12h_memmap.py",
            "climatology.py",
        ):
            if evaluation_code.get(required_source) != _sha256(
                source_paths[required_source]
            ):
                errors.append(
                    f"evaluation code hash mismatch: {required_source}"
                )

    if require_window_metrics:
        relative = payload.get("window_metrics_file")
        if not relative:
            errors.append("missing window_metrics_file")
        else:
            window_path = artifact_path.parent / str(relative)
            if not window_path.is_file() or window_path.stat().st_size == 0:
                errors.append("window metrics artifact is missing")
            else:
                window_provenance = payload.get(
                    "window_metrics_provenance"
                )
                if not isinstance(window_provenance, dict):
                    errors.append("missing window metrics provenance")
                else:
                    _matches_int(
                        window_provenance.get("size_bytes"),
                        window_path.stat().st_size,
                        label="window metrics size",
                        errors=errors,
                    )
                    window_hash = window_provenance.get("sha256")
                    if not isinstance(window_hash, str) or not window_hash:
                        errors.append("missing window metrics content hash")
                    elif window_hash != _sha256(window_path):
                        errors.append("window metrics content hash mismatch")
                    if (
                        window_provenance.get("index_sha256")
                        != protocol.get("index_sha256")
                    ):
                        errors.append("window metrics index hash mismatch")
                try:
                    with np.load(window_path, allow_pickle=False) as windows:
                        available = set(windows.files)
                        year = np.asarray(windows["year"])
                        t0 = np.asarray(windows["t0"])
                        tau = np.asarray(windows["tau"])
                        window_channels = [
                            str(value)
                            for value in windows["channel_names"]
                        ]
                        window_acc_channels = [
                            str(value)
                            for value in windows["acc_channel_names"]
                        ]
                        arrays = {
                            name: np.asarray(windows[name])
                            for name in available
                        }
                except (OSError, ValueError, KeyError) as error:
                    errors.append(f"cannot read window metrics: {error}")
                else:
                    if not (year.size == t0.size == tau.size):
                        errors.append("window metrics index length mismatch")
                    else:
                        index_text = "\n".join(
                            f"{int(y)},{int(start)},{int(hour)}"
                            for y, start, hour in zip(year, t0, tau)
                        )
                        index_hash = hashlib.sha256(
                            index_text.encode("utf-8")
                        ).hexdigest()
                        if index_hash != protocol.get("index_sha256"):
                            errors.append(
                                "window metrics index content mismatch"
                            )
                        if year.size != num_samples:
                            errors.append(
                                "window metrics sample count mismatch"
                            )
                        if set(map(int, tau.tolist())) != set(
                            required_taus
                        ):
                            errors.append(
                                "window metrics tau set mismatch"
                            )
                    if window_channels != channel_names:
                        errors.append("window metric channel order mismatch")
                    if window_acc_channels != acc_channel_names:
                        errors.append(
                            "window ACC channel order mismatch"
                        )
                    for method in ("model", "bilinear"):
                        mse_key = f"mse_norm_{method}"
                        acc_key = f"acc_{method}"
                        if mse_key not in arrays:
                            errors.append(
                                f"window metrics missing {mse_key}"
                            )
                        else:
                            mse = arrays[mse_key]
                            if mse.shape != (
                                num_samples,
                                len(channel_names),
                            ):
                                errors.append(
                                    f"window metrics {mse_key} shape mismatch"
                                )
                            elif (
                                not np.isfinite(mse).all()
                                or (mse < 0.0).any()
                            ):
                                errors.append(
                                    f"window metrics {mse_key} is invalid"
                                )
                        if acc_key not in arrays:
                            errors.append(
                                f"window metrics missing {acc_key}"
                            )
                        else:
                            acc_values = arrays[acc_key]
                            if acc_values.shape != (
                                num_samples,
                                len(acc_channel_names),
                            ):
                                errors.append(
                                    f"window metrics {acc_key} shape mismatch"
                                )
                            elif (
                                not np.isfinite(acc_values).all()
                                or (np.abs(acc_values) > 1.000001).any()
                            ):
                                errors.append(
                                    f"window metrics {acc_key} is invalid"
                                )
                    if require_physical_metrics:
                        for method in ("model", "bilinear"):
                            for diagnostic in PHYSICAL_DIAGNOSTICS:
                                key = f"physical_{diagnostic}_{method}"
                                values = arrays.get(key)
                                expected_shape = (
                                    num_samples,
                                    len(DIAGNOSTIC_COMPONENTS[diagnostic]),
                                )
                                if values is None:
                                    errors.append(
                                        f"window metrics missing {key}"
                                    )
                                elif values.shape != expected_shape:
                                    errors.append(
                                        f"window metrics {key} shape mismatch"
                                    )
                                elif (
                                    not np.isfinite(values).all()
                                    or (values < 0.0).any()
                                ):
                                    errors.append(
                                        f"window metrics {key} is invalid"
                                    )
                    if require_temporal_metrics:
                        temporal_year = arrays.get("temporal_year")
                        temporal_t0 = arrays.get("temporal_t0")
                        center_count = arrays.get("temporal_center_count")
                        centers = arrays.get("temporal_centers")
                        if (
                            temporal_year is None
                            or temporal_t0 is None
                            or center_count is None
                            or centers is None
                        ):
                            errors.append(
                                "window metrics missing temporal index arrays"
                            )
                        else:
                            expected_windows = (
                                num_samples // len(required_taus)
                                if required_taus
                                else 0
                            )
                            if not (
                                temporal_year.shape
                                == temporal_t0.shape
                                == center_count.shape
                                == (expected_windows,)
                            ):
                                errors.append(
                                    "temporal window index shape mismatch"
                                )
                            expected_centers = np.arange(
                                1,
                                int(round(float(payload["delta_t_hours"]))),
                            )
                            if not np.array_equal(centers, expected_centers):
                                errors.append(
                                    "temporal centre schedule mismatch"
                                )
                            if (
                                center_count.size
                                and not np.all(
                                    center_count == len(expected_centers)
                                )
                            ):
                                errors.append(
                                    "temporal centre count mismatch"
                                )
                            temporal_index_text = "\n".join(
                                f"{int(year_value)},{int(t0_value)}"
                                for year_value, t0_value in zip(
                                    temporal_year,
                                    temporal_t0,
                                )
                            )
                            temporal_index_hash = hashlib.sha256(
                                temporal_index_text.encode("utf-8")
                            ).hexdigest()
                            if temporal_index_hash != protocol.get(
                                "temporal_index_sha256"
                            ):
                                errors.append(
                                    "temporal window index content mismatch"
                                )
                        for method in ("model", "bilinear"):
                            key = f"temporal_curvature_mse_{method}"
                            values = arrays.get(key)
                            expected_shape = (
                                (
                                    num_samples // len(required_taus),
                                    len(channel_names),
                                )
                                if required_taus
                                else (0, len(channel_names))
                            )
                            if values is None:
                                errors.append(
                                    f"window metrics missing {key}"
                                )
                            elif values.shape != expected_shape:
                                errors.append(
                                    f"window metrics {key} shape mismatch"
                                )
                            elif (
                                not np.isfinite(values).all()
                                or (values < 0.0).any()
                            ):
                                errors.append(
                                    f"window metrics {key} is invalid"
                                )

    return {
        "valid": not errors,
        "artifact": str(artifact_path),
        "checkpoint": str(checkpoint),
        "required_taus": list(required_taus),
        "acc_mode": acc_mode,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--required-taus", required=True)
    parser.add_argument(
        "--acc-mode",
        choices=("enabled", "disabled"),
        required=True,
    )
    parser.add_argument("--allow-economy", action="store_true")
    parser.add_argument("--allow-missing-window-metrics", action="store_true")
    parser.add_argument("--allow-legacy-rmse", action="store_true")
    parser.add_argument("--require-physical-metrics", action="store_true")
    parser.add_argument("--require-temporal-metrics", action="store_true")
    parser.add_argument(
        "--expected-climatology-window",
        default="1990-2019",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    result = validate_eval_artifact(
        args.artifact,
        checkpoint_path=args.checkpoint,
        required_taus=tuple(
            int(value)
            for value in args.required_taus.split(",")
            if value.strip()
        ),
        acc_mode=args.acc_mode,
        require_full_year=not args.allow_economy,
        require_window_metrics=not args.allow_missing_window_metrics,
        require_physical_metrics=args.require_physical_metrics,
        require_temporal_metrics=args.require_temporal_metrics,
        require_proper_rmse=not args.allow_legacy_rmse,
        expected_climatology_window=args.expected_climatology_window,
    )
    if not args.quiet or not result["valid"]:
        print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["valid"] else 1)


if __name__ == "__main__":
    main()
