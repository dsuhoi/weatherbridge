#!/usr/bin/env python3
"""Select the journal architecture from matched full-year evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CANDIDATES = ("weatherbridge_detail", "refine", "flow_spectral")
PARAMETERS_M = {
    "weatherbridge_detail": 14.266733,
    "refine": 13.651208,
    "flow_spectral": 14.260565,
    "dcae": 14.365049,
}
METRIC_FILES = {
    6: {
        "weatherbridge_detail": "weatherbridge_detail.json",
        "refine": "refine.json",
        "flow_spectral": "flow_spectral.json",
        "reference": "weatherdcae_14m_6yr.json",
        "reference_name": "weatherdcae_14m_6yr",
        "held": (2, 4),
    },
    12: {
        "weatherbridge_detail": "weatherbridge_detail.json",
        "refine": "refine.json",
        "flow_spectral": "flow_spectral.json",
        "reference": "weatherdcae_14m.json",
        "reference_name": "weatherdcae_14m",
        "held": (4, 6, 8),
    },
}
SPECTRAL_NAMES = {
    "weatherbridge_detail": "weatherbridge",
    "refine": "refine",
    "flow_spectral": "flow_spectral",
}
SPECTRAL_METRICS = (
    "energy_log_error",
    "shape_log_error",
    "coherence",
    "signed_cospectrum",
)
SEEDS = (202707, 202708, 202709)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, context: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context}: expected a finite value")
    return result


def _load_cost_artifact(path: Path, used: dict[str, str]) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: wrong inference-cost schema")
    if "A100" not in str(payload.get("device", "")):
        raise ValueError(f"{path}: expected an A100 benchmark")
    if payload.get("input_seed") != 2027 or payload.get("tau_values") != [
        0.25,
        0.5,
        0.75,
    ]:
        raise ValueError(f"{path}: wrong inference-cost input protocol")

    repo_root = Path(__file__).resolve().parents[2]
    benchmark_source = repo_root / "tools/eval/benchmark_capmatched_inference.py"
    if payload.get("evaluation_script_sha256") != _sha256(benchmark_source):
        raise ValueError(f"{path}: stale inference benchmark source")
    used[str(benchmark_source)] = _sha256(benchmark_source)

    supporting = payload.get("supporting_code_sha256")
    required_supporting = {
        "tools/eval/capmatched_loader.py",
        "tools/train/train_capacity_matched_6h.py",
        "weather_time_interp/model/weatherbridge_flow_model.py",
        "weather_time_interp/model/dcae_adaln_model.py",
    }
    if not isinstance(supporting, dict) or not required_supporting.issubset(
        supporting
    ):
        raise ValueError(f"{path}: incomplete inference supporting-code provenance")
    for source_name, expected in supporting.items():
        source = repo_root / str(source_name)
        if not source.is_file() or _sha256(source) != expected:
            raise ValueError(f"{path}: stale inference source {source}")
        used[str(source)] = str(expected)

    static = payload.get("static_features")
    if not isinstance(static, dict):
        raise TypeError(f"{path}: missing static-feature provenance")
    _validate_file_record(static, f"{path}: static features")
    used[str(static["path"])] = str(static["sha256"])

    models = payload.get("models")
    expected_names = {*CANDIDATES, "dcae"}
    if not isinstance(models, dict) or set(models) != expected_names:
        raise ValueError(f"{path}: wrong inference-cost model set")
    for name in expected_names:
        row = models[name]
        expected_protocol = {
            "batch_size": 1,
            "height": 360,
            "width": 720,
            "warmup": 5,
            "iterations": 20,
            "repeats": 7,
            "input_seed": 2027,
            "tau_values": [0.25, 0.5, 0.75],
        }
        if any(row.get(key) != value for key, value in expected_protocol.items()):
            raise ValueError(f"{path}: wrong {name} inference protocol")
        params_m = _finite(row.get("params_m"), f"{name} parameters")
        if not math.isclose(
            params_m,
            PARAMETERS_M[name],
            rel_tol=0.0,
            abs_tol=5e-7,
        ):
            raise ValueError(f"{path}: {name} parameter count mismatch")
        _finite(row.get("latency_ms"), f"{name} latency")
        checkpoint = Path(str(row.get("checkpoint", "")))
        checkpoint_sha256 = row.get("checkpoint_sha256")
        if (
            not checkpoint.is_file()
            or not checkpoint_sha256
            or _sha256(checkpoint) != checkpoint_sha256
        ):
            raise ValueError(f"{path}: stale {name} inference checkpoint")
        used[str(checkpoint)] = str(checkpoint_sha256)

    used[str(path)] = _sha256(path)
    return payload


def _validate_bound_pairwise_inputs(
    payload: dict[str, Any],
    *,
    artifact_path: Path,
    left_path: Path,
    reference: str,
    right_path: Path,
) -> None:
    if payload.get("left_sha256") != _sha256(left_path):
        raise ValueError(f"{artifact_path}: stale left window metric")
    if payload.get("right_sha256", {}).get(reference) != _sha256(right_path):
        raise ValueError(f"{artifact_path}: stale reference window metric")


def _validate_file_record(record: dict[str, Any], context: str) -> None:
    path_value = record.get("path")
    expected = record.get("sha256")
    if not path_value or not expected:
        raise ValueError(f"{context}: incomplete file provenance")
    path = Path(str(path_value))
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"{context}: stale provenance file {path}")


def _resolve_completion_source(
    marker_path: Path,
    source_name: str,
    expected_sha256: str,
) -> Path:
    source = Path(source_name)
    candidates = [source] if source.is_absolute() else [
        marker_path.parent / "source_snapshot" / source,
        *(parent / source for parent in marker_path.parents),
    ]
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == expected_sha256:
            return candidate.resolve()
    raise ValueError(f"{marker_path}: stale source file {source}")


def _validate_completion_marker(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    if payload.get("status") != "complete":
        raise ValueError(f"{path}: incomplete evaluation marker")
    source_hashes = payload.get("source_sha256")
    if not isinstance(source_hashes, dict):
        source_hashes = payload.get("source_files_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"{path}: missing source-file provenance")
    resolved = {}
    for source_name, expected in source_hashes.items():
        source = _resolve_completion_source(path, str(source_name), str(expected))
        resolved[str(source)] = str(expected)
    return resolved


def _collect_file_records(
    value: Any,
    *,
    used: dict[str, str],
    context: str,
) -> None:
    """Validate and retain nested provenance records with path/SHA-256 pairs."""
    if isinstance(value, dict):
        path_value = value.get("path")
        expected = value.get("sha256")
        if path_value is not None and expected is not None:
            if not path_value or not expected:
                raise ValueError(f"{context}: empty nested file provenance")
            path = Path(str(path_value))
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"{context}: stale nested provenance file {path}")
            used[str(path)] = str(expected)
        for nested in value.values():
            _collect_file_records(nested, used=used, context=context)
    elif isinstance(value, list):
        for nested in value:
            _collect_file_records(nested, used=used, context=context)


def _load_metric(path: Path, horizon: int, year: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    protocol = payload.get("evaluation_protocol", {})
    if payload.get("years") != [year]:
        raise ValueError(f"{path}: wrong evaluation year")
    if float(payload.get("delta_t_hours", -1)) != float(horizon):
        raise ValueError(f"{path}: wrong interpolation horizon")
    if protocol.get("full_year") is not True:
        raise ValueError(f"{path}: refusing non-full-year metrics")
    if protocol.get("latitude_grid") != "wb2_0p25_2x2_block_average_v1":
        raise ValueError(f"{path}: wrong or unverified latitude grid")
    if protocol.get("eval_hours") != list(range(1, horizon)):
        raise ValueError(f"{path}: incomplete query-hour grid")
    if not protocol.get("index_sha256"):
        raise ValueError(f"{path}: missing window-index provenance")
    channels = tuple(str(value) for value in payload.get("channel_names", []))
    if len(channels) != 24:
        raise ValueError(f"{path}: expected 24 canonical fields")
    for tau in range(1, horizon):
        methods = payload.get("per_tau", {}).get(str(tau), {})
        if "model" not in methods:
            raise ValueError(f"{path}: missing model metrics at tau={tau}")
        for channel in channels:
            rmse = _finite(
                methods["model"].get(f"rmse_norm_{channel}"),
                f"{path}: tau={tau} field={channel} RMSE",
            )
            acc = _finite(
                methods["model"].get(f"acc_{channel}"),
                f"{path}: tau={tau} field={channel} ACC",
            )
            if rmse <= 0.0 or not -1.0001 <= acc <= 1.0001:
                raise ValueError(f"{path}: invalid metric range")
    return payload


def _validate_cellwise_grid(
    comparison: dict[str, Any],
    *,
    expected_taus: tuple[int, ...],
    expected_channels: tuple[str, ...],
    context: str,
) -> None:
    per_channel_tau = comparison.get("per_channel_tau")
    if not isinstance(per_channel_tau, dict):
        raise TypeError(f"{context}: missing cellwise field-hour grid")
    expected_tau_keys = {str(value) for value in expected_taus}
    if set(per_channel_tau) != expected_tau_keys:
        raise ValueError(f"{context}: incomplete cellwise hour grid")
    expected_channel_keys = set(expected_channels)
    for tau, channel_rows in per_channel_tau.items():
        if not isinstance(channel_rows, dict) or set(channel_rows) != (
            expected_channel_keys
        ):
            raise ValueError(
                f"{context}: incomplete cellwise field grid at tau={tau}"
            )
    family = comparison.get("cellwise_family", {})
    expected_hypotheses = len(expected_taus) * len(expected_channels)
    if (
        family.get("correction") != "Holm-Bonferroni"
        or _finite(family.get("alpha"), f"{context}: family alpha") != 0.05
        or int(family.get("n_hypotheses", -1)) != expected_hypotheses
    ):
        raise ValueError(f"{context}: incomplete multiplicity family")
    for name in (
        "n_pointwise_left_better",
        "n_pointwise_left_worse",
        "n_significant_left_better_holm",
        "n_significant_left_worse_holm",
    ):
        if name in family and not 0 <= int(family[name]) <= expected_hypotheses:
            raise ValueError(f"{context}: invalid multiplicity count {name}")


def _mean(payload: dict[str, Any], key: str, taus: tuple[int, ...]) -> float:
    channels = [str(value) for value in payload["channel_names"]]
    values = [
        _finite(
            payload["per_tau"][str(tau)]["model"][f"{key}_{channel}"],
            f"tau={tau} field={channel} {key}",
        )
        for tau in taus
        for channel in channels
    ]
    return sum(values) / len(values)


def _load_pairwise(
    path: Path,
    *,
    left: str,
    reference: str,
    index_sha256: str,
    left_path: Path,
    right_path: Path,
    expected_taus: tuple[int, ...],
    expected_channels: tuple[str, ...],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1 or payload.get("left") != left:
        raise ValueError(f"{path}: wrong pairwise identity")
    if payload.get("index_sha256") != index_sha256:
        raise ValueError(f"{path}: pairwise window index mismatch")
    _validate_bound_pairwise_inputs(
        payload,
        artifact_path=path,
        left_path=left_path,
        reference=reference,
        right_path=right_path,
    )
    try:
        comparison = payload["comparisons"][reference]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing cellwise DCAE comparison") from error
    _validate_cellwise_grid(
        comparison,
        expected_taus=expected_taus,
        expected_channels=expected_channels,
        context=f"{path}: RMSE",
    )
    return comparison


def _load_acc_pairwise(
    path: Path,
    *,
    left: str,
    reference: str,
    index_sha256: str,
    left_path: Path,
    right_path: Path,
    expected_taus: tuple[int, ...],
    expected_channels: tuple[str, ...],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2 or payload.get("left") != left:
        raise ValueError(f"{path}: wrong ACC pairwise identity")
    if payload.get("metric") != "acc":
        raise ValueError(f"{path}: expected ACC comparison")
    if payload.get("index_sha256") != index_sha256:
        raise ValueError(f"{path}: ACC window index mismatch")
    _validate_bound_pairwise_inputs(
        payload,
        artifact_path=path,
        left_path=left_path,
        reference=reference,
        right_path=right_path,
    )
    try:
        result = payload["comparisons"][reference]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing DCAE ACC comparison") from error
    ci = result.get("delta_ci95", [])
    if len(ci) != 3 or not all(math.isfinite(float(value)) for value in ci):
        raise ValueError(f"{path}: invalid ACC confidence interval")
    _validate_cellwise_grid(
        result,
        expected_taus=expected_taus,
        expected_channels=expected_channels,
        context=f"{path}: ACC",
    )
    return result


def _load_hard_window_pairwise(
    path: Path,
    *,
    left: str,
    reference: str,
    index_sha256: str,
    left_path: Path,
    right_path: Path,
    expected_taus: tuple[int, ...],
    expected_channels: tuple[str, ...],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 1
        or payload.get("left") != left
        or payload.get("metric") != "hard_window_rmse"
        or payload.get("selection_is_model_independent") is not True
        or float(payload.get("quantile", -1.0)) != 0.95
        or payload.get("index_sha256") != index_sha256
    ):
        raise ValueError(f"{path}: wrong hard-window comparison identity")
    _validate_bound_pairwise_inputs(
        payload,
        artifact_path=path,
        left_path=left_path,
        reference=reference,
        right_path=right_path,
    )
    try:
        result = payload["comparisons"][reference]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing hard-window DCAE comparison") from error
    _validate_cellwise_grid(
        result,
        expected_taus=expected_taus,
        expected_channels=expected_channels,
        context=f"{path}: hard-window RMSE",
    )
    return result


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(order) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _load_aux_pairwise(
    path: Path,
    *,
    left: str,
    reference: str,
    metric: str,
    index_sha256: str | None = None,
    left_path: Path | None = None,
    right_path: Path | None = None,
    expected_channels: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("left") != left
        or payload.get("metric") != metric
    ):
        raise ValueError(f"{path}: wrong auxiliary comparison identity")
    if index_sha256 is not None and payload.get("index_sha256") != index_sha256:
        raise ValueError(f"{path}: auxiliary window index mismatch")
    if left_path is not None and right_path is not None:
        _validate_bound_pairwise_inputs(
            payload,
            artifact_path=path,
            left_path=left_path,
            reference=reference,
            right_path=right_path,
        )
    try:
        result = payload["comparisons"][reference]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path}: missing DCAE auxiliary comparison") from error

    if metric == "temporal_curvature":
        ci = result.get("delta_ci95", [])
        if len(ci) != 3 or not all(math.isfinite(float(value)) for value in ci):
            raise ValueError(f"{path}: invalid temporal confidence interval")
        if expected_channels is None:
            raise ValueError(f"{path}: missing expected temporal fields")
        _validate_cellwise_grid(
            result,
            expected_taus=(0,),
            expected_channels=expected_channels,
            context=f"{path}: temporal curvature",
        )
        family = result["cellwise_family"]
        return {
            "diagnostic_count": int(family["n_hypotheses"]),
            "pointwise_improvements": int(
                family.get("n_pointwise_left_better", 0)
            ),
            "all_pointwise_left_better": bool(
                family.get("all_pointwise_left_better", False)
            ),
            "significant_regressions_holm": (
                int(float(ci[0]) > 0.0)
                + int(family["n_significant_left_worse_holm"])
            ),
            "significant_improvements_holm": (
                int(float(ci[2]) < 0.0)
                + int(family["n_significant_left_better_holm"])
            ),
            "delta_ci95": [float(value) for value in ci],
        }

    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, dict) or not diagnostics:
        raise ValueError(f"{path}: missing physical diagnostics")
    p_values = []
    deltas = []
    for name, row in diagnostics.items():
        p_value = _finite(
            row.get("p_paired_block_permutation"),
            f"{path}: {name} physical p-value",
        )
        delta = _finite(
            row.get("delta_left_minus_right"),
            f"{path}: {name} physical delta",
        )
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"{path}: invalid physical p-value")
        p_values.append(p_value)
        deltas.append(delta)
    adjusted = _holm_adjust(p_values)
    return {
        "diagnostic_count": len(diagnostics),
        "pointwise_improvements": sum(delta < 0.0 for delta in deltas),
        "all_pointwise_left_better": all(delta < 0.0 for delta in deltas),
        "significant_regressions_holm": sum(
            delta > 0.0 and p_value < 0.05
            for delta, p_value in zip(deltas, adjusted, strict=True)
        ),
        "significant_improvements_holm": sum(
            delta < 0.0 and p_value < 0.05
            for delta, p_value in zip(deltas, adjusted, strict=True)
        ),
    }


def _load_spectral(
    path: Path,
    *,
    year: int,
    reference: str,
    expected_taus: list[int],
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: wrong spectral summary schema")
    if payload.get("reference") != reference:
        raise ValueError(f"{path}: wrong spectral reference")
    if payload.get("years") != [year] or payload.get("taus") != expected_taus:
        raise ValueError(f"{path}: wrong spectral year or tau grid")
    input_sha256 = payload.get("input_sha256", {})
    if len(input_sha256) != len(expected_taus):
        raise ValueError(f"{path}: incomplete spectral source binding")
    for source_name, expected_hash in input_sha256.items():
        source = Path(source_name)
        if _sha256(source) != expected_hash:
            raise ValueError(f"{path}: stale spectral pairwise artifact {source}")
        pairwise = json.loads(source.read_text())
        left_source = Path(pairwise.get("left_path", ""))
        if pairwise.get("left_sha256") != _sha256(left_source):
            raise ValueError(f"{source}: stale spectral left input")
        for reference_name, right_name in pairwise.get("right_paths", {}).items():
            right_source = Path(right_name)
            if pairwise.get("right_sha256", {}).get(reference_name) != _sha256(
                right_source
            ):
                raise ValueError(f"{source}: stale spectral right input")
    metrics = payload.get("yearly", {}).get(str(year), {})
    wins = 0
    cells = 0
    significant_regressions = 0
    metric_cell_count: int | None = None
    for name in SPECTRAL_METRICS:
        row = metrics.get(name, {})
        cell_count = int(row.get("cell_count", -1))
        if cell_count <= 0:
            raise ValueError(f"{path}: incomplete spectral family {name}")
        family = row.get("multiplicity_family", {})
        if (
            family.get("method") != "Holm-Bonferroni"
            or family.get("dimensions") != "tau_x_channel"
            or int(family.get("n_hypotheses", -1)) != cell_count
            or _finite(family.get("alpha"), f"{path}: spectral alpha") != 0.05
        ):
            raise ValueError(f"{path}: invalid spectral multiplicity family {name}")
        if metric_cell_count is None:
            metric_cell_count = cell_count
        elif cell_count != metric_cell_count:
            raise ValueError(f"{path}: inconsistent spectral family sizes")
        metric_wins = int(row.get("wins", -1))
        if not 0 <= metric_wins <= cell_count:
            raise ValueError(f"{path}: invalid spectral win count {name}")
        wins += metric_wins
        cells += cell_count
        significant_regressions += sum(
            float(failure["p_holm_global_tau_channel"]) < 0.05
            for failure in row.get("failures", [])
        )
    return {
        "win_fraction": wins / cells,
        "wins": wins,
        "cell_count": cells,
        "significant_regressions_holm": significant_regressions,
    }


def _seed_stem(candidate: str, horizon: int, seed: int) -> str:
    if candidate == "weatherbridge_detail":
        infix = "" if horizon == 6 else "_2017_19"
        return f"exp_flow_pp3_detail_14m_{horizon}h{infix}_refinev1_s{seed}_bs4"
    if candidate == "refine":
        infix = "" if horizon == 6 else "_2017_19"
        return f"exp_flow_universal_latent_refine_14m_{horizon}h{infix}_s{seed}_v1_bs4"
    if candidate == "flow_spectral":
        infix = "" if horizon == 6 else "_2017_19"
        return f"exp_flow_pp3_spectral_14m_{horizon}h{infix}_refinev1_s{seed}_bs4"
    if candidate == "dcae":
        infix = "_6yr" if horizon == 6 else "_2017_19"
        return f"exp_weatherdcae_14m_{horizon}h{infix}_refinev1_s{seed}_bs4"
    raise ValueError(f"unknown seed family {candidate}")


def _ifs_mean(path: Path, horizon: int) -> tuple[float, str]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: wrong IFS schema")
    if payload.get("protocol", {}).get("delta_t_hours") != horizon:
        raise ValueError(f"{path}: wrong IFS horizon")
    if payload.get("protocol", {}).get("latitude_grid") != (
        "wb2_0p25_2x2_block_average_v1"
    ):
        raise ValueError(f"{path}: unverified IFS latitude grid")
    if payload.get("protocol", {}).get("area_weighting") != (
        "spherical_latitude_strip_area"
    ):
        raise ValueError(f"{path}: unverified IFS area weighting")
    if payload.get("protocol", {}).get("max_inits") != 16:
        raise ValueError(f"{path}: expected the frozen 16-init IFS subset")
    provenance = payload.get("provenance", {})
    if provenance.get("forecast_anchors", {}).get("init_count") != 16:
        raise ValueError(f"{path}: IFS provenance does not bind 16 inits")
    _validate_file_record(provenance.get("evaluator", {}), f"{path} evaluator")
    evaluation_code = provenance.get("evaluation_code", {})
    if not isinstance(evaluation_code, dict) or not evaluation_code:
        raise ValueError(f"{path}: missing IFS evaluation-code provenance")
    for name, record in evaluation_code.items():
        _validate_file_record(record, f"{path} evaluation code {name}")
    model_artifact = provenance.get("model", {}).get("artifact")
    if not isinstance(model_artifact, dict):
        raise TypeError(f"{path}: missing IFS model-artifact provenance")
    _validate_file_record(model_artifact, f"{path} model artifact")
    paired_record = payload.get("paired_artifact", {})
    _validate_file_record(paired_record, f"{path} paired artifact")
    expected_windows = {6: 3200, 12: 6864}[horizon]
    if paired_record.get("n_windows") != expected_windows:
        raise ValueError(
            f"{path}: expected {expected_windows} paired IFS window-hours"
        )
    paired_path = Path(str(paired_record["path"]))
    with np.load(paired_path, allow_pickle=False) as arrays:
        expected_keys = {
            "init_time_hours",
            "anchor_lead_hours",
            "tau_hours",
            "squared_error_norm",
            "valid_channel",
            "channel_names",
        }
        if set(arrays.files) != expected_keys:
            raise ValueError(f"{path}: malformed paired IFS artifact")
        init = np.asarray(arrays["init_time_hours"])
        lead = np.asarray(arrays["anchor_lead_hours"])
        tau = np.asarray(arrays["tau_hours"])
        squared = np.asarray(arrays["squared_error_norm"])
        valid = np.asarray(arrays["valid_channel"])
        if (
            init.shape != (expected_windows,)
            or lead.shape != (expected_windows,)
            or tau.shape != (expected_windows,)
            or squared.shape != (expected_windows, 24)
            or valid.shape != squared.shape
            or np.asarray(arrays["channel_names"]).shape != (24,)
        ):
            raise ValueError(f"{path}: wrong paired IFS array shape")
        expected_pairs = expected_windows // (horizon - 1)
        values, counts = np.unique(tau, return_counts=True)
        if not np.array_equal(values, np.arange(1, horizon)) or not np.all(
            counts == expected_pairs
        ):
            raise ValueError(f"{path}: incomplete paired IFS tau grid")
        pairs = np.stack((init, lead), axis=1)
        if np.unique(pairs, axis=0).shape[0] != expected_pairs:
            raise ValueError(f"{path}: wrong paired IFS anchor-pair count")
        if not np.all(np.isfinite(squared[valid])):
            raise ValueError(f"{path}: non-finite valid paired IFS errors")
        digest = hashlib.sha256()
        for array in (init, lead, tau):
            contiguous = np.ascontiguousarray(array)
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(str(contiguous.shape).encode("ascii"))
            digest.update(contiguous.view(np.uint8))
        if digest.hexdigest() != paired_record.get("window_index_sha256"):
            raise ValueError(f"{path}: paired IFS index hash mismatch")
    values = []
    for tau in range(1, horizon):
        methods = payload.get("per_tau", {}).get(str(tau), {})
        if len(methods) != 1:
            raise ValueError(f"{path}: ambiguous IFS model identity")
        metrics = next(iter(methods.values()))
        channel_values = [
            _finite(value, f"{path}: IFS RMSE")
            for key, value in metrics.items()
            if key.startswith("rmse_norm_")
        ]
        if len(channel_values) != 24:
            raise ValueError(f"{path}: expected 24 IFS fields at tau={tau}")
        values.extend(channel_values)
    index = paired_record.get("window_index_sha256")
    if not index:
        raise ValueError(f"{path}: missing paired IFS index")
    return sum(values) / len(values), str(index)


def _seed_confirmation(
    root: Path,
    candidate: str,
    used: dict[str, str],
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    missing: list[str] = []
    for horizon in (6, 12):
        deltas = []
        rows = []
        for seed in SEEDS:
            candidate_path = root / f"{horizon}h" / (
                _seed_stem(candidate, horizon, seed) + ".json"
            )
            reference_path = root / f"{horizon}h" / (
                _seed_stem("dcae", horizon, seed) + ".json"
            )
            if not candidate_path.is_file() or not reference_path.is_file():
                missing.extend(
                    str(path)
                    for path in (candidate_path, reference_path)
                    if not path.is_file()
                )
                continue
            candidate_mean, candidate_index = _ifs_mean(candidate_path, horizon)
            reference_mean, reference_index = _ifs_mean(reference_path, horizon)
            _collect_file_records(
                json.loads(candidate_path.read_text()),
                used=used,
                context=str(candidate_path),
            )
            _collect_file_records(
                json.loads(reference_path.read_text()),
                used=used,
                context=str(reference_path),
            )
            if candidate_index != reference_index:
                raise ValueError("paired IFS window-index mismatch")
            delta = candidate_mean / reference_mean - 1.0
            deltas.append(delta)
            rows.append(
                {
                    "seed": seed,
                    "candidate_rmse": candidate_mean,
                    "reference_rmse": reference_mean,
                    "relative_delta": delta,
                    "index_sha256": candidate_index,
                }
            )
            used[str(candidate_path)] = _sha256(candidate_path)
            used[str(reference_path)] = _sha256(reference_path)
        complete = len(rows) == len(SEEDS)
        wins = sum(delta < 0.0 for delta in deltas)
        horizons[str(horizon)] = {
            "complete": complete,
            "rows": rows,
            "wins": wins,
            "mean_relative_delta": (
                sum(deltas) / len(deltas) if deltas else None
            ),
            "pass": bool(
                complete
                and (
                    candidate == "dcae"
                    or (wins >= 2 and sum(deltas) / len(deltas) <= 0.0)
                )
            ),
        }
    return {
        "complete": not missing,
        "missing": sorted(set(missing)),
        "horizons": horizons,
        "pass": all(row["pass"] for row in horizons.values()),
    }


def select_champion(
    detailed_root: Path,
    spectra_root: Path,
    seed_root: Path,
    cost_path: Path,
    *,
    latency_ratio_limit: float = 1.25,
) -> dict[str, Any]:
    used: dict[str, str] = {}
    for marker in (
        detailed_root / "6h/.complete",
        detailed_root / "12h/.complete",
        spectra_root / "state/.complete",
    ):
        if not marker.is_file():
            raise FileNotFoundError(f"missing completion marker {marker}")
        marker_sources = _validate_completion_marker(marker)
        used[str(marker)] = _sha256(marker)
        used.update(marker_sources)
    cost = _load_cost_artifact(cost_path, used)
    cost_models = cost.get("models", {})
    latencies = {
        name: _finite(cost_models.get(name, {}).get("latency_ms"), f"{name} latency")
        for name in CANDIDATES
    }
    fastest_latency = min(latencies.values())
    reference_latency = _finite(
        cost_models.get("dcae", {}).get("latency_ms"),
        "dcae latency",
    )

    rows: dict[str, Any] = {}
    for candidate in CANDIDATES:
        evidence: dict[str, Any] = {}
        selection_failures: list[str] = []
        ood_failures: list[str] = []
        for horizon in (6, 12):
            config = METRIC_FILES[horizon]
            horizon_rows: dict[str, Any] = {}
            for year in (2020, 2021):
                full_year = detailed_root / f"{horizon}h/{year}/full_year"
                candidate_path = full_year / str(config[candidate])
                reference_path = full_year / str(config["reference"])
                candidate_payload = _load_metric(candidate_path, horizon, year)
                reference_payload = _load_metric(reference_path, horizon, year)
                if horizon == 6 and year == 2020:
                    candidate_checkpoint_sha = candidate_payload.get(
                        "checkpoint_provenance", {}
                    ).get("sha256")
                    reference_checkpoint_sha = reference_payload.get(
                        "checkpoint_provenance", {}
                    ).get("sha256")
                    if (
                        candidate_checkpoint_sha
                        != cost_models[candidate]["checkpoint_sha256"]
                    ):
                        raise ValueError(
                            f"{candidate}: inference/evaluation checkpoint mismatch"
                        )
                    if (
                        reference_checkpoint_sha
                        != cost_models["dcae"]["checkpoint_sha256"]
                    ):
                        raise ValueError(
                            "dcae: inference/evaluation checkpoint mismatch"
                        )
                channels = tuple(
                    str(value) for value in candidate_payload["channel_names"]
                )
                _collect_file_records(
                    candidate_payload,
                    used=used,
                    context=str(candidate_path),
                )
                _collect_file_records(
                    reference_payload,
                    used=used,
                    context=str(reference_path),
                )
                candidate_index = candidate_payload["evaluation_protocol"]["index_sha256"]
                reference_index = reference_payload["evaluation_protocol"]["index_sha256"]
                if candidate_index != reference_index:
                    raise ValueError(f"{horizon}h {year}: metric index mismatch")
                candidate_window = full_year / "window_metrics" / f"{candidate}.npz"
                reference_window = full_year / "window_metrics" / (
                    str(config["reference_name"]) + ".npz"
                )
                used[str(candidate_window)] = _sha256(candidate_window)
                used[str(reference_window)] = _sha256(reference_window)
                taus = tuple(range(1, horizon))
                held = tuple(int(value) for value in config["held"])
                candidate_rmse = _mean(candidate_payload, "rmse_norm", taus)
                reference_rmse = _mean(reference_payload, "rmse_norm", taus)
                candidate_held = _mean(candidate_payload, "rmse_norm", held)
                reference_held = _mean(reference_payload, "rmse_norm", held)
                candidate_acc = _mean(candidate_payload, "acc", taus)
                reference_acc = _mean(reference_payload, "acc", taus)

                rmse_path = full_year / f"paired_rmse_{candidate}.json"
                rmse_pair = _load_pairwise(
                    rmse_path,
                    left=candidate,
                    reference=str(config["reference_name"]),
                    index_sha256=candidate_index,
                    left_path=candidate_window,
                    right_path=reference_window,
                    expected_taus=taus,
                    expected_channels=channels,
                )
                hard_path = full_year / f"paired_hard_window_{candidate}.json"
                hard_pair = _load_hard_window_pairwise(
                    hard_path,
                    left=candidate,
                    reference=str(config["reference_name"]),
                    index_sha256=candidate_index,
                    left_path=candidate_window,
                    right_path=reference_window,
                    expected_taus=taus,
                    expected_channels=channels,
                )
                acc_path = full_year / f"paired_acc_{candidate}.json"
                acc_pair = _load_acc_pairwise(
                    acc_path,
                    left=candidate,
                    reference=str(config["reference_name"]),
                    index_sha256=candidate_index,
                    left_path=candidate_window,
                    right_path=reference_window,
                    expected_taus=taus,
                    expected_channels=channels,
                )
                temporal_path = full_year / (
                    f"paired_temporal_curvature_{candidate}.json"
                )
                temporal_pair = _load_aux_pairwise(
                    temporal_path,
                    left=candidate,
                    reference=str(config["reference_name"]),
                    metric="temporal_curvature",
                    left_path=candidate_window,
                    right_path=reference_window,
                    expected_channels=channels,
                )
                physical_path = full_year / f"paired_physical_{candidate}.json"
                physical_pair = _load_aux_pairwise(
                    physical_path,
                    left=candidate,
                    reference=str(config["reference_name"]),
                    metric="physical",
                    index_sha256=candidate_index,
                    left_path=candidate_window,
                    right_path=reference_window,
                )
                spectral_name = SPECTRAL_NAMES[candidate]
                spectral_reference = "weatherdcae_14m"
                spectrum_dir = spectra_root / f"{horizon}h_{year}"
                scalar_path = spectrum_dir / (
                    f"global_scalar_{spectral_name}_vs_{spectral_reference}.json"
                )
                vector_path = spectrum_dir / (
                    f"global_vector_{spectral_name}_vs_{spectral_reference}.json"
                )
                scalar = _load_spectral(
                    scalar_path,
                    year=year,
                    reference=spectral_reference,
                    expected_taus=list(taus),
                )
                vector = _load_spectral(
                    vector_path,
                    year=year,
                    reference=spectral_reference,
                    expected_taus=list(taus),
                )
                for spectral_summary in (scalar_path, vector_path):
                    summary_payload = json.loads(spectral_summary.read_text())
                    for source_name, source_hash in summary_payload[
                        "input_sha256"
                    ].items():
                        source = Path(source_name)
                        used[str(source)] = str(source_hash)
                        pairwise_payload = json.loads(source.read_text())
                        left_source = Path(pairwise_payload["left_path"])
                        used[str(left_source)] = str(
                            pairwise_payload["left_sha256"]
                        )
                        for reference_name, right_name in pairwise_payload[
                            "right_paths"
                        ].items():
                            used[str(Path(right_name))] = str(
                                pairwise_payload["right_sha256"][reference_name]
                            )
                rmse_relative = candidate_rmse / reference_rmse - 1.0
                held_relative = candidate_held / reference_held - 1.0
                acc_delta = candidate_acc - reference_acc
                rmse_regressions = int(
                    rmse_pair["cellwise_family"]["n_significant_left_worse_holm"]
                )
                hard_relative = _finite(
                    hard_pair.get("relative_delta_pct"),
                    f"{hard_path}: hard-window relative delta",
                ) / 100.0
                hard_regressions = int(
                    hard_pair["cellwise_family"]["n_significant_left_worse_holm"]
                )
                acc_significant_regressions = (
                    int(float(acc_pair["delta_ci95"][2]) < 0.0)
                    + int(
                        acc_pair["cellwise_family"][
                            "n_significant_left_worse_holm"
                        ]
                    )
                )
                spectral_regressions = (
                    scalar["significant_regressions_holm"]
                    + vector["significant_regressions_holm"]
                )
                temporal_regressions = int(
                    temporal_pair["significant_regressions_holm"]
                )
                physical_regressions = int(
                    physical_pair["significant_regressions_holm"]
                )
                gates = {
                    "aggregate_rmse_no_worse": rmse_relative <= 0.0,
                    "held_rmse_no_worse": held_relative <= 0.0,
                    "no_rmse_cell_regression_holm": rmse_regressions == 0,
                    "hard_window_rmse_no_worse": hard_relative <= 0.0,
                    "no_hard_window_cell_regression_holm": hard_regressions == 0,
                    "no_acc_regression_holm": acc_significant_regressions == 0,
                    "no_spectral_regression_holm": spectral_regressions == 0,
                    "no_temporal_regression_95ci": temporal_regressions == 0,
                    "no_physical_regression_holm": physical_regressions == 0,
                }
                strict_gates = {
                    "all_rmse_cells_pointwise_better": bool(
                        rmse_pair["cellwise_family"].get(
                            "all_pointwise_left_better", False
                        )
                    ),
                    "all_hard_window_cells_pointwise_better": bool(
                        hard_pair["cellwise_family"].get(
                            "all_pointwise_left_better", False
                        )
                    ),
                    "all_acc_cells_pointwise_better": bool(
                        acc_pair["cellwise_family"].get(
                            "all_pointwise_left_better", False
                        )
                    ),
                    "all_scalar_spectral_cells_pointwise_better": (
                        scalar["wins"] == scalar["cell_count"]
                    ),
                    "all_vector_spectral_cells_pointwise_better": (
                        vector["wins"] == vector["cell_count"]
                    ),
                    "all_temporal_fields_pointwise_better": bool(
                        temporal_pair["all_pointwise_left_better"]
                    ),
                    "all_physical_diagnostics_pointwise_better": bool(
                        physical_pair["all_pointwise_left_better"]
                    ),
                }
                row = {
                    "aggregate_rmse": candidate_rmse,
                    "reference_rmse": reference_rmse,
                    "aggregate_rmse_relative": rmse_relative,
                    "held_rmse_relative": held_relative,
                    "acc_delta": acc_delta,
                    "acc_significant_regressions_holm": (
                        acc_significant_regressions
                    ),
                    "rmse_significant_regressions_holm": rmse_regressions,
                    "hard_window_rmse_relative": hard_relative,
                    "hard_window_significant_regressions_holm": hard_regressions,
                    "spectral_significant_regressions_holm": spectral_regressions,
                    "temporal_significant_regressions": temporal_regressions,
                    "physical_significant_regressions_holm": physical_regressions,
                    "physical_significant_improvements_holm": int(
                        physical_pair["significant_improvements_holm"]
                    ),
                    "scalar_spectral_win_fraction": scalar["win_fraction"],
                    "vector_spectral_win_fraction": vector["win_fraction"],
                    "gates": gates,
                    "pass": all(gates.values()),
                    "strict_dominance_gates": strict_gates,
                    "strict_dominance_pass": all(strict_gates.values()),
                    "index_sha256": candidate_index,
                }
                horizon_rows[str(year)] = row
                failures = selection_failures if year == 2020 else ood_failures
                failures.extend(
                    f"{horizon}h:{name}"
                    for name, passed in gates.items()
                    if not passed
                )
                for path in (
                    candidate_path,
                    reference_path,
                    rmse_path,
                    hard_path,
                    acc_path,
                    temporal_path,
                    physical_path,
                    scalar_path,
                    vector_path,
                ):
                    used[str(path)] = _sha256(path)
            evidence[f"{horizon}h"] = horizon_rows

        latency_ratio = latencies[candidate] / fastest_latency
        compute_pass = (
            PARAMETERS_M[candidate] <= 15.0
            and latency_ratio <= latency_ratio_limit
        )
        if not compute_pass:
            selection_failures.append("compute")
        seed_confirmation = _seed_confirmation(seed_root, candidate, used)
        rows[candidate] = {
            "parameters_m": PARAMETERS_M[candidate],
            "latency_ms": latencies[candidate],
            "latency_ratio_to_fastest": latency_ratio,
            "compute_pass": compute_pass,
            "evidence": evidence,
            "selection_failures": selection_failures,
            "ood_failures": ood_failures,
            "selection_pass": not selection_failures,
            "ood_pass": not ood_failures,
            "seed_confirmation": seed_confirmation,
        }

    eligible = [name for name in CANDIDATES if rows[name]["selection_pass"]]
    reference_confirmation = _seed_confirmation(seed_root, "dcae", used)
    reference = {
        "name": "weatherdcae_14m",
        "parameters_m": PARAMETERS_M["dcae"],
        "latency_ms": reference_latency,
        "seed_confirmation": reference_confirmation,
    }

    def rank(name: str) -> tuple[float, ...]:
        row = rows[name]
        selection = [
            row["evidence"][f"{horizon}h"]["2020"]
            for horizon in (6, 12)
        ]
        return (
            len(row["selection_failures"]),
            sum(item["rmse_significant_regressions_holm"] for item in selection),
            sum(
                item["hard_window_significant_regressions_holm"]
                for item in selection
            ),
            max(item["held_rmse_relative"] for item in selection),
            sum(item["aggregate_rmse_relative"] for item in selection) / 2.0,
            sum(item["spectral_significant_regressions_holm"] for item in selection),
            sum(item["physical_significant_regressions_holm"] for item in selection),
            sum(item["temporal_significant_regressions"] for item in selection),
            -sum(
                item["scalar_spectral_win_fraction"]
                + item["vector_spectral_win_fraction"]
                for item in selection
            ),
            row["latency_ms"],
        )

    diagnostic_leader = min(CANDIDATES, key=rank)
    strict_dominance_candidates = [
        name
        for name in CANDIDATES
        if rows[name]["compute_pass"]
        and rows[name]["seed_confirmation"]["pass"]
        and all(
            rows[name]["evidence"][f"{horizon}h"][str(year)][
                "strict_dominance_pass"
            ]
            for horizon in (6, 12)
            for year in (2020, 2021)
        )
    ]
    strict_dominance_winner = (
        min(strict_dominance_candidates, key=rank)
        if strict_dominance_candidates
        else None
    )
    ood_eligible = [name for name in eligible if rows[name]["ood_pass"]]
    preliminary_winner = min(ood_eligible, key=rank) if ood_eligible else None
    if preliminary_winner is None:
        status = "reference_retained"
        action = "promote_reference"
        winner = "weatherdcae_14m" if reference_confirmation["pass"] else None
    else:
        winner_row = rows[preliminary_winner]
        if not winner_row["seed_confirmation"]["complete"]:
            status = "confirmation_required"
            action = (
                "train_detail_confirmation"
                if preliminary_winner == "weatherbridge_detail"
                else "wait_existing_seed_queue"
            )
            winner = None
        elif not winner_row["seed_confirmation"]["pass"]:
            status = "reference_retained"
            action = "promote_reference"
            winner = "weatherdcae_14m" if reference_confirmation["pass"] else None
        else:
            status = "confirmed"
            action = "promote"
            winner = preliminary_winner

    return {
        "schema_version": 1,
        "selection_rule": {
            "selection_unit": "trained architecture and objective configuration",
            "selection_year": 2020,
            "ood_confirmation_year": 2021,
            "reference": "WeatherDCAE-14M",
            "horizons": [6, 12],
            "held_hours": {"6": [2, 4], "12": [4, 6, 8]},
            "rmse_family": "Holm over all field_x_hour cells per horizon",
            "acc_family": "Holm over all field_x_hour cells per horizon",
            "hard_window_family": (
                "95th-percentile bilinear-hard windows with Holm over all "
                "field_x_hour cells per horizon"
            ),
            "spectral_family": "Holm over all field_x_hour cells per metric",
            "physical_family": "Holm over all physical diagnostics per horizon-year",
            "temporal_rule": (
                "no positive aggregate 95% CI and no Holm-significant "
                "field regression for curvature error"
            ),
            "seed_rule": (
                "paired IFS HRES forecast-anchor win in at least 2/3 seeds "
                "and mean no worse"
            ),
            "latency_ratio_limit_to_fastest": latency_ratio_limit,
            "ranking": "lexicographic gates; no weighted metric score",
            "strict_dominance": (
                "descriptive point-estimate diagnostic: all RMSE, hard-window, "
                "ACC, scalar/vector spectral, temporal, and physical cells "
                "improve in both years and horizons"
            ),
        },
        "candidates": rows,
        "reference": reference,
        "eligible_candidates": eligible,
        "ood_eligible_candidates": ood_eligible,
        "diagnostic_leader": diagnostic_leader,
        "strict_dominance_candidates": strict_dominance_candidates,
        "strict_dominance_winner": strict_dominance_winner,
        "preliminary_winner": preliminary_winner,
        "winner": winner,
        "status": status,
        "action": action,
        "input_sha256": used,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detailed-root", type=Path, required=True)
    parser.add_argument("--spectra-root", type=Path, required=True)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--cost-json", type=Path, required=True)
    parser.add_argument("--latency-ratio-limit", type=float, default=1.25)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.latency_ratio_limit) or args.latency_ratio_limit < 1.0:
        parser.error("--latency-ratio-limit must be finite and at least 1")
    result = select_champion(
        args.detailed_root,
        args.spectra_root,
        args.seed_root,
        args.cost_json,
        latency_ratio_limit=args.latency_ratio_limit,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "action": result["action"],
                "preliminary_winner": result["preliminary_winner"],
                "winner": result["winner"],
                "diagnostic_leader": result["diagnostic_leader"],
            }
        )
    )


if __name__ == "__main__":
    main()
