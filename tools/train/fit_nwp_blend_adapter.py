#!/usr/bin/env python3
"""Fit a tiny Linear/model residual adapter for HRES-to-ERA5 interpolation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    FORECAST_LEAD_BINS,
    CheckpointModel,
    Era5Cache,
    LinearInterp,
    _forecast_manifest_provenance,
    forecast_lead_bin,
    read_init,
    select_anchor_pairs,
)
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.grid import (
    latitude_strip_weights,
    wb2_block_average_latitudes,
)

SCALE_CANDIDATES = (1.0, 1.0 / 1.03, 1.0 / 1.1, 1.0 / 1.3, 0.5, 0.25, 0.0)
DEFAULT_MIN_LEAD_RELATIVE_RMSE_GAIN = 1.0e-3
from weather_time_interp.normalization import (
    file_provenance,
    load_channel_stats,
    static_feature_provenance,
)


def _new_moments(taus: list[int], channels: int) -> dict[str, np.ndarray]:
    shape = (max(taus) + 1, channels)
    return {
        "linear_sq": np.zeros(shape, dtype=np.float64),
        "delta_sq": np.zeros(shape, dtype=np.float64),
        "linear_delta": np.zeros(shape, dtype=np.float64),
        "linear_mean": np.zeros(shape, dtype=np.float64),
        "delta_mean": np.zeros(shape, dtype=np.float64),
        "count": np.zeros(shape, dtype=np.int64),
    }


def _area_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (values * weights[None, :, None]).sum(axis=(-2, -1)) / values.shape[-1]


def _accumulate(
    moments: dict[str, np.ndarray],
    tau: int,
    linear: np.ndarray,
    model: np.ndarray,
    target: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
) -> None:
    linear_error = (linear.astype(np.float64) - target) / std[:, None, None]
    model_delta = (model.astype(np.float64) - linear) / std[:, None, None]
    finite = (
        np.isfinite(linear_error).all(axis=(-2, -1))
        & np.isfinite(model_delta).all(axis=(-2, -1))
    )
    values = {
        "linear_sq": _area_mean(np.square(linear_error), weights),
        "delta_sq": _area_mean(np.square(model_delta), weights),
        "linear_delta": _area_mean(linear_error * model_delta, weights),
        "linear_mean": _area_mean(linear_error, weights),
        "delta_mean": _area_mean(model_delta, weights),
    }
    for name, value in values.items():
        moments[name][tau, finite] += value[finite]
    moments["count"][tau, finite] += 1


def _averages(
    moments: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = moments["count"]
    return {
        name: np.divide(
            value,
            count,
            out=np.full_like(value, np.nan, dtype=np.float64),
            where=count > 0,
        )
        for name, value in moments.items()
        if name != "count"
    }


def _mse(
    averages: dict[str, np.ndarray],
    gate: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return np.maximum(
        0.0,
        averages["linear_sq"]
        + 2.0 * gate * averages["linear_delta"]
        + np.square(gate) * averages["delta_sq"]
        + 2.0
        * bias
        * (
            averages["linear_mean"]
            + gate * averages["delta_mean"]
        )
        + np.square(bias),
    )


def _macro_rmse(mse: np.ndarray, taus: list[int]) -> float:
    return float(np.nanmean(np.sqrt(mse[taus])))


def _interpolate_rows(
    values: np.ndarray,
    fit_taus: list[int],
    eval_taus: list[int],
) -> np.ndarray:
    output = np.zeros_like(values)
    for channel in range(values.shape[1]):
        output[eval_taus, channel] = np.interp(
            eval_taus,
            fit_taus,
            values[fit_taus, channel],
        )
    return output


def fit_adapter(
    fit: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    fit_taus: list[int],
    eval_taus: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fit_avg = _averages(fit)
    val_avg = _averages(validation)
    denominator = np.maximum(fit_avg["delta_sq"], 1.0e-12)
    raw_gate = np.clip(-fit_avg["linear_delta"] / denominator, 0.0, 1.0)
    candidates: list[dict[str, object]] = []
    for gate_scale in SCALE_CANDIDATES:
        fit_gate = raw_gate * gate_scale
        raw_bias = np.clip(
            -(
                fit_avg["linear_mean"]
                + fit_gate * fit_avg["delta_mean"]
            ),
            -0.25,
            0.25,
        )
        for bias_scale in SCALE_CANDIDATES:
            fit_bias = raw_bias * bias_scale
            gate = _interpolate_rows(fit_gate, fit_taus, eval_taus)
            bias = _interpolate_rows(fit_bias, fit_taus, eval_taus)
            validation_mse = _mse(val_avg, gate, bias)
            candidates.append(
                {
                    "gate_scale": gate_scale,
                    "bias_scale": bias_scale,
                    "selection_rmse": _macro_rmse(
                        validation_mse,
                        fit_taus,
                    ),
                    "gate": gate,
                    "bias": bias,
                }
            )
    selected = min(candidates, key=lambda item: item["selection_rmse"])
    gate = np.asarray(selected.pop("gate"))
    bias = np.asarray(selected.pop("bias"))
    fit_mse = _mse(fit_avg, gate, bias)
    val_mse = _mse(val_avg, gate, bias)
    zero = np.zeros_like(gate)
    one = np.ones_like(gate)
    diagnostics = {
        **selected,
        "fit_macro_rmse": {
            "linear": _macro_rmse(_mse(fit_avg, zero, zero), fit_taus),
            "model": _macro_rmse(_mse(fit_avg, one, zero), fit_taus),
            "adapted": _macro_rmse(fit_mse, fit_taus),
        },
        "validation_macro_rmse": {
            "linear_fit_taus": _macro_rmse(
                _mse(val_avg, zero, zero),
                fit_taus,
            ),
            "model_fit_taus": _macro_rmse(
                _mse(val_avg, one, zero),
                fit_taus,
            ),
            "adapted_fit_taus": _macro_rmse(val_mse, fit_taus),
            "linear_all_taus": _macro_rmse(
                _mse(val_avg, zero, zero),
                eval_taus,
            ),
            "model_all_taus": _macro_rmse(
                _mse(val_avg, one, zero),
                eval_taus,
            ),
            "adapted_all_taus": _macro_rmse(val_mse, eval_taus),
        },
    }
    return gate, bias, diagnostics


def fit_lead_scales(
    validation_by_lead: dict[str, dict[str, np.ndarray]],
    gate: np.ndarray,
    bias: np.ndarray,
    eval_taus: list[int],
    min_relative_rmse_gain: float = 0.0,
) -> tuple[dict[str, dict[str, float | int]], float]:
    """Select robust shrinkage per forecast-age bin on validation data."""
    combined_sse = np.zeros_like(gate, dtype=np.float64)
    combined_count = np.zeros_like(gate, dtype=np.float64)
    records: dict[str, dict[str, float | int]] = {}
    for name, lower, upper in FORECAST_LEAD_BINS:
        moments = validation_by_lead[name]
        averages = _averages(moments)
        candidates = []
        for scale in SCALE_CANDIDATES:
            mse = _mse(averages, gate * scale, bias * scale)
            candidates.append(
                (scale, _macro_rmse(mse, eval_taus), mse)
            )
        unconstrained_scale, unconstrained_rmse, unconstrained_mse = min(
            candidates,
            key=lambda item: item[1],
        )
        linear_mse = _mse(
            averages,
            np.zeros_like(gate),
            np.zeros_like(bias),
        )
        linear_rmse = _macro_rmse(linear_mse, eval_taus)
        unconstrained_relative_gain = (
            (linear_rmse - unconstrained_rmse) / linear_rmse
            if linear_rmse > 0.0
            else 0.0
        )
        fallback_triggered = (
            unconstrained_scale != 0.0
            and unconstrained_relative_gain < min_relative_rmse_gain
        )
        if fallback_triggered:
            scale = 0.0
            selected_rmse = linear_rmse
            selected_mse = linear_mse
        else:
            scale = unconstrained_scale
            selected_rmse = unconstrained_rmse
            selected_mse = unconstrained_mse
        unscaled_mse = _mse(averages, gate, bias)
        records[name] = {
            "lower_hours": lower,
            "upper_hours": upper,
            "scale": float(scale),
            "linear_validation_rmse": linear_rmse,
            "unscaled_validation_rmse": _macro_rmse(
                unscaled_mse,
                eval_taus,
            ),
            "selected_validation_rmse": float(selected_rmse),
            "unconstrained_scale": float(unconstrained_scale),
            "unconstrained_validation_rmse": float(unconstrained_rmse),
            "unconstrained_relative_rmse_gain": float(
                unconstrained_relative_gain
            ),
            "minimum_relative_rmse_gain": float(min_relative_rmse_gain),
            "linear_fallback_triggered": fallback_triggered,
            "validation_observations": int(
                moments["count"][eval_taus].sum()
            ),
        }
        count = moments["count"].astype(np.float64)
        for tau in eval_taus:
            combined_sse[tau] += selected_mse[tau] * count[tau]
            combined_count[tau] += count[tau]
    combined_mse = np.divide(
        combined_sse,
        combined_count,
        out=np.full_like(combined_sse, np.nan),
        where=combined_count > 0,
    )
    return records, _macro_rmse(combined_mse, eval_taus)


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-dir", required=True)
    parser.add_argument("--era5-memmap-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-years", nargs="+", type=int, required=True)
    parser.add_argument("--validation-years", nargs="+", type=int, required=True)
    parser.add_argument("--delta-t-hours", type=int, default=6)
    parser.add_argument("--fit-taus", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--eval-taus", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--stats-path", required=True)
    parser.add_argument("--surface-stats-path", required=True)
    parser.add_argument("--static-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--lead-stride-hours",
        type=int,
        default=24,
        help="fit only anchor leads divisible by this value",
    )
    parser.add_argument(
        "--minimum-lead-relative-rmse-gain",
        type=float,
        default=DEFAULT_MIN_LEAD_RELATIVE_RMSE_GAIN,
        help=(
            "fall back exactly to Linear when a lead-bin correction improves "
            "validation RMSE by less than this relative fraction"
        ),
    )
    args = parser.parse_args()
    if set(args.fit_years) & set(args.validation_years):
        parser.error("fit and validation years must be disjoint")
    if not set(args.fit_taus) <= set(args.eval_taus):
        parser.error("fit taus must be a subset of eval taus")
    if args.fit_taus != sorted(set(args.fit_taus)):
        parser.error("fit taus must be sorted and unique")
    if args.eval_taus != list(range(1, args.delta_t_hours)):
        parser.error("eval taus must cover every interior hour")
    if args.lead_stride_hours < args.delta_t_hours:
        parser.error("lead stride must be at least the anchor separation")
    if args.minimum_lead_relative_rmse_gain < 0.0:
        parser.error("minimum lead relative RMSE gain must be non-negative")

    stats = load_channel_stats(
        args.stats_path,
        args.surface_stats_path,
        CHANNELS_ORDER,
    )
    model = CheckpointModel(
        args.checkpoint,
        args.arch,
        args.device,
        stats,
        static_path=args.static_path,
        delta_t_hours=args.delta_t_hours,
        taus=args.eval_taus,
    )
    era5 = Era5Cache(Path(args.era5_memmap_dir))
    forecast_dir = Path(args.forecast_dir)
    archive_manifest = json.loads(
        (forecast_dir / "forecast_archive_manifest.json").read_text()
    )
    init_files = [
        forecast_dir / name
        for name in sorted(archive_manifest.get("files", {}))
    ]
    fit_years = set(args.fit_years)
    validation_years = set(args.validation_years)
    selected_files: list[Path] = []
    fit = _new_moments(args.eval_taus, len(CHANNELS_ORDER))
    validation = _new_moments(args.eval_taus, len(CHANNELS_ORDER))
    validation_by_lead = {
        name: _new_moments(args.eval_taus, len(CHANNELS_ORDER))
        for name, _, _ in FORECAST_LEAD_BINS
    }
    weights = latitude_strip_weights(wb2_block_average_latitudes())
    weights = weights / weights.sum()
    for position, binary in enumerate(init_files):
        array, leads, metadata = read_init(binary, binary.with_suffix(".json"))
        init_time = np.datetime64(metadata["init_time"], "h")
        year = int(str(init_time)[:4])
        if year in fit_years:
            target_moments = fit
            target_lead_moments = None
            taus = args.fit_taus
        elif year in validation_years:
            target_moments = validation
            target_lead_moments = validation_by_lead
            taus = args.eval_taus
        else:
            continue
        selected_files.append(binary)
        for left, right in select_anchor_pairs(leads, args.delta_t_hours):
            lead = int(leads[left])
            if lead % args.lead_stride_hours:
                continue
            lead_bin = forecast_lead_bin(lead)
            if lead_bin is None:
                continue
            x0 = array[left]
            xT = array[right]
            predictions = model.predict_taus(
                x0,
                xT,
                taus,
                dt=args.delta_t_hours,
                chunk_size=2,
                anchor_lead_hours=lead,
            )
            for tau, prediction in zip(taus, predictions, strict=True):
                valid_time = init_time + np.timedelta64(lead + tau, "h")
                target = era5.hour(valid_time)
                if target is None:
                    raise ValueError(f"missing ERA5 target at {valid_time}")
                linear = LinearInterp()(x0, xT, tau, args.delta_t_hours)
                _accumulate(
                    target_moments,
                    tau,
                    linear,
                    prediction,
                    target,
                    stats.std,
                    weights,
                )
                if target_lead_moments is not None:
                    _accumulate(
                        target_lead_moments[lead_bin],
                        tau,
                        linear,
                        prediction,
                        target,
                        stats.std,
                        weights,
                    )
        print(
            f"[{position + 1}/{len(init_files)}] {str(init_time)[:13]} year={year}",
            flush=True,
        )
    if not selected_files:
        raise SystemExit("no forecast initialisations matched the requested years")
    if any((fit["count"][tau] == 0).any() for tau in args.fit_taus):
        raise SystemExit("fit split lacks required field/tau observations")
    if any((validation["count"][tau] == 0).any() for tau in args.eval_taus):
        raise SystemExit("validation split lacks required field/tau observations")
    if any(
        (moments["count"][tau] == 0).any()
        for moments in validation_by_lead.values()
        for tau in args.eval_taus
    ):
        raise SystemExit("validation split lacks a required lead-bin observation")

    gate, bias, diagnostics = fit_adapter(
        fit,
        validation,
        args.fit_taus,
        args.eval_taus,
    )
    lead_scale_bins, lead_scaled_validation_rmse = fit_lead_scales(
        validation_by_lead,
        gate,
        bias,
        args.eval_taus,
        args.minimum_lead_relative_rmse_gain,
    )
    validation_diagnostics = diagnostics["validation_macro_rmse"]
    validation_diagnostics["adapted_all_taus_unscaled"] = (
        validation_diagnostics["adapted_all_taus"]
    )
    validation_diagnostics["adapted_all_taus"] = lead_scaled_validation_rmse
    diagnostics["lead_scale_bins"] = lead_scale_bins
    payload = {
        "schema_version": 2,
        "kind": "nwp_linear_residual_blend",
        "arch": args.arch,
        "delta_t_hours": args.delta_t_hours,
        "channel_order": CHANNELS_ORDER,
        "fit_years": args.fit_years,
        "validation_years": args.validation_years,
        "fit_taus": args.fit_taus,
        "held_out_taus": sorted(set(args.eval_taus) - set(args.fit_taus)),
        "held_out_gate_policy": "piecewise_linear_interpolation_in_tau",
        "fit_anchor_lead_stride_hours": args.lead_stride_hours,
        "fit_forecast_initialization_count": sum(
            int(str(json.loads(path.with_suffix(".json").read_text())["init_time"])[:4])
            in fit_years
            for path in selected_files
        ),
        "validation_forecast_initialization_count": sum(
            int(str(json.loads(path.with_suffix(".json").read_text())["init_time"])[:4])
            in validation_years
            for path in selected_files
        ),
        "gate_constraint": "channelwise_[0,1]",
        "bias_constraint": "channelwise_[-0.25,0.25]_training_std",
        "bias_units": "ERA5_training_standard_deviations",
        "lead_scale_policy": (
            "validation_selected_scalar_shrinkage_with_minimum_gain_fallback"
        ),
        "minimum_lead_relative_rmse_gain": (
            args.minimum_lead_relative_rmse_gain
        ),
        "lead_scale_bins": lead_scale_bins,
        "gates": {
            str(tau): gate[tau].tolist() for tau in args.eval_taus
        },
        "bias_normalized": {
            str(tau): bias[tau].tolist() for tau in args.eval_taus
        },
        "selection": diagnostics,
        "base_checkpoint": file_provenance(args.checkpoint),
        "provenance": {
            "fitter": file_provenance(__file__),
            "forecast_archive": _forecast_manifest_provenance(
                forecast_dir,
                selected_files,
            ),
            "era5": memmap_dataset_provenance(
                args.era5_memmap_dir,
                sorted(fit_years | validation_years),
            ),
            "pressure_stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(args.surface_stats_path),
            "static_features": static_feature_provenance(args.static_path),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, payload)
    print(json.dumps(diagnostics, indent=2), flush=True)


if __name__ == "__main__":
    main()
