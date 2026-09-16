#!/usr/bin/env python3
"""Paired spherical spectra for a checkpoint with optional NWP adaptation.

The evaluator shares forecast/ERA5 selection and model loading with
``batch_eval_forecast_anchor.py``. It computes the analytic linear baseline
and the model in one pass. When an adapter is supplied, neural inference is
skipped wherever its guarded lead scale is exactly zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    CheckpointModel,
    Era5Cache,
    LinearInterp,
    ResidualBlendAdapter,
    _forecast_manifest_provenance,
    _sha256_arrays,
    forecast_evaluation_source_paths,
    forecast_lead_bin,
    read_init,
    select_anchor_pairs,
    select_init_files,
)
from tools.eval.sh_energy_spectra_12h import (
    SPECTRAL_SCHEMA_VERSION,
    _band_window_diagnostics,
    _build_sht,
)
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.metrics.spherical_spectra import (
    coefficient_cross,
    coefficient_power,
)
from weather_time_interp.normalization import (
    file_provenance,
    load_channel_stats,
    static_feature_provenance,
)

DEFAULT_CHANNELS = ("Q850", "U850")
DEFAULT_TAUS = (2, 3)


def lead_model_scale(
    adapter: ResidualBlendAdapter,
    lead_hours: int,
) -> float:
    """Return the neural-correction scale for one left-anchor lead."""
    if adapter.lead_scales is None:
        return 1.0
    name = forecast_lead_bin(int(lead_hours))
    if name is None or name not in adapter.lead_scales:
        raise ValueError(f"anchor lead {lead_hours} is outside adapter bins")
    return float(adapter.lead_scales[name])


def hour_of_year(valid_time: np.datetime64) -> tuple[int, int]:
    """Return calendar year and zero-based hour within that year."""
    value = np.datetime64(valid_time, "h")
    year = int(str(value)[:4])
    start = np.datetime64(f"{year}-01-01T00", "h")
    hour = int((value - start) / np.timedelta64(1, "h"))
    return year, hour


def _new_accumulator(
    n_channels: int,
    lmax: int,
    device: torch.device,
) -> dict[str, Any]:
    return {
        "pred_power": torch.zeros(
            n_channels, lmax + 1, dtype=torch.float64, device=device
        ),
        "truth_power": torch.zeros(
            n_channels, lmax + 1, dtype=torch.float64, device=device
        ),
        "cross": torch.zeros(
            n_channels, lmax + 1, dtype=torch.complex128, device=device
        ),
        "init_time_hours": [],
        "anchor_lead_hours": [],
        "year": [],
        "t0": [],
        "energy_ratio": [],
        "shape_error": [],
        "coherence": [],
        "signed_cospectrum": [],
    }


def _append_window(
    accumulator: dict[str, Any],
    *,
    pred_power: torch.Tensor,
    truth_power: torch.Tensor,
    cross: torch.Tensor,
    init_time_hours: int,
    anchor_lead_hours: int,
    year: int,
    t0: int,
    hf_ell_min: int,
) -> None:
    accumulator["pred_power"] += pred_power
    accumulator["truth_power"] += truth_power
    accumulator["cross"] += cross
    ratio, shape, coherence, signed = _band_window_diagnostics(
        pred_power.unsqueeze(0),
        truth_power.unsqueeze(0),
        cross.unsqueeze(0),
        hf_ell_min,
    )
    accumulator["init_time_hours"].append(init_time_hours)
    accumulator["anchor_lead_hours"].append(anchor_lead_hours)
    accumulator["year"].append(year)
    accumulator["t0"].append(t0)
    accumulator["energy_ratio"].append(ratio[0].cpu().numpy().astype(np.float32))
    accumulator["shape_error"].append(shape[0].cpu().numpy().astype(np.float32))
    accumulator["coherence"].append(
        coherence[0].cpu().numpy().astype(np.float32)
    )
    accumulator["signed_cospectrum"].append(
        signed[0].cpu().numpy().astype(np.float32)
    )


def _atomic_savez(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _finalize_payload(
    accumulator: dict[str, Any],
    *,
    model_name: str,
    tau: int,
    channels: list[str],
    height: int,
    width: int,
    lmax: int,
    hf_ell_min: int,
    provenance: dict[str, Any],
) -> dict[str, Any]:
    n_samples = len(accumulator["year"])
    if n_samples < 1:
        raise RuntimeError(f"no spectral windows for tau={tau}")
    year = np.asarray(accumulator["year"], dtype=np.int16)
    t0 = np.asarray(accumulator["t0"], dtype=np.int32)
    if len(set(zip(year.tolist(), t0.tolist(), strict=True))) != n_samples:
        raise RuntimeError(
            f"tau={tau}: duplicate valid times cannot enter the block bootstrap"
        )
    init_time = np.asarray(accumulator["init_time_hours"], dtype=np.int64)
    lead = np.asarray(accumulator["anchor_lead_hours"], dtype=np.int16)
    tau_array = np.full(n_samples, tau, dtype=np.int8)
    pred_power = (accumulator["pred_power"] / n_samples).cpu().numpy()
    truth_power = (accumulator["truth_power"] / n_samples).cpu().numpy()
    cross = (accumulator["cross"] / n_samples).cpu().numpy()
    denominator = np.clip(pred_power * truth_power, 1.0e-30, None)
    return {
        "schema_version": np.asarray(SPECTRAL_SCHEMA_VERSION, dtype=np.int16),
        "ell": np.arange(lmax + 1, dtype=np.int16),
        "channel_names": np.asarray(channels),
        "pred_El": pred_power,
        "gt_El": truth_power,
        "cross_El_real": cross.real,
        "cross_El_imag": cross.imag,
        "coherence_l": np.abs(cross) ** 2 / denominator,
        "signed_cospectrum_l": cross.real / np.sqrt(denominator),
        "window_year": year,
        "window_t0": t0,
        "window_init_time_hours": init_time,
        "window_anchor_lead_hours": lead,
        "window_hf_energy_ratio": np.stack(accumulator["energy_ratio"]),
        "window_hf_log_shape_error": np.stack(accumulator["shape_error"]),
        "window_hf_coherence": np.stack(accumulator["coherence"]),
        "window_hf_signed_cospectrum": np.stack(
            accumulator["signed_cospectrum"]
        ),
        "n_samples": np.asarray(n_samples, dtype=np.int32),
        "tau": np.asarray(tau, dtype=np.int16),
        "H": np.asarray(height, dtype=np.int16),
        "W": np.asarray(width, dtype=np.int16),
        "lmax": np.asarray(lmax, dtype=np.int16),
        "hf_ell_min": np.asarray(hf_ell_min, dtype=np.int16),
        "model_name": np.asarray(model_name),
        "sht_grid": np.asarray(
            "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht"
        ),
        "spectral_field_units": np.asarray(
            "physical_anomaly_units_via_channel_std"
        ),
        "window_index_sha256": np.asarray(
            _sha256_arrays(init_time, lead, tau_array)
        ),
        "provenance_json": np.asarray(json.dumps(provenance, sort_keys=True)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forecast-dir", required=True)
    parser.add_argument("--era5-memmap-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--blend-adapter", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path", default="data/surface_stats_0p5.json"
    )
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--delta-t-hours", type=int, default=6)
    parser.add_argument("--forecast-lead-stride-hours", type=int, default=None)
    parser.add_argument(
        "--forecast-maximum-left-lead-hours",
        type=int,
        default=None,
    )
    parser.add_argument("--taus", nargs="+", type=int, default=list(DEFAULT_TAUS))
    parser.add_argument("--channels", nargs="+", default=list(DEFAULT_CHANNELS))
    parser.add_argument("--lmax", type=int, default=180)
    parser.add_argument("--hf-ell-min", type=int, default=80)
    parser.add_argument("--max-inits", type=int, default=16)
    parser.add_argument("--tau-batch-size", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.delta_t_hours != 6
        or not args.taus
        or len(args.taus) != len(set(args.taus))
        or any(not 1 <= tau < args.delta_t_hours for tau in args.taus)
    ):
        raise SystemExit("this frozen evaluator requires unique interior 6h taus")
    if args.hf_ell_min < 1 or args.hf_ell_min > args.lmax:
        raise SystemExit("--hf-ell-min must lie in [1, lmax]")
    if (
        args.forecast_lead_stride_hours is not None
        and args.forecast_lead_stride_hours <= 0
    ):
        raise SystemExit("--forecast-lead-stride-hours must be positive")
    if (
        args.forecast_maximum_left_lead_hours is not None
        and args.forecast_maximum_left_lead_hours <= 0
    ):
        raise SystemExit(
            "--forecast-maximum-left-lead-hours must be positive"
        )
    if len(args.channels) != len(set(args.channels)):
        raise SystemExit("--channels must be unique")
    try:
        channel_indices = [CHANNELS_ORDER.index(name) for name in args.channels]
    except ValueError as exc:
        raise SystemExit(f"unknown channel in {args.channels}") from exc

    forecast_dir = Path(args.forecast_dir)
    era5_dir = Path(args.era5_memmap_dir)
    out_dir = Path(args.out_dir)
    init_files = select_init_files(
        sorted(forecast_dir.glob("init_*.bin")), args.max_inits
    )
    if not init_files:
        raise SystemExit(f"no forecast initializations in {forecast_dir}")

    stats = load_channel_stats(
        args.stats_path, args.surface_stats_path, CHANNELS_ORDER
    )
    base = CheckpointModel(
        args.checkpoint,
        args.arch,
        args.device,
        stats,
        static_path=args.static_path,
        delta_t_hours=args.delta_t_hours,
        taus=args.taus,
    )
    adapter = (
        ResidualBlendAdapter(
            base,
            args.blend_adapter,
            args.checkpoint,
            args.arch,
            args.delta_t_hours,
        )
        if args.blend_adapter
        else None
    )
    model_name = args.model_name or (
        "adapted_model" if adapter is not None else "model"
    )
    if model_name == "linear":
        raise SystemExit("--model-name=linear is reserved for the baseline")
    era5 = Era5Cache(era5_dir)
    device = torch.device(args.device)
    first_meta = json.loads(init_files[0].with_suffix(".json").read_text())
    height, width = map(int, first_meta["shape"][-2:])
    sht = _build_sht(height, width, args.lmax, device)
    accumulators = {
        name: {
            tau: _new_accumulator(len(args.channels), args.lmax, device)
            for tau in args.taus
        }
        for name in ("linear", model_name)
    }
    mean = stats.mean[channel_indices].reshape(1, -1, 1, 1)
    target_years: set[int] = set()
    skipped_model_windows = 0
    neural_model_windows = 0

    with torch.no_grad():
        for init_number, bin_path in enumerate(init_files, start=1):
            values, leads, meta = read_init(bin_path, bin_path.with_suffix(".json"))
            init_time = np.datetime64(meta["init_time"], "h")
            init_time_hours = int(init_time.astype(np.int64))
            for left_index, right_index in select_anchor_pairs(
                leads,
                args.delta_t_hours,
                lead_stride_hours=args.forecast_lead_stride_hours,
                maximum_left_lead_hours=(
                    args.forecast_maximum_left_lead_hours
                ),
            ):
                lead = int(leads[left_index])
                x0 = values[left_index]
                xT = values[right_index]
                linear = np.stack(
                    [
                        LinearInterp()(x0, xT, tau, args.delta_t_hours)
                        for tau in args.taus
                    ]
                )
                scale = lead_model_scale(adapter, lead) if adapter else 1.0
                if adapter is not None and scale == 0.0:
                    model_prediction = linear
                    skipped_model_windows += len(args.taus)
                else:
                    if adapter is None:
                        model_prediction = base.predict_taus(
                            x0,
                            xT,
                            args.taus,
                            dt=args.delta_t_hours,
                            chunk_size=args.tau_batch_size,
                            anchor_lead_hours=lead,
                        )
                    else:
                        model_prediction = adapter.predict_taus(
                            x0,
                            xT,
                            args.taus,
                            dt=args.delta_t_hours,
                            chunk_size=args.tau_batch_size,
                            anchor_lead_hours=lead,
                        )
                    neural_model_windows += len(args.taus)

                targets = []
                valid_times = []
                for tau in args.taus:
                    valid = init_time + np.timedelta64(lead + tau, "h")
                    target = era5.hour(valid)
                    if target is None:
                        raise RuntimeError(f"missing ERA5 target at {valid}")
                    targets.append(target)
                    valid_times.append(valid)
                    target_years.add(int(str(valid)[:4]))
                selected = {
                    "linear": linear[:, channel_indices],
                    model_name: model_prediction[:, channel_indices],
                    "truth": np.stack(targets)[:, channel_indices],
                }
                if any(not np.isfinite(field).all() for field in selected.values()):
                    raise RuntimeError(
                        f"non-finite selected field at init={init_time}, lead={lead}"
                    )

                def transform(field: np.ndarray) -> torch.Tensor:
                    anomalies = (field - mean).astype(np.float32)
                    return sht(torch.from_numpy(anomalies).to(device).double())

                coefficients = {
                    "truth": transform(selected["truth"]),
                    "linear": transform(selected["linear"]),
                }
                coefficients[model_name] = (
                    coefficients["linear"]
                    if adapter is not None and scale == 0.0
                    else transform(selected[model_name])
                )
                truth_power = coefficient_power(coefficients["truth"])
                for evaluated_name in ("linear", model_name):
                    pred_power = coefficient_power(coefficients[evaluated_name])
                    cross = coefficient_cross(
                        coefficients[evaluated_name], coefficients["truth"]
                    )
                    for tau_index, (tau, valid) in enumerate(
                        zip(args.taus, valid_times, strict=True)
                    ):
                        year, t0 = hour_of_year(valid)
                        _append_window(
                            accumulators[evaluated_name][tau],
                            pred_power=pred_power[tau_index],
                            truth_power=truth_power[tau_index],
                            cross=cross[tau_index],
                            init_time_hours=init_time_hours,
                            anchor_lead_hours=lead,
                            year=year,
                            t0=t0,
                            hf_ell_min=args.hf_ell_min,
                        )
            print(
                f"[{init_number}/{len(init_files)}] init={str(init_time)[:13]}",
                flush=True,
            )

    source_paths = {
        "nwp_blend_spectra.py": Path(__file__).resolve(),
        "spectral_evaluator": Path(
            "tools/eval/sh_energy_spectra_12h.py"
        ).resolve(),
        "spherical_spectra": Path(
            "weather_time_interp/metrics/spherical_spectra.py"
        ).resolve(),
        **{
            f"forecast_inference/{name}": path
            for name, path in forecast_evaluation_source_paths().items()
        },
    }
    provenance = {
        "evaluation_code": {
            name: file_provenance(path) for name, path in source_paths.items()
        },
        "checkpoint": file_provenance(args.checkpoint),
        "adapter": (
            file_provenance(args.blend_adapter)
            if args.blend_adapter
            else None
        ),
        "forecast_anchors": _forecast_manifest_provenance(
            forecast_dir, init_files
        ),
        "era5_truth": memmap_dataset_provenance(
            era5_dir, sorted(target_years)
        ),
        "pressure_stats": file_provenance(args.stats_path),
        "surface_stats": file_provenance(args.surface_stats_path),
        "static_features": static_feature_provenance(args.static_path),
        "protocol": {
            "task": "6h_NWP_anchor_to_ERA5_interpolation",
            "taus": args.taus,
            "channels": args.channels,
            "lmax": args.lmax,
            "hf_ell_min": args.hf_ell_min,
            "model_name": model_name,
            "blend_adapter_enabled": adapter is not None,
            "model_inference_skipped_when_guard_scale_zero": (
                adapter is not None
            ),
            "neural_model_windows": neural_model_windows,
            "skipped_model_windows": skipped_model_windows,
            "forecast_lead_stride_hours": (
                args.forecast_lead_stride_hours
            ),
            "maximum_left_forecast_lead_hours_exclusive": (
                args.forecast_maximum_left_lead_hours
            ),
        },
    }
    artifacts: dict[str, str] = {}
    for model_name, per_tau in accumulators.items():
        for tau, accumulator in per_tau.items():
            path = out_dir / f"{model_name}_tau{tau}.npz"
            payload = _finalize_payload(
                accumulator,
                model_name=model_name,
                tau=tau,
                channels=args.channels,
                height=height,
                width=width,
                lmax=args.lmax,
                hf_ell_min=args.hf_ell_min,
                provenance=provenance,
            )
            _atomic_savez(path, payload)
            artifacts[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            print(f"[write] {path}", flush=True)
    _atomic_json(
        out_dir / "spectra_complete.json",
        {
            "schema_version": 1,
            "status": "complete",
            "artifacts": artifacts,
            "protocol": provenance["protocol"],
        },
    )


if __name__ == "__main__":
    main()
