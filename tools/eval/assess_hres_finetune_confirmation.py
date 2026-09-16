#!/usr/bin/env python3
"""Apply the frozen RMSE and spectral gates to HRES fine-tuning outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.summarize_forecast_anchor import (
    ForecastArtifact,
    load_artifact,
    summarize_artifacts,
)
from weather_time_interp.normalization import file_provenance


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid name=path specification: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in result:
            raise ValueError(f"duplicate or empty artifact name: {name!r}")
        result[name] = Path(raw_path)
    return result


def _load_training_selections(
    marker_path: Path,
    protocol_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    marker = json.loads(marker_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    if (
        marker.get("status") != "training_complete_selection_frozen"
        or marker.get("protocol_sha256") != _sha256(protocol_path)
    ):
        raise ValueError("training marker does not bind the frozen protocol")
    records = marker.get("selections", {})
    expected = {
        f"{family}_s{seed}"
        for family in ("weatherbridge", "weatherdcae")
        for seed in protocol["optimization"]["seeds"]
    }
    if set(records) != expected:
        raise ValueError("training marker has an unexpected run set")
    selections: dict[str, dict[str, Any]] = {}
    for name, record in records.items():
        path = Path(record["path"])
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"stale selected checkpoint record: {name}")
        selection = json.loads(path.read_text())
        if selection.get("status") != "selected_on_2020_validation":
            raise ValueError(f"{name}: checkpoint selection is not frozen")
        selections[name] = selection
    return selections, protocol


def _validate_confirmation_data(
    artifact: ForecastArtifact,
    expected_init_times: list[str],
    expected_left_leads: list[int],
    maximum_left_lead_hours: int,
) -> None:
    record = artifact.data_provenance["forecast_anchors"]["archive_manifest"]
    path = Path(record["path"])
    if _sha256(path) != record.get("sha256"):
        raise ValueError("confirmation forecast manifest hash mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("init_times") != expected_init_times:
        raise ValueError("confirmation initialisations differ from the frozen index")
    evaluation = json.loads(artifact.json_path.read_text()).get("protocol", {})
    if (
        evaluation.get("forecast_lead_stride_hours") != 24
        or evaluation.get(
            "maximum_left_forecast_lead_hours_exclusive"
        )
        != maximum_left_lead_hours
    ):
        raise ValueError("confirmation forecast-lead protocol mismatch")
    expected_init_hours = np.asarray(
        [
            int(np.datetime64(value, "h").astype(np.int64))
            for value in expected_init_times
        ],
        dtype=np.int64,
    )
    expected_taus = np.arange(1, artifact.delta_t_hours, dtype=np.int64)
    if (
        np.unique(artifact.init_time).tolist()
        != expected_init_hours.tolist()
        or np.unique(artifact.lead).tolist() != expected_left_leads
        or np.unique(artifact.tau).tolist() != expected_taus.tolist()
        or np.any(artifact.lead >= maximum_left_lead_hours)
    ):
        raise ValueError("confirmation paired index violates frozen lead support")
    triplets = np.stack(
        (artifact.init_time, artifact.lead, artifact.tau),
        axis=1,
    )
    expected_count = (
        len(expected_init_times)
        * len(expected_left_leads)
        * len(expected_taus)
    )
    if triplets.shape != (expected_count, 3) or np.unique(
        triplets,
        axis=0,
    ).shape[0] != expected_count:
        raise ValueError("confirmation paired index is incomplete or duplicated")


def _load_spectral_pair(
    fine_dir: Path,
    raw_dir: Path,
    *,
    fine_name: str,
    raw_name: str,
    taus: list[int],
    channels: list[str],
    ell_min: int,
    fine_checkpoint_sha256: str,
    raw_checkpoint_sha256: str,
    expected_left_leads: list[int] | None = None,
    expected_init_count: int | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for tau in taus:
        fine_path = fine_dir / f"{fine_name}_tau{tau}.npz"
        raw_path = raw_dir / f"{raw_name}_tau{tau}.npz"
        with np.load(fine_path, allow_pickle=False) as fine, np.load(
            raw_path, allow_pickle=False
        ) as raw:
            if (
                fine["schema_version"].item() != raw["schema_version"].item()
                or fine["model_name"].item() != fine_name
                or raw["model_name"].item() != raw_name
                or fine["channel_names"].tolist() != channels
                or raw["channel_names"].tolist() != channels
                or fine["tau"].item() != tau
                or raw["tau"].item() != tau
                or fine["hf_ell_min"].item() != ell_min
                or raw["hf_ell_min"].item() != ell_min
                or fine["window_index_sha256"].item()
                != raw["window_index_sha256"].item()
            ):
                raise ValueError(f"tau={tau}: incompatible spectral artifacts")
            fine_provenance = json.loads(fine["provenance_json"].item())
            raw_provenance = json.loads(raw["provenance_json"].item())
            fine_protocol = dict(fine_provenance.get("protocol", {}))
            raw_protocol = dict(raw_provenance.get("protocol", {}))
            fine_protocol.pop("model_name", None)
            raw_protocol.pop("model_name", None)
            if (
                fine_provenance.get("checkpoint", {}).get("sha256")
                != fine_checkpoint_sha256
                or raw_provenance.get("checkpoint", {}).get("sha256")
                != raw_checkpoint_sha256
                or fine_provenance.get("forecast_anchors")
                != raw_provenance.get("forecast_anchors")
                or fine_provenance.get("era5_truth")
                != raw_provenance.get("era5_truth")
                or fine_protocol != raw_protocol
            ):
                raise ValueError(f"tau={tau}: incompatible spectral provenance")
            if expected_left_leads is not None:
                if expected_init_count is None or expected_init_count < 1:
                    raise ValueError("expected_init_count is required")
                expected_samples = expected_init_count * len(
                    expected_left_leads
                )
                fine_leads = fine["window_anchor_lead_hours"]
                fine_inits = fine["window_init_time_hours"]
                pairs = np.stack((fine_inits, fine_leads), axis=1)
                if (
                    np.unique(fine_leads).tolist() != expected_left_leads
                    or pairs.shape != (expected_samples, 2)
                    or np.unique(pairs, axis=0).shape[0] != expected_samples
                    or fine_protocol.get("forecast_lead_stride_hours") != 24
                    or fine_protocol.get(
                        "maximum_left_forecast_lead_hours_exclusive"
                    )
                    != 120
                ):
                    raise ValueError(
                        f"tau={tau}: spectral lead support violates protocol"
                    )
            fine_shape = fine["window_hf_log_shape_error"].mean(axis=0)
            raw_shape = raw["window_hf_log_shape_error"].mean(axis=0)
            fine_coherence = fine["window_hf_coherence"].mean(axis=0)
            raw_coherence = raw["window_hf_coherence"].mean(axis=0)
            for index, channel in enumerate(channels):
                rows.append(
                    {
                        "tau": tau,
                        "channel": channel,
                        "shape_error_change": float(
                            fine_shape[index] - raw_shape[index]
                        ),
                        "coherence_change": float(
                            fine_coherence[index] - raw_coherence[index]
                        ),
                    }
                )
    return {
        "rows": rows,
        "fine_artifacts": {
            path.name: file_provenance(path)
            for path in sorted(fine_dir.glob(f"{fine_name}_tau*.npz"))
        },
        "raw_artifacts": {
            path.name: file_provenance(path)
            for path in sorted(raw_dir.glob(f"{raw_name}_tau*.npz"))
        },
    }


def assess(
    *,
    protocol_path: Path,
    training_marker_path: Path,
    artifact_paths: dict[str, Path],
    raw_checkpoints: dict[str, Path],
    fine_spectra_dir: Path,
    raw_spectra_dir: Path,
    draws: int,
    random_seed: int,
) -> dict[str, Any]:
    selections, protocol = _load_training_selections(
        training_marker_path,
        protocol_path,
    )
    seeds = [int(value) for value in protocol["optimization"]["seeds"]]
    primary_seed = int(protocol["optimization"]["primary_seed"])
    expected_names = {
        "linear",
        "weatherbridge_raw",
        "weatherdcae_raw",
        *{f"weatherbridge_s{seed}" for seed in seeds},
        *{f"weatherdcae_s{seed}" for seed in seeds},
    }
    if set(artifact_paths) != expected_names:
        raise ValueError("confirmation artifact set does not match the protocol")
    if set(raw_checkpoints) != {"weatherbridge_raw", "weatherdcae_raw"}:
        raise ValueError("both raw checkpoint bindings are required")

    artifacts = {
        name: load_artifact(
            name,
            path,
            expected_target_year=2022,
            expected_init_count=24,
        )
        for name, path in artifact_paths.items()
    }
    expected_init_times = protocol["splits"]["date_level_confirmation"][
        "init_times"
    ]
    expected_left_leads = protocol["data_contract"][
        "left_forecast_leads_hours"
    ]
    maximum_left_lead_hours = protocol["data_contract"][
        "maximum_left_forecast_lead_hours_exclusive"
    ]
    for artifact in artifacts.values():
        _validate_confirmation_data(
            artifact,
            expected_init_times,
            expected_left_leads,
            maximum_left_lead_hours,
        )

    expected_checkpoint_hashes = {
        name: selection["checkpoint"]["sha256"]
        for name, selection in selections.items()
    }
    expected_checkpoint_hashes.update(
        {name: _sha256(path) for name, path in raw_checkpoints.items()}
    )
    for name, expected_hash in expected_checkpoint_hashes.items():
        if artifacts[name].checkpoint_sha256 != expected_hash:
            raise ValueError(f"{name}: confirmation checkpoint mismatch")

    per_seed: dict[str, Any] = {}
    required_comparators = (
        "weatherbridge_raw",
        "linear",
    )
    point_wins = 0
    mean_deltas: dict[str, list[float]] = {
        "weatherbridge_raw": [],
        "linear": [],
        "matched_weatherdcae": [],
    }
    for seed in seeds:
        winner = f"weatherbridge_s{seed}"
        matched_dcae = f"weatherdcae_s{seed}"
        subset = {
            name: artifacts[name]
            for name in (
                winner,
                matched_dcae,
                "weatherbridge_raw",
                "weatherdcae_raw",
                "linear",
            )
        }
        summary = summarize_artifacts(
            subset,
            winner,
            draws=draws,
            seed=random_seed + seed,
        )
        all_scores = summary["families"]["all"]["scores"]
        held_scores = summary["families"]["unseen_tau"]["scores"]
        comparators = (*required_comparators, matched_dcae)
        lower_all = all(
            all_scores[winner]["rmse_norm"] < all_scores[name]["rmse_norm"]
            for name in comparators
        )
        no_held_regression = all(
            held_scores[winner]["rmse_norm"] <= held_scores[name]["rmse_norm"]
            for name in comparators
        )
        point_wins += int(lower_all and no_held_regression)
        for label, comparator in (
            ("weatherbridge_raw", "weatherbridge_raw"),
            ("linear", "linear"),
            ("matched_weatherdcae", matched_dcae),
        ):
            mean_deltas[label].append(
                float(all_scores[winner]["rmse_norm"])
                - float(all_scores[comparator]["rmse_norm"])
            )
        per_seed[str(seed)] = {
            "lower_all_hour_rmse_than_all_comparators": lower_all,
            "no_held_hour_rmse_regression": no_held_regression,
            "paired_summary": summary,
        }

    primary = per_seed[str(primary_seed)]["paired_summary"]
    primary_scores = primary["families"]["all"]["scores"]
    primary_comparators = (
        "weatherbridge_raw",
        "linear",
        f"weatherdcae_s{primary_seed}",
    )
    strongest = min(
        primary_comparators,
        key=lambda name: primary_scores[name]["rmse_norm"],
    )
    strongest_comparison = primary["families"]["all"][
        "comparisons_to_winner"
    ][strongest]

    spectral_protocol = protocol["promotion_rule"]["spectral_gate"]
    spectral = _load_spectral_pair(
        fine_spectra_dir,
        raw_spectra_dir,
        fine_name=f"weatherbridge_s{primary_seed}",
        raw_name="weatherbridge_raw",
        taus=spectral_protocol["query_hours"],
        channels=spectral_protocol["fields"],
        ell_min=spectral_protocol["ell_min"],
        fine_checkpoint_sha256=expected_checkpoint_hashes[
            f"weatherbridge_s{primary_seed}"
        ],
        raw_checkpoint_sha256=expected_checkpoint_hashes["weatherbridge_raw"],
        expected_left_leads=expected_left_leads,
        expected_init_count=len(expected_init_times),
    )
    shape_margin = float(spectral_protocol["max_shape_error_increase"])
    coherence_margin = float(spectral_protocol["max_coherence_decrease"])
    spectral_passed = all(
        row["shape_error_change"] <= shape_margin
        and row["coherence_change"] >= -coherence_margin
        for row in spectral["rows"]
    )
    mean_delta_passed = all(
        float(np.mean(values)) <= 0.0 for values in mean_deltas.values()
    )
    significance_passed = bool(
        strongest_comparison["winner_significantly_better"]
    )
    promotion_passed = bool(
        point_wins >= 2
        and mean_delta_passed
        and significance_passed
        and spectral_passed
    )
    return {
        "schema_version": 1,
        "status": "promotion_passed" if promotion_passed else "promotion_failed",
        "promotion_passed": promotion_passed,
        "primary_seed": primary_seed,
        "point_estimate_seed_wins": point_wins,
        "mean_rmse_delta_finetuned_minus_comparator": {
            name: float(np.mean(values)) for name, values in mean_deltas.items()
        },
        "mean_delta_gate_passed": mean_delta_passed,
        "strongest_primary_comparator": strongest,
        "primary_significance": strongest_comparison,
        "primary_significance_gate_passed": significance_passed,
        "spectral_gate": {**spectral, "passed": spectral_passed},
        "per_seed": per_seed,
        "provenance": {
            "protocol": file_provenance(protocol_path),
            "training_marker": file_provenance(training_marker_path),
            "rmse_artifacts": {
                name: file_provenance(path)
                for name, path in artifact_paths.items()
            },
            "raw_checkpoints": {
                name: file_provenance(path)
                for name, path in raw_checkpoints.items()
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--training-marker", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[], required=True)
    parser.add_argument("--raw-checkpoint", action="append", default=[], required=True)
    parser.add_argument("--fine-spectra-dir", type=Path, required=True)
    parser.add_argument("--raw-spectra-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1000:
        raise SystemExit("--draws must be at least 1000")
    payload = assess(
        protocol_path=args.protocol,
        training_marker_path=args.training_marker,
        artifact_paths=_parse_named_paths(args.artifact),
        raw_checkpoints=_parse_named_paths(args.raw_checkpoint),
        fine_spectra_dir=args.fine_spectra_dir,
        raw_spectra_dir=args.raw_spectra_dir,
        draws=args.draws,
        random_seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps({"output": str(args.output), "status": payload["status"]}))


if __name__ == "__main__":
    main()
