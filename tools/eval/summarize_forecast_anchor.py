"""Summarize paired forecast-anchor OOD errors after candidate selection."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    _sha256_arrays,
    forecast_evaluation_source_paths,
)
from weather_time_interp.normalization import file_provenance


FORECAST_RMSE_NONINFERIORITY_MARGIN = 0.02


@dataclass(frozen=True)
class ForecastArtifact:
    name: str
    json_path: Path
    npz_path: Path
    init_time: np.ndarray
    lead: np.ndarray
    tau: np.ndarray
    squared_error: np.ndarray
    valid_channel: np.ndarray
    index_sha256: str
    delta_t_hours: int
    protocol_taus: tuple[int, ...]
    model_kind: str
    checkpoint_sha256: str | None
    data_provenance: dict[str, Any]


def load_artifact(
    name: str,
    json_path: str | Path,
    *,
    expected_target_year: int = 2021,
    expected_init_count: int = 16,
) -> ForecastArtifact:
    if expected_init_count < 2:
        raise ValueError("expected_init_count must be at least two")
    path = Path(json_path)
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(f"{path}: expected schema_version=2")
    if payload.get("model_name") != name:
        raise ValueError(
            f"{path}: model_name={payload.get('model_name')!r} != {name!r}"
        )
    protocol = payload.get("protocol", {})
    delta_t_hours = protocol.get("delta_t_hours")
    protocol_taus = protocol.get("taus")
    if (
        not isinstance(delta_t_hours, int)
        or delta_t_hours < 2
        or not isinstance(protocol_taus, list)
        or not protocol_taus
        or any(
            not isinstance(tau, int) or not 1 <= tau < delta_t_hours
            for tau in protocol_taus
        )
        or len(protocol_taus) != len(set(protocol_taus))
        or payload.get("channels") != CHANNELS_ORDER
        or protocol.get("anchor_pairing")
        != "same_initialization_forecast_leads"
        or protocol.get("target_source")
        != "ERA5_at_intermediate_valid_time"
        or protocol.get("future_analysis_as_input") is not False
        or protocol.get("area_weighting")
        != "spherical_latitude_strip_area"
        or protocol.get("latitude_grid")
        != "wb2_0p25_2x2_block_average_v1"
        or protocol.get("normalization")
        != "(x - training_mean) / training_std"
        or protocol.get("missing_anchor_channel_imputation")
        != "training_mean"
        or protocol.get("missing_channels_excluded_from_metrics") is not True
        or protocol.get("init_selection")
        != "evenly_spaced_over_sorted_archive"
        or protocol.get("max_inits") != expected_init_count
    ):
        raise ValueError(f"{path}: invalid forecast protocol")
    provenance = payload.get("provenance", {})
    evaluation_code = provenance.get("evaluation_code")
    expected_sources = forecast_evaluation_source_paths()
    if (
        not isinstance(evaluation_code, dict)
        or set(evaluation_code) != set(expected_sources)
    ):
        raise ValueError(f"{path}: invalid forecast evaluation provenance")
    for source_name, source_path in expected_sources.items():
        recorded_source = evaluation_code.get(source_name)
        current_source = file_provenance(source_path)
        if (
            not isinstance(recorded_source, dict)
            or recorded_source.get("sha256") != current_source["sha256"]
            or recorded_source.get("size_bytes")
            != current_source["size_bytes"]
        ):
            raise ValueError(
                f"{path}: stale forecast evaluation source {source_name}"
            )
    model_provenance = provenance.get("model", {})
    data_provenance = {
        key: provenance.get(key)
        for key in (
            "forecast_anchors",
            "era5_truth",
            "pressure_stats",
            "surface_stats",
        )
    }
    if (
        not all(isinstance(value, dict) and value for value in data_provenance.values())
        or data_provenance["era5_truth"].get("years")
        != [expected_target_year]
        or not isinstance(
            data_provenance["forecast_anchors"].get("init_count"),
            int,
        )
        or data_provenance["forecast_anchors"]["init_count"]
        != expected_init_count
        or data_provenance["forecast_anchors"].get("grid", {}).get(
            "latitude_order"
        )
        != "north_to_south"
        or not isinstance(
            data_provenance["forecast_anchors"].get("archive_manifest"),
            dict,
        )
    ):
        raise ValueError(f"{path}: invalid forecast data provenance")
    model_kind = model_provenance.get("kind")
    if model_kind == "linear":
        checkpoint_sha256 = None
    elif model_kind in {
        "capmatched_checkpoint",
        "capmatched_checkpoint_with_nwp_blend",
        "bare_blob",
    }:
        checkpoint_sha256 = model_provenance.get("artifact", {}).get(
            "sha256"
        )
        if not isinstance(checkpoint_sha256, str) or len(
            checkpoint_sha256
        ) != 64:
            raise ValueError(f"{path}: invalid model artifact provenance")
        if model_kind == "capmatched_checkpoint_with_nwp_blend":
            adapter = model_provenance.get("blend_adapter")
            if (
                not isinstance(adapter, dict)
                or not isinstance(adapter.get("sha256"), str)
                or len(adapter["sha256"]) != 64
            ):
                raise ValueError(f"{path}: invalid blend-adapter provenance")
    else:
        raise ValueError(f"{path}: invalid model provenance kind")
    paired = payload.get("paired_artifact", {})
    npz_path = Path(paired.get("path", ""))
    if not npz_path.is_file():
        raise ValueError(f"{path}: missing paired artifact {npz_path}")
    current_npz = file_provenance(npz_path)
    if current_npz["sha256"] != paired.get("sha256"):
        raise ValueError(f"{path}: paired artifact SHA-256 mismatch")
    with np.load(npz_path, allow_pickle=False) as artifact:
        init_time = artifact["init_time_hours"]
        lead = artifact["anchor_lead_hours"]
        tau = artifact["tau_hours"]
        squared_error = artifact["squared_error_norm"]
        valid_channel = artifact["valid_channel"]
        channels = artifact["channel_names"].tolist()
    if channels != CHANNELS_ORDER:
        raise ValueError(f"{npz_path}: channel order mismatch")
    n_windows = tau.size
    if (
        init_time.shape != (n_windows,)
        or lead.shape != (n_windows,)
        or squared_error.shape != (n_windows, len(CHANNELS_ORDER))
        or valid_channel.shape != squared_error.shape
    ):
        raise ValueError(f"{npz_path}: inconsistent paired array shapes")
    index_sha256 = _sha256_arrays(init_time, lead, tau)
    if index_sha256 != paired.get("window_index_sha256"):
        raise ValueError(f"{path}: window index SHA-256 mismatch")
    if not np.isfinite(squared_error[valid_channel]).all():
        raise ValueError(f"{npz_path}: non-finite valid errors")
    return ForecastArtifact(
        name=name,
        json_path=path,
        npz_path=npz_path,
        init_time=init_time,
        lead=lead,
        tau=tau,
        squared_error=squared_error,
        valid_channel=valid_channel,
        index_sha256=index_sha256,
        delta_t_hours=delta_t_hours,
        protocol_taus=tuple(protocol_taus),
        model_kind=model_kind,
        checkpoint_sha256=checkpoint_sha256,
        data_provenance=data_provenance,
    )


def validate_selection_lineage(
    artifacts: dict[str, ForecastArtifact],
    selection: dict[str, object],
) -> None:
    selection_models = selection.get("models")
    if not isinstance(selection_models, dict):
        raise ValueError("selection lacks frozen model provenance")
    for name, artifact in artifacts.items():
        if name == "linear":
            if artifact.model_kind != "linear":
                raise ValueError("linear artifact has non-linear provenance")
            continue
        selected = selection_models.get(name)
        if not isinstance(selected, dict):
            raise ValueError(f"{name}: not present in frozen selection")
        if artifact.checkpoint_sha256 != selected.get("checkpoint_sha256"):
            raise ValueError(f"{name}: forecast/selection checkpoint mismatch")


def _family_masks(
    tau: np.ndarray,
    lead: np.ndarray,
    *,
    seen_taus: tuple[int, ...],
    unseen_taus: tuple[int, ...],
) -> dict[str, np.ndarray]:
    masks = {
        "all": np.ones(tau.shape, dtype=bool),
        "seen_tau": np.isin(tau, seen_taus),
        "unseen_tau": np.isin(tau, unseen_taus),
        "lead_fresh_0_48h": (lead >= 0) & (lead < 48),
        "lead_medium_48_120h": (lead >= 48) & (lead < 120),
        "lead_long_120_240h": (lead >= 120) & (lead < 240),
    }
    for hour in sorted(np.unique(tau).tolist()):
        masks[f"tau_h{int(hour)}"] = tau == hour
    return masks


def _block_means(
    artifact: ForecastArtifact,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    block_ids = np.unique(artifact.init_time[mask])
    values = np.where(
        artifact.valid_channel,
        artifact.squared_error,
        np.nan,
    )
    means = np.asarray(
        [
            np.nanmean(
                values[mask & (artifact.init_time == block_id)]
            )
            for block_id in block_ids
        ],
        dtype=np.float64,
    )
    finite = np.isfinite(means)
    return block_ids[finite], means[finite]


def _paired_test(
    difference: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict[str, float | int]:
    if difference.size < 2:
        raise ValueError("paired inference requires at least two blocks")
    rng = np.random.default_rng(seed)
    observed = float(difference.mean())
    bootstrap_indices = rng.integers(
        0,
        difference.size,
        size=(draws, difference.size),
    )
    bootstrap = difference[bootstrap_indices].mean(axis=1)
    signs = rng.choice(
        np.asarray([-1.0, 1.0]),
        size=(draws, difference.size),
    )
    null = (signs * difference).mean(axis=1)
    p_value = (
        1.0 + np.count_nonzero(np.abs(null) >= abs(observed))
    ) / (draws + 1.0)
    return {
        "n_blocks": int(difference.size),
        "delta_mse_challenger_minus_winner": observed,
        "bootstrap_ci95_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci95_high": float(np.quantile(bootstrap, 0.975)),
        "p_paired_sign_flip": float(p_value),
    }


def _holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, name in enumerate(ordered):
        value = min(1.0, (total - rank) * p_values[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def summarize_artifacts(
    artifacts: dict[str, ForecastArtifact],
    winner: str,
    *,
    draws: int,
    seed: int,
    seen_taus: tuple[int, ...] = (1, 3, 5),
    unseen_taus: tuple[int, ...] = (2, 4),
) -> dict[str, object]:
    if winner not in artifacts:
        raise ValueError(f"winner {winner!r} has no forecast artifact")
    reference = artifacts[winner]
    protocol_taus = set(reference.protocol_taus)
    observed_taus = set(int(value) for value in np.unique(reference.tau))
    seen_set = set(seen_taus)
    unseen_set = set(unseen_taus)
    if not seen_set or not unseen_set or seen_set & unseen_set:
        raise ValueError("seen and unseen taus must be non-empty and disjoint")
    if seen_set | unseen_set != protocol_taus:
        raise ValueError(
            "seen and unseen taus must partition the evaluated protocol"
        )
    if observed_taus != protocol_taus:
        raise ValueError("paired artifact does not cover every protocol tau")
    for name, artifact in artifacts.items():
        if (
            artifact.delta_t_hours != reference.delta_t_hours
            or artifact.protocol_taus != reference.protocol_taus
            or artifact.data_provenance != reference.data_provenance
        ):
            raise ValueError(f"{name}: forecast protocol differs from winner")
        if artifact.index_sha256 != reference.index_sha256:
            raise ValueError(f"{name}: forecast window index differs from winner")
        for label, left, right in (
            ("init_time", artifact.init_time, reference.init_time),
            ("lead", artifact.lead, reference.lead),
            ("tau", artifact.tau, reference.tau),
            (
                "valid_channel",
                artifact.valid_channel,
                reference.valid_channel,
            ),
        ):
            if not np.array_equal(left, right):
                raise ValueError(f"{name}: paired {label} differs from winner")

    family_results: dict[str, object] = {}
    for family, mask in _family_masks(
        reference.tau,
        reference.lead,
        seen_taus=seen_taus,
        unseen_taus=unseen_taus,
    ).items():
        if not mask.any():
            continue
        scores: dict[str, dict[str, float | int]] = {}
        block_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, artifact in artifacts.items():
            block_id, block_mean = _block_means(artifact, mask)
            block_values[name] = (block_id, block_mean)
            valid = artifact.valid_channel[mask]
            error = artifact.squared_error[mask]
            mse = float(error[valid].mean())
            scores[name] = {
                "mse_norm": mse,
                "rmse_norm": float(np.sqrt(mse)),
                "n_windows": int(mask.sum()),
                "n_valid_values": int(valid.sum()),
                "n_init_blocks": int(block_id.size),
            }

        comparisons: dict[str, dict[str, float | int | bool]] = {}
        raw_p: dict[str, float] = {}
        winner_blocks, winner_means = block_values[winner]
        for challenger in artifacts:
            if challenger == winner:
                continue
            challenger_blocks, challenger_means = block_values[challenger]
            if not np.array_equal(challenger_blocks, winner_blocks):
                raise ValueError(
                    f"{family}/{challenger}: paired blocks differ from winner"
                )
            digest = hashlib.sha256(
                f"{family}\0{challenger}".encode("utf-8")
            ).digest()
            local_seed = seed + int.from_bytes(digest[:4], "little")
            result = _paired_test(
                challenger_means - winner_means,
                draws=draws,
                seed=local_seed,
            )
            result["delta_rmse_percent_challenger_minus_winner"] = float(
                100.0
                * (
                    scores[challenger]["rmse_norm"]
                    / scores[winner]["rmse_norm"]
                    - 1.0
                )
            )
            comparisons[challenger] = result
            raw_p[challenger] = float(result["p_paired_sign_flip"])
        adjusted = _holm_adjust(raw_p)
        for challenger, p_holm in adjusted.items():
            comparison = comparisons[challenger]
            comparison["p_holm"] = p_holm
            comparison["winner_significantly_better"] = bool(
                p_holm < 0.05
                and comparison["delta_mse_challenger_minus_winner"] > 0
            )
            comparison["winner_significantly_worse"] = bool(
                p_holm < 0.05
                and comparison["delta_mse_challenger_minus_winner"] < 0
            )
            challenger_mse = float(scores[challenger]["mse_norm"])
            margin_mse = (
                (1.0 + FORECAST_RMSE_NONINFERIORITY_MARGIN) ** 2
                - 1.0
            ) * challenger_mse
            comparison["winner_noninferiority_margin_rmse"] = (
                FORECAST_RMSE_NONINFERIORITY_MARGIN
            )
            comparison["winner_noninferiority_margin_mse"] = margin_mse
            comparison["winner_noninferior"] = bool(
                float(comparison["bootstrap_ci95_low"]) >= -margin_mse
                and float(scores[winner]["rmse_norm"])
                <= (
                    (1.0 + FORECAST_RMSE_NONINFERIORITY_MARGIN)
                    * float(scores[challenger]["rmse_norm"])
                )
            )
        family_results[family] = {
            "scores": scores,
            "comparisons_to_winner": comparisons,
        }

    unseen_comparisons = family_results["unseen_tau"][
        "comparisons_to_winner"
    ]
    significant_unseen_regressions = [
        model
        for model, comparison in unseen_comparisons.items()
        if comparison["winner_significantly_worse"]
    ]
    significant_unseen_dominance = [
        model
        for model, comparison in unseen_comparisons.items()
        if comparison["winner_significantly_better"]
    ]
    unseen_noninferiority_failures = [
        model
        for model, comparison in unseen_comparisons.items()
        if not comparison["winner_noninferior"]
    ]
    return {
        "schema_version": 1,
        "winner": winner,
        "diagnostic_only": True,
        "selection_independent": True,
        "block_unit": "forecast_initialization",
        "draws": draws,
        "seed": seed,
        "protocol": {
            "delta_t_hours": reference.delta_t_hours,
            "evaluated_taus": list(reference.protocol_taus),
            "seen_taus": list(seen_taus),
            "unseen_taus": list(unseen_taus),
            "unseen_noninferiority_margin_rmse": (
                FORECAST_RMSE_NONINFERIORITY_MARGIN
            ),
        },
        "forecast_data_provenance_sha256": hashlib.sha256(
            json.dumps(
                reference.data_provenance,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "no_significant_unseen_forecast_regression": (
            not significant_unseen_regressions
        ),
        "significant_unseen_forecast_regressions": (
            significant_unseen_regressions
        ),
        "significant_unseen_forecast_dominance": (
            significant_unseen_dominance
        ),
        "unseen_forecast_noninferior_2pct_rmse": (
            not unseen_noninferiority_failures
        ),
        "unseen_forecast_noninferiority_failures": (
            unseen_noninferiority_failures
        ),
        "families": family_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help="name=path/to/forecast_anchor.json; repeat per model",
    )
    parser.add_argument("--selection", required=True)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--seen-taus", default="1,3,5")
    parser.add_argument("--unseen-taus", default="2,4")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.draws < 100:
        raise SystemExit("--draws must be at least 100")

    selection_path = Path(args.selection)
    selection = json.loads(selection_path.read_text())
    if selection.get("ood_attached_at_selection_time") is not False:
        raise SystemExit("candidate selection was not frozen before OOD")
    winner = selection["winner"]
    artifact_paths: dict[str, Path] = {}
    for spec in args.artifact:
        if "=" not in spec:
            raise SystemExit(f"invalid --artifact {spec!r}")
        name, raw_path = spec.split("=", 1)
        if not name or name in artifact_paths:
            raise SystemExit(f"duplicate or empty artifact name {name!r}")
        artifact_paths[name] = Path(raw_path)
    artifacts = {
        name: load_artifact(name, path)
        for name, path in artifact_paths.items()
    }
    validate_selection_lineage(artifacts, selection)
    seen_taus = tuple(
        int(value) for value in args.seen_taus.split(",") if value
    )
    unseen_taus = tuple(
        int(value) for value in args.unseen_taus.split(",") if value
    )
    output = summarize_artifacts(
        artifacts,
        winner,
        draws=args.draws,
        seed=args.seed,
        seen_taus=seen_taus,
        unseen_taus=unseen_taus,
    )
    output["provenance"] = {
        "summarizer": file_provenance(__file__),
        "selection": file_provenance(selection_path),
        "artifacts": {
            name: {
                "json": file_provenance(artifact.json_path),
                "paired_npz": file_provenance(artifact.npz_path),
            }
            for name, artifact in artifacts.items()
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[write] {output_path}")


if __name__ == "__main__":
    main()
