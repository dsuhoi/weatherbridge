#!/usr/bin/env python3
"""Paired regional and seasonal OOD comparisons for a frozen winner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.eval.paired_block_bootstrap import (
    WindowMetrics,
    paired_block_bootstrap,
)
from tools.eval.region_season_12h_eval import (
    REGIONS,
    SEASONS,
    SURFACE_REGIMES,
    region_season_evaluation_source_paths,
)
from tools.eval.validate_upr_lite_selection import _annotate_holm
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


REGION_SEASON_NONINFERIORITY_MARGIN = 0.05


EXPECTED_REGIONS = tuple(REGIONS) + tuple(SURFACE_REGIMES)
EXPECTED_DAY_PICKS = (1, 5, 9, 13, 16, 20, 24, 28)
EXPECTED_UNSEEN_TAUS = {
    6: (2, 4),
    12: (4, 6, 8),
}


@dataclass(frozen=True)
class RegionWindows:
    model: str
    year: np.ndarray
    t0: np.ndarray
    tau: np.ndarray
    mse: np.ndarray
    regions: tuple[str, ...]
    channels: tuple[str, ...]
    index_sha256: str
    delta_t_hours: int
    test_year: int
    input_provenance: dict[str, Any]
    dataset_provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _index_sha256(year: np.ndarray, t0: np.ndarray, tau: np.ndarray) -> str:
    return hashlib.sha256(
        json.dumps(
            [
                (int(row_year), int(row_t0), int(row_tau))
                for row_year, row_t0, row_tau in zip(year, t0, tau)
            ],
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _validate_evaluation_code(
    payload: dict[str, Any],
    artifact_path: Path,
) -> None:
    recorded = payload.get("evaluation_code_provenance")
    current_paths = region_season_evaluation_source_paths()
    if (
        not isinstance(recorded, dict)
        or set(recorded) != set(current_paths)
    ):
        raise ValueError(
            f"{artifact_path}: evaluation source manifest mismatch"
        )
    for name, source_path in current_paths.items():
        if recorded.get(name) != _sha256(source_path):
            raise ValueError(
                f"{artifact_path}: evaluation source hash mismatch: {name}"
            )


def load_windows(
    path: Path,
    model: str,
    selection: dict[str, Any],
    expected_year: int,
) -> RegionWindows:
    payload = json.loads(path.read_text())
    selected = selection.get("models", {}).get(model)
    checkpoint_provenance = payload.get("checkpoint_provenance")
    if (
        payload.get("schema_version") != 6
        or payload.get("model_name") != model
        or not isinstance(selected, dict)
        or not isinstance(checkpoint_provenance, dict)
        or checkpoint_provenance.get("sha256")
        != payload.get("checkpoint_sha256")
        or payload.get("checkpoint_sha256")
        != selected.get("checkpoint_sha256")
    ):
        raise ValueError(f"{path}: model or checkpoint provenance mismatch")
    _validate_evaluation_code(payload, path)
    paired = payload.get("paired_region_windows", {})
    npz_path = path.parent / str(paired.get("path", ""))
    if (
        not npz_path.is_file()
        or npz_path.stat().st_size != paired.get("size_bytes")
        or _sha256(npz_path) != paired.get("sha256")
    ):
        raise ValueError(f"{path}: paired region artifact mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        year = np.asarray(arrays["year"])
        t0 = np.asarray(arrays["t0"])
        tau = np.asarray(arrays["tau"])
        mse = np.asarray(arrays["mse_norm"], dtype=np.float64)
        regions = tuple(str(value) for value in arrays["region_names"])
        channels = tuple(str(value) for value in arrays["channel_names"])
    if (
        regions != EXPECTED_REGIONS
        or tuple(payload.get("regions", ())) != EXPECTED_REGIONS
        or channels != CANONICAL_24_CHANNELS
        or tuple(payload.get("channel_names", ()))
        != CANONICAL_24_CHANNELS
    ):
        raise ValueError(f"{path}: regional coverage mismatch")
    if (
        mse.shape != (year.size, len(regions), len(channels))
        or t0.shape != year.shape
        or tau.shape != year.shape
        or not np.all(np.isfinite(mse))
        or np.any(mse < 0.0)
    ):
        raise ValueError(f"{path}: invalid paired region arrays")
    index_sha256 = _index_sha256(year, t0, tau)
    if index_sha256 != paired.get("window_index_sha256"):
        raise ValueError(f"{path}: paired region index hash mismatch")
    delta_t_hours = int(float(payload.get("delta_t_hours", -1)))
    test_year = payload.get("test_year")
    expected_taus = EXPECTED_UNSEEN_TAUS.get(delta_t_hours)
    input_provenance = payload.get("evaluation_input_provenance")
    dataset_provenance = payload.get("evaluation_dataset_provenance")
    model_environment = payload.get("model_environment")
    if (
        expected_taus is None
        or tuple(sorted(set(int(value) for value in tau))) != expected_taus
        or tuple(int(value) for value in payload.get("eval_hours", ()))
        != expected_taus
        or isinstance(test_year, bool)
        or test_year != expected_year
        or set(int(value) for value in year) != {expected_year}
        or payload.get("samples_per_date") != 2
        or payload.get("eval_days_per_month") != 8
        or tuple(payload.get("eval_day_picks", ())) != EXPECTED_DAY_PICKS
        or payload.get("seasons") != list(SEASONS)
        or payload.get("region_types")
        != {
            **{name: "geographic" for name in REGIONS},
            **{name: "surface_fraction" for name in SURFACE_REGIMES},
        }
        or payload.get("surface_regime_definition")
        != {
            "Land": "land_sea_fraction",
            "Ocean": "1 - land_sea_fraction",
        }
        or not isinstance(input_provenance, dict)
        or not input_provenance
        or not isinstance(dataset_provenance, dict)
        or not dataset_provenance
        or not isinstance(model_environment, dict)
        or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in model_environment.items()
        )
        or paired.get("n_windows") != year.size
        or payload.get("evaluation_index_sha256") != index_sha256
    ):
        raise ValueError(f"{path}: regional evaluation protocol mismatch")
    return RegionWindows(
        model=model,
        year=year,
        t0=t0,
        tau=tau,
        mse=mse,
        regions=regions,
        channels=channels,
        index_sha256=index_sha256,
        delta_t_hours=delta_t_hours,
        test_year=test_year,
        input_provenance=input_provenance,
        dataset_provenance=dataset_provenance,
    )


def _months(windows: RegionWindows) -> np.ndarray:
    months = np.empty(windows.t0.shape, dtype=np.int8)
    for year in np.unique(windows.year):
        mask = windows.year == year
        timestamps = (
            np.datetime64(f"{int(year):04d}-01-01T00", "h")
            + windows.t0[mask].astype("timedelta64[h]")
        )
        months[mask] = (
            timestamps.astype("datetime64[M]").astype(np.int64) % 12 + 1
        )
    return months


def compare(
    selection: dict[str, Any],
    roots: dict[str, Path],
    models: tuple[str, ...],
    *,
    block_days: int,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if (
        isinstance(selection.get("schema_version"), bool)
        or not isinstance(selection.get("schema_version"), int)
        or selection["schema_version"] < 12
        or selection.get("ood_attached_at_selection_time") is not False
    ):
        raise ValueError("invalid frozen selection")
    if not models or len(models) != len(set(models)):
        raise ValueError("regional model set must be non-empty and unique")
    selection_models = selection.get("models")
    if (
        not isinstance(selection_models, dict)
        or set(models) != set(selection_models)
    ):
        raise ValueError("regional model set differs from frozen selection")
    winner = str(selection["winner"])
    if winner not in models:
        raise ValueError("frozen winner is absent from regional model set")
    comparisons: dict[str, Any] = {}
    source_artifacts: dict[str, Any] = {}
    input_provenance_sha256: str | None = None
    dataset_provenance_sha256: dict[str, str] = {}
    shared_delta_t: int | None = None
    regressions: list[dict[str, Any]] = []
    noninferiority_failures: list[dict[str, Any]] = []
    for year, root in roots.items():
        windows = {
            model: load_windows(
                root / f"{model}.json",
                model,
                selection,
                int(year),
            )
            for model in models
        }
        reference = windows[winner]
        for model, item in windows.items():
            if (
                item.index_sha256 != reference.index_sha256
                or item.regions != reference.regions
                or item.channels != reference.channels
                or item.delta_t_hours != reference.delta_t_hours
                or item.test_year != reference.test_year
                or item.input_provenance != reference.input_provenance
                or item.dataset_provenance != reference.dataset_provenance
                or not np.array_equal(item.year, reference.year)
                or not np.array_equal(item.t0, reference.t0)
                or not np.array_equal(item.tau, reference.tau)
            ):
                raise ValueError(f"{year}/{model}: regional pairing mismatch")
        current_input_sha256 = _canonical_sha256(
            reference.input_provenance
        )
        if input_provenance_sha256 is None:
            input_provenance_sha256 = current_input_sha256
        elif input_provenance_sha256 != current_input_sha256:
            raise ValueError("regional input provenance differs across years")
        dataset_provenance_sha256[year] = _canonical_sha256(
            reference.dataset_provenance
        )
        if shared_delta_t is None:
            shared_delta_t = reference.delta_t_hours
        elif shared_delta_t != reference.delta_t_hours:
            raise ValueError("regional horizon differs across years")
        source_artifacts[year] = {
            model: {
                "path": str((root / f"{model}.json").resolve()),
                "sha256": _sha256(root / f"{model}.json"),
            }
            for model in models
        }
        months = _months(reference)
        taus = np.asarray(sorted(set(int(value) for value in reference.tau)))
        year_comparisons: dict[str, Any] = {}
        family: list[dict[str, Any]] = []
        for region_index, region in enumerate(reference.regions):
            year_comparisons[region] = {}
            for season, season_months in SEASONS.items():
                mask = np.isin(months, season_months)
                left = WindowMetrics(
                    reference.year[mask],
                    reference.t0[mask],
                    reference.tau[mask],
                    reference.mse[mask, region_index],
                    reference.channels,
                )
                cell = {
                    model: paired_block_bootstrap(
                        left,
                        WindowMetrics(
                            item.year[mask],
                            item.t0[mask],
                            item.tau[mask],
                            item.mse[mask, region_index],
                            item.channels,
                        ),
                        taus=taus,
                        block_days=block_days,
                        draws=draws,
                        seed=seed,
                    )
                    for model, item in windows.items()
                    if model != winner
                }
                family.extend(cell.values())
                year_comparisons[region][season] = cell
        _annotate_holm(
            family,
            output_key="p_holm_region_season_family",
        )
        for region, seasons in year_comparisons.items():
            for season, cell in seasons.items():
                for model, result in cell.items():
                    margin = (
                        REGION_SEASON_NONINFERIORITY_MARGIN
                        * float(result["right_rmse"])
                    )
                    result["winner_noninferiority_margin_rmse"] = (
                        margin
                    )
                    result["winner_noninferior"] = bool(
                        float(result["delta_ci95"][2]) <= margin
                        and float(result["left_rmse"])
                        <= (
                            (1.0 + REGION_SEASON_NONINFERIORITY_MARGIN)
                            * float(result["right_rmse"])
                        )
                    )
        if year == "2021":
            for region, seasons in year_comparisons.items():
                for season, cell in seasons.items():
                    for model, result in cell.items():
                        if not result["winner_noninferior"]:
                            noninferiority_failures.append(
                                {
                                    "region": region,
                                    "season": season,
                                    "model": model,
                                    "delta_ci95_high": result[
                                        "delta_ci95"
                                    ][2],
                                    "margin_rmse": result[
                                        "winner_noninferiority_margin_rmse"
                                    ],
                                }
                            )
                        if (
                            result["delta_left_minus_right"] > 0.0
                            and result["p_holm_region_season_family"] < 0.05
                        ):
                            regressions.append(
                                {
                                    "region": region,
                                    "season": season,
                                    "model": model,
                                    "relative_delta_pct": result[
                                        "relative_delta_pct"
                                    ],
                                    "p_raw": result[
                                        "p_paired_block_permutation"
                                    ],
                                    "p_holm": result[
                                        "p_holm_region_season_family"
                                    ],
                                }
                            )
        comparisons[year] = year_comparisons
    return {
        "schema_version": 1,
        "diagnostic_only": True,
        "winner": winner,
        "protocol": {
            "delta_t_hours": shared_delta_t,
            "regions": list(EXPECTED_REGIONS),
            "channels": list(CANONICAL_24_CHANNELS),
            "samples_per_date": 2,
            "eval_days_per_month": 8,
            "block_days": block_days,
            "draws": draws,
            "seed": seed,
            "noninferiority_margin_rmse": (
                REGION_SEASON_NONINFERIORITY_MARGIN
            ),
            "multiplicity": (
                "Holm within each year across every region-season-challenger"
            ),
        },
        "evaluation_input_provenance_sha256": input_provenance_sha256,
        "evaluation_dataset_provenance_sha256": (
            dataset_provenance_sha256
        ),
        "source_artifacts": source_artifacts,
        "comparisons": comparisons,
        "no_significant_2021_region_season_regression": not regressions,
        "significant_2021_region_season_regressions": regressions,
        "winner_2021_region_season_noninferior_5pct": (
            not noninferiority_failures
        ),
        "winner_2021_region_season_noninferiority_failures": (
            noninferiority_failures
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--root-2020", type=Path, required=True)
    parser.add_argument("--root-2021", type=Path, required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--block-days", type=int, default=7)
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    models = tuple(
        value.strip() for value in args.models.split(",") if value.strip()
    )
    report = compare(
        selection,
        {"2020": args.root_2020, "2021": args.root_2021},
        models,
        block_days=args.block_days,
        draws=args.draws,
        seed=args.seed,
    )
    report["selection_manifest"] = {
        "path": str(args.selection.resolve()),
        "sha256": _sha256(args.selection),
    }
    report["generator_sha256"] = _sha256(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "winner": report["winner"],
                "no_ood_regression": report[
                    "no_significant_2021_region_season_regression"
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
