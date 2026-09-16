import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.assess_flow_spherical_ep_pilot import assess
from tools.eval.region_season_12h_eval import (
    REGIONS,
    SURFACE_REGIMES,
    _day_picks,
    _write_paired_region_windows,
    region_season_evaluation_source_paths,
)
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


def _write_curve(path: Path, epoch_one_hours: tuple[float, ...]) -> None:
    fields = [
        "epoch",
        "step",
        "val/recon_l1",
        *(f"val/rmse_h{hour}" for hour in range(1, 6)),
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch, values in (
            (0, tuple(value * 1.1 for value in epoch_one_hours)),
            (1, epoch_one_hours),
        ):
            writer.writerow(
                {
                    "epoch": epoch,
                    "step": (epoch + 1) * 1642 - 1,
                    "val/recon_l1": sum(values) / len(values),
                    **{
                        f"val/rmse_h{hour}": values[hour - 1]
                        for hour in range(1, 6)
                    },
                }
            )


def _write_region_metrics(
    path: Path,
    *,
    model: str,
    checkpoint: Path,
    rmse: float,
    eval_hours: tuple[int, ...] = (2, 4),
) -> None:
    checkpoint.write_bytes(f"{model}-checkpoint".encode())
    regions = tuple(REGIONS) + tuple(SURFACE_REGIMES)
    start = np.datetime64("2020-01-01T00", "h")
    window_keys = []
    for month in range(1, 13):
        for day in _day_picks(8):
            date = np.datetime64(
                f"2020-{month:02d}-{day:02d}T00",
                "h",
            )
            t0 = int((date - start) / np.timedelta64(1, "h"))
            for tau in eval_hours:
                window_keys.append((2020, t0, tau))
    template = np.full(
        (len(regions), len(CANONICAL_24_CHANNELS)),
        rmse * rmse,
        dtype=np.float32,
    )
    paired = _write_paired_region_windows(
        path.with_suffix(".paired.npz"),
        window_keys,
        [template.copy() for _ in window_keys],
        region_names=list(regions),
        channel_names=list(CANONICAL_24_CHANNELS),
    )
    code_provenance = {
        name: hashlib.sha256(source.read_bytes()).hexdigest()
        for name, source in region_season_evaluation_source_paths().items()
    }
    path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "model_name": model,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "checkpoint_provenance": {
                    "sha256": hashlib.sha256(
                        checkpoint.read_bytes()
                    ).hexdigest(),
                },
                "model_environment": {},
                "test_year": 2020,
                "delta_t_hours": 6.0,
                "eval_hours": list(eval_hours),
                "samples_per_date": 2,
                "eval_days_per_month": 8,
                "eval_day_picks": [1, 5, 9, 13, 16, 20, 24, 28],
                "regions": list(regions),
                "region_types": {
                    **{name: "geographic" for name in REGIONS},
                    **{
                        name: "surface_fraction"
                        for name in SURFACE_REGIMES
                    },
                },
                "surface_regime_definition": {
                    "Land": "land_sea_fraction",
                    "Ocean": "1 - land_sea_fraction",
                },
                "seasons": ["DJF", "MAM", "JJA", "SON"],
                "channel_names": list(CANONICAL_24_CHANNELS),
                "evaluation_input_provenance": {"static": "same"},
                "evaluation_dataset_provenance": {"year": 2020},
                "evaluation_index_sha256": paired[
                    "window_index_sha256"
                ],
                "evaluation_code_provenance": code_provenance,
                "paired_region_windows": paired,
            }
        )
    )


def test_promotes_compute_matched_pilot_within_limits(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(pilot, (0.061, 0.091, 0.101, 0.091, 0.061))

    report = assess(pilot, reference)

    assert report["promoted"] is True
    assert all(report["checks"].values())
    assert report["promotion_path"] == "global_gate"


def test_rejects_hidden_single_hour_regression(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(pilot, (0.064, 0.09, 0.10, 0.09, 0.06))

    report = assess(pilot, reference)

    assert report["promoted"] is False
    assert report["checks"]["per_hour_relative_delta_max"] is False


def test_polar_gain_rescues_bounded_global_regression(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    pilot_json = tmp_path / "pilot.json"
    reference_json = tmp_path / "reference.json"
    pilot_ckpt = tmp_path / "pilot.ckpt"
    reference_ckpt = tmp_path / "reference.ckpt"
    reference_hours = (0.06, 0.09, 0.10, 0.09, 0.06)
    _write_curve(reference, reference_hours)
    _write_curve(pilot, tuple(value * 1.03 for value in reference_hours))
    _write_region_metrics(
        pilot_json,
        model="flow_spherical_ep",
        checkpoint=pilot_ckpt,
        rmse=0.095,
    )
    _write_region_metrics(
        reference_json,
        model="upr_implicit_global_14m",
        checkpoint=reference_ckpt,
        rmse=0.100,
    )

    report = assess(
        pilot,
        reference,
        polar_pilot_json=pilot_json,
        polar_reference_json=reference_json,
        polar_pilot_checkpoint=pilot_ckpt,
        polar_reference_checkpoint=reference_ckpt,
    )

    assert report["promoted"] is True
    assert report["promotion_path"] == "polar_geometry_rescue"
    assert report["observed"]["polar_relative_delta"] == pytest.approx(-0.05)
    assert report["geometry_rescue_inputs"]["pilot_checkpoint"][
        "sha256"
    ] == hashlib.sha256(pilot_ckpt.read_bytes()).hexdigest()


def test_supports_state_compatible_spherical_upr_names(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    pilot_json = tmp_path / "pilot.json"
    reference_json = tmp_path / "reference.json"
    pilot_ckpt = tmp_path / "pilot.ckpt"
    reference_ckpt = tmp_path / "reference.ckpt"
    reference_hours = (0.06, 0.09, 0.10, 0.09, 0.06)
    _write_curve(reference, reference_hours)
    _write_curve(pilot, tuple(value * 1.03 for value in reference_hours))
    _write_region_metrics(
        pilot_json,
        model="upr_spherical_implicit_global_14m",
        checkpoint=pilot_ckpt,
        rmse=0.095,
    )
    _write_region_metrics(
        reference_json,
        model="upr_implicit_global_14m",
        checkpoint=reference_ckpt,
        rmse=0.100,
    )

    report = assess(
        pilot,
        reference,
        pilot_name="upr_spherical_implicit_global_14m",
        reference_name="upr_implicit_global_14m",
        polar_pilot_json=pilot_json,
        polar_reference_json=reference_json,
        polar_pilot_checkpoint=pilot_ckpt,
        polar_reference_checkpoint=reference_ckpt,
    )

    assert report["promoted"] is True
    assert report["promotion_path"] == "polar_geometry_rescue"
    assert report["pilot_name"] == "upr_spherical_implicit_global_14m"


@pytest.mark.parametrize(
    ("global_scale", "pilot_polar"),
    ((1.06, 0.090), (1.03, 0.101)),
)
def test_polar_rescue_rejects_weak_candidate(
    tmp_path: Path,
    global_scale: float,
    pilot_polar: float,
) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    pilot_json = tmp_path / "pilot.json"
    reference_json = tmp_path / "reference.json"
    pilot_ckpt = tmp_path / "pilot.ckpt"
    reference_ckpt = tmp_path / "reference.ckpt"
    reference_hours = (0.06, 0.09, 0.10, 0.09, 0.06)
    _write_curve(reference, reference_hours)
    _write_curve(
        pilot,
        tuple(value * global_scale for value in reference_hours),
    )
    _write_region_metrics(
        pilot_json,
        model="flow_spherical_ep",
        checkpoint=pilot_ckpt,
        rmse=pilot_polar,
    )
    _write_region_metrics(
        reference_json,
        model="upr_implicit_global_14m",
        checkpoint=reference_ckpt,
        rmse=0.100,
    )

    report = assess(
        pilot,
        reference,
        polar_pilot_json=pilot_json,
        polar_reference_json=reference_json,
        polar_pilot_checkpoint=pilot_ckpt,
        polar_reference_checkpoint=reference_ckpt,
    )

    assert report["promoted"] is False
    assert report["promotion_path"] == "excluded"


def test_polar_rescue_fails_closed_on_protocol_mismatch(
    tmp_path: Path,
) -> None:
    pilot = tmp_path / "pilot.csv"
    reference = tmp_path / "reference.csv"
    pilot_json = tmp_path / "pilot.json"
    reference_json = tmp_path / "reference.json"
    pilot_ckpt = tmp_path / "pilot.ckpt"
    reference_ckpt = tmp_path / "reference.ckpt"
    _write_curve(reference, (0.06, 0.09, 0.10, 0.09, 0.06))
    _write_curve(pilot, (0.061, 0.091, 0.101, 0.091, 0.061))
    _write_region_metrics(
        pilot_json,
        model="flow_spherical_ep",
        checkpoint=pilot_ckpt,
        rmse=0.09,
        eval_hours=(2,),
    )
    _write_region_metrics(
        reference_json,
        model="upr_implicit_global_14m",
        checkpoint=reference_ckpt,
        rmse=0.10,
    )

    with pytest.raises(ValueError, match="protocol mismatch"):
        assess(
            pilot,
            reference,
            polar_pilot_json=pilot_json,
            polar_reference_json=reference_json,
            polar_pilot_checkpoint=pilot_ckpt,
            polar_reference_checkpoint=reference_ckpt,
        )
