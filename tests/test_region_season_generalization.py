import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.eval.region_season_12h_eval import (
    REGIONS,
    SURFACE_REGIMES,
    _build_region_masks,
    _day_picks,
    _parse_model_entries,
    _temporary_environment,
    _write_paired_region_windows,
    region_season_evaluation_source_paths,
)
from tools.eval.summarize_region_season_generalization import (
    _sha256,
    compare,
    load_windows,
)
from weather_time_interp.metrics.physical_consistency import (
    CANONICAL_24_CHANNELS,
)


EXPECTED_REGIONS = tuple(REGIONS) + tuple(SURFACE_REGIMES)


def _evaluation_code_provenance() -> dict[str, str]:
    return {
        name: _sha256(path)
        for name, path in region_season_evaluation_source_paths().items()
    }


def _window_keys(year: int) -> list[tuple[int, int, int]]:
    start = np.datetime64(f"{year}-01-01T00", "h")
    keys = []
    for month in range(1, 13):
        for day in _day_picks(8):
            date = np.datetime64(
                f"{year}-{month:02d}-{day:02d}T00",
                "h",
            )
            t0 = int((date - start) / np.timedelta64(1, "h"))
            for tau in (2, 4):
                keys.append((year, t0, tau))
    return keys


def test_region_sampling_spans_the_full_month() -> None:
    days = _day_picks(8)

    assert days == [1, 5, 9, 13, 16, 20, 24, 28]
    assert len(days) == len(set(days))


def test_surface_regimes_form_fractional_land_ocean_partition() -> None:
    land_fraction = torch.tensor(
        [[0.0, 0.25, 0.75, 1.0]] * 4,
        dtype=torch.float32,
    )

    masks, _ = _build_region_masks(
        4,
        4,
        torch.device("cpu"),
        land_sea_fraction=land_fraction,
    )

    torch.testing.assert_close(
        masks["Land"][0, 0],
        land_fraction,
    )
    torch.testing.assert_close(
        masks["Land"] + masks["Ocean"],
        torch.ones(1, 1, 4, 4),
    )


def _write_model(
    root: Path,
    year: int,
    model: str,
    checkpoint_sha256: str,
    mse_value: float,
    *,
    region_names: tuple[str, ...] = EXPECTED_REGIONS,
    channel_names: tuple[str, ...] = CANONICAL_24_CHANNELS,
) -> None:
    keys = _window_keys(year)
    template = np.ones(
        (len(region_names), len(channel_names)),
        dtype=np.float32,
    )
    if "Tropics" in region_names:
        template[region_names.index("Tropics")] = mse_value
    values = [template.copy() for _ in keys]
    paired = _write_paired_region_windows(
        root / f"{model}.paired.npz",
        keys,
        values,
        region_names=list(region_names),
        channel_names=list(channel_names),
    )
    (root / f"{model}.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "model_name": model,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_provenance": {
                    "sha256": checkpoint_sha256,
                },
                "model_environment": {},
                "delta_t_hours": 6.0,
                "test_year": year,
                "samples_per_date": 2,
                "eval_days_per_month": 8,
                "eval_day_picks": _day_picks(8),
                "regions": list(region_names),
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
                "channel_names": list(channel_names),
                "eval_hours": [2, 4],
                "evaluation_input_provenance": {"static": "same"},
                "evaluation_dataset_provenance": {"year": year},
                "evaluation_index_sha256": paired[
                    "window_index_sha256"
                ],
                "evaluation_code_provenance": (
                    _evaluation_code_provenance()
                ),
                "paired_region_windows": paired,
            }
        )
    )


def _write_surface_model(
    root: Path,
    year: int,
    model: str,
    checkpoint_sha256: str,
    land_mse: float,
    ocean_mse: float,
) -> None:
    keys = _window_keys(year)
    template = np.ones(
        (len(EXPECTED_REGIONS), len(CANONICAL_24_CHANNELS)),
        dtype=np.float32,
    )
    template[EXPECTED_REGIONS.index("Land")] = land_mse
    template[EXPECTED_REGIONS.index("Ocean")] = ocean_mse
    values = [template.copy() for _ in keys]
    paired = _write_paired_region_windows(
        root / f"{model}.paired.npz",
        keys,
        values,
        region_names=list(EXPECTED_REGIONS),
        channel_names=list(CANONICAL_24_CHANNELS),
    )
    (root / f"{model}.json").write_text(
        json.dumps(
            {
                "schema_version": 6,
                "model_name": model,
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_provenance": {
                    "sha256": checkpoint_sha256,
                },
                "model_environment": {},
                "delta_t_hours": 6.0,
                "test_year": year,
                "samples_per_date": 2,
                "eval_days_per_month": 8,
                "eval_day_picks": _day_picks(8),
                "regions": list(EXPECTED_REGIONS),
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
                "eval_hours": [2, 4],
                "evaluation_input_provenance": {"static": "same"},
                "evaluation_dataset_provenance": {"year": year},
                "evaluation_index_sha256": paired[
                    "window_index_sha256"
                ],
                "evaluation_code_provenance": (
                    _evaluation_code_provenance()
                ),
                "paired_region_windows": paired,
            }
        )
    )


def test_paired_region_writer_rejects_misaligned_values(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="misaligned"):
        _write_paired_region_windows(
            tmp_path / "bad.npz",
            [(2021, 0, 2)],
            [],
            region_names=["Tropics"],
            channel_names=["t2m"],
        )


def test_region_model_parser_is_strict_and_environment_is_scoped(
    monkeypatch,
) -> None:
    parsed = _parse_model_entries(
        "first:/a.ckpt:FOO=one,second:/b.ckpt:BAR='two words'"
    )
    assert parsed == [
        ("first", "/a.ckpt", {"FOO": "one"}),
        ("second", "/b.ckpt", {"BAR": "two words"}),
    ]
    with pytest.raises(ValueError, match="invalid"):
        _parse_model_entries("missing_checkpoint")
    with pytest.raises(ValueError, match="unique"):
        _parse_model_entries("same:/a.ckpt,same:/b.ckpt")

    monkeypatch.setenv("FOO", "original")
    monkeypatch.delenv("NEW_VALUE", raising=False)
    with _temporary_environment({"FOO": "temporary", "NEW_VALUE": "set"}):
        assert os.environ["FOO"] == "temporary"
        assert os.environ["NEW_VALUE"] == "set"
    assert os.environ["FOO"] == "original"
    assert "NEW_VALUE" not in os.environ


def test_region_source_manifest_contains_existing_dependencies() -> None:
    sources = region_season_evaluation_source_paths()

    assert len(sources) >= 20
    assert all(path.is_file() for path in sources.values())
    assert "weather_time_interp/memmap_dataset.py" in sources
    assert "legacy/scripts/train_atm_vfi_12h_oddskip.py" in sources


def test_region_loader_rejects_stale_evaluator_source(
    tmp_path: Path,
) -> None:
    _write_model(tmp_path, 2021, "winner", "a" * 64, 1.0)
    artifact = tmp_path / "winner.json"
    payload = json.loads(artifact.read_text())
    source_name = next(iter(payload["evaluation_code_provenance"]))
    payload["evaluation_code_provenance"][source_name] = "0" * 64
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evaluation source hash mismatch"):
        load_windows(
            artifact,
            "winner",
            {"models": {"winner": {"checkpoint_sha256": "a" * 64}}},
            2021,
        )


def test_region_loader_rejects_unbound_evaluation_index(
    tmp_path: Path,
) -> None:
    _write_model(tmp_path, 2021, "winner", "a" * 64, 1.0)
    artifact = tmp_path / "winner.json"
    payload = json.loads(artifact.read_text())
    payload["evaluation_index_sha256"] = "0" * 64
    artifact.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="evaluation protocol mismatch"):
        load_windows(
            artifact,
            "winner",
            {"models": {"winner": {"checkpoint_sha256": "a" * 64}}},
            2021,
        )


@pytest.mark.parametrize("missing", ("Ocean", "mslp"))
def test_region_summary_rejects_incomplete_coverage(
    tmp_path: Path,
    missing: str,
) -> None:
    roots = {
        str(year): tmp_path / str(year)
        for year in (2020, 2021)
    }
    for root in roots.values():
        root.mkdir()
    selection = {
        "schema_version": 12,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {"winner": {"checkpoint_sha256": "a" * 64}},
    }
    regions = tuple(
        name for name in EXPECTED_REGIONS if name != missing
    )
    channels = tuple(
        name for name in CANONICAL_24_CHANNELS if name != missing
    )
    for year in (2020, 2021):
        _write_model(
            roots[str(year)],
            year,
            "winner",
            "a" * 64,
            1.0,
            region_names=regions,
            channel_names=channels,
        )

    with pytest.raises(ValueError, match="regional coverage mismatch"):
        compare(
            selection,
            roots,
            ("winner",),
            block_days=7,
            draws=100,
            seed=7,
        )


def test_region_summary_rejects_pre_robust_selection(
    tmp_path: Path,
) -> None:
    selection = {
        "schema_version": 11,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {"winner": {"checkpoint_sha256": "a" * 64}},
    }

    with pytest.raises(ValueError, match="invalid frozen selection"):
        compare(
            selection,
            {"2020": tmp_path},
            ("winner",),
            block_days=7,
            draws=100,
            seed=7,
        )


def test_region_summary_rejects_cross_model_provenance_mismatch(
    tmp_path: Path,
) -> None:
    roots = {
        str(year): tmp_path / str(year)
        for year in (2020, 2021)
    }
    for root in roots.values():
        root.mkdir()
    selection = {
        "schema_version": 12,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {
            "winner": {"checkpoint_sha256": "a" * 64},
            "reference": {"checkpoint_sha256": "b" * 64},
        },
    }
    for year in (2020, 2021):
        _write_model(
            roots[str(year)], year, "winner", "a" * 64, 1.0
        )
        _write_model(
            roots[str(year)], year, "reference", "b" * 64, 1.0
        )
    reference_path = roots["2021"] / "reference.json"
    payload = json.loads(reference_path.read_text())
    payload["evaluation_input_provenance"] = {"static": "different"}
    reference_path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="regional pairing mismatch"):
        compare(
            selection,
            roots,
            ("winner", "reference"),
            block_days=7,
            draws=100,
            seed=7,
        )


def test_region_season_summary_detects_ood_regression(
    tmp_path: Path,
) -> None:
    roots = {
        str(year): tmp_path / str(year)
        for year in (2020, 2021)
    }
    for root in roots.values():
        root.mkdir()
    selection = {
        "schema_version": 12,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {
            "winner": {"checkpoint_sha256": "a" * 64},
            "reference": {"checkpoint_sha256": "b" * 64},
        },
    }
    _write_model(roots["2020"], 2020, "winner", "a" * 64, 1.0)
    _write_model(roots["2020"], 2020, "reference", "b" * 64, 4.0)
    _write_model(roots["2021"], 2021, "winner", "a" * 64, 4.0)
    _write_model(roots["2021"], 2021, "reference", "b" * 64, 1.0)

    report = compare(
        selection,
        roots,
        ("winner", "reference"),
        block_days=7,
        draws=10000,
        seed=7,
    )

    assert not report["no_significant_2021_region_season_regression"]
    assert not report["winner_2021_region_season_noninferior_5pct"]
    assert report[
        "winner_2021_region_season_noninferiority_failures"
    ]
    regressions = report[
        "significant_2021_region_season_regressions"
    ]
    assert report["protocol"]["regions"] == list(EXPECTED_REGIONS)
    assert report["protocol"]["channels"] == list(
        CANONICAL_24_CHANNELS
    )
    assert {
        (item["region"], item["season"], item["model"])
        for item in regressions
    } == {
        ("Tropics", season, "reference")
        for season in ("DJF", "MAM", "JJA", "SON")
    }


def test_region_summary_detects_ocean_only_ood_regression(
    tmp_path: Path,
) -> None:
    roots = {
        str(year): tmp_path / str(year)
        for year in (2020, 2021)
    }
    for root in roots.values():
        root.mkdir()
    selection = {
        "schema_version": 12,
        "winner": "winner",
        "ood_attached_at_selection_time": False,
        "models": {
            "winner": {"checkpoint_sha256": "a" * 64},
            "reference": {"checkpoint_sha256": "b" * 64},
        },
    }
    _write_surface_model(
        roots["2020"],
        2020,
        "winner",
        "a" * 64,
        1.0,
        1.0,
    )
    _write_surface_model(
        roots["2020"],
        2020,
        "reference",
        "b" * 64,
        2.0,
        2.0,
    )
    _write_surface_model(
        roots["2021"],
        2021,
        "winner",
        "a" * 64,
        1.0,
        4.0,
    )
    _write_surface_model(
        roots["2021"],
        2021,
        "reference",
        "b" * 64,
        1.0,
        1.0,
    )

    report = compare(
        selection,
        roots,
        ("winner", "reference"),
        block_days=7,
        draws=10000,
        seed=8,
    )

    assert {
        (item["region"], item["season"], item["model"])
        for item in report[
            "significant_2021_region_season_regressions"
        ]
    } == {
        ("Ocean", season, "reference")
        for season in ("DJF", "MAM", "JJA", "SON")
    }
