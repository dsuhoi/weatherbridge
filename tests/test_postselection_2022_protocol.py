from __future__ import annotations

import json
from pathlib import Path

from tools.data.build_postselection_2022_memmap import (
    CHANNEL_NAMES,
    ROOT,
    _marker_matches,
    selected_relative_hours,
)

ROOT_DIR = Path(__file__).resolve().parents[1]


def test_holdout_manifest_is_frozen_before_access() -> None:
    manifest = json.loads(
        (ROOT_DIR / "repro/postselection_holdout_2022.json").read_text()
    )
    assert manifest["status"] == "frozen_before_data_access"
    assert manifest["year"] == 2022
    assert manifest["source"] == f"gs://{ROOT}"
    assert manifest["day_of_month"] == [1, 8, 15, 22]
    assert manifest["anchor_hours_utc"] == [0]
    assert manifest["fields"] == list(CHANNEL_NAMES[:24])
    assert "cannot change" in manifest["selection_policy"]


def test_holdout_hours_are_disjoint_and_cover_each_window() -> None:
    hours = selected_relative_hours(2022, [1, 8, 15, 22], [0], 12)
    assert len(hours) == 48 * 13
    assert len(hours) == len(set(hours))
    assert min(hours) == 0
    assert max(hours) < 365 * 24


def test_holdout_uses_only_post_training_year() -> None:
    manifest = json.loads(
        (ROOT_DIR / "repro/postselection_holdout_2022.json").read_text()
    )
    assert manifest["year"] > 2021
    assert manifest["failure_policy"].startswith("If either primary horizon fails")


def test_resume_marker_is_bound_to_frozen_manifest(tmp_path: Path) -> None:
    marker = tmp_path / "temperature.complete.json"
    marker.write_text(
        json.dumps({"manifest_sha256": "frozen", "selected_hour_count": 624})
    )
    assert _marker_matches(marker, "frozen")
    assert not _marker_matches(marker, "changed")


def test_postselection_marker_binds_result_tex_and_champion() -> None:
    script = (
        ROOT_DIR / "scripts/run_postselection_2022_queue_cloudru.sh"
    ).read_text()

    assert '"tex_sha256"' in script
    assert '"champion_sha256"' in script
    assert '"champion_marker_sha256"' in script
    assert "paper/postselection_2022.tex" in script
    assert "reusing verified ${horizon}h evaluation artifacts" in script
    assert 'if [[ "$reuse_evaluation" -eq 1 ]]' in script


def test_rmse_map_queue_uses_canonical_weatherbridge_artifact() -> None:
    script = (
        ROOT_DIR / "scripts/run_weatherbridge_rmse_maps_cloudru.sh"
    ).read_text()

    assert "weather_time_interpolation_spectral_v6" in script
    assert "weatherbridge_flow_spectral_6h_bare.pt" in script
    assert 'RUNTIME_ROOT="${RUNTIME_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"' in script
    assert "exp_weatherdcae_14m_6h_6yr_refinev1_s202707_bs4" in script
    assert "/home/jovyan/dsuhoi/weather_time_interpolation/logs" in script
    assert "exp_fuxi_24ch_6yr_v2/epoch=7-step=17520.ckpt" in script
    assert "exp_sdyff_24ch_6yr/epoch=7-step=35032.ckpt" in script
    assert "journal_champion_v1/final.json" in script
    assert 'payload.get("winner") != marker.get("winner")' in script
    assert '"selected_model": selection["winner"]' in script
    assert '"schema_version": 2' in script
    assert '"artifact_sha256"' in script
    assert '"figure_sha256"' in script
    assert "exp_atmvfi_24ch_6h_2014_19_sparse135_lr1e4_s202707_protocol_v2" in script
    assert 'export RMSE_FORCE="${RMSE_FORCE:-0}"' in script
    assert 'export SDYFF_NLAT="${SDYFF_NLAT:-360}"' in script
    assert 'export SDYFF_NLON="${SDYFF_NLON:-720}"' in script
    assert 'export SDYFF_LAT_CROP="${SDYFF_LAT_CROP:-0}"' in script
    assert 'repo / "trainer_weather_hermite.py"' in script
    assert 'repo / "legacy" / "scripts" / "trainer_weather_hermite.py"' in script

    plotter = (
        ROOT_DIR / "scripts/make_fig_rmse_maps_5tau_per_field.py"
    ).read_text()
    assert 'fig.savefig(out_pdf, dpi=300, bbox_inches="tight")' in plotter
    assert 'RMSE_PLOT_ONLY' in plotter
    assert 'metadata.get("schema_version") != 1' in plotter
    assert 'metadata.get("sample_selection_sha256")' in plotter
    assert 'metadata.get("sample_strategy") != "uniform_over_full_year"' in plotter
