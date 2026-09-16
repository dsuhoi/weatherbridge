import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from weather_time_interp.hres_finetune_dataset import (
    HRESForecastAnchorDataset,
    hres_finetune_dataset_provenance,
)
from weather_time_interp.normalization import ChannelStats, PAPER_CHANNELS_24


def _sparse_file(path, size: int) -> None:
    with path.open("wb") as handle:
        handle.truncate(size)


def _write_archive(root) -> None:
    shape = (2, 24, 360, 720)
    binary = root / "init_2017-01-05T00.bin"
    sidecar = binary.with_suffix(".json")
    _sparse_file(binary, int(np.prod(shape)) * 4)
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "init_time": "2017-01-05T00",
                "lead_hours": [0, 6],
                "shape": list(shape),
                "channel_order": list(PAPER_CHANNELS_24),
                "dtype": "float32",
            }
        )
    )
    stat = binary.stat()
    (root / "forecast_archive_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "archive_kind": "weatherbench2_ifs_hres_forecast_anchors",
                "files": {
                    binary.name: {
                        "sidecar": sidecar.name,
                        "binary_size_bytes": stat.st_size,
                        "binary_mtime_ns": stat.st_mtime_ns,
                    }
                },
            }
        )
    )


def _write_era5(root) -> None:
    shape = (7, 27, 360, 720)
    (root / "wb2_2017.json").write_text(
        json.dumps(
            {
                "T": shape[0],
                "n_channels": shape[1],
                "H": shape[2],
                "W": shape[3],
            }
        )
    )
    _sparse_file(root / "wb2_2017.bin", int(np.prod(shape)) * 4)


def test_hres_split_indexes_only_declared_query_hours(tmp_path, monkeypatch) -> None:
    forecast = tmp_path / "forecast"
    era5 = tmp_path / "era5"
    forecast.mkdir()
    era5.mkdir()
    _write_archive(forecast)
    _write_era5(era5)
    stats = ChannelStats(
        channel_names=tuple(PAPER_CHANNELS_24),
        mean=np.zeros(24, dtype=np.float32),
        std=np.ones(24, dtype=np.float32),
    )
    monkeypatch.setattr(
        "weather_time_interp.hres_finetune_dataset.load_channel_stats",
        lambda *_: stats,
    )
    monkeypatch.setattr(
        "weather_time_interp.hres_finetune_dataset._forecast_manifest_provenance",
        lambda root, files: {
            "path": str(root),
            "init_count": len(files),
            "manifest_sha256": "a" * 64,
        },
    )

    dataset = HRESForecastAnchorDataset(
        forecast,
        era5,
        [2017],
        train=True,
        train_hours=[1, 3, 5],
        lead_stride_hours=24,
        maximum_left_lead_hours=120,
        stats_path=tmp_path / "unused.nc",
        surface_stats_path=tmp_path / "unused.json",
    )

    assert len(dataset) == 3
    assert [record[-1] for record in dataset.index] == [1, 3, 5]
    provenance = hres_finetune_dataset_provenance(dataset)
    assert provenance["future_analysis_as_input"] is False
    assert provenance["target_source"] == "ERA5_at_intermediate_valid_time"
    assert provenance["sample_count"] == 3
    assert provenance["forecast_archive"]["init_count"] == 1


def test_hres_weight_finetuning_rejects_12h(tmp_path) -> None:
    with pytest.raises(ValueError, match="6 h task"):
        HRESForecastAnchorDataset(
            tmp_path,
            tmp_path,
            [2017],
            max_tau_hours=12,
            stats_path=tmp_path / "unused.nc",
            surface_stats_path=tmp_path / "unused.json",
        )


def test_hres_forecast_error_augmentation_scales_paired_anchor_errors(
    tmp_path,
    monkeypatch,
) -> None:
    forecast = tmp_path / "forecast"
    era5 = tmp_path / "era5"
    forecast.mkdir()
    era5.mkdir()
    _write_archive(forecast)
    _write_era5(era5)
    stats = ChannelStats(
        channel_names=tuple(PAPER_CHANNELS_24),
        mean=np.zeros(24, dtype=np.float32),
        std=np.ones(24, dtype=np.float32),
    )
    monkeypatch.setattr(
        "weather_time_interp.hres_finetune_dataset.load_channel_stats",
        lambda *_: stats,
    )
    monkeypatch.setattr(
        "weather_time_interp.hres_finetune_dataset._forecast_manifest_provenance",
        lambda root, files: {"init_count": len(files)},
    )
    dataset = HRESForecastAnchorDataset(
        forecast,
        era5,
        [2017],
        train=True,
        train_hours=[1],
        forecast_error_augmentation_repeats=2,
        forecast_error_scale_min=0.5,
        forecast_error_scale_max=1.5,
        forecast_error_augmentation_seed=17,
        stats_path=tmp_path / "unused.nc",
        surface_stats_path=tmp_path / "unused.json",
    )
    forecast_values = np.empty((2, 24, 360, 720), dtype=np.float32)
    forecast_values[0].fill(1.0)
    forecast_values[1].fill(2.0)
    dataset._forecast_arrays[0] = forecast_values
    monkeypatch.setattr(
        dataset,
        "_era5_hour",
        lambda _: np.zeros((24, 360, 720), dtype=np.float32),
    )

    original = dataset[0]
    augmented = dataset[1]

    assert len(dataset) == 2
    assert original["forecast_error_scale"].item() == 1.0
    scale = augmented["forecast_error_scale"].item()
    assert 0.5 <= scale <= 1.5
    assert scale != 1.0
    np.testing.assert_allclose(augmented["x0"].numpy(), scale)
    np.testing.assert_allclose(augmented["x1"].numpy(), 2.0 * scale)
    assert augmented["forecast_lead_hours"].item() == 0.0


def test_hres_launcher_builds_the_frozen_108x24_archive() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/run_hres_finetune_6h_cloudru.sh").read_text()
    protocol = json.loads((root / "repro/hres_finetune_protocol.json").read_text())

    assert "wti_hres_finetune_2017_2020_108x24_v2" in script
    assert "build_hres_forecast_memmap.py" in script
    assert 'len(manifest.get("files", {})) != 132' in script
    assert len(protocol["splits"]["optimization"]["init_times"]) == 108
    assert (
        len(
            protocol["splits"]["validation_and_checkpoint_selection"][
                "init_times"
            ]
        )
        == 24
    )
    assert protocol["optimization"]["epochs"] == 10
    qc = protocol["data_contract"]["source_quality_policy"]
    assert qc["minimum_finite_native_values_per_block"] == 2
    assert (
        qc["maximum_nonfinite_fraction_per_channel_and_initialisation"]
        == 1.0e-6
    )
    assert (
        "2018-10-01T00"
        in protocol["splits"]["optimization"]["init_times"]
    )


def test_confirmation_cache_covers_the_frozen_hres_index() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "repro/hres_finetune_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads(
        (
            root / "repro/hres_finetune_confirmation_era5_2022.json"
        ).read_text()
    )
    script = (
        root / "scripts/run_hres_finetune_confirmation_2022_cloudru.sh"
    ).read_text()

    assert manifest["parent_manifest_sha256"] == hashlib.sha256(
        protocol_path.read_bytes()
    ).hexdigest()
    assert manifest["init_times"] == protocol["splits"][
        "date_level_confirmation"
    ]["init_times"]
    assert manifest["left_forecast_leads_hours"] == protocol[
        "data_contract"
    ]["left_forecast_leads_hours"]
    assert manifest["maximum_forecast_lead_hours"] == 102
    assert "wb2_0p5_hres_confirmation_2022_108x24_v2" in script
    assert "--forecast-lead-stride-hours 24" in script
    assert "--forecast-maximum-left-lead-hours 120" in script
