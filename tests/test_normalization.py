from __future__ import annotations

import hashlib

import numpy as np
import torch

from weather_time_interp.normalization import (
    STATIC_FEATURES_3,
    STATIC_COSINE_LATITUDE_GRID,
    ChannelStats,
    file_provenance,
    static_feature_provenance,
    zarr_store_provenance,
)


def test_channel_stats_round_trip() -> None:
    stats = ChannelStats(
        ("a", "b"),
        np.array([10.0, -3.0], dtype=np.float32),
        np.array([2.0, 4.0], dtype=np.float32),
    )
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 2, 2)

    reconstructed = stats.denormalize(stats.normalize(values))

    np.testing.assert_allclose(reconstructed, values, rtol=0.0, atol=1e-6)


def test_file_provenance(tmp_path) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"weatherbridge")

    provenance = file_provenance(path)

    assert provenance["path"] == str(path.resolve())
    assert provenance["size_bytes"] == len(b"weatherbridge")
    assert provenance["sha256"] == hashlib.sha256(b"weatherbridge").hexdigest()


def test_static_feature_schema_is_explicit_and_ordered() -> None:
    assert STATIC_FEATURES_3 == (
        "land_sea_mask",
        "normalized_orography",
        "cosine_latitude",
    )


def test_production_static_geometry_is_identified(tmp_path) -> None:
    latitude = np.linspace(89.75, -89.75, 360, dtype=np.float32)
    values = torch.zeros(3, 360, 720)
    values[2] = torch.from_numpy(np.cos(np.deg2rad(latitude)))[:, None]
    path = tmp_path / "static.pt"
    torch.save(values, path)

    provenance = static_feature_provenance(path)

    semantic = provenance["semantic"]
    assert semantic["cosine_latitude_grid"] == STATIC_COSINE_LATITUDE_GRID
    assert semantic["offset_from_wb2_block_centres_degrees"] == -0.125
    assert semantic["max_abs_value_difference_from_block_centres"] < 0.0022


def test_zarr_store_provenance_tracks_metadata_and_chunk_state(
    tmp_path,
) -> None:
    store = tmp_path / "climatology.zarr"
    chunks = store / "t"
    chunks.mkdir(parents=True)
    (store / ".zmetadata").write_text('{"zarr_consolidated_format": 1}')
    (chunks / ".zarray").write_text('{"shape": [2]}')
    (chunks / "0").write_bytes(b"first")
    (chunks / "1").write_bytes(b"second")

    before = zarr_store_provenance(store, sampled_data_files=2)
    (chunks / "1").write_bytes(b"changed")
    after = zarr_store_provenance(store, sampled_data_files=2)

    assert before["file_count"] == 4
    assert before["metadata_file_count"] == 2
    assert before["data_file_count"] == 2
    assert before["sampled_data_file_count"] == 2
    assert before["metadata_sha256"] == after["metadata_sha256"]
    assert before["layout_sha256"] != after["layout_sha256"]
    assert (
        before["sampled_chunk_slices_sha256"]
        != after["sampled_chunk_slices_sha256"]
    )
    assert before["sampled_total_bytes"] > 0
    assert before["cache_identity_sha256"] != after["cache_identity_sha256"]
