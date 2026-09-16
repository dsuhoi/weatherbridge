import hashlib
import json
import os

import numpy as np
import pytest

from tools.eval.batch_eval_forecast_anchor import (
    CHANNELS_ORDER,
    HRES_QC_POLICY,
    LEGACY_HRES_QC_POLICY,
    CheckpointModel,
    Era5Cache,
    _atomic_savez,
    _atomic_write_json,
    _forecast_manifest_provenance,
    area_weighted_channel_mse,
    normalize_forecast_anchors,
    read_init,
    select_anchor_pairs,
    select_forecast_target_index,
    select_init_files,
)
from weather_time_interp.grid import (
    WB2_BLOCK_GRID_NAME,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)
from weather_time_interp.normalization import ChannelStats


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _write_forecast_archive_manifest(
    tmp_path,
    binary,
    sidecar,
    *,
    sidecar_policy=LEGACY_HRES_QC_POLICY,
    manifest_policy=None,
) -> None:
    builder = tmp_path / "builder.py"
    builder.write_text("# frozen builder\n")
    binary_stat = binary.stat()
    binary_sha = _sha256(binary)
    grid = {
        "name": WB2_BLOCK_GRID_NAME,
        "latitude_order": "north_to_south",
        "latitude_sha256": _array_sha256(wb2_block_average_latitudes()),
        "longitude_sha256": _array_sha256(wb2_block_average_longitudes()),
    }
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": (
                    4 if sidecar_policy == HRES_QC_POLICY else 3
                ),
                "channel_order": CHANNELS_ORDER,
                "channels_available": CHANNELS_ORDER,
                "grid": grid,
                "binary": {
                    "size_bytes": binary_stat.st_size,
                    "mtime_ns": binary_stat.st_mtime_ns,
                    "sha256": binary_sha,
                },
                "source_qc": {
                    "policy": sidecar_policy,
                    "channels": {
                        channel: {
                            "source_nonfinite_values": 0,
                            "affected_output_values": 0,
                            "minimum_finite_native_values_per_output": 4,
                            "output_nonfinite_values": 0,
                            "source_nonfinite_fraction": 0.0,
                        }
                        for channel in CHANNELS_ORDER
                    },
                    "source_nonfinite_values": 0,
                    "affected_output_values": 0,
                    "minimum_finite_native_values_per_output": 4,
                    "output_nonfinite_values": 0,
                },
            }
        )
    )
    manifest = {
        "schema_version": 2,
        "archive_kind": "weatherbench2_ifs_hres_forecast_anchors",
        "source_qc_policy": manifest_policy or sidecar_policy,
        "grid": grid,
        "builder": {"path": str(builder), "sha256": _sha256(builder)},
        "files": {
            binary.name: {
                "sidecar": sidecar.name,
                "sidecar_sha256": _sha256(sidecar),
                "binary_sha256": binary_sha,
                "binary_size_bytes": binary_stat.st_size,
                "binary_mtime_ns": binary_stat.st_mtime_ns,
            }
        },
    }
    (tmp_path / "forecast_archive_manifest.json").write_text(
        json.dumps(manifest)
    )


def _stats() -> ChannelStats:
    return ChannelStats(
        channel_names=("a", "b"),
        mean=np.asarray([10.0, -4.0], dtype=np.float32),
        std=np.asarray([2.0, 4.0], dtype=np.float32),
    )


def test_forecast_anchors_use_training_mean_and_std() -> None:
    values = np.asarray(
        [
            [[12.0, 8.0], [10.0, np.nan]],
            [[0.0, -8.0], [-4.0, 4.0]],
        ],
        dtype=np.float32,
    )

    normalized = normalize_forecast_anchors(values, _stats())

    np.testing.assert_allclose(
        normalized,
        np.asarray(
            [
                [[1.0, -1.0], [0.0, 0.0]],
                [[1.0, -1.0], [0.0, 2.0]],
            ],
            dtype=np.float32,
        ),
    )


def test_area_weighted_mse_downweights_polar_rows() -> None:
    prediction = np.zeros((2, 4, 3), dtype=np.float32)
    target = np.zeros_like(prediction)
    prediction[0, 0] = 2.0
    prediction[1, 1] = 4.0
    std = np.asarray([2.0, 4.0], dtype=np.float32)

    result = area_weighted_channel_mse(prediction, target, std)

    latitude = np.asarray([67.5, 22.5, -22.5, -67.5])
    weights = np.cos(np.deg2rad(latitude))
    weights /= weights.sum()
    np.testing.assert_allclose(result, [weights[0], weights[1]])
    assert result[0] < 0.25


def test_forecast_sidecar_must_use_canonical_channel_order(tmp_path) -> None:
    binary = tmp_path / "init_2021-01-01T00.bin"
    sidecar = binary.with_suffix(".json")
    shape = (1, len(CHANNELS_ORDER), 1, 1)
    np.zeros(shape, dtype=np.float32).tofile(binary)
    sidecar.write_text(
        json.dumps(
            {
                "shape": shape,
                "lead_hours": [0],
                "channel_order": list(reversed(CHANNELS_ORDER)),
                "channels_available": CHANNELS_ORDER,
                "dtype": "float32",
            }
        )
    )

    with pytest.raises(ValueError, match="channel_order"):
        read_init(binary, sidecar)


def test_init_subsample_spans_the_archive(tmp_path) -> None:
    files = [tmp_path / f"init_{index:02d}.bin" for index in range(10)]

    selected = select_init_files(files, 4)

    assert selected == [files[0], files[3], files[6], files[9]]


def test_forecast_manifest_detects_same_size_binary_change(
    tmp_path,
) -> None:
    binary = tmp_path / "init_2021-01-01T00.bin"
    sidecar = binary.with_suffix(".json")
    binary.write_bytes(bytes(range(64)))
    _write_forecast_archive_manifest(tmp_path, binary, sidecar)
    original_mtime = binary.stat().st_mtime_ns

    before = _forecast_manifest_provenance(
        tmp_path,
        [binary],
        sampled_bytes_per_binary=8,
    )
    with binary.open("r+b") as handle:
        handle.seek(28)
        handle.write(b"changed!")
    os.utime(binary, ns=(original_mtime, original_mtime))
    after = _forecast_manifest_provenance(
        tmp_path,
        [binary],
        sampled_bytes_per_binary=8,
    )

    assert binary.stat().st_size == 64
    assert before["manifest_sha256"] != after["manifest_sha256"]
    assert (
        before["files"][binary.name]["binary"]["sampled_sha256"]
        != after["files"][binary.name]["binary"]["sampled_sha256"]
    )


def test_forecast_manifest_preserves_recorded_builder_identity(tmp_path) -> None:
    binary = tmp_path / "init_2021-01-01T00.bin"
    sidecar = binary.with_suffix(".json")
    binary.write_bytes(bytes(range(64)))
    _write_forecast_archive_manifest(tmp_path, binary, sidecar)
    manifest = json.loads(
        (tmp_path / "forecast_archive_manifest.json").read_text()
    )
    recorded_sha256 = manifest["builder"]["sha256"]
    (tmp_path / "builder.py").write_text("# later builder revision\n")

    provenance = _forecast_manifest_provenance(tmp_path, [binary])

    assert provenance["builder"]["sha256"] == recorded_sha256
    assert provenance["builder"]["current_path_exists"] is True
    assert (
        provenance["builder"]["current_path_matches_recorded_sha256"]
        is False
    )


def test_current_manifest_accepts_stricter_legacy_sidecar(tmp_path) -> None:
    binary = tmp_path / "init_2021-01-01T00.bin"
    sidecar = binary.with_suffix(".json")
    binary.write_bytes(bytes(range(64)))
    _write_forecast_archive_manifest(
        tmp_path,
        binary,
        sidecar,
        sidecar_policy=LEGACY_HRES_QC_POLICY,
        manifest_policy=HRES_QC_POLICY,
    )

    provenance = _forecast_manifest_provenance(tmp_path, [binary])

    assert provenance["source_qc_policy"] == HRES_QC_POLICY


def test_current_manifest_accepts_current_qc_sidecar(tmp_path) -> None:
    binary = tmp_path / "init_2021-01-01T00.bin"
    sidecar = binary.with_suffix(".json")
    binary.write_bytes(bytes(range(64)))
    _write_forecast_archive_manifest(
        tmp_path,
        binary,
        sidecar,
        sidecar_policy=HRES_QC_POLICY,
    )

    provenance = _forecast_manifest_provenance(tmp_path, [binary])

    assert provenance["source_qc_policy"] == HRES_QC_POLICY


def test_forecast_outputs_are_atomically_replaced(tmp_path) -> None:
    npz_path = tmp_path / "paired.npz"
    json_path = tmp_path / "result.json"
    _atomic_savez(
        npz_path,
        values=np.asarray([1.0, 2.0], dtype=np.float32),
    )
    _atomic_write_json(json_path, {"complete": True})

    with np.load(npz_path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["values"], [1.0, 2.0])
    assert json.loads(json_path.read_text()) == {"complete": True}
    assert not list(tmp_path.glob(".*.tmp*"))


def test_anchor_pairs_support_window_wider_than_forecast_step() -> None:
    leads = [0, 6, 12, 18, 24]

    assert select_anchor_pairs(leads, 12) == [(0, 2), (1, 3), (2, 4)]
    assert select_anchor_pairs(leads, 6) == [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
    ]
    assert select_anchor_pairs(
        leads,
        6,
        lead_stride_hours=12,
        maximum_left_lead_hours=24,
    ) == [(0, 1), (2, 3)]


def test_anchor_pairs_reject_ambiguous_lead_metadata() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_anchor_pairs([0, 6, 6, 12], 6)
    with pytest.raises(ValueError, match="sorted"):
        select_anchor_pairs([0, 12, 6], 6)
    with pytest.raises(ValueError, match="lead_stride_hours"):
        select_anchor_pairs([0, 6], 6, lead_stride_hours=0)


def test_era5_cache_rejects_unwritten_sparse_hours(tmp_path) -> None:
    metadata = {
        "shape": [4, 24, 2, 2],
        "sparse_file": True,
        "selected_relative_hours": [0, 2],
    }
    (tmp_path / "wb2_2022.json").write_text(json.dumps(metadata))
    values = np.memmap(
        tmp_path / "wb2_2022.bin",
        dtype="float32",
        mode="w+",
        shape=tuple(metadata["shape"]),
    )
    values[0] = 1.0
    values[2] = 2.0
    values.flush()

    cache = Era5Cache(tmp_path)

    assert np.all(cache.hour(np.datetime64("2022-01-01T00")) == 1.0)
    assert cache.hour(np.datetime64("2022-01-01T01")) is None
    assert np.all(cache.hour(np.datetime64("2022-01-01T02")) == 2.0)


def test_same_forecast_target_requires_an_available_intermediate_lead() -> None:
    leads = [0, 6, 12, 18]

    assert select_forecast_target_index(leads, 0, 6) == 1
    assert select_forecast_target_index(leads, 1, 6) == 2
    assert select_forecast_target_index(leads, 0, 3) is None


def test_query_match_checkpoint_uses_inference_loader(
    monkeypatch,
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    from tools.eval import capmatched_loader

    checkpoint_path = tmp_path / "query.ckpt"
    checkpoint_path.touch()
    checkpoint = {
        "epoch": 1,
        "global_step": 3284,
        "hyper_parameters": {
            "arch": "upr_query_match_14m",
            "delta_t": 6.0,
        },
    }
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    calls = []

    class FakeInference:
        def __call__(self, x0, xT, tau):
            calls.append((x0.shape, xT.shape, tau.tolist()))
            return xT, {"aux": True}

    def fake_loader(path, device, *, static_path):
        calls.append((path, device.type, static_path))
        return FakeInference(), "capmatched_upr_query_match_14m"

    monkeypatch.setattr(
        capmatched_loader,
        "load_capmatched_checkpoint",
        fake_loader,
    )
    model = CheckpointModel(
        str(checkpoint_path),
        "upr_query_match_14m",
        "cpu",
        _stats(),
        static_path="static.pt",
        delta_t_hours=6,
        taus=[1, 2, 3, 4, 5],
    )
    x0 = np.zeros((2, 2, 2), dtype=np.float32)
    xT = np.full_like(x0, 2.0)

    prediction = model(x0, xT, tau_h=2)

    np.testing.assert_allclose(prediction, xT)
    assert calls[0] == (str(checkpoint_path), "cpu", "static.pt")
    assert calls[1][2] == [pytest.approx(2 / 6)]


@pytest.mark.parametrize(
    "arch",
    ["flow_pp3_hres_aug", "flow_pp3_hres_residual"],
)
def test_checkpoint_model_forwards_anchor_lead_conditioning(
    monkeypatch,
    tmp_path,
    arch,
) -> None:
    torch = pytest.importorskip("torch")
    from tools.eval import capmatched_loader

    checkpoint_path = tmp_path / "lead.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "hyper_parameters": {
                "arch": arch,
                "delta_t": 6.0,
            }
        },
    )
    observed = []

    class FakeInference:
        def __call__(self, x0, xT, tau, lead):
            observed.append(lead.detach().cpu().tolist())
            return xT

    monkeypatch.setattr(
        capmatched_loader,
        "load_capmatched_checkpoint",
        lambda *args, **kwargs: (FakeInference(), "capmatched"),
    )
    model = CheckpointModel(
        str(checkpoint_path),
        arch,
        "cpu",
        _stats(),
        static_path="static.pt",
        delta_t_hours=6,
        taus=[1, 2],
    )
    values = np.zeros((2, 2, 2), dtype=np.float32)

    model.predict_taus(
        values,
        values,
        [1, 2],
        anchor_lead_hours=72,
    )

    assert observed == [[72.0, 72.0]]


def test_checkpoint_arch_mismatch_fails_before_model_construction(
    monkeypatch,
    tmp_path,
) -> None:
    torch = pytest.importorskip("torch")
    from tools.eval import capmatched_loader

    checkpoint_path = tmp_path / "wrong.ckpt"
    checkpoint_path.touch()
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: {
            "hyper_parameters": {
                "arch": "flow_pp3",
                "delta_t": 6.0,
            }
        },
    )

    def unexpected_loader(*args, **kwargs):
        raise AssertionError("loader must not run for mismatched metadata")

    monkeypatch.setattr(
        capmatched_loader,
        "load_capmatched_checkpoint",
        unexpected_loader,
    )
    with pytest.raises(ValueError, match="requested arch"):
        CheckpointModel(
            str(checkpoint_path),
            "upr_query_match_14m",
            "cpu",
            _stats(),
            static_path="static.pt",
            delta_t_hours=6,
            taus=[1, 2, 3, 4, 5],
        )
