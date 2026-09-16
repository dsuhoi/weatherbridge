from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.eval.align_window_artifact import align_artifact, sha256_file
from tools.eval.paired_block_bootstrap import window_index_sha256


def _write_artifact(
    path: Path,
    rows: list[tuple[int, int, int]],
    *,
    temporal_rows: list[tuple[int, int]] | None = None,
) -> None:
    window = path.parent / "window_metrics" / f"{path.stem}.npz"
    window.parent.mkdir(parents=True, exist_ok=True)
    year = np.asarray([row[0] for row in rows], dtype=np.int16)
    t0 = np.asarray([row[1] for row in rows], dtype=np.int32)
    tau = np.asarray([row[2] for row in rows], dtype=np.int8)
    values = np.asarray([[row[1] + row[2]] for row in rows], dtype=np.float32)
    arrays = dict(
        year=year,
        t0=t0,
        tau=tau,
        channel_names=np.asarray(["T850", "Q850", "U850", "V850"]),
        acc_channel_names=np.asarray(["T850", "Q850", "U850", "V850"]),
        mse_norm_model=values,
    )
    if temporal_rows is not None:
        temporal_year = np.asarray(
            [row[0] for row in temporal_rows], dtype=np.int16
        )
        temporal_t0 = np.asarray(
            [row[1] for row in temporal_rows], dtype=np.int32
        )
        arrays.update(
            temporal_year=temporal_year,
            temporal_t0=temporal_t0,
            temporal_center_count=np.full(len(temporal_rows), 5, dtype=np.int8),
            temporal_centers=np.arange(1, 6, dtype=np.int8),
            temporal_curvature_mse_model=np.asarray(
                [[row[1] + 0.5] for row in temporal_rows], dtype=np.float32
            ),
        )
    np.savez_compressed(window, **arrays)
    index_hash = window_index_sha256(year, t0, tau)
    protocol = {"index_sha256": index_hash}
    if temporal_rows is not None:
        temporal_text = "\n".join(
            f"{current_year},{current_t0}"
            for current_year, current_t0 in temporal_rows
        )
        protocol["temporal_index_sha256"] = hashlib.sha256(
            temporal_text.encode("utf-8")
        ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "evaluation_protocol": protocol,
                "window_metrics_file": str(window.relative_to(path.parent)),
                "window_metrics_provenance": {
                    "size_bytes": window.stat().st_size,
                    "sha256": sha256_file(window),
                    "index_sha256": index_hash,
                },
            }
        )
    )


def test_align_artifact_reorders_every_window_array(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    reference = tmp_path / "reference.json"
    output = tmp_path / "published" / "model.json"
    output_window = output.parent / "window_metrics" / "model.npz"
    source_rows = [(2020, 0, 1), (2020, 12, 1), (2020, 0, 2), (2020, 12, 2)]
    reference_rows = [(2020, 0, 1), (2020, 0, 2), (2020, 12, 1), (2020, 12, 2)]
    source_temporal = [(2020, 24), (2020, 0)]
    reference_temporal = [(2020, 0), (2020, 24)]
    _write_artifact(source, source_rows, temporal_rows=source_temporal)
    _write_artifact(reference, reference_rows, temporal_rows=reference_temporal)

    payload = align_artifact(
        source_json=source,
        reference_json=reference,
        output_json=output,
        output_window=output_window,
    )

    with np.load(output_window, allow_pickle=False) as data:
        actual_rows = list(
            zip(data["year"].tolist(), data["t0"].tolist(), data["tau"].tolist())
        )
        assert actual_rows == reference_rows
        assert data["mse_norm_model"][:, 0].tolist() == [1.0, 2.0, 13.0, 14.0]
        assert data["channel_names"].tolist() == [
            "T850",
            "Q850",
            "U850",
            "V850",
        ]
        assert data["acc_channel_names"].tolist() == [
            "T850",
            "Q850",
            "U850",
            "V850",
        ]
        assert data["temporal_t0"].tolist() == [0, 24]
        assert data["temporal_curvature_mse_model"][:, 0].tolist() == [
            0.5,
            24.5,
        ]
        assert data["temporal_centers"].tolist() == [1, 2, 3, 4, 5]
    assert payload["evaluation_protocol"]["index_sha256"] == json.loads(
        reference.read_text()
    )["evaluation_protocol"]["index_sha256"]
    assert payload["publication_window_alignment"]["window_count"] == 4
    assert payload["publication_window_alignment"]["temporal_window_count"] == 2
    assert payload["window_metrics_provenance"]["sha256"] == sha256_file(
        output_window
    )


def test_align_artifact_rejects_different_window_sets(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    reference = tmp_path / "reference.json"
    _write_artifact(source, [(2020, 0, 1), (2020, 12, 1)])
    _write_artifact(reference, [(2020, 0, 1), (2020, 24, 1)])

    with pytest.raises(ValueError, match="window sets differ"):
        align_artifact(
            source_json=source,
            reference_json=reference,
            output_json=tmp_path / "out.json",
            output_window=tmp_path / "out.npz",
        )


def test_align_artifact_rejects_duplicate_windows(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    reference = tmp_path / "reference.json"
    duplicate_rows = [(2020, 0, 1), (2020, 0, 1)]
    _write_artifact(source, duplicate_rows)
    _write_artifact(reference, duplicate_rows)

    with pytest.raises(ValueError, match="duplicate"):
        align_artifact(
            source_json=source,
            reference_json=reference,
            output_json=tmp_path / "out.json",
            output_window=tmp_path / "out.npz",
        )


def test_align_artifact_allows_reference_only_diagnostics(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    reference = tmp_path / "reference.json"
    rows = [(2020, 0, 1), (2020, 0, 2)]
    _write_artifact(source, rows[::-1])
    _write_artifact(reference, rows, temporal_rows=[(2020, 0)])
    payload = align_artifact(source_json=source, reference_json=reference,
                             output_json=tmp_path / "out.json",
                             output_window=tmp_path / "out.npz")
    with np.load(tmp_path / "out.npz") as arrays:
        assert arrays["tau"].tolist() == [1, 2]
        assert "temporal_year" not in arrays.files
    assert "temporal_window_count" not in payload["publication_window_alignment"]


def test_align_artifact_rejects_different_temporal_window_sets(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    reference = tmp_path / "reference.json"
    rows = [(2020, 0, 1), (2020, 12, 1)]
    _write_artifact(source, rows, temporal_rows=[(2020, 0), (2020, 12)])
    _write_artifact(reference, rows, temporal_rows=[(2020, 0), (2020, 24)])

    with pytest.raises(ValueError, match="temporal window sets differ"):
        align_artifact(
            source_json=source,
            reference_json=reference,
            output_json=tmp_path / "out.json",
            output_window=tmp_path / "out.npz",
        )
