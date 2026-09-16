#!/usr/bin/env python3
"""Align a full-year metric artifact to a reference window order."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

import numpy as np

from tools.eval.paired_block_bootstrap import window_index_sha256


INDEX_COLUMNS = ("year", "t0", "tau")
TEMPORAL_INDEX_COLUMNS = ("temporal_year", "temporal_t0")
WINDOW_VALUE_PREFIXES = ("mse_norm_", "acc_", "physical_")
WINDOW_METADATA_NAMES = ("channel_names", "acc_channel_names")
TEMPORAL_VALUE_PREFIXES = ("temporal_curvature_mse_",)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_path(payload_path: Path, payload: Mapping[str, object]) -> Path:
    value = payload.get("window_metrics_file")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{payload_path}: missing window_metrics_file")
    path = Path(value)
    return path if path.is_absolute() else payload_path.parent / path


def _load_artifact(payload_path: Path) -> tuple[dict, Path, dict[str, np.ndarray]]:
    payload = json.loads(payload_path.read_text())
    window_path = _window_path(payload_path, payload)
    if not window_path.is_file():
        raise ValueError(f"{payload_path}: missing window metrics {window_path}")
    with np.load(window_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    for name in INDEX_COLUMNS:
        if name not in arrays:
            raise ValueError(f"{window_path}: missing index column {name}")
    year, t0, tau = (arrays[name] for name in INDEX_COLUMNS)
    if not (year.ndim == t0.ndim == tau.ndim == 1):
        raise ValueError(f"{window_path}: index columns must be vectors")
    if not (year.size == t0.size == tau.size):
        raise ValueError(f"{window_path}: index columns have different lengths")
    if year.size == 0:
        raise ValueError(f"{window_path}: empty window index")
    if not all(np.issubdtype(values.dtype, np.integer) for values in (year, t0, tau)):
        raise ValueError(f"{window_path}: index columns must have integer dtype")
    keys = list(zip(year.tolist(), t0.tolist(), tau.tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{window_path}: duplicate (year, t0, tau) windows")

    index_hash = window_index_sha256(year, t0, tau)
    protocol = payload.get("evaluation_protocol", {})
    provenance = payload.get("window_metrics_provenance", {})
    if not isinstance(protocol, dict) or protocol.get("index_sha256") != index_hash:
        raise ValueError(f"{payload_path}: protocol index hash mismatch")
    if not isinstance(provenance, dict):
        raise ValueError(f"{payload_path}: invalid window provenance")
    if provenance.get("index_sha256") != index_hash:
        raise ValueError(f"{payload_path}: provenance index hash mismatch")
    if provenance.get("sha256") != sha256_file(window_path):
        raise ValueError(f"{payload_path}: window file hash mismatch")
    temporal_columns = [name in arrays for name in TEMPORAL_INDEX_COLUMNS]
    if any(temporal_columns) and not all(temporal_columns):
        raise ValueError(f"{window_path}: incomplete temporal index")
    if all(temporal_columns):
        temporal_year, temporal_t0 = (
            arrays[name] for name in TEMPORAL_INDEX_COLUMNS
        )
        if (
            temporal_year.ndim != 1
            or temporal_t0.ndim != 1
            or temporal_year.size != temporal_t0.size
            or temporal_year.size == 0
        ):
            raise ValueError(f"{window_path}: invalid temporal index")
        if not all(
            np.issubdtype(values.dtype, np.integer)
            for values in (temporal_year, temporal_t0)
        ):
            raise ValueError(f"{window_path}: temporal index must be integer")
        temporal_keys = list(
            zip(temporal_year.tolist(), temporal_t0.tolist(), strict=True)
        )
        if len(temporal_keys) != len(set(temporal_keys)):
            raise ValueError(f"{window_path}: duplicate temporal windows")
        expected_temporal_hash = _temporal_index_sha256(
            temporal_year,
            temporal_t0,
        )
        if protocol.get("temporal_index_sha256") != expected_temporal_hash:
            raise ValueError(f"{payload_path}: temporal index hash mismatch")
        for name, values in arrays.items():
            if (
                name == "temporal_center_count"
                or name.startswith(TEMPORAL_VALUE_PREFIXES)
            ) and (values.ndim == 0 or values.shape[0] != len(temporal_keys)):
                raise ValueError(f"{window_path}: invalid {name} shape")
    return payload, window_path, arrays


def _temporal_index_sha256(year: np.ndarray, t0: np.ndarray) -> str:
    text = "\n".join(
        f"{current_year},{current_t0}"
        for current_year, current_t0 in zip(year, t0, strict=True)
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _exact_reorder(
    source_arrays: Mapping[str, np.ndarray],
    reference_arrays: Mapping[str, np.ndarray],
    columns: tuple[str, ...],
    *,
    context: str,
) -> np.ndarray:
    source_keys = list(
        zip(*(source_arrays[name].tolist() for name in columns), strict=True)
    )
    reference_keys = list(
        zip(*(reference_arrays[name].tolist() for name in columns), strict=True)
    )
    if len(source_keys) != len(reference_keys) or set(source_keys) != set(
        reference_keys
    ):
        missing = len(set(reference_keys) - set(source_keys))
        extra = len(set(source_keys) - set(reference_keys))
        raise ValueError(
            f"{context} sets differ: "
            f"source={len(source_keys)} reference={len(reference_keys)} "
            f"missing={missing} extra={extra}"
        )
    source_row = {key: index for index, key in enumerate(source_keys)}
    return np.asarray([source_row[key] for key in reference_keys], dtype=np.int64)


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def align_artifact(
    *,
    source_json: Path,
    reference_json: Path,
    output_json: Path,
    output_window: Path,
) -> dict:
    """Write a source artifact reordered exactly like the reference artifact."""
    source, source_window, source_arrays = _load_artifact(source_json)
    reference, reference_window, reference_arrays = _load_artifact(reference_json)

    order = _exact_reorder(
        source_arrays,
        reference_arrays,
        INDEX_COLUMNS,
        context="window",
    )
    n_windows = order.size
    aligned_arrays = {
        name: (
            values[order]
            if name in INDEX_COLUMNS
            or (
                name.startswith(WINDOW_VALUE_PREFIXES)
                and name not in WINDOW_METADATA_NAMES
                and values.ndim > 0
                and values.shape[0] == n_windows
            )
            else values
        )
        for name, values in source_arrays.items()
    }
    aligned_hash = window_index_sha256(
        *(aligned_arrays[name] for name in INDEX_COLUMNS)
    )
    reference_hash = reference["evaluation_protocol"]["index_sha256"]
    if aligned_hash != reference_hash:
        raise ValueError("aligned index hash does not match reference")
    output_payload = dict(source)
    output_payload["evaluation_protocol"] = dict(source["evaluation_protocol"])
    output_payload["evaluation_protocol"]["index_sha256"] = aligned_hash
    source_has_temporal = all(
        name in source_arrays for name in TEMPORAL_INDEX_COLUMNS
    )
    reference_has_temporal = all(
        name in reference_arrays for name in TEMPORAL_INDEX_COLUMNS
    )
    # Reference-only diagnostics do not affect alignment of source RMSE/ACC.
    if source_has_temporal and not reference_has_temporal:
        raise ValueError("source and reference temporal-index availability differs")
    if source_has_temporal:
        temporal_order = _exact_reorder(
            source_arrays,
            reference_arrays,
            TEMPORAL_INDEX_COLUMNS,
            context="temporal window",
        )
        for name, values in source_arrays.items():
            if (
                name in TEMPORAL_INDEX_COLUMNS
                or name == "temporal_center_count"
                or name.startswith(TEMPORAL_VALUE_PREFIXES)
            ):
                aligned_arrays[name] = values[temporal_order]
        temporal_hash = _temporal_index_sha256(
            *(aligned_arrays[name] for name in TEMPORAL_INDEX_COLUMNS)
        )
        reference_temporal_hash = reference["evaluation_protocol"].get(
            "temporal_index_sha256"
        )
        if temporal_hash != reference_temporal_hash:
            raise ValueError("aligned temporal index does not match reference")
        output_payload["evaluation_protocol"][
            "temporal_index_sha256"
        ] = temporal_hash
    _write_npz_atomic(output_window, aligned_arrays)
    output_window_hash = sha256_file(output_window)
    output_payload["window_metrics_file"] = os.path.relpath(
        output_window,
        start=output_json.parent,
    )
    output_payload["window_metrics_provenance"] = {
        "size_bytes": output_window.stat().st_size,
        "sha256": output_window_hash,
        "index_sha256": aligned_hash,
    }
    tool_path = Path(__file__).resolve()
    output_payload["publication_window_alignment"] = {
        "method": "exact_key_reorder_v1",
        "key_columns": list(INDEX_COLUMNS),
        "window_count": n_windows,
        "source_json_sha256": sha256_file(source_json),
        "source_window_sha256": sha256_file(source_window),
        "source_index_sha256": source["evaluation_protocol"]["index_sha256"],
        "reference_json_sha256": sha256_file(reference_json),
        "reference_window_sha256": sha256_file(reference_window),
        "reference_index_sha256": reference_hash,
        "tool_sha256": sha256_file(tool_path),
    }
    if source_has_temporal:
        output_payload["publication_window_alignment"].update(
            {
                "temporal_window_count": int(temporal_order.size),
                "source_temporal_index_sha256": source["evaluation_protocol"][
                    "temporal_index_sha256"
                ],
                "reference_temporal_index_sha256": reference[
                    "evaluation_protocol"
                ]["temporal_index_sha256"],
            }
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_json.write_text(json.dumps(output_payload, sort_keys=True) + "\n")
    os.replace(temporary_json, output_json)

    verified, verified_window, _ = _load_artifact(output_json)
    if verified_window.resolve() != output_window.resolve():
        raise ValueError("output JSON points to an unexpected window artifact")
    return verified


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-window", type=Path)
    args = parser.parse_args()
    output_window = args.output_window or (
        args.output_json.parent / "window_metrics" / f"{args.output_json.stem}.npz"
    )
    payload = align_artifact(
        source_json=args.source_json,
        reference_json=args.reference_json,
        output_json=args.output_json,
        output_window=output_window,
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_window": str(output_window),
                "index_sha256": payload["evaluation_protocol"]["index_sha256"],
                "window_count": payload["publication_window_alignment"]["window_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
