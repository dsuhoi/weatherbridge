"""Fail-fast validation for matched interpolation training recipes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_resume_lineage(path: str | Path) -> dict[str, Any]:
    """Capture the exact checkpoint and prior protocol used for a resume."""
    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size <= 0:
        raise ValueError(f"{checkpoint_path}: expected a non-empty checkpoint")
    import torch

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    lineage: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "epoch": int(checkpoint.get("epoch", -1)),
        "global_step": int(checkpoint.get("global_step", -1)),
        "model": {
            key: hparams.get(key)
            for key in (
                "arch",
                "total_steps",
                "delta_t",
                "training_seed",
            )
        },
    }
    for key in (
        "training_protocol",
        "training_input_provenance",
        "training_code_sha256",
        "resume_lineage",
        "initialization_lineage",
    ):
        if key in hparams:
            lineage[f"previous_{key}"] = hparams[key]
    return lineage


def validate_training_protocol_args(args: argparse.Namespace) -> None:
    """Reject split or recipe drift before opening terabyte-scale memmaps."""
    for label in ("years", "val_years", "train_tau_subset", "eval_tau"):
        values = list(getattr(args, label))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{label} must be a non-empty unique list")
    overlap = set(args.years).intersection(args.val_years)
    if overlap:
        raise ValueError(
            f"training and validation years overlap: {sorted(overlap)}"
        )
    if args.window_hours < 2:
        raise ValueError("window_hours must be at least 2")
    for label in ("train_tau_subset", "eval_tau"):
        invalid = [
            value
            for value in getattr(args, label)
            if value <= 0 or value >= args.window_hours
        ]
        if invalid:
            raise ValueError(
                f"{label} contains endpoint or out-of-range values: "
                f"{invalid}"
            )
    if not set(args.train_tau_subset).issubset(args.eval_tau):
        raise ValueError("train_tau_subset must be a subset of eval_tau")
    for label in (
        "samples_per_date_train",
        "samples_per_date_val",
    ):
        value = int(getattr(args, label))
        if value <= 0 or value > 24 or 24 % value:
            raise ValueError(f"{label} must be a positive divisor of 24")
    for label in ("bs", "val_bs", "accumulate", "max_epochs"):
        if int(getattr(args, label)) <= 0:
            raise ValueError(f"{label} must be positive")
    if args.workers < 0 or args.val_workers < 0:
        raise ValueError("worker counts must be non-negative")
    if (
        not args.gpus
        or len(args.gpus) != len(set(args.gpus))
        or any(gpu < 0 for gpu in args.gpus)
    ):
        raise ValueError("gpus must be a non-empty unique non-negative list")
    if not math.isfinite(args.lr) or args.lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if (
        not math.isfinite(args.limit_train_batches)
        or args.limit_train_batches <= 0.0
        or args.limit_train_batches > 1.0
    ):
        raise ValueError("limit_train_batches must lie in (0, 1]")
    if (
        not math.isfinite(args.limit_val_batches)
        or args.limit_val_batches <= 0.0
        or args.limit_val_batches > 1.0
    ):
        raise ValueError("limit_val_batches must lie in (0, 1]")
    if args.train_batches_per_epoch < 0:
        raise ValueError("train_batches_per_epoch must be non-negative")
    if (
        args.train_batches_per_epoch
        and args.train_batches_per_epoch % args.accumulate
    ):
        raise ValueError(
            "train_batches_per_epoch must contain complete accumulation groups"
        )


def memmap_dataset_provenance(
    memmap_dir: str | Path,
    years: list[int],
    *,
    sampled_bytes_per_file: int = 1024 * 1024,
) -> dict[str, Any]:
    """Fingerprint large yearly memmaps without reading them in full."""
    if not years or len(years) != len(set(years)):
        raise ValueError("years must be a non-empty unique list")
    if sampled_bytes_per_file < 3:
        raise ValueError("sampled_bytes_per_file must be at least 3")
    root = Path(memmap_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"{root}: expected a memmap directory")
    files: dict[str, dict[str, Any]] = {}
    sampled_total_bytes = 0
    for year in years:
        metadata_path = root / f"wb2_{year}.json"
        data_path = root / f"wb2_{year}.bin"
        metadata_bytes = metadata_path.read_bytes()
        metadata = json.loads(metadata_bytes)
        try:
            shape = [
                int(metadata[key])
                for key in ("T", "n_channels", "H", "W")
            ]
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{metadata_path}: invalid memmap shape metadata"
            ) from error
        if any(value <= 0 for value in shape):
            raise ValueError(f"{metadata_path}: shape values must be positive")
        stat = data_path.stat()
        expected_size = math.prod(shape) * 4
        if stat.st_size != expected_size:
            raise ValueError(
                f"{data_path}: size {stat.st_size} does not match "
                f"float32 shape {shape} ({expected_size} bytes)"
            )
        segment_size = min(
            stat.st_size,
            max(1, sampled_bytes_per_file // 3),
        )
        offsets = sorted(
            {
                0,
                max(0, (stat.st_size - segment_size) // 2),
                max(0, stat.st_size - segment_size),
            }
        )
        sampled_digest = hashlib.sha256()
        with data_path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(segment_size)
                sampled_total_bytes += len(chunk)
                sampled_digest.update(
                    f"{offset}\0{len(chunk)}\0".encode("utf-8")
                )
                sampled_digest.update(chunk)
        files[str(year)] = {
            "metadata_path": str(metadata_path.resolve()),
            "data_path": str(data_path.resolve()),
            "shape": shape,
            "dtype": "float32",
            "metadata_sha256": hashlib.sha256(
                metadata_bytes
            ).hexdigest(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sample_offsets": offsets,
            "sampled_slices_sha256": sampled_digest.hexdigest(),
        }
    identity = {
        "root": str(root),
        "years": list(years),
        "sampled_bytes_per_file_limit": sampled_bytes_per_file,
        "files": files,
    }
    return {
        **identity,
        "sampled_total_bytes": sampled_total_bytes,
        "identity_sha256": hashlib.sha256(
            json.dumps(
                identity,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
