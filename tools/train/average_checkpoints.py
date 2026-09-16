#!/usr/bin/env python3
"""Build an evaluation-only checkpoint by averaging final training epochs."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_epoch_checkpoints(
    checkpoint_dir: Path,
    *,
    last_n: int,
) -> list[Path]:
    """Return the final ``last_n`` unique epoch checkpoints in epoch order."""
    if last_n < 2:
        raise ValueError("last_n must be at least 2")
    candidates = sorted(
        path
        for path in checkpoint_dir.glob("*.ckpt")
        if path.name != "last.ckpt"
    )
    by_epoch: dict[int, Path] = {}
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("checkpoint_average"):
            continue
        if "epoch" not in checkpoint:
            continue
        epoch = int(checkpoint["epoch"])
        if epoch in by_epoch:
            raise ValueError(
                f"duplicate checkpoint epoch {epoch}: "
                f"{by_epoch[epoch]} and {path}"
            )
        by_epoch[epoch] = path
    if len(by_epoch) < last_n:
        raise ValueError(
            f"{checkpoint_dir}: found {len(by_epoch)} epoch checkpoints, "
            f"need {last_n}"
        )
    selected_epochs = sorted(by_epoch)[-last_n:]
    if selected_epochs != list(
        range(selected_epochs[0], selected_epochs[-1] + 1)
    ):
        raise ValueError(
            f"final checkpoints are not consecutive: {selected_epochs}"
        )
    return [by_epoch[epoch] for epoch in selected_epochs]


def _validate_metadata(
    checkpoints: list[dict[str, Any]],
    paths: list[Path],
) -> None:
    reference_hparams = checkpoints[-1].get("hyper_parameters", {})
    reference_keys = tuple(checkpoints[-1]["state_dict"])
    for checkpoint, path in zip(checkpoints, paths, strict=True):
        if checkpoint.get("hyper_parameters", {}) != reference_hparams:
            raise ValueError(f"{path}: hyper_parameters do not match")
        if tuple(checkpoint.get("state_dict", {})) != reference_keys:
            raise ValueError(f"{path}: state_dict keys or order do not match")


def average_checkpoints(
    paths: list[Path],
    output: Path,
) -> dict[str, Any]:
    """Average floating state tensors and atomically save an eval checkpoint."""
    if len(paths) < 2:
        raise ValueError("at least two checkpoints are required")
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in paths
    ]
    _validate_metadata(checkpoints, paths)

    newest = checkpoints[-1]
    averaged_state: dict[str, torch.Tensor] = {}
    for key, newest_value in newest["state_dict"].items():
        values = [checkpoint["state_dict"][key] for checkpoint in checkpoints]
        if any(value.shape != newest_value.shape for value in values):
            raise ValueError(f"{key}: tensor shapes do not match")
        if newest_value.is_floating_point() or newest_value.is_complex():
            accumulator_dtype = (
                torch.complex128
                if newest_value.is_complex()
                else torch.float64
            )
            accumulator = values[0].to(accumulator_dtype)
            for value in values[1:]:
                accumulator.add_(value.to(accumulator_dtype))
            averaged_state[key] = (
                accumulator.div_(len(values)).to(newest_value.dtype)
            )
        else:
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise ValueError(f"{key}: non-floating tensors do not match")
            averaged_state[key] = newest_value.clone()

    result = dict(newest)
    result["state_dict"] = averaged_state
    result["optimizer_states"] = []
    result["lr_schedulers"] = []
    result["checkpoint_average"] = {
        "method": "uniform_parameter_mean",
        "evaluation_only": True,
        "epochs": [int(checkpoint["epoch"]) for checkpoint in checkpoints],
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "torch_version": torch.__version__,
        "inputs": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in paths
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(result, temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return result["checkpoint_average"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--last-n", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = discover_epoch_checkpoints(
        args.checkpoint_dir,
        last_n=args.last_n,
    )
    metadata = average_checkpoints(paths, args.output)
    print(
        f"averaged epochs={metadata['epochs']} "
        f"output={args.output} evaluation_only=true"
    )


if __name__ == "__main__":
    main()
