#!/usr/bin/env python3
"""Blend compatible same-architecture checkpoints for evaluation."""
from __future__ import annotations

import argparse
import hashlib
import math
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


def blend_checkpoints(
    left_path: Path,
    right_path: Path,
    output: Path,
    *,
    left_weight: float,
) -> dict[str, Any]:
    if not math.isfinite(left_weight) or not 0.0 < left_weight < 1.0:
        raise ValueError("left_weight must be finite and strictly between 0 and 1")
    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (left_path, right_path)
    ]
    left, right = checkpoints
    left_hparams = left.get("hyper_parameters", {})
    right_hparams = right.get("hyper_parameters", {})
    for key in ("arch", "delta_t", "total_steps", "training_seed"):
        if left_hparams.get(key) != right_hparams.get(key):
            raise ValueError(f"checkpoint metadata mismatch at {key}")
    if left.get("epoch") != right.get("epoch"):
        raise ValueError("checkpoints must come from the same epoch")
    left_state = left.get("state_dict", {})
    right_state = right.get("state_dict", {})
    if tuple(left_state) != tuple(right_state):
        raise ValueError("state_dict keys or order do not match")

    blended_state: dict[str, torch.Tensor] = {}
    right_weight = 1.0 - left_weight
    for key, left_value in left_state.items():
        right_value = right_state[key]
        if left_value.shape != right_value.shape:
            raise ValueError(f"{key}: tensor shapes do not match")
        if left_value.is_floating_point() or left_value.is_complex():
            accumulator_dtype = (
                torch.complex128
                if left_value.is_complex()
                else torch.float64
            )
            blended_state[key] = (
                left_value.to(accumulator_dtype) * left_weight
                + right_value.to(accumulator_dtype) * right_weight
            ).to(left_value.dtype)
        else:
            if not torch.equal(left_value, right_value):
                raise ValueError(f"{key}: non-floating tensors do not match")
            blended_state[key] = left_value.clone()

    metadata = {
        "method": "linear_parameter_blend",
        "evaluation_only": True,
        "left_weight": left_weight,
        "right_weight": right_weight,
        "epoch": int(left["epoch"]),
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "inputs": [
            {
                "role": role,
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for role, path in (
                ("left", left_path),
                ("right", right_path),
            )
        ],
    }
    result = dict(right)
    result["state_dict"] = blended_state
    result["optimizer_states"] = []
    result["lr_schedulers"] = []
    result["callbacks"] = {}
    result["checkpoint_blend"] = metadata
    result["hyper_parameters"] = dict(right_hparams)
    result["hyper_parameters"]["deployment_transform"] = metadata

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
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = blend_checkpoints(
        args.left,
        args.right,
        args.output,
        left_weight=args.left_weight,
    )
    print(
        f"blended epoch={metadata['epoch']} "
        f"left_weight={metadata['left_weight']:g} "
        f"output={args.output} evaluation_only=true"
    )


if __name__ == "__main__":
    main()
