#!/usr/bin/env python3
"""Create an evaluation-only checkpoint with a scaled Lagrange head."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch


HEAD_KEYS = (
    "net.base_knot_head.weight",
    "net.base_knot_head.bias",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scale_checkpoint(
    source: Path,
    output: Path,
    *,
    factor: float,
) -> None:
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("factor must be finite and positive")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    hparams = checkpoint.get("hyper_parameters", {})
    if hparams.get("arch") != "flow_compact_lagrange_l":
        raise ValueError("source is not a Lagrange checkpoint")
    if hparams.get("trainable_scope") != "base_knot_head":
        raise ValueError("source was not trained with a frozen backbone")
    state = checkpoint.get("state_dict", {})
    for key in HEAD_KEYS:
        if key not in state:
            raise ValueError(f"missing Lagrange head tensor: {key}")
        state[key] = state[key] * factor

    hparams["deployment_transform"] = {
        "kind": "lagrange_head_scale",
        "factor": factor,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": _sha256(source),
        "scaled_state_keys": list(HEAD_KEYS),
        "selection_only": True,
    }
    checkpoint["optimizer_states"] = []
    checkpoint["lr_schedulers"] = []
    checkpoint["callbacks"] = {}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    os.replace(temporary, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--factor", type=float, required=True)
    args = parser.parse_args()
    scale_checkpoint(args.source, args.output, factor=args.factor)


if __name__ == "__main__":
    main()
