#!/usr/bin/env python3
"""Validate the matched WeatherDCAE Skip/NoSkip checkpoint pair."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


EXPECTED_UPDATES = 6560
EXPECTED_EPOCH = 7
EXPECTED_EXTRA_KEYS = {"net.skip_gates.0", "net.skip_gates.1"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _protocol_without_arch(protocol: dict[str, Any]) -> dict[str, Any]:
    result = dict(protocol)
    result.pop("requested_arch", None)
    result.pop("canonical_arch", None)
    return result


def validate_pair(base_path: Path, skip_path: Path) -> dict[str, Any]:
    import torch

    checkpoints = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in (base_path, skip_path)
    ]
    base, skip = checkpoints
    if [item.get("epoch") for item in checkpoints] != [EXPECTED_EPOCH] * 2:
        raise ValueError("both checkpoints must be saved after epoch 8")
    if [item.get("global_step") for item in checkpoints] != [EXPECTED_UPDATES] * 2:
        raise ValueError("both checkpoints must have exactly 6560 updates")

    hparams = [item.get("hyper_parameters", {}) for item in checkpoints]
    if [item.get("arch") for item in hparams] != ["dcae_14m", "wb_skip"]:
        raise ValueError("unexpected ablation architecture identifiers")
    protocols = [item.get("training_protocol") for item in hparams]
    if not all(isinstance(item, dict) for item in protocols):
        raise ValueError("checkpoint lacks structured training protocol")
    if _protocol_without_arch(protocols[0]) != _protocol_without_arch(protocols[1]):
        raise ValueError("Skip/NoSkip training protocols differ")
    if protocols[0].get("optimizer_steps_per_epoch") != 820:
        raise ValueError("unexpected optimizer steps per epoch")
    if protocols[0].get("seed") != 202707:
        raise ValueError("unexpected ablation seed")
    if protocols[0].get("train_years") != [2017, 2018, 2019]:
        raise ValueError("unexpected ablation training years")
    if protocols[0].get("train_tau_hours") != [1, 3, 5]:
        raise ValueError("unexpected ablation query-hour sampler")

    for field in ("training_input_provenance", "training_code_provenance"):
        values = [item.get(field) for item in hparams]
        if not values[0] or values[0] != values[1]:
            raise ValueError(f"Skip/NoSkip {field} differs")

    states = [item.get("state_dict", {}) for item in checkpoints]
    common = set(states[0]) & set(states[1])
    extra_skip = set(states[1]) - set(states[0])
    extra_base = set(states[0]) - set(states[1])
    if extra_skip != EXPECTED_EXTRA_KEYS or extra_base:
        raise ValueError("checkpoint key difference is not exactly two skip gates")
    mismatched_shapes = {
        key: [list(states[0][key].shape), list(states[1][key].shape)]
        for key in common
        if states[0][key].shape != states[1][key].shape
    }
    if mismatched_shapes:
        raise ValueError(f"common parameter shapes differ: {mismatched_shapes}")
    if any(states[1][key].numel() != 1 for key in EXPECTED_EXTRA_KEYS):
        raise ValueError("skip gates must be scalar")
    base_parameters = sum(value.numel() for value in states[0].values())
    skip_parameters = sum(value.numel() for value in states[1].values())
    if skip_parameters - base_parameters != 2:
        raise ValueError("Skip arm must add exactly two parameters")

    protocol_digest = hashlib.sha256(
        json.dumps(
            _protocol_without_arch(protocols[0]),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "comparison": "weatherdcae_skip_vs_noskip",
        "matched": True,
        "epoch": EXPECTED_EPOCH,
        "optimizer_updates": EXPECTED_UPDATES,
        "optimizer_steps_per_epoch": 820,
        "seed": 202707,
        "training_protocol_sha256": protocol_digest,
        "common_state_keys": len(common),
        "extra_skip_state_keys": sorted(extra_skip),
        "base_parameter_tensors_numel": base_parameters,
        "skip_parameter_tensors_numel": skip_parameters,
        "checkpoints": {
            "noskip": {
                "path": str(base_path.resolve()),
                "sha256": sha256_file(base_path),
                "size_bytes": base_path.stat().st_size,
            },
            "skip": {
                "path": str(skip_path.resolve()),
                "sha256": sha256_file(skip_path),
                "size_bytes": skip_path.stat().st_size,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--noskip", type=Path, required=True)
    parser.add_argument("--skip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = validate_pair(args.noskip, args.skip)
    _atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
