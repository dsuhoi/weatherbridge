#!/usr/bin/env python3
"""Build a hash-locked manifest for a capacity-matched output ensemble."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--expected-arch", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoints) < 2:
        parser.error("at least two checkpoints are required")

    members = []
    seeds: set[int] = set()
    code_fingerprints: set[str] = set()
    input_fingerprints: set[str] = set()
    for raw_path in args.checkpoints:
        path = raw_path.expanduser().resolve()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        hparams = checkpoint.get("hyper_parameters", {})
        if hparams.get("arch") != args.expected_arch:
            raise ValueError(f"{path}: expected arch {args.expected_arch!r}")
        seed = int(hparams["training_seed"])
        if seed in seeds:
            raise ValueError(f"duplicate training seed {seed}")
        seeds.add(seed)
        state = checkpoint.get("state_dict", {})
        parameters = sum(
            value.numel()
            for key, value in state.items()
            if key.startswith("net.")
        )
        code_fingerprints.add(
            hashlib.sha256(
                json.dumps(
                    hparams.get("training_code_sha256", {}),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        input_fingerprints.add(
            hashlib.sha256(
                json.dumps(
                    hparams.get("training_input_provenance", {}),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
        )
        members.append(
            {
                "checkpoint": str(path),
                "sha256": _sha256(path),
                "seed": seed,
                "global_step": int(checkpoint.get("global_step", -1)),
                "parameters": parameters,
            }
        )
    if len(code_fingerprints) != 1:
        raise ValueError("members do not share training-code provenance")
    if len(input_fingerprints) != 1:
        raise ValueError("members do not share training-input provenance")
    if len({member["parameters"] for member in members}) != 1:
        raise ValueError("members do not share parameter count")

    manifest = {
        "schema_version": 1,
        "type": "capmatched_output_ensemble",
        "name": args.name,
        "arch": args.expected_arch,
        "reduction": "mean",
        "members": members,
        "training_code_fingerprint": next(iter(code_fingerprints)),
        "training_input_fingerprint": next(iter(input_fingerprints)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
