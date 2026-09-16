#!/usr/bin/env python3
"""Relabel a checkpoint only after strict state-compatible control loading."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import torch

from tools.train.train_capacity_matched_6h import (
    build_net,
    canonical_arch_name,
)


SUPPORTED_CONVERSIONS = {
    (
        "upr_implicit_global_14m",
        "upr_endpoint_implicit_global_14m",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _net_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("state_dict", checkpoint)
    result = {
        key.removeprefix("net."): value
        for key, value in state.items()
        if key.startswith("net.")
    }
    if not result:
        raise ValueError("checkpoint contains no net.* state")
    return result


def convert_checkpoint(
    source: Path,
    output: Path,
    *,
    expected_source_arch: str,
    target_arch: str,
) -> dict[str, Any]:
    conversion = (expected_source_arch, target_arch)
    if conversion not in SUPPORTED_CONVERSIONS:
        raise ValueError(f"unsupported architecture conversion {conversion}")
    if source.resolve() == output.resolve():
        raise ValueError("source and output checkpoint must differ")
    source_sha256 = _sha256(source)
    if output.exists():
        existing = torch.load(
            output,
            map_location="cpu",
            weights_only=False,
        )
        provenance = (
            existing.get("architecture_conversion")
            if isinstance(existing, dict)
            else None
        )
        if (
            not isinstance(provenance, dict)
            or provenance.get("source_checkpoint_sha256") != source_sha256
            or provenance.get("source_arch") != expected_source_arch
            or provenance.get("target_arch") != target_arch
            or provenance.get("weights_unchanged") is not True
        ):
            raise FileExistsError(
                f"refusing to overwrite incompatible {output}"
            )
        return {
            **provenance,
            "output_checkpoint": str(output.resolve()),
            "output_checkpoint_sha256": _sha256(output),
            "reused": True,
        }

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise ValueError("checkpoint lacks hyper_parameters")
    source_arch = str(hparams.get("arch", ""))
    if source_arch != expected_source_arch:
        raise ValueError(
            f"source architecture {source_arch!r} != "
            f"{expected_source_arch!r}"
        )

    state = _net_state(checkpoint)
    source_model, _, _ = build_net(source_arch, "")
    target_model, _, _ = build_net(target_arch, "")
    source_model.load_state_dict(state, strict=True)
    target_model.load_state_dict(state, strict=True)
    if tuple(source_model.state_dict()) != tuple(target_model.state_dict()):
        raise ValueError("source and target state-key order differs")
    if sum(p.numel() for p in source_model.parameters()) != sum(
        p.numel() for p in target_model.parameters()
    ):
        raise ValueError("source and target parameter counts differ")

    code_sha256 = _sha256(Path(__file__))
    provenance = {
        "kind": "zero_shot_state_compatible_control",
        "source_arch": source_arch,
        "target_arch": target_arch,
        "source_checkpoint": str(source.resolve()),
        "source_checkpoint_sha256": source_sha256,
        "conversion_code_sha256": code_sha256,
        "weights_unchanged": True,
    }
    hparams = dict(hparams)
    hparams["arch"] = target_arch
    hparams["canonical_arch_name"] = canonical_arch_name(target_arch)
    hparams["trained_arch"] = source_arch
    hparams["architecture_conversion"] = provenance
    checkpoint["hyper_parameters"] = hparams
    checkpoint["architecture_conversion"] = provenance

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(output)
    return {
        **provenance,
        "output_checkpoint": str(output.resolve()),
        "output_checkpoint_sha256": _sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-source-arch", required=True)
    parser.add_argument("--target-arch", required=True)
    args = parser.parse_args()
    report = convert_checkpoint(
        args.source,
        args.output,
        expected_source_arch=args.expected_source_arch,
        target_arch=args.target_arch,
    )
    print(report)


if __name__ == "__main__":
    main()
