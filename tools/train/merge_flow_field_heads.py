#!/usr/bin/env python3
"""Merge selected field-specific Flow head rows from a distilled checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import torch


FIELD_NAMES = ("Z1000", "Z925", "Z850", "Z700", "mslp")
FIELD_INDICES = (16, 17, 18, 19, 23)
ROW_KEYS = (
    "net.scale",
    "net.warp_gate",
    "net.blend_head.weight",
    "net.blend_head.bias",
    "net.res_coarse.weight",
    "net.res_coarse.bias",
    "net.res_fine.weight",
    "net.res_fine.bias",
    "net.spectral.pout.weight",
    "net.spectral.pout.bias",
)
HYDRO_KEYS = ("net.hydro.weight", "net.hydro.bias")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def merge_field_heads(
    control_path: Path,
    distilled_path: Path,
    output_path: Path,
    *,
    alpha: float,
    include_shared_trunk: bool = False,
) -> dict[str, object]:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0, 1]")
    control = torch.load(control_path, map_location="cpu", weights_only=False)
    distilled = torch.load(distilled_path, map_location="cpu", weights_only=False)
    control_state = control["state_dict"]
    distilled_state = distilled["state_dict"]
    if tuple(control_state) != tuple(distilled_state):
        raise ValueError("checkpoint state dictionaries differ")
    if control["hyper_parameters"].get("arch") != "flow_pp3":
        raise ValueError("control must use flow_pp3")
    if distilled["hyper_parameters"].get("arch") != "flow_pp3":
        raise ValueError("distilled checkpoint must use flow_pp3")

    merged_state = {key: value.clone() for key, value in control_state.items()}
    if include_shared_trunk:
        for key, control_value in control_state.items():
            distilled_value = distilled_state[key]
            if control_value.shape != distilled_value.shape:
                raise ValueError(f"checkpoint tensor shape differs: {key}")
            if control_value.is_floating_point() or control_value.is_complex():
                merged_state[key].copy_(torch.lerp(control_value, distilled_value, alpha))
    for key in ROW_KEYS:
        control_value = control_state[key]
        distilled_value = distilled_state[key]
        if control_value.shape != distilled_value.shape or control_value.size(0) != 24:
            raise ValueError(f"invalid field-row tensor: {key}")
        for index in range(24):
            if index in FIELD_INDICES:
                merged_state[key][index].copy_(
                    torch.lerp(control_value[index], distilled_value[index], alpha)
                )
            elif include_shared_trunk:
                merged_state[key][index].copy_(control_value[index])
    for key in HYDRO_KEYS:
        control_value = control_state[key]
        distilled_value = distilled_state[key]
        if control_value.shape != distilled_value.shape or control_value.size(0) != 5:
            raise ValueError(f"invalid hydro tensor: {key}")
        merged_state[key].copy_(torch.lerp(control_value, distilled_value, alpha))

    result = dict(control)
    result["state_dict"] = merged_state
    result["optimizer_states"] = []
    result["lr_schedulers"] = []
    metadata = {
        "schema_version": 1,
        "method": (
            "shared_trunk_target_head_linear_parameter_merge"
            if include_shared_trunk
            else "field_specific_linear_parameter_merge"
        ),
        "evaluation_only": True,
        "alpha": alpha,
        "include_shared_trunk": include_shared_trunk,
        "fields": list(FIELD_NAMES),
        "field_indices": list(FIELD_INDICES),
        "row_keys": list(ROW_KEYS),
        "hydro_keys": list(HYDRO_KEYS),
        "control": {"path": str(control_path.resolve()), "sha256": _sha256(control_path)},
        "distilled": {"path": str(distilled_path.resolve()), "sha256": _sha256(distilled_path)},
        "generator_sha256": _sha256(Path(__file__).resolve()),
        "torch_version": torch.__version__,
    }
    result["checkpoint_field_merge"] = metadata
    result.setdefault("hyper_parameters", {})["field_merge"] = metadata

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(result, temporary)
        os.chmod(temporary, 0o644)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--distilled", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--include-shared-trunk", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = merge_field_heads(
        args.control,
        args.distilled,
        args.output,
        alpha=args.alpha,
        include_shared_trunk=args.include_shared_trunk,
    )
    print(
        f"output={args.output} alpha={metadata['alpha']} "
        f"fields={','.join(metadata['fields'])}"
    )


if __name__ == "__main__":
    main()
