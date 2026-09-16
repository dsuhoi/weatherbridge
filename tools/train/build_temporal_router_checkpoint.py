"""Pack frozen interpolation experts and a 2020-selected route for inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source(
    name: str,
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    path = Path(str(metadata.get("checkpoint_path", "")))
    expected_hash = metadata.get("checkpoint_sha256")
    if (
        not path.is_file()
        or not isinstance(expected_hash, str)
        or file_sha256(path) != expected_hash
    ):
        raise ValueError(f"{name}: stale route checkpoint provenance")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    hparams = dict(checkpoint.get("hyper_parameters", {}))
    arch = hparams.get("arch")
    if not isinstance(arch, str) or not arch:
        raise ValueError(f"{name}: source checkpoint lacks architecture")
    if arch == "temporal_expert_router":
        raise ValueError("nested temporal routers are not supported")
    recorded_arch = metadata.get("checkpoint_arch")
    if recorded_arch is not None and recorded_arch != arch:
        raise ValueError(f"{name}: route/source architecture mismatch")
    state = checkpoint.get("state_dict", checkpoint)
    net_state = {
        key.removeprefix("net."): value.detach().cpu()
        for key, value in state.items()
        if key.startswith("net.")
    }
    if not net_state:
        raise ValueError(f"{name}: source checkpoint has no net state")
    return {
        "arch": arch,
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": expected_hash,
        "delta_t": float(hparams.get("delta_t", 6.0)),
        "global_step": int(checkpoint.get("global_step", 0)),
        "epoch": int(checkpoint.get("epoch", -1)),
    }, net_state


def build_checkpoint(route_path: Path) -> dict[str, Any]:
    route = json.loads(route_path.read_text())
    if (
        route.get("schema_version") != 2
        or route.get("evidence_level")
        != "frozen_full_year_2020_tau_router"
        or route.get("selection_year") != 2020
        or route.get("ood_year_loaded") is not False
    ):
        raise ValueError("route artifact is not a frozen 2020 selection")
    selection_source = (
        Path(__file__).resolve().parents[1]
        / "eval"
        / "select_temporal_expert_route.py"
    )
    if route.get("selection_code_sha256") != file_sha256(selection_source):
        raise ValueError("route selection code hash is stale")

    route_by_tau = route.get("route_by_tau")
    expert_metadata = route.get("experts")
    if not isinstance(route_by_tau, dict) or not isinstance(
        expert_metadata,
        dict,
    ):
        raise TypeError("route artifact lacks experts or route")
    if set(route_by_tau.values()) - set(expert_metadata):
        raise ValueError("route references an unknown expert")
    routed_experts = set(route_by_tau.values())

    packed_state: dict[str, torch.Tensor] = {}
    packed_experts: dict[str, dict[str, Any]] = {}
    delta_values: set[float] = set()
    steps: list[int] = []
    epochs: list[int] = []
    for name in sorted(routed_experts):
        metadata = expert_metadata[name]
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("expert names must be valid identifiers")
        if not isinstance(metadata, dict):
            raise TypeError(f"{name}: invalid expert metadata")
        source, net_state = _load_source(name, metadata)
        packed_experts[name] = {
            key: source[key]
            for key in (
                "arch",
                "checkpoint_path",
                "checkpoint_sha256",
            )
        }
        delta_values.add(source["delta_t"])
        steps.append(source["global_step"])
        epochs.append(source["epoch"])
        for key, tensor in net_state.items():
            packed_state[f"net.experts.{name}.{key}"] = tensor
    if len(delta_values) != 1:
        raise ValueError("router experts use different interpolation windows")
    delta_t = next(iter(delta_values))
    routed_hours = {int(hour) for hour in route_by_tau}
    if any(hour <= 0 or hour >= delta_t for hour in routed_hours):
        raise ValueError("route contains a target outside the anchor window")

    return {
        "state_dict": packed_state,
        "hyper_parameters": {
            "arch": "temporal_expert_router",
            "delta_t": delta_t,
            "router_schema_version": 2,
            "router_route_by_tau": {
                str(int(hour)): str(name)
                for hour, name in route_by_tau.items()
            },
            "router_experts": packed_experts,
            "router_route_artifact": str(route_path.resolve()),
            "router_route_artifact_sha256": file_sha256(route_path),
            "router_selection_year": 2020,
            "router_ood_year_loaded": False,
        },
        "global_step": min(steps),
        "epoch": min(epochs),
        "temporal_router_provenance": {
            "route_artifact": str(route_path.resolve()),
            "route_artifact_sha256": file_sha256(route_path),
            "builder_code_sha256": file_sha256(Path(__file__)),
            "experts": packed_experts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = build_checkpoint(args.route)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": file_sha256(args.output),
                "route": checkpoint["hyper_parameters"][
                    "router_route_by_tau"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
