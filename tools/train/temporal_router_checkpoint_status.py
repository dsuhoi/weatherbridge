"""Fail-closed status check for a frozen temporal-router checkpoint."""
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


def verify(checkpoint_path: Path, required_taus: set[int]) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    hparams = checkpoint.get("hyper_parameters", {})
    if (
        hparams.get("arch") != "temporal_expert_router"
        or hparams.get("router_schema_version") != 2
        or hparams.get("router_selection_year") != 2020
        or hparams.get("router_ood_year_loaded") is not False
    ):
        raise ValueError("invalid temporal-router checkpoint metadata")

    route_path = Path(str(hparams.get("router_route_artifact", "")))
    if (
        not route_path.is_file()
        or file_sha256(route_path)
        != hparams.get("router_route_artifact_sha256")
    ):
        raise ValueError("stale temporal-router route artifact")
    route = json.loads(route_path.read_text())
    if (
        route.get("schema_version") != 2
        or route.get("selection_year") != 2020
        or route.get("ood_year_loaded") is not False
        or route.get("route_by_tau") != hparams.get("router_route_by_tau")
    ):
        raise ValueError("route/checkpoint metadata mismatch")

    recorded_route = {
        int(hour): str(name)
        for hour, name in hparams["router_route_by_tau"].items()
    }
    if set(recorded_route) != required_taus:
        raise ValueError("temporal-router target-hour set mismatch")
    experts = hparams.get("router_experts")
    if not isinstance(experts, dict) or set(recorded_route.values()) - set(
        experts
    ):
        raise ValueError("temporal-router expert set mismatch")
    state = checkpoint.get("state_dict", {})
    for name, metadata in experts.items():
        source_path = Path(str(metadata.get("checkpoint_path", "")))
        source_hash = metadata.get("checkpoint_sha256")
        if (
            not source_path.is_file()
            or not isinstance(source_hash, str)
            or file_sha256(source_path) != source_hash
        ):
            raise ValueError(f"{name}: stale source checkpoint")
        prefix = f"net.experts.{name}."
        if not any(key.startswith(prefix) for key in state):
            raise ValueError(f"{name}: packed state is missing")
    provenance = checkpoint.get("temporal_router_provenance", {})
    builder = Path(__file__).with_name("build_temporal_router_checkpoint.py")
    if provenance.get("builder_code_sha256") != file_sha256(builder):
        raise ValueError("temporal-router builder code hash is stale")
    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "route_by_tau": recorded_route,
        "experts": sorted(experts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--required-taus", default="1,2,3,4,5")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    report = verify(
        args.checkpoint,
        {
            int(value)
            for value in args.required_taus.split(",")
            if value.strip()
        },
    )
    if not args.quiet:
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
