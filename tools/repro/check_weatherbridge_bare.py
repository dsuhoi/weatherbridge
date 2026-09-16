#!/usr/bin/env python3
"""Validate the deployable WeatherBridge bare artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    parser.add_argument("--skip-forward", action="store_true")
    args = parser.parse_args()

    import torch

    from weather_time_interp.model.weatherbridge_flow_model import WeatherBridgeModel
    from weather_time_interp.normalization import file_provenance

    artifact = Path(args.artifact)
    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    if blob.get("arch") != "WeatherBridgeModel":
        raise ValueError(f"unexpected bare architecture: {blob.get('arch')}")
    if blob.get("kwargs", {}).get("anchor_detail_bypass", False):
        raise ValueError("deployable artifact unexpectedly enables DetailBypass")
    model = WeatherBridgeModel(**blob["kwargs"])
    model.load_state_dict(blob["state_dict"], strict=True)
    model = model.eval().to(args.device)
    if model.detail_gain_head is not None:
        raise ValueError("deployable model unexpectedly has a detail gain head")

    output_shape = None
    if not args.skip_forward:
        x0 = torch.randn(1, 24, args.height, args.width, device=args.device)
        xT = torch.randn_like(x0)
        static = torch.randn(1, 3, args.height, args.width, device=args.device)
        with torch.inference_mode():
            output = model(
                x0,
                xT,
                torch.tensor([0.5], device=args.device),
                torch.tensor([6.0], device=args.device),
                static=static,
            )[0]
        if output.shape != x0.shape or not torch.isfinite(output).all():
            raise RuntimeError(f"invalid forward output: {tuple(output.shape)}")
        output_shape = list(output.shape)
    print(
        json.dumps(
            {
                "artifact": file_provenance(artifact),
                "architecture": blob["arch"],
                "paper_name": "WeatherBridge",
                "anchor_detail_bypass": False,
                "parameter_count": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "output_shape": output_shape,
                "finite": None if args.skip_forward else True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
