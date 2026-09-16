#!/usr/bin/env python3
"""Load one paper checkpoint and run a full-grid inference smoke test."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch

from tools.eval.batch_eval_12h_memmap import _load_model_safe
from weather_time_interp.memmap_dataset import ERA5MemmapDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--horizon", type=int, choices=(6, 12), required=True)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--width", type=int, default=720)
    args = parser.parse_args()

    device = torch.device(args.device)
    dataset = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[2020],
        max_tau_hours=args.horizon,
        samples_per_date=1,
        train=False,
        eval_hours=[args.horizon // 2],
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
    )
    model, model_type = _load_model_safe(
        args.checkpoint,
        device,
        dataset.channel_groups,
        static_path=args.static_path,
    )

    input_channels = int(
        getattr(
            model,
            "in_channels",
            getattr(getattr(model, "net", None), "in_channels", 24),
        )
    )
    x0 = torch.zeros(
        1, input_channels, args.height, args.width, device=device
    )
    xT = torch.ones(
        1, input_channels, args.height, args.width, device=device
    ) * 0.01
    static = torch.load(
        Path(args.static_path),
        map_location=device,
        weights_only=False,
    )[:3, : args.height, : args.width].unsqueeze(0)
    tau = torch.tensor([0.5], device=device)
    cond = torch.tensor([float(args.horizon)], device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(x0, xT, tau, cond, static=static)
    if isinstance(output, tuple):
        output = output[0]
    print(
        f"name={args.name} model_type={model_type} "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"input_channels={input_channels} "
        f"shape={tuple(output.shape)} finite={bool(torch.isfinite(output).all())} "
        f"mean={output.float().mean().item():.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
