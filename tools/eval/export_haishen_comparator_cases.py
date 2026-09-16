#!/usr/bin/env python3
"""Export matched-model Haishen panels for supplementary case studies."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from tools.eval.batch_eval_memmap import load_model_safe
from tools.eval.export_weatherbridge_cases import _write_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    parser.add_argument("--static-path", default="data/static_features_0p5.pt")
    parser.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    parser.add_argument(
        "--surface-stats-path",
        default="data/surface_stats_0p5.json",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device(args.device)
    model, model_type = load_model_safe(str(checkpoint), device, {})
    _write_case(
        model=model,
        model_label=args.model_label,
        model_type=model_type,
        checkpoint=str(checkpoint),
        memmap_dir=args.memmap_dir,
        static_path=args.static_path,
        stats_path=args.stats_path,
        surface_stats_path=args.surface_stats_path,
        year=2020,
        init_time=datetime(2020, 9, 7, 0),
        lat=np.arange(54.875, 19.874, -0.5, dtype=np.float32),
        lon=np.arange(100.125, 145.126, 0.5, dtype=np.float32),
        channels={"mslp": 23},
        output_dir=Path(args.output_root) / "typhoon_haishen",
        device=device,
    )


if __name__ == "__main__":
    main()
