#!/usr/bin/env python3
"""Interpolate the shipped ERA5 window with every released model.

    python examples/interpolate.py                 # all models, all interior hours
    python examples/interpolate.py --model weatherbridge-6h --tau 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import weatherbridge as wb


def latitude_weights(height: int) -> torch.Tensor:
    edges = torch.linspace(90.0, -90.0, height + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = torch.cos(torch.deg2rad(centres)).clamp_min(0.0)
    return (weights / weights.mean()).view(1, 1, -1, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None)
    parser.add_argument("--tau", type=int, action="append", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data = np.load(str(root / "data" / "sample_era5_2020070100.npz"))
    delta_t = int(data["delta_t_hours"])
    taus = args.tau or [int(t) for t in data["taus"]]
    x0 = torch.from_numpy(data["x0"].astype(np.float32)).unsqueeze(0)
    xT = torch.from_numpy(data["xT"].astype(np.float32)).unsqueeze(0)
    weights = latitude_weights(x0.shape[-2])

    names = args.model or [n for n in wb.list_models() if n.endswith("-6h")]
    print(f"anchors {data['anchor_valid_time']} +0 h / +{delta_t} h, device {args.device}\n")
    for name in names:
        model = wb.load_model(name, device=args.device)
        row = []
        for tau in taus:
            target = torch.from_numpy(
                data[f"target_tau{tau}"].astype(np.float32)
            ).unsqueeze(0)
            pred = wb.predict(model, x0, xT, tau).cpu()
            base = wb.linear_interpolation(x0, xT, tau, delta_t)
            rmse = float(torch.sqrt((weights * (pred - target) ** 2).mean()))
            ref = float(torch.sqrt((weights * (base - target) ** 2).mean()))
            row.append(f"tau={tau}: {rmse:.4f} ({100 * (ref - rmse) / ref:+.1f}%)")
        print(f"{wb.model_info(name)['display']:<18} " + "  ".join(row))
        del model
    print("\nRMSE is latitude-weighted over 24 normalised channels; the percentage "
          "is the reduction against linear interpolation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
