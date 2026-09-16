#!/usr/bin/env python3
"""Record what each released model scores on the shipped ERA5 sample.

The numbers this writes are what ``tests/test_reference_scores.py`` asserts,
so they pin the released weights to a reproducible behaviour: if a checkpoint
is ever swapped, re-converted or silently corrupted, the test fails.

Errors are latitude-weighted, matching the paper: a grid cell's area on the
sphere goes as the cosine of its latitude, so an unweighted mean would let the
poles dominate. Scores are reported against linear interpolation, which is the
baseline the paper measures every model against.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weatherbridge import linear_interpolation, list_models, load_model, predict  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
REFERENCE_SEED = 0


def latitude_weights(height: int) -> torch.Tensor:
    """cos(latitude) on the 0.5 degree cell-centre grid, normalised to mean one."""
    edges = torch.linspace(90.0, -90.0, height + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weights = torch.cos(torch.deg2rad(centres)).clamp_min(0.0)
    return (weights / weights.mean()).view(1, 1, -1, 1)


def weighted_rmse(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> float:
    return float(torch.sqrt((weights * (pred - target) ** 2).mean()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path,
                        default=REPO / "data" / "sample_era5_2020070100.npz")
    parser.add_argument("--out", type=Path,
                        default=REPO / "data" / "reference_scores.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    data = np.load(str(args.sample))
    delta_t = int(data["delta_t_hours"])
    taus = [int(t) for t in data["taus"]]
    x0 = torch.from_numpy(data["x0"].astype(np.float32)).unsqueeze(0)
    xT = torch.from_numpy(data["xT"].astype(np.float32)).unsqueeze(0)
    targets = {
        tau: torch.from_numpy(data[f"target_tau{tau}"].astype(np.float32)).unsqueeze(0)
        for tau in taus
    }
    weights = latitude_weights(x0.shape[-2])

    scores: dict[str, dict] = {
        "linear": {
            str(tau): weighted_rmse(
                linear_interpolation(x0, xT, tau, delta_t), targets[tau], weights
            )
            for tau in taus
        }
    }
    print(f"{'model':<22}" + "".join(f"  tau={t}" for t in taus) + "     mean   vs linear")
    linear_mean = float(np.mean(list(scores["linear"].values())))
    print(f"{'linear':<22}" + "".join(f"  {scores['linear'][str(t)]:.4f}" for t in taus)
          + f"   {linear_mean:.4f}        --")

    for name in list_models():
        info_taus = taus if int(delta_t) == 6 else taus
        model = load_model(name, device=args.device)
        if abs(model._wb_delta_t - delta_t) > 1e-6:
            print(f"{name:<22}  skipped: trained for a "
                  f"{model._wb_delta_t:.0f} h anchor gap, sample is {delta_t} h")
            continue
        per_tau = {}
        for tau in info_taus:
            # S-DYff samples noise inside its forward; seeding makes its output
            # reproducible without pretending the model is deterministic.
            torch.manual_seed(REFERENCE_SEED)
            pred = predict(model, x0, xT, tau).cpu()
            per_tau[str(tau)] = weighted_rmse(pred, targets[tau], weights)
        scores[name] = per_tau
        mean = float(np.mean(list(per_tau.values())))
        gain = 100.0 * (linear_mean - mean) / linear_mean
        print(f"{name:<22}" + "".join(f"  {per_tau[str(t)]:.4f}" for t in info_taus)
              + f"   {mean:.4f}   {gain:+6.2f}%")
        del model

    payload = {
        "schema": 1,
        "sample": args.sample.name,
        "anchor_valid_time": str(data["anchor_valid_time"]),
        "delta_t_hours": delta_t,
        "taus": taus,
        "metric": "latitude-weighted RMSE over 24 normalised channels",
        "reference_seed": REFERENCE_SEED,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
        },
        "scores": scores,
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
