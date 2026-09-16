#!/usr/bin/env python3
"""Build the Figure 5 high-frequency preservation summary from SH caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


MODELS = {
    "weatherbridge_pp3": "WeatherBridge",
    "weatherdcae_14m": "WeatherDCAE-14M",
    "atm_vfi_v2_135only": "PixelAttn-VFI",
    "fuxi_24ch_6yr_ep8": "SwinV2",
    "modafno_24ch_6yr_ep8": "ModAFNO",
    "sdyff_24ch_6yr_ep8": "S-DYff",
    "bilinear": "Linear Interp.",
}
CHANNELS = ("t2m", "mslp", "u10", "v10", "T850")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz-dir",
        default="metrics/sh_spectra_6h_unified_v3",
    )
    parser.add_argument("--taus", default="2,3")
    parser.add_argument("--ell-min", type=int, default=90)
    parser.add_argument(
        "--out",
        default="metrics/sh_spectra_6h_unified_v3/paper_hf_summary.json",
    )
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir)
    taus = sorted({int(value) for value in args.taus.split(",")})
    payload = {
        "metric": "sum(pred_El[ell>=ell_min]) / sum(gt_El[ell>=ell_min])",
        "ell_min": args.ell_min,
        "taus": taus,
        "channels": list(CHANNELS),
        "models": {},
    }

    for stem, display in MODELS.items():
        per_tau = []
        for tau in taus:
            path = npz_dir / f"{stem}_tau{tau}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            data = np.load(path, allow_pickle=True)
            channel_names = [str(name) for name in data["channel_names"]]
            mask = data["ell"] >= args.ell_min
            values = {}
            for channel in CHANNELS:
                index = channel_names.index(channel)
                numerator = float(data["pred_El"][index, mask].sum())
                denominator = float(data["gt_El"][index, mask].sum())
                values[channel] = numerator / max(denominator, 1e-30)
            per_tau.append(values)

        channel_mean = {
            channel: float(np.mean([row[channel] for row in per_tau]))
            for channel in CHANNELS
        }
        payload["models"][display] = {
            "stem": stem,
            "per_tau": {
                str(tau): per_tau[index] for index, tau in enumerate(taus)
            },
            "mean_over_tau": channel_mean,
            "mean_over_tau_and_channels": float(
                np.mean(list(channel_mean.values()))
            ),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
