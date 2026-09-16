#!/usr/bin/env python3
"""Build static features tensor for our 181×360 ERA5 grid.

Sources from ladcast (https://github.com/tonyzyl/ladcast/tree/master/ladcast/static):
- 240x121_land_sea_mask.pt  →  resize to 181×360, [0..1] mask
- 240x121_orography.pt       →  4-channel (elevation + sub-grid stats),
                                 use channel 0 (raw orography) + normalize

Output: data/static_features.pt
    shape (3, 181, 360)
    channels:
        0: land_sea_mask           (raw [0,1])
        1: orography_normalized    ((x - mean) / std)
        2: latitude_cos            (cos(lat * π/180), [-1, 1])

Usage:
    python tools/build_static_features.py \
        --src /tmp/ladcast/ladcast/static \
        --out data/static_features.pt
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/tmp/ladcast/ladcast/static")
    p.add_argument("--out", default="/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features.pt")
    p.add_argument("--lat", type=int, default=181)
    p.add_argument("--lon", type=int, default=360)
    args = p.parse_args()

    src = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Land-sea mask (1, 121, 240) in [0, 1]
    lsm_raw = torch.load(src / "240x121_land_sea_mask.pt", weights_only=False)
    if lsm_raw.dim() == 2:
        lsm_raw = lsm_raw.unsqueeze(0)
    lsm = F.interpolate(
        lsm_raw.unsqueeze(0).float(),
        size=(args.lat, args.lon),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)
    print(f"lsm: shape={tuple(lsm.shape)} range=[{lsm.min():.3f}, {lsm.max():.3f}]")

    # 2. Orography (4, 121, 240) — first channel is raw geopotential at surface
    oro_raw = torch.load(src / "240x121_orography.pt", weights_only=False)
    print(f"oro_raw shape: {tuple(oro_raw.shape)}")
    # Use channel 0 (elevation/geopotential at surface)
    oro = oro_raw[0:1].unsqueeze(0).float()
    oro = F.interpolate(oro, size=(args.lat, args.lon), mode="bilinear", align_corners=False).squeeze(0)
    # Normalize: paper uses mean=3790.77, std=8342.16 for geopotential_at_surface (1979-2018)
    # but our orography values are different (max ~673, much smaller). Compute on actual data.
    oro_mean = float(oro.mean())
    oro_std = float(oro.std()) + 1e-6
    oro_norm = (oro - oro_mean) / oro_std
    print(f"oro: shape={tuple(oro_norm.shape)} pre-norm range=[{oro.min():.2f}, {oro.max():.2f}] mean={oro_mean:.2f} std={oro_std:.2f}")

    # 3. Latitude cosine map (geographic info)
    lats_deg = torch.linspace(90.0, -90.0, args.lat)  # ERA5 convention
    lat_cos = torch.cos(lats_deg * math.pi / 180.0).view(1, args.lat, 1).expand(1, args.lat, args.lon).contiguous()
    print(f"lat_cos: shape={tuple(lat_cos.shape)} range=[{lat_cos.min():.3f}, {lat_cos.max():.3f}]")

    # Stack: (3, lat, lon)
    static = torch.cat([lsm, oro_norm, lat_cos], dim=0)
    print(f"static: shape={tuple(static.shape)} dtype={static.dtype}")

    torch.save(static, str(out_path))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
