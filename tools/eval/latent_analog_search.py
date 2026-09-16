"""Latent nearest-neighbour analog search for a target atmospheric state.

Motivation for npj CAS: reanalysis embeddings usable for downstream ML.
Show that WeatherDCAE encoder latents preserve synoptic structure —
querying an ERA5 2021 target against a historical pool retrieves
physically-similar states (same season, similar circulation, similar
TC activity).

Approach:
  1. Encode the target state → mean-pool over spatial → 512-d embedding
     (or 128-d for lighter models).
  2. Encode every 6-hourly state from a pool year → embed same way.
  3. L2 distance between target and pool → rank, report top-k analogs.

Pool = ERA5 memmap (wb2_YYYY.bin/json). Target = specific hour in
another year memmap.

Output JSON:
  {
    "target": {"year": int, "hour_index": int, "iso": "..."},
    "pool_year": int,
    "n_pool": int,
    "top_k": [{"hour_index": int, "iso": "...", "l2_dist": float}, ...],
    "embedding_dim": int,
    "encode_time_sec": float,
  }
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]


def load_canonical_stds() -> np.ndarray:
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from paper_units import canonical_stds
    d = canonical_stds()
    return np.array([d[c] for c in CHANNELS_ORDER], dtype=np.float32)


def open_era5(era5_dir: Path, year: int):
    bin_p = era5_dir / f"wb2_{year}.bin"
    json_p = era5_dir / f"wb2_{year}.json"
    meta = json.loads(json_p.read_text())
    mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                   shape=tuple(meta["shape"]))
    return mm, meta


def load_wb_encoder(blob_path: str, device: str):
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    from examples._bare_loader import load_bare
    model = load_bare(blob_path, device)
    return model


def encode_state(model, x24: np.ndarray, stds_t, static_t,
                 device: str) -> np.ndarray:
    """x24: (24, H, W) physical. Returns mean-pooled embedding (C_latent,).

    The DC-AE encoder's ResBlocks modulate on a time embedding, so we
    must feed a temb. We use a fixed τ=0.5 embedding (analog structure
    is τ-agnostic — the same state regardless of interpolation fraction).
    Encoder input is cat(x0, xT, static); for a single state we duplicate
    the frame (x, x) which is the τ=0.5 self-consistent input.
    """
    x = torch.from_numpy(x24[None]).to(device) / stds_t
    concat = torch.cat([x, x, static_t], dim=1)  # (1, 51, H, W)
    # lat-crop/pad to match the encoder's expected input, mirroring the model.
    if hasattr(model, "_crop_lat"):
        concat = model._crop_lat(concat)
    tau = torch.tensor([0.5], device=device, dtype=torch.float32)
    t_emb = model.time_mlp(tau)
    with torch.no_grad():
        z = model.encoder(concat, t_emb)
    if not torch.is_tensor(z):
        z = z[0] if isinstance(z, (list, tuple)) else None
    if z is None:
        raise RuntimeError("encoder did not return a tensor")
    pooled = z.mean(dim=(0, 2, 3))
    return pooled.cpu().numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--target-year", type=int, required=True)
    ap.add_argument("--target-iso", required=True,
                    help="ISO datetime, e.g. 2021-08-29T06")
    ap.add_argument("--pool-year", type=int, required=True)
    ap.add_argument("--pool-stride-hours", type=int, default=6)
    ap.add_argument("--wb-blob", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stds = load_canonical_stds()
    era5_dir = Path(args.era5_dir)

    # Load target
    tgt_mm, tgt_meta = open_era5(era5_dir, args.target_year)
    year_start = np.datetime64(f"{args.target_year}-01-01T00", "h")
    target_dt = np.datetime64(args.target_iso, "h")
    tgt_h = int((target_dt - year_start) / np.timedelta64(1, "h"))
    print(f"[target] {args.target_iso} → hour {tgt_h} of {args.target_year}", flush=True)
    x_target = np.asarray(tgt_mm[tgt_h, :24])

    # Load model
    model = load_wb_encoder(args.wb_blob, args.device)
    print(f"[model] params={sum(p.numel() for p in model.parameters())/1e6:.1f}M",
          flush=True)

    stds_t = torch.from_numpy(stds).to(args.device).view(1, -1, 1, 1)
    root = Path(__file__).resolve().parents[2]
    static_full = torch.load(str(root / "data" / "static_features_0p5.pt"),
                              weights_only=False).float()
    if static_full.dim() == 3:
        static_full = static_full.unsqueeze(0)
    n_static = getattr(model, "n_static_features", 3)
    static_t = static_full[:, :n_static].to(args.device)

    # Encode target
    t0 = time.time()
    emb_target = encode_state(model, x_target, stds_t, static_t, args.device)
    print(f"[target] embedding dim={emb_target.shape[0]}, "
          f"encode={time.time()-t0:.2f}s", flush=True)

    # Iterate pool year
    pool_mm, pool_meta = open_era5(era5_dir, args.pool_year)
    pool_T = pool_meta["shape"][0]
    hour_ids = list(range(0, pool_T, args.pool_stride_hours))
    print(f"[pool] year {args.pool_year} T={pool_T} → sampling every "
          f"{args.pool_stride_hours}h → {len(hour_ids)} states", flush=True)

    dists = np.zeros(len(hour_ids), dtype=np.float32)
    t0 = time.time()
    for i, hi in enumerate(hour_ids):
        x = np.asarray(pool_mm[hi, :24])
        emb = encode_state(model, x, stds_t, static_t, args.device)
        dists[i] = np.linalg.norm(emb - emb_target)
        if (i + 1) % 60 == 0 or i == 0:
            print(f"[{i+1}/{len(hour_ids)}] t+{time.time()-t0:.1f}s "
                  f"h={hi} dist={dists[i]:.3f}", flush=True)
    print(f"[done] {time.time()-t0:.1f}s", flush=True)

    # Rank
    ranking = np.argsort(dists)[: args.top_k]
    pool_start = np.datetime64(f"{args.pool_year}-01-01T00", "h")
    top_k = []
    for r in ranking:
        h = hour_ids[int(r)]
        dt = pool_start + np.timedelta64(h, "h")
        top_k.append({
            "hour_index": int(h),
            "iso": str(np.datetime_as_string(dt, unit="h")),
            "l2_dist": float(dists[r]),
        })

    out = {
        "target": {"year": args.target_year, "hour_index": tgt_h,
                    "iso": args.target_iso},
        "pool_year": args.pool_year,
        "pool_stride_hours": args.pool_stride_hours,
        "n_pool": len(hour_ids),
        "embedding_dim": int(emb_target.shape[0]),
        "top_k": top_k,
        "encoder": args.wb_blob,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
