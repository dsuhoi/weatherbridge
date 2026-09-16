"""Wind capacity-factor bias induced by 6h→1h temporal interpolation.

Motivation (npj CAS narrative): hourly wind is the input to any
wind-farm operations model — capacity factor CF = mean_t f(V(t))
where f is a manufacturer power curve. 6h-linear interpolation
under-samples the temporal wind variance (esp. in complex terrain
and coastal sea-breeze regimes), biasing CF low. A learned temporal
interpolator that preserves sub-6h variance should recover CF closer
to the ERA5 hourly truth.

Sites (single 0.5° gridcell each, coordinates ≈ real offshore/onshore
wind farm locations):
  * hornsea_uk         (53.9N, 1.8E)     — North Sea offshore
  * horns_rev_dk       (55.5N, 7.9E)     — Danish offshore
  * altamont_us        (37.7N, -121.6E → 238.4)  — California onshore, complex terrain
  * gansu_cn           (40.5N, 96.0E)    — Gobi onshore, monsoon-modulated
  * cabo_verde_offshore (15.5N, -22.5E → 337.5)  — trade winds

Power curve: reference Vestas V90 2 MW, cut-in 3.5 m/s, rated 15 m/s,
cut-out 25 m/s. Manufacturer curve linearised for reproducibility.

Output:
  metrics/downstream_wind_cf/<model_name>.json:
    per_site: {
      site_name: {
        cf_true, cf_model, cf_linear, cf_anchor_only,
        cf_bias_model, cf_bias_linear, cf_bias_anchor,
        rmse_speed_model, rmse_speed_linear,
      }
    }
    aggregate: {
      abs_cf_bias_mean_model, abs_cf_bias_mean_linear,
      site_count, hours_used
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from weather_time_interp.grid import (
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)

try:
    import torch
except ImportError:
    torch = None

CHANNELS_ORDER_24 = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
U10_IDX = CHANNELS_ORDER_24.index("u10")
V10_IDX = CHANNELS_ORDER_24.index("v10")

SITES = {
    "hornsea_uk":          {"lat": 53.9, "lon":   1.8},
    "horns_rev_dk":        {"lat": 55.5, "lon":   7.9},
    "altamont_us":         {"lat": 37.7, "lon": 238.4},  # 360-121.6
    "gansu_cn":            {"lat": 40.5, "lon":  96.0},
    "cabo_verde_offshore": {"lat": 15.5, "lon": 337.5},  # 360-22.5
}

def build_grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the coordinates encoded by the canonical WB2 memmap."""
    return wb2_block_average_latitudes(H), wb2_block_average_longitudes(W)


def nearest_ij(lat_arr: np.ndarray, lon_arr: np.ndarray,
                lat: float, lon: float) -> tuple[int, int]:
    return int(np.argmin(np.abs(lat_arr - lat))), int(np.argmin(np.abs(lon_arr - lon)))


def power_curve(v: np.ndarray, cut_in: float = 3.5, rated: float = 15.0,
                 cut_out: float = 25.0) -> np.ndarray:
    """Return normalised power ∈ [0,1] for wind speed v (m/s) — piecewise
    linear on cube-root of speed between cut-in and rated. Below cut-in
    or above cut-out → 0. Between rated and cut-out → 1.
    """
    v = np.asarray(v, dtype=np.float32)
    p = np.zeros_like(v)
    ramp = (v >= cut_in) & (v < rated)
    plateau = (v >= rated) & (v <= cut_out)
    p[ramp] = ((v[ramp] - cut_in) / (rated - cut_in)) ** 3
    p[plateau] = 1.0
    return p


def load_canonical_stds() -> np.ndarray:
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from paper_units import canonical_stds
    d = canonical_stds()
    return np.array([d[c] for c in CHANNELS_ORDER_24], dtype=np.float32)


def open_era5_year(era5_dir: Path, year: int):
    bin_p = era5_dir / f"wb2_{year}.bin"
    json_p = era5_dir / f"wb2_{year}.json"
    meta = json.loads(json_p.read_text())
    mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                   shape=tuple(meta["shape"]))
    return mm, meta


def linear_frames(a: np.ndarray, b: np.ndarray, taus: list[int], dt: int = 6):
    return [(1 - t / dt) * a + (t / dt) * b for t in taus]


def load_static_for(model, device: str):
    n_static = getattr(model, "n_static_features", 0)
    if n_static <= 0:
        return None
    root = Path(__file__).resolve().parents[2]
    s = torch.load(str(root / "data" / "static_features_0p5.pt"),
                    weights_only=False).float()
    if s.dim() == 3:
        s = s.unsqueeze(0)
    return s[:, :n_static].to(device)


def model_frames(model, a: np.ndarray, b: np.ndarray, taus: list[int],
                  stds: np.ndarray, device: str, dt: int = 6,
                  static: "torch.Tensor | None" = None):
    stds_t = torch.from_numpy(stds).to(device).view(1, -1, 1, 1)
    a_t = torch.from_numpy(a[None]).to(device) / stds_t
    b_t = torch.from_numpy(b[None]).to(device) / stds_t
    is_atmvfi = type(model).__name__ == "PixelAttentionVFI"
    cond = torch.tensor([float(dt)], device=device, dtype=torch.float32)
    kwargs = {"static": static} if static is not None else {}
    out = []
    with torch.no_grad():
        for t in taus:
            tt = torch.tensor([[t / dt]], device=device, dtype=torch.float32)
            if is_atmvfi:
                f = model.net(a_t, b_t, tt)
            else:
                f = model(a_t, b_t, tt, cond, **kwargs)
            if isinstance(f, tuple):
                f = f[0]
            out.append((f * stds_t).cpu().numpy()[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--model-blob", required=True,
                    help="'linear' / 'anchor-only' / *.pt bare blob")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    stds = load_canonical_stds()
    mm, meta = open_era5_year(Path(args.era5_dir), args.year)
    T, _, H, W = meta["shape"]
    lat_arr, lon_arr = build_grid(H, W)

    is_linear = args.model_blob == "linear"
    is_anchor = args.model_blob == "anchor-only"
    model = None
    if not (is_linear or is_anchor):
        if torch is None:
            raise RuntimeError("torch required for bare-blob model")
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from examples._bare_loader import load_bare
        model = load_bare(args.model_blob, args.device)
        static_t = load_static_for(model, args.device)

    site_ij = {s: nearest_ij(lat_arr, lon_arr, **c) for s, c in SITES.items()}
    print(f"[sites] {site_ij}", flush=True)

    # Extract truth u10/v10 for full year at all sites (cheap: T × 2 × 5)
    truth_u = {s: np.asarray(mm[:T, U10_IDX, i, j]) for s, (i, j) in site_ij.items()}
    truth_v = {s: np.asarray(mm[:T, V10_IDX, i, j]) for s, (i, j) in site_ij.items()}

    # Build reconstructed hourly u,v at each site
    recon_u = {s: v.copy() for s, v in truth_u.items()}
    recon_v = {s: v.copy() for s, v in truth_v.items()}
    n_pairs = 0
    for h_start in range(0, T - 6, 6):
        # For model: need full field (24,H,W) at both anchors
        if not (is_linear or is_anchor):
            a_full = np.asarray(mm[h_start, :24])
            b_full = np.asarray(mm[h_start + 6, :24])
        taus = [1, 2, 3, 4, 5]

        if is_anchor:
            # copy anchor forward
            for t in taus:
                for s, (i, j) in site_ij.items():
                    recon_u[s][h_start + t] = truth_u[s][h_start]
                    recon_v[s][h_start + t] = truth_v[s][h_start]
        elif is_linear:
            for t in taus:
                w = t / 6.0
                for s, (i, j) in site_ij.items():
                    recon_u[s][h_start + t] = (1 - w) * truth_u[s][h_start] + w * truth_u[s][h_start + 6]
                    recon_v[s][h_start + t] = (1 - w) * truth_v[s][h_start] + w * truth_v[s][h_start + 6]
        else:
            outs = model_frames(model, a_full, b_full, taus, stds, args.device,
                                 static=static_t)
            for i_tau, t in enumerate(taus):
                f = outs[i_tau]
                for s, (i, j) in site_ij.items():
                    recon_u[s][h_start + t] = f[U10_IDX, i, j]
                    recon_v[s][h_start + t] = f[V10_IDX, i, j]
        n_pairs += 1

    # Compute CF at each site
    per_site = {}
    cf_bias_list = []
    for s in SITES:
        v_true = np.sqrt(truth_u[s] ** 2 + truth_v[s] ** 2)
        v_recon = np.sqrt(recon_u[s] ** 2 + recon_v[s] ** 2)
        cf_true = float(power_curve(v_true).mean())
        cf_recon = float(power_curve(v_recon).mean())
        rmse_speed = float(np.sqrt(((v_true - v_recon) ** 2).mean()))
        per_site[s] = {
            "cf_true": cf_true,
            "cf_recon": cf_recon,
            "cf_bias": cf_recon - cf_true,
            "cf_bias_rel_pct": (cf_recon - cf_true) / max(cf_true, 1e-6) * 100.0,
            "rmse_speed_ms": rmse_speed,
            "mean_speed_true_ms": float(v_true.mean()),
        }
        cf_bias_list.append(abs(cf_recon - cf_true))

    out = {
        "model_name": args.model_name,
        "year": args.year,
        "sites": SITES,
        "n_pairs": n_pairs,
        "per_site": per_site,
        "aggregate": {
            "mean_abs_cf_bias": float(np.mean(cf_bias_list)),
            "site_count": len(SITES),
            "hours_used": T,
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
