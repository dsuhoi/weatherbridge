"""Physical-consistency checks for temporal interpolation output.

Two cheap dynamical-balance residuals that any physically-plausible
mid-latitude atmospheric state should approximately satisfy. If a
learned interpolator produces a state that violates them by more than
the truth-baseline residual, that's a red flag for reviewers of a
climate-science submission.

  * Geostrophic residual
        u_g = -(1/f) · ∂Z/∂y,   v_g = (1/f) · ∂Z/∂x
    Ageostrophic magnitude |v - v_g| at 850 and 700 hPa mid-latitudes.
    Truth ERA5 gives the baseline scale; a model that doubles this
    residual has drifted off the geostrophic manifold.

  * Hydrostatic residual
        ∂Z/∂(ln p) ≈ -R T ⇒ ΔZ / Δ(ln p) ≈ -R T_mean
    For our 4 PL levels {1000, 925, 850, 700} we check the residual
    between adjacent-layer geopotential differences and the hypsometric
    estimate from layer-mean virtual temperature. Virtual temperature uses
    the standard first-order moist-air correction.

Metric shape:
  per_tau[str(τ)] -> {
    model_name: {
      "ageo_rmse_ms_850": float,
      "ageo_rmse_ms_700": float,
      "hydrostatic_rmse_m2s2": float,
      "hydrostatic_rmse_m2s2_truth": float,
      "ageo_ratio_850": float,   # model / truth ratio
      "ageo_ratio_700": float,
      "hydrostatic_ratio": float,
      "n_pairs": int,
    }
  }

Reads:
  * ERA5 27-ch memmap (preprocess_0p5_to_memmap contract)
  * canonical per-channel mean/std used by the training dataset

Baselines "linear" and "anchor-only" reuse the shared plumbing from
eval_wind_capacity_factor / eval_diurnal_amplitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from weather_time_interp.grid import (
    latitude_strip_weights,
    wb2_block_average_latitudes,
    wb2_block_average_longitudes,
)
from weather_time_interp.normalization import (
    PAPER_CHANNELS_24,
    file_provenance,
    load_channel_stats,
)
from tools.train.training_protocol import memmap_dataset_provenance

try:
    import torch
except ImportError:
    torch = None

IDX = {c: i for i, c in enumerate(PAPER_CHANNELS_24)}
LEVELS = [1000, 925, 850, 700]
R_DRY = 287.05          # J / (kg K)
EARTH_RADIUS_KM = 6371.0

# Mid-latitude bands (skip |lat|<15° where f -> 0)
MIDLAT_ABS_MIN = 15.0
MIDLAT_ABS_MAX = 70.0


def build_grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the coordinates encoded by the canonical WB2 memmap."""
    return wb2_block_average_latitudes(H), wb2_block_average_longitudes(W)


def open_era5(era5_dir: Path, year: int):
    bin_p = era5_dir / f"wb2_{year}.bin"
    json_p = era5_dir / f"wb2_{year}.json"
    meta = json.loads(json_p.read_text())
    mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                   shape=tuple(meta["shape"]))
    return mm, meta


def coriolis(lat_deg: np.ndarray) -> np.ndarray:
    """Coriolis parameter f = 2Ω sin(φ), Ω = 7.2921e-5 rad/s."""
    return 2.0 * 7.2921e-5 * np.sin(np.deg2rad(lat_deg))


def geostrophic_wind(Z: np.ndarray, lat_arr: np.ndarray, lon_arr: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Central-diff geostrophic (u_g, v_g) from geopotential Z (m²/s²).

    Z shape (H, W), lat_arr (H,), lon_arr (W,) in degrees.
    Returns u_g, v_g (H, W). At |lat|<1° f≈0 → set NaN to skip.
    """
    H, W = Z.shape
    f = coriolis(lat_arr)  # (H,)
    dphi_deg = float(lat_arr[1] - lat_arr[0])  # negative (90 → -90)
    dy_m = np.deg2rad(dphi_deg) * EARTH_RADIUS_KM * 1000.0  # negative
    dx_deg = float(lon_arr[1] - lon_arr[0])
    dx_base = np.deg2rad(dx_deg) * EARTH_RADIUS_KM * 1000.0
    dx_m = dx_base * np.cos(np.deg2rad(lat_arr))[:, None]  # (H, 1)

    dZ_dy = (np.roll(Z, -1, axis=0) - np.roll(Z, 1, axis=0)) / (2.0 * dy_m)
    dZ_dx = (np.roll(Z, -1, axis=1) - np.roll(Z, 1, axis=1)) / (2.0 * dx_m)

    f_bcast = f[:, None]  # (H,1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u_g = -dZ_dy / f_bcast
        v_g = dZ_dx / f_bcast
    mask = np.abs(lat_arr) < 1.0
    u_g[mask, :] = np.nan
    v_g[mask, :] = np.nan
    # boundary rows are wrap-around garbage — mask
    u_g[[0, -1], :] = np.nan
    v_g[[0, -1], :] = np.nan
    return u_g, v_g


def midlat_mask(lat_arr: np.ndarray, W: int) -> np.ndarray:
    row = (np.abs(lat_arr) >= MIDLAT_ABS_MIN) & (np.abs(lat_arr) <= MIDLAT_ABS_MAX)
    return np.broadcast_to(row[:, None], (lat_arr.shape[0], W))


def ageo_rmse(u: np.ndarray, v: np.ndarray, Z: np.ndarray,
              lat_arr: np.ndarray, lon_arr: np.ndarray,
              mmask: np.ndarray) -> float:
    u_g, v_g = geostrophic_wind(Z, lat_arr, lon_arr)
    du = u - u_g
    dv = v - v_g
    ageo = np.sqrt(du ** 2 + dv ** 2)
    valid = ~np.isnan(ageo) & mmask
    if not valid.any():
        return float("nan")
    weights = np.broadcast_to(latitude_strip_weights(lat_arr)[:, None], ageo.shape)
    return float(
        np.sqrt(
            np.sum(weights[valid] * ageo[valid] ** 2)
            / np.sum(weights[valid])
        )
    )


def hydrostatic_resid(state: np.ndarray, lat_arr: np.ndarray) -> float:
    """|ΔZ - (-R T Δln p)| across adjacent PL layers, RMSE (m²/s²).

    state: (24, H, W) — physical values.
    Levels order in indices 0..3 (T) and 16..19 (Z): 1000,925,850,700.
    Layer pairs: (1000,925), (925,850), (850,700).
    """
    T = state[[IDX[f"T{lv}"] for lv in LEVELS]]           # (4, H, W) K
    Q = state[[IDX[f"Q{lv}"] for lv in LEVELS]]           # (4, H, W) kg/kg
    Z = state[[IDX[f"Z{lv}"] for lv in LEVELS]]           # (4, H, W) m²/s²
    virtual_T = T * (1.0 + 0.61 * Q)
    Tbar = 0.5 * (virtual_T[:-1] + virtual_T[1:])          # (3, H, W)
    dZ = Z[1:] - Z[:-1]                                    # (3, H, W)
    lnp = np.log(np.array(LEVELS, dtype=np.float32))       # (4,)
    dlnp = (lnp[1:] - lnp[:-1])[:, None, None]             # (3,1,1)
    expected_dZ = -R_DRY * Tbar * dlnp                     # hydrostatic
    resid = dZ - expected_dZ
    weights = latitude_strip_weights(lat_arr)[None, :, None]
    denominator = resid.shape[0] * float(weights.sum()) * resid.shape[-1]
    return float(np.sqrt(np.sum(weights * resid ** 2) / denominator))


class LinearInterp:
    def __call__(self, a, b, t, dt=6):
        w = t / dt
        return (1 - w) * a + w * b


class AnchorOnly:
    def __call__(self, a, b, t, dt=6):
        return a  # copy start anchor forward


class BareModel:
    def __init__(
        self,
        blob_path: str,
        device: str,
        means: np.ndarray,
        stds: np.ndarray,
        static_path: str,
    ):
        if torch is None:
            raise RuntimeError("torch required")
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from examples._bare_loader import load_bare
        self.net = load_bare(blob_path, device)
        self.device = device
        self.means = torch.from_numpy(means).to(device).view(1, -1, 1, 1)
        self.stds = torch.from_numpy(stds).to(device).view(1, -1, 1, 1)
        self.static = None
        n_static = getattr(self.net, "n_static_features", 0)
        if n_static > 0:
            s = torch.load(static_path, weights_only=False).float()
            if s.dim() == 3:
                s = s.unsqueeze(0)
            self.static = s[:, :n_static].to(device)

    def __call__(self, a, b, t, dt=6):
        with torch.no_grad():
            a_t = (
                torch.from_numpy(np.array(a, copy=True)[None]).to(self.device)
                - self.means
            ) / self.stds
            b_t = (
                torch.from_numpy(np.array(b, copy=True)[None]).to(self.device)
                - self.means
            ) / self.stds
            tau = torch.tensor([[t / dt]], device=self.device, dtype=torch.float32)
            if type(self.net).__name__ == "PixelAttentionVFI":
                f = self.net.net(a_t, b_t, tau)
            else:
                cond = torch.tensor([float(dt)], device=self.device, dtype=torch.float32)
                kwargs = {"static": self.static} if self.static is not None else {}
                f = self.net(a_t, b_t, tau, cond, **kwargs)
            if isinstance(f, tuple):
                f = f[0]
            return (f * self.stds + self.means).cpu().numpy()[0]


def select_start_hours(
    total_hours: int,
    delta_t_hours: int,
    stride_hours: int,
    max_pairs: int,
) -> list[int]:
    candidates = list(range(0, total_hours - delta_t_hours, stride_hours))
    if max_pairs <= 0 or len(candidates) <= max_pairs:
        return candidates
    positions = np.linspace(0, len(candidates) - 1, max_pairs)
    return [candidates[index] for index in np.rint(positions).astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--model-blob", required=True,
                    help="'linear' / 'anchor-only' / *.pt bare blob")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta-t-hours", type=int, default=6)
    ap.add_argument("--taus", nargs="+", type=int, default=None)
    ap.add_argument("--sample-stride-hours", type=int, default=48,
                    help="stride between eval samples (anchor pairs)")
    ap.add_argument("--max-pairs", type=int, default=180,
                    help="uniformly sample at most this many pairs over the full year; 0 keeps all")
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    args = ap.parse_args()
    dt = int(args.delta_t_hours)
    taus = args.taus if args.taus is not None else list(range(1, dt))
    taus = [int(t) for t in taus if 1 <= int(t) < dt]
    if not taus:
        raise SystemExit(f"no valid interior taus for delta_t={dt}")

    stats = load_channel_stats(
        args.stats_path,
        args.surface_stats_path,
        PAPER_CHANNELS_24,
    )
    means, stds = stats.mean, stats.std
    mm, meta = open_era5(Path(args.era5_dir), args.year)
    T, _, H, W = meta["shape"]
    lat_arr, lon_arr = build_grid(H, W)
    mmask = midlat_mask(lat_arr, W)
    print(f"[era5] T={T} H×W={H}×{W}", flush=True)

    if args.model_blob == "linear":
        interp = LinearInterp()
    elif args.model_blob == "anchor-only":
        interp = AnchorOnly()
    else:
        interp = BareModel(
            args.model_blob,
            args.device,
            means,
            stds,
            args.static_path,
        )

    # accumulate per-tau residuals
    acc = {
        t: {
            "ageo_850": [],
            "ageo_700": [],
            "ageo_true_850": [],
            "ageo_true_700": [],
            "hydro": [],
            "hydro_true": [],
        }
        for t in taus
    }
    n_pairs_used = 0

    start_hours = select_start_hours(
        T,
        dt,
        args.sample_stride_hours,
        args.max_pairs,
    )
    for h_start in start_hours:
        a_full = np.asarray(mm[h_start,       :24])  # (24, H, W)
        b_full = np.asarray(mm[h_start + dt,  :24])
        for tau in taus:
            truth = np.asarray(mm[h_start + tau, :24])
            pred = interp(a_full, b_full, tau, dt=dt)

            # The 24-channel protocol contains 1000/925/850/700 hPa.
            Z_700 = pred[IDX["Z700"]]
            u_700 = pred[IDX["U700"]]
            v_700 = pred[IDX["V700"]]
            Z_850 = pred[IDX["Z850"]]
            u_850 = pred[IDX["U850"]]
            v_850 = pred[IDX["V850"]]

            r700 = ageo_rmse(u_700, v_700, Z_700, lat_arr, lon_arr, mmask)
            r850 = ageo_rmse(u_850, v_850, Z_850, lat_arr, lon_arr, mmask)
            # Truth baseline for ratio
            t700 = ageo_rmse(truth[IDX["U700"]], truth[IDX["V700"]],
                              truth[IDX["Z700"]], lat_arr, lon_arr, mmask)
            t850 = ageo_rmse(truth[IDX["U850"]], truth[IDX["V850"]],
                              truth[IDX["Z850"]], lat_arr, lon_arr, mmask)
            hydro = hydrostatic_resid(pred, lat_arr)
            hydro_true = hydrostatic_resid(truth, lat_arr)

            acc[tau]["ageo_850"].append(r850)
            acc[tau]["ageo_700"].append(r700)
            acc[tau]["ageo_true_850"].append(t850)
            acc[tau]["ageo_true_700"].append(t700)
            acc[tau]["hydro"].append(hydro)
            acc[tau]["hydro_true"].append(hydro_true)
        n_pairs_used += 1
        if n_pairs_used % 10 == 0:
            print(f"[pairs] {n_pairs_used}/{args.max_pairs}", flush=True)

    per_tau = {}
    for tau in taus:
        r850 = np.nanmean(acc[tau]["ageo_850"]) if acc[tau]["ageo_850"] else float("nan")
        r700 = np.nanmean(acc[tau]["ageo_700"]) if acc[tau]["ageo_700"] else float("nan")
        t850 = np.nanmean(acc[tau]["ageo_true_850"]) if acc[tau]["ageo_true_850"] else float("nan")
        t700 = np.nanmean(acc[tau]["ageo_true_700"]) if acc[tau]["ageo_true_700"] else float("nan")
        hyd = np.nanmean(acc[tau]["hydro"]) if acc[tau]["hydro"] else float("nan")
        hyd_true = (
            np.nanmean(acc[tau]["hydro_true"])
            if acc[tau]["hydro_true"]
            else float("nan")
        )
        per_tau[str(tau)] = {
            args.model_name: {
                "ageo_rmse_ms_850": float(r850),
                "ageo_rmse_ms_700": float(r700),
                "ageo_rmse_ms_850_truth": float(t850),
                "ageo_rmse_ms_700_truth": float(t700),
                "ageo_ratio_850": float(r850 / t850) if t850 else float("nan"),
                "ageo_ratio_700": float(r700 / t700) if t700 else float("nan"),
                "hydrostatic_rmse_m2s2": float(hyd),
                "hydrostatic_rmse_m2s2_truth": float(hyd_true),
                "hydrostatic_ratio": (
                    float(hyd / hyd_true) if hyd_true else float("nan")
                ),
                "n_pairs": len(acc[tau]["ageo_700"]),
            }
        }

    out = {
        "per_tau": per_tau,
        "model_name": args.model_name,
        "midlat_band_deg": [MIDLAT_ABS_MIN, MIDLAT_ABS_MAX],
        "year": args.year,
        "delta_t_hours": dt,
        "taus": taus,
        "n_pairs_used": n_pairs_used,
        "sample_start_hours": start_hours,
        "sampling": {
            "strategy": "uniform_over_year",
            "candidate_stride_hours": args.sample_stride_hours,
            "max_pairs": args.max_pairs,
            "index_sha256": hashlib.sha256(
                "\n".join(map(str, start_hours)).encode("utf-8")
            ).hexdigest(),
        },
        "normalization": {
            "scheme": "(x - mean) / std",
            "stats": file_provenance(args.stats_path),
            "surface_stats": file_provenance(args.surface_stats_path),
            "static_features": file_provenance(args.static_path),
            "channels": list(PAPER_CHANNELS_24),
        },
        "evaluation_dataset_provenance": memmap_dataset_provenance(
            args.era5_dir,
            [args.year],
        ),
        "model_provenance": (
            {"kind": args.model_blob}
            if args.model_blob in {"linear", "anchor-only"}
            else file_provenance(args.model_blob)
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
