"""Diurnal amplitude preservation for temporal interpolation.

Motivation (npj CAS narrative): coarse (6h) fields alias the diurnal
cycle — the 12/18 UTC peaks in tropical land regions get chopped down,
so any downstream application driven by 6h-linear forcing sees biased
insolation-response variables (t2m, u10, v10 sea breeze). We measure
how well temporal interpolators recover the true hourly diurnal
amplitude at fixed monitor boxes.

Regions (5×5 gridcell boxes at 0.5°, ~2.5° × 2.5° each):
  * Sahel        (10-15°N, 0-5°E)
  * Amazon       (5°S-0°N, 65-60°W)
  * Congo        (2°S-3°N, 22-27°E)
  * SE Australia (35-30°S, 145-150°E)

Method:
  For each region, extract truth t2m/u10/v10 at every hour of the year
  from the ERA5 hourly memmap (channel indices from paper_units).
  For 6h boundaries {00, 06, 12, 18 UTC} run the model between adjacent
  anchors, output τ ∈ {1..5}h, splice with anchors → full hourly series.
  Compute diurnal amplitude:
      A(t2m) = max_hour(mean_across_days(t2m)) - min_hour(...)
  per month.

  For linear baseline just replace model output with linear-interp of
  the two anchors. Anchor-only baseline (skip interior) copies each
  anchor forward to next.

Output:
  metrics/downstream_diurnal/<model_name>.json with per region-month:
    diurnal_amp_true, diurnal_amp_model, diurnal_amp_linear,
    amp_ratio_model, amp_ratio_linear, peak_hour_true, peak_hour_model
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

SURF_CH_IDX = {c: 20 + i for i, c in enumerate(["t2m", "u10", "v10", "mslp"])}

REGIONS = {
    "sahel":   {"lat": (10.0, 15.0), "lon": (0.0, 5.0)},
    "amazon":  {"lat": (-5.0, 0.0),  "lon": (295.0, 300.0)},  # 65W-60W
    "congo":   {"lat": (-2.0, 3.0),  "lon": (22.0, 27.0)},
    "se_aus":  {"lat": (-35.0, -30.0), "lon": (145.0, 150.0)},
}

def build_grid(H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the coordinates encoded by the canonical WB2 memmap."""
    return wb2_block_average_latitudes(H), wb2_block_average_longitudes(W)


def region_slice(lat_arr: np.ndarray, lon_arr: np.ndarray, reg: dict) -> tuple[slice, slice]:
    lat_lo, lat_hi = reg["lat"]
    lon_lo, lon_hi = reg["lon"]
    lat_i = np.where((lat_arr >= lat_lo) & (lat_arr <= lat_hi))[0]
    lon_i = np.where((lon_arr >= lon_lo) & (lon_arr <= lon_hi))[0]
    return slice(lat_i.min(), lat_i.max() + 1), slice(lon_i.min(), lon_i.max() + 1)


def open_era5_year(era5_dir: Path, year: int):
    bin_p = era5_dir / f"wb2_{year}.bin"
    json_p = era5_dir / f"wb2_{year}.json"
    meta = json.loads(json_p.read_text())
    mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                   shape=tuple(meta["shape"]))
    return mm, meta


def build_linear_interp(anchor_a: np.ndarray, anchor_b: np.ndarray, taus: list[int], dt: int = 6):
    """anchor_[a,b]: (C, H, W). Returns list of (C, H, W) for tau=1..dt-1."""
    return [(1 - t / dt) * anchor_a + (t / dt) * anchor_b for t in taus]


def build_model_interp(model, anchor_a: np.ndarray, anchor_b: np.ndarray,
                       taus: list[int], means: np.ndarray, stds: np.ndarray,
                       device: str,
                       static: torch.Tensor | None = None,
                       dt: int = 6):
    means_t = torch.from_numpy(means).to(device).view(1, -1, 1, 1)
    stds_t = torch.from_numpy(stds).to(device).view(1, -1, 1, 1)
    a = (
        torch.from_numpy(np.array(anchor_a, copy=True)[None]).to(device)
        - means_t
    ) / stds_t
    b = (
        torch.from_numpy(np.array(anchor_b, copy=True)[None]).to(device)
        - means_t
    ) / stds_t
    is_atmvfi = type(model).__name__ == "PixelAttentionVFI"
    cond = torch.tensor([float(dt)], device=device, dtype=torch.float32)
    kwargs = {"static": static} if static is not None else {}
    outs = []
    with torch.no_grad():
        for t in taus:
            tau = torch.tensor([[t / dt]], device=device, dtype=torch.float32)
            if is_atmvfi:
                out = model.net(a, b, tau)
            else:
                out = model(a, b, tau, cond, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            outs.append((out * stds_t + means_t).cpu().numpy()[0])
    return outs


def diurnal_amplitude(
    hourly: np.ndarray, utc_offset_hours: int = 0
) -> tuple[float, int]:
    """hourly: (N_days*24, ...) — mean across grid dims already applied.

    Returns (amp, peak_hour) where amp = max-min of climatological
    diurnal profile (24 values, one per hour).
    """
    n = hourly.shape[0]
    n_days = n // 24
    if n_days == 0:
        return float("nan"), -1
    profile = hourly[: n_days * 24].reshape(n_days, 24).mean(axis=0)
    profile = np.roll(profile, int(utc_offset_hours))
    return float(profile.max() - profile.min()), int(np.argmax(profile))


def region_utc_offset(region: dict) -> int:
    """Approximate local solar-time offset from the region centre longitude."""
    lon = 0.5 * (float(region["lon"][0]) + float(region["lon"][1]))
    if lon > 180.0:
        lon -= 360.0
    return int(np.rint(lon / 15.0))


def month_hour_bounds(
    year: int,
    month: int,
    total_hours: int,
) -> tuple[int, int, int]:
    """Return a complete-day month window with an available right anchor.

    A single-year memmap has no 00 UTC anchor from the following year.  The
    final calendar day is therefore excluded instead of leaking truth into its
    five interior slots or indexing one element beyond the memmap.
    """
    year_start = np.datetime64(f"{year}-01-01T00", "h")
    start = np.datetime64(f"{year}-{month:02d}-01T00", "h")
    if month == 12:
        following = np.datetime64(f"{year + 1}-01-01T00", "h")
    else:
        following = np.datetime64(f"{year}-{month + 1:02d}-01T00", "h")
    start_h = int((start - year_start) / np.timedelta64(1, "h"))
    calendar_end_h = int(
        (following - year_start) / np.timedelta64(1, "h")
    )
    end_h = min(calendar_end_h, int(total_hours))
    dropped_hours = 0
    if end_h == total_hours:
        bounded_hours = max(0, ((end_h - start_h - 1) // 24) * 24)
        bounded_end_h = start_h + bounded_hours
        dropped_hours = end_h - bounded_end_h
        end_h = bounded_end_h
    return start_h, end_h, dropped_hours


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--era5-dir", required=True)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--model-blob", required=True, help="'linear'/'anchor-only' or *.pt bare blob")
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta-t-hours", type=int, default=6)
    ap.add_argument("--taus", nargs="+", type=int, default=None)
    ap.add_argument("--surface-channels", nargs="+", default=["t2m", "u10", "v10"])
    ap.add_argument("--anchor-hours", nargs="+", type=int, default=[0, 6, 12, 18])
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
    mm, meta = open_era5_year(Path(args.era5_dir), args.year)
    T, _, H, W = meta["shape"]
    lat_arr, lon_arr = build_grid(H, W)
    print(f"[era5] year={args.year} T={T} H×W={H}×{W} regions={list(REGIONS)}", flush=True)

    reg_slices = {r: region_slice(lat_arr, lon_arr, REGIONS[r]) for r in REGIONS}

    # Load model if needed
    model = None
    static_t = None
    is_linear = args.model_blob == "linear"
    is_anchor_only = args.model_blob == "anchor-only"
    if not (is_linear or is_anchor_only):
        if torch is None:
            raise RuntimeError("torch required for bare-blob model")
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from examples._bare_loader import load_bare
        model = load_bare(args.model_blob, args.device)
        n_static = getattr(model, "n_static_features", 0)
        if n_static > 0:
            s = torch.load(args.static_path, weights_only=False).float()
            if s.dim() == 3:
                s = s.unsqueeze(0)
            static_t = s[:, :n_static].to(args.device)

    # Iterate months
    monthly = {r: {} for r in REGIONS}
    month_coverage = {}

    for month in range(1, 13):
        start_h, end_h, dropped_hours = month_hour_bounds(
            args.year,
            month,
            T,
        )
        n_hours = end_h - start_h
        if n_hours < 24 * 7:
            continue
        month_coverage[f"{args.year}-{month:02d}"] = {
            "n_hours": n_hours,
            "n_complete_days": n_hours // 24,
            "dropped_tail_hours_without_right_anchor": dropped_hours,
        }

        # Extract truth once per region
        surf_idx = [SURF_CH_IDX[c] for c in args.surface_channels]
        truth_by_region = {}
        recon_by_region = {}
        for r_name, sl in reg_slices.items():
            lat_sl, lon_sl = sl
            truth_by_region[r_name] = np.asarray(mm[start_h:end_h, surf_idx, lat_sl, lon_sl])
            recon_by_region[r_name] = np.copy(truth_by_region[r_name])

        # Single anchor-pair pass — one model call per (h_start,τ), extract per region
        for h_start in range(0, n_hours - dt + 1, dt):
            absolute_hour = start_h + h_start
            if absolute_hour % 24 not in set(args.anchor_hours):
                continue
            anchor_a_full = np.asarray(mm[start_h + h_start, :24])
            anchor_b_full = np.asarray(mm[start_h + h_start + dt, :24])
            if is_anchor_only:
                interp_full = [anchor_a_full for _ in taus]
            elif is_linear:
                interp_full = build_linear_interp(anchor_a_full, anchor_b_full, taus, dt=dt)
            else:
                interp_full = build_model_interp(model, anchor_a_full, anchor_b_full,
                                                 taus, means, stds, args.device,
                                                 static=static_t, dt=dt)
            for i_t, t in enumerate(taus):
                surf_full = interp_full[i_t][surf_idx]  # (n_ch, H, W)
                for r_name, sl in reg_slices.items():
                    lat_sl, lon_sl = sl
                    recon_by_region[r_name][h_start + t] = surf_full[:, lat_sl, lon_sl]

        # Per-region diurnal amplitude
        for r_name in reg_slices:
            truth = truth_by_region[r_name]
            recon = recon_by_region[r_name]
            utc_offset = region_utc_offset(REGIONS[r_name])
            entry = {}
            for ci, ch in enumerate(args.surface_channels):
                truth_mean = truth[:, ci].mean(axis=(1, 2))
                recon_mean = recon[:, ci].mean(axis=(1, 2))
                a_t, ph_t = diurnal_amplitude(truth_mean, utc_offset)
                a_r, ph_r = diurnal_amplitude(recon_mean, utc_offset)
                peak_error = min(abs(ph_r - ph_t), 24 - abs(ph_r - ph_t))
                entry[ch] = {
                    "amp_true": a_t, "amp_recon": a_r,
                    "amp_ratio": a_r / a_t if a_t else float("nan"),
                    "peak_hour_true": ph_t, "peak_hour_recon": ph_r,
                    "peak_hour_shift": ph_r - ph_t,
                    "peak_hour_error": peak_error,
                    "utc_offset_hours": utc_offset,
                }
            monthly[r_name][f"{args.year}-{month:02d}"] = entry
        print(f"[m{month:02d}] done", flush=True)

    out = {
        "model_name": args.model_name,
        "year": args.year,
        "delta_t_hours": dt,
        "taus": taus,
        "regions": REGIONS,
        "surface_channels": args.surface_channels,
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
        "coverage": {
            "anchor_hours_utc": sorted(set(args.anchor_hours)),
            "months": 12,
            "last_anchor_interval_included": True,
            "month_windows": month_coverage,
            "time_basis": "approximate local solar time",
        },
        "model_provenance": (
            {"kind": args.model_blob}
            if is_linear or is_anchor_only
            else file_provenance(args.model_blob)
        ),
        "monthly": monthly,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
