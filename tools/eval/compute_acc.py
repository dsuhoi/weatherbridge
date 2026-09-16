#!/usr/bin/env python3
"""Compute ACC (Anomaly Correlation Coefficient) for a single model checkpoint
on the WB2 0.5°→1° test set (2020).

Output: JSON with per-hour × per-channel ACC + per-hour aggregate.

Usage:
    python tools/eval/compute_acc.py \
        --model-checkpoint logs/exp_dcae_v9_skip/last.ckpt \
        --output metrics/acc_2020/dcae_skip.json \
        [--data-dir /workspace.../zarrs_1deg ...]

Aggregation uses streaming sums:
    ACC = sum(w * (p-c)*(t-c)) / sqrt(sum(w*(p-c)^2) * sum(w*(t-c)^2))
where w = cos(lat). Accumulated per (hour, channel) over all evaluation windows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dataset import ERA5ResNetODEDataset  # noqa: E402
from evaluate_baselines import (  # noqa: E402
    interpolate_time_with_f_interpolate,
    _forward_model_once,
    forward_ensemble,
)
from trainer_weather_hermite import WeatherHermiteLightningModule  # noqa: E402
from tools.eval.climatology import (  # noqa: E402
    climatology_time_weights,
    validate_climatology_archive,
)


def load_weather_hermite_safe(ckpt_path: Path, device, channel_groups):
    """Load by reading state_dict and inferring block_out_channels from tensor shapes.

    Bypasses Lightning's broken kwargs-default behavior on load_from_checkpoint.
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    hparams = dict(ckpt.get("hyper_parameters", {}))

    # For DC-AE models, use block_out_channels from ckpt hparams if length≥3 (per-stage);
    # otherwise fall back to default (128, 256, 512). The preset 'stable' default
    # (128,128,256,256) is for ResNetODE family; trainer line 594 enforces fallback for DC-AE.
    mt = hparams.get("model_type", "")
    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        if len(boc) < 3:
            hparams["block_out_channels"] = (128, 256, 512)
        else:
            hparams["block_out_channels"] = boc
        # Layers should be 3-tuple for 3-stage DC-AE; pad or trim
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        if len(lpb) > 3:
            hparams["layers_per_block"] = lpb[:3]
        elif len(lpb) < 3:
            hparams["layers_per_block"] = (lpb[0],) * 3 if lpb else (3, 3, 3)
        print(f"  DC-AE arch: block_out_channels={hparams['block_out_channels']}, layers_per_block={hparams['layers_per_block']}")

    # Cleanup keys Lightning auto-injects but __init__ does not accept
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)

    init_kwargs = dict(hparams)
    init_kwargs["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**init_kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  state loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:3]:
        print(f"    missing sample: {missing[:3]}")
    if unexpected[:3]:
        print(f"    unexpected sample: {unexpected[:3]}")
    model.to(device).eval()
    return model

# Channel groups order: PL_VARS × levels first (5×4=20), then surface (6).
PL_VARS = ["T", "U", "V", "Q", "Z"]
PL_LEVELS = [1000, 925, 850, 700]
SURF_VARS_FULL = ["t2m", "u10", "v10", "mslp", "sst", "tcc"]  # climatology-backed surface vars
SURF_VARS_WITH_TISR = SURF_VARS_FULL + ["tisr"]  # tisr is analytic, excluded from ACC

# Climatology var names → channel index mapping built dynamically.
CLIM_PL_NAMES = {"T": "t", "U": "u", "V": "v", "Q": "q", "Z": "z"}


class ClimatologyLookup:
    """Loads climatology (PL + surface) and provides linear interp lookup for (doy, hour).

    Accepts both surface-only climatology (no 'level' dim) and full PL+surface climatology
    (with 'level' dim). Channel names of form 'T1000', 'Q925' indicate PL channels; the
    rest are looked up as surface vars (e.g. 't2m', 'mslp').
    """
    def __init__(self, path: Path, channel_names: List[str], device: torch.device):
        self.device = device
        ds = xr.open_zarr(str(path), consolidated=True).load()
        self.archive_provenance = validate_climatology_archive(ds, path)
        H = ds.sizes["latitude"]; W = ds.sizes["longitude"]
        n_hour = ds.sizes["hour"]; n_doy = ds.sizes["dayofyear"]
        self.n_doy = int(n_doy)
        n_c = len(channel_names)
        arr = np.zeros((n_c, n_hour, n_doy, H, W), dtype=np.float32)
        has_levels = "level" in ds.coords
        src_levels = list(map(int, ds["level"].values)) if has_levels else []
        for ci, name in enumerate(channel_names):
            if len(name) > 1 and name[0] in CLIM_PL_NAMES and name[1:].isdigit():
                if not has_levels:
                    raise ValueError(f"climatology has no 'level' dim, cannot map PL channel '{name}'")
                var = name[0]
                lvl = int(name[1:])
                if lvl not in src_levels:
                    raise ValueError(f"climatology missing level {lvl} for '{name}'; have {src_levels}")
                short = CLIM_PL_NAMES[var]
                lvl_idx = src_levels.index(lvl)
                arr[ci] = ds[short].isel(level=lvl_idx).values
            else:
                if name not in ds.data_vars:
                    raise ValueError(f"climatology missing var '{name}'; have {list(ds.data_vars)}")
                arr[ci] = ds[name].values
        # SST: replace NaN over land with global mean (won't matter for ACC since target is also NaN)
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            mean_per_ch = np.nanmean(arr.reshape(n_c, -1), axis=1)
            for ci in range(n_c):
                arr[ci] = np.where(np.isnan(arr[ci]), mean_per_ch[ci], arr[ci])
        # Keep climatology on CPU (39 GB on 0.5° won't fit alongside model on GPU).
        # Transfer per-sample at lookup time.
        self.clim = torch.from_numpy(arr)  # (C, 4, 366, H, W) on CPU
        self.lat = torch.from_numpy(ds["latitude"].values).to(device)
        ds.close()

    def lookup(
        self,
        doy: int,
        hour_frac: float,
        year: int,
    ) -> torch.Tensor:
        """Linear interp between climatology 6h-points. hour_frac in [0, 24)."""
        h0_idx, h1_idx, doy0_idx, doy1_idx, w = (
            climatology_time_weights(
                day_of_year=doy,
                hour=hour_frac,
                year=year,
                climatology_days=self.n_doy,
            )
        )
        c0 = self.clim[:, h0_idx, doy0_idx]
        c1 = self.clim[:, h1_idx, doy1_idx]
        return (1.0 - w) * c0 + w * c1  # (C, H, W)


class ACCAccumulator:
    """Per-hour × per-channel streaming ACC sums."""
    def __init__(self, n_channels: int, lat_weights: torch.Tensor):
        # lat_weights: (H,) cos(lat) normalized
        self.n_channels = n_channels
        self.w = lat_weights.view(1, 1, -1, 1)  # broadcast to (1, 1, H, 1)
        # per_hour[h]["model" or "bilinear"] → dict of sum_xx, sum_yy, sum_xy per channel
        self.acc: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}

    def _ensure(self, hour: int, method: str):
        if hour not in self.acc:
            self.acc[hour] = {}
        if method not in self.acc[hour]:
            self.acc[hour][method] = {
                "sxy": torch.zeros(self.n_channels, dtype=torch.float64,
                                    device=self.w.device),
                "sxx": torch.zeros(self.n_channels, dtype=torch.float64,
                                    device=self.w.device),
                "syy": torch.zeros(self.n_channels, dtype=torch.float64,
                                    device=self.w.device),
            }

    def update(self, hour: int, method: str,
               pred_anom: torch.Tensor, tgt_anom: torch.Tensor):
        """pred_anom, tgt_anom: (B, C, H, W) physical-units anomalies."""
        self._ensure(hour, method)
        w = self.w  # (1, 1, H, 1)
        # Per-channel sums (sum over B, H, W).
        sxy = (w * pred_anom * tgt_anom).sum(dim=(0, 2, 3)).to(torch.float64)
        sxx = (w * pred_anom * pred_anom).sum(dim=(0, 2, 3)).to(torch.float64)
        syy = (w * tgt_anom * tgt_anom).sum(dim=(0, 2, 3)).to(torch.float64)
        d = self.acc[hour][method]
        d["sxy"] += sxy
        d["sxx"] += sxx
        d["syy"] += syy

    def finalize(self, channel_names: List[str]) -> Dict[int, Dict[str, Dict[str, float]]]:
        """Return dict[hour][method]["acc_<channel>"] = float."""
        out: Dict[int, Dict[str, Dict[str, float]]] = {}
        for hour, methods in self.acc.items():
            out[hour] = {}
            for m, d in methods.items():
                acc_ch = d["sxy"] / torch.sqrt(d["sxx"] * d["syy"] + 1e-12)
                row = {}
                for ci, name in enumerate(channel_names):
                    row[f"acc_{name}"] = float(acc_ch[ci].item())
                row["acc_mean"] = float(acc_ch.mean().item())
                out[hour][m] = row
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-checkpoint", type=str, default=None)
    ap.add_argument("--model-type", type=str, default=None)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--climatology",
                    default="/workspace-SR006.nfs2/weather_data/time_interpolation_0p5/climatology_1990-2019_0p5_canonical_v2.zarr",
                    help="Climatology zarr (default: 0.5° full PL+surface)")
    ap.add_argument("--data-dir", type=str, required=True,
                    help="Per-year PL zarrs root (1°)")
    ap.add_argument("--surface-data-dir", type=str, default=None)
    ap.add_argument("--stats-path", type=str, default="data/json_stats.nc")
    ap.add_argument("--surface-stats-path", type=str, default=None)
    ap.add_argument("--static-path", type=str, default="data/static_features.pt")
    ap.add_argument("--years", type=int, nargs="+", default=[2020])
    ap.add_argument("--variables", type=str, nargs="+", default=PL_VARS)
    ap.add_argument("--pressure-levels", type=int, nargs="+", default=PL_LEVELS)
    ap.add_argument("--surface-variables", type=str, nargs="+", default=SURF_VARS_WITH_TISR)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--eval-days-per-month", type=int, default=4)
    ap.add_argument("--delta-t-hours", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--ensemble-passes", type=int, default=1)
    ap.add_argument("--cache-in-ram", action="store_true")
    ap.add_argument("--analytic-tisr", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"checkpoint: {args.model_checkpoint}")
    print(f"output: {args.output}")

    eval_hours = list(range(args.delta_t_hours + 1))
    dataset = ERA5ResNetODEDataset(
        data_dir=args.data_dir,
        years=args.years,
        variables=args.variables,
        pressure_levels=args.pressure_levels,
        max_tau_hours=int(args.delta_t_hours),
        samples_per_date=args.samples_per_date,
        train=False,
        static_path=args.static_path,
        stats_path=args.stats_path,
        eval_hours=eval_hours,
        cache_in_ram=args.cache_in_ram,
        surface_data_dir=args.surface_data_dir,
        surface_variables=args.surface_variables,
        surface_stats_path=args.surface_stats_path,
        use_analytic_tisr=args.analytic_tisr,
    )
    # Apply economy day filter
    if args.eval_days_per_month is not None:
        K = max(1, int(args.eval_days_per_month))
        day_picks = {3: [1, 11, 21], 4: [1, 8, 15, 22], 5: [1, 7, 14, 21, 28]}.get(
            K, sorted({1 + i * (30 // K) for i in range(K)})
        )
        allowed = set(day_picks)
        import datetime as _dt
        filt = []
        for entry in dataset.index:
            y, t0, _, _ = entry
            doy = t0 // 24
            try:
                d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
            except Exception:
                continue
            if d.day in allowed:
                filt.append(entry)
        print(f"  economy filter: {len(filt)}/{len(dataset.index)} index entries")
        dataset.index = filt

    full_channel_names = list(dataset.channel_names) + list(dataset.surface_variables)
    print(f"all model channels ({len(full_channel_names)}): {full_channel_names}")
    # ACC: include all channels except tisr (analytic) and any not present in climatology.
    skip = {"tisr"}
    # Pre-probe climatology to skip missing surface vars
    _probe = xr.open_zarr(str(args.climatology), consolidated=True)
    _clim_vars = set(_probe.data_vars)
    _probe.close()
    for c in full_channel_names:
        is_pl = len(c) > 1 and c[0] in CLIM_PL_NAMES and c[1:].isdigit()
        if is_pl:
            short = CLIM_PL_NAMES[c[0]]
            if short not in _clim_vars:
                skip.add(c)
        else:
            if c not in _clim_vars and c not in skip:
                skip.add(c)
    if skip:
        print(f"ACC skip channels (not in climatology / analytic): {sorted(skip)}")
    channel_names = [c for c in full_channel_names if c not in skip]
    acc_ch_indices = [i for i, c in enumerate(full_channel_names) if c not in skip]
    n_channels = len(channel_names)
    print(f"ACC channels ({n_channels}): {channel_names}")
    print(f"ACC channel indices in model output: {acc_ch_indices}")

    # Lat weights from data grid (180 lat × 360 lon)
    sample_ds = list(dataset.datasets.values())[0]
    lat = torch.from_numpy(sample_ds["latitude"].values).to(device)
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.sum()
    print(f"lat range: [{float(lat.min()):.1f}, {float(lat.max()):.1f}], H={len(lat)}")

    # Climatology
    print(f"loading climatology from {args.climatology}...")
    clim = ClimatologyLookup(Path(args.climatology), channel_names, device)
    print(f"climatology shape: {tuple(clim.clim.shape)}")

    # Stats for denormalization (channel-level)
    mu_all = torch.cat([dataset.mu, dataset.surface_mu], dim=0).to(device) \
        if dataset.surface_mu is not None else dataset.mu.to(device)
    sigma_all = torch.cat([dataset.sigma, dataset.surface_sigma], dim=0).to(device) \
        if dataset.surface_sigma is not None else dataset.sigma.to(device)
    mu_all = mu_all.view(1, -1, 1, 1)
    sigma_all = sigma_all.view(1, -1, 1, 1)

    # Model
    model, model_type = None, None
    if args.model_checkpoint:
        model = load_weather_hermite_safe(
            Path(args.model_checkpoint), device, dataset.channel_groups
        )
        model_type = args.model_type or "weather_hermite"
        print(f"model loaded, type={model_type}, "
              f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")

    acc = ACCAccumulator(n_channels, w_lat)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    t_start = time.time()
    n_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x0 = batch["x0"].to(device)
            x1 = batch["x1"].to(device)
            tau = batch["tau"].to(device)
            target = batch["target"].to(device)
            time_emb = batch["time_emb"].to(device)
            static = batch.get("static")
            if static is not None:
                static = static.to(device)
            bsz = x0.size(0)

            # Compute predictions
            pred_bil = interpolate_time_with_f_interpolate(x0, x1, tau, mode="bilinear")
            if model is not None:
                pred_model, _ = forward_ensemble(
                    model, model_type, x0, x1, tau, time_emb, static, device,
                    n_passes=args.ensemble_passes,
                )

            # Denormalize back to physical units (full channel set incl. tisr)
            target_phys = target * sigma_all + mu_all
            pred_bil_phys = pred_bil * sigma_all + mu_all
            if model is not None:
                pred_model_phys = pred_model * sigma_all + mu_all
            # Restrict to ACC channels (exclude tisr)
            acc_idx_t = torch.tensor(acc_ch_indices, device=device, dtype=torch.long)
            target_phys = target_phys.index_select(1, acc_idx_t)
            pred_bil_phys = pred_bil_phys.index_select(1, acc_idx_t)
            if model is not None:
                pred_model_phys = pred_model_phys.index_select(1, acc_idx_t)

            # Per-sample climatology lookup & ACC accumulation
            for i in range(bsz):
                idx_global = batch_idx * args.batch_size + i
                if idx_global >= len(dataset.index):
                    break
                year, t0, tau_hours, _ = dataset.index[idx_global]
                ds_year = dataset.datasets[year]
                t_target = ds_year.time.values[t0 + tau_hours]
                tt = datetime.utcfromtimestamp(t_target.astype("datetime64[s]").astype(int))
                doy = tt.timetuple().tm_yday
                hour_frac = tt.hour + tt.minute / 60.0
                clim_at_t = clim.lookup(
                    doy,
                    hour_frac,
                    tt.year,
                ).unsqueeze(0).to(device)  # (1, C_acc, H, W)

                tgt_anom = target_phys[i:i+1] - clim_at_t
                bil_anom = pred_bil_phys[i:i+1] - clim_at_t
                acc.update(tau_hours, "bilinear", bil_anom, tgt_anom)
                if model is not None:
                    mdl_anom = pred_model_phys[i:i+1] - clim_at_t
                    acc.update(tau_hours, "model", mdl_anom, tgt_anom)

            n_processed += bsz
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)} processed={n_processed} "
                      f"elapsed={time.time()-t_start:.0f}s")

    print(f"total processed: {n_processed} in {(time.time()-t_start)/60:.1f} min")

    # Finalize ACC
    result_per_hour = acc.finalize(channel_names)
    payload = {
        "checkpoint": args.model_checkpoint,
        "model_type": model_type,
        "num_samples": n_processed,
        "years": args.years,
        "channel_names": channel_names,
        "per_hour": result_per_hour,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved → {out_path}")

    # Quick summary
    print("\n=== ACC summary (mean across channels) ===")
    for hour in sorted(result_per_hour.keys()):
        for m, vals in result_per_hour[hour].items():
            print(f"  h={hour:2d}  {m:>10s}  ACC_mean={vals['acc_mean']:.4f}")


if __name__ == "__main__":
    main()
