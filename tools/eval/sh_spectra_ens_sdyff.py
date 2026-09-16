#!/usr/bin/env python3
"""SH angular power spectra for S-DYff *ensemble* (paper-critical experiment).

Answers: does the ensemble-mean preserve HF energy, or does it blur high-
frequencies via regression-to-mean?

For each (date, channel) at a fixed tau (default 3h), this script:
  1. Computes the ERA5 ground-truth field at t0+tau.
  2. Runs the S-DYff Lightning model N times with MC-dropout active (matches
     ``eval_ensemble_crps.py`` ``sdyff_mc`` mode).
  3. Records SH angular power spectrum E(l) = sum_m |a_{l,m}|^2 for:
       * ground truth
       * single MC-dropout sample (sample index 0)
       * ensemble mean of all N samples
  4. Aggregates (mean over dates) per channel and saves per-channel NPZ.

It then derives the HF energy ratio
  R_HF = sum_{l >= l_lo} E(l) / sum_l E(l),   l_lo = floor(lmax/2) = 90
for each variant and writes a summary JSON + LaTeX table elsewhere
(see ``--out-dir``; companion plot/table scripts are stand-alone).

Notes on environment:
  * The S-DYff ckpt was trained with ``SDYFF_NLAT=360 SDYFF_NLON=720
    SDYFF_LAT_CROP=0`` and these MUST be in env at construction time.
  * ``torch_harmonics==0.6.5`` is required. The default ``wti-train:v1``
    image does not have it; the launch script ``pip install``s before
    starting the eval.

Output schema (NPZ, one per channel)::

    {
        "ell":              (lmax+1,) integer l axis,
        "channel":          str,
        "gt_spectrum":      (lmax+1,) E(l) for ground truth,
        "single_spectrum":  (lmax+1,) E(l) for one MC-dropout sample (idx 0),
        "ens_mean_spectrum":(lmax+1,) E(l) for ensemble mean (N samples),
        "R_HF_gt":          scalar HF energy ratio for ground truth,
        "R_HF_single":      scalar HF energy ratio for single sample,
        "R_HF_ens":         scalar HF energy ratio for ensemble mean,
        "ell_lo":           int (HF cutoff used),
        "n_dates":          int,
        "n_ensemble":       int,
        "tau_hours":        int,
        "H":                int, "W": int, "lmax": int,
        "ckpt":             str,
    }

Summary JSON (one file aggregating all channels): see ``--out-dir``/
``sdyff_ens<N>_tau<tau>_summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, "/workspace/code/wti")
torch.set_float32_matmul_precision("high")


# Standard 24-channel order produced by ERA5MemmapDataset for the WTI paper:
#   PL (T, U, V, Q, Z) x (1000, 925, 850, 700) hPa = 20 channels
#   surface (t2m, u10, v10, mslp)                   =  4 channels
ALL_24_CHANNELS = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]


# --------------------------- model loading ---------------------------

def load_sdyff_lit(ckpt_path: str, channel_groups: Dict[str, List[int]],
                   device: torch.device, envs: str = ""):
    """Load WeatherHermiteLightningModule for the S-DYff DYffusion ckpt."""
    for kv in envs.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v
    from trainer_weather_hermite import WeatherHermiteLightningModule  # type: ignore

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(state.get("hyper_parameters", {}))
    hparams["channel_groups"] = channel_groups
    for k in ("_class_path", "model_type_save"):
        hparams.pop(k, None)
    model = WeatherHermiteLightningModule(**hparams)
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  state load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model


def _set_dropout_train(module: nn.Module, train_mode: bool = True) -> None:
    """Toggle dropout/stochastic-depth layers to train mode for MC sampling.

    Mirrors `eval_ensemble_crps.py:_set_dropout_train`.
    """
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train(train_mode)
        cls = m.__class__.__name__.lower()
        if "droppath" in cls or "stochasticdepth" in cls:
            m.train(train_mode)


@torch.no_grad()
def sdyff_ensemble_predict(model, x0, xT, tau_norm, cond, static,
                           n_samples: int, n_ch_model: int) -> torch.Tensor:
    """N MC-dropout forward passes. Returns (N, B, 24, H, W) in normalised units."""
    model.eval()
    _set_dropout_train(model, train_mode=True)
    samples = []
    for _ in range(n_samples):
        out = model(x0[:, :n_ch_model], xT[:, :n_ch_model], tau_norm, cond, static)
        if isinstance(out, tuple):
            out = out[0]
        samples.append(out[:, :24])
    _set_dropout_train(model, train_mode=False)
    return torch.stack(samples, dim=0)


# --------------------------- SHT helpers ---------------------------

def _build_sht(H: int, W: int, lmax: int, device: torch.device):
    """Real spherical-harmonic transform on the ERA5 equiangular grid."""
    import torch_harmonics as th  # type: ignore
    sht = th.RealSHT(H, W, lmax=lmax + 1, mmax=lmax + 1, grid="equiangular").to(device)
    return sht


def _angular_power(field: torch.Tensor, sht) -> torch.Tensor:
    """E(l) = sum_m |a_{l,m}|^2 per (C). Input (B, C, H, W), out (C, lmax+1) f64."""
    coeffs = sht(field.double())                                    # (B, C, L, M) cf64
    p = coeffs.real.pow(2) + coeffs.imag.pow(2)                     # double m>0 below
    if p.size(-1) > 1:
        p[..., 1:] = p[..., 1:] * 2.0
    return p.sum(dim=-1).sum(dim=0)                                 # (C, lmax+1)


# --------------------------- main loop ---------------------------

def _select_channel_indices(channel_names: Sequence[str],
                            selected: Sequence[str]) -> List[int]:
    idx: List[int] = []
    for s in selected:
        if s not in channel_names:
            raise SystemExit(f"channel {s!r} not in dataset; have {channel_names[:8]} ...")
        idx.append(channel_names.index(s))
    return idx


def _subsample_dates(ds_base, n_dates: int, tau_h: int) -> None:
    """Keep at most `n_dates` distinct days from `ds_base.index`, spread over the year.

    Each kept day still contributes `samples_per_date` start-hour windows.
    """
    import datetime as _dt
    # Collect all unique (year, day-of-year) for entries that hit tau_h.
    keep_days: Dict[tuple, list] = {}
    for entry in ds_base.index:
        y, t0_h, eval_h, sph = entry
        if int(eval_h) != tau_h:
            continue
        doy = int(t0_h) // 24
        try:
            d = _dt.date(int(y), 1, 1) + _dt.timedelta(days=int(doy))
        except Exception:
            continue
        keep_days.setdefault((d.year, d.month, d.day), []).append(entry)
    all_days = sorted(keep_days.keys())
    if not all_days:
        raise SystemExit(f"no dataset entries at tau_h={tau_h}h")
    if len(all_days) > n_dates:
        # Uniform stride over the year.
        idxs = np.linspace(0, len(all_days) - 1, num=n_dates).round().astype(int)
        picked = [all_days[i] for i in idxs]
    else:
        picked = all_days
    new_index = []
    for day in picked:
        new_index.extend(keep_days[day])
    ds_base.index = new_index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="S-DYff Lightning ckpt path.")
    ap.add_argument("--memmap_dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--data_root", default="/workspace/code/wti")
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--n_ensemble", type=int, default=16)
    ap.add_argument("--n_dates", type=int, default=24,
                    help="Approx number of distinct calendar days to use.")
    ap.add_argument("--tau_hours", type=int, default=3,
                    help="Single tau to evaluate (default 3h, hardest interior point).")
    ap.add_argument("--channels",
                    default="t2m,u10,v10,mslp,T850,U850,Q850,Z700",
                    help="Comma-separated channels (8 representative by default).")
    ap.add_argument("--lmax", type=int, default=180)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--samples_per_date", type=int, default=4)
    ap.add_argument("--max_tau_hours", type=int, default=6)
    ap.add_argument("--out_dir", default="metrics/sh_spectra_ens")
    ap.add_argument("--model_name", default="sdyff",
                    help="Output prefix: <model_name>_ens<N>_tau<tau>_<ch>.npz")
    ap.add_argument("--envs", default="SDYFF_NLAT=360 SDYFF_NLON=720 SDYFF_LAT_CROP=0",
                    help="Env vars to set BEFORE importing trainer_weather_hermite.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # Apply --envs BEFORE importing the dataset / trainer (they read os.environ
    # at import / construct time).
    for kv in args.envs.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            os.environ[k] = v

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()
    tau_h = int(args.tau_hours)
    sel_channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    print(f"device={device}  tau_h={tau_h}  N={args.n_ensemble}  "
          f"n_dates={args.n_dates}  channels({len(sel_channels)})={sel_channels}")

    # Dataset: restrict eval to single tau for efficiency.
    data_root = Path(args.data_root)
    from weather_time_interp.memmap_dataset import ERA5MemmapDataset
    from weather_time_interp.eval_datasets import ERA5WeatherHermiteDataset

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[int(args.year)],
        max_tau_hours=int(args.max_tau_hours),
        samples_per_date=int(args.samples_per_date),
        train=False,
        eval_hours=[tau_h],
        static_path=str(data_root / "data/static_features_0p5.pt"),
        stats_path=str(data_root / "data/json_stats_0p5.nc"),
        surface_stats_path=str(data_root / "data/surface_stats_0p5.json"),
    )
    print(f"  raw test items at tau={tau_h}h: {len(ds_base.index)}")
    _subsample_dates(ds_base, args.n_dates, tau_h)
    print(f"  after date sub-sample (~{args.n_dates} days): {len(ds_base.index)} items")

    channel_names_full = list(ds_base.channel_names) + list(ds_base.surface_variables)
    channel_names = channel_names_full[:24]
    sel_idx = _select_channel_indices(channel_names, sel_channels)
    print(f"  selected channel indices: {sel_idx}")
    channel_groups = ds_base.channel_groups

    wrapped = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=float(args.max_tau_hours))
    loader = DataLoader(wrapped, batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers), shuffle=False,
                        pin_memory=True)

    # Model
    model = load_sdyff_lit(args.ckpt, channel_groups, device, args.envs)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  S-DYff loaded, params={n_params:.1f}M")
    n_ch_model = 27   # S-DYff was trained on 27ch; outputs sliced to 24 inside helper.

    # Probe H, W
    first = next(iter(loader))
    H, W = first["x0"].shape[-2:]
    print(f"  H x W = {H} x {W}")
    sht = _build_sht(H, W, args.lmax, device)
    lmax1 = args.lmax + 1

    # Accumulators per channel
    C = len(sel_idx)
    gt_acc = torch.zeros(C, lmax1, dtype=torch.float64, device=device)
    single_acc = torch.zeros(C, lmax1, dtype=torch.float64, device=device)
    ensmean_acc = torch.zeros(C, lmax1, dtype=torch.float64, device=device)
    n_acc = 0
    max_tau_f = float(args.max_tau_hours)

    print("  starting forward passes ...", flush=True)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_hour_all = batch["tau_hour"].long()   # (B, nH, 1)
            target_all = batch["target"].to(device, non_blocking=True)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)

            B = x0.size(0)
            nH = tau_hour_all.size(1)
            cond = torch.full((B, 1), max_tau_f, device=device, dtype=torch.float32)

            for h_idx in range(nH):
                tau_h_int = tau_hour_all[:, h_idx, 0]
                if int(tau_h_int[0].item()) != tau_h:
                    continue
                tau_norm = tau_h_int.float().to(device) / max_tau_f
                target_h = target_all[:, h_idx]

                samples = sdyff_ensemble_predict(
                    model, x0, xT, tau_norm, cond, static,
                    int(args.n_ensemble), n_ch_model,
                )   # (N, B, 24, H, W)
                single = samples[0]                          # (B, 24, H, W)
                ens_mean = samples.mean(dim=0)               # (B, 24, H, W)

                gt_sel = target_h[:, sel_idx]
                single_sel = single[:, sel_idx]
                ens_sel = ens_mean[:, sel_idx]

                gt_acc += _angular_power(gt_sel, sht)
                single_acc += _angular_power(single_sel, sht)
                ensmean_acc += _angular_power(ens_sel, sht)
                n_acc += int(gt_sel.size(0))

            if bi % 5 == 0:
                elapsed = (time.time() - t0)
                print(f"  batch {bi}/{len(loader)}  t={elapsed:.0f}s "
                      f"({elapsed/max(1,bi+1):.1f}s/batch)  n_acc={n_acc}", flush=True)

    if n_acc == 0:
        raise SystemExit("zero samples accumulated; check tau_h / dataset.")

    print(f"  total samples accumulated: {n_acc}")
    gt_arr = (gt_acc / n_acc).cpu().numpy()              # (C, lmax+1)
    single_arr = (single_acc / n_acc).cpu().numpy()
    ens_arr = (ensmean_acc / n_acc).cpu().numpy()

    ell = np.arange(lmax1, dtype=np.int32)
    ell_lo = int(args.lmax // 2)   # 90 for lmax=180

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "ckpt": args.ckpt,
        "tau_hours": tau_h,
        "n_ensemble": int(args.n_ensemble),
        "n_dates_target": int(args.n_dates),
        "n_samples_accumulated": int(n_acc),
        "lmax": int(args.lmax),
        "ell_lo": ell_lo,
        "year": int(args.year),
        "channels": sel_channels,
        "per_channel": [],
        "metric_note": ("R_HF = sum_{l>=l_lo} E(l) / sum_l E(l). "
                        "If R_HF(ens_mean) < R_HF(single) <= R_HF(gt), "
                        "the ensemble mean blurs HF -> regression-to-mean."),
    }
    for ci, ch in enumerate(sel_channels):
        gt_sp = gt_arr[ci]
        sg_sp = single_arr[ci]
        en_sp = ens_arr[ci]
        gt_tot = float(gt_sp.sum()) + 1e-30
        sg_tot = float(sg_sp.sum()) + 1e-30
        en_tot = float(en_sp.sum()) + 1e-30
        r_gt = float(gt_sp[ell_lo:].sum()) / gt_tot
        r_sg = float(sg_sp[ell_lo:].sum()) / sg_tot
        r_en = float(en_sp[ell_lo:].sum()) / en_tot
        # Relative HF blur of ensemble vs single sample (positive => blurring).
        rel_blur = (r_sg - r_en) / max(r_sg, 1e-12) * 100.0

        npz_path = out_dir / f"{args.model_name}_ens{args.n_ensemble}_tau{tau_h}_{ch}.npz"
        np.savez(npz_path,
                 ell=ell,
                 channel=ch,
                 gt_spectrum=gt_sp,
                 single_spectrum=sg_sp,
                 ens_mean_spectrum=en_sp,
                 R_HF_gt=r_gt,
                 R_HF_single=r_sg,
                 R_HF_ens=r_en,
                 ell_lo=ell_lo,
                 n_dates=int(args.n_dates),
                 n_samples_accumulated=int(n_acc),
                 n_ensemble=int(args.n_ensemble),
                 tau_hours=int(tau_h),
                 H=int(H), W=int(W),
                 lmax=int(args.lmax),
                 ckpt=args.ckpt)
        print(f"  wrote {npz_path}  R_HF gt={r_gt:.4f} single={r_sg:.4f} "
              f"ens={r_en:.4f}  blur(ens vs single)={rel_blur:+.2f}%")
        summary["per_channel"].append({
            "channel": ch,
            "R_HF_gt": round(r_gt, 6),
            "R_HF_single": round(r_sg, 6),
            "R_HF_ens": round(r_en, 6),
            "relative_blur_pct": round(rel_blur, 3),
        })

    # Headline numbers
    sg_arr = np.array([row["R_HF_single"] for row in summary["per_channel"]])
    en_arr = np.array([row["R_HF_ens"] for row in summary["per_channel"]])
    gt_arr_r = np.array([row["R_HF_gt"] for row in summary["per_channel"]])
    summary["aggregate"] = {
        "mean_R_HF_gt": float(gt_arr_r.mean()),
        "mean_R_HF_single": float(sg_arr.mean()),
        "mean_R_HF_ens": float(en_arr.mean()),
        "mean_relative_blur_pct": float(((sg_arr - en_arr) / np.maximum(sg_arr, 1e-12) * 100.0).mean()),
    }
    summary_path = out_dir / f"{args.model_name}_ens{args.n_ensemble}_tau{tau_h}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved summary: {summary_path}")
    print(f"  aggregate: gt={summary['aggregate']['mean_R_HF_gt']:.4f}  "
          f"single={summary['aggregate']['mean_R_HF_single']:.4f}  "
          f"ens={summary['aggregate']['mean_R_HF_ens']:.4f}  "
          f"mean blur={summary['aggregate']['mean_relative_blur_pct']:+.2f}%")
    print(f"=== DONE in {(time.time() - t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
