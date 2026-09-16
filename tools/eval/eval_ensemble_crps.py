#!/usr/bin/env python3
"""Ensemble RMSE + CRPS (lat-weighted) eval for stochastic VFI models.

Supports two stochastic models:

  --mode corrdiff_fm  --base_ckpt PATH (frozen base) --fm_ckpt PATH (FM head ckpt)
      Forward:
        x_base = base(x_0, x_T, tau_vfi)  # frozen deterministic
        For k in 1..N:
          x_init_k = randn
          delta_pred_k = Euler-N-step ODE solve from x_init_k
          x_VFI_k = x_base + scale * delta_pred_k

  --mode sdyff_mc  --ckpt PATH (S-DYff Lightning ckpt)
      Forward (MC-dropout): enable dropout at inference, run N forward passes.
        x_VFI_k = model(x_0, x_T, tau, ...)  with dropout active

Outputs two JSONs:
  --out_rmse PATH   ensemble-mean RMSE (lat-weighted, normalised), schema like
                    metrics/eval_6h_2020_paper_leaderboard/*_24ch_*_ep8.json
  --out_crps PATH   fair-CRPS (Hersbach) per-channel per-tau, lat-weighted

Schema for CRPS JSON:
{
  "model_name": ..., "ckpt": ..., "n_channels": 24, "channels": [...],
  "tau_hours": [1..5], "n_ensemble": N, "n_samples_per_tau": [...],
  "crps_norm": list[5] of list[24],     # normalised units (consistent with rmse_model_norm)
  "crps_phys": list[5] of list[24],     # physical units
  "stds_for_denorm": [...]
}

CRPS fair estimator (per-pixel, per-channel):
  CRPS = mean_i |X_i - y| - 0.5 * mean_{i,j} |X_i - X_j|
Lat-weighted average across pixels per channel.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Repo root: prefer /workspace/code/wti (docker on fibo), else find from this file
# (cloud.ru native: /home/jovyan/dsuhoi/weather_time_interpolation/tools/eval/this).
_DOCKER_ROOT = "/workspace/code/wti"
if os.path.isdir(_DOCKER_ROOT):
    _REPO_ROOT = _DOCKER_ROOT
else:
    _REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)
torch.set_float32_matmul_precision("high")

from weather_time_interp.memmap_dataset import ERA5MemmapDataset  # noqa: E402
from trainer_weather_hermite import WeatherHermiteLightningModule  # noqa: E402


CH_24 = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]
CH_27 = CH_24 + ["sst", "tcc", "tcwv"]
TAU_HOURS = [1, 2, 3, 4, 5]


def lat_weights(H, device, dtype):
    lat = np.linspace(89.75, -89.75, H, dtype=np.float32) if H == 360 else np.linspace(90.0, -90.0, H, dtype=np.float32)
    w = np.cos(np.deg2rad(lat))
    w = w / w.sum()
    return torch.from_numpy(w).to(device=device, dtype=dtype).view(1, 1, -1, 1)


def get_channel_stds(stats_path, surface_stats_path, channel_names):
    import xarray as xr
    pl_ds = xr.open_dataset(stats_path)
    pl_params = [str(p) for p in pl_ds.params.values.tolist()]
    pl_stds = pl_ds.climate_statistics.sel(stats="std").values
    pl_map = {pl_params[i]: float(pl_stds[i]) for i in range(len(pl_params))}
    surf = json.load(open(surface_stats_path))
    stds = []
    for ch in channel_names:
        if ch in pl_map:
            stds.append(pl_map[ch])
        elif ch in surf:
            stds.append(float(surf[ch]["std"]))
        else:
            raise KeyError(f"no std for channel {ch}")
    return np.array(stds, dtype=np.float32)


def fair_crps_per_pixel(samples, target, lat_w):
    """Compute fair CRPS estimator with lat-weighted averaging across H,W.

    samples: (N, B, C, H, W)
    target:  (B, C, H, W)
    lat_w:   (1, 1, H, 1) with sum_h lat_w == 1, broadcast over W.

    Returns lat-weighted per-pixel-mean CRPS averaged over (H, W) and summed
    over B (so caller can divide by total count). Per channel.

    Memory: O(N * B * C * H * W) which is large but tractable for N=16 on A100/B300.
    """
    N, B, C, H, W = samples.shape
    # |X_i - y| term: shape (N, B, C, H, W)
    abs_diff_y = (samples - target.unsqueeze(0)).abs()
    mean_abs_y = abs_diff_y.mean(dim=0)  # (B, C, H, W)

    # |X_i - X_j| pairwise term: mean over i,j (N^2 pairs). Use efficient sort-based
    # formula: 2/N^2 * sum_i (2i - N - 1) * X_(i) where X_(i) are sorted samples.
    samples_sorted, _ = samples.sort(dim=0)
    idx = torch.arange(N, device=samples.device, dtype=samples.dtype).view(N, 1, 1, 1, 1)
    weights = 2 * idx - N + 1  # for fair estimator: sum_i (2i - N + 1) * X_(i) / (N*(N-1))
    pairwise_term = (weights * samples_sorted).sum(dim=0) / (N * (N - 1))  # (B, C, H, W)

    crps_field = mean_abs_y - 0.5 * pairwise_term  # actually fair has * (1 - 1/N); see below

    # NOTE: the unbiased fair CRPS estimator is
    #   CRPS_fair = mean_i |X_i - y| - 1/(2 N (N-1)) * sum_{i!=j} |X_i - X_j|
    # which equals  mean_abs_y - 1/(N(N-1)) * sum_{i<j} |X_i - X_j|.
    # The sort-based formula above gives the standard (biased) CRPS = mean_abs_y -
    # 1/(2 N^2) * sum_{i,j} |X_i - X_j|, equivalently mean_abs_y - 1/N^2 * sum_{i<j} 2|X_i-X_j|.
    # For the fair version we re-scale: cancel /N^2 to get /(N(N-1)).
    # To compute fair: replace the pairwise_term factor.
    # Re-derive cleanly: pairwise_abs_sum = 2 * sum_{i<j} |X_i - X_j| = sum_{i,j} |X_i - X_j|
    # Sort-trick: sum_{i,j}|X_i - X_j| = sum_i (2i - N + 1) * X_(i)
    # So  fair_pairwise_mean = sum_{i,j}|X_i - X_j| / (N * (N - 1)) = pairwise_term  (as computed).
    # But /2 because in CRPS the spread term has coefficient 1/2 * mean_pairwise
    # = (1 / (2 N (N-1))) * sum_{i,j ne i} |X_i - X_j| =
    # = (1 / (N (N-1))) * sum_{i<j} |X_i - X_j|
    # = pairwise_term  (already / (N(N-1)) above).
    # Hence the fair CRPS:
    crps_field_fair = mean_abs_y - 0.5 * pairwise_term

    # Lat-weighted mean over (H,W): lat_w has shape (1,1,H,1) with sum_h == 1.
    # Mean over W is just divide by W. So per (B,C):
    crps_pc = (crps_field_fair * lat_w).sum(dim=(-2, -1)) / W  # (B, C)
    return crps_pc


# ---------------- CorrDiff/FM mode ----------------

def load_corrdiff_fm(fm_ckpt, base_ckpt, device):
    """Load a CorrDiffFMVFI Lightning ckpt. Returns a callable model and metadata.

    Compatible with train_corrdiff_fm_weatherdcae.py (24ch base) and
    train_corrdiff_fm_v2.py (27ch base / DC-AE Skip).
    """
    sys.path.insert(0, _REPO_ROOT)
    # Dynamically detect the trainer module — try both names.
    state = torch.load(fm_ckpt, map_location="cpu", weights_only=False)
    hparams = dict(state["hyper_parameters"])
    in_channels = hparams.get("in_channels", 24)
    hidden = hparams.get("hidden", 64)
    n_levels = hparams.get("n_levels", 3)
    n_ode_steps = hparams.get("n_ode_steps", 5)

    # Import the script as a module to get model classes.
    import importlib.util
    if in_channels == 24:
        # Try canonical 6h script first; fall back to the 12h variant (cloud.ru).
        for candidate in (
            f"{_REPO_ROOT}/train_corrdiff_fm_weatherdcae.py",
            f"{_REPO_ROOT}/train_corrdiff_fm_weatherdcae_12h.py",
        ):
            if Path(candidate).exists():
                script_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"no CorrDiffFMVFI trainer script found at {_REPO_ROOT}/")
    else:
        script_path = f"{_REPO_ROOT}/train_corrdiff_fm_v2.py"
    spec = importlib.util.spec_from_file_location("corrdiff_module", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Initialize with the base ckpt (will load frozen base weights), then overwrite all weights
    # via the FM head ckpt's state_dict.
    Module = mod.CorrDiffFMVFI
    # Force use of provided base_ckpt at construction (will load frozen base).
    init_kwargs = dict(hparams)
    if "base_ckpt" in init_kwargs:
        init_kwargs["base_ckpt"] = base_ckpt
    elif "dcae_skip_ckpt" in init_kwargs:
        init_kwargs["dcae_skip_ckpt"] = base_ckpt
    # Remove keys not in __init__
    sig = init_kwargs
    model = Module(**sig)
    # Load the FM ckpt state (this overwrites base weights too, but they were already frozen
    # to the WeatherDCAE values anyway since base_ckpt is the same path).
    missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
    if missing or unexpected:
        print(f"  state load: missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device).eval()
    return model, in_channels, n_ode_steps


@torch.no_grad()
def corrdiff_ensemble_predict(model, x0_n, xT_n, target_n, tau_vfi, static, n_samples):
    """Generate N stochastic predictions via random x_init seeds. Returns (N, B, C, H, W)."""
    B = x0_n.size(0)
    static_dev = static.to(x0_n.device) if static is not None else None
    x_base = model._baseline_pred(x0_n, xT_n, tau_vfi, static_dev)
    samples = []
    s = torch.tanh(model.scale).view(1, -1, 1, 1)
    for _ in range(n_samples):
        xt = torch.randn_like(x_base) * 0  # zero-init Euler integrator (consistent with training)
        # train_corrdiff_fm_v2 uses xt = zeros at inference. To get ensemble variation,
        # use random x_init per sample.
        xt = torch.randn_like(x_base)
        N = model.n_ode_steps
        for k in range(N):
            t_k = torch.full((B,), k / N, device=x_base.device)
            v_k = model.velocity(xt, x0_n, xT_n, x_base, t_k, tau_vfi)
            xt = xt + v_k / N
        x_final = x_base + s * xt
        samples.append(x_final)
    return torch.stack(samples, dim=0), x_base


# ---------------- S-DYff MC-dropout mode ----------------

def load_sdyff_lit(ckpt_path, channel_groups, device):
    """Load S-DYff via WeatherHermiteLightningModule. Returns wrapped model.

    For ensemble sampling we'll set the inner model to .train() mode (MC dropout).
    """
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(state.get("hyper_parameters", {}))
    hparams["channel_groups"] = channel_groups
    for k in ("_class_path", "model_type_save"):
        hparams.pop(k, None)
    model = WeatherHermiteLightningModule(**hparams)
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(device).eval()
    return model


def _set_dropout_train(module, train_mode=True):
    """Toggle dropout layers to train mode (for MC dropout) while keeping rest in eval."""
    for m in module.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train(train_mode)
        # SDyff may use DropPath/timm-style stochastic depth — detect by classname
        cls_name = m.__class__.__name__.lower()
        if "droppath" in cls_name or "stochasticdepth" in cls_name:
            m.train(train_mode)


@torch.no_grad()
def sdyff_ensemble_predict(model, x0, xT, tau, cond, static, n_samples, n_ch_model):
    """N stochastic forward passes via MC-dropout."""
    # Reset all to eval, then re-enable dropout-only modules
    model.eval()
    _set_dropout_train(model, train_mode=True)
    samples = []
    for _ in range(n_samples):
        out = model(x0[:, :n_ch_model], xT[:, :n_ch_model], tau, cond, static)
        if isinstance(out, tuple):
            out = out[0]
        samples.append(out[:, :24])  # slice to 24 paper channels
    # Restore eval
    _set_dropout_train(model, train_mode=False)
    return torch.stack(samples, dim=0)


# ---------------- Main loop ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["corrdiff_fm", "sdyff_mc"])
    ap.add_argument("--ckpt", help="Lightning ckpt path (sdyff_mc mode)")
    ap.add_argument("--fm_ckpt", help="CorrDiff FM head ckpt (corrdiff_fm mode)")
    ap.add_argument("--base_ckpt", help="Frozen base ckpt (corrdiff_fm mode)")
    ap.add_argument("--n_ensemble", type=int, default=16)
    ap.add_argument("--memmap_dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--data_root", default=_REPO_ROOT)
    ap.add_argument("--year", type=int, default=2020)
    ap.add_argument("--batch_size", type=int, default=2,
                    help="Lower than non-ensemble — N samples held in memory.")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--samples_per_date", type=int, default=4)
    ap.add_argument("--out_rmse", required=True)
    ap.add_argument("--out_crps", required=True)
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--max_tau_hours", type=int, default=6,
                    help="Interpolation window in hours (6 or 12). Drives tau-normalization "
                         "and the set of eval hours (1..max_tau_hours-1).")
    args = ap.parse_args()

    # NOTE: HOURS_PER_TAU_UNIT is hardcoded to 6.0 in weather_time_interp/config.py
    # but reads WTI_HOURS_PER_TAU env var at import time. We import the dataset at
    # top-of-module, so the env var must be set in the shell before launching this
    # script (run_eval_ensemble_corrdiff_weatherdcae_12h.sh does this). We re-set
    # it here for safety in case downstream code re-reads it.
    os.environ["WTI_HOURS_PER_TAU"] = str(float(args.max_tau_hours))

    # Derive tau-hour list dynamically from the window.
    global TAU_HOURS
    TAU_HOURS = list(range(1, args.max_tau_hours))

    device = torch.device("cuda")
    data_root = Path(args.data_root)

    ds = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[args.year],
        max_tau_hours=args.max_tau_hours,
        samples_per_date=args.samples_per_date,
        train=False,
        eval_hours=list(range(1, args.max_tau_hours)),
        static_path=str(data_root / "data/static_features_0p5.pt"),
        stats_path=str(data_root / "data/json_stats_0p5.nc"),
        surface_stats_path=str(data_root / "data/surface_stats_0p5.json"),
    )
    print(f"test items: {len(ds)}", flush=True)

    H, W = 360, 720
    lw = lat_weights(H, device, torch.float32)
    stds_24 = get_channel_stds(
        str(data_root / "data/json_stats_0p5.nc"),
        str(data_root / "data/surface_stats_0p5.json"),
        CH_24,
    )

    # Load model
    if args.mode == "corrdiff_fm":
        assert args.fm_ckpt and args.base_ckpt
        model, n_ch_model, n_ode_steps = load_corrdiff_fm(args.fm_ckpt, args.base_ckpt, device)
        slice_24 = (n_ch_model == 24)  # already 24ch
        meta_kind = "corrdiff_fm_ensemble"
    else:
        # sdyff_mc
        assert args.ckpt
        model = load_sdyff_lit(args.ckpt, ds.channel_groups, device)
        n_ch_model = 27
        slice_24 = True
        meta_kind = "sdyff_mc_dropout_ensemble"

    # Accumulators (per tau, 24ch)
    sums_sq = {h: torch.zeros(24, dtype=torch.float64, device=device) for h in TAU_HOURS}
    sums_crps = {h: torch.zeros(24, dtype=torch.float64, device=device) for h in TAU_HOURS}
    counts = {h: 0 for h in TAU_HOURS}

    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    t0 = time.time()
    for i, batch in enumerate(loader):
        x0 = batch["x0"].to(device)
        xT = batch["x1"].to(device)
        target = batch["target"].to(device)
        tau = batch["tau"].to(device)
        tau_h = batch["tau_hour"].to(device).squeeze(-1)
        cond = torch.full((x0.shape[0], 1), float(args.max_tau_hours), dtype=torch.float32, device=device)
        static = batch.get("static")
        if static is not None:
            static = static.to(device)

        # 27ch -> 24ch slice for target
        target_24 = target[:, :24]

        with torch.no_grad():
            if args.mode == "corrdiff_fm":
                x0_n = x0[:, :n_ch_model]
                xT_n = xT[:, :n_ch_model]
                samples, x_base = corrdiff_ensemble_predict(
                    model, x0_n, xT_n, target_24, tau, static, args.n_ensemble,
                )
                # samples: (N, B, 24 or 27, H, W) → take first 24
                samples = samples[:, :, :24, :, :]
            else:
                samples = sdyff_ensemble_predict(
                    model, x0, xT, tau, cond, static, args.n_ensemble, n_ch_model,
                )
                # samples: (N, B, 24, H, W)

        # Ensemble mean for RMSE
        mean = samples.mean(dim=0)  # (B, 24, H, W)
        err_m = (mean - target_24) ** 2
        err_m_lw = (err_m * lw).sum(dim=(-2, -1)) / W  # (B, 24)

        # Fair CRPS per pixel, lat-weighted
        crps_pc = fair_crps_per_pixel(samples, target_24, lw)  # (B, 24)

        for k, h in enumerate(tau_h.tolist()):
            h = int(h)
            if h in sums_sq:
                sums_sq[h] += err_m_lw[k].double()
                sums_crps[h] += crps_pc[k].double()
                counts[h] += 1

        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  batch {i}/{len(loader)} t={elapsed:.0f}s ({elapsed/max(1,i+1):.1f}s/batch)", flush=True)

    # Finalize
    rmse_norm = []
    rmse_phys = []
    crps_norm = []
    crps_phys = []
    n_samp = []
    stds_np = stds_24.astype(np.float64)
    for h in TAU_HOURS:
        n = counts[h]
        n_samp.append(n)
        if n == 0:
            rmse_norm.append([float("nan")] * 24)
            rmse_phys.append([float("nan")] * 24)
            crps_norm.append([float("nan")] * 24)
            crps_phys.append([float("nan")] * 24)
            continue
        rmse_n = (sums_sq[h] / n).sqrt().cpu().numpy()
        crps_n = (sums_crps[h] / n).cpu().numpy()
        rmse_norm.append([float(x) for x in rmse_n])
        rmse_phys.append([float(x * s) for x, s in zip(rmse_n, stds_np)])
        crps_norm.append([float(x) for x in crps_n])
        crps_phys.append([float(x * s) for x, s in zip(crps_n, stds_np)])

    rmse_payload = {
        "model_name": args.model_name + "_ensN" + str(args.n_ensemble),
        "kind": meta_kind,
        "ckpt": args.fm_ckpt if args.mode == "corrdiff_fm" else args.ckpt,
        "base_ckpt": args.base_ckpt if args.mode == "corrdiff_fm" else None,
        "n_channels": 24,
        "channels": CH_24,
        "tau_hours": TAU_HOURS,
        "n_ensemble": args.n_ensemble,
        "rmse_model_norm": rmse_norm,
        "rmse_model_phys": rmse_phys,
        "n_samples_per_tau": n_samp,
        "stds_for_denorm": stds_np.tolist(),
        "units": "normalized (mean=0, std=1), ensemble-mean lat-weighted RMSE",
    }
    crps_payload = {
        "model_name": args.model_name + "_ensN" + str(args.n_ensemble),
        "kind": meta_kind,
        "ckpt": args.fm_ckpt if args.mode == "corrdiff_fm" else args.ckpt,
        "base_ckpt": args.base_ckpt if args.mode == "corrdiff_fm" else None,
        "n_channels": 24,
        "channels": CH_24,
        "tau_hours": TAU_HOURS,
        "n_ensemble": args.n_ensemble,
        "crps_norm": crps_norm,
        "crps_phys": crps_phys,
        "n_samples_per_tau": n_samp,
        "stds_for_denorm": stds_np.tolist(),
        "estimator": "fair_hersbach",
        "units": "normalized (mean=0, std=1), lat-weighted CRPS per pixel averaged over (H,W)",
    }
    Path(args.out_rmse).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_crps).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_rmse).write_text(json.dumps(rmse_payload, indent=2))
    Path(args.out_crps).write_text(json.dumps(crps_payload, indent=2))
    print(f"\nsaved {args.out_rmse}")
    print(f"saved {args.out_crps}")
    mid_idx = len(TAU_HOURS) // 2
    mid_h = TAU_HOURS[mid_idx]
    print(f"tau={mid_h} RMSE_norm mean: {np.mean(rmse_norm[mid_idx]):.4f}")
    print(f"tau={mid_h} CRPS_norm mean: {np.mean(crps_norm[mid_idx]):.4f}")
    print(f"runtime: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
