"""CorrDiff-FM angular power spectra for 12 h interpolation.

Same on-grid SHT pipeline as ``tools/eval/sh_energy_spectra_12h.py`` —
but for a stochastic CorrDiff-FM model (n_ensemble=1 sample per call,
because we already average over hundreds of timesteps per τ).

Outputs an npz with the SAME schema as ``sh_energy_spectra_12h.py``:
  ell, pred_El, gt_El, channel_names, n_samples, tau, H, W, lmax,
  model_name, ckpt.

So the existing ``scripts/make_fig3_spectra.py`` consumes it without
changes — just add the stem to ``MODELS``.

Run on fibo via the same wti-train:v1 image used by sh_energy_spectra_12h.
"""
from __future__ import annotations
import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Repo root — script lives in tools/eval/.
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _REPO_ROOT)

from weather_time_interp.memmap_dataset import ERA5MemmapDataset  # noqa: E402
from trainer_weather_hermite import ERA5WeatherHermiteDataset  # noqa: E402

import torch_harmonics as th  # noqa: E402


def _build_sht(H: int, W: int, lmax: int, device: torch.device):
    return th.RealSHT(H, W, lmax=lmax + 1, mmax=lmax + 1,
                      grid="equiangular").to(device)


@torch.no_grad()
def _angular_power(field: torch.Tensor, sht) -> torch.Tensor:
    """field: (B, C, H, W) → (C, lmax+1) real angular power."""
    coeffs = sht(field.double())          # (B, C, lmax+1, mmax+1) complex
    p = (coeffs.real ** 2 + coeffs.imag ** 2)
    return p.sum(dim=-1).sum(dim=0)        # sum over m and batch


def _select_channel_indices(channel_names, sel):
    if sel == "all" or sel == ["all"]:
        return list(range(len(channel_names)))
    if isinstance(sel, str):
        sel = [c.strip() for c in sel.split(",") if c.strip()]
    return [channel_names.index(c) for c in sel if c in channel_names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", required=True)
    ap.add_argument("--fm-ckpt", required=True)
    ap.add_argument("--base-ckpt", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--taus", default="2,3")
    ap.add_argument("--lmax", type=int, default=180)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-tau-hours", type=int, default=12)
    ap.add_argument("--samples-per-date", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=200,
                    help="Cap total batches (for shared-GPU politeness).")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--keep-n-channels", type=int, default=24)
    ap.add_argument("--channels", default="all")
    ap.add_argument("--ode-steps", type=int, default=None)
    ap.add_argument("--sampler", default="euler", choices=["euler", "heun"])
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    taus = sorted({int(x) for x in args.taus.split(",") if x.strip()})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing:
        present = [t for t in taus
                   if (out_dir / f"{args.model_name}_tau{t}.npz").exists()]
        taus_todo = [t for t in taus if t not in present]
        if not taus_todo:
            print(f"  [skip-existing] all τ done for {args.model_name}")
            return
        taus = taus_todo

    # Dataset — same wiring as sh_energy_spectra_12h.py.
    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir,
        years=[2020], max_tau_hours=args.max_tau_hours,
        samples_per_date=args.samples_per_date, train=False,
    )
    wrapped = ERA5WeatherHermiteDataset(
        ds_base, delta_t_hours=float(args.max_tau_hours))
    loader = DataLoader(wrapped, batch_size=args.batch_size, num_workers=0,
                        shuffle=False, pin_memory=True)

    channel_names = list(ds_base.channel_names)[: args.keep_n_channels]
    sel_idx = _select_channel_indices(channel_names, args.channels)
    print(f"  channels: {len(channel_names)}, selected: {len(sel_idx)}")

    # Load CorrDiff-FM via the canonical eval-side loader.
    from tools.eval.eval_ensemble_crps import (load_corrdiff_fm,
                                                corrdiff_ensemble_predict)
    model, n_ch_model, n_ode_steps = load_corrdiff_fm(
        args.fm_ckpt, args.base_ckpt, device)
    ode_steps = args.ode_steps if args.ode_steps is not None else n_ode_steps
    print(f"  CorrDiff loaded: n_ch={n_ch_model}, ode_steps={ode_steps}, "
          f"sampler={args.sampler}")

    sample = next(iter(loader))
    H, W = sample["x0"].shape[-2:]
    print(f"  H×W = {H}×{W}; lmax={args.lmax}")
    sht = _build_sht(H, W, args.lmax, device)
    lmax_plus_1 = args.lmax + 1

    pred_acc = {t: torch.zeros(len(sel_idx), lmax_plus_1, dtype=torch.float64,
                                device=device) for t in taus}
    gt_acc = {t: torch.zeros(len(sel_idx), lmax_plus_1, dtype=torch.float64,
                              device=device) for t in taus}
    n_per_tau = {t: 0 for t in taus}
    max_tau = float(args.max_tau_hours)

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= args.max_batches:
                print(f"  reached max-batches={args.max_batches}, stopping")
                break
            x0 = batch["x0"].to(device, non_blocking=True)
            xT = batch["xT"].to(device, non_blocking=True)
            tau_hour_all = batch["tau_hour"].long()
            target_all = batch["target"].to(device, non_blocking=True)
            static = batch.get("static")
            if static is not None:
                static = static.to(device, non_blocking=True)
            K = args.keep_n_channels
            x0 = x0[:, :K].contiguous()
            xT = xT[:, :K].contiguous()
            target_all = target_all[:, :, :K].contiguous()
            B = x0.size(0)
            nH = tau_hour_all.size(1)

            for h_idx in range(nH):
                tau_h_int = tau_hour_all[:, h_idx, 0]
                tau_norm = tau_h_int.float().to(device) / max_tau
                target_h = target_all[:, h_idx]

                # CorrDiff stochastic sample (n_ensemble=1).
                pred_N = corrdiff_ensemble_predict(
                    model, x0, xT, target_h, tau_norm, static,
                    n_samples=1, ode_steps=ode_steps, sampler=args.sampler,
                    sharpen_gamma=0.0, ensemble_chunk=1)
                pred = pred_N[0]   # (B, C, H, W)

                pred_sel = pred[:, sel_idx]
                tgt_sel = target_h[:, sel_idx]
                for i in range(B):
                    tau_i = int(tau_h_int[i].item())
                    if tau_i not in pred_acc:
                        continue
                    pred_acc[tau_i] += _angular_power(pred_sel[i:i+1], sht)
                    gt_acc[tau_i] += _angular_power(tgt_sel[i:i+1], sht)
                    n_per_tau[tau_i] += 1
            if bi % 10 == 0:
                print(f"  batch {bi}/{min(args.max_batches, len(loader))}",
                      flush=True)

    ell = np.arange(lmax_plus_1, dtype=np.int32)
    sel_names = [channel_names[i] for i in sel_idx]
    for tau in taus:
        n = max(1, n_per_tau[tau])
        pred_arr = (pred_acc[tau] / n).cpu().numpy().astype(np.float64)
        gt_arr = (gt_acc[tau] / n).cpu().numpy().astype(np.float64)
        out_path = out_dir / f"{args.model_name}_tau{tau}.npz"
        np.savez(out_path,
                 ell=ell, pred_El=pred_arr, gt_El=gt_arr,
                 channel_names=np.array(sel_names, dtype=object),
                 n_samples=int(n_per_tau[tau]), tau=int(tau),
                 H=int(H), W=int(W), lmax=int(args.lmax),
                 model_name=args.model_name,
                 ckpt=os.path.basename(args.fm_ckpt))
        print(f"  wrote {out_path}  (n={n_per_tau[tau]})")


if __name__ == "__main__":
    main()
