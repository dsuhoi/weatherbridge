"""Plot SH angular power spectra E(ℓ) for the 12h paper, all models on
the same axes per channel, one mosaic per τ.

Reads .npz files from ``metrics/sh_spectra_12h_ep10/`` (output of
``sh_energy_spectra_12h.py``) and overlays them per channel, with the
ground-truth (ERA5) curve plotted once in black.

Output: ``figs/fig_sh_spectra_12h_ep10.{png,pdf}`` — one mosaic with
(rows × cols) = (n_taus × n_channels) panels.

Also prints a small HF-preservation table per channel/τ:

    HF energy ratio = sum_{ℓ ≥ ℓ_lo} E_pred(ℓ) / sum_{ℓ ≥ ℓ_lo} E_gt(ℓ)

with ``ℓ_lo = lmax // 3`` (default). A ratio close to 1 means the model
preserves high-wavenumber power; values << 1 indicate spectral truncation
("spectral bias").
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Default model display info (per-model line style overrides).
MODEL_STYLE = {
    "DC-AE_NoSkip_3yr_12h_fibo":      ("DC-AE NoSkip 3yr",     "#1b9e77", "-",  1.6),
    "DC-AE_Skip_3yr_12h_fibo":        ("DC-AE Skip 3yr",       "#66a61e", "-",  1.6),
    "FuXi_3yr_12h_fibo":              ("FuXi 3yr",             "#e7298a", "--", 1.4),
    "ModAFNO_3yr_12h_fibo":           ("ModAFNO 3yr",          "#d95f02", ":",  1.4),
    "SDyff_3yr_12h_fibo":             ("S-DYff 3yr",           "#7570b3", ":",  1.4),
    "ATM-VFI_3yr_12h_fibo":           ("ATM-VFI 3yr",          "#e6ab02", "-.", 1.4),
    "WeatherDCAE_NoSkip_6yr_12h":     ("WeatherDCAE 6yr (ours)", "#d62728", "-",  2.2),
}


def _load_dir(npz_dir: Path) -> Dict[Tuple[str, int], dict]:
    """Group .npz files by (model_name, tau)."""
    out: Dict[Tuple[str, int], dict] = {}
    for p in sorted(npz_dir.glob("*.npz")):
        d = np.load(p, allow_pickle=True)
        model_name = str(d["model_name"]) if "model_name" in d.files else p.stem
        tau = int(d["tau"]) if "tau" in d.files else -1
        out[(model_name, tau)] = dict(
            ell=d["ell"], pred_El=d["pred_El"], gt_El=d["gt_El"],
            channels=[str(c) for c in d["channel_names"]],
            n=int(d["n_samples"]),
        )
    return out


def _hf_ratio(pred_El: np.ndarray, gt_El: np.ndarray,
              ell_lo: int) -> np.ndarray:
    """Per-channel HF energy ratio sum_{ℓ ≥ ℓ_lo} E_pred / sum E_gt."""
    p = pred_El[:, ell_lo:].sum(axis=1)
    g = np.clip(gt_El[:, ell_lo:].sum(axis=1), 1e-30, None)
    return p / g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz-dir", default="metrics/sh_spectra_12h_ep10")
    ap.add_argument("--out", default="figs/fig_sh_spectra_12h_ep10")
    ap.add_argument("--taus", default="2,3,5,8")
    ap.add_argument("--channels", default="t2m,mslp,u10,T850")
    ap.add_argument("--hf-frac", type=float, default=0.5,
                    help="ℓ_lo = floor(hf_frac × lmax) for HF-preservation ratio.")
    args = ap.parse_args()

    npz_dir = Path(args.npz_dir)
    if not npz_dir.is_dir():
        raise SystemExit(f"--npz-dir not found: {npz_dir}")
    data = _load_dir(npz_dir)
    if not data:
        raise SystemExit(f"no .npz in {npz_dir}")

    taus = sorted({int(t) for t in args.taus.split(",")})
    channels = [c.strip() for c in args.channels.split(",")]
    n_taus = len(taus)
    n_chs = len(channels)

    fig, axes = plt.subplots(n_taus, n_chs, figsize=(3.6 * n_chs, 3.0 * n_taus),
                              squeeze=False)
    legend_labels: List[Tuple[str, str, str, float]] = []
    legend_done = False

    print("\n=== HF-preservation summary (ratio = ΣE_pred / ΣE_gt for ℓ ≥ ℓ_lo) ===")
    for ti, tau in enumerate(taus):
        for ci, ch_name in enumerate(channels):
            ax = axes[ti, ci]
            gt_drawn = False
            for (model_name, t), d in data.items():
                if t != tau:
                    continue
                if ch_name not in d["channels"]:
                    continue
                idx = d["channels"].index(ch_name)
                ell = d["ell"]
                pred = d["pred_El"][idx]
                gt = d["gt_El"][idx]
                disp, color, ls, lw = MODEL_STYLE.get(
                    model_name, (model_name, "#888888", "-", 1.0)
                )
                m = (ell > 0) & (pred > 0)
                ax.loglog(ell[m], pred[m], color=color, ls=ls, lw=lw, label=disp)
                if not gt_drawn:
                    mg = (ell > 0) & (gt > 0)
                    ax.loglog(ell[mg], gt[mg], color="black", ls="-", lw=1.4,
                              label="ERA5 (GT)")
                    gt_drawn = True
                if not legend_done:
                    legend_labels.append((disp, color, ls, lw))

                lmax = int(ell.max())
                ell_lo = max(1, int(args.hf_frac * lmax))
                ratio = _hf_ratio(d["pred_El"], d["gt_El"], ell_lo)[idx]
                print(f"  τ={tau:>2} {ch_name:>5s} {disp:>26s}: "
                      f"HF ratio (ℓ≥{ell_lo}) = {ratio:.3f}")
            ax.set_title(f"τ={tau}h — {ch_name}", fontsize=9)
            ax.grid(True, which="both", alpha=0.25)
            if ti == n_taus - 1:
                ax.set_xlabel("ℓ")
            if ci == 0:
                ax.set_ylabel("E(ℓ)")
            legend_done = True

    # One shared legend at the figure bottom.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)),
                   fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Angular SH power spectra E(ℓ) — 12h interpolation, ep10 ckpts",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out}.png")
    print(f"wrote {out}.pdf")


if __name__ == "__main__":
    main()
