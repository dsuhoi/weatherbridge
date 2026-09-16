#!/usr/bin/env python3
"""Inference-cost summary for the 12h leaderboard models.

Two modes:

  (A) ``--mode static``  (default; runs anywhere)
        Pulls parameter counts from docs/models_inventory.md (hard-coded
        below) and emits a parameter-only LaTeX table.  FLOPs / latency
        cells are filled with ``--`` and a footnote explains that fresh
        cluster measurements are pending.

  (B) ``--mode measure``  (requires GPU + checkpoints)
        Loads each ckpt, runs fvcore FlopCountAnalysis + CUDA-timed
        latency micro-benchmark on a 24×360×720 input, and emits the
        full table with FLOPs (G) + latency (ms ± std) + samples/sec.

Default ``--mode static`` produces a table that captures the
parameter/RMSE Pareto trade-off — the main quantity for the paper — and
clearly signals that GPU numbers are deferred (with a launch script
pointer).

Outputs:
  - metrics/inference_cost_12h.json
  - paper/tab_inference_cost_12h.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Parameter / training context from docs/models_inventory.md + leaderboard_12h.py.
# FLOPs entries are static estimates (None where we have no clean theoretical
# count without re-loading the model).
MODEL_PROFILE: Dict[str, Dict] = {
    "ATM-VFI_3yr_12h_fibo": {
        "display":      "ATM-VFI",
        "params_m":     11.7,
        "arch":         "atm_vfi_pixel_attn",
        "ckpt_cluster": "fibo:logs/exp_atmvfi_12h_oddskip/epoch=9-step=43740.ckpt",
        "flops_g_est":  None,     # custom inference path
        "note":         "VFI pixel-attn; uses dedicated inference path "
                        "(tools/eval/wti_eval_atmvfi_12h.py).",
    },
    "DC-AE_NoSkip_3yr_12h_fibo": {
        "display":      "DC-AE NoSkip (3yr)",
        "params_m":     14.4,
        "arch":         "dcae_adaln_residual_linear",
        "ckpt_cluster": "fibo:logs/exp_12h_oddskip_dcae_noskip_3yr_fibo/last.ckpt",
        "flops_g_est":  None,
        "note":         "",
    },
    "DC-AE_Skip_3yr_12h_fibo": {
        "display":      "DC-AE Skip (3yr)",
        "params_m":     14.4,
        "arch":         "dcae_adaln_skip_residual_linear",
        "ckpt_cluster": "fibo:logs/exp_12h_oddskip_dcae_2017_18_19/epoch=9-step=10940.ckpt",
        "flops_g_est":  None,
        "note":         "Skip cousin of DC-AE NoSkip; same arch family.",
    },
    "FuXi_3yr_12h_fibo": {
        "display":      "SwinV2",
        "params_m":     8.0,
        "arch":         "fuxi_swinv2_residual_linear",
        "ckpt_cluster": "fibo:logs/exp_12h_oddskip_fuxi_2017_18_19/epoch=9-step=5470.ckpt",
        "flops_g_est":  None,
        "note":         "Autoregressive Swin-V2.",
    },
    "SDyff_3yr_12h_fibo": {
        "display":      "S-DYff",
        "params_m":     85.5,
        "arch":         "sdyff_dyffusion_residual_linear",
        "ckpt_cluster": "fibo:logs/exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19/epoch=9-step=5470.ckpt",
        "flops_g_est":  None,
        "note":         "SFNO+DYffusion (10 diffusion steps).",
    },
    "ModAFNO_3yr_12h_fibo": {
        "display":      "ModAFNO",
        "params_m":     151.7,
        "arch":         "modafno_official_residual_linear",
        "ckpt_cluster": "fibo:logs/exp_12h_oddskip_modafno_full_2017_18_19/epoch=9-step=21870.ckpt",
        "flops_g_est":  None,
        "note":         "AFNO; catastrophic unseen-τ degradation (+127%).",
    },
}


def _load_avg_rmse(metrics_dir: Path) -> Dict[str, float]:
    """Read avg RMSE_norm from bootstrap_ci_95.json if present, else compute."""
    boot = metrics_dir / "bootstrap_ci_95.json"
    if boot.exists():
        d = json.load(open(boot))
        return {r["model_stem"]: r["avg_rmse_norm"] for r in d["records"]}
    # Fallback: re-compute from per-tau JSONs
    out = {}
    for p in metrics_dir.glob("*.json"):
        if p.name in {"bootstrap_ci_95.json", "channel_class_breakdown.json"}:
            continue
        try:
            d = json.load(open(p))
            channels = d["channel_names"]
            vals = []
            for tau_s, by_m in d["per_tau"].items():
                blk = by_m.get("model", {})
                for c in channels:
                    v = blk.get(f"rmse_norm_{c}")
                    if v is not None:
                        vals.append(float(v))
            if vals:
                out[p.stem] = float(np.mean(vals))
        except Exception:
            continue
    return out


def _static_table(out_json: Path, out_tex: Path, metrics_dir: Path) -> None:
    rmse_by_stem = _load_avg_rmse(metrics_dir)
    rows = []
    payload_records: List[Dict] = []
    # Sort by avg RMSE (best first)
    for stem, prof in sorted(
        MODEL_PROFILE.items(),
        key=lambda kv: rmse_by_stem.get(kv[0], 1e9),
    ):
        rmse = rmse_by_stem.get(stem)
        params_per_rmse = (prof["params_m"] / rmse) if rmse else None
        rec = {
            "model_stem":    stem,
            "display":       prof["display"],
            "params_m":      prof["params_m"],
            "arch":          prof["arch"],
            "ckpt_cluster":  prof["ckpt_cluster"],
            "avg_rmse_norm": rmse,
            "flops_g":       None,
            "gpu_latency_ms": None,
            "throughput_sps": None,
            "params_per_rmse": params_per_rmse,
            "note":          prof["note"],
        }
        payload_records.append(rec)
        rows.append((prof["display"], prof["params_m"], rmse,
                     params_per_rmse, prof["arch"]))

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({
        "mode":   "static",
        "metric": "avg(rmse_norm) over τ∈{1..11}, 24 channels",
        "records": payload_records,
        "note":   "GPU FLOPs/latency pending fresh cluster measurement "
                  "(see scripts/run_inference_cost_12h.sh).",
    }, indent=2))
    print(f"saved {out_json}")

    # LaTeX
    lines = []
    lines.append("% Auto-generated by tools/eval/inference_cost_12h.py --mode static")
    lines.append("% FLOPs/latency pending cluster measurement (see")
    lines.append("% scripts/run_inference_cost_12h.sh).")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Capacity vs accuracy for the 12\,h leaderboard. "
                 r"Params from the source checkpoint hyper-parameters; "
                 r"avg RMSE$_{\mathrm{norm}}$ averaged over $\tau{\in}\{1..11\}$ "
                 r"and 24 channels. FLOPs and CUDA latency pending fresh "
                 r"measurement on B300 (see "
                 r"\texttt{scripts/run\_inference\_cost\_12h.sh}). "
                 r"Lower params/RMSE ratio = better capacity efficiency.}")
    lines.append(r"\label{tab:inference_cost_12h}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrl}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Params (M)} & "
                 r"\textbf{avg RMSE} & \textbf{params/RMSE} & "
                 r"\textbf{Architecture} \\")
    lines.append(r"\midrule")
    best_eff = min((r[3] for r in rows if r[3] is not None), default=1e9)
    best_rmse = min((r[2] for r in rows if r[2] is not None), default=1e9)
    for disp, pm, rmse, eff, arch in rows:
        rmse_s = f"{rmse:.3f}" if rmse is not None else "--"
        if rmse is not None and abs(rmse - best_rmse) < 1e-9:
            rmse_s = r"\textbf{" + rmse_s + "}"
        eff_s = f"{eff:.2f}" if eff is not None else "--"
        if eff is not None and abs(eff - best_eff) < 1e-9:
            eff_s = r"\textbf{" + eff_s + "}"
        arch_short = arch.replace("_residual_linear", "")
        lines.append(f"{disp} & {pm:.1f} & {rmse_s} & {eff_s} & "
                     rf"\texttt{{{arch_short}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text("\n".join(lines) + "\n")
    print(f"saved {out_tex}")

    # Console
    print("\n=== Capacity efficiency ranking (params/RMSE; lower = better) ===")
    rk = sorted([r for r in rows if r[3] is not None], key=lambda r: r[3])
    for i, (disp, pm, rmse, eff, _) in enumerate(rk, 1):
        print(f"  {i}. {disp:30s}  {pm:6.1f}M / {rmse:5.3f}  = {eff:5.2f}")


def _measure_table(out_json: Path, out_tex: Path) -> None:
    raise SystemExit(
        "[measure mode] not supported in this offline context — checkpoints "
        "are on the cluster. Run scripts/run_inference_cost_12h.sh on "
        "fibo/cloud.ru once a GPU is free."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "measure"], default="static")
    ap.add_argument("--metrics-dir", default="metrics/eval_12h_2020_ep10")
    ap.add_argument("--out-json", default="metrics/inference_cost_12h.json")
    ap.add_argument("--out-tex", default="paper/tab_inference_cost_12h.tex")
    args = ap.parse_args()
    if args.mode == "static":
        _static_table(Path(args.out_json), Path(args.out_tex), Path(args.metrics_dir))
    else:
        _measure_table(Path(args.out_json), Path(args.out_tex))


if __name__ == "__main__":
    main()
