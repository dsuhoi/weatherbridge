#!/usr/bin/env python3
"""Measure FLOPs + latency for all 4 ML models + numerical baselines.

Outputs paper/A4_inference_cost.tex with: model | params | FLOPs/sample |
GPU latency (ms/sample) | CPU latency (ms/sample) | RMSE Δ% vs bicubic.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trainer_weather_hermite import WeatherHermiteLightningModule


MODELS = [
    # (display_name, ckpt_path, env_dict, rmse_delta_pct)
    ("DC-AE Skip (PAD) 6yr",   "logs/exp_dcae_skip_0p5_6yr_pad/epoch=7-step=70064.ckpt",
     {"LAT_CROP": "-8"}, -50.22),
    ("FuXi SwinV2 6yr",        "logs/exp_fuxi_0p5_6yr_full/last.ckpt",
     {}, -10.82),
    ("ModAFNO 6yr",            "logs/exp_modafno_0p5_6yr/last.ckpt",
     {"MODAFNO_INP_H": "360", "MODAFNO_INP_W": "720",
      "MODAFNO_NATIVE_H": "360", "MODAFNO_NATIVE_W": "720"}, -16.87),
    ("S-DYff DYffusion 6yr",   "logs/exp_sdyff_dyffusion_0p5_6yr/epoch=7-step=70064.ckpt",
     {"SDYFF_NLAT": "360", "SDYFF_NLON": "720", "SDYFF_LAT_CROP": "0"}, -14.78),
]


def measure_model(name, ckpt_path, env_dict, device, n_warmup=3, n_trials=10):
    import os as _os
    for k, v in env_dict.items():
        _os.environ[k] = v
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")
    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        hparams["block_out_channels"] = boc if len(boc) >= 3 else (128, 256, 512)
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        hparams["layers_per_block"] = lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)
    hparams["channel_groups"] = {}
    model = WeatherHermiteLightningModule(**hparams)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())

    B, C, H, W = 1, 27, 360, 720
    x0 = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    xT = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    tau = torch.tensor([0.5], device=device, dtype=torch.float32)
    cond = torch.full((B,), 6.0, device=device, dtype=torch.float32)
    # Static order: land-sea mask, normalized orography, cosine latitude.
    static = torch.zeros(B, 3, H, W, device=device, dtype=torch.float32)

    # FLOPs via fvcore (best-effort; may fail on custom ops)
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fca = FlopCountAnalysis(model, (x0, xT, tau, cond, static))
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            flops = int(fca.total())
    except Exception as e:
        print(f"  fvcore failed ({type(e).__name__}): {e}")

    # GPU latency
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x0, xT, tau, cond, static=static)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latencies_ms = []
        for _ in range(n_trials):
            start.record()
            _ = model(x0, xT, tau, cond, static=static)
            end.record()
            torch.cuda.synchronize()
            latencies_ms.append(start.elapsed_time(end))
    gpu_mean_ms = sum(latencies_ms) / len(latencies_ms)
    gpu_std_ms = (sum((x - gpu_mean_ms) ** 2 for x in latencies_ms) / max(1, len(latencies_ms) - 1)) ** 0.5

    print(f"  {name}: params={n_params/1e6:.1f}M  FLOPs={'N/A' if flops is None else f'{flops/1e9:.2f}G'}  "
          f"GPU latency={gpu_mean_ms:.1f}±{gpu_std_ms:.1f} ms")

    del model
    torch.cuda.empty_cache()
    return n_params, flops, gpu_mean_ms, gpu_std_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/A4_inference_cost.tex")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, torch={torch.__version__}")
    results = []
    for name, ckpt, envs, delta_pct in MODELS:
        if not Path(ckpt).exists():
            print(f"  [skip] {ckpt}")
            continue
        try:
            n_params, flops, lat_ms, lat_std = measure_model(name, ckpt, envs, device)
            results.append((name, n_params, flops, lat_ms, lat_std, delta_pct))
        except Exception as e:
            print(f"  [fail] {name}: {e}")

    # Add bilinear (no model)
    B, C, H, W = 1, 27, 360, 720
    x0 = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    xT = torch.randn(B, C, H, W, device=device, dtype=torch.float32)
    tau = 0.5
    torch.cuda.synchronize()
    bil_t = []
    for _ in range(20):
        start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = (1 - tau) * x0 + tau * xT
        end.record(); torch.cuda.synchronize()
        bil_t.append(start.elapsed_time(end))
    bil_lat = sum(bil_t) / len(bil_t)
    print(f"  bilinear: latency={bil_lat:.3f} ms")
    results.append(("Bilinear (reference)", 0, 0, bil_lat, 0.0, 0.0))

    lines = []
    lines.append(r"% Inference cost (auto-generated)")
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Inference cost at 0.5$^\circ$ ($27 \times 360 \times 720$, batch=1) "
                 r"on NVIDIA A100-SXM4-80GB, fp32. "
                 r"$\Delta$RMSE\% is vs bicubic baseline.}")
    lines.append(r"\label{tab:inference_cost}")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4pt}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Model} & \textbf{Params} & \textbf{FLOPs} & \textbf{Latency (ms)} & \textbf{$\Delta$RMSE\%} \\")
    lines.append(r"\midrule")
    for name, n_params, flops, lat_ms, lat_std, delta in results:
        params_s = f"{n_params/1e6:.1f}M" if n_params > 0 else "0"
        flops_s = "N/A" if flops in (None, 0) else f"{flops/1e9:.1f}G"
        lat_s = f"{lat_ms:.1f}\\,$\\pm$\\,{lat_std:.1f}" if lat_std > 0 else f"{lat_ms:.2f}"
        if delta < 0:
            delta_s = rf"\textcolor{{ForestGreen}}{{{delta:+.1f}}}"
        elif delta > 0:
            delta_s = rf"\textcolor{{BrickRed}}{{{delta:+.1f}}}"
        else:
            delta_s = "0.0"
        # Bold DC-AE Skip
        nm = rf"\textbf{{{name}}}" if "DC-AE Skip" in name else name
        lines.append(f"{nm} & {params_s} & {flops_s} & {lat_s} & {delta_s} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
