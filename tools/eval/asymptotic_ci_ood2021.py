#!/usr/bin/env python3
r"""Asymptotic 95% CI for per-tau, per-channel RMSE on OOD 2021 eval JSONs.

Per-window MSE was not logged during the OOD 2021 eval (the existing JSONs
in ``metrics/eval_0p5_2021_ood/`` only carry the aggregate ``rmse_<CH>``
under each ``per_hour[tau].{model,bilinear,bicubic}``). A proper bootstrap
would require re-running the eval with per-window output; since RMSE over
N = 960 windows is well-approximated by a Gaussian sampling distribution,
we instead report the *asymptotic* delta-method 95% CI.

Derivation
==========

If the per-window squared errors :math:`Y_i = \|f(x_i) - y_i\|^2` are roughly
iid with finite variance, then

    Var(MSE) ≈ Var(Y) / N.

For approximately Gaussian residuals the standard result is
:math:`\mathrm{Var}(Y) \approx 2 \cdot \mathrm{MSE}^2`, giving

    SE(MSE) ≈ MSE · sqrt(2 / N).

Applying the delta method to :math:`g(\mathrm{MSE}) = \sqrt{\mathrm{MSE}}`:

    SE(RMSE) = SE(MSE) / (2 · RMSE) = RMSE / sqrt(2N).

So the asymptotic 95% half-width is

    half_width = 1.96 · RMSE / sqrt(2 · N)
               ≈ 0.0447 · RMSE   for N = 960.

This is an honest estimate of credibility for a single (τ, channel) cell;
it cannot capture cross-channel correlations the way a stratified bootstrap
on per-window cubes would. We label the band as
"±1.96·SE asymptotic CI" in figure captions to make this explicit.

Output schema (per JSON file, mirrors input):
::

    {
      "schema_version": 1,
      "ci_method": "asymptotic_normal_delta_method",
      "ci_alpha": 0.05,
      "num_samples": 960,
      "channel_names": [...],
      "per_hour": {
        "1": {
          "model": {
            "rmse_<CH>":          <point estimate, copy>,
            "rmse_<CH>_ci_low":   <point - 1.96 · rmse / sqrt(2N)>,
            "rmse_<CH>_ci_high":  <point + 1.96 · rmse / sqrt(2N)>,
            ...
          },
          "bilinear": {... same triple ...}
        },
        ...
      }
    }

Usage::

    python tools/eval/asymptotic_ci_ood2021.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_IN = ROOT / "metrics" / "eval_0p5_2021_ood"
DEFAULT_OUT = ROOT / "metrics" / "eval_0p5_2021_ood_ci"

ALPHA = 0.05
Z_975 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)


def _ci_block(point_block: Dict[str, float], n: int) -> Dict[str, float]:
    """Return a dict that mirrors point_block plus ci_low/ci_high siblings."""
    out: Dict[str, float] = {}
    factor = Z_975 / math.sqrt(2.0 * n)
    for k, v in point_block.items():
        out[k] = v
        if not isinstance(v, (int, float)):
            continue
        if k.startswith("rmse_") and not k.endswith(("_ci_low", "_ci_high")):
            hw = factor * float(v)
            out[f"{k}_ci_low"] = max(float(v) - hw, 0.0)
            out[f"{k}_ci_high"] = float(v) + hw
    return out


def annotate_one(path_in: Path, path_out: Path) -> Dict[str, Any]:
    d = json.load(open(path_in))
    n = int(d.get("num_samples", 0))
    if n <= 0:
        raise ValueError(f"{path_in}: missing or non-positive num_samples")
    out = {
        "schema_version": 1,
        "source_json": str(path_in.relative_to(ROOT)),
        "ci_method": "asymptotic_normal_delta_method",
        "ci_formula": "half_width = z_0.975 * rmse / sqrt(2 * N)",
        "ci_alpha": ALPHA,
        "num_samples": n,
        "channel_names": d.get("channel_names", []),
        "years": d.get("years", []),
        "per_hour": {},
    }
    for tau_key, tau_block in d.get("per_hour", {}).items():
        new_tau: Dict[str, Any] = {}
        for sub in ("model", "bilinear", "bicubic"):
            if sub in tau_block and isinstance(tau_block[sub], dict):
                new_tau[sub] = _ci_block(tau_block[sub], n)
        out["per_hour"][tau_key] = new_tau
    path_out.parent.mkdir(parents=True, exist_ok=True)
    path_out.write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    paths = sorted(args.in_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no JSONs in {args.in_dir}")
    print(f"[asymptotic_ci_ood2021] {len(paths)} JSONs from {args.in_dir}")
    print(f"  z_0.975 = {Z_975:.6f}")
    for p in paths:
        outp = args.out_dir / p.name
        rec = annotate_one(p, outp)
        n = rec["num_samples"]
        hw_pct = 100.0 * Z_975 / math.sqrt(2.0 * n)
        # Pick a diagnostic cell: t2m at tau=3 for the model
        try:
            t2m_rmse = rec["per_hour"]["3"]["model"]["rmse_t2m"]
            t2m_lo = rec["per_hour"]["3"]["model"]["rmse_t2m_ci_low"]
            t2m_hi = rec["per_hour"]["3"]["model"]["rmse_t2m_ci_high"]
            diag = (f"  t2m@τ=3: {t2m_rmse:.4f}  CI95=[{t2m_lo:.4f},"
                    f" {t2m_hi:.4f}]  (±{hw_pct:.2f}%)")
        except KeyError:
            diag = f"  (±{hw_pct:.2f}% width, N={n})"
        print(f"  {p.stem:50s}  ->  {outp.relative_to(ROOT)}\n{diag}")
    print(f"[done] wrote {len(paths)} CI JSONs to {args.out_dir}")


if __name__ == "__main__":
    main()
