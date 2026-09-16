#!/usr/bin/env python3
"""Compare two outputs of `verify_hydra_equivalence.py` (legacy vs hydra).

Reads two JSON files produced by the harness and emits:

  - Loss curve overlap (% relative diff per step).
  - Final-state param L2 / max-abs delta.
  - Pass/fail vs configurable threshold.

Usage:
    python compare_hydra_verify.py logs/hydra_verify/legacy.json \
                                   logs/hydra_verify/hydra.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("legacy_json")
    ap.add_argument("hydra_json")
    ap.add_argument("--threshold-pct", type=float, default=1.0,
                    help="Pass if max abs %% diff <= this. Default 1.0")
    args = ap.parse_args()

    with open(args.legacy_json) as f:
        L = json.load(f)
    with open(args.hydra_json) as f:
        H = json.load(f)

    print("=" * 70)
    print("Hydra-vs-legacy equivalence check")
    print("=" * 70)

    print(f"\n[legacy] param_count={L.get('param_count')} param_l2={L.get('param_l2'):.4f} "
          f"max_abs={L.get('param_max_abs'):.4f}")
    print(f"[hydra ] param_count={H.get('param_count')} param_l2={H.get('param_l2'):.4f} "
          f"max_abs={H.get('param_max_abs'):.4f}")

    same_count = L.get("param_count") == H.get("param_count")
    print(f"\nparam_count match     : {same_count}")
    if not same_count:
        print("  → model architectures differ; pure equivalence check not meaningful")

    # Param-norm comparison
    if L.get("param_l2") and H.get("param_l2"):
        rel_l2 = abs(L["param_l2"] - H["param_l2"]) / max(abs(L["param_l2"]), 1e-9) * 100
        print(f"param_l2 rel diff     : {rel_l2:.4f}%")
        rel_max = abs(L["param_max_abs"] - H["param_max_abs"]) / max(abs(L["param_max_abs"]), 1e-9) * 100
        print(f"param_max_abs rel diff: {rel_max:.4f}%")

    # Train loss curve comparison
    Lt = L.get("train_losses", [])
    Ht = H.get("train_losses", [])
    n = min(len(Lt), len(Ht))
    print(f"\ntrain loss steps      : legacy={len(Lt)}, hydra={len(Ht)}, compared={n}")
    if n > 0:
        max_rel = 0.0
        sum_rel = 0.0
        for i in range(n):
            denom = max(abs(Lt[i]), 1e-9)
            rel = abs(Lt[i] - Ht[i]) / denom * 100
            sum_rel += rel
            max_rel = max(max_rel, rel)
        mean_rel = sum_rel / n
        print(f"  mean rel diff       : {mean_rel:.4f}%")
        print(f"  max  rel diff       : {max_rel:.4f}%")
        print(f"\nlegacy train_loss[0]  : {Lt[0]:.4f}")
        print(f"hydra  train_loss[0]  : {Ht[0]:.4f}")
        print(f"legacy train_loss[-1] : {Lt[-1]:.4f}")
        print(f"hydra  train_loss[-1] : {Ht[-1]:.4f}")
        legacy_last = Lt[-1]
        hydra_last = Ht[-1]
        rel_last = abs(legacy_last - hydra_last) / max(abs(legacy_last), 1e-9) * 100
        print(f"\nfinal train_loss rel diff: {rel_last:.4f}%")

    Lv = L.get("val_losses", [])
    Hv = H.get("val_losses", [])
    if Lv and Hv:
        print(f"\nval losses            : legacy={Lv[-1]:.4f}, hydra={Hv[-1]:.4f}")
        rel_v = abs(Lv[-1] - Hv[-1]) / max(abs(Lv[-1]), 1e-9) * 100
        print(f"val_loss rel diff     : {rel_v:.4f}%")

    print("\n" + "=" * 70)
    if n > 0 and max_rel <= args.threshold_pct:
        print(f"PASS: max train-loss rel diff {max_rel:.4f}% ≤ {args.threshold_pct}% threshold")
        sys.exit(0)
    else:
        print(f"DIVERGE: max rel diff {max_rel:.4f}% > {args.threshold_pct}% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
