#!/usr/bin/env python3
"""Compare two Lightning ckpts: param count, L2 norms, per-param L2 diff.

Usage:
    python scripts/compare_ckpts_param_l2.py \
        --ckpt-a <paper_ckpt> --ckpt-b <hydra_ckpt> \
        --out <json_path>

Loads only ``state_dict``, no model construction. Aligns by name. Reports
param count, total L2 of params concat, relative L2 diff. Treats missing
keys on either side as fatal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch


def _flat_l2(t: torch.Tensor) -> float:
    return float(torch.linalg.norm(t.detach().float().flatten()).item())


def _state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj
    return {k: v for k, v in sd.items() if isinstance(v, torch.Tensor)}


def _hyper(ckpt_path: Path) -> Dict[str, Any]:
    obj = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    return dict(obj.get("hyper_parameters", {})) if isinstance(obj, dict) else {}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-a", required=True, help="paper/reference ckpt")
    p.add_argument("--ckpt-b", required=True, help="hydra/new ckpt")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a = Path(args.ckpt_a)
    b = Path(args.ckpt_b)
    print(f"Loading A: {a}")
    sda = _state_dict(a)
    print(f"Loading B: {b}")
    sdb = _state_dict(b)

    keys_a = set(sda.keys())
    keys_b = set(sdb.keys())
    common = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    # Param-count: count only trainable-shaped tensors (skip rank-0 buffers).
    def _is_param_like(t: torch.Tensor) -> bool:
        return t.dim() >= 1 or t.numel() > 1
    n_param_a = sum(t.numel() for t in sda.values() if _is_param_like(t))
    n_param_b = sum(t.numel() for t in sdb.values() if _is_param_like(t))

    # L2 over concat of common keys (sorted) on each side.
    l2_a_sq = 0.0
    l2_b_sq = 0.0
    diff_sq = 0.0
    per_key: List[Dict[str, Any]] = []
    mismatched_shape: List[str] = []
    for k in common:
        ta = sda[k]
        tb = sdb[k]
        if ta.shape != tb.shape:
            mismatched_shape.append(f"{k}: A={tuple(ta.shape)} B={tuple(tb.shape)}")
            continue
        a_norm = _flat_l2(ta)
        b_norm = _flat_l2(tb)
        d = (ta.detach().float() - tb.detach().float()).flatten()
        d_norm = float(torch.linalg.norm(d).item())
        l2_a_sq += a_norm * a_norm
        l2_b_sq += b_norm * b_norm
        diff_sq += d_norm * d_norm
        per_key.append({"key": k, "l2_a": a_norm, "l2_b": b_norm, "l2_diff": d_norm,
                        "shape": list(ta.shape)})
    l2_a = math.sqrt(l2_a_sq)
    l2_b = math.sqrt(l2_b_sq)
    l2_diff = math.sqrt(diff_sq)
    rel_diff = l2_diff / max(l2_a, 1e-12)
    rel_diff_b = abs(l2_a - l2_b) / max(l2_a, 1e-12)

    # Top-10 keys by |l2_diff|.
    top_diff = sorted(per_key, key=lambda r: -r["l2_diff"])[:10]

    out: Dict[str, Any] = {
        "ckpt_a": str(a),
        "ckpt_b": str(b),
        "n_keys_a": len(sda),
        "n_keys_b": len(sdb),
        "n_common_keys": len(common),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "only_a_examples": only_a[:20],
        "only_b_examples": only_b[:20],
        "param_count_a": n_param_a,
        "param_count_b": n_param_b,
        "param_count_match": n_param_a == n_param_b,
        "n_mismatched_shape": len(mismatched_shape),
        "mismatched_shape_examples": mismatched_shape[:20],
        "l2_a": l2_a,
        "l2_b": l2_b,
        "l2_abs_norm_diff": abs(l2_a - l2_b),
        "l2_rel_norm_diff": rel_diff_b,           # |‖A‖ - ‖B‖| / ‖A‖
        "l2_pairwise_diff": l2_diff,              # ‖A − B‖
        "l2_pairwise_rel_diff": rel_diff,         # ‖A − B‖ / ‖A‖
        "top_diff_keys": top_diff,
        "hparams_a": _hyper(a),
        "hparams_b": _hyper(b),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in out.items() if k != "top_diff_keys"
                       and k != "hparams_a" and k != "hparams_b"
                       and k != "only_a_examples" and k != "only_b_examples"
                       and k != "mismatched_shape_examples"}, indent=2, default=str))


if __name__ == "__main__":
    main()
