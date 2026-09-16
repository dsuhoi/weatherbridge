#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals on avg-RMSE_norm for the 6h leaderboard.

Pure-JSON analysis: consumes ``metrics/eval_6h_2020_paper_leaderboard/*.json``.

The 6h JSONs come in TWO schemas:

  (A) 24-channel "fast" schema
        Keys: ``channels`` (24), ``tau_hours`` ([1..5]),
              ``rmse_model_norm`` (list[5] of list[24]) — OR — dict
              keyed by tau string (e.g. atm_vfi_v2_135only_rmse.json).
        These are the headline 24ch models in sec_main_table_v2.

  (B) 27-channel "legacy" schema
        Keys: ``channel_names`` (27), ``per_hour`` ({"1".."5"})
              with ``model``/``bilinear``/``bicubic`` dicts of
              ``rmse_<CH>`` (PHYSICAL units, no rmse_norm).
        Used by 01_main_table_0p5_6yr.tex; we normalize by the
        per-channel bilinear baseline RMSE so the resulting
        "rmse_norm" is comparable across channels (it is the ratio
        model_rmse / bilinear_rmse — values < 1 = better than bilinear).
        This is the SAME normalisation used by the published 27ch
        bootstrap CI script (tools/eval/bootstrap_ci.py).

We build a (n_tau × n_channel) matrix of either rmse_norm (24ch) or
the bilinear-relative ratio (27ch) and bootstrap-resample cells with
replacement, B=2000, joint (τ × channel).

Outputs:
  - ``metrics/eval_6h_2020_paper_leaderboard/bootstrap_ci_95_24ch.json``
  - ``metrics/eval_6h_2020_paper_leaderboard/bootstrap_ci_95_27ch.json``

The two leaderboards use different metric semantics (raw rmse_norm vs
ratio-to-bilinear), so we deliberately keep them in separate JSONs.

Usage::

    python tools/eval/bootstrap_ci_6h.py --bootstraps 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _deterministic_seed(label: str, base_seed: int) -> int:
    """Return a stable 32-bit seed derived from `label` + `base_seed`.

    Python's builtin ``hash()`` is randomised per-process (PYTHONHASHSEED),
    so it must not be used for reproducible RNG seeding. We use MD5 over the
    UTF-8 bytes and XOR with the user-supplied base seed.
    """
    digest = hashlib.md5(label.encode("utf-8")).digest()[:8]
    return (int.from_bytes(digest, "big") ^ base_seed) & 0xFFFFFFFF


ALL_TAUS_6H = (1, 2, 3, 4, 5)


# ---------------- 24-channel schema ----------------

def _build_matrix_24ch(d: dict, taus=ALL_TAUS_6H) -> Tuple[np.ndarray, List[str], List[int]]:
    """Return (n_tau, n_channel) matrix of rmse_norm for 24ch JSONs."""
    channels = d.get("channels") or []
    tau_hours = d.get("tau_hours") or list(taus)
    rmse = d.get("rmse_model_norm")
    if rmse is None:
        return np.empty((0, 0)), channels, []
    rows = []
    used_taus = []
    if isinstance(rmse, list):
        # list of lists in tau_hours order
        for i, tau in enumerate(tau_hours):
            if tau not in taus:
                continue
            row = rmse[i]
            if any(v is None for v in row):
                continue
            rows.append(row)
            used_taus.append(tau)
    elif isinstance(rmse, dict):
        # dict keyed by tau string
        for tau in taus:
            key = str(tau)
            if key not in rmse:
                continue
            row = rmse[key]
            if any(v is None for v in row):
                continue
            rows.append(row)
            used_taus.append(tau)
    return np.array(rows, dtype=np.float64), channels, used_taus


# ---------------- 27-channel schema ----------------

def _build_matrix_27ch(d: dict, taus=ALL_TAUS_6H) -> Tuple[np.ndarray, List[str], List[int]]:
    """Return (n_tau, n_channel) matrix for 27ch JSONs.

    Metric: rmse_model / rmse_bilinear (per channel, per τ). Values < 1.0
    = better than bilinear. We average this ratio across τ × channels
    — same shape as 24ch matrix so CIs are interpretable on the same
    bootstrap protocol.
    """
    channels = d["channel_names"]
    per_hour = d.get("per_hour", {})
    rows = []
    used_taus = []
    for tau in taus:
        key = str(tau)
        if key not in per_hour:
            continue
        node = per_hour[key]
        model = node.get("model", {})
        bilinear = node.get("bilinear") or node.get("bicubic") or {}
        if not bilinear:
            continue
        row = []
        for c in channels:
            mv = model.get(f"rmse_{c}")
            bv = bilinear.get(f"rmse_{c}")
            if mv is None or bv is None or bv <= 0:
                row.append(None)
            else:
                row.append(float(mv) / float(bv))
        if any(v is None for v in row):
            continue
        rows.append(row)
        used_taus.append(tau)
    return np.array(rows, dtype=np.float64), channels, used_taus


# ---------------- Bootstrap ----------------

def _bootstrap_mean_ci(M: np.ndarray, B: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> Tuple[float, float, float, float, np.ndarray]:
    """Bootstrap CI on mean(M) — resample cells jointly."""
    rng = np.random.default_rng(seed)
    flat = M.ravel()
    n = flat.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), np.empty(0)
    draws = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        draws[b] = flat[idx].mean()
    point = float(flat.mean())
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    se = float(draws.std(ddof=1))
    return point, lo, hi, se, draws


def _pairwise_pvalue(draws_a: np.ndarray, draws_b: np.ndarray) -> float:
    """Two-sided bootstrap p-value for paired draws (same B)."""
    if draws_a.size != draws_b.size or draws_a.size == 0:
        return float("nan")
    diff = draws_a - draws_b
    p_a_ge_b = float((diff >= 0).mean())
    p_a_le_b = float((diff <= 0).mean())
    return 2.0 * min(p_a_ge_b, p_a_le_b)


# ---------------- 24ch display map ----------------

TAG_24CH = {
    # Strict ep8 leaderboard (paper sec_main_table_v2.tex uniform-epoch row)
    "weatherdcae_noskip_24ch_6yr_ep8":   "WeatherDCAE NoSkip (6yr, ep8)",
    "modafno_24ch_6yr_ep8":              "ModAFNO 24ch (6yr, ep8)",
    "sdyff_24ch_6yr_ep8":                "S-DYff 24ch (6yr, ep8)",
    "fuxi_24ch_6yr_ep8":                 "FuXi 24ch (6yr, ep8)",
    "dcae_skip_24ch_6yr_ep8":            "DC-AE Skip 24ch (6yr, ep8)",
    "bilinear_24ch":                     "Bilinear (24ch)",
    "atm_vfi_24ch_6yr":                  "ATM-VFI 24ch (6yr)",
    "atm_vfi_v2_static_24ch_6yr":        "ATM-VFI v2 static 24ch (6yr)",
    "atm_vfi_v2_135only_rmse":           "ATM-VFI v2 (135 only)",
    "corrdiff_fm_bilinear_24ch":         "CorrDiff-FM / Bilinear (24ch)",
    "corrdiff_fm_dcae_skip_24ch":        "CorrDiff-FM / DC-AE Skip (24ch)",
    # Legacy ep10 (archived; kept in tag map so CI script still labels it if present)
    "weatherdcae_noskip_24ch_6yr_ep10":  "WeatherDCAE NoSkip (6yr, ep10) [archived]",
}

TAG_27CH = {
    "dcae_skip_0p5_6yr_pad":         "DC-AE Skip (PAD) 6yr",
    "dcae_skip_0p5_pad":             "DC-AE Skip (PAD) 3yr",
    "dcae_skip_frozen_gates":        "DC-AE NoSkip-frozen 6yr",
    "dcae_skip_frozen_gates_ep7":    "DC-AE NoSkip-frozen 6yr (ep7)",
    "fuxi_0p5_6yr":                  "FuXi SwinV2 6yr",
    "fuxi_0p5_2018":                 "FuXi SwinV2 1yr (2018)",
    "modafno_0p5_6yr":               "ModAFNO 6yr",
    "modafno_0p5_2018":              "ModAFNO 1yr (2018)",
    "sdyff_dyffusion_0p5_6yr":       "S-DYff DYffusion 6yr",
}


def _process(json_paths: List[Path], schema: str, B: int, seed: int) -> Dict:
    records: List[dict] = []
    draws_by_model: Dict[str, np.ndarray] = {}
    tag_map = TAG_24CH if schema == "24ch" else TAG_27CH

    for p in json_paths:
        try:
            d = json.load(open(p))
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if schema == "24ch":
            M, channels, used = _build_matrix_24ch(d)
        else:
            M, channels, used = _build_matrix_27ch(d)
        if M.size == 0:
            print(f"  [skip] {p.name}: empty matrix")
            continue
        seed_i = _deterministic_seed(p.stem, seed)
        point, lo, hi, se, draws = _bootstrap_mean_ci(M, B=B, seed=seed_i)
        rec = {
            "model_stem": p.stem,
            "display_name": tag_map.get(p.stem, p.stem),
            "schema": schema,
            "n_channels": len(channels),
            "n_tau_used": M.shape[0],
            "taus_used": used,
            "avg_rmse_norm" if schema == "24ch" else "avg_ratio_to_bilinear":
                round(point, 6),
            "ci_95_low": round(lo, 6),
            "ci_95_high": round(hi, 6),
            "ci_95_half_width": round(max(point - lo, hi - point), 6),
            "se": round(se, 6),
            "B": B,
            "stratification": "joint (τ × channel) cell resampling",
        }
        records.append(rec)
        draws_by_model[p.stem] = draws
        metric_key = "avg_rmse_norm" if schema == "24ch" else "avg_ratio_to_bilinear"
        print(f"  {rec['display_name']:38s}  "
              f"{metric_key}={point:.4f}  CI95=[{lo:.4f}, {hi:.4f}]  "
              f"±{rec['ci_95_half_width']:.4f}")

    key_metric = "avg_rmse_norm" if schema == "24ch" else "avg_ratio_to_bilinear"
    records.sort(key=lambda r: r[key_metric])

    # Pairwise significance vs leader
    if records:
        leader = records[0]
        leader_draws = draws_by_model[leader["model_stem"]]
        for r in records:
            if r["model_stem"] == leader["model_stem"]:
                r["p_vs_leader"] = None
                r["sig_vs_leader_at_0p05"] = None
                continue
            p_val = _pairwise_pvalue(leader_draws, draws_by_model[r["model_stem"]])
            r["p_vs_leader"] = round(p_val, 6)
            r["sig_vs_leader_at_0p05"] = bool(p_val < 0.05)

    payload = {
        "schema_version": 1,
        "schema": schema,
        "B": B,
        "seed": seed,
        "stratification": "joint (τ × channel) cell resampling",
        "metric": ("avg(rmse_norm) over τ∈{1..5}, 24 channels" if schema == "24ch"
                   else "avg(model_rmse / bilinear_rmse) over τ∈{1..5}, 27 channels"),
        "leader_stem": records[0]["model_stem"] if records else None,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="metrics/eval_6h_2020_paper_leaderboard")
    ap.add_argument("--out-24ch", default="metrics/eval_6h_2020_paper_leaderboard/bootstrap_ci_95_24ch.json")
    ap.add_argument("--out-27ch", default="metrics/eval_6h_2020_paper_leaderboard/bootstrap_ci_95_27ch.json")
    ap.add_argument("--bootstraps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260610)
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)

    # Classify by schema
    paths_24ch: List[Path] = []
    paths_27ch: List[Path] = []
    for p in sorted(metrics_dir.glob("*.json")):
        if p.name.startswith("bootstrap_ci_"):
            continue
        if p.name in {"channel_class_breakdown_24ch.json",
                      "channel_class_breakdown_27ch.json"}:
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        if "channels" in d and "tau_hours" in d:
            paths_24ch.append(p)
        elif "channel_names" in d and "per_hour" in d:
            paths_27ch.append(p)

    t0 = time.time()
    print(f"[bootstrap_ci_6h] B={args.bootstraps}  seed={args.seed}")
    print(f"  metrics_dir = {metrics_dir}")
    print(f"  24ch JSONs: {len(paths_24ch)}, 27ch JSONs: {len(paths_27ch)}")

    print("\n=== 24-channel leaderboard (raw rmse_norm) ===")
    payload_24 = _process(paths_24ch, "24ch", args.bootstraps, args.seed)
    Path(args.out_24ch).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_24ch).write_text(json.dumps(payload_24, indent=2))
    print(f"  saved {args.out_24ch}")

    print("\n=== 27-channel leaderboard (ratio to bilinear) ===")
    payload_27 = _process(paths_27ch, "27ch", args.bootstraps, args.seed)
    Path(args.out_27ch).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_27ch).write_text(json.dumps(payload_27, indent=2))
    print(f"  saved {args.out_27ch}")

    print(f"\n{time.time() - t0:.1f}s elapsed")


if __name__ == "__main__":
    main()
