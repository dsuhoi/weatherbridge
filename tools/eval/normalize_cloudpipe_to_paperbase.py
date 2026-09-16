#!/usr/bin/env python3
"""Convert cloudpipe eval JSONs (6h / 12h) to paper-baseline-comparable units.

Root cause and conversion table: see ``docs/Q_CHANNEL_UNIT_DRIFT.md``.

In short, the cloudpipe scripts (``batch_eval_memmap.py`` /
``batch_eval_12h_memmap.py``) accumulate ``((pred-tgt)**2 * w_lat).sum(H, W)``
without dividing by ``W``, so every per-channel RMSE in those JSONs is
``sqrt(W) ≈ 26.833`` times the true per-pixel lat-weighted RMSE at 0.5°
(W=720). This affects all channels uniformly; the per-channel "phys" ratio
explodes for Q (small σ) and collapses for Z/mslp (huge σ), but the underlying
defect is W-only.

This script does NOT touch the cloudpipe scripts themselves (their outputs
are consumed by other tools that expect the inflated schema). It post-hoc
emits a normalised copy with:

  * ``rmse_norm_<CH>`` = cloudpipe value / sqrt(W)    (per-pixel lat-weighted norm)
  * ``rmse_phys_<CH>`` = rmse_norm_<CH> × stds_for_denorm[<CH>]
  * a ``_conversion_metadata`` block recording the audit trail.

The output schema is uniform 12h-style (``per_tau[<tau>].model.rmse_{norm,phys}_<CH>``)
for both 6h and 12h inputs, so downstream tools can read either with one code path.

Usage::

    python tools/eval/normalize_cloudpipe_to_paperbase.py \\
        --in-dir  metrics/eval_0p5_2020_cloudpipe \\
        --out-dir metrics/eval_0p5_2020_cloudpipe_normalized \\
        --horizon 6h

    python tools/eval/normalize_cloudpipe_to_paperbase.py \\
        --in-dir  metrics/eval_12h_2021_ood \\
        --out-dir metrics/eval_12h_2021_ood_normalized \\
        --horizon 12h
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

# ---------------------------------------------------------------------------
# Canonical 0.5° ERA5 24-channel stds (per-channel σ used to denormalise).
# Source: data/json_stats_0p5.nc; mirrored in every paper-base JSON's
# ``stds_for_denorm`` field (e.g.
# metrics/eval_6h_2020_paper_leaderboard/dcae_skip_24ch_6yr_ep8.json).
# ---------------------------------------------------------------------------
STDS_24CH_0P5: Dict[str, float] = {
    "T1000": 16.949722290039062,
    "T925":  15.9839448928833,
    "T850":  15.35348892211914,
    "T700":  14.31494140625,
    "U1000": 5.518161773681641,
    "U925":  7.160628795623779,
    "U850":  7.5362019538879395,
    "U700":  8.530590057373047,
    "V1000": 3.07429575920105,
    "V925":  3.1294267177581787,
    "V850":  2.508357286453247,
    "V700":  2.236948251724243,
    "Q1000": 0.007229719776660204,
    "Q925":  0.006158000789582729,
    "Q850":  0.004850391298532486,
    "Q700":  0.0029231621883809566,
    "Z1000": 913.3009643554688,
    "Z925":  1111.41162109375,
    "Z850":  1393.02978515625,
    "Z700":  2121.437744140625,
    "t2m":   19.600000381469727,
    "u10":   4.940000057220459,
    "v10":   2.7799999713897705,
    "mslp":  1115.699951171875,
}

# 0.5° grid width: 720 longitude points.
GRID_W = 720
SQRT_W = math.sqrt(GRID_W)


def _normalize_6h_block(model_block: Dict[str, float],
                        channels: List[str]) -> Dict[str, float]:
    """6h cloudpipe schema → paper-base-comparable {rmse_norm,rmse_phys}_<CH>.

    Input ``model_block`` has keys ``rmse_<CH>`` whose values are
    ``true_norm_rmse × sqrt(W)``. We emit both norm and phys.
    """
    out: Dict[str, float] = {
        key: float(value)
        for key, value in model_block.items()
        if key.startswith("acc_") and isinstance(value, (int, float))
    }
    for c in channels:
        v = model_block.get(f"rmse_{c}")
        if v is None or not isinstance(v, (int, float)):
            continue
        v_norm = float(v) / SQRT_W
        out[f"rmse_norm_{c}"] = v_norm
        if c in STDS_24CH_0P5:
            out[f"rmse_phys_{c}"] = v_norm * STDS_24CH_0P5[c]
    return out


def _normalize_12h_block(model_block: Dict[str, float],
                         channels: List[str]) -> Dict[str, float]:
    """12h cloudpipe schema → paper-base-comparable {rmse_norm,rmse_phys}_<CH>.

    Input has both ``rmse_norm_<CH>`` and ``rmse_phys_<CH>`` inflated by sqrt(W).
    We divide both by sqrt(W). (Equivalent to: divide norm by sqrt(W) then
    multiply by σ; both methods agree to numerical precision.)
    """
    out: Dict[str, float] = {
        key: float(value)
        for key, value in model_block.items()
        if key.startswith("acc_") and isinstance(value, (int, float))
    }
    for c in channels:
        nv = model_block.get(f"rmse_norm_{c}")
        if nv is not None:
            out[f"rmse_norm_{c}"] = float(nv) / SQRT_W
        pv = model_block.get(f"rmse_phys_{c}")
        if pv is not None:
            out[f"rmse_phys_{c}"] = float(pv) / SQRT_W
        # If phys absent but norm present and σ available, synthesise phys.
        if pv is None and nv is not None and c in STDS_24CH_0P5:
            out[f"rmse_phys_{c}"] = (float(nv) / SQRT_W) * STDS_24CH_0P5[c]
    return out


def normalize_one(in_path: Path, horizon: str) -> Dict:
    """Return a normalised copy of a single cloudpipe JSON.

    ``horizon`` ∈ {"6h", "12h"} selects the input schema.
    """
    d = json.loads(in_path.read_text())
    channels = d.get("channel_names") or d.get("channels") or []
    if not channels:
        raise ValueError(f"{in_path.name}: no channel_names/channels block")
    out = copy.deepcopy(d)

    if horizon == "6h":
        per_in = d.get("per_hour", {})
        per_out: Dict[str, Dict[str, Dict[str, float]]] = {}
        for tau, sub in per_in.items():
            per_out[tau] = {}
            for actor in ("model", "bilinear", "bicubic"):
                if actor in sub:
                    per_out[tau][actor] = _normalize_6h_block(sub[actor], channels)
        # Discard the inflated per_hour; emit a 12h-style per_tau block so
        # downstream tools see a uniform schema. Keep per_hour for backward
        # compat by also writing the normalised copy under the original key.
        out["per_tau"] = per_out
        out["per_hour"] = per_out
    elif horizon == "12h":
        per_in = d.get("per_tau") or d.get("per_hour") or {}
        per_out = {}
        for tau, sub in per_in.items():
            per_out[tau] = {}
            for actor in ("model", "bilinear", "bicubic"):
                if actor in sub:
                    per_out[tau][actor] = _normalize_12h_block(sub[actor], channels)
        out["per_tau"] = per_out
        if "per_hour" in d:
            out["per_hour"] = per_out
    else:
        raise ValueError(f"unsupported horizon {horizon!r}")

    out["stds_for_denorm"] = {c: STDS_24CH_0P5.get(c) for c in channels}
    out["_conversion_metadata"] = {
        "source_path": str(in_path),
        "source_horizon": horizon,
        "grid_W": GRID_W,
        "sqrt_W_factor_removed": SQRT_W,
        "stds_source": "data/json_stats_0p5.nc (canonical 24ch 0.5° σ)",
        "scheme": "rmse_norm = cloud / sqrt(W); rmse_phys = rmse_norm × σ",
        "doc": "docs/Q_CHANNEL_UNIT_DRIFT.md",
    }
    return out


def run(in_dir: Path, out_dir: Path, horizon: str) -> List[Path]:
    """Normalise every JSON in ``in_dir`` to ``out_dir`` and return the list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for p in sorted(in_dir.glob("*.json")):
        # Skip aggregate / non-eval JSONs that don't carry per-tau or per_hour.
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if not (d.get("per_hour") or d.get("per_tau")):
            print(f"  [skip] {p.name}: no per_hour/per_tau block")
            continue
        try:
            normed = normalize_one(p, horizon)
        except Exception as e:
            print(f"  [fail] {p.name}: {e}")
            continue
        out_path = out_dir / p.name
        out_path.write_text(json.dumps(normed, indent=2))
        written.append(out_path)
        print(f"  wrote {out_path}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True,
                    help="Cloudpipe input directory.")
    ap.add_argument("--out-dir", required=True,
                    help="Normalised-output directory.")
    ap.add_argument("--horizon", required=True, choices=("6h", "12h"),
                    help="Input schema horizon.")
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"in-dir not a directory: {in_dir}")
    written = run(in_dir, out_dir, args.horizon)
    print(f"\nnormalised {len(written)} JSONs into {out_dir}")


if __name__ == "__main__":
    main()
