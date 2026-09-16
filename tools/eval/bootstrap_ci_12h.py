#!/usr/bin/env python3
"""Bootstrap 95% confidence intervals on avg-RMSE_norm for the 12h leaderboard.

Pure-JSON analysis: consumes ``metrics/eval_12h_2020_ep10/*.json`` (strict ep10
schema with ``per_tau[1..11].model.rmse_norm_<CH>``) and resamples the
(τ × channel) pairs with replacement to obtain a 95% CI on the per-model
average normalised RMSE that drives the main leaderboard.

Stratification:
  * For each model we build an (n_tau × n_channels) matrix of rmse_norm values
    across τ ∈ {1..11} (11 τ × 24 channels = 264 cells).
  * Cells are resampled with replacement (B=2000) → percentile 95% CI on the
    mean.
  * SE = std of bootstrap means.

Why this stratification?
  * Channel-level resampling alone (the 6h variant) ignores τ-correlation.
  * τ-only resampling would have only 11 samples, too coarse.
  * Joint (τ, channel) cell resampling captures both axes of variability and
    is conservative w.r.t. the headline ``avg_rmse_norm`` number.

Outputs:
  - ``metrics/eval_12h_2020_ep10/bootstrap_ci_95.json`` — per-model
    {avg_rmse_norm, ci_95_low, ci_95_high, se, B} records, plus pairwise
    significance vs the leader (DeLong-style sign-bootstrap test for the
    difference of means).

The script also patches ``paper/tab_main_12h.tex`` to append a "95% CI"
column (point ± half-width) — generation is opt-in via ``--write-tex``.

Usage::

    python tools/eval/bootstrap_ci_12h.py --bootstraps 2000 --write-tex
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


SEEN = (1, 2, 3, 5, 7, 9, 10, 11)
UNSEEN = (4, 6, 8)
ALL_TAUS = tuple(range(1, 12))


def _build_matrix(d: dict, taus=ALL_TAUS) -> Tuple[np.ndarray, List[str]]:
    """Return (n_tau, n_channel) matrix of rmse_norm for the model block."""
    channels = d["channel_names"]
    rows = []
    used_taus = []
    for tau in taus:
        key = str(tau)
        if key not in d["per_tau"]:
            continue
        block = d["per_tau"][key]["model"]
        row = [block.get(f"rmse_norm_{c}") for c in channels]
        if any(v is None for v in row):
            continue
        rows.append(row)
        used_taus.append(tau)
    return np.array(rows, dtype=np.float64), channels


def _bootstrap_mean_ci(M: np.ndarray, B: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> Tuple[float, float, float, float, np.ndarray]:
    """Bootstrap CI on mean(M) — resample the n_rows × n_cols cells jointly.

    Returns (point, lo_95, hi_95, se, draws).
    """
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


def _seen_unseen_avg(M: np.ndarray, used_taus: List[int],
                     subset: Tuple[int, ...]) -> float:
    idxs = [i for i, t in enumerate(used_taus) if t in subset]
    if not idxs:
        return float("nan")
    return float(M[idxs].mean())


def _pairwise_pvalue(draws_a: np.ndarray, draws_b: np.ndarray) -> float:
    """Two-sided bootstrap p-value for mean(a) < mean(b).

    p = 2 * min( P(a >= b), P(a <= b) ) over paired bootstrap draws.
    """
    if draws_a.size != draws_b.size or draws_a.size == 0:
        return float("nan")
    diff = draws_a - draws_b
    p_a_ge_b = float((diff >= 0).mean())
    p_a_le_b = float((diff <= 0).mean())
    return 2.0 * min(p_a_ge_b, p_a_le_b)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", default="metrics/eval_12h_2020_ep10")
    ap.add_argument("--out", default="metrics/eval_12h_2020_ep10/bootstrap_ci_95.json")
    ap.add_argument("--bootstraps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260610)
    ap.add_argument("--write-tex", action="store_true",
                    help="Patch paper/tab_main_12h.tex to add a 95%% CI column.")
    ap.add_argument("--tex-out", default="paper/tab_main_12h.tex")
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    excluded_stems = {"WeatherDCAE_NoSkip_6yr_12h"}
    json_paths = sorted(
        p
        for p in metrics_dir.glob("*.json")
        if p.name != "bootstrap_ci_95.json" and p.stem not in excluded_stems
    )

    # Display tag map (kept compatible with leaderboard_12h.py)
    TAG_DISPLAY = {
        "ATM-VFI_3yr_12h_fibo":          "ATM-VFI",
        "DC-AE_NoSkip_3yr_12h_fibo":     "DC-AE NoSkip (3yr)",
        "WeatherBridge_PP3_3yr_12h":     "WeatherBridge",
        "DC-AE_Skip_3yr_12h_fibo":       "DC-AE Skip (3yr)",
        "FuXi_3yr_12h_fibo":             "SwinV2",
        "SDyff_3yr_12h_fibo":            "S-DYff",
        "ModAFNO_3yr_12h_fibo":          "ModAFNO",
    }

    records: List[dict] = []
    draws_by_model: Dict[str, np.ndarray] = {}
    t0 = time.time()

    print(f"[bootstrap_ci_12h] B={args.bootstraps}  seed={args.seed}")
    print(f"  metrics_dir = {metrics_dir}")
    print(f"  found {len(json_paths)} JSONs")

    for p in json_paths:
        d = json.load(open(p))
        try:
            M, channels = _build_matrix(d)
        except Exception as e:
            print(f"  [skip] {p.name}: {e}")
            continue
        if M.size == 0:
            print(f"  [skip] {p.name}: empty matrix")
            continue
        used_taus = [int(k) for k in d["per_tau"].keys()
                     if all(d["per_tau"][k]["model"].get(f"rmse_norm_{c}") is not None
                            for c in channels)]
        seed_i = _deterministic_seed(p.stem, args.seed)
        point, lo, hi, se, draws = _bootstrap_mean_ci(
            M, B=args.bootstraps, seed=seed_i
        )
        seen_avg = _seen_unseen_avg(M, used_taus, SEEN)
        unseen_avg = _seen_unseen_avg(M, used_taus, UNSEEN)
        rec = {
            "model_stem": p.stem,
            "display_name": TAG_DISPLAY.get(p.stem, p.stem),
            "n_channels": len(channels),
            "n_tau_used": M.shape[0],
            "taus_used": used_taus,
            "avg_rmse_norm": round(point, 6),
            "ci_95_low": round(lo, 6),
            "ci_95_high": round(hi, 6),
            "ci_95_half_width": round(max(point - lo, hi - point), 6),
            "se": round(se, 6),
            "avg_rmse_norm_seen": round(seen_avg, 6),
            "avg_rmse_norm_unseen": round(unseen_avg, 6),
            "delta_unseen_pct": round((unseen_avg / seen_avg - 1.0) * 100.0, 3)
            if (seen_avg and unseen_avg) else None,
            "B": args.bootstraps,
            "stratification": "joint (τ × channel) cell resampling",
            "_metadata_source": d.get("_metadata", {}).get("checkpoint_path_on_cluster", "?"),
            "_eval_methodology": d.get("_metadata", {}).get("eval_methodology", "?"),
        }
        records.append(rec)
        draws_by_model[p.stem] = draws
        print(f"  {rec['display_name']:35s}  "
              f"avg={point:.4f}  CI95=[{lo:.4f}, {hi:.4f}]  "
              f"±{rec['ci_95_half_width']:.4f}  se={se:.4f}")

    records.sort(key=lambda r: r["avg_rmse_norm"])

    # Pairwise significance vs the leader
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
        "B": args.bootstraps,
        "seed": args.seed,
        "stratification": "joint (τ × channel) cell resampling",
        "metric": "avg(rmse_norm) over τ∈{1..11}, 24 channels (intersection of per-tau availability)",
        "source_dir": str(metrics_dir),
        "leader_stem": records[0]["model_stem"] if records else None,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records": records,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved {out_path}  ({time.time() - t0:.1f}s elapsed)")

    # ───── Optional: patch paper/tab_main_12h.tex with a CI column ─────
    if args.write_tex:
        tex_path = Path(args.tex_out)
        if not tex_path.exists():
            print(f"[write-tex] {tex_path} not found — skipping.")
            return
        # Build a re-generated table with an extra "95% CI" column.
        # We re-read each JSON for per-tau RMSE to keep one source of truth.
        json_by_stem = {p.stem: json.load(open(p)) for p in json_paths}
        rec_by_stem = {r["model_stem"]: r for r in records}
        # Static table:  Model | Params (M) | Train yrs | τ=3 | τ=6 | τ=9 | seen | unseen | Δ% | 95% CI
        TAG_TABLE = {
            "ATM-VFI_3yr_12h_fibo":       ("ATM-VFI", 11.7, "2017-19"),
            "DC-AE_NoSkip_3yr_12h_fibo":  ("DC-AE NoSkip (3yr)", 14.4, "2017-19"),
            "WeatherBridge_PP3_3yr_12h":    ("WeatherBridge", 14.3, "2017-19"),
            "DC-AE_Skip_3yr_12h_fibo":    ("DC-AE Skip (3yr)", 14.4, "2017-19"),
            "FuXi_3yr_12h_fibo":          ("SwinV2", 8.0, "2017-19"),
            "SDyff_3yr_12h_fibo":         ("S-DYff", 85.5, "2017-19"),
            "ModAFNO_3yr_12h_fibo":       ("ModAFNO", 151.7, "2017-19"),
        }

        def _avg_tau(d, tau):
            ch = d["channel_names"]
            block = d["per_tau"].get(str(tau), {}).get("model", {})
            vals = [block.get(f"rmse_norm_{c}") for c in ch]
            vals = [v for v in vals if v is not None]
            return float(np.mean(vals)) if vals else None

        rows = []
        best = min(r["avg_rmse_norm"] for r in records)
        for r in records:
            stem = r["model_stem"]
            disp, params_m, train_yrs = TAG_TABLE.get(
                stem, (r["display_name"], 0.0, "?")
            )
            d = json_by_stem[stem]
            t3 = _avg_tau(d, 3)
            t6 = _avg_tau(d, 6)
            t9 = _avg_tau(d, 9)
            seen_v = r["avg_rmse_norm_seen"]
            unseen_v = r["avg_rmse_norm_unseen"]
            delta = (unseen_v / seen_v - 1.0) * 100.0 if seen_v and unseen_v else None
            avg_all = r["avg_rmse_norm"]
            avg_all_str = f"{avg_all:.3f}"
            if abs(avg_all - best) < 1e-9:
                avg_all_str = r"\textbf{" + avg_all_str + "}"
            ci_str = rf"{avg_all:.3f}\,$\pm$\,{r['ci_95_half_width']:.3f}"
            if abs(avg_all - best) < 1e-9:
                ci_str = rf"\textbf{{{ci_str}}}"
            row = (
                f"{disp} & {params_m:.1f} & {train_yrs} & "
                f"{t3:.3f} & {t6:.3f} & {t9:.3f} & "
                f"{seen_v:.3f} & {unseen_v:.3f} & {delta:.1f}\\% & {ci_str} \\\\"
            )
            rows.append(row)

        # Pull existing numerical-baseline block from current table. Match
        # either the 9-column legacy header or the 10-column patched header so
        # the block is preserved across re-runs.
        existing = tex_path.read_text().splitlines()
        try:
            mid_idx = next(
                i for i, line in enumerate(existing)
                if (
                    r"\multicolumn{9}{l}{\textit{Numerical baselines}}" in line
                    or r"\multicolumn{10}{l}{\textit{Numerical baselines}}" in line
                )
            )
            end_idx = next(i for i, line in enumerate(existing[mid_idx:], start=mid_idx)
                           if line.strip().startswith(r"\bottomrule"))
            num_block_lines = existing[mid_idx + 1:end_idx]
            # Detect whether the existing rows already have 10 columns (last
            # column = "--" for CI) or only 9; add "& --" only if needed.
            patched_num = []
            for ln in num_block_lines:
                stripped = ln.rstrip()
                if not stripped.endswith(r"\\"):
                    patched_num.append(ln)
                    continue
                # Count column separators (& outside of math; safe heuristic
                # because the numeric-baseline rows have no math {}.
                body = stripped[:-2].rstrip()
                n_cols = body.count("&") + 1
                if n_cols >= 10:
                    patched_num.append(f"{body} \\\\")
                else:
                    patched_num.append(f"{body} & -- \\\\")
        except StopIteration:
            patched_num = []

        tex = (
            "% Auto-generated by tools/eval/bootstrap_ci_12h.py (--write-tex)\n"
            "% Source: metrics/eval_12h_2020_ep10/*.json + bootstrap_ci_95.json\n"
            "% 95% CI = joint (τ × channel) bootstrap, B=2000\n"
            r"\begin{table*}[t]" + "\n"
            r"\centering" + "\n"
            r"\caption{12\,h interpolation leaderboard on 2020 ERA5 0.5\textdegree{} "
            r"(4 days/month, 24 channels). Avg seen / unseen split: "
            r"$\tau{\in}\{1,2,3,5,7,9,10,11\}$ seen, $\tau{\in}\{4,6,8\}$ held out. "
            r"$\Delta\%$ is unseen vs seen relative gap. 95\% CI from joint "
            r"$(\tau \times \mathrm{channel})$ bootstrap ($B{=}2000$). "
            r"Lowest avg RMSE in \textbf{bold}.}" + "\n"
            r"\label{tab:main-12h}" + "\n"
            r"\small" + "\n"
            r"\setlength{\tabcolsep}{4pt}" + "\n"
            r"\begin{tabular}{lrrrrrrrrr}" + "\n"
            r"\toprule" + "\n"
            r"Model & Params (M) & Train yrs & RMSE@$\tau$=3 & RMSE@$\tau$=6 & "
            r"RMSE@$\tau$=9 & RMSE seen & RMSE unseen & $\Delta\%$ & avg $\pm$ 95\% CI \\" + "\n"
            r"\midrule" + "\n"
            + "\n".join(rows) + "\n"
            r"\midrule" + "\n"
            r"\multicolumn{10}{l}{\textit{Numerical baselines}} \\" + "\n"
            + "\n".join(patched_num) + "\n"
            r"\bottomrule" + "\n"
            r"\end{tabular}" + "\n"
            r"\end{table*}" + "\n"
        )
        tex_path.write_text(tex)
        print(f"patched {tex_path}")


if __name__ == "__main__":
    main()
