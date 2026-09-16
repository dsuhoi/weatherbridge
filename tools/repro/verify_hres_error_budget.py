#!/usr/bin/env python3
"""Recompute every forecast-anchor error-budget number quoted in the manuscript.

The budget mixes two estimators that are easy to confuse: a window-count
weighted aggregate over all leads, and per-lead-bin values. Quoting one where
the other was computed silently changes the number by several percentage
points, so this script prints both and asserts the values the paper states.

Usage:
    python tools/repro/verify_hres_error_budget.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = Path(
    os.environ.get(
        "WTI_HRES_ERROR_BUDGET_JSON",
        ROOT / "metrics" / "hres_degradation_aware_2021_dev_v1" / "error_budget_by_lead.json",
    )
)

# Left-anchor lead range of the frozen 2022 confirmation set.
CONFIRMATION_MAX_LEAD_H = 120.0

# (label, computed value, value stated in the manuscript, tolerance)
TOLERANCE = 0.06


def _rms(values: np.ndarray, weights: np.ndarray) -> float:
    """Window-count weighted RMSE, i.e. the aggregate averages in MSE space."""
    return float(np.sqrt((values**2 * weights).sum() / weights.sum()))


def main() -> int:
    rows = json.loads(SRC.read_text())
    lead = np.array([r["lead_hours"] for r in rows])
    n = np.array([r["n"] for r in rows], dtype=float)
    anchor = np.array([r["anchor"] for r in rows])
    linear = np.array([r["linear"] for r in rows])
    model = np.array([r["model"] for r in rows])

    checks: list[tuple[str, float, float]] = []

    for label, mask in (("all leads", np.ones_like(lead, dtype=bool)),
                        ("<=120 h", lead <= CONFIRMATION_MAX_LEAD_H)):
        a, l, m, w = anchor[mask], linear[mask], model[mask], n[mask]
        a_r, l_r, m_r = _rms(a, w), _rms(l, w), _rms(m, w)
        print(f"[{label}]")
        print(f"  anchor / linear                 {a_r / l_r:.4f}")
        print(f"  linear adds over anchors        {100 * (l_r - a_r) / a_r:.2f} %")
        print(f"  perfect-interpolator ceiling    {100 * (l_r - a_r) / l_r:.2f} %")
        print(f"  adapted model vs linear         {100 * (l_r - m_r) / l_r:.2f} %")
        print(f"  adapted model vs anchor floor   {100 * (a_r - m_r) / a_r:.2f} %")
        if label == "all leads":
            checks += [
                ("anchor/linear ratio", a_r / l_r, 0.972),
                ("linear adds over anchors (%)", 100 * (l_r - a_r) / a_r, 2.9),
                ("model below anchor floor (%)", 100 * (a_r - m_r) / a_r, 8.1),
            ]
            ratio = a_r / l_r
            for gain, stated in ((20.0, 17.7), (30.0, 28.0)):
                needed = 100 * (gain / 100 - (1 - ratio)) / ratio
                print(f"  beating linear by {gain:.0f}% needs   {needed:.1f} % over the anchors")
                checks.append((f"requirement for {gain:.0f}% (%)", needed, stated))
        else:
            checks += [
                ("confirmation-range ceiling (%)", 100 * (l_r - a_r) / l_r, 5.3),
                ("confirmation-range model gain (%)", 100 * (l_r - m_r) / l_r, 7.0),
            ]

    share = 100 * (linear - anchor) / linear
    gain = 100 * (linear - model) / linear
    below = 100 * (anchor - model) / anchor
    print("\n[per lead bin]")
    print("  lead   share%   gain%   below-floor%")
    for L, s, g, b in zip(lead, share, gain, below):
        print(f"  {L:5.0f}  {s:6.1f}  {g:6.2f}  {b:12.2f}")
    checks += [
        ("interpolation share at lead 0 (%)", float(share[0]), 21.1),
        ("interpolation share at lead 216 (%)", float(share[-1]), 1.5),
        ("gain at lead 0 (%)", float(gain[0]), 17.7),
        ("gain at lead 24 (%)", float(gain[1]), 13.3),
        ("gain at lead 72 (%)", float(gain[3]), 4.5),
        ("gain at lead 216 (%)", float(gain[-1]), 12.8),
        ("below floor at lead 96 (%)", float(below[4]), 2.8),
        ("below floor at lead 168 (%)", float(below[7]), 9.0),
        ("below floor at lead 216 (%)", float(below[-1]), 11.4),
    ]

    print("\n[manuscript claims]")
    failures = 0
    for name, computed, stated in checks:
        ok = abs(computed - stated) <= TOLERANCE * max(1.0, abs(stated))
        failures += not ok
        print(f"  {'ok ' if ok else 'FAIL'}  {name:<36} computed {computed:8.3f}  paper {stated:8.3f}")
    if failures:
        print(f"\n{failures} manuscript claim(s) do not match the artifact.", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
