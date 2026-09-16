#!/usr/bin/env python3
"""Summarize Lightning CSV-logger metrics.csv into per-epoch JSON.

Reads `metrics.csv` (produced by lightning.pytorch.loggers.CSVLogger),
extracts train/loss + val/loss aggregated per epoch, and emits a small JSON
summary that's easy to drop into a verification doc.

Usage:
    python scripts/summarize_lightning_csv.py path/to/metrics.csv \
        --out path/to/summary.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


def _f(v: Optional[str]) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=str)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = []
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"error": "no rows"}, f, indent=2)
        return

    train_loss_by_epoch: Dict[int, List[float]] = defaultdict(list)
    val_loss_by_epoch: Dict[int, List[float]] = defaultdict(list)
    train_loss_steps: List[Dict[str, float]] = []
    val_keys = [k for k in rows[0].keys() if k.startswith("val/")]
    train_keys = [k for k in rows[0].keys() if k.startswith("train/")]

    for row in rows:
        ep = _f(row.get("epoch"))
        st = _f(row.get("step"))
        if ep is None:
            continue
        ep_i = int(ep)
        tl = _f(row.get("train/loss")) or _f(row.get("train_loss"))
        if tl is not None:
            train_loss_by_epoch[ep_i].append(tl)
            if st is not None:
                train_loss_steps.append({"epoch": ep_i, "step": int(st), "train_loss": tl})
        vl = _f(row.get("val/loss")) or _f(row.get("val_loss"))
        if vl is not None:
            val_loss_by_epoch[ep_i].append(vl)

    def _avg(xs: List[float]) -> Optional[float]:
        return sum(xs) / len(xs) if xs else None

    epochs = sorted(set(list(train_loss_by_epoch.keys()) + list(val_loss_by_epoch.keys())))
    per_epoch = []
    for ep in epochs:
        per_epoch.append({
            "epoch": ep,
            "train_loss_mean": _avg(train_loss_by_epoch.get(ep, [])),
            "train_loss_last": (train_loss_by_epoch.get(ep, [])[-1]
                                if train_loss_by_epoch.get(ep) else None),
            "val_loss_mean": _avg(val_loss_by_epoch.get(ep, [])),
            "val_loss_last": (val_loss_by_epoch.get(ep, [])[-1]
                              if val_loss_by_epoch.get(ep) else None),
            "n_train_logs": len(train_loss_by_epoch.get(ep, [])),
            "n_val_logs": len(val_loss_by_epoch.get(ep, [])),
        })

    out = {
        "csv": args.csv,
        "n_rows": len(rows),
        "available_keys": list(rows[0].keys()),
        "train_keys": train_keys,
        "val_keys": val_keys,
        "per_epoch": per_epoch,
        "train_loss_last_step": train_loss_steps[-1] if train_loss_steps else None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
