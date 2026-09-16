#!/usr/bin/env python3
"""Promote strict-ep10 metrics into the canonical metrics/eval_12h_2020_pre_ep10/ dir.

Reads metrics/eval_12h_2020_ep10/*.json (already produced by re-runs against
``epoch=9-step=*.ckpt`` for each model), enriches each JSON with explicit
``checkpoint_epoch`` and ``checkpoint_note`` fields, and overwrites the
corresponding metrics/eval_12h_2020_pre_ep10/<name>.json so downstream figures and
LaTeX tables reflect the strict-ep10 leaderboard.

This is a metadata-only promotion (no recomputation here). The actual model
inference happened in the ep10 re-eval batch on fibo/cloudru.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC_DIR = REPO / "metrics" / "eval_12h_2020_ep10"
DST_DIR = REPO / "metrics" / "eval_12h_2020_pre_ep10"

# (filename stem) → (declared epoch, note). All epochs are Lightning 0-indexed
# converted to 1-indexed for human reading (epoch=9 ckpt → epoch 10 of training).
PROMOTION = {
    "DC-AE_NoSkip_3yr_12h_fibo": (
        10,
        "epoch=9-step=10940.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "DC-AE_Skip_3yr_12h_fibo": (
        10,
        "epoch=9-step=10940.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "FuXi_3yr_12h_fibo": (
        10,
        "epoch=9-step=5470.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "ModAFNO_3yr_12h_fibo": (
        10,
        "epoch=9-step=21870.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "SDyff_3yr_12h_fibo": (
        10,
        "epoch=9-step=5470.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "ATM-VFI_3yr_12h_fibo": (
        10,
        "epoch=9-step=43740.ckpt (Lightning 0-indexed); strict ep10",
    ),
    "WeatherDCAE_NoSkip_6yr_12h": (
        6,
        "last.ckpt = epoch=5-step=13128.ckpt (max_epochs=6); "
        "ep10 to be added after resume training completes",
    ),
}


def main() -> None:
    if not SRC_DIR.is_dir():
        raise SystemExit(f"source dir missing: {SRC_DIR}")
    DST_DIR.mkdir(parents=True, exist_ok=True)
    for stem, (epoch, note) in PROMOTION.items():
        src = SRC_DIR / f"{stem}.json"
        dst = DST_DIR / f"{stem}.json"
        if not src.is_file():
            raise SystemExit(f"missing ep10 JSON: {src}")
        d = json.loads(src.read_text())
        d["checkpoint_path"] = d.get("checkpoint")
        d["checkpoint_epoch"] = epoch
        d["checkpoint_note"] = note
        d["eval_protocol"] = "strict_ep10"
        dst.write_text(json.dumps(d, indent=2))
        print(f"  -> {dst.relative_to(REPO)}  epoch={epoch}")
    print("done")


if __name__ == "__main__":
    main()
