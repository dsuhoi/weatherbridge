#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
from typing import Dict, List, Optional


KEY_METRICS = [
    "val/loss",
    "val/loss_recon",
    "val/loss_latent",
    "val/loss_reg_d",
    "val/rmse",
    "val/rmse_temperature",
    "val/rmse_u_component_of_wind",
    "val/rmse_v_component_of_wind",
    "val/rmse_specific_humidity",
    "val/rmse_geopotential",
    "train/loss_epoch",
    "train/rmse",
]


def to_float(v: str) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        out = float(s)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def pick_metrics_file(run_dir: str) -> str:
    pattern = os.path.join(run_dir, "lightning_logs", "version_*", "metrics.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"metrics.csv not found under {pattern}")

    def version_key(path: str) -> int:
        vname = os.path.basename(os.path.dirname(path))
        if vname.startswith("version_"):
            try:
                return int(vname.split("_", 1)[1])
            except ValueError:
                pass
        return -1

    return sorted(matches, key=version_key)[-1]


def load_rows(metrics_csv: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(metrics_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def last_nonempty(rows: List[Dict[str, str]], key: str) -> Optional[Dict[str, str]]:
    for row in reversed(rows):
        if to_float(row.get(key, "")) is not None:
            return row
    return None


def best_row(rows: List[Dict[str, str]], key: str, mode: str = "min") -> Optional[Dict[str, str]]:
    cand = []
    for row in rows:
        val = to_float(row.get(key, ""))
        if val is not None:
            cand.append((val, row))
    if not cand:
        return None
    cand.sort(key=lambda x: x[0], reverse=(mode == "max"))
    return cand[0][1]


def format_row(row: Optional[Dict[str, str]], keys: List[str]) -> str:
    if row is None:
        return "n/a"
    parts = []
    epoch = row.get("epoch", "")
    step = row.get("step", "")
    if epoch != "":
        parts.append(f"epoch={epoch}")
    if step != "":
        parts.append(f"step={step}")
    for k in keys:
        v = to_float(row.get(k, ""))
        if v is not None:
            parts.append(f"{k}={v:.6f}")
    return ", ".join(parts) if parts else "n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze one WeatherHermite training run")
    ap.add_argument("run_name", help="Name under logs/<run_name>")
    ap.add_argument("--root", default="/home/jovyan/data/time_interpolation")
    args = ap.parse_args()

    run_dir = os.path.join(args.root, "logs", args.run_name)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"Run directory not found: {run_dir}")

    metrics_csv = pick_metrics_file(run_dir)
    rows = load_rows(metrics_csv)

    print(f"run_name: {args.run_name}")
    print(f"run_dir: {run_dir}")
    print(f"metrics_csv: {metrics_csv}")
    print(f"rows: {len(rows)}")

    best_val = best_row(rows, "val/loss", mode="min")
    last_val = last_nonempty(rows, "val/loss")
    last_train = last_nonempty(rows, "train/loss_epoch")

    print("\nBest validation:")
    print(format_row(best_val, [
        "val/loss",
        "val/loss_recon",
        "val/loss_latent",
        "val/loss_reg_d",
        "val/rmse",
        "val/rmse_temperature",
        "val/rmse_u_component_of_wind",
        "val/rmse_v_component_of_wind",
        "val/rmse_specific_humidity",
        "val/rmse_geopotential",
    ]))

    print("\nLast validation row:")
    print(format_row(last_val, KEY_METRICS))

    print("\nLast train epoch row:")
    print(format_row(last_train, ["train/loss_epoch", "train/rmse", "train/grad_norm_2", "train/residual_scale"]))

    ckpts = sorted(glob.glob(os.path.join(run_dir, "*.ckpt")))
    print("\nCheckpoints:")
    print(f"count={len(ckpts)}")
    if ckpts:
        print(f"latest={ckpts[-1]}")


if __name__ == "__main__":
    main()
