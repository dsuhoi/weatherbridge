"""Official-weighting evaluation of the lead-routed ensemble (full-lead below
the threshold, dense_fresh at/above it). Reuses the audited evaluator's own
CheckpointModel, area_weighted_channel_mse, read_init, select_anchor_pairs and
CHANNELS_ORDER so the numbers are directly comparable to every other row in
the paper's tables. The audited script itself is untouched.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6")
sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba")
sys.path.insert(0, "/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim")

from tools.eval.batch_eval_forecast_anchor import (  # noqa: E402
    CHANNELS_ORDER,
    CheckpointModel,
    Era5Cache,
    area_weighted_channel_mse,
    load_channel_stats,
    read_init,
    select_anchor_pairs,
    select_init_files,
)


class LeadRoutedModel:
    """Duck-types CheckpointModel.predict_taus, routing by anchor lead."""

    def __init__(self, short, long, threshold_hours):
        self.short = short
        self.long = long
        self.threshold_hours = threshold_hours
        self.checkpoint_metadata = {
            "short_below_threshold": short.checkpoint_metadata,
            "long_at_or_above_threshold": long.checkpoint_metadata,
            "threshold_hours": threshold_hours,
        }

    def predict_taus(self, x0, xT, tau_hours, *, dt=6, chunk_size=2, anchor_lead_hours=None):
        model = self.long if (anchor_lead_hours or 0) >= self.threshold_hours else self.short
        return model.predict_taus(
            x0, xT, tau_hours, dt=dt, chunk_size=chunk_size, anchor_lead_hours=anchor_lead_hours
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-dir", required=True)
    ap.add_argument("--era5-memmap-dir", required=True)
    ap.add_argument("--checkpoint-short", required=True)
    ap.add_argument("--checkpoint-long", required=True)
    ap.add_argument("--arch", default="flow_pp3_hres_aug")
    ap.add_argument("--threshold-hours", type=float, default=48.0)
    ap.add_argument("--model-name", default="lead_ensemble")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--stats-path", required=True)
    ap.add_argument("--surface-stats-path", required=True)
    ap.add_argument("--static-path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--delta-t-hours", type=int, default=6)
    ap.add_argument("--taus", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--tau-batch-size", type=int, default=5)
    ap.add_argument("--forecast-lead-stride-hours", type=int, required=True)
    ap.add_argument("--forecast-maximum-left-lead-hours", type=int, required=True)
    ap.add_argument("--max-inits", type=int, default=-1)
    args = ap.parse_args()

    stats = load_channel_stats(args.stats_path, args.surface_stats_path, CHANNELS_ORDER)
    stds = stats.std
    era5_cache = Era5Cache(Path(args.era5_memmap_dir))

    short = CheckpointModel(
        args.checkpoint_short, args.arch, args.device, stats,
        static_path=args.static_path, delta_t_hours=args.delta_t_hours, taus=args.taus,
    )
    long = CheckpointModel(
        args.checkpoint_long, args.arch, args.device, stats,
        static_path=args.static_path, delta_t_hours=args.delta_t_hours, taus=args.taus,
    )
    interp = LeadRoutedModel(short, long, args.threshold_hours)
    model_key = args.model_name

    n_ch = len(CHANNELS_ORDER)

    def _new_accum():
        return {tau: {"sxx": np.zeros(n_ch, dtype=np.float64),
                       "cnt": np.zeros(n_ch, dtype=np.int64), "n": 0} for tau in args.taus}

    acc_all = _new_accum()
    forecast_dir = Path(args.forecast_dir)
    init_files = sorted(forecast_dir.glob("init_*.bin"))
    init_files = select_init_files(init_files, args.max_inits)
    if not init_files:
        raise SystemExit(f"no init_*.bin files in {forecast_dir}")
    print(f"[init] {len(init_files)} forecast inits, model={model_key}", flush=True)

    routing_counts = {"short": 0, "long": 0}
    for fi, bin_p in enumerate(init_files):
        json_p = bin_p.with_suffix(".json")
        arr, leads, meta = read_init(bin_p, json_p)
        init_time = np.datetime64(meta["init_time"])
        avail_mask = np.array([c in meta["channels_available"] for c in CHANNELS_ORDER])
        n_processed = 0
        for left_index, right_index in select_anchor_pairs(
            leads, args.delta_t_hours,
            lead_stride_hours=args.forecast_lead_stride_hours,
            maximum_left_lead_hours=args.forecast_maximum_left_lead_hours,
        ):
            lead_a = int(leads[left_index])
            routing_counts["long" if lead_a >= args.threshold_hours else "short"] += 1
            a = arr[left_index]
            b = arr[right_index]
            predictions = interp.predict_taus(
                a, b, args.taus, dt=args.delta_t_hours,
                chunk_size=args.tau_batch_size, anchor_lead_hours=lead_a,
            )
            for tau, pred in zip(args.taus, predictions, strict=True):
                valid = init_time + np.timedelta64(lead_a + tau, "h")
                tgt = era5_cache.hour(valid)
                if tgt is None:
                    continue
                mean_sq_ch = area_weighted_channel_mse(pred, tgt, stds)
                finite_anchors = (
                    np.isfinite(a).all(axis=(1, 2)) & np.isfinite(b).all(axis=(1, 2))
                )
                valid_ch_mask = avail_mask & finite_anchors & np.isfinite(mean_sq_ch)
                mean_sq_ch = np.where(valid_ch_mask, mean_sq_ch, 0.0)
                acc_all[tau]["sxx"] += mean_sq_ch
                acc_all[tau]["cnt"] += valid_ch_mask.astype(np.int64)
                acc_all[tau]["n"] += 1
                n_processed += 1
        if (fi + 1) % 8 == 0 or fi == 0:
            print(f"[{fi+1}/{len(init_files)}] init {str(init_time)[:13]} pairs+={n_processed}",
                  flush=True)

    per_tau = {}
    for tau in args.taus:
        rec = {}
        for ci, c in enumerate(CHANNELS_ORDER):
            n = int(acc_all[tau]["cnt"][ci])
            rec[f"rmse_norm_{c}"] = float(np.sqrt(acc_all[tau]["sxx"][ci] / max(n, 1))) if n else None
            rec[f"n_pairs_{c}"] = n
        rec["n_pairs_any"] = int(acc_all[tau]["n"])
        per_tau[str(tau)] = {model_key: rec}

    out = {
        "schema_version": 1,
        "model_name": model_key,
        "per_tau": per_tau,
        "routing_counts": routing_counts,
        "threshold_hours": args.threshold_hours,
        "protocol": {
            "checkpoint_short": str(args.checkpoint_short),
            "checkpoint_long": str(args.checkpoint_long),
        },
    }
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print("routing_counts:", routing_counts)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
