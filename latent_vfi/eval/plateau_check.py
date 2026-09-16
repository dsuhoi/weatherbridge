"""Analyze convergence trend + estimate plateau distance for both AE trainings."""
import os, sys
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def analyze(log_dir, label):
    if not os.path.exists(log_dir):
        print(f"--- {label}: no log dir ---")
        return
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if "val/recon_l1" not in ea.Tags()["scalars"]:
        print(f"--- {label}: no val/recon_l1 ---")
        return
    evs = ea.Scalars("val/recon_l1")
    if len(evs) < 5:
        print(f"--- {label}: too few val points ({len(evs)}) ---")
        return
    steps = np.array([e.step for e in evs])
    vals = np.array([e.value for e in evs])

    print(f"\n=== {label} ===")
    print(f"  N val checks: {len(evs)}  last step={steps[-1]}  last val/recon={vals[-1]:.5f}")

    # Last 5 vs prior 5 — slope analysis
    if len(vals) >= 10:
        early_slope = (vals[-6] - vals[-10]) / (steps[-6] - steps[-10])
        late_slope = (vals[-1] - vals[-5]) / (steps[-1] - steps[-5])
        print(f"  early slope (last 5-10 ago): {early_slope*1000:.4f}  (per 1000 steps)")
        print(f"  late slope (last 5):         {late_slope*1000:.4f}")
        slope_ratio = abs(late_slope / early_slope) if early_slope != 0 else 0
        print(f"  late/early slope ratio: {slope_ratio:.2f} (1.0=same, <1=plateau)")

    # Exponential fit: vals[k] = a + b * exp(-c * k)
    # Simpler: estimate plateau by extrapolating from last N points
    last_n = min(8, len(vals))
    last_vals = vals[-last_n:]
    last_steps = steps[-last_n:]
    # Estimate "if we double epochs"
    if len(last_vals) >= 4:
        # Linear in log-step
        recent_drop_per_epoch = (last_vals[0] - last_vals[-1]) / max(1, len(last_vals) - 1)
        print(f"  recent drop per val-check: {recent_drop_per_epoch*1000:.4f} (×1e-3)")
        # Extrapolate: if same rate continues for next N epochs
        for next_n in [5, 10, 20]:
            est = max(0.05, last_vals[-1] - recent_drop_per_epoch * next_n)
            pct_improve = (1 - est/last_vals[-1]) * 100
            print(f"  +{next_n} val-checks @ same rate: {est:.4f} ({pct_improve:+.1f}%)")

    # Last 5 values
    print(f"  recent val sequence: {[f'{v:.4f}' for v in vals[-5:]]}")


analyze("/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_static_6yr/lightning_logs/version_0",
        "cloud.ru lat=64 6yr")
analyze("/workspace/code/wti/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/lightning_logs/version_0",
        "fibo lat=128 3yr")
