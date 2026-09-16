"""Compare cloud.ru vs fibo at data-equivalent epochs.
cloud.ru: bs=8 global, 5474 steps/epoch, 43790 samples/epoch (6yr × 7298 items)
fibo:     bs=40 global, 547 steps/epoch,  21885 samples/epoch (3yr × 7295 items)

Match by samples seen: step_cloud × 8 = step_fibo × 40
→ step_fibo = step_cloud / 5
"""
import os, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CH_27 = ["T1000","T925","T850","T700","U1000","U925","U850","U700","V1000","V925","V850","V700",
         "Q1000","Q925","Q850","Q700","Z1000","Z925","Z850","Z700",
         "t2m","u10","v10","mslp","sst","tcc","tcwv"]

def get_at_step(log_dir, target_step, tol=2000):
    ea = EventAccumulator(log_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    # Find val_rmse/<channel> closest to target_step
    res = {}
    for c in CH_27:
        tag = f"val_rmse/{c}"
        if tag in tags:
            evs = ea.Scalars(tag)
            # Find closest step
            best = min(evs, key=lambda e: abs(e.step - target_step))
            if abs(best.step - target_step) <= tol:
                res[c] = (best.step, best.value)
    return res


CR_LOG = "/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_static_6yr/lightning_logs/version_0"
FB_LOG = "/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/lightning_logs/version_0"

if not os.path.exists(FB_LOG):
    FB_LOG = "/workspace/code/wti/logs/exp_dcae_ae_f16_lat128_3yr_fibo_ddp/lightning_logs/version_0"

# target: 175k samples seen
# cloud.ru: 175000/8 = 21875 step
# fibo:     175000/40 = 4375 step
cr = get_at_step(CR_LOG, 21875)
fb = get_at_step(FB_LOG, 4375)

# Load surface stats
sfc_path = "/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json"
if not os.path.exists(sfc_path):
    sfc_path = "/workspace/code/wti/data/surface_stats_0p5.json"
sfc = json.load(open(sfc_path))

APPROX = {"T1000":13.5,"T925":14.2,"T850":14.0,"T700":13.0,"U1000":5.8,"U925":6.5,"U850":7.5,"U700":9.5,
          "V1000":4.8,"V925":5.5,"V850":6.5,"V700":7.5,"Q1000":0.0058,"Q925":0.0055,"Q850":0.0048,"Q700":0.0028,
          "Z1000":87,"Z925":105,"Z850":130,"Z700":195}
std = dict(APPROX)
for c in ["t2m","u10","v10","mslp","sst","tcc","tcwv"]:
    if c in sfc and isinstance(sfc[c], dict):
        std[c] = float(sfc[c].get("std", 1.0))

print("=" * 78)
print("Data-equivalent comparison: ~175k samples seen")
print(f"  cloud.ru @ ~step 21875 (≈ 4 epochs of 6yr data)")
print(f"  fibo     @ ~step 4375  (≈ 8 epochs of 3yr data)")
print("=" * 78)
print(f"{'channel':<8} {'unit':<8} {'cloud norm':>10} {'cloud phys':>12} "
      f"{'fibo norm':>10} {'fibo phys':>12} {'fibo/cloud':>10}")
print("-" * 78)
for c in CH_27:
    cr_info = cr.get(c)
    fb_info = fb.get(c)
    cr_n = cr_info[1] if cr_info else float("nan")
    fb_n = fb_info[1] if fb_info else float("nan")
    s = std[c]
    cr_p = cr_n * s
    fb_p = fb_n * s
    ratio = fb_n / cr_n if (cr_n and cr_n > 0) else float("nan")
    unit = {"t2m":"K","sst":"K","mslp":"Pa","tcc":"0-1","tcwv":"kg/m2"}.get(c, "K" if c.startswith("T") else ("m/s" if c[0] in "UVuv" else ""))
    print(f"{c:<8} {unit:<8} {cr_n:>10.4f} {cr_p:>12.4g} {fb_n:>10.4f} {fb_p:>12.4g} {ratio:>10.2f}")

# Print actual matched steps for diagnostics
if cr and fb:
    sample_c = list(cr.values())[0][0]
    sample_f = list(fb.values())[0][0]
    print(f"\nMatched steps: cloud.ru={sample_c} (target=21875), fibo={sample_f} (target=4375)")
