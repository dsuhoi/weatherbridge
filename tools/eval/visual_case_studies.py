#!/usr/bin/env python3
"""Generate visual case-study figures for the paper:

  1. Hurricane case  — strong tropical cyclone, MSLP + wind speed plots
  2. Heatwave case   — 2020 Siberian heatwave, T2m anomaly plots
  3. Extratropical jet — Q700/Z500 plot showing mesoscale detail

For each case, plot a 5-row x 3-column grid:
  rows = methods (ours, bilinear, ModAFNO, S-DYff, ground truth)
  cols = (raw field, abs. error, error spectrum)

Driver loads:
  - One model checkpoint per method (passed via --ckpt-<method>)
  - ERA5 anchors at the chosen case datetime
  - Computes interpolation at h=3 (midpoint), compares to GT.

Output: paper/case_studies/<case_name>.png
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import WeatherHermiteLightningModule

CASES = {
    "hurricane_laura_2020": dict(
        date=dt.date(2020, 8, 26), hour=18, interior_h=3,
        bbox=(15, 35, -100, -75),  # (lat_min, lat_max, lon_min, lon_max)
        channels=["mslp", "u10", "v10"],
        title="Hurricane Laura (Aug 26 2020, 18:00–24:00 UTC, interp at +3h)",
    ),
    "siberian_heatwave_2020": dict(
        date=dt.date(2020, 6, 20), hour=12, interior_h=3,
        bbox=(50, 80, 70, 130),
        channels=["t2m", "T850", "Z700"],
        title="Siberian heatwave (Jun 20 2020, 12:00–18:00 UTC, interp at +3h)",
    ),
    "atlantic_jet_2020": dict(
        date=dt.date(2020, 12, 15), hour=6, interior_h=3,
        bbox=(30, 70, -40, 20),
        channels=["U700", "Q700", "Z700"],
        title="Atlantic jet (Dec 15 2020, 06:00–12:00 UTC, interp at +3h)",
    ),
}


def load_model(ckpt_path, device, channel_groups):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hparams = dict(ckpt.get("hyper_parameters", {}))
    state = ckpt.get("state_dict", ckpt)
    mt = hparams.get("model_type", "")
    if "dcae" in mt:
        boc = tuple(hparams.get("block_out_channels", ()))
        hparams["block_out_channels"] = boc if len(boc) >= 3 else (128, 256, 512)
        lpb = tuple(hparams.get("layers_per_block", (3, 3, 3)))
        hparams["layers_per_block"] = lpb[:3] if len(lpb) > 3 else (lpb or (3, 3, 3))
    for k in ("channel_groups", "_class_path", "model_type_save"):
        hparams.pop(k, None)
    hparams["channel_groups"] = channel_groups
    model = WeatherHermiteLightningModule(**hparams)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model, mt


def get_anchors(ds, year, month, day, hour, max_tau_hours=6):
    """Return entries matching the requested case window (any tau_h)."""
    import calendar
    doy0 = sum(calendar.monthrange(year, m)[1] for m in range(1, month)) + (day - 1)
    t0_hour_of_year = doy0 * 24 + hour
    # Index format: (year, t0_hour_of_year, tau_h, tau_norm) with tau_h ∈ {1..5}
    matches = [e for e in ds.index
               if e[0] == year and e[1] == t0_hour_of_year]
    if not matches:
        raise SystemExit(f"no test sample with y={year}, t0={t0_hour_of_year}h")
    return matches


def fmt_units(ch):
    return {"T": "K", "U": "m/s", "V": "m/s", "Q": "kg/kg", "Z": "m²/s²",
            "t2m": "K", "u10": "m/s", "v10": "m/s",
            "mslp": "Pa", "sst": "K", "tcc": "-", "tcwv": "kg/m²"}.get(
        ch[0] if ch[0] in "TUVQZ" else ch, "")


def crop_to_bbox(field, bbox, H, W):
    """Crop a (C, H, W) field to the given bbox (lat_min, lat_max, lon_min, lon_max).
    Lat is north-to-south in the array; longitude is 0..360."""
    lat_min, lat_max, lon_min, lon_max = bbox
    # Convert lon [-180,180] convention to [0,360]
    if lon_min < 0:
        lon_min += 360
    if lon_max < 0:
        lon_max += 360
    lat_full = np.linspace(89.75, -89.75, H) if H == 360 else np.linspace(90, -90, H)
    lon_full = np.linspace(0, 360, W, endpoint=False)
    i0 = int(np.argmin(np.abs(lat_full - lat_max)))
    i1 = int(np.argmin(np.abs(lat_full - lat_min))) + 1
    if lon_min < lon_max:
        j0 = int(np.argmin(np.abs(lon_full - lon_min)))
        j1 = int(np.argmin(np.abs(lon_full - lon_max))) + 1
        return field[..., i0:i1, j0:j1]
    # Wrap-around case
    j0 = int(np.argmin(np.abs(lon_full - lon_min)))
    j1 = int(np.argmin(np.abs(lon_full - lon_max))) + 1
    left = field[..., i0:i1, j0:]
    right = field[..., i0:i1, :j1]
    return np.concatenate([left, right], axis=-1)


def make_case_figure(case_name, case, methods_data, channel_names, out_path):
    """methods_data: dict of method_name -> dict(field_phys: (C, H, W), title)
    channel_names: list of all C channel names (for indexing).
    """
    chs = case["channels"]
    bbox = case["bbox"]
    H, W = next(iter(methods_data.values()))["field_phys"].shape[-2:]
    crops = {m: crop_to_bbox(d["field_phys"], bbox, H, W)
             for m, d in methods_data.items()}
    gt = crops["ground_truth"]
    n_rows = len(methods_data) - 1   # exclude GT (shown as overlay or rightmost col)
    fig, axes = plt.subplots(
        len(methods_data), len(chs), figsize=(len(chs) * 4.0, len(methods_data) * 3.0),
        sharex=True, sharey=True, squeeze=False,
    )
    method_order = ["ground_truth"] + [m for m in methods_data if m != "ground_truth"]
    for ri, method in enumerate(method_order):
        d = crops[method]
        for ci, ch in enumerate(chs):
            ax = axes[ri, ci]
            cidx = channel_names.index(ch)
            field = d[cidx]
            if ri == 0:
                # GT: show raw field
                vmax = float(np.percentile(np.abs(field - field.mean()), 99))
                im = ax.imshow(field, cmap="RdBu_r",
                               vmin=field.mean() - vmax, vmax=field.mean() + vmax,
                               extent=[bbox[2], bbox[3], bbox[0], bbox[1]],
                               origin="upper", aspect="auto")
                ax.set_title(f"{ch} ({fmt_units(ch)})  — GT", fontsize=9)
            else:
                # Error map
                err = field - gt[channel_names.index(ch)]
                vmax = float(np.percentile(np.abs(err), 99)) or 1e-6
                im = ax.imshow(err, cmap="RdBu_r", vmin=-vmax, vmax=+vmax,
                               extent=[bbox[2], bbox[3], bbox[0], bbox[1]],
                               origin="upper", aspect="auto")
                rmse = float(np.sqrt(np.mean(err**2)))
                ax.set_title(f"{methods_data[method]['title']}  RMSE={rmse:.3f}",
                             fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.04)
            ax.tick_params(labelsize=7)
            if ci == 0:
                ax.set_ylabel("lat")
            if ri == len(method_order) - 1:
                ax.set_xlabel("lon")
    fig.suptitle(case["title"], y=1.005, fontsize=12, fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--test-year", type=int, default=2020)
    ap.add_argument("--stats-path", default="data/json_stats_0p5.nc")
    ap.add_argument("--surface-stats-path", default="data/surface_stats_0p5.json")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--out-dir", default="paper/case_studies")
    ap.add_argument("--cases", nargs="+", default=list(CASES))
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="name=path pairs, e.g. ours=/path/last.ckpt bilinear=__bilinear__")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_map = {}
    for kv in args.ckpts:
        if "=" not in kv:
            raise SystemExit(f"--ckpts entry must be name=path, got: {kv}")
        n, p = kv.split("=", 1)
        ckpt_map[n] = p

    ds_base = ERA5MemmapDataset(
        memmap_dir=args.memmap_dir, years=[args.test_year], max_tau_hours=6,
        samples_per_date=4, train=False,
        eval_hours=list(range(7)), static_path=args.static_path,
        stats_path=args.stats_path, surface_stats_path=args.surface_stats_path,
    )
    channel_names = list(ds_base.channel_names) + list(ds_base.surface_variables)
    mu = torch.cat([ds_base.mu, ds_base.surface_mu]).view(1, -1, 1, 1)
    sigma = torch.cat([ds_base.sigma, ds_base.surface_sigma]).view(1, -1, 1, 1)

    out_root = Path(args.out_dir)

    # Load models
    print("Loading models...")
    models = {}
    for name, p in ckpt_map.items():
        if p == "__bilinear__":
            models[name] = None  # placeholder
            continue
        m, mt = load_model(p, device, ds_base.channel_groups)
        models[name] = m
        print(f"  {name}: {mt} ({sum(p.numel() for p in m.parameters()) / 1e6:.1f}M)")

    for case_name in args.cases:
        case = CASES[case_name]
        print(f"\n=== {case_name} ===")
        try:
            matches = get_anchors(
                ds_base, args.test_year,
                case["date"].month, case["date"].day, case["hour"])
        except SystemExit as e:
            print(f"  skip: {e}")
            continue
        # e[2] = tau_h (1..5); pick entry matching the requested interior hour.
        entry = next((e for e in matches if e[2] == case["interior_h"]), matches[0])
        sample = ds_base[ds_base.index.index(entry)]
        x0 = torch.from_numpy(np.ascontiguousarray(sample["x0"])).float().unsqueeze(0).to(device)
        xT = torch.from_numpy(np.ascontiguousarray(sample["x1"])).float().unsqueeze(0).to(device)
        target = torch.from_numpy(np.ascontiguousarray(sample["target"])).float().unsqueeze(0).to(device)
        static = torch.from_numpy(np.ascontiguousarray(sample["static"])).float().unsqueeze(0).to(device) \
            if "static" in sample else None
        h = case["interior_h"]
        tau = torch.full((1,), h / 6.0, device=device)
        cond = torch.full((1,), 6.0, device=device)

        methods_data = {}
        # Ground truth (physical units)
        tgt_phys = (target * sigma.to(device) + mu.to(device)).cpu().numpy()[0]
        methods_data["ground_truth"] = dict(field_phys=tgt_phys, title="Ground truth")

        for name, model in models.items():
            if model is None:  # bilinear
                pred = (1 - tau) * x0 + tau * xT
            else:
                with torch.no_grad():
                    out = model(x0, xT, tau, cond, static=static)
                pred = out[0] if isinstance(out, tuple) else out
            pred_phys = (pred * sigma.to(device) + mu.to(device)).cpu().numpy()[0]
            methods_data[name] = dict(field_phys=pred_phys, title=name)

        make_case_figure(case_name, case, methods_data, channel_names,
                         out_root / f"{case_name}.png")


if __name__ == "__main__":
    main()
