"""Eval a capacity-matched CapMatchedLit checkpoint: cross-source anchor
(HRES 2021 × ERA5 2021) + tropical diurnal amplitude (2020 val).

Loads the CapMatchedLit wrapper directly (rebuilds the backbone via the
trainer's build_net, loads state_dict, reuses its ``_forward`` so the
inference path is identical to training — no bare-blob roundtrip, works
for both the WB-family (x0,xT,tau,cond,static) and ATM-VFI (x0,xT,tau)).

  python eval_capmatched.py --ckpt .../last.ckpt --arch mamba \
    --out-prefix metrics/capmatched/wb_mamba
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CH = ["T1000","T925","T850","T700","U1000","U925","U850","U700",
      "V1000","V925","V850","V700","Q1000","Q925","Q850","Q700",
      "Z1000","Z925","Z850","Z700","t2m","u10","v10","mslp"]
SURF = {"t2m": 20, "u10": 21, "v10": 22, "mslp": 23}
REGIONS = {"sahel": ((10.,15.),(0.,5.)), "amazon": ((-5.,0.),(295.,300.)),
           "congo": ((-2.,3.),(22.,27.)), "se_aus": ((-35.,-30.),(145.,150.))}


def stds_vec():
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from paper_units import canonical_stds
    d = canonical_stds()
    return np.array([d[c] for c in CH], dtype=np.float32)


def load_model(ckpt, arch, static_path, device):
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tools" / "train"))
    from train_capacity_matched_6h import CapMatchedLit
    lit = CapMatchedLit(arch=arch, static_path=static_path, lr=1e-4,
                        delta_t=6.0, eval_taus=(1, 2, 3, 4, 5))
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["state_dict"] if "state_dict" in ck else ck
    lit.load_state_dict(sd, strict=False)
    return lit.to(device).eval()


def open_year(era5_dir, y):
    meta = json.loads((era5_dir / f"wb2_{y}.json").read_text())
    mm = np.memmap(str(era5_dir / f"wb2_{y}.bin"), dtype="float32", mode="r",
                   shape=tuple(meta["shape"]))
    return mm, meta


@torch.no_grad()
def predict(lit, a, b, tau_h, device, dt=6):
    tau = torch.tensor([tau_h / dt], device=device, dtype=torch.float32)
    x0 = torch.from_numpy(a[None]).to(device)
    xT = torch.from_numpy(b[None]).to(device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = lit._forward(x0, xT, tau)
    return out.float().cpu().numpy()[0]


def anchor_eval(lit, forecast_dir, era5_dir, stds, device, taus, max_inits):
    cache = {}
    def era5_hour(valid):
        y = int(str(valid)[:4])
        if y not in cache:
            cache[y] = open_year(era5_dir, y)
        mm, meta = cache[y]
        t0 = np.datetime64(f"{y}-01-01T00", "h")
        hi = int((np.datetime64(valid, "h") - t0) / np.timedelta64(1, "h"))
        if not (0 <= hi < meta["shape"][0]):
            return None
        return np.asarray(mm[hi, :24])

    sxx = {t: np.zeros(24) for t in taus}
    cnt = {t: np.zeros(24, dtype=np.int64) for t in taus}
    files = sorted(forecast_dir.glob("init_*.bin"))
    if max_inits > 0:
        files = files[:max_inits]
    for fi, bp in enumerate(files):
        meta = json.loads(bp.with_suffix(".json").read_text())
        arr = np.asarray(np.memmap(str(bp), dtype="float32", mode="r",
                                    shape=tuple(meta["shape"])))
        it = np.datetime64(meta["init_time"])
        leads = meta["lead_hours"]
        avail = np.array([c in meta["channels_available"] for c in CH])
        for i in range(len(leads) - 1):
            if leads[i+1] - leads[i] != 6:
                continue
            for t in taus:
                tgt = era5_hour(it + np.timedelta64(leads[i] + t, "h"))
                if tgt is None:
                    continue
                pred = predict(lit, arr[i], arr[i+1], t, device)
                sq = (pred - tgt) ** 2 / (stds[:, None, None] ** 2)
                m = avail & ~np.isnan(sq).any(axis=(1, 2))
                sxx[t] += np.where(m, np.nanmean(sq, axis=(1, 2)), 0.0)
                cnt[t] += m.astype(np.int64)
        if (fi+1) % 4 == 0:
            print(f"[anchor {fi+1}/{len(files)}]", flush=True)
    out = {}
    for t in taus:
        out[str(t)] = {"model": {f"rmse_norm_{c}": (float(np.sqrt(sxx[t][i]/cnt[t][i]))
                                                     if cnt[t][i] else None)
                                  for i, c in enumerate(CH)}}
    return {"per_tau": out}


def grid(H, W):
    return np.linspace(90, -90, H+1)[:H], np.linspace(0, 360, W+1)[:W]


def rslice(lat, lon, reg):
    (la, lb), (oa, ob) = reg
    li = np.where((lat >= la) & (lat <= lb))[0]
    oi = np.where((lon >= oa) & (lon <= ob))[0]
    return slice(li.min(), li.max()+1), slice(oi.min(), oi.max()+1)


def diurnal_eval(lit, era5_dir, device, year, month=7):
    mm, meta = open_year(era5_dir, year)
    T, _, H, W = meta["shape"]
    lat, lon = grid(H, W)
    sl = {r: rslice(lat, lon, REGIONS[r]) for r in REGIONS}
    sh = int((np.datetime64(f"{year}-{month:02d}-01T00", "h")
              - np.datetime64(f"{year}-01-01T00", "h")) / np.timedelta64(1, "h"))
    eh = int((np.datetime64(f"{year}-{month+1:02d}-01T00", "h")
              - np.datetime64(f"{year}-01-01T00", "h")) / np.timedelta64(1, "h"))
    eh = min(eh, T)
    surf_idx = [SURF["t2m"], SURF["u10"], SURF["v10"]]
    truth = {r: np.asarray(mm[sh:eh, surf_idx, sl[r][0], sl[r][1]]) for r in REGIONS}
    recon = {r: truth[r].copy() for r in REGIONS}
    for h in range(0, eh - sh - 6, 6):
        a = np.asarray(mm[sh+h, :24]); b = np.asarray(mm[sh+h+6, :24])
        for t in (1, 2, 3, 4, 5):
            p = predict(lit, a, b, t, device)[surf_idx]
            for r in REGIONS:
                recon[r][h+t] = p[:, sl[r][0], sl[r][1]]

    def amp(x):
        n = x.shape[0] // 24
        prof = x[:n*24].reshape(n, 24).mean(0)
        return float(prof.max() - prof.min())
    out = {}
    for r in REGIONS:
        out[r] = {}
        for ci, c in enumerate(("t2m", "u10", "v10")):
            at = amp(truth[r][:, ci].mean(axis=(1, 2)))
            ar = amp(recon[r][:, ci].mean(axis=(1, 2)))
            out[r][c] = {"amp_true": at, "amp_recon": ar,
                          "amp_ratio": ar/at if at else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--forecast-dir",
                    default="/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021")
    ap.add_argument("--era5-dir",
                    default="/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_era5")
    ap.add_argument("--diurnal-era5-dir", default="/tmp/wb2_0p5_cache")
    ap.add_argument("--static-path", default="data/static_features_0p5.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-inits", type=int, default=12)
    args = ap.parse_args()

    lit = load_model(args.ckpt, args.arch, args.static_path, args.device)
    stds = stds_vec()
    print(f"[model] {args.arch} params={sum(p.numel() for p in lit.net.parameters())/1e6:.2f}M",
          flush=True)

    an = anchor_eval(lit, Path(args.forecast_dir), Path(args.era5_dir), stds,
                     args.device, [1, 2, 3, 4, 5], args.max_inits)
    Path(args.out_prefix + "_anchor.json").parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_prefix + "_anchor.json").write_text(json.dumps(an, indent=2))
    print(f"[write] {args.out_prefix}_anchor.json", flush=True)

    di = diurnal_eval(lit, Path(args.diurnal_era5_dir), args.device, 2020)
    Path(args.out_prefix + "_diurnal.json").write_text(json.dumps(di, indent=2))
    print(f"[write] {args.out_prefix}_diurnal.json", flush=True)


if __name__ == "__main__":
    main()
