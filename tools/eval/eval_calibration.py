"""Variance calibration + tail percentile + 1D radial spectrum eval.

Motivation for npj CAS reviewer story: aggregate RMSE hides two failure
modes that are individually critical for reanalysis-grade output —

  * Variance hallucination: model output std over-shooting the truth
    means the model injects spurious sub-6h variance. For downstream
    users (statistical downscaling, ensemble spread, extremes) this
    biases every histogram. `var_ratio = std(pred) / std(truth)` per
    channel — should be ≈ 1.00 for a faithful interpolator.
  * Tail-percentile RMSE: aggregate MSE is dominated by the bulk of
    the distribution. 95th and 99th percentiles of |pred - truth|
    expose tail behaviour, which matters for extreme-weather users.
  * 1D radial power spectrum ratio: model_power(k) / truth_power(k)
    at high wavenumbers reveals HF energy over-injection vs
    over-smoothing. `hf_ratio_ge_k = mean of power_ratio at k >= k*`.

Runs on HRES 2021 forecast anchors × ERA5 2021 truth, τ=3 midpoint,
subsampled to 12 inits × 40 pair × 5 τ = ~2400 samples for statistical
significance. All 24 canonical channels.

Output JSON per model:
  {
    "var_ratio_per_channel": {ch -> float},
    "p95_norm_per_channel":  {ch -> float},   # 95th percentile of |Δ|/σ
    "p99_norm_per_channel":  {ch -> float},
    "hf_power_ratio_ge_10":  {ch -> float},   # k>=10 model/truth ratio
    "hf_power_ratio_ge_50":  {ch -> float},
    "n_samples": int,
  }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

CHANNELS_ORDER = [
    "T1000", "T925", "T850", "T700",
    "U1000", "U925", "U850", "U700",
    "V1000", "V925", "V850", "V700",
    "Q1000", "Q925", "Q850", "Q700",
    "Z1000", "Z925", "Z850", "Z700",
    "t2m", "u10", "v10", "mslp",
]


def read_init(memmap_bin: Path, memmap_json: Path):
    meta = json.loads(memmap_json.read_text())
    arr = np.memmap(str(memmap_bin), dtype="float32", mode="r",
                    shape=tuple(meta["shape"]))
    return np.asarray(arr), meta["lead_hours"], meta


class Era5Cache:
    def __init__(self, era5_dir: Path):
        self.era5_dir = era5_dir
        self._maps: dict[int, tuple[np.memmap, dict]] = {}

    def _open(self, year: int):
        if year in self._maps:
            return self._maps[year]
        bin_p = self.era5_dir / f"wb2_{year}.bin"
        json_p = self.era5_dir / f"wb2_{year}.json"
        meta = json.loads(json_p.read_text())
        mm = np.memmap(str(bin_p), dtype="float32", mode="r",
                       shape=tuple(meta["shape"]))
        self._maps[year] = (mm, meta)
        return self._maps[year]

    def hour(self, valid_time: np.datetime64) -> np.ndarray | None:
        y = int(str(valid_time)[:4])
        try:
            mm, meta = self._open(y)
        except FileNotFoundError:
            return None
        t0 = np.datetime64(f"{y}-01-01T00", "h")
        hi = int((np.datetime64(valid_time, "h") - t0) / np.timedelta64(1, "h"))
        if not (0 <= hi < meta["shape"][0]):
            return None
        return np.asarray(mm[hi, :24])


def load_canonical_stds() -> np.ndarray:
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    from paper_units import canonical_stds
    d = canonical_stds()
    return np.array([d[c] for c in CHANNELS_ORDER], dtype=np.float32)


class LinearInterp:
    def __call__(self, x0, xT, tau_h, dt=6):
        w = tau_h / dt
        return (1 - w) * x0 + w * xT


class BareModel:
    def __init__(self, blob_path: str, device: str, stds: np.ndarray):
        if torch is None:
            raise RuntimeError("torch required")
        import sys
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        from examples._bare_loader import load_bare
        self.net = load_bare(blob_path, device)
        self.device = device
        self.stds = torch.from_numpy(stds).to(device).view(1, -1, 1, 1)
        self.static = None
        n_static = getattr(self.net, "n_static_features", 0)
        if n_static > 0:
            s = torch.load(str(root / "data" / "static_features_0p5.pt"),
                            weights_only=False).float()
            if s.dim() == 3:
                s = s.unsqueeze(0)
            self.static = s[:, :n_static].to(device)

    def __call__(self, x0, xT, tau_h, dt=6):
        with torch.no_grad():
            x0_t = torch.from_numpy(x0[None]).to(self.device) / self.stds
            xT_t = torch.from_numpy(xT[None]).to(self.device) / self.stds
            tau_t = torch.tensor([[tau_h / dt]], device=self.device, dtype=torch.float32)
            if type(self.net).__name__ == "PixelAttentionVFI":
                out = self.net.net(x0_t, xT_t, tau_t)
            else:
                cond_t = torch.tensor([float(dt)], device=self.device, dtype=torch.float32)
                kwargs = {"static": self.static} if self.static is not None else {}
                out = self.net(x0_t, xT_t, tau_t, cond_t, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            return (out * self.stds).cpu().numpy()[0]


def radial_power_spectrum_1d(field: np.ndarray) -> np.ndarray:
    """2D FFT power → binned radial spectrum. field (H, W) → (n_bins,)."""
    H, W = field.shape
    f = np.fft.rfft2(field)
    p = (f * f.conj()).real
    ky = np.fft.fftfreq(H, d=1.0) * H
    kx = np.fft.rfftfreq(W, d=1.0) * W
    kr = np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2)
    max_k = int(min(H, W) // 2)
    bins = np.arange(0, max_k + 1)
    idx = np.clip(kr.astype(int), 0, max_k)
    spec = np.bincount(idx.ravel(), weights=p.ravel(), minlength=max_k + 1)
    cnt = np.bincount(idx.ravel(), minlength=max_k + 1)
    spec = spec / np.maximum(cnt, 1)
    return spec[:max_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--forecast-dir", required=True)
    ap.add_argument("--era5-memmap-dir", required=True)
    ap.add_argument("--model-blob", required=True,
                    help="'linear' or *.pt bare blob")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau", type=int, default=3)
    ap.add_argument("--max-inits", type=int, default=12)
    args = ap.parse_args()

    forecast_dir = Path(args.forecast_dir)
    era5_cache = Era5Cache(Path(args.era5_memmap_dir))
    stds = load_canonical_stds()
    n_ch = len(CHANNELS_ORDER)

    if args.model_blob == "linear":
        interp = LinearInterp()
        model_key = "linear"
    else:
        interp = BareModel(args.model_blob, args.device, stds)
        model_key = "model"

    # accumulators per channel:
    # variance: E[pred²] - E[pred]², E[truth²] - E[truth]²
    # tail: collect flattened |pred-truth|/σ samples for quantile
    sum_pred = np.zeros(n_ch, dtype=np.float64)
    sum_pred2 = np.zeros(n_ch, dtype=np.float64)
    sum_truth = np.zeros(n_ch, dtype=np.float64)
    sum_truth2 = np.zeros(n_ch, dtype=np.float64)
    cnt = np.zeros(n_ch, dtype=np.int64)
    # Reservoir sampling for percentiles: keep up to N samples per channel
    RESERVOIR_N = 500_000
    tail_bufs = [np.zeros(0, dtype=np.float32) for _ in range(n_ch)]

    # Spectrum accumulators (mean over samples)
    max_k = 180
    spec_pred = np.zeros((n_ch, max_k), dtype=np.float64)
    spec_truth = np.zeros((n_ch, max_k), dtype=np.float64)
    n_spec = np.zeros(n_ch, dtype=np.int64)

    init_files = sorted(forecast_dir.glob("init_*.bin"))
    if args.max_inits > 0:
        init_files = init_files[: args.max_inits]
    print(f"[init] {len(init_files)} inits, tau={args.tau} model={model_key}", flush=True)

    for fi, bin_p in enumerate(init_files):
        json_p = bin_p.with_suffix(".json")
        arr, leads, meta = read_init(bin_p, json_p)
        init_time = np.datetime64(meta["init_time"])
        avail_mask = np.array([c in meta["channels_available"] for c in CHANNELS_ORDER])
        tau = args.tau
        for i in range(len(leads) - 1):
            lead_a, lead_b = leads[i], leads[i + 1]
            if lead_b - lead_a != 6:
                continue
            a = arr[i]; b = arr[i + 1]
            valid = init_time + np.timedelta64(lead_a + tau, "h")
            tgt = era5_cache.hour(valid)
            if tgt is None:
                continue
            pred = interp(a, b, tau)  # (n_ch, H, W)
            for ci in range(n_ch):
                if not avail_mask[ci]:
                    continue
                p = pred[ci]; t = tgt[ci]
                if np.isnan(p).any() or np.isnan(t).any():
                    continue
                sig = float(stds[ci])
                sum_pred[ci]   += p.mean()
                sum_pred2[ci]  += (p ** 2).mean()
                sum_truth[ci]  += t.mean()
                sum_truth2[ci] += (t ** 2).mean()
                cnt[ci] += 1

                # spectra: 1 sample = one image
                spec_pred[ci] += radial_power_spectrum_1d(p)[:max_k]
                spec_truth[ci] += radial_power_spectrum_1d(t)[:max_k]
                n_spec[ci] += 1

                # sample errors for percentiles — subsample for memory
                err = np.abs(p - t) / sig
                sub = err.ravel()[::10]  # every 10th pixel
                room = RESERVOIR_N - tail_bufs[ci].size
                if room > 0:
                    take = sub[: max(0, min(room, sub.size))]
                    tail_bufs[ci] = np.concatenate([tail_bufs[ci], take.astype(np.float32)])
        if (fi + 1) % 4 == 0 or fi == 0:
            print(f"[{fi+1}/{len(init_files)}] init {str(init_time)[:13]}", flush=True)

    # Compute stats
    var_ratio = {}
    p95, p99 = {}, {}
    hf10, hf50 = {}, {}
    for ci, c in enumerate(CHANNELS_ORDER):
        n = int(cnt[ci])
        if n == 0:
            var_ratio[c] = None; p95[c] = None; p99[c] = None
            hf10[c] = None; hf50[c] = None; continue
        mp = sum_pred[ci] / n; mp2 = sum_pred2[ci] / n
        mt = sum_truth[ci] / n; mt2 = sum_truth2[ci] / n
        vp = max(mp2 - mp ** 2, 1e-12)
        vt = max(mt2 - mt ** 2, 1e-12)
        var_ratio[c] = float(np.sqrt(vp / vt))  # std ratio
        p95[c] = float(np.quantile(tail_bufs[ci], 0.95))
        p99[c] = float(np.quantile(tail_bufs[ci], 0.99))
        ns = int(n_spec[ci])
        if ns:
            sp = spec_pred[ci] / ns
            st = spec_truth[ci] / ns
            ratio = sp / np.maximum(st, 1e-20)
            hf10[c] = float(np.mean(ratio[10:]))
            hf50[c] = float(np.mean(ratio[50:]))
        else:
            hf10[c] = None; hf50[c] = None

    out = {
        "model_key": model_key,
        "n_samples": int(cnt.max()),
        "std_ratio_per_channel": var_ratio,
        "p95_norm_per_channel": p95,
        "p99_norm_per_channel": p99,
        "hf_power_ratio_ge_k10": hf10,
        "hf_power_ratio_ge_k50": hf50,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
