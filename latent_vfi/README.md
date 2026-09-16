# `latent_vfi/` — Latent Residual VFI for ERA5 atmospheric interpolation

Self-contained module implementing the **3-phase latent VFI pipeline** for AAAI 2027 paper:

```
Phase 1 → Phase 2 → Phase 3
  AE       Residual    Latent VFI (deterministic)
                       └─ later: BBDM ensemble for probabilistic forecast
```

See **`docs/design.md`** for the full design rationale, including the bilinear-residual
ablation that addresses encoder/decoder reconstruction floor (see `docs/colleague_message.md`
for a concise summary suitable for forwarding).

## Directory layout

```
latent_vfi/
├── trainers/               # Training scripts for each phase + variants
│   ├── phase1_ae_cloudru.py            # baseline AE: f=16, lat_ch=64, 6yr  (2× A100)
│   ├── phase1_ae_fibo.py               # generic AE trainer, exposes --latent_channels
│   ├── phase1_ae_f32_fibo.py           # ablation: f=32 (10×24 latent), lat_ch=64, 3yr
│   ├── phase1_ae_f32_lat512_fibo.py    # ablation: f=32, lat_ch=512 (DiT-friendly grid)
│   ├── phase2_residual_simple.py       # Phase 2 without τ-conditioning (per-channel scale)
│   ├── phase2_adaln_zero.py            # Phase 2 with AdaLN-Zero τ-conditioning at latent
│   └── phase3_latent_vfi.py            # Phase 3 deterministic latent VFI + AdaLN-Zero
│
├── eval/                   # Offline evaluation utilities
│   ├── eval_phase2_per_tau.py          # per-τ × per-channel RMSE (paper plots)
│   ├── eval_ae_per_channel.py          # AE reconstruction quality (1-shot)
│   ├── encode_dataset_to_latents.py    # pre-extract latents for downstream training
│   ├── denorm_status.py                # convert normalized RMSE → physical units
│   ├── plateau_check.py                # late-slope analysis (have we converged?)
│   └── compare_data_equiv.py           # data-equivalent comparison (epochs × yr)
│
├── scripts/                # Cluster queue runners (chain Phase 1 → Phase 2 / extend)
│   ├── queue_phase2_adaln.sh           # cloud.ru: wait Phase 1 → launch Phase 2 AdaLN
│   ├── queue_phase2_adaln_fibo.sh      # fibo: same, gloo backend
│   └── queue_fibo_extend_phase1.sh     # fibo: extend Phase 1 from 10→20 epochs
│
└── docs/
    ├── design.md                       # full architectural rationale
    └── colleague_message.md            # short-form summary for sharing
```

## The 3-phase pipeline

### Phase 1 — DC-AE autoencoder for weather

Pure single-frame reconstruction:

```
x ──encoder──▶ z ──decoder──▶ x_rec
loss = lat-weighted L1 with LadCast-style SST sentinel masking
```

Key recipe (LadCast / DC-AE 1.5 conventions):
- **SST masking**: `torch.where(land_mask, sentinel, sst)` on both pred & target → zero
  loss contribution over land (avoids encoder wasting capacity on trivial land-SST values)
- **Static conditioning** into decoder: latent + downsampled `(lsm, orography, lat_cos)`
  → 1×1 projection back to latent_ch (identity-init, static gets learned weight)
- **Circular lon padding** at I/O boundary (existing `SphereConv2d` at conv_in/conv_out
  already does this; interior convs use DC-AE default zero-pad — see `design.md` for
  why deeper sphere-conv patch failed)
- **bf16-mixed precision**, AdamW(0.9, 0.95), lr 2e-4
- **DDP**: `gloo` backend on B300 (NCCL hangs on fibo at bootstrap), `nccl` on cloud.ru

Variants explored:
- `lat_ch=64, f=16` (cloud.ru baseline, 6yr, plateau at val/recon_l1 ≈ 0.074)
- `lat_ch=128, f=16` (fibo 3yr, 2× channel bandwidth — better fine-scale but worse smooth)
- `lat_ch=512, f=32` (fibo 3yr, 4× channel × 4× more compression — DiT-friendly grid)

### Phase 2 — Decoder residual fine-tune

Frozen encoder; decoder learns δ such that:

```
x_target = (1−τ)·x_0 + τ·x_T   +   scale · decoder(z_τ)
           └──── bilinear ────┘    └──── δ residual ────┘
```

Two flavors:
- `phase2_residual_simple.py` — per-channel `scale ∈ ℝ²⁷` learnable. No τ in decoder.
- `phase2_adaln_zero.py` — adds **AdaLN-Zero τ conditioning** at static_proj level via
  sinusoidal time embedding + MLP → (γ, β, gate). **Important**: γ and β init must be
  *non-zero* (small random), only gate=0 — otherwise gate gradient is dead and
  τ-conditioning never activates.

### Phase 3 — Latent VFI (deterministic predictor)

Frozen Phase 2; trains `LatentVFINet(z_0, z_T, τ) → z_τ_pred`:

```
z_lin = (1−τ)·z_0 + τ·z_T               # latent-space bilinear
z_τ_pred = z_lin + small UNet(τ-cond)   # residual to z_lin
```

Then frozen Phase 2 decoder produces δ, added to physical-space bilinear. The
PreservedAdaLN-Zero pattern is also used inside the UNet for τ conditioning.

`phase3_latent_vfi.py` logs **per-τ × per-channel** val RMSE (`val_rmse_h{1..5}/{ch}`)
out of the box — this is the paper-ready metric set.

## How to use

### Train Phase 1 on cloud.ru (DDP 2× A100)

```bash
python latent_vfi/trainers/phase1_ae_cloudru.py \
  --years 2014 2015 2016 2017 2018 2019 \
  --bs 4 --gpus 0 1 --max_epochs 10 \
  --exp_name exp_dcae_ae_static_6yr
```

### Train Phase 1 on fibo (DDP 2× B300, gloo backend)

```bash
DDP_BACKEND=gloo python latent_vfi/trainers/phase1_ae_fibo.py \
  --years 2017 2018 2019 \
  --bs 16 --gpus 0 1 --max_epochs 20 --lr 4e-4 \
  --latent_channels 64 \
  --exp_name exp_dcae_ae_f16_lat64_3yr_fibo_cosine
```

### Train Phase 2 with AdaLN-Zero (on top of Phase 1 ckpt)

```bash
PHASE1_TRAINER=latent_vfi/trainers/phase1_ae_cloudru.py \
python latent_vfi/trainers/phase2_adaln_zero.py \
  --phase1_ckpt logs/exp_dcae_ae_static_6yr/last.ckpt \
  --gpus 0 1 --max_epochs 8 \
  --exp_name exp_dcae_phase2_adaln_6yr
```

### Train Phase 3 (on top of Phase 2 ckpt)

```bash
PHASE2_TRAINER=latent_vfi/trainers/phase2_adaln_zero.py \
python latent_vfi/trainers/phase3_latent_vfi.py \
  --phase2_ckpt logs/exp_dcae_phase2_adaln_6yr/last.ckpt \
  --gpus 0 1 --max_epochs 10
```

### Offline per-τ × per-channel eval (for paper plots)

```bash
PHASE2_TRAINER=latent_vfi/trainers/phase2_adaln_zero.py \
CKPT=logs/exp_dcae_phase2_adaln_6yr/last.ckpt \
OUT_NAME=phase2_adaln_final \
python latent_vfi/eval/eval_phase2_per_tau.py
```

Output: `metrics/eval_phase2_per_tau/phase2_adaln_final.json` with
`per_hour_model`, `per_hour_bilinear`, `per_hour_relative_improvement` for all
(h ∈ {1,2,3,4,5}, channel ∈ 27) combinations.

## Key references

- **LadCast** (Zhu 2024) — SST sentinel + DC-AE adaptation: github.com/tonyzyl/ladcast
- **DC-AE 1.5 / Sana** (Han Lab) — hybrid ResBlock + EfficientViTBlock, f=32 latent
- **LDMVFI** (Danier, AAAI 2024) — VAE + LDM for video frame interpolation
- **TLB-VFI** (ICCV 2025) — Brownian-bridge in latent + 3D wavelet (Phase 3 candidate)
- **DYffusion / GenCast** — probabilistic atmospheric forecast in residual / latent space

## Known issues / TODO

- [ ] **AdaLN-Zero gate dead-init bug** fixed in `phase2_adaln_zero.py` and
      `phase3_latent_vfi.py`; older cloud.ru Phase 2 run has gate stuck at 0
      (functions as plain residual decoder, +35% improvement vs bilinear is still
      a valid lower-bound result)
- [ ] **Sphere-conv inside DC-AE blocks** failed (assertion clash with existing
      `weather_time_interp.model.sphere_conv.SphereConv2d`); current pipeline uses
      circular lon-pad only at I/O. Internal convs use zero-pad
- [ ] **Phase 3 BBDM** (probabilistic ensemble) is a future variant; current
      `phase3_latent_vfi.py` is deterministic. BBDM scaffold to be added.
- [ ] **NCCL DDP on fibo B300** hangs at bootstrap (CUDA-13 driver vs cu12.2 NCCL?);
      currently using `gloo` backend (~25% slower than NCCL but works)
- [ ] **Cosine LR schedule** in `phase1_ae_fibo.py` has 10% floor (not 1% standard);
      `phase3_latent_vfi.py` already uses 1% floor + clamped progress
