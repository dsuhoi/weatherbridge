# Hierarchical DC-AE training (24ch, 0.5°, TISR + static)

## Layout

| File | Purpose |
|---|---|
| `train_hier_compress_ae_v3.py` | v3: capacious inner cascade (6 attn × FFN8) + unfrozen Phase 1 + split-LR optimizer |
| `train_hier_compress_ae_v31.py` | v3.1 = v3 + 4-level latent split heads (planetary/synoptic/meso/local) |
| `hier_v3_sanity.py` | Param counts + forward-shape verification (CPU) for v3 |
| `hier_v31_sanity.py` | Sanity for v3.1: also verifies telescoping `decompose_z1_4lvl` is lossless |
| `read_tb_v31.py` | Read TensorBoard scalars from a running v3.x experiment (since stdout under nohup is buffered) |

## Phase scheme

- **Phase 1** (32× compression): 3-stage AutoencoderDC (diffusers), `latent_ch_phase1=48`. Trained first to convergence on 3yr (~8 ep). Ckpt → `logs/exp_hier_compress_v2_phase1_32x_3yr/last.ckpt` on cloudru.
- **Phase 2** (~47× compression): inner cascade (PixelUnshuffle 2× → 1×1 → transformer-style attn stack → PixelShuffle 2×) on z₂ = (128, h/2, w/2). Phase 1 is **unfrozen** (v3) and weights co-adapt with lr × `lr_ratio_p1`.

## Loss composition (Phase 2)

```
loss = pixel_mse * (lat × ch weights)               # main reconstruction
     + λ_latent · MSE(z₁_rec, z₁.detach())          # latent supervision (= 0.5)
     + λ_spec   · SH_high_freq(x_rec, x)            # spectral preservation (l>30, = 0.05)
     + λ_split  · Σ_k MSE(split_head_k(z₂_g_k), z₁_band_k)  # v3.1 only (= 0.3)
```

`z₁` band decomposition is the lossless **telescoping** sum:
```
z₁ = pool8↑ + (pool4↑ − pool8↑) + (pool2↑ − pool4↑) + (z₁ − pool2↑)
   = planetary + synoptic + meso + local
```

## Launch (cloud.ru, 2× A100 DDP)

Phase 1 was already trained — Phase 2 v3 example:
```bash
python train_hier_compress_ae_v3.py \
  --phase 2 \
  --phase1_ckpt logs/exp_hier_compress_v2_phase1_32x_3yr/last.ckpt \
  --years 2014 2015 2016 2017 2018 2019 \
  --max_epochs 10 --bs 4 --lr 2e-4 \
  --gpus 0 1 --precision 32-true \
  --exp_name exp_hier_v3_capacious2_6yr
```

Phase 2 v3.1 example (4-level latent split, λ_split=0.3):
```bash
python train_hier_compress_ae_v31.py \
  --phase 2 \
  --phase1_ckpt logs/exp_hier_compress_v2_phase1_32x_3yr/last.ckpt \
  --years 2014 2015 2016 2017 2018 2019 \
  --max_epochs 10 --bs 4 --lr 2e-4 \
  --lambda_split 0.3 --split_groups 24 40 40 24 \
  --gpus 0 1 --precision 32-true \
  --exp_name exp_hier_v31_split4lvl_6yr
```

## Reading metrics during training

Lightning's `nohup` stdout is line-buffered for the process, so `tail -f <log>` shows print-statements only after `on_validation_epoch_end` (i.e. once per epoch). Per-step `train_loss`, `pixel_loss`, `latent_loss`, `sh_hf`, and `split_*` go to TensorBoard event files. Use `read_tb_v31.py` (or `tensorboard --logdir`) for mid-epoch progress.

## Validation channels (24)

```
T1000, T925, T850, T700
U1000, U925, U850, U700
V1000, V925, V850, V700
Q1000, Q925, Q850, Q700
Z1000, Z925, Z850, Z700
t2m, u10, v10, mslp
```

Channel weights (in `CH_WEIGHTS` dict) prioritise t2m (3.0), Q-levels (1.0-1.5), mslp (1.5) over upper-tropospheric winds (0.5-1.2).

## Next: v3.2 = v3.1 + WeatherGAN

Planned: PatchGAN discriminator (~5-6M params, spectral-norm), hinge GAN + R1 grad penalty, Lightning `automatic_optimization=False`. Implementation will live alongside as `train_hier_compress_ae_v32.py` and discriminator module `weather_patch_gan.py`. Expected RMSE drop ~10-15% on top of v3.1, but ~30% risk of GAN divergence requiring re-tune.
