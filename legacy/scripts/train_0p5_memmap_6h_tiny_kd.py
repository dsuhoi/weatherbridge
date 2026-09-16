#!/usr/bin/env python3
"""WB-mini ~8M student + full Block-Level KD from 37M NoSkip teacher.

Trains on τ ∈ {1, 3, 5}; τ ∈ {2, 4} are honest held-out.

Architecture decisions (verified against actual teacher state_dict shapes —
the saved hparams.yaml has stale tuples; the real ckpt is 3-stage):
  Student: 3-stage NoSkip DC-AE
      block_out=(64, 128, 256), layers=(2,2,2), latent=128  → ~8M params
  Teacher: 3-stage NoSkip DC-AE (frozen)
      block_out=(128, 256, 512), layers=(2,2,2), latent=256 → ~37M params
      block_type=("ResBlock", "ResBlock", "EfficientViTBlock")

  Both 3-stage with same downsample factors (×2/×4/×8), so spatial sizes
  of intermediate features ALIGN at every stage. Block-KD pairs stage-i
  features via 1x1 conv projections (only ~200K extra trainable params).

Loss:
  L = L_GT (Aurora lat-weighted MSE)
    + λ_out·MSE(x̂_s, x̂_t)
    + λ_block·Σᵢ MSE(proj_enc_i(enc_s_i), enc_t_i)
    + λ_block·Σᵢ MSE(proj_dec_i(dec_s_i), dec_t_i)
    + λ_lat·MSE(proj_lat(z_s), z_t)
    + L_residual + L_anchor (inherited from parent _shared_step)

  Defaults: λ_out=0.3, λ_block=0.1, λ_lat=0.05  (env-overridable).

Env knobs:
  YEARS, MAX_EPOCHS, BATCH_SIZE, LR, NUM_WORKERS, OUT_DIR
  TEACHER_CKPT     (path; if empty → KD disabled, pure baseline at this size)
  LAMBDA_KD_OUT    default 0.3
  LAMBDA_KD_BLOCK  default 0.1
  LAMBDA_KD_LAT    default 0.05
"""
import os
import sys
import time

sys.path.insert(0, "/home/jovyan/dsuhoi/weather_time_interpolation")
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from weather_time_interp.memmap_dataset import ERA5MemmapDataset
from trainer_weather_hermite import (
    WeatherHermiteLightningModule,
    ERA5WeatherHermiteDataset,
)


N_KEEP = 24  # 20 PL + 4 surface (drops sst/tcc/tcwv)

# Stage-end indices in encoder.down_blocks / decoder.up_blocks
# (Holds for layers_per_block=(2,2,2) → 8 entries: B,B,D,B,B,D,B,B
#  stage ends are after the last block of each group: indices 1, 4, 7.)
KD_STAGE_INDICES = (1, 4, 7)


class TruncateChannelsWrapper(Dataset):
    def __init__(self, base, n_keep=N_KEEP):
        self.base = base
        self.n_keep = n_keep

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        out = self.base[idx]
        out["x0"] = out["x0"][..., : self.n_keep, :, :].clone()
        out["xT"] = out["xT"][..., : self.n_keep, :, :].clone()
        out["target"] = out["target"][..., : self.n_keep, :, :].clone()
        return out


class BlockKDStudentModule(WeatherHermiteLightningModule):
    """8M student + frozen 37M teacher.  Output KD + per-stage encoder/decoder
    KD + latent KD via 1x1 conv projections (student-ch → teacher-ch).

    Strategy:
      • Forward hooks on student & teacher modules capture per-stage features
        into two dicts (`_feats_s`, `_feats_t`).
      • Patched `model.forward` runs the (frozen, no-grad) teacher right after
        student forward, so both feature dicts are populated.
      • `_shared_step` is overridden minimally: call super to get the base
        loss, then add output-KD + block-KD + latent-KD terms.
    """

    def __init__(
        self,
        *args,
        teacher_model: nn.Module | None = None,
        student_chs=(64, 128, 256),
        teacher_chs=(128, 256, 512),
        student_latent: int = 128,
        teacher_latent: int = 256,
        lambda_kd_out: float = 0.05,
        lambda_kd_block: float = 0.5,
        lambda_kd_lat: float = 0.2,
        lambda_kd_at: float = 0.3,       # v3: attention transfer on bottleneck
        lambda_kd_temb: float = 0.5,     # v3: τ-pathway embedding KD
        lambda_kd_film: float = 0.2,     # v3: FiLM scale/shift KD
        kd_ramp_epochs: int = 2,         # v3: ramp factor 0.2→1.0 over N epochs
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_teacher", teacher_model)
        self._lambda_kd_out = float(lambda_kd_out)
        self._lambda_kd_block = float(lambda_kd_block)
        self._lambda_kd_lat = float(lambda_kd_lat)
        self._lambda_kd_at = float(lambda_kd_at)
        self._lambda_kd_temb = float(lambda_kd_temb)
        self._lambda_kd_film = float(lambda_kd_film)
        self._kd_ramp_epochs = int(kd_ramp_epochs)
        self._kd_active = teacher_model is not None and (
            self._lambda_kd_out + self._lambda_kd_block + self._lambda_kd_lat
            + self._lambda_kd_at + self._lambda_kd_temb + self._lambda_kd_film > 0.0
        )
        # Flag flips True during anchor/tau-smooth aux forwards so kd_forward
        # skips both teacher invocation and feature-cache overwrite. Result:
        # _feats_s / _feats_t / _last_x_hat always reflect the MAIN forward at
        # random τ — not the degenerate τ=0/τ=1 anchors.
        self._in_aux_forward = False
        # Cached student x_hat from main forward (for true output-space KD).
        self._last_x_hat_s: torch.Tensor | None = None
        self._last_x_hat_t: torch.Tensor | None = None
        # v3: extra KD signals.
        self._temb_s: torch.Tensor | None = None     # τ-pathway embedding (B, 256)
        self._temb_t: torch.Tensor | None = None
        self._film_s: dict = {}    # film_proj outputs per ResBlock hook
        self._film_t: dict = {}

        # 1x1 projections (Identity when channels match).
        def _proj(s, t):
            return nn.Conv2d(s, t, 1) if s != t else nn.Identity()

        # Encoder: stage_i feature has block_out[i] channels.
        self.proj_enc = nn.ModuleList(
            [_proj(s, t) for s, t in zip(student_chs, teacher_chs)]
        )
        # Decoder iterates in REVERSE (deepest stage first), so the feature
        # at decoder hook position i has block_out[N-1-i] channels.
        self.proj_dec = nn.ModuleList(
            [_proj(s, t) for s, t in zip(reversed(student_chs), reversed(teacher_chs))]
        )
        self.proj_lat = _proj(student_latent, teacher_latent)
        # v3: FiLM time_emb_porj output projections. Each ResBlock's
        # time_emb_porj outputs 2*C dims (scale + shift). We KD-distill at
        # stage-end ResBlocks only (skipping EfficientViT stage which has
        # nested attn.time_emb_porj with different semantics):
        # enc stage-0 (idx 1): student 2*64=128, teacher 2*128=256
        # enc stage-1 (idx 4): student 2*128=256, teacher 2*256=512
        # dec stage-1 (idx 4): same as enc stage-1
        # dec stage-0 (idx 7): same as enc stage-0
        def _lin(s, t):
            return nn.Linear(s, t) if s != t else nn.Identity()
        self.proj_film_enc = nn.ModuleList([
            _lin(2*student_chs[0], 2*teacher_chs[0]),
            _lin(2*student_chs[1], 2*teacher_chs[1]),
        ])
        self.proj_film_dec = nn.ModuleList([
            _lin(2*student_chs[1], 2*teacher_chs[1]),  # dec_film_1 at up_blocks[4]
            _lin(2*student_chs[0], 2*teacher_chs[0]),  # dec_film_2 at up_blocks[7]
        ])
        # v3: τ-embedding pathway: time_mlp output is (B, time_emb_dim=256)
        # for both student and teacher (same default), so no projection.

        # Always init feature caches — _shared_step accesses them
        # unconditionally; in NoKD mode they stay empty.
        self._feats_s: dict = {}
        self._feats_t: dict = {}
        if self._kd_active:
            teacher_model.eval()
            for p in teacher_model.parameters():
                p.requires_grad_(False)
            self._patch_forward_and_hooks()

    def _patch_forward_and_hooks(self) -> None:
        """Register hooks on student & teacher stage-end blocks + latent, and
        wrap student model forward to also run teacher in no_grad."""

        def mk_hook(store, key):
            def hook(mod, inputs, output):
                store[key] = output
            return hook

        # IMPORTANT — hook indices vs anchor protection:
        # The hooks fire on EVERY forward of the wrapped module. To prevent
        # anchor / tau-smooth aux forwards from polluting the cache, we read
        # `self._in_aux_forward` inside each hook and skip the store. The flag
        # is flipped True in `kd_forward` when the parent _shared_step issues
        # extra forwards (after the main forward already ran).
        store_s = self._feats_s
        store_t = self._feats_t
        outer = self

        def mk_hook_guarded(store, key):
            def hook(mod, inputs, output):
                if outer._in_aux_forward:
                    return
                store[key] = output
            return hook

        # Student hooks
        for i, idx in enumerate(KD_STAGE_INDICES):
            self.model.encoder.down_blocks[idx].register_forward_hook(
                mk_hook_guarded(self._feats_s, f"enc_{i}")
            )
            self.model.decoder.up_blocks[idx].register_forward_hook(
                mk_hook_guarded(self._feats_s, f"dec_{i}")
            )
        # FIX 2: hook entire encoder (not conv_out) — captures full latent
        # z = conv_out(x) + out_shortcut(x). That's the tensor actually fed to
        # the decoder; out_shortcut=True is the WeatherDCAEAdaLNModel default.
        self.model.encoder.register_forward_hook(
            mk_hook_guarded(self._feats_s, "latent")
        )

        # Teacher hooks
        for i, idx in enumerate(KD_STAGE_INDICES):
            self._teacher.encoder.down_blocks[idx].register_forward_hook(
                mk_hook_guarded(self._feats_t, f"enc_{i}")
            )
            self._teacher.decoder.up_blocks[idx].register_forward_hook(
                mk_hook_guarded(self._feats_t, f"dec_{i}")
            )
        self._teacher.encoder.register_forward_hook(
            mk_hook_guarded(self._feats_t, "latent")
        )

        # v3: τ-pathway embedding hooks (single vector each forward).
        def temb_hook(store_name):
            def hook(mod, inputs, output):
                if outer._in_aux_forward:
                    return
                if store_name == "s":
                    outer._temb_s = output
                else:
                    outer._temb_t = output
            return hook
        self.model.time_mlp.register_forward_hook(temb_hook("s"))
        self._teacher.time_mlp.register_forward_hook(temb_hook("t"))

        # v3: FiLM time_emb_porj hooks at stage-end ResBlock positions.
        # Encoder: down_blocks[1] (stage 0 end), down_blocks[4] (stage 1 end)
        # Decoder: up_blocks[4] (stage 1 end), up_blocks[7] (stage 0 end)
        film_keys_enc = [("enc_film_0", 1), ("enc_film_1", 4)]
        film_keys_dec = [("dec_film_0", 4), ("dec_film_1", 7)]
        for key, idx in film_keys_enc:
            if hasattr(self.model.encoder.down_blocks[idx], "time_emb_porj"):
                self.model.encoder.down_blocks[idx].time_emb_porj.register_forward_hook(
                    mk_hook_guarded(self._film_s, key))
            if hasattr(self._teacher.encoder.down_blocks[idx], "time_emb_porj"):
                self._teacher.encoder.down_blocks[idx].time_emb_porj.register_forward_hook(
                    mk_hook_guarded(self._film_t, key))
        for key, idx in film_keys_dec:
            if hasattr(self.model.decoder.up_blocks[idx], "time_emb_porj"):
                self.model.decoder.up_blocks[idx].time_emb_porj.register_forward_hook(
                    mk_hook_guarded(self._film_s, key))
            if hasattr(self._teacher.decoder.up_blocks[idx], "time_emb_porj"):
                self._teacher.decoder.up_blocks[idx].time_emb_porj.register_forward_hook(
                    mk_hook_guarded(self._film_t, key))

        # Patch student model.forward to also run teacher (no_grad) and cache
        # both x_hat tensors. Aux forwards (anchor, tau-smooth) come in with
        # `self._in_aux_forward=True` — skip teacher + skip x_hat caching.
        orig_forward = self.model.forward

        def kd_forward(x0, xT, tau, cond, static=None):
            x_hat, aux = orig_forward(x0, xT, tau, cond, static=static)
            if outer.training and not outer._in_aux_forward:
                with torch.no_grad():
                    t_out = outer._teacher(x0, xT, tau, cond, static=static)
                teacher_pred = t_out[0] if isinstance(t_out, tuple) else t_out
                # Cache for true output-space KD; x_hat is still differentiable.
                outer._last_x_hat_s = x_hat
                outer._last_x_hat_t = teacher_pred.detach()
                if isinstance(aux, dict):
                    aux["x_hat_teacher"] = teacher_pred.detach()
            return x_hat, aux

        self.model.forward = kd_forward

    # ------------------- step override ----------------------

    def _shared_step(self, batch, stage):
        # Reset cached features each new batch — the main forward of the
        # parent _shared_step will repopulate them; subsequent aux forwards
        # are silenced via `_in_aux_forward`.
        self._feats_s.clear()
        self._feats_t.clear()
        self._film_s.clear()
        self._film_t.clear()
        self._last_x_hat_s = None
        self._last_x_hat_t = None
        self._temb_s = None
        self._temb_t = None
        self._in_aux_forward = False

        # The parent's _shared_step issues a main forward first, then aux
        # forwards (anchor / tau-smooth). We patch `self.forward` so the
        # SECOND and later forwards inside this step flip `_in_aux_forward`
        # to True (and hooks then silently bypass the cache, no teacher run).
        n_calls = {"v": 0}
        orig_self_forward = self.forward

        def main_then_aux_forward(x0, xT, tau, cond, static=None):
            n_calls["v"] += 1
            if n_calls["v"] > 1:
                self._in_aux_forward = True
            try:
                return orig_self_forward(x0, xT, tau, cond, static=static)
            finally:
                pass

        self.forward = main_then_aux_forward  # type: ignore
        try:
            base_loss = super()._shared_step(batch, stage)
        finally:
            self.forward = orig_self_forward  # type: ignore
            self._in_aux_forward = False

        if stage != "train" or not self._kd_active:
            return base_loss
        if self._last_x_hat_s is None or self._last_x_hat_t is None:
            return base_loss

        kd_loss = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

        # --- true output-space KD: pixel-space MSE on x_hat ---
        if self._lambda_kd_out > 0:
            kd_out = F.mse_loss(self._last_x_hat_s, self._last_x_hat_t)
            kd_loss = kd_loss + self._lambda_kd_out * kd_out
            self.log("train/kd_out", kd_out, on_step=True, on_epoch=True,
                     prog_bar=True, sync_dist=False)

        # --- block KD: averaged MSE across 3 encoder + 3 decoder stages ---
        if self._lambda_kd_block > 0:
            kd_blocks = []
            for i in range(len(KD_STAGE_INDICES)):
                fs = self._feats_s.get(f"enc_{i}")
                ft = self._feats_t.get(f"enc_{i}")
                if fs is not None and ft is not None:
                    kd_blocks.append(F.mse_loss(self.proj_enc[i](fs), ft))
                fs = self._feats_s.get(f"dec_{i}")
                ft = self._feats_t.get(f"dec_{i}")
                if fs is not None and ft is not None:
                    kd_blocks.append(F.mse_loss(self.proj_dec[i](fs), ft))
            if kd_blocks:
                kd_block = torch.stack(kd_blocks).mean()
                kd_loss = kd_loss + self._lambda_kd_block * kd_block
                self.log("train/kd_block", kd_block, on_step=True, on_epoch=True,
                         prog_bar=False, sync_dist=False)

        # --- latent KD: full encoder output (z = conv_out + out_shortcut) ---
        if self._lambda_kd_lat > 0:
            zs = self._feats_s.get("latent")
            zt = self._feats_t.get("latent")
            if zs is not None and zt is not None:
                kd_lat = F.mse_loss(self.proj_lat(zs), zt)
                kd_loss = kd_loss + self._lambda_kd_lat * kd_lat
                self.log("train/kd_lat", kd_lat, on_step=True, on_epoch=True,
                         prog_bar=False, sync_dist=False)

        # --- v3 (fixed): Attention Transfer on bottleneck (EfficientViT stage) ---
        # Spatial attention map = ||F||² collapsed over channel dim. Compare
        # student vs teacher attention maps directly via MSE.
        # NOTE: previous version applied L2 normalisation per-sample which
        # collapsed MSE between unit-vectors of length ~1000 to ~0.002, making
        # the AT gradient negligible (effective contribution → 0). Removing
        # the normalisation restores a meaningful signal. To keep magnitudes
        # comparable across stages we still divide by mean(||F_t||²) so the
        # scale doesn't vary wildly with channel count.
        if self._lambda_kd_at > 0:
            at_terms = []
            for key in ("enc_2", "dec_0"):
                fs = self._feats_s.get(key); ft = self._feats_t.get(key)
                if fs is None or ft is None:
                    continue
                a_s = fs.pow(2).mean(dim=1)              # (B, H, W)
                a_t = ft.pow(2).mean(dim=1).detach()
                # Scale invariance: normalise both maps by the teacher's mean
                # magnitude so the loss is scale-comparable across stages.
                denom = a_t.detach().mean().clamp_min(1e-6)
                at_terms.append(F.mse_loss(a_s / denom, a_t / denom))
            if at_terms:
                kd_at = torch.stack(at_terms).mean()
                kd_loss = kd_loss + self._lambda_kd_at * kd_at
                self.log("train/kd_at", kd_at, on_step=True, on_epoch=True,
                         prog_bar=False, sync_dist=False)

        # --- v3: τ-pathway embedding KD ---
        # Single vector per batch; both student and teacher use time_emb_dim=256
        # so no projection. MSE on the 256-d vector.
        if self._lambda_kd_temb > 0 and self._temb_s is not None and self._temb_t is not None:
            kd_temb = F.mse_loss(self._temb_s, self._temb_t.detach())
            kd_loss = kd_loss + self._lambda_kd_temb * kd_temb
            self.log("train/kd_temb", kd_temb, on_step=True, on_epoch=True,
                     prog_bar=False, sync_dist=False)

        # --- v3: FiLM scale/shift KD (only at ResBlock stage ends) ---
        if self._lambda_kd_film > 0:
            film_terms = []
            for proj, key in [
                (self.proj_film_enc[0], "enc_film_0"),
                (self.proj_film_enc[1], "enc_film_1"),
                (self.proj_film_dec[0], "dec_film_0"),
                (self.proj_film_dec[1], "dec_film_1"),
            ]:
                fs = self._film_s.get(key); ft = self._film_t.get(key)
                if fs is None or ft is None:
                    continue
                film_terms.append(F.mse_loss(proj(fs), ft))
            if film_terms:
                kd_film = torch.stack(film_terms).mean()
                kd_loss = kd_loss + self._lambda_kd_film * kd_film
                self.log("train/kd_film", kd_film, on_step=True, on_epoch=True,
                         prog_bar=False, sync_dist=False)

        # --- v3: ramp KD signal up from 0.2 → 1.0 over first N epochs ---
        if self._kd_ramp_epochs > 0:
            ramp = min(1.0, 0.2 + (1.0 - 0.2) * self.current_epoch / self._kd_ramp_epochs)
            kd_loss = kd_loss * ramp
            self.log("train/kd_ramp", torch.tensor(ramp), on_step=False, on_epoch=True,
                     prog_bar=False, sync_dist=False)

        self.log("train/loss_kd_total", kd_loss, on_step=True, on_epoch=True,
                 prog_bar=True, sync_dist=False)
        return base_loss + kd_loss


def build_kwargs(model_type: str, block_out, layers_per_block, latent_channels,
                 cg_24, lr: float):
    return dict(
        model_type=model_type,
        channel_groups=cg_24,
        n_pl_channels=20,
        n_surface_channels=4,
        n_static_features=3,
        latent_channels=latent_channels,
        block_out_channels=tuple(block_out),
        layers_per_block=tuple(layers_per_block),
        lat_weighted_loss=True,
        lat_crop=-8,
        lambda_residual=1.0,
        residual_scale_floor=0.05,
        residual_scale_init=0.30,
        lambda_anchor=0.5,
        anchor_every_n_batches=16,
        lr=lr,
        weight_decay=1e-5,
        use_aurora_weights=True,
        use_physical_scales_loss=False,
        max_tau_hours=6,
    )


def load_teacher(ckpt_path: str, cg_24) -> nn.Module:
    """Reconstruct 37M NoSkip teacher and load weights."""
    kwargs = build_kwargs(
        model_type="dcae_adaln_residual_linear",
        block_out=(128, 256, 512),
        layers_per_block=(2, 2, 2),
        latent_channels=256,
        cg_24=cg_24, lr=1e-4,
    )
    teacher = WeatherHermiteLightningModule(**kwargs)
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("state_dict", sd)
    missing, unexpected = teacher.load_state_dict(state, strict=False)
    print(f"[teacher] missing={len(missing)} unexpected={len(unexpected)} "
          f"(first 3 missing: {missing[:3]})")
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher.model  # inner WeatherDCAEAdaLNModel


def main():
    years = list(map(int, os.environ.get("YEARS",
                                          "2014 2015 2016 2017 2018 2019").split()))
    max_epochs = int(os.environ.get("MAX_EPOCHS", "10"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    lr = float(os.environ.get("LR", "1e-4"))
    num_workers = int(os.environ.get("NUM_WORKERS", "8"))
    teacher_ckpt = os.environ.get("TEACHER_CKPT", "")
    lambda_kd_out = float(os.environ.get("LAMBDA_KD_OUT", "0.05"))
    lambda_kd_block = float(os.environ.get("LAMBDA_KD_BLOCK", "0.5"))
    lambda_kd_lat = float(os.environ.get("LAMBDA_KD_LAT", "0.2"))
    lambda_kd_at = float(os.environ.get("LAMBDA_KD_AT", "0.3"))
    lambda_kd_temb = float(os.environ.get("LAMBDA_KD_TEMB", "0.5"))
    lambda_kd_film = float(os.environ.get("LAMBDA_KD_FILM", "0.2"))
    kd_ramp_epochs = int(os.environ.get("KD_RAMP_EPOCHS", "2"))
    out_dir = os.environ.get("OUT_DIR",
                              "logs/exp_dcae_noskip_mini_8M_6yr_blockkd")

    print("=== WB-mini (8M) + Block-Level KD from 37M ===")
    print(f"  years={years}, epochs={max_epochs}, batch={batch_size}, lr={lr}")
    print(f"  teacher_ckpt={teacher_ckpt or '(disabled — pure baseline)'}")
    print(f"  λ_out={lambda_kd_out} λ_block={lambda_kd_block} λ_lat={lambda_kd_lat}")
    print(f"  λ_at={lambda_kd_at} λ_temb={lambda_kd_temb} λ_film={lambda_kd_film}")
    print(f"  kd_ramp_epochs={kd_ramp_epochs}")
    print(f"  out={out_dir}", flush=True)
    t0 = time.time()

    ds_base = ERA5MemmapDataset(
        memmap_dir="/tmp/wb2_0p5_cache",
        years=years,
        max_tau_hours=6,
        samples_per_date=4,
        train=True,
        train_hours=[1, 3, 5],
        static_path="data/static_features_0p5.pt",
        stats_path="data/json_stats_0p5.nc",
        surface_stats_path="data/surface_stats_0p5.json",
    )
    cg_full = ds_base.channel_groups
    cg_24 = {k: v for k, v in cg_full.items() if k not in ("sst", "tcc", "tcwv")}
    print(f"  channel_groups (24-ch): {list(cg_24.keys())}")

    ds_train = ERA5WeatherHermiteDataset(ds_base, delta_t_hours=6.0)
    ds_train = TruncateChannelsWrapper(ds_train, n_keep=N_KEEP)
    print(f"train samples: {len(ds_train)}", flush=True)

    loader = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    student_kwargs = build_kwargs(
        model_type="dcae_adaln_residual_linear",
        block_out=(64, 128, 256),
        layers_per_block=(2, 2, 2),
        latent_channels=128,
        cg_24=cg_24, lr=lr,
    )

    teacher_inner = None
    if teacher_ckpt:
        print(f"[teacher] loading {teacher_ckpt}", flush=True)
        teacher_inner = load_teacher(teacher_ckpt, cg_24)
        n_t = sum(p.numel() for p in teacher_inner.parameters()) / 1e6
        print(f"[teacher] params: {n_t:.1f}M (frozen)", flush=True)

    model = BlockKDStudentModule(
        **student_kwargs,
        teacher_model=teacher_inner,
        student_chs=(64, 128, 256),
        teacher_chs=(128, 256, 512),
        student_latent=128,
        teacher_latent=256,
        lambda_kd_out=lambda_kd_out if teacher_ckpt else 0.0,
        lambda_kd_block=lambda_kd_block if teacher_ckpt else 0.0,
        lambda_kd_lat=lambda_kd_lat if teacher_ckpt else 0.0,
        lambda_kd_at=lambda_kd_at if teacher_ckpt else 0.0,
        lambda_kd_temb=lambda_kd_temb if teacher_ckpt else 0.0,
        lambda_kd_film=lambda_kd_film if teacher_ckpt else 0.0,
        kd_ramp_epochs=kd_ramp_epochs if teacher_ckpt else 0,
    )

    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    n_proj = (
        sum(p.numel() for p in model.proj_enc.parameters() if p.requires_grad)
        + sum(p.numel() for p in model.proj_dec.parameters() if p.requires_grad)
        + sum(p.numel() for p in model.proj_lat.parameters() if p.requires_grad)
    ) / 1e6
    print(f"student total params: {n_total:.2f}M  trainable: {n_trainable:.2f}M"
          f"  (proj overhead: {n_proj:.2f}M)", flush=True)

    # Move teacher to device once Lightning has placed pl_module on device.
    callbacks = [ModelCheckpoint(dirpath=out_dir, save_last=True,
                                  save_top_k=-1, every_n_epochs=2)]
    if teacher_inner is not None:
        class _TeacherMover(pl.Callback):
            def on_fit_start(self, trainer, pl_module):
                if pl_module._teacher is not None:
                    pl_module._teacher.to(pl_module.device)
                    pl_module._teacher.eval()
        callbacks.append(_TeacherMover())

    trainer = pl.Trainer(
        max_epochs=max_epochs, log_every_n_steps=20,
        devices=1, accelerator="gpu", precision="bf16-mixed",
        limit_val_batches=0, num_sanity_val_steps=0,
        enable_checkpointing=True, enable_progress_bar=False,
        callbacks=callbacks, default_root_dir=out_dir,
    )
    trainer.fit(model, train_dataloaders=loader)
    print(f"=== DONE in {(time.time()-t0)/60:.1f} min ===")


if __name__ == "__main__":
    main()
