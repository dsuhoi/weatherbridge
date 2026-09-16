"""Sanity check Hier v3.1: 4-level split heads + telescoping decomposition."""
import sys, os
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
os.chdir('/home/jovyan/dsuhoi/weather_time_interpolation')

import importlib.util
spec = importlib.util.spec_from_file_location('hcav31', '/tmp/train_hier_compress_ae_v31.py')
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

import torch, torch.nn.functional as F

# 1) telescoping decomposition lossless test
z1 = torch.randn(2, 48, 92, 184)
zp, zs, zm, zl = mod.decompose_z1_4lvl(z1)
recon = zp + zs + zm + zl
err = (recon - z1).abs().max().item()
print(f'4-level decomposition reconstruction error (max abs): {err:.2e}')
assert err < 1e-5, 'telescoping decomp not lossless'

# 2) full P2 v3.1 model build
P1_CKPT = '/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_hier_compress_v2_phase1_32x_3yr/last.ckpt'
m2 = mod.HierCompressAEv2(
    phase=2, phase1_ckpt=P1_CKPT,
    in_channels=24, latent_ch_phase1=48, latent_ch_phase2=128,
    lr=2e-4, lambda_spec=0.05, lambda_latent=0.5,
    unfreeze_phase1=True, lr_ratio_p1=0.1,
    attn_heads=4, ffn_ratio=8,
    n_attn_blocks=6, n_outer_res=4, n_inner_res=2,
    lambda_split=0.3, split_groups=(24, 40, 40, 24))
n_total = sum(p.numel() for p in m2.parameters())
n_trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
print(f'\nFull v3.1 P2 model: {n_total/1e6:.2f}M  trainable: {n_trainable/1e6:.2f}M')
print(f'split_heads: {[h.weight.shape for h in m2.split_heads]}')
n_heads = sum(p.numel() for p in m2.split_heads.parameters())
print(f'split_heads params: {n_heads/1e6:.4f}M')

# 3) Forward with return_splits
x = torch.randn(1, 24, 360, 720)
tisr = torch.randn(1, 1, 360, 720)
static = torch.randn(1, 3, 360, 720)
with torch.no_grad():
    x_rec, z1, z1_rec, splits = m2.forward(
        x, tisr, static, return_z=True, return_splits=True)
print(f'\nx_rec={tuple(x_rec.shape)}  z1={tuple(z1.shape)}  z1_rec={tuple(z1_rec.shape)}')
for band, p in zip(('planetary', 'synoptic', 'meso', 'local'), splits):
    print(f'  split[{band}]={tuple(p.shape)}')

# 4) Verify split heads produce zero on first forward (zero init)
splits_norms = [p.abs().max().item() for p in splits]
print(f'\nsplit heads outputs (should be 0 at init): {splits_norms}')
assert max(splits_norms) < 1e-6, 'split heads not zero-init!'

# 5) Compute split loss on actual data
z_planet, z_synop, z_meso, z_local = mod.decompose_z1_4lvl(z1)
targets = (z_planet, z_synop, z_meso, z_local)
losses = [F.mse_loss(p, t) for p, t in zip(splits, targets)]
print(f'\nSplit losses (init): planet={losses[0]:.4f} synop={losses[1]:.4f} '
      f'meso={losses[2]:.4f} local={losses[3]:.4f}')
print(f'Mean split loss: {sum(losses)/4:.4f}')
print('\nALL OK ✓')
