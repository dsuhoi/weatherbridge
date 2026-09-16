"""Sanity check Hier v3: param counts, forward shape, full model assembly."""
import sys, os
sys.path.insert(0, '/tmp')
sys.path.insert(0, '/home/jovyan/dsuhoi/weather_time_interpolation')
os.chdir('/home/jovyan/dsuhoi/weather_time_interpolation')

import importlib.util
spec = importlib.util.spec_from_file_location('hcav3', '/tmp/train_hier_compress_ae_v3.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import torch

# 1) Inner cascade only (CAPACIOUS: 6 attn blocks @ ffn_ratio=8)
inner_down = mod.InnerDown(in_ch=48, out_ch=128, attn_heads=4, ffn_ratio=8,
                            n_attn_blocks=6, n_outer_res=4, n_inner_res=2)
inner_up = mod.InnerUp(in_ch=128, out_ch=48, attn_heads=4, ffn_ratio=8,
                        n_attn_blocks=6, n_outer_res=4, n_inner_res=2)
n_d = sum(p.numel() for p in inner_down.parameters())
n_u = sum(p.numel() for p in inner_up.parameters())
print(f'InnerDown(48→128): {n_d/1e6:.3f}M params')
print(f'InnerUp(128→48):   {n_u/1e6:.3f}M params')
print(f'Inner cascade total: {(n_d+n_u)/1e6:.3f}M params')

x = torch.randn(2, 48, 46, 92)
z2 = inner_down(x)
print(f'forward: (48,46,92) -> {tuple(z2.shape[1:])}  expected (128, 23, 46)')
z1 = inner_up(z2)
print(f'reverse: {tuple(z2.shape[1:])} -> {tuple(z1.shape[1:])}  expected (48, 46, 92)')
assert z1.shape == x.shape, f'shape mismatch: {z1.shape}'

# 2) Compression ratio: input 24 ch * 360 * 720 = 6_220_800 → z2 128 * 23 * 46 = 135_424
n_input = 24 * 360 * 720
n_z2 = 128 * 23 * 46
print(f'Compression ratio: {n_input/n_z2:.1f}× (v3 z₂ 128ch)')
print(f'  vs v2 z₂ 96ch: {n_input/(96*23*46):.1f}×')
print(f'  vs P1 z₁ 48ch: {n_input/(48*46*92):.1f}×')

# 3) Full model assembly (P1, no ckpt) — verify state_dict layout
m1 = mod.HierCompressAEv2(
    phase=1, in_channels=24, latent_ch_phase1=48,
    lr=2e-4, lambda_spec=0.0, lambda_latent=0.0)
n_p1 = sum(p.numel() for p in m1.parameters())
print(f'\nFull P1 model: {n_p1/1e6:.2f}M params')

# 4) Phase 2 from existing P1 ckpt (unfrozen, v3)
P1_CKPT = '/home/jovyan/dsuhoi/weather_time_interpolation/logs/exp_hier_compress_v2_phase1_32x_3yr/last.ckpt'
if os.path.exists(P1_CKPT):
    m2 = mod.HierCompressAEv2(
        phase=2, phase1_ckpt=P1_CKPT,
        in_channels=24, latent_ch_phase1=48, latent_ch_phase2=128,
        lr=2e-4, lambda_spec=0.05, lambda_latent=0.5,
        unfreeze_phase1=True, lr_ratio_p1=0.1,
        attn_heads=4, ffn_ratio=8,
        n_attn_blocks=6, n_outer_res=4, n_inner_res=2)
    n_p2 = sum(p.numel() for p in m2.parameters())
    n_trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f'Full P2 v3 model: {n_p2/1e6:.2f}M  trainable: {n_trainable/1e6:.2f}M')

    # Forward pass test
    x = torch.randn(1, 24, 360, 720)
    tisr = torch.randn(1, 1, 360, 720)
    static = torch.randn(1, 3, 360, 720)
    with torch.no_grad():
        out, z1, z1_rec = m2.forward(x, tisr, static, return_z=True)
    print(f'Forward OK: x_rec={tuple(out.shape)}  z1={tuple(z1.shape)}  z1_rec={tuple(z1_rec.shape)}')
else:
    print(f'\nP1 ckpt missing: {P1_CKPT}')
