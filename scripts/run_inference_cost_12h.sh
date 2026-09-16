#!/usr/bin/env bash
# Cluster-side: measure FLOPs + CUDA latency for 12h leaderboard ckpts.
#
# Adapts tools/eval/inference_cost.py's measurement code for 12h paths.
# Output: metrics/inference_cost_12h_measured.json + appended into
# paper/tab_inference_cost_12h.tex (overwrites the static placeholder).
#
# Run on fibo (B300, 1 GPU) once active training jobs are not on the
# target GPU. Total ETA ~ 8 min (6 hermite models + 1 ATM-VFI).
set -euo pipefail

WTI_ROOT="${WTI_ROOT:-/workspace/code/wti}"
LOG="${LOG_DIR:-${WTI_ROOT}/logs}/inference_cost_12h_$(date -u +%Y%m%dT%H%M%SZ).log"

# Reuses tools/eval/inference_cost.py with --models list adjusted to 12h
# ckpts. The script's MODELS array currently encodes 6h ckpts; below we
# pass the 12h ones via env overrides at runtime.
python -c "
import json, os, sys, time, torch
sys.path.insert(0, '${WTI_ROOT}')
from pathlib import Path
from trainer_weather_hermite import WeatherHermiteLightningModule

MODELS = [
    ('DC-AE NoSkip (3yr)',
     '${WTI_ROOT}/logs/exp_12h_oddskip_dcae_noskip_3yr_fibo/last.ckpt', {}),
    ('DC-AE Skip (3yr)',
     '${WTI_ROOT}/logs/exp_12h_oddskip_dcae_2017_18_19/epoch=9-step=10940.ckpt', {}),
    ('FuXi',
     '${WTI_ROOT}/logs/exp_12h_oddskip_fuxi_2017_18_19/epoch=9-step=5470.ckpt', {}),
    ('ModAFNO',
     '${WTI_ROOT}/logs/exp_12h_oddskip_modafno_full_2017_18_19/epoch=9-step=21870.ckpt',
     {'MODAFNO_INP_H':'362','MODAFNO_INP_W':'720','MODAFNO_NATIVE_H':'360','MODAFNO_NATIVE_W':'720'}),
    ('S-DYff',
     '${WTI_ROOT}/logs/exp_12h_oddskip_sdyff_dyff_0p5_2017_18_19/epoch=9-step=5470.ckpt',
     {'SDYFF_NLAT':'360','SDYFF_NLON':'720','SDYFF_LAT_CROP':'0'}),
    ('WeatherDCAE NoSkip (6yr)',
     '${WTI_ROOT}/logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last.ckpt', {}),
]

device = torch.device('cuda')
results = []
for name, ckpt_path, env in MODELS:
    if not Path(ckpt_path).exists():
        print(f'  [skip-missing] {name}: {ckpt_path}')
        continue
    for k,v in env.items():
        os.environ[k] = v
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    hp = dict(ckpt.get('hyper_parameters', {}))
    state = ckpt.get('state_dict', ckpt)
    mt = hp.get('model_type','')
    if 'dcae' in mt:
        boc = tuple(hp.get('block_out_channels', ()))
        hp['block_out_channels'] = boc if len(boc)>=3 else (128,256,512)
        lpb = tuple(hp.get('layers_per_block', (3,3,3)))
        hp['layers_per_block'] = lpb[:3] if len(lpb)>3 else (lpb or (3,3,3))
    for k in ('channel_groups','_class_path','model_type_save'): hp.pop(k, None)
    hp['channel_groups'] = {}
    model = WeatherHermiteLightningModule(**hp); model.load_state_dict(state, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    B,C,H,W = 1,24,360,720
    x0 = torch.randn(B,C,H,W, device=device)
    xT = torch.randn(B,C,H,W, device=device)
    tau = torch.tensor([0.5], device=device)
    cond = torch.full((B,), 12.0, device=device)
    static = torch.zeros(B,3,H,W, device=device)

    # FLOPs
    flops = None
    try:
        from fvcore.nn import FlopCountAnalysis
        with torch.no_grad():
            fca = FlopCountAnalysis(model, (x0,xT,tau,cond,static))
            fca.unsupported_ops_warnings(False); fca.uncalled_modules_warnings(False)
            flops = int(fca.total())
    except Exception as e:
        print(f'  fvcore failed: {e}')

    # Latency
    with torch.no_grad():
        for _ in range(3): _ = model(x0,xT,tau,cond, static=static)
        torch.cuda.synchronize()
        lats = []
        for _ in range(10):
            s = torch.cuda.Event(enable_timing=True); e_ = torch.cuda.Event(enable_timing=True)
            s.record(); _ = model(x0,xT,tau,cond, static=static); e_.record()
            torch.cuda.synchronize()
            lats.append(s.elapsed_time(e_))
    mean_ms = sum(lats)/len(lats)
    std_ms = (sum((x-mean_ms)**2 for x in lats)/max(1,len(lats)-1))**0.5
    sps = 1000.0/mean_ms
    print(f'  {name}: params={n_params/1e6:.1f}M  FLOPs={flops/1e9 if flops else None:.2f}G  '
          f'lat={mean_ms:.1f}±{std_ms:.1f}ms  thr={sps:.2f} samples/s')
    results.append({'name': name, 'params_m': n_params/1e6,
                    'flops_g': (flops/1e9 if flops else None),
                    'latency_ms_mean': mean_ms, 'latency_ms_std': std_ms,
                    'throughput_sps': sps})
    del model; torch.cuda.empty_cache()

out = Path('${WTI_ROOT}/metrics/inference_cost_12h_measured.json')
out.write_text(json.dumps({'device': str(device), 'B':1, 'C':24, 'H':360, 'W':720,
                           'records': results,
                           'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())},
                          indent=2))
print(f'saved {out}')
" 2>&1 | tee "${LOG}"

echo ""
echo "Next: scp metrics/inference_cost_12h_measured.json back to local,"
echo "      then regenerate paper/tab_inference_cost_12h.tex by editing"
echo "      tools/eval/inference_cost_12h.py to read the measured JSON."
