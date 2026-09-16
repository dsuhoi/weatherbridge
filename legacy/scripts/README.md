# `legacy/scripts/` — archived trainers and runners

Pre-Hydra entry-points kept here only for exact paper-cited reproducibility.
The canonical replacement is `python train.py` / `python eval.py` driven by
the `conf/` Hydra groups (see top-level `README.md` and `Makefile`).

| File | Why archived |
|---|---|
| `train_0p5_memmap.py` | Env-var-configured 6 h trainer; superseded by `train.py +legacy=train_weatherdcae_noskip_24ch_6yr` / `train.py model=<x>`. |
| `train_0p5_memmap_12h_oddskip.py` | 12 h oddskip trainer; superseded by `train.py +legacy=train_atm_vfi_12h_oddskip` and `conf/data/era5_0p5_12h.yaml`. |
| `train_0p5_memmap_6h_tiny_kd.py` | KD experiment trainer; never productionised, kept for the record. |
| `train_0p5_memmap_smoke.py` | Smoke-test trainer; replaced by `scripts/run_hydra_smoke_test.sh` + `trainer=smoke`. |
| `train_atm_vfi_12h_oddskip.py` | ATM-VFI 12 h trainer; superseded by `train.py +legacy=train_atm_vfi_12h_oddskip`. |
| `train_corrdiff_fm_weatherdcae.py` | Deprecated 6 h diffusion-correction experiment; negative result, not part of paper narrative. |
| `train_corrdiff_fm_weatherdcae_12h.py` | Deprecated 12 h variant of the above; negative result. |
| `train_weatherbridge_hybrid.py` | Experimental hybrid trainer; not used in headline runs. |
| `trainer_weather_hermite.py` | Lightning module used by all env-var trainers; replaced by `weather_time_interp/` module dispatch via Hydra. |
| `bilinear_baseline_12h.py` | Standalone bilinear-only 12 h eval; replaced by `tools/eval/numerical_baseline_eval.py`. |
| `evaluate_baselines.py` | Legacy 1° eval CLI; replaced by `eval.py +legacy=eval_2020_paper_baseline_24ch`. |
| `compare_ckpts_param_l2.py` | One-off checkpoint diff utility; kept for reproducibility audits. |
| `compare_hydra_verify.py` | Hydra-vs-legacy parity check; one-off during migration. |
| `verify_hydra_equivalence.py` | Same purpose, run-time differential check; one-off during migration. |
| `make_fig_bias_maps_4arch.py` | Earlier 4-arch bias-map figure; superseded by `scripts/make_fig_bias_maps_7arch.py`. |
| `visualize_tb_logs.py` | TensorBoard-log plotter; replaced by metrics JSON pipeline in `metrics/`. |
| `run_weather_hermite_quality.sh` | Shell launcher for `trainer_weather_hermite.py`; obsolete with Hydra. |

If a file needs to come back to active use, move it back to repo root or
`scripts/` and re-export from the Hydra config if appropriate.
