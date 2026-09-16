#!/bin/bash
# Resume WeatherDCAE NoSkip 6yr 12h training ep6→ep10 on cloud.ru DDP
set -e
cd /home/jovyan/dsuhoi/weather_time_interpolation
export PATH=/home/jovyan/.mlspace/envs/ai_scientist/bin:$PATH
export WTI_ROOT=/home/jovyan/dsuhoi/weather_time_interpolation
export MODEL_TYPE=dcae_adaln_residual_linear
export YEARS="2014 2015 2016 2017 2018 2019"
export VAL_YEARS=2020
export MAX_EPOCHS=10
export BATCH_SIZE=1
export VAL_BATCH_SIZE=1
export LR=5e-5
export LATENT_CHANNELS=256
export KEEP_24CH=1
export CKPT_EVERY_N_EPOCHS=1
export PRECISION=bf16-mixed
export NUM_WORKERS=4
export DEVICES=2
export DCAE_BLOCK_CHANNELS=128,256,512
export DCAE_LAYERS_PER_BLOCK=2,2,2
export OUT_DIR=logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19_resume_ep10
export RESUME_CKPT=logs/exp_12h_oddskip_dcae_noskip_cloudru_2014_19/last-v1.ckpt
exec /home/jovyan/.mlspace/envs/ai_scientist/bin/python -u scripts/train_0p5_memmap_12h_oddskip.py
