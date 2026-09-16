# WTI Hydra entry-point shortcuts.
#
# Usage:
#   make train MODEL=weatherdcae           # generic train (override args via VAR=...)
#   make eval EVAL=memmap_6h MODELS='dcae_skip:logs/.../last.ckpt'
#   make smoke-test [GPU=1]                # full Hydra smoke suite on fibo
#   make legacy-train-dcae-skip            # paper SOTA: DC-AE Skip 14.4M 6yr
#   make legacy-train-weatherdcae          # paper SOTA: WeatherDCAE NoSkip 37.5M 6yr
#   make legacy-train-atm-vfi              # paper SOTA 12h: ATM-VFI
#   make legacy-eval-2020                  # paper 6h baseline eval table
#
PYTHON ?= python
MODEL ?= weatherdcae
TRAINER ?= default
DATA ?= era5_0p5_6h
EVAL ?= memmap_6h
GPU ?= 1
MODELS ?=

.PHONY: train eval smoke-test prod-grid-test \
        legacy-train-dcae-skip legacy-train-weatherdcae \
        legacy-train-atm-vfi legacy-train-corrdiff-fm-6h legacy-train-corrdiff-fm-12h \
        legacy-eval-2020 help

train:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) train.py model=$(MODEL) trainer=$(TRAINER) data=$(DATA)

eval:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) eval.py eval=$(EVAL) data=$(DATA) ++eval.models='$(MODELS)'

smoke-test:
	bash scripts/run_hydra_smoke_test.sh $(GPU)

# Prod-grid smoke: S-DYff + ModAFNO on real 360x720 ERA5 memmap. Requires
# `/tmp/wb2_0p5_cache/wb2_2020.bin` on the host and a CUDA GPU; gated by the
# `prod_grid` pytest marker so it does not run as part of `make smoke-test`.
prod-grid-test:
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) -m pytest tests/test_hydra_prod_grid.py \
	    -m prod_grid -v --tb=short

legacy-train-dcae-skip:
	$(PYTHON) train.py +legacy=train_dcae_skip_24ch_6yr

legacy-train-weatherdcae:
	$(PYTHON) train.py +legacy=train_weatherdcae_noskip_24ch_6yr

legacy-train-atm-vfi:
	$(PYTHON) train.py +legacy=train_atm_vfi_12h_oddskip

legacy-train-corrdiff-fm-6h:
	$(PYTHON) train.py +legacy=train_corrdiff_fm_6h_3yr

legacy-train-corrdiff-fm-12h:
	$(PYTHON) train.py +legacy=train_corrdiff_fm_12h_6yr_ddp

legacy-eval-2020:
	$(PYTHON) eval.py +legacy=eval_2020_paper_baseline_24ch

help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | grep -v '^\.PHONY' | sed 's/:.*//' | sed 's/^/  make /'
