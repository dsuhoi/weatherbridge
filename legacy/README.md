# Checkpoint compatibility code

PixelAttn-VFI imports its encoder from `scripts/train_atm_vfi_12h_oddskip.py`.
Other retained trainers support the baseline comparisons.

`training_snapshots/` preserves source hashes recorded by training runs.
Change current experiments through `tools/train/`, not these snapshots.
