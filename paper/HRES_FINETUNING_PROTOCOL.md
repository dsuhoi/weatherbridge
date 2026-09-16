# HRES Weight Fine-Tuning Protocol

This protocol separates two experiments that must not be conflated. The paper's
current 27 kB HRES adapter estimates frozen residual coefficients and never
updates network weights. The planned experiment below fine-tunes WeatherBridge
and WeatherDCAE-14M parameters from their ERA5-trained checkpoints.

## Data Contract

The primary task remains six-hour interpolation. Both anchors are states from
one IFS HRES forecast trajectory, separated by six hours; ERA5 at the matching
interior valid time is the target. No future analysis is an input. Training uses
2017--2019 initialisations and query hours 1, 3 and 5. The 2020 split selects the
checkpoint using all five interior hours, so hours 2 and 4 test temporal
interpolation within the validation split.

The archive contains three 00 UTC initialisations per month for optimisation
and two per month for validation: 108 trajectories from 2017--2019 and 24 from
2020. From each trajectory we retain left-anchor leads 0, 24, 48, 72 and 96 h.
This yields 1,620 training queries and 600 validation queries while limiting
dependence between adjacent forecast leads. Native-grid missingness is handled
by a finite-only 2x2 mean requiring at least two available values and a
per-channel missing fraction no larger than $10^{-6}$; any output non-finite
value rejects the initialisation. This policy was fixed before model fitting.

The already inspected 2021 archive is development-only. Existing 2022 results
also cannot serve as a new independent-year test. A disjoint set of 24 2022
initialisations, fixed in `repro/hres_finetune_protocol.json`, provides only
date-level confirmation. A genuinely independent calendar-year claim requires
a later forecast archive.

## Matched Training

Both models receive exactly the same anchor pairs, targets, normalization,
seeds, batch size, learning rate and update budget. Fine-tuning starts from the
published six-year ERA5 checkpoint, updates all parameters for 10 epochs, and
does not use anchor swapping or the residual adapter. WeatherBridge retains its
declared local high-pass (0.05) and advected-field FFT (0.02) objectives;
WeatherDCAE retains its reconstruction objective with both weights set to
zero. These values were read from the ERA5 warm-start checkpoint contract and
fixed during preflight, before any HRES optimisation step. This preserves model
identity instead of tuning a new loss separately for each method.

Seed 202707 is fixed as the primary replicate; 202708 and 202709 measure
training variability. This designation is made before any confirmation-date
prediction is generated.

## Evaluation and Promotion

The primary endpoints are latitude-weighted normalized RMSE averaged over all
24 fields at hours 1--5 and at held hours 2 and 4. Comparators are the raw
WeatherBridge checkpoint, fine-tuned WeatherDCAE-14M and Linear Interp. The
spectral audit uses the same schema-v6 SHT implementation and predeclared
high-wavenumber range. For Q850 and U850 at hours 2 and 3, the primary replicate
must not increase mean log-shape error by more than 0.02 or reduce coherence by
more than 0.01 relative to raw WeatherBridge over degrees 80--180. Results enter
the main paper only if the frozen promotion rule passes; mixed or negative
results remain in the audit artifacts.

Training starts with `scripts/run_hres_finetune_6h_cloudru.sh`; the disjoint
confirmation and promotion audit use
`scripts/run_hres_finetune_confirmation_2022_cloudru.sh`.
