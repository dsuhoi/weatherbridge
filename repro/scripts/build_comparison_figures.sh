#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
cd "$ROOT"

journal_root="${WTI_JOURNAL_METRICS_ROOT:-metrics/journal_unified}"
spectra_root="${WTI_JOURNAL_SPECTRA_ROOT:-metrics/journal_spectra_v6}"
required=(
  "$journal_root/6h_2020/weatherbridge_detail_14m_6yr_ep8.json"
  "$journal_root/6h_2021/weatherbridge_detail_14m_6yr_ep8.json"
  "$journal_root/12h_2020/weatherbridge_detail_14m_3yr_ep10.json"
  "$spectra_root/12h_2020/weatherbridge_tau5.npz"
  "$spectra_root/12h_2020/weatherbridge_tau8.npz"
  "metrics/hres_same_trajectory_12h_v1/linear.json"
  "metrics/hres_same_trajectory_12h_v1/pixelattn_vfi.json"
  "metrics/hres_same_trajectory_12h_v1/weatherdcae_14m.json"
  "metrics/hres_same_trajectory_12h_v1/flow_spectral.json"
)
for path in "${required[@]}"; do
  test -s "$path" || {
    echo "missing canonical WeatherBridge figure input: $path" >&2
    exit 1
  }
done

export WTI_JOURNAL_METRICS_ROOT="$journal_root"
export WTI_JOURNAL_SPECTRA_ROOT="$spectra_root"
"$PYTHON" scripts/make_fig1_rmse_per_tau.py
"$PYTHON" scripts/make_fig2_acc_per_tau.py
"$PYTHON" scripts/make_fig_main_scores.py
"$PYTHON" scripts/make_fig_specific_channels_phys.py --mode all
"$PYTHON" scripts/make_fig3_spectra.py
"$PYTHON" scripts/make_fig_perclass.py --mode all
"$PYTHON" scripts/make_fig_hres_best_fields.py
