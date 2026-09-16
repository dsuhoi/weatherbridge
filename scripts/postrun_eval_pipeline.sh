#!/usr/bin/env bash
# Post-training evaluation pipeline for the AAAI 2027 submission.
# Run this AFTER both no-skip and DC-AE Skip +A+C trainings complete.
#
# Stages:
#   1. RMSE+ACC eval on 2020 for both new ckpts (4 days/month, 4 starts/day).
#   2. Energy spectra eval (radial E(k)) for both new ckpts + 4 existing models + GT.
#   3. Plot energy spectra mosaic.
#   4. Update tab:noskip-ablation with computed test RMSEs (fills TBD cells).
#   5. Optional: multi-year OOD eval on 2021 (--with-2021).
#
# Usage:
#   bash scripts/postrun_eval_pipeline.sh \
#       [--ac-ckpt /workspace-SR006.nfs2/.../exp_dcae_skip_AC_0p5_6yr/last.ckpt] \
#       [--noskip-ckpt /workspace-SR006.nfs2/.../exp_dcae_noskip_0p5_6yr/last.ckpt] \
#       [--with-2021] \
#       [--with-spectra] \
#       [--with-cases]
set -euo pipefail

cd "$(dirname "$0")/.."

AC_CKPT="/workspace-SR006.nfs2/weather_data/experiments/exp_dcae_skip_AC_0p5_6yr/last.ckpt"
NOSKIP_CKPT="/workspace-SR006.nfs2/weather_data/experiments/exp_dcae_noskip_0p5_6yr/last.ckpt"
SKIP_BASE_CKPT="/workspace-SR006.nfs2/weather_data/experiments/exp_dcae_skip_0p5_6yr_pad/last.ckpt"

WITH_2021=0
WITH_SPECTRA=0
WITH_CASES=0
SKIP_RMSE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ac-ckpt)      AC_CKPT="$2"; shift 2 ;;
    --noskip-ckpt)  NOSKIP_CKPT="$2"; shift 2 ;;
    --with-2021)    WITH_2021=1; shift ;;
    --with-spectra) WITH_SPECTRA=1; shift ;;
    --with-cases)   WITH_CASES=1; shift ;;
    --skip-rmse)    SKIP_RMSE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

LOGDIR="logs/postrun_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOGDIR"
echo "logs -> $LOGDIR"

run() {
  local tag="$1"; shift
  echo "=== [$tag] $* ===" | tee -a "$LOGDIR/00_master.log"
  "$@" 2>&1 | tee "$LOGDIR/${tag}.log"
}

# -----------------------------------------------------------------------------
# Stage 1: RMSE + ACC for both new ckpts
# -----------------------------------------------------------------------------
if [[ "$SKIP_RMSE" == "0" ]]; then
  echo ">>> Stage 1: RMSE+ACC on 2020 (new ckpts)"
  if [[ ! -f "$AC_CKPT" ]]; then
    echo "  AC ckpt missing: $AC_CKPT (skipping AC eval)"
  else
    run eval_AC python tools/eval/batch_eval_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year 2020 \
      --eval-days-per-month 4 \
      --samples-per-date 4 \
      --models "dcae_skip_AC:$AC_CKPT" \
      --out-rmse-dir metrics/eval_6h_2020_paper_leaderboard \
      --out-acc-dir  metrics/acc_6h_2020_paper_leaderboard
  fi
  if [[ ! -f "$NOSKIP_CKPT" ]]; then
    echo "  NoSkip ckpt missing: $NOSKIP_CKPT (skipping NoSkip eval)"
  else
    run eval_noskip python tools/eval/batch_eval_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache \
      --test-year 2020 \
      --eval-days-per-month 4 \
      --samples-per-date 4 \
      --models "dcae_noskip:$NOSKIP_CKPT" \
      --out-rmse-dir metrics/eval_6h_2020_paper_leaderboard \
      --out-acc-dir  metrics/acc_6h_2020_paper_leaderboard
  fi
fi

# -----------------------------------------------------------------------------
# Stage 2: Energy spectra (--with-spectra)
# -----------------------------------------------------------------------------
if [[ "$WITH_SPECTRA" == "1" ]]; then
  echo ">>> Stage 2: Energy spectra"
  for entry in \
      "dcae_skip_AC:$AC_CKPT" \
      "dcae_noskip:$NOSKIP_CKPT" \
      "dcae_skip:$SKIP_BASE_CKPT"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    if [[ ! -f "$path" ]]; then
      echo "  skip spectra for $name (ckpt missing: $path)"
      continue
    fi
    INCL_GT_FLAG=""
    [[ "$name" == "dcae_skip_AC" ]] && INCL_GT_FLAG="--include-ground-truth"
    run "spectra_$name" python tools/eval/energy_spectra.py \
      --ckpt "$path" \
      --method-name "$name" \
      --out-dir metrics/energy_spectra_0p5_2020 \
      $INCL_GT_FLAG
  done
  run plot_spectra python tools/eval/plot_energy_spectra.py \
    --npz-dir metrics/energy_spectra_0p5_2020 \
    --out-dir paper/energy_spectra
fi

# -----------------------------------------------------------------------------
# Stage 3: Update tab:noskip-ablation with real numbers
# -----------------------------------------------------------------------------
echo ">>> Stage 3: Update tab:noskip-ablation"
python - <<'PY' 2>&1 | tee "$LOGDIR/update_noskip_table.log"
import json
from pathlib import Path

def mean_test_rmse(p):
    d = json.loads(Path(p).read_text())
    # aggregate across hours and channels (normalised)
    vals = []
    for h, hd in d.get("per_hour", {}).items():
        m = hd.get("model", {})
        for k, v in m.items():
            if k.startswith("rmse_"):
                vals.append(float(v))
    return sum(vals) / max(1, len(vals))

p_ac = Path("metrics/eval_6h_2020_paper_leaderboard/dcae_skip_AC.json")
p_skip = Path("metrics/eval_6h_2020_paper_leaderboard/dcae_skip_0p5_6yr_pad.json")
p_noskip = Path("metrics/eval_6h_2020_paper_leaderboard/dcae_noskip.json")
out = []
out.append(f"% Auto-generated after eval. Do not hand-edit.")
out.append("\\begin{table}[t]")
out.append("\\centering")
out.append("\\caption{Skip-connection ablation on 0.5$^\\circ$ ERA5 (360$\\times$720 grid, 24 prognostic fields, 2020 test, 4 days/month $\\times$ 4 starts/day). Both models share the same DC-AE backbone, AdaLN-Zero time conditioning, and bilinear-residual scaffold. The only difference is removal of the U-Net skip connections.}")
out.append("\\label{tab:noskip-ablation}")
out.append("\\small")
out.append("\\setlength{\\tabcolsep}{4pt}")
out.append("\\begin{tabular}{lrrr}")
out.append("\\toprule")
out.append("\\textbf{Variant} & \\textbf{Params} & \\textbf{Test RMSE (norm)} & \\textbf{Test RMSE +A+C} \\\\")
out.append("\\midrule")
r_skip = mean_test_rmse(p_skip) if p_skip.exists() else None
r_noskip = mean_test_rmse(p_noskip) if p_noskip.exists() else None
r_ac = mean_test_rmse(p_ac) if p_ac.exists() else None
def fmt(v): return f"{v:.4f}" if v is not None else "—"
out.append(f"DC-AE Skip (full)         & 14.4 & {fmt(r_skip)} & {fmt(r_ac)} \\\\")
out.append(f"\\quad $-$ skip connections & 13.1 & {fmt(r_noskip)} & --- \\\\")
out.append("\\midrule")
if r_skip and r_noskip:
    out.append(f"\\textit{{Degradation factor}} & & $\\mathbf{{{r_noskip/r_skip:.2f}\\times}}$ & --- \\\\")
out.append("\\bottomrule")
out.append("\\end{tabular}")
out.append("\\end{table}")
Path("paper/11_noskip_ablation.tex").write_text("\n".join(out) + "\n")
print(f"wrote paper/11_noskip_ablation.tex  (skip={r_skip}, noskip={r_noskip}, ac={r_ac})")
PY

# -----------------------------------------------------------------------------
# Stage 4: Multi-year OOD on 2021 (--with-2021)
# -----------------------------------------------------------------------------
if [[ "$WITH_2021" == "1" ]]; then
  echo ">>> Stage 4: Multi-year OOD on 2021"
  if [[ -f "$AC_CKPT" ]]; then
    run eval_AC_2021 python tools/eval/batch_eval_memmap.py \
      --memmap-dir /tmp/wb2_0p5_cache_2021 \
      --test-year 2021 \
      --eval-days-per-month 4 \
      --samples-per-date 4 \
      --models "dcae_skip_AC:$AC_CKPT" \
      --out-rmse-dir metrics/eval_0p5_2021_fast \
      --out-acc-dir  metrics/acc_0p5_2021_fast
  fi
fi

# -----------------------------------------------------------------------------
# Stage 5: Visual case studies (--with-cases)
# -----------------------------------------------------------------------------
if [[ "$WITH_CASES" == "1" ]] && [[ -f "$AC_CKPT" ]]; then
  echo ">>> Stage 5: Visual case studies"
  run cases python tools/eval/visual_case_studies.py \
    --ckpts "ours=$AC_CKPT" "bilinear=__bilinear__" \
    --out-dir paper/case_studies
fi

echo "=== DONE.  Logs in $LOGDIR ==="
