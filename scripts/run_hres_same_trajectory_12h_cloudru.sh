#!/usr/bin/env bash
# Evaluate the true +6 h HRES state between same-initialisation +0/+12 h anchors.
set -euo pipefail

SOURCE=${SOURCE:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_spectral_v6}
RUNTIME=${RUNTIME:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}
LOG_ROOT=${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}
PY=${PY:-/home/jovyan/.mlspace/envs/ai_scientist/bin/python}
HRES=${HRES:-/home/jovyan/shares/SR006.nfs2/dsuhoi/memmap_hres_2021_canonical_v3}
STATIC=${STATIC:-/home/jovyan/dsuhoi/weather_time_interpolation/data/static_features_0p5.pt}
STATS=${STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/json_stats_0p5.nc}
SURFACE_STATS=${SURFACE_STATS:-/home/jovyan/dsuhoi/weather_time_interpolation/data/surface_stats_0p5.json}
OUT_ROOT=${OUT_ROOT:-$RUNTIME/metrics/hres_same_trajectory_12h_v1}
MAX_INITS=${MAX_INITS:-16}

cd "$SOURCE"
export PYTHONPATH="$SOURCE:/home/jovyan/shares/SR006.nfs2/dsuhoi/pylibs_mamba:/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_shim"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
exec 9>"$OUT_ROOT/launch.lock"
if ! flock -n 9; then
  echo "same-trajectory HRES benchmark already active"
  exit 0
fi

EVALUATOR=tools/eval/batch_eval_forecast_anchor.py
EVALUATOR_SHA=$(sha256sum "$EVALUATOR" | awk '{print $1}')
LOG="$LOG_ROOT/hres_same_trajectory_12h_v1.log"
exec > >(tee -a "$LOG") 2>&1

run_eval() {
  local gpu="$1" name="$2" arch="$3" checkpoint="$4"
  local output="$OUT_ROOT/$name.json"
  local lock="$LOG_ROOT/.upr_lite_gpu${gpu}.lock"
  (
    exec 8>"$lock"
    flock 8
    [[ "$(sha256sum "$EVALUATOR" | awk '{print $1}')" == "$EVALUATOR_SHA" ]]
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$EVALUATOR" \
      --forecast-dir "$HRES" --target-source same_forecast \
      --checkpoint "$checkpoint" --arch "$arch" --model-name "$name" \
      --out-json "$output" --stats-path "$STATS" \
      --surface-stats-path "$SURFACE_STATS" --static-path "$STATIC" \
      --device cuda --delta-t-hours 12 --taus 6 --max-inits "$MAX_INITS"
  )
}

"$PY" -u "$EVALUATOR" \
  --forecast-dir "$HRES" --target-source same_forecast \
  --model-blob linear --model-name linear \
  --out-json "$OUT_ROOT/linear.json" --stats-path "$STATS" \
  --surface-stats-path "$SURFACE_STATS" \
  --device cpu --delta-t-hours 12 --taus 6 --max-inits "$MAX_INITS"

DCAE="$LOG_ROOT/exp_weatherdcae_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
FLOW="$LOG_ROOT/exp_flow_pp3_spectral_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
DETAIL="$LOG_ROOT/exp_flow_pp3_detail_14m_12h_2017_19_refinev1_s202707_bs4/last.ckpt"
for checkpoint in "$DCAE" "$FLOW" "$DETAIL"; do
  [[ -s "$checkpoint" ]] || {
    echo "missing checkpoint: $checkpoint" >&2
    exit 1
  }
done

run_eval 0 weatherdcae_14m dcae_14m "$DCAE" &
pid_dcae=$!
run_eval 1 flow_spectral flow_pp3 "$FLOW" &
pid_flow=$!
wait "$pid_dcae" "$pid_flow"
run_eval 0 weatherbridge flow_pp3_detail "$DETAIL"

"$PY" - "$OUT_ROOT" "$EVALUATOR_SHA" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
evaluator_sha = sys.argv[2]
names = ("linear", "weatherdcae_14m", "flow_spectral", "weatherbridge")
payloads = {name: json.loads((root / f"{name}.json").read_text()) for name in names}
indices = {item["paired_artifact"]["window_index_sha256"] for item in payloads.values()}
if len(indices) != 1:
    raise SystemExit("models do not share one paired HRES index")
for name, payload in payloads.items():
    protocol = payload["protocol"]
    if (
        protocol["target_source"] != "same_initialization_forecast_lead"
        or protocol["delta_t_hours"] != 12
        or protocol["taus"] != [6]
    ):
        raise SystemExit(f"invalid protocol for {name}")
    if payload["provenance"]["evaluator"]["sha256"] != evaluator_sha:
        raise SystemExit(f"stale evaluator provenance for {name}")
marker = {
    "schema_version": 1,
    "status": "complete",
    "target_source": "same_initialization_forecast_lead",
    "horizon_hours": 12,
    "tau_hours": [6],
    "model_names": list(names),
    "window_index_sha256": indices.pop(),
    "artifacts": {
        name: hashlib.sha256((root / f"{name}.json").read_bytes()).hexdigest()
        for name in names
    },
}
temporary = root / ".complete.tmp"
temporary.write_text(json.dumps(marker, sort_keys=True) + "\n")
os.replace(temporary, root / "complete.json")
PY
echo "[$(date -Is)] same-trajectory HRES benchmark complete"
