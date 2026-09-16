#!/usr/bin/env bash
# Local watcher: polls fibo + cloud.ru for the two ensemble CRPS JSONs.
# Once both exist (or after timeout), fetches them to local and rebuilds
# paper/tab_crps_comparison.tex.
#
# Runs locally, talks to fibo + cloud.ru via SSH every 5 minutes.
#
# Designed to be launched via:
#   nohup bash /home/dsuhoi/.../scripts/local_watcher_crps_table.sh \
#     > /tmp/local_watcher_crps_table.log 2>&1 &
set -uo pipefail

LOCAL_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FIBO_REPO=/home/d.sukhorukov/weather_time_interpolation
CLOUDRU_KEY="$LOCAL_REPO/keys/weather_projects_cloudru_private_key"
CLOUDRU_HOST="weather-projects.ai0001053-01333@ssh-sr006-jupyter.ai.cloud.ru"
CLOUDRU_REPO=/home/jovyan/dsuhoi/weather_time_interpolation

# Outputs we are waiting on:
FIBO_CRPS="${FIBO_REPO}/metrics/crps_0p5_2020/corrdiff_fm_weatherdcae.json"
FIBO_RMSE="${FIBO_REPO}/metrics/eval_6h_2020_paper_leaderboard/corrdiff_fm_weatherdcae_24ch_3yr_ep10.json"
CLOUDRU_CRPS="${CLOUDRU_REPO}/metrics/crps_12h_2020/corrdiff_fm_weatherdcae_12h_6yr.json"
CLOUDRU_RMSE="${CLOUDRU_REPO}/metrics/eval_12h_2020_ep10/corrdiff_fm_weatherdcae_12h_6yr_ens16.json"

# Local destinations (mirror layout)
LOCAL_FIBO_CRPS="${LOCAL_REPO}/metrics/crps_0p5_2020/corrdiff_fm_weatherdcae.json"
LOCAL_FIBO_RMSE="${LOCAL_REPO}/metrics/eval_6h_2020_paper_leaderboard/corrdiff_fm_weatherdcae_24ch_3yr_ep10.json"
LOCAL_CLOUDRU_CRPS="${LOCAL_REPO}/metrics/crps_12h_2020/corrdiff_fm_weatherdcae_12h_6yr.json"
LOCAL_CLOUDRU_RMSE="${LOCAL_REPO}/metrics/eval_12h_2020_ep10/corrdiff_fm_weatherdcae_12h_6yr_ens16.json"

mkdir -p "$(dirname "${LOCAL_FIBO_CRPS}")" \
         "$(dirname "${LOCAL_FIBO_RMSE}")" \
         "$(dirname "${LOCAL_CLOUDRU_CRPS}")" \
         "$(dirname "${LOCAL_CLOUDRU_RMSE}")"

scp_cloudru() {
  scp -i "${CLOUDRU_KEY}" -o IdentitiesOnly=yes -P 2222 -o StrictHostKeyChecking=no \
    "${CLOUDRU_HOST}:${1}" "${2}"
}

check_remote() {
  # $1 = host (fibonacci or cloudru), $2 = path
  case "$1" in
    fibonacci)
      ssh -o BatchMode=yes fibonacci "[ -f '${2}' ]" 2>/dev/null
      ;;
    cloudru)
      ssh -i "${CLOUDRU_KEY}" -o IdentitiesOnly=yes -p 2222 -o StrictHostKeyChecking=no -o BatchMode=yes \
        "${CLOUDRU_HOST}" "[ -f '${2}' ]" 2>/dev/null
      ;;
  esac
}

echo "[local watcher start] $(date -u +%FT%TZ)"
echo "Waiting for:"
echo "  fibo:   ${FIBO_CRPS}"
echo "  cloudru: ${CLOUDRU_CRPS}"

fibo_done=0
cloudru_done=0
TIMEOUT_HOURS=24
DEADLINE=$(($(date +%s) + TIMEOUT_HOURS * 3600))

while [[ $fibo_done -eq 0 || $cloudru_done -eq 0 ]]; do
  if [[ $fibo_done -eq 0 ]] && check_remote fibonacci "${FIBO_CRPS}"; then
    echo "[poll] fibo CRPS exists at $(date -u +%FT%TZ); fetching..."
    if scp fibonacci:"${FIBO_CRPS}" "${LOCAL_FIBO_CRPS}" 2>/dev/null \
       && scp fibonacci:"${FIBO_RMSE}" "${LOCAL_FIBO_RMSE}" 2>/dev/null; then
      fibo_done=1
      echo "[poll] fibo files fetched OK"
    fi
  fi
  if [[ $cloudru_done -eq 0 ]] && check_remote cloudru "${CLOUDRU_CRPS}"; then
    echo "[poll] cloud.ru CRPS exists at $(date -u +%FT%TZ); fetching..."
    if scp_cloudru "${CLOUDRU_CRPS}" "${LOCAL_CLOUDRU_CRPS}" 2>/dev/null \
       && scp_cloudru "${CLOUDRU_RMSE}" "${LOCAL_CLOUDRU_RMSE}" 2>/dev/null; then
      cloudru_done=1
      echo "[poll] cloud.ru files fetched OK"
    fi
  fi
  if [[ $(date +%s) -ge $DEADLINE ]]; then
    echo "[timeout] hit ${TIMEOUT_HOURS}h deadline at $(date -u +%FT%TZ); building table with whatever's available"
    break
  fi
  if [[ $fibo_done -eq 0 || $cloudru_done -eq 0 ]]; then
    sleep 300  # 5 minutes
  fi
done

echo "[build] running build_crps_comparison.py at $(date -u +%FT%TZ)"
cd "${LOCAL_REPO}"
python3 tools/eval/build_crps_comparison.py \
  --out "${LOCAL_REPO}/paper/tab_crps_comparison.tex"

echo "[local watcher done] $(date -u +%FT%TZ); fibo_done=${fibo_done} cloudru_done=${cloudru_done}"
