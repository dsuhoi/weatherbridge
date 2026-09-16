#!/usr/bin/env bash
# Перенос ERA5 zarr с S3 (read-only s3fs FUSE) на NFS workspace.
# Source: /mnt/s3/bucket8tb/weather_data/time_interpolation/
# Dest:   /workspace-SR006.nfs2/weather_data/time_interpolation/
#
# 2020 уже на NFS, копируем 2012-2019 (PL+surface, ~57 GiB/year × 8 = ~456 GiB).
set -eo pipefail

SRC=/mnt/s3/bucket8tb/weather_data/time_interpolation
DST=/workspace-SR006.nfs2/weather_data/time_interpolation
mkdir -p "$DST"

YEARS=(2012 2013 2014 2015 2016 2017 2018 2019)
LOG=/home/jovyan/dsuhoi/weather_time_interpolation/logs/runner/transfer_s3_nfs.log

echo "=== TRANSFER S3 → NFS started ===" | tee -a "$LOG"
date | tee -a "$LOG"
echo "Source:      $SRC" | tee -a "$LOG"
echo "Destination: $DST" | tee -a "$LOG"
echo "Years:       ${YEARS[*]}" | tee -a "$LOG"

for YEAR in "${YEARS[@]}"; do
  for PREFIX in zarr surface; do
    SRC_DIR="$SRC/${PREFIX}_${YEAR}.zarr"
    DST_DIR="$DST/${PREFIX}_${YEAR}.zarr"
    if [[ ! -d "$SRC_DIR" ]]; then
      echo "[SKIP] $SRC_DIR not found" | tee -a "$LOG"
      continue
    fi
    if [[ -d "$DST_DIR" ]]; then
      EXISTING=$(du -sb "$DST_DIR" 2>/dev/null | awk '{print $1}')
      EXPECTED=$(du -sb "$SRC_DIR" 2>/dev/null | awk '{print $1}')
      if [[ "$EXISTING" == "$EXPECTED" ]] && [[ -n "$EXISTING" ]]; then
        echo "[SKIP] $DST_DIR already complete ($EXISTING B)" | tee -a "$LOG"
        continue
      fi
      echo "[RETRY] $DST_DIR exists but incomplete — removing and re-copy" | tee -a "$LOG"
      rm -rf "$DST_DIR"
    fi
    echo "=== ${PREFIX}_${YEAR}.zarr → $DST_DIR ===" | tee -a "$LOG"
    date | tee -a "$LOG"
    START=$SECONDS
    cp -a "$SRC_DIR" "$DST_DIR" 2>&1 | tee -a "$LOG"
    ELAPSED=$(( SECONDS - START ))
    SIZE_MB=$(du -sm "$DST_DIR" | awk '{print $1}')
    SPEED=$(( SIZE_MB / (ELAPSED + 1) ))
    echo "[done] ${PREFIX}_${YEAR}: ${SIZE_MB} MB in ${ELAPSED}s (~${SPEED} MB/s)" | tee -a "$LOG"
  done
done

echo "=== TRANSFER S3 → NFS DONE ===" | tee -a "$LOG"
date | tee -a "$LOG"
df -h "$DST" | tail -2 | tee -a "$LOG"
