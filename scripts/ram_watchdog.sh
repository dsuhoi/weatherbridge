#!/usr/bin/env bash
# Soft RAM watchdog: убивает наименее важный training, если total RAM > 1.7 TB.
# Лог в logs/runner/ram_watchdog.log.
LOG=/home/jovyan/dsuhoi/weather_time_interpolation/logs/runner/ram_watchdog.log
THRESHOLD_GIB=1700  # 1.7 TB warning; 1.8 TB action

while true; do
  USED_KB=$(grep "^MemTotal\|^MemAvailable" /proc/meminfo | awk 'NR==1{t=$2}NR==2{a=$2}END{print t-a}')
  USED_GIB=$((USED_KB / 1024 / 1024))
  ts=$(date "+%H:%M:%S")
  if [ "$USED_GIB" -gt 1800 ]; then
    echo "[$ts] CRITICAL: RAM=${USED_GIB}GiB > 1800 GiB — killing newest train job" >> "$LOG"
    PID=$(ps -eo pid,etime,cmd | grep -E 'train_weather_hermite.*xattn|train_weather_hermite.*swin' | grep -v grep | sort -k2 | head -1 | awk '{print $1}')
    if [ -n "$PID" ]; then
      kill -KILL "$PID" 2>/dev/null
      echo "[$ts]   killed PID=$PID" >> "$LOG"
    fi
  elif [ "$USED_GIB" -gt "$THRESHOLD_GIB" ]; then
    echo "[$ts] WARN: RAM=${USED_GIB}GiB > ${THRESHOLD_GIB}GiB" >> "$LOG"
  else
    echo "[$ts] OK: RAM=${USED_GIB}GiB" >> "$LOG"
  fi
  sleep 60
done
