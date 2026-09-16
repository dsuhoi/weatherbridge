#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/weather_time_interpolation_runtime}"
TARGET_ROOT="${TARGET_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/wti_spherical_overlay_20260727}"
LOG_ROOT="${LOG_ROOT:-/home/jovyan/shares/SR006.nfs2/dsuhoi/capmatched_logs}"
WAIT_SEC="${WAIT_SEC:-30}"
QUIET_SEC="${QUIET_SEC:-60}"
REQUIRE_TERMINALS="${REQUIRE_TERMINALS:-1}"
SYNC_LOG="${SYNC_LOG:-$LOG_ROOT/local_corr_inference_source_sync.log}"
MANIFEST="${MANIFEST:-$LOG_ROOT/local_corr_inference_source_sync.json}"

files=(
  tools/eval/capmatched_loader.py
  weather_time_interp/model/weatherbridge_upr_lite_model.py
  weather_time_interp/model/weatherbridge_upr_scaled_model.py
)
hashes=(
  31e91d4db0bd23c03c06ec3c0656c487e9ca230f99b8090dbbe799d71e1c34a4
  ea26a8fea9fc0bfba4a6812aef49cceecc6daff1d10f186192d528ce9a77ba38
  f3d6927e297068a5dfc13c4473f532eeb95fae33175595fb5e1a06d1246b38f6
)
blocking_experiments=(
  exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2
  exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1
  exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707
  exp_upr_endpoint_implicit_global_14m_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707
  exp_upr_implicit_global_14m_nohf_24ch_6h_2014_19_sparse135_lr1e4_eb16_s202707
  exp_flow_pp3_hf_135_14m_6h_s202707_protocol_v2
)

mkdir -p "$LOG_ROOT"
exec 9>"$LOG_ROOT/.local_corr_inference_source_sync.lock"
if ! flock -n 9; then
  printf '[%s] source-sync watcher already active\n' "$(date -Is)" \
    | tee -a "$SYNC_LOG"
  exit 0
fi

log() {
  printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$SYNC_LOG"
}

validate_files() {
  local root="$1"
  local index path actual
  for index in "${!files[@]}"; do
    path="$root/${files[$index]}"
    [[ -f "$path" ]] || return 1
    actual="$(sha256sum "$path" | awk '{print $1}')"
    [[ "$actual" == "${hashes[$index]}" ]] || return 1
  done
}

active_target_trainers() {
  local proc cwd cmdline
  for proc in /proc/[0-9]*; do
    [[ -r "$proc/cmdline" ]] || continue
    cwd="$(readlink "$proc/cwd" 2>/dev/null || true)"
    [[ "$cwd" == "$TARGET_ROOT" ]] || continue
    cmdline="$(tr '\0' ' ' <"$proc/cmdline" 2>/dev/null || true)"
    case "$cmdline" in
      *train_capacity_matched_6h.py*)
        printf '%s %s\n' "${proc##*/}" "$cmdline"
        ;;
    esac
  done
}

write_manifest() {
  local tmp="$MANIFEST.tmp.$$"
  {
    printf '{\n'
    printf '  "schema_version": 1,\n'
    printf '  "synced_at": "%s",\n' "$(date -Is)"
    printf '  "source_root": "%s",\n' "$SOURCE_ROOT"
    printf '  "target_root": "%s",\n' "$TARGET_ROOT"
    printf '  "files": {\n'
    printf '    "%s": "%s",\n' "${files[0]}" "${hashes[0]}"
    printf '    "%s": "%s",\n' "${files[1]}" "${hashes[1]}"
    printf '    "%s": "%s"\n' "${files[2]}" "${hashes[2]}"
    printf '  }\n'
    printf '}\n'
  } >"$tmp"
  mv -f "$tmp" "$MANIFEST"
}

if ! validate_files "$SOURCE_ROOT"; then
  log "refuse sync: source hashes do not match frozen LocalCorr inference files"
  exit 2
fi

if validate_files "$TARGET_ROOT"; then
  write_manifest
  log "target already contains frozen LocalCorr inference files"
  exit 0
fi

if [[ "$REQUIRE_TERMINALS" -eq 1 ]]; then
  for experiment in "${blocking_experiments[@]}"; do
    marker="$LOG_ROOT/${experiment}.terminal"
    while [[ ! -e "$marker" ]]; do
      log "wait terminal experiment=$experiment"
      sleep "$WAIT_SEC"
    done
  done
fi

while trainers="$(active_target_trainers)" && [[ -n "$trainers" ]]; do
  log "wait active target training process"
  sleep "$WAIT_SEC"
done
if [[ "$QUIET_SEC" != 0 ]]; then
  sleep "$QUIET_SEC"
fi
if trainers="$(active_target_trainers)" && [[ -n "$trainers" ]]; then
  log "target training restarted during quiet interval"
  exit 3
fi

for index in "${!files[@]}"; do
  relative="${files[$index]}"
  source="$SOURCE_ROOT/$relative"
  target="$TARGET_ROOT/$relative"
  temporary="$target.sync.$$"
  mkdir -p "$(dirname "$target")"
  cp -p "$source" "$temporary"
  actual="$(sha256sum "$temporary" | awk '{print $1}')"
  if [[ "$actual" != "${hashes[$index]}" ]]; then
    rm -f "$temporary"
    log "temporary copy hash mismatch file=$relative"
    exit 4
  fi
  mv -f "$temporary" "$target"
done

if ! validate_files "$TARGET_ROOT"; then
  log "post-sync target validation failed"
  exit 5
fi
write_manifest
log "synchronized frozen LocalCorr inference files"
