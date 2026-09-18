#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(sed -n 's/^version: "\(.*\)"/\1/p' "$ROOT/CITATION.cff" | head -1)"
OUT="${OUT_DIR:-$ROOT/zenodo/dist}"
PREFIX="weatherbridge-source-v${VERSION}"
SOURCE_ARCHIVE="$OUT/${PREFIX}.tar.gz"
MODEL_RELEASE="$ROOT/weatherbridge-release"
MODEL_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$MODEL_RELEASE/pyproject.toml" | head -1)"
MODEL_ARCHIVE="$MODEL_RELEASE/dist/weatherbridge-models-v${MODEL_VERSION}.tar.gz"

if [[ -z "$VERSION" ]]; then
  printf 'Cannot read version from CITATION.cff\n' >&2
  exit 1
fi

mkdir -p "$OUT" "$ROOT/tmp"
BUILD_DIR="$(mktemp -d "$ROOT/tmp/weatherbridge-zenodo.XXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT
FILE_LIST="$BUILD_DIR/source-files.list"
STAGE="$BUILD_DIR/$PREFIX"

COMMIT="$(git -C "$ROOT" rev-parse HEAD)"
if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=normal)" ]]; then
  DIRTY=true
else
  DIRTY=false
fi
if [[ "$DIRTY" == true && "${ALLOW_DIRTY:-0}" != 1 ]]; then
  printf 'Refusing a final archive from a dirty working tree. Commit the release or set ALLOW_DIRTY=1 for a draft.\n' >&2
  exit 1
fi

git -C "$ROOT" ls-files --cached --others --exclude-standard | while IFS= read -r path; do
  case "$path" in
    .dockerignore|.gitignore|CITATION.cff|LICENSE|Makefile|PROJECT_MAP.md|THIRD_PARTY_NOTICES.md|pyproject.toml|uv.lock|requirements-hydra.txt|requirements-publication.txt|dataset.py|eval.py|evaluate_baselines.py|train.py|trainer.py|trainer_weather_hermite.py)
      printf '%s\n' "$path"
      ;;
    LICENSES/*|conf/*|data/*|demo/*.py|demo/*.json|examples/*|legacy/*|repro/*|scripts/*|tests/*|tools/*|weather_time_interp/*|zenodo/*|weatherbridge-release/*)
      case "$path" in
        zenodo/dist/*|weatherbridge-release/dist/*|weatherbridge-release/weights/*.pt|weatherbridge-release/data/sample_era5_2020070100.npz|*.pyc|*/__pycache__/*|*/.pytest_cache/*|*/.ruff_cache/*)
          ;;
        *)
          printf '%s\n' "$path"
          ;;
      esac
      ;;
    docs/METRICS_INDEX.md|docs/REGIONAL_EVAL_SPEC.md|docs/calculations_audit.md|docs/results/baselines_6h_12h.md|docs/results/per_channel_per_hour_tables.md)
      printf '%s\n' "$path"
      ;;
    metrics/*)
      printf '%s\n' "$path"
      ;;
    paper/*)
      case "$path" in
        paper/main.pdf|paper/manuscript_npj.pdf|paper/supplementary.pdf|paper/*.zip|paper/*.zip.sha256)
          ;;
        *)
          printf '%s\n' "$path"
          ;;
      esac
      ;;
  esac
done | LC_ALL=C sort -u > "$FILE_LIST"

REPRO_FIXTURES=(
  "metrics/nwp_blend_spectra_6h_2022_v2_endpoint_guard/flow_adapted_vs_linear_tau2.json"
  "metrics/nwp_blend_spectra_6h_2022_v2_endpoint_guard/flow_adapted_vs_linear_tau3.json"
  "demo/precomputed/typhoon_haishen/East_Asia__mslp.npz"
  "metrics/case_studies_weatherdcae_14m/typhoon_haishen/mslp.npz"
  "metrics/case_studies_pixelattn_vfi/typhoon_haishen/mslp.npz"
  "metrics/case_studies_weatherbridge/typhoon_haishen/mslp.npz"
  "metrics/case_studies_weatherdcae_14m/hurricane_laura_2020/wind10.npz"
  "metrics/case_studies_pixelattn_vfi/hurricane_laura_2020/wind10.npz"
  "metrics/case_studies_weatherbridge/hurricane_laura_2020/wind10.npz"
)
for path in "${REPRO_FIXTURES[@]}"; do
  if [[ ! -f "$ROOT/$path" ]]; then
    printf 'Missing required reproducibility fixture: %s\n' "$path" >&2
    exit 1
  fi
  printf '%s\n' "$path" >> "$FILE_LIST"
done
LC_ALL=C sort -u -o "$FILE_LIST" "$FILE_LIST"

if [[ ! -s "$FILE_LIST" ]]; then
  printf 'Source allowlist is empty\n' >&2
  exit 1
fi

mkdir -p "$STAGE"
tar -C "$ROOT" -cf - --files-from "$FILE_LIST" | tar -C "$STAGE" -xf -
cp "$ROOT/zenodo/SOURCE_README.md" "$STAGE/README.md"

{
  printf 'WeatherBridge source release v%s\n' "$VERSION"
  printf 'git_commit=%s\n' "$COMMIT"
  printf 'git_dirty=%s\n' "$DIRTY"
  printf 'built_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'builder=zenodo/build_release.sh\n'
} > "$STAGE/RELEASE_PROVENANCE.txt"

(
  cd "$STAGE"
  find . -type f ! -name SOURCE_MANIFEST.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SOURCE_MANIFEST.sha256
)

tar --sort=name --mtime='2026-08-27 00:00:00Z' \
  --owner=0 --group=0 --numeric-owner \
  -C "$BUILD_DIR" -czf "$SOURCE_ARCHIVE" "$PREFIX"

ARCHIVES=("$(basename "$SOURCE_ARCHIVE")")
if [[ "${SOURCE_ONLY:-0}" != 1 ]]; then
if [[ "${SKIP_MODEL_BUILD:-0}" != 1 ]]; then
  "$MODEL_RELEASE/scripts/make_archive.sh"
fi
if [[ ! -s "$MODEL_ARCHIVE" ]]; then
  printf 'Missing model archive: %s\n' "$MODEL_ARCHIVE" >&2
  exit 1
fi
ln -f "$MODEL_ARCHIVE" "$OUT/$(basename "$MODEL_ARCHIVE")"
ARCHIVES+=("$(basename "$MODEL_ARCHIVE")")
fi

(
  cd "$OUT"
  sha256sum "${ARCHIVES[@]}" > SHA256SUMS
)

printf 'Built Zenodo artifacts:\n'
du -h "$SOURCE_ARCHIVE"
cat "$OUT/SHA256SUMS"
if [[ "$DIRTY" == true ]]; then
  printf 'WARNING: source archive was built from a dirty working tree; rebuild from the final tag.\n' >&2
fi
