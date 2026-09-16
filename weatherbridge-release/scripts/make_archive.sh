#!/usr/bin/env bash
# Build the uploadable archive, with a manifest that covers what is in it.
#
# Deterministic: entries are sorted, timestamps and ownership are pinned, so
# the same tree always produces the same bytes and the archive can itself be
# checksummed in a citation.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)"
OUT="dist/weatherbridge-models-v${VERSION}"
mkdir -p dist

echo "==> regenerating MANIFEST.sha256"
: > MANIFEST.sha256
{
  printf '# sha256 of every file in the WeatherBridge model release v%s\n' "$VERSION"
  find weatherbridge weights data docs examples scripts tests LICENSES \
       README.md LICENSE THIRD_PARTY_NOTICES.md CITATION.cff \
       pyproject.toml zenodo_metadata.json \
       -type f ! -name '*.pyc' ! -path '*__pycache__*' -print0 \
    | sort -z | xargs -0 sha256sum
} > MANIFEST.sha256
printf '    %s files\n' "$(grep -cv '^#' MANIFEST.sha256)"

echo "==> packing $OUT.tar.gz"
tar --sort=name --mtime='2026-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='.venv' --exclude='dist' \
    -czf "$OUT.tar.gz" \
    weatherbridge weights data docs examples scripts tests LICENSES \
    README.md LICENSE THIRD_PARTY_NOTICES.md CITATION.cff pyproject.toml \
    zenodo_metadata.json MANIFEST.sha256

sha256sum "$OUT.tar.gz" > "$OUT.tar.gz.sha256"
printf '    %s (%s)\n' "$OUT.tar.gz" "$(du -h "$OUT.tar.gz" | cut -f1)"
cat "$OUT.tar.gz.sha256"
