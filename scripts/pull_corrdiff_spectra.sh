#!/usr/bin/env bash
# Pull CorrDiff SH-spectra npz from fibo /mnt/storage back into local repo,
# then regenerate Fig.5.
set -euo pipefail

NAME="${NAME:-corrdiff_fm_weatherdcae_12h_6yr}"
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DIR="$REPO_ROOT/metrics/sh_spectra_12h_ep10_24ch"
REMOTE_DIR=/mnt/storage/d.sukhorukov/wti_storage/metrics/sh_spectra_12h_ep10_24ch

mkdir -p "${LOCAL_DIR}"

for tau in 2 3; do
  src="${REMOTE_DIR}/${NAME}_tau${tau}.npz"
  scp -q "fibonacci:${src}" "${LOCAL_DIR}/" || echo "  [miss] ${src}"
done

ls -la "${LOCAL_DIR}/${NAME}"_tau{2,3}.npz 2>/dev/null || echo "  no files yet"

# Regenerate Fig.5
cd "$REPO_ROOT"
python3 scripts/make_fig3_spectra.py
cd paper && pdflatex -interaction=nonstopmode main.tex >/dev/null
echo "  paper rebuilt: $(pwd)/main.pdf"
