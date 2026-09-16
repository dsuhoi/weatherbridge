#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main_with_appendix.tex

main_pages="$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')"
full_pages="$(pdfinfo main_with_appendix.pdf | awk '/^Pages:/ {print $2}')"

if (( main_pages > 9 )); then
  printf 'AAAI-27 submission exceeds 9 total pages: %s\n' "$main_pages" >&2
  exit 1
fi
if (( full_pages <= main_pages )); then
  printf 'Combined build does not contain an appendix.\n' >&2
  exit 1
fi

cp main.pdf main_submission.pdf

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
pdfseparate \
  -f "$((main_pages + 1))" \
  -l "$full_pages" \
  main_with_appendix.pdf \
  "$tmp_dir/page-%d.pdf"

appendix_pages=()
for ((page = main_pages + 1; page <= full_pages; page++)); do
  appendix_pages+=("$tmp_dir/page-$page.pdf")
done
pdfunite "${appendix_pages[@]}" supplement.pdf

printf 'main_submission.pdf: %s pages\n' "$main_pages"
printf 'supplement.pdf: %s pages\n' "$((full_pages - main_pages))"
printf 'main_with_appendix.pdf: %s pages\n' "$full_pages"
