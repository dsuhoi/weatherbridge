#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${FINAL_PUBLICATION:-0}" == 1 ]]; then
  required=(
    main_metrics_v2.tex
    postselection_2022.tex
    supplementary_data_1_statistics.csv
    supplementary_data_1_statistics.manifest.json
    supplementary_data_2_postselection_2022.json
    supplementary_data_2_postselection_2022.manifest.json
    tab_sh_per_channel_6h_v6.tex
    tab_sh_per_channel_12h_v6.tex
    images/.weatherbridge_full_year_figures_complete
    images/.weatherbridge_rmse_maps_complete
    images/.hres_field_figures_complete
    images/fig_hres_fields_6h.pdf
    images/fig_hres_fields_12h.pdf
    ../metrics/journal_spectra_v6/state/.complete
    code_archive_statement.tex
    funding_statement.tex
    ../LICENSE
    ../CITATION.cff
    ../requirements-publication.txt
  )
  missing=()
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      missing+=("$path")
    fi
  done
  citation_status=0
  if [[ -s ../CITATION.cff ]]; then
    if ! python - <<'PY'
from pathlib import Path

import yaml

path = Path("../CITATION.cff")
payload = yaml.safe_load(path.read_text()) or {}
errors = []
for key in ("version", "date-released"):
    if not payload.get(key):
        errors.append(f"CITATION.cff is incomplete: missing {key}")
authors = payload.get("authors") or []
if not authors:
    errors.append("CITATION.cff is incomplete: missing authors")
elif not any(author.get("orcid") for author in authors):
    errors.append("CITATION.cff is incomplete: no author ORCID identifiers")
for error in errors:
    print(error)
raise SystemExit(bool(errors))
PY
    then
      citation_status=1
    fi
  fi
  if (( ${#missing[@]} || citation_status )); then
    for path in "${missing[@]}"; do
      printf 'Final publication artifact is missing: %s\n' "$path" >&2
    done
    exit 1
  fi
fi

# The supplementary is built first: main.tex resolves its Supplementary figure
# numbers through xr from supplementary.aux, so that file has to exist and be
# current before main is typeset.
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

for stem in main supplementary; do
  if grep -Eq 'LaTeX Warning: (Reference|Citation).*undefined|There were undefined references' "$stem.log"; then
    printf 'Unresolved references or citations remain in %s.log.\n' "$stem" >&2
    exit 1
  fi

  # sn-jnl's output routine reports the rotated pdflscape page box as
  # overfull even when the page content is within its landscape media box.
  # Content overflows use the normal "in paragraph" form and remain fatal.
  if grep 'Overfull \\[hv]box' "$stem.log" | grep -vq 'while \\output is active'; then
    printf 'Overfull boxes remain in %s.log.\n' "$stem" >&2
    exit 1
  fi

  if ! pdffonts "$stem.pdf" | awk 'NR > 2 && $(NF-4) != "yes" {bad=1} END {exit bad}'; then
    printf 'At least one font is not embedded in %s.pdf.\n' "$stem" >&2
    exit 1
  fi
done

python - <<'PY'
import re
from pathlib import Path

source = Path("main.tex").read_text()
start = source.index(r"\abstract{") + len(r"\abstract{")
depth = 1
end = start
while depth:
    depth += (source[end] == "{") - (source[end] == "}")
    end += 1
abstract = source[start:end - 1]
abstract = re.sub(r"\\[A-Za-z]+(?:\[[^]]*\])?", " ", abstract)
abstract = re.sub(r"[{}$~_^=\\]", " ", abstract)
words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract)
if len(words) > 150:
    raise SystemExit(f"Abstract exceeds 150 words: {len(words)}")
print(f"abstract: approximately {len(words)} words")
PY

if [[ "${FINAL_PUBLICATION:-0}" == 1 ]]; then
  text_file="$(mktemp)"
  trap 'rm -f "$text_file"' EXIT
  pdftotext main.pdf "$text_file"
  forbidden=(
    pending-v2
    "pending validation"
    "working draft"
    "working build"
    "withheld from this interim build"
    "to be inserted before submission"
    "will be deposited"
  )
  for phrase in "${forbidden[@]}"; do
    if grep -Fqi "$phrase" "$text_file"; then
      printf 'Final publication text still contains: %s\n' "$phrase" >&2
      exit 1
    fi
  done
  python ../tools/repro/preflight_npj_submission.py --paper-dir .
  # Figure text has to stay legible at the size it is actually placed at, not
  # the size its plotting script asked for.
  python ../tools/repro/audit_figure_legibility.py --paper-dir . --strict
fi

cp main.pdf manuscript_npj.pdf

pages="$(pdfinfo manuscript_npj.pdf | awk '/^Pages:/ {print $2}')"
words="$(pdftotext manuscript_npj.pdf - | wc -w)"
supp_pages="$(pdfinfo supplementary.pdf | awk '/^Pages:/ {print $2}')"
supp_words="$(pdftotext supplementary.pdf - | wc -w)"
printf 'manuscript_npj.pdf: %s pages, approximately %s extracted words\n' \
  "$pages" "$words"
printf 'supplementary.pdf: %s pages, approximately %s extracted words\n' \
  "$supp_pages" "$supp_words"
if [[ "${FINAL_PUBLICATION:-0}" != 1 ]]; then
  printf 'Working build only; run with FINAL_PUBLICATION=1 for submission preflight.\n'
fi
