#!/usr/bin/env python3
"""Fail-closed checks for the final npj Climate and Atmospheric Science PDF."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from pathlib import Path

FORBIDDEN_FINAL_PHRASES = (
    "pending-v2",
    "pending validation",
    "working draft",
    "working build",
    "interim build",
    "withheld from",
    "to be inserted before submission",
    "will be deposited",
    "will be created before publication",
    "must be completed before submission",
    "authors withheld",
    "author names withheld",
    "corresponding author email",
    "funding and acknowledgement information",
    "contribution statement using author initials",
    "must complete the competing-interest declaration",
    "affiliations to be inserted",
)

REQUIRED_SECTIONS = (
    "Data availability",
    "Code availability",
    "Acknowledgements",
    "Author contributions",
    "Competing interests",
)

PERFORMANCE_FIGURES = (
    "fig1_rmse_per_tau.pdf",
    "fig2_acc_per_tau.pdf",
    "fig_channels_body_6h_phys.pdf",
    "fig_weatherbridge_spectra_tau23.pdf",
    "fig_spectra_ratio_hard.pdf",
    "fig_reference_architectures.pdf",
    "fig_channels_body_12h_phys.pdf",
    "fig_haishen_mslp.pdf",
    "fig_laura_wind10.pdf",
    "fig_channels_body_ood2021.pdf",
    "fig_channels_app_ood2021.pdf",
    "fig_case_ida_wind.pdf",
    "fig_channels_app_12h_phys.pdf",
    "fig_channels_app_6h_phys.pdf",
    "fig_rmse_maps_t2m_5tau.pdf",
    "fig_rmse_maps_u10_5tau.pdf",
    "fig_rmse_maps_mslp_5tau.pdf",
    "fig_rmse_maps_Q1000_5tau.pdf",
    "fig_hres_fields_6h.pdf",
)

BASELINE_ONLY_FIGURES = {
    "fig_reference_architectures.pdf",
}

COMPLETION_MARKERS = (
    ("paper", "images/.weatherbridge_full_year_figures_complete"),
    ("paper", "images/.weatherbridge_rmse_maps_complete"),
    ("paper", "images/.hres_field_figures_complete"),
    ("project", "metrics/journal_spectra_v6/state/.complete"),
)

MAX_MAIN_DISPLAY_ITEMS = 12


def _run_text(*command: str) -> str:
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _pdf_text(path: Path) -> str:
    return _run_text("pdftotext", str(path), "-")


def _normalized_pdf_text(value: str) -> str:
    """Normalize PDF extraction whitespace for robust phrase checks."""
    return re.sub(r"\s+", " ", value).strip().lower()


def _expanded_tex_source(path: Path, seen: set[Path] | None = None) -> str:
    """Expand local LaTeX input files for source-level editorial checks."""
    path = path.resolve()
    seen = set() if seen is None else seen
    if path in seen:
        return ""
    seen.add(path)
    source = path.read_text()
    chunks = [source]
    for argument in _command_arguments(source, "input"):
        child = path.parent / argument
        if child.suffix == "":
            child = child.with_suffix(".tex")
        if child.is_file():
            chunks.append(_expanded_tex_source(child, seen))
    return "\n".join(chunks)


def _display_item_errors(
    source: str,
    *,
    limit: int = MAX_MAIN_DISPLAY_ITEMS,
) -> list[str]:
    figures = len(re.findall(r"\\begin\{figure\*?\}", source))
    tables = len(re.findall(r"\\begin\{table\*?\}", source))
    total = figures + tables
    if total > limit:
        return [
            f"main text has {total} display items ({figures} figures, "
            f"{tables} tables); internal editorial limit is {limit}"
        ]
    return []


def _supplementary_display_reference_errors(
    main_source: str,
    supplementary_source: str,
) -> list[str]:
    """Require every supplementary figure and table to be cited from main."""
    environments = (
        "figure",
        "figure*",
        "sidewaysfigure",
        "table",
        "table*",
        "sidewaystable",
    )
    labels: set[str] = set()
    for environment in environments:
        pattern = re.compile(
            rf"\\begin\{{{re.escape(environment)}\}}"
            rf"(.*?)\\end\{{{re.escape(environment)}\}}",
            re.DOTALL,
        )
        for body in pattern.findall(supplementary_source):
            labels.update(_command_arguments(body, "label"))
    references = set(_command_arguments(main_source, "ref"))
    missing = sorted(
        label
        for label in labels
        if label not in references and f"S-{label}" not in references
    )
    if not missing:
        return []
    return [
        "supplementary display items not cited in main text: "
        + ", ".join(missing)
    ]


def _spectral_figure_text_errors(text: str) -> list[str]:
    errors: list[str] = []
    if "Signed cospectrum" not in text:
        errors.append("spectral figure does not label the signed cospectrum")
    if "Spectral coherence" in text:
        errors.append("spectral figure still labels coherence")
    return errors


def _font_errors(
    pdffonts_output: str,
    *,
    document_label: str = "final PDF",
) -> list[str]:
    errors: list[str] = []
    for line in pdffonts_output.splitlines()[2:]:
        match = re.match(
            r"^(\S+)\s+(.+?)\s+(\S+)\s+(yes|no)\s+(yes|no)\s+(yes|no)"
            r"\s+\d+\s+\d+\s*$",
            line,
        )
        if match is None:
            errors.append(f"unparseable pdffonts row: {line}")
            continue
        name, font_type, _, embedded, _, _ = match.groups()
        if embedded != "yes":
            errors.append(f"unembedded font in {document_label}: {name}")
        if font_type.strip() == "Type 3":
            errors.append(f"Type 3 font in {document_label}: {name}")
    return errors


def _raster_errors(
    pdfimages_output: str,
    *,
    min_ppi: float = 300.0,
) -> list[str]:
    errors: list[str] = []
    allowed_colors = {"rgb", "gray", "index"}
    for line in pdfimages_output.splitlines()[2:]:
        if not line.strip():
            continue
        columns = line.split()
        if len(columns) < 16:
            errors.append(f"unparseable pdfimages row: {line}")
            continue
        page, number, image_type = columns[:3]
        if image_type != "image":
            continue
        color = columns[5].lower()
        try:
            x_ppi = float(columns[12])
            y_ppi = float(columns[13])
        except ValueError:
            errors.append(f"unparseable raster resolution row: {line}")
            continue
        label = f"page {page} image {number}"
        if color not in allowed_colors:
            errors.append(f"unsupported raster color space in {label}: {color}")
        if min(x_ppi, y_ppi) < min_ppi:
            errors.append(
                f"raster resolution below {min_ppi:g} ppi in {label}: "
                f"{x_ppi:g} x {y_ppi:g}"
            )
    return errors


def _graphics_paths(source: str, paper_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for value in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", source):
        path = paper_dir / value
        if path.suffix:
            paths.append(path)
            continue
        for suffix in (".pdf", ".png", ".jpg", ".jpeg"):
            candidate = path.with_suffix(suffix)
            if candidate.exists():
                paths.append(candidate)
                break
        else:
            paths.append(path)
    return paths


def _command_arguments(source: str, command: str) -> list[str]:
    """Return balanced mandatory arguments for a LaTeX command."""
    pattern = re.compile(
        rf"\\{re.escape(command)}(?:\[[^]]*\])?\s*\{{"
    )
    arguments: list[str] = []
    for match in pattern.finditer(source):
        start = match.end()
        depth = 1
        index = start
        while index < len(source) and depth:
            character = source[index]
            if character == "\\" and index + 1 < len(source):
                index += 2
                continue
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
            index += 1
        if depth:
            raise ValueError(f"unterminated \\{command} argument")
        arguments.append(source[start : index - 1])
    return arguments


def _latex_word_count(value: str) -> int:
    without_comments = re.sub(r"(?<!\\)%[^\n]*", " ", value)
    without_commands = re.sub(r"\\(?:[A-Za-z@]+|.)", " ", without_comments)
    plain = re.sub(r"[{}$~_^=]", " ", without_commands)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", plain))


def _caption_errors(source: str, *, word_limit: int = 350) -> list[str]:
    errors: list[str] = []
    for index, caption in enumerate(_command_arguments(source, "caption"), start=1):
        count = _latex_word_count(caption)
        if count > word_limit:
            errors.append(
                f"figure/table caption {index} exceeds {word_limit} words: {count}"
            )
    return errors


def _frontmatter_errors(source: str) -> list[str]:
    errors: list[str] = []
    titles = _command_arguments(source, "title")
    abstracts = _command_arguments(source, "abstract")
    if len(titles) != 1:
        errors.append(f"expected one full title, found {len(titles)}")
    elif (count := _latex_word_count(titles[0])) > 15:
        errors.append(f"title exceeds 15 words: {count}")
    if len(abstracts) != 1:
        errors.append(f"expected one abstract, found {len(abstracts)}")
    elif (count := _latex_word_count(abstracts[0])) > 150:
        errors.append(f"abstract exceeds 150 words: {count}")
    return errors


def _npj_source_style_errors(
    main_source: str,
    supplementary_source: str,
) -> list[str]:
    """Check source conventions that are not reliably visible in PDF text."""
    errors: list[str] = []
    if r"\bmhead{Funding}" in main_source:
        errors.append("funding must be included under Acknowledgements")
    if re.search(
        r"\\(?:sub)*section\{Supplementary Methods(?:[^}]*)\}",
        supplementary_source,
    ):
        errors.append("Supplementary Methods section is not permitted")

    author_pattern = re.compile(
        r"\\author\*?\[[^]]*\]\{\\fnm\{[^}]+\}\s*\\sur\{([^}]+)\}\}"
    )
    main_authors = author_pattern.findall(main_source)
    supplementary_authors = author_pattern.findall(supplementary_source)
    if main_authors != supplementary_authors:
        errors.append(
            "main and supplementary author lists differ: "
            f"{main_authors!r} != {supplementary_authors!r}"
        )

    legacy_panel_labels = re.findall(
        r"\\textbf\{([a-z](?:--[a-z])?)\},",
        main_source + "\n" + supplementary_source,
    )
    if legacy_panel_labels:
        errors.append(
            "panel labels must use lower-case letters with closing parentheses: "
            + ", ".join(legacy_panel_labels)
        )
    return errors


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completion_marker_errors(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    for root_name, relative_path in COMPLETION_MARKERS:
        root = paper_dir if root_name == "paper" else paper_dir.parent
        path = root / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing completion marker: {path}")
            continue
        try:
            marker = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"invalid completion marker {path}: {error}")
            continue
        if marker.get("status") != "complete":
            errors.append(f"incomplete completion marker: {path}")
    return errors


def _statistics_errors(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    project_dir = paper_dir.parent
    csv_path = paper_dir / "supplementary_data_1_statistics.csv"
    manifest_path = paper_dir / "supplementary_data_1_statistics.manifest.json"
    champion_path = project_dir / "metrics/journal_champion_v1/final.json"
    exporter_path = project_dir / "tools/eval/export_journal_statistics.py"
    for path in (csv_path, manifest_path, champion_path, exporter_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing statistics artifact: {path}")
    if errors:
        return errors

    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"invalid statistics manifest: {error}"]
    if manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
        errors.append("statistics manifest is not complete schema 1")
    try:
        champion = json.loads(champion_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        errors.append(f"invalid champion artifact: {error}")
        champion = {}
    if not champion.get("winner") or manifest.get("winner") != champion.get("winner"):
        errors.append("statistics manifest winner does not match the selector")
    expected_hashes = (
        (champion_path, manifest.get("champion_sha256"), "champion"),
        (csv_path, manifest.get("csv_sha256"), "statistics CSV"),
        (exporter_path, manifest.get("exporter_sha256"), "statistics exporter"),
    )
    for path, expected, label in expected_hashes:
        if not isinstance(expected, str) or _sha256(path) != expected:
            errors.append(f"{label} hash mismatch")

    source_hashes = manifest.get("source_sha256", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        errors.append("statistics manifest has no source hashes")
    else:
        for path_string, expected in source_hashes.items():
            path = Path(path_string)
            if not path.is_absolute():
                path = project_dir / path
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"statistics source hash mismatch: {path}")

    try:
        with csv_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = set(reader.fieldnames or ())
    except (csv.Error, OSError) as error:
        errors.append(f"invalid statistics CSV: {error}")
        return errors
    required_columns = {
        "candidate",
        "reference",
        "family",
        "metric",
        "delta_left_minus_right",
        "ci95_low",
        "ci95_median",
        "ci95_high",
        "p_raw_two_sided",
        "p_censored",
        "source_path",
        "source_sha256",
    }
    missing_columns = sorted(required_columns - fieldnames)
    if missing_columns:
        errors.append(
            "statistics CSV is missing columns: " + ", ".join(missing_columns)
        )
    if not rows or manifest.get("row_count") != len(rows):
        errors.append("statistics CSV row count does not match its manifest")
    for index, row in enumerate(rows, start=2):
        if row.get("p_censored") not in {"true", "false"}:
            errors.append(
                f"statistics CSV row {index} has invalid p_censored flag"
            )
        for field in (
            "delta_left_minus_right",
            "ci95_low",
            "ci95_median",
            "ci95_high",
            "p_raw_two_sided",
        ):
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                errors.append(f"statistics CSV row {index} has invalid {field}")
                continue
            if not math.isfinite(value):
                errors.append(f"statistics CSV row {index} has non-finite {field}")
            if field == "p_raw_two_sided" and not 0.0 <= value <= 1.0:
                errors.append(f"statistics CSV row {index} has invalid p-value")
    return errors


def _postselection_data_errors(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    project_dir = paper_dir.parent
    data_path = paper_dir / "supplementary_data_2_postselection_2022.json"
    manifest_path = data_path.with_name(data_path.stem + ".manifest.json")
    exporter_path = project_dir / "tools/repro/export_postselection_supplementary_data.py"
    for path in (data_path, manifest_path, exporter_path):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing post-selection data artifact: {path}")
    if errors:
        return errors
    try:
        data = json.loads(data_path.read_text())
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError) as error:
        return [f"invalid post-selection data artifact: {error}"]
    if data.get("status") != "complete" or data.get("schema_version") != 1:
        errors.append("post-selection data is not complete schema 1")
    if data.get("candidate", {}).get("public_name") != "WeatherBridge":
        errors.append("post-selection data does not identify WeatherBridge")
    if not data.get("aggregate_confirmation_pass"):
        errors.append("post-selection aggregate confirmation did not pass")
    if data.get("universal_dominance_claim_allowed") is not False:
        errors.append("post-selection dominance boundary is missing")
    if manifest.get("status") != "complete" or manifest.get("schema_version") != 1:
        errors.append("post-selection manifest is not complete schema 1")
    if manifest.get("data_sha256") != _sha256(data_path):
        errors.append("post-selection data hash mismatch")
    if manifest.get("exporter_sha256") != _sha256(exporter_path):
        errors.append("post-selection exporter hash mismatch")
    sources = manifest.get("source_sha256", {})
    if not isinstance(sources, dict) or not sources:
        errors.append("post-selection manifest has no source hashes")
    else:
        for path_string, expected in sources.items():
            source = Path(path_string)
            if not source.is_absolute():
                source = project_dir / source
            if not source.is_file() or _sha256(source) != expected:
                errors.append(f"post-selection source hash mismatch: {source}")
    return errors


def collect_errors(paper_dir: Path) -> list[str]:
    errors: list[str] = []
    tex_path = paper_dir / "main.tex"
    pdf_path = paper_dir / "main.pdf"
    log_path = paper_dir / "main.log"
    supplementary_tex_path = paper_dir / "supplementary.tex"
    supplementary_pdf_path = paper_dir / "supplementary.pdf"
    supplementary_log_path = paper_dir / "supplementary.log"
    for path in (
        tex_path,
        pdf_path,
        log_path,
        supplementary_tex_path,
        supplementary_pdf_path,
        supplementary_log_path,
        paper_dir / "code_archive_statement.tex",
        paper_dir / "funding_statement.tex",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing final-build artifact: {path}")
    if errors:
        return errors

    source = tex_path.read_text()
    expanded_source = _expanded_tex_source(tex_path)
    source_lower = _normalized_pdf_text(source)
    pdf_text = _pdf_text(pdf_path)
    log_text = log_path.read_text(errors="replace")
    pdf_lower = _normalized_pdf_text(pdf_text)
    supplementary_source = supplementary_tex_path.read_text()
    expanded_supplementary_source = _expanded_tex_source(
        supplementary_tex_path
    )
    supplementary_source_lower = _normalized_pdf_text(supplementary_source)
    supplementary_pdf_text = _pdf_text(supplementary_pdf_path)
    supplementary_log_text = supplementary_log_path.read_text(errors="replace")
    supplementary_pdf_lower = _normalized_pdf_text(supplementary_pdf_text)

    for phrase in FORBIDDEN_FINAL_PHRASES:
        if phrase.lower() in pdf_lower or phrase.lower() in source_lower:
            errors.append(f"final PDF contains placeholder phrase: {phrase}")
        if (
            phrase.lower() in supplementary_pdf_lower
            or phrase.lower() in supplementary_source_lower
        ):
            errors.append(
                f"supplementary PDF contains placeholder phrase: {phrase}"
            )
    legacy_names = (
        "weatherbridge-flow-spectral",
        "flow-spectral",
        "weatherbridge-detail",
        "detail-bypass ablation",
    )
    for legacy_name in legacy_names:
        if legacy_name in pdf_lower:
            errors.append(f"legacy model name remains in final PDF: {legacy_name}")
        if legacy_name in supplementary_pdf_lower:
            errors.append(
                "legacy model name remains in supplementary PDF: "
                f"{legacy_name}"
            )
    if "weatherbridge" not in pdf_lower:
        errors.append("canonical WeatherBridge name is absent")
    for section in REQUIRED_SECTIONS:
        if section.lower() not in pdf_lower:
            errors.append(f"missing required statement: {section}")

    labels = re.findall(r"\\label\{([^}]+)\}", source)
    duplicate_labels = sorted(
        label for label, count in Counter(labels).items() if count > 1
    )
    if duplicate_labels:
        errors.append(f"duplicate LaTeX labels: {', '.join(duplicate_labels)}")
    uncited_figures = sorted(
        label
        for label in labels
        if label.startswith("fig:") and f"\\ref{{{label}}}" not in source
    )
    if uncited_figures:
        errors.append("uncited figures: " + ", ".join(uncited_figures))
    if re.search(
        r"LaTeX Warning: (?:Reference|Citation).*undefined|"
        r"There were undefined references",
        log_text,
    ):
        errors.append("unresolved LaTeX reference or citation")
    if any(
        "Overfull \\" in line and "while \\output is active" not in line
        for line in log_text.splitlines()
    ):
        errors.append("overfull LaTeX box remains")
    supplementary_labels = re.findall(r"\\label\{([^}]+)\}", supplementary_source)
    duplicate_supplementary_labels = sorted(
        label
        for label, count in Counter(supplementary_labels).items()
        if count > 1
    )
    if duplicate_supplementary_labels:
        errors.append(
            "duplicate supplementary LaTeX labels: "
            + ", ".join(duplicate_supplementary_labels)
        )
    uncited_supplementary_figures = sorted(
        label
        for label in supplementary_labels
        if label.startswith("fig:")
        and f"\\ref{{{label}}}" not in supplementary_source
    )
    if uncited_supplementary_figures:
        errors.append(
            "uncited supplementary figures: "
            + ", ".join(uncited_supplementary_figures)
        )
    if re.search(
        r"LaTeX Warning: (?:Reference|Citation).*undefined|"
        r"There were undefined references",
        supplementary_log_text,
    ):
        errors.append("unresolved supplementary LaTeX reference or citation")
    if any(
        "Overfull \\" in line and "while \\output is active" not in line
        for line in supplementary_log_text.splitlines()
    ):
        errors.append("overfull supplementary LaTeX box remains")
    errors.extend(_frontmatter_errors(source))
    errors.extend(_npj_source_style_errors(source, supplementary_source))
    errors.extend(_caption_errors(source))
    errors.extend(_caption_errors(supplementary_source))
    errors.extend(_display_item_errors(expanded_source))
    errors.extend(
        _supplementary_display_reference_errors(
            expanded_source,
            expanded_supplementary_source,
        )
    )

    main_titles = _command_arguments(source, "title")
    supplementary_titles = _command_arguments(supplementary_source, "title")
    if len(main_titles) == 1 and len(supplementary_titles) == 1:
        expected_title = "Supplementary Information for " + " ".join(
            main_titles[0].split()
        )
        actual_title = " ".join(supplementary_titles[0].split())
        if actual_title != expected_title:
            errors.append("supplementary title does not match the main title")

    for path in _graphics_paths(source, paper_dir) + _graphics_paths(
        supplementary_source, paper_dir
    ):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing figure: {path}")
            continue
        if path.suffix.lower() == ".pdf":
            info = _run_text("pdfinfo", str(path))
            match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
            if match is None or int(match.group(1)) != 1:
                errors.append(f"figure is not a one-page PDF: {path}")

    for filename in PERFORMANCE_FIGURES:
        path = paper_dir / "images" / filename
        if not path.is_file():
            errors.append(f"missing performance figure: {path}")
            continue
        text = _pdf_text(path)
        if filename not in BASELINE_ONLY_FIGURES and "WeatherBridge" not in text:
            errors.append(f"WeatherBridge is absent from performance figure: {path}")
        normalized_text = _normalized_pdf_text(text)
        for legacy_name in legacy_names:
            if legacy_name in normalized_text:
                errors.append(
                    f"legacy model name remains in figure {path}: {legacy_name}"
                )
        if filename == "fig_weatherbridge_spectra_tau23.pdf":
            errors.extend(
                f"{path}: {error}"
                for error in _spectral_figure_text_errors(text)
            )

    errors.extend(_statistics_errors(paper_dir))
    errors.extend(_postselection_data_errors(paper_dir))
    errors.extend(_completion_marker_errors(paper_dir))

    errors.extend(_font_errors(_run_text("pdffonts", str(pdf_path))))
    errors.extend(
        _raster_errors(_run_text("pdfimages", "-list", str(pdf_path)))
    )
    errors.extend(
        _font_errors(
            _run_text("pdffonts", str(supplementary_pdf_path)),
            document_label="supplementary PDF",
        )
    )
    errors.extend(
        "supplementary PDF: " + error
        for error in _raster_errors(
            _run_text("pdfimages", "-list", str(supplementary_pdf_path))
        )
    )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    args = parser.parse_args()
    errors = collect_errors(args.paper_dir.resolve())
    if errors:
        print("NPJ final-publication preflight failed:")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("NPJ final-publication preflight passed")


if __name__ == "__main__":
    main()
