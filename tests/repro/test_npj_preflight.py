import csv
import hashlib
import json
from pathlib import Path

from tools.repro.preflight_npj_submission import (
    collect_errors,
    _caption_errors,
    _command_arguments,
    _completion_marker_errors,
    _display_item_errors,
    _expanded_tex_source,
    _font_errors,
    _frontmatter_errors,
    _graphics_paths,
    _latex_word_count,
    _normalized_pdf_text,
    _npj_source_style_errors,
    _raster_errors,
    _spectral_figure_text_errors,
    _statistics_errors,
    _supplementary_display_reference_errors,
)


def test_final_preflight_requires_author_declarations(tmp_path: Path) -> None:
    for filename in (
        "main.tex", "main.pdf", "main.log", "supplementary.tex",
        "supplementary.pdf", "supplementary.log",
    ):
        (tmp_path / filename).write_text("present")
    errors = collect_errors(tmp_path)
    assert len(errors) == 2
    assert any("code_archive_statement.tex" in error for error in errors)
    assert any("funding_statement.tex" in error for error in errors)


def test_pdf_text_normalization_joins_wrapped_placeholder_phrases() -> None:
    assert _normalized_pdf_text(
        "Contribution statement must be completed\n before submission."
    ) == "contribution statement must be completed before submission."


def test_font_preflight_rejects_type3_and_unembedded_fonts() -> None:
    output = """name type encoding emb sub uni object ID
---- ---- -------- --- --- --- ---------
AAAAAA+CMR10 Type 1 Builtin yes yes yes 1 0
BBBBBB+Sans Type 3 Custom yes yes yes 2 0
Helvetica Type 1 WinAnsi no no no 3 0
"""

    assert _font_errors(output) == [
        "Type 3 font in final PDF: BBBBBB+Sans",
        "unembedded font in final PDF: Helvetica",
    ]
    assert _font_errors(output, document_label="supplementary PDF") == [
        "Type 3 font in supplementary PDF: BBBBBB+Sans",
        "unembedded font in supplementary PDF: Helvetica",
    ]


def test_every_supplementary_figure_is_cited() -> None:
    source = Path("paper/supplementary.tex").read_text()
    labels = {
        line.split(r"\label{", 1)[1].split("}", 1)[0]
        for line in source.splitlines()
        if r"\label{fig:" in line
    }

    assert labels
    assert all(rf"\ref{{{label}}}" in source for label in labels)


def test_every_main_figure_is_cited() -> None:
    source = Path("paper/main.tex").read_text()
    labels = {
        line.split(r"\label{", 1)[1].split("}", 1)[0]
        for line in source.splitlines()
        if r"\label{fig:" in line
    }

    assert labels
    assert all(rf"\ref{{{label}}}" in source for label in labels)


def test_completion_markers_must_exist_and_be_complete(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    expected = (
        paper_dir / "images/.weatherbridge_full_year_figures_complete",
        paper_dir / "images/.weatherbridge_rmse_maps_complete",
        paper_dir / "images/.hres_field_figures_complete",
        tmp_path / "metrics/journal_spectra_v6/state/.complete",
    )

    assert len(_completion_marker_errors(paper_dir)) == len(expected)
    for path in expected:
        _write(path, '{"status": "complete"}\n')
    assert _completion_marker_errors(paper_dir) == []

    expected[0].write_text('{"status": "running"}\n')
    assert _completion_marker_errors(paper_dir) == [
        f"incomplete completion marker: {expected[0]}"
    ]


def test_raster_preflight_rejects_low_resolution_and_cmyk() -> None:
    output = """page num type width height color comp bpc enc interp object ID x-ppi y-ppi size ratio
---- --- ---- ----- ------ ----- ---- --- --- ------ ------ -- ----- ----- ---- -----
1 0 image 100 100 rgb 3 8 image no 10 0 300 320 1K 10%
2 1 image 100 100 cmyk 4 8 image no 11 0 299 301 1K 10%
2 2 smask 100 100 gray 1 8 image no 12 0 72 72 1K 10%
"""

    assert _raster_errors(output) == [
        "unsupported raster color space in page 2 image 1: cmyk",
        "raster resolution below 300 ppi in page 2 image 1: 299 x 301",
    ]


def test_spectral_figure_preflight_requires_signed_cospectrum_label() -> None:
    assert _spectral_figure_text_errors("Signed cospectrum") == []
    assert _spectral_figure_text_errors("Spectral coherence") == [
        "spectral figure does not label the signed cospectrum",
        "spectral figure still labels coherence",
    ]


def test_graphics_parser_resolves_extensionless_pdf(tmp_path: Path) -> None:
    image = tmp_path / "images" / "figure.pdf"
    image.parent.mkdir()
    image.write_bytes(b"pdf")

    assert _graphics_paths(
        r"\includegraphics[width=\textwidth]{images/figure}",
        tmp_path,
    ) == [image]


def test_graphics_parser_preserves_explicit_suffix(tmp_path: Path) -> None:
    assert _graphics_paths(
        r"\includegraphics{images/figure.png}",
        tmp_path,
    ) == [tmp_path / "images" / "figure.png"]


def test_caption_parser_handles_nested_latex() -> None:
    source = (
        r"\caption[Short]{WeatherBridge at $\tau{=}3$ with "
        r"\textbf{nested {detail}}.}"
    )

    assert _command_arguments(source, "caption") == [
        r"WeatherBridge at $\tau{=}3$ with \textbf{nested {detail}}."
    ]
    assert _latex_word_count(_command_arguments(source, "caption")[0]) == 6
    assert _caption_errors(source, word_limit=5) == [
        "figure/table caption 1 exceeds 5 words: 6"
    ]


def test_caption_preflight_accepts_limit_exactly() -> None:
    assert _caption_errors(r"\caption{one two three}", word_limit=3) == []


def test_display_item_preflight_expands_inputs_and_enforces_limit(
    tmp_path: Path,
) -> None:
    paper_dir = tmp_path / "paper"
    paper_dir.mkdir()
    (paper_dir / "main.tex").write_text(
        "\n".join(
            [r"\begin{figure}" for _ in range(4)]
            + [r"\input{tables}"]
        )
    )
    (paper_dir / "tables.tex").write_text(
        "\n".join(r"\begin{table*}" for _ in range(5))
    )

    expanded = _expanded_tex_source(paper_dir / "main.tex")
    assert _display_item_errors(expanded) == []
    assert _display_item_errors(expanded, limit=9) == []
    assert _display_item_errors(expanded, limit=8) == [
        "main text has 9 display items (4 figures, 5 tables); "
        "internal editorial limit is 8"
    ]


def test_supplementary_display_items_must_be_cited_from_main() -> None:
    supplementary = r"""
\begin{figure}\label{fig:case}\end{figure}
\begin{table*}\label{tab:scores}\end{table*}
"""
    assert _supplementary_display_reference_errors(
        r"See Fig.~\ref{S-fig:case} and Table~\ref{S-tab:scores}.",
        supplementary,
    ) == []
    assert _supplementary_display_reference_errors(
        r"See Fig.~\ref{S-fig:case}.",
        supplementary,
    ) == [
        "supplementary display items not cited in main text: tab:scores"
    ]


def test_frontmatter_preflight_enforces_word_limits() -> None:
    source = r"\title[Short]{one two three}\abstract{four five six}"
    assert _frontmatter_errors(source) == []

    long_title = " ".join(f"word{index}" for index in range(16))
    assert _frontmatter_errors(
        rf"\title{{{long_title}}}\abstract{{short}}"
    ) == ["title exceeds 15 words: 16"]


def test_npj_source_style_rejects_split_authors_and_funding_section() -> None:
    main = r"""
\author*[1]{\fnm{Ada} \sur{Lovelace}}
\author[1]{\fnm{Grace} \sur{Hopper}}
\bmhead{Acknowledgements}
\bmhead{Funding}
\caption{\textbf{a}, Legacy panel label.}
"""
    supplement = r"""
\author*[1]{\fnm{Ada} \sur{Lovelace}}
\section{Supplementary Methods}
"""

    assert _npj_source_style_errors(main, supplement) == [
        "funding must be included under Acknowledgements",
        "Supplementary Methods section is not permitted",
        "main and supplementary author lists differ: "
        "['Lovelace', 'Hopper'] != ['Lovelace']",
        "panel labels must use lower-case letters with closing parentheses: a",
    ]


def test_npj_source_style_accepts_aligned_documents() -> None:
    authors = r"""
\author*[1]{\fnm{Ada} \sur{Lovelace}}
\author[1]{\fnm{Grace} \sur{Hopper}}
"""
    main = authors + r"\bmhead{Acknowledgements}\caption{\textbf{a)} Panel.}"
    supplement = authors + r"\section{Additional diagnostics}"
    assert _npj_source_style_errors(main, supplement) == []


def _write(path: Path, contents: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_statistics_preflight_binds_csv_and_sources(tmp_path: Path) -> None:
    paper_dir = tmp_path / "paper"
    source = tmp_path / "metrics/source.json"
    source_hash = _write(source, "{}\n")
    champion = tmp_path / "metrics/journal_champion_v1/final.json"
    champion_hash = _write(
        champion,
        json.dumps({"status": "confirmed", "winner": "weatherbridge_detail"}),
    )
    exporter = tmp_path / "tools/eval/export_journal_statistics.py"
    exporter_hash = _write(exporter, "# exporter\n")
    csv_path = paper_dir / "supplementary_data_1_statistics.csv"
    csv_path.parent.mkdir(parents=True)
    row = {
        "candidate": "weatherbridge_detail",
        "reference": "weatherdcae_14m",
        "family": "rmse",
        "metric": "rmse",
        "delta_left_minus_right": "-0.1",
        "ci95_low": "-0.2",
        "ci95_median": "-0.1",
        "ci95_high": "-0.01",
        "p_raw_two_sided": "0.02",
        "p_censored": "false",
        "source_path": str(source),
        "source_sha256": source_hash,
    }
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "winner": "weatherbridge_detail",
        "champion_sha256": champion_hash,
        "exporter_sha256": exporter_hash,
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "row_count": 1,
        "source_sha256": {str(source): source_hash},
    }
    (paper_dir / "supplementary_data_1_statistics.manifest.json").write_text(
        json.dumps(manifest)
    )

    assert _statistics_errors(paper_dir) == []


def test_statistics_preflight_rejects_stale_source(tmp_path: Path) -> None:
    test_statistics_preflight_binds_csv_and_sources(tmp_path)
    source = tmp_path / "metrics/source.json"
    source.write_text('{"changed": true}\n')

    assert any(
        "statistics source hash mismatch" in error
        for error in _statistics_errors(tmp_path / "paper")
    )
