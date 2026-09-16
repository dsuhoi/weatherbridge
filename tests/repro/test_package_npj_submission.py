from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from tools.repro.package_npj_submission import (
    collect_source_files,
    package_submission,
    write_deterministic_zip,
)


def test_deterministic_zip_is_byte_identical(tmp_path: Path) -> None:
    entries = {"b.txt": b"second\n", "a.txt": b"first\n"}
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"

    write_deterministic_zip(first, entries)
    write_deterministic_zip(second, entries)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(
        second.read_bytes()
    ).digest()
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.txt", "b.txt"]
        assert archive.getinfo("a.txt").date_time == (1980, 1, 1, 0, 0, 0)


def test_source_collection_uses_only_local_recorded_dependencies(
    tmp_path: Path,
) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    required_contents = {
        "main.tex": "\\input{table.tex}\n",
        "main.bbl": "% bibliography\n",
        "supplementary.tex": "Supplement\n",
        "references.bib": "% references\n",
        "sn-jnl.cls": "% class\n",
        "sn-nature.bst": "% style\n",
        "manuscript_npj.pdf": "pdf\n",
        "supplementary.pdf": "supplementary pdf\n",
        "supplementary_data_1_statistics.csv": "p_raw_two_sided\n0.1\n",
        "supplementary_data_1_statistics.manifest.json": "{}\n",
        "supplementary_data_2_postselection_2022.json": "{}\n",
        "supplementary_data_2_postselection_2022.manifest.json": "{}\n",
        ".journal_figures_verified": json.dumps({"status": "complete"}),
        "table.tex": "table\n",
    }
    for name, contents in required_contents.items():
        (paper / name).write_text(contents)
    (paper / "main.fls").write_text(
        "INPUT ./main.tex\n"
        "INPUT ./table.tex\n"
        "INPUT /usr/share/texmf/system.sty\n"
    )
    (paper / "supplementary.fls").write_text(
        "INPUT ./supplementary.tex\n"
    )

    collected = {
        path.relative_to(paper).as_posix()
        for path in collect_source_files(paper)
    }

    assert "table.tex" in collected
    assert "main.tex" in collected
    assert "supplementary.tex" in collected
    assert "manuscript_npj.pdf" not in collected
    assert "supplementary.pdf" not in collected
    assert all("system.sty" not in name for name in collected)


def test_submission_archive_contains_supplementary_pdf(tmp_path: Path) -> None:
    paper = tmp_path / "paper"
    paper.mkdir()
    manuscript = b"main pdf\n"
    supplementary = b"supplementary pdf\n"
    required_contents = {
        "main.tex": b"Main\n",
        "main.bbl": b"% bibliography\n",
        "main.fls": b"INPUT ./main.tex\n",
        "supplementary.tex": b"Supplement\n",
        "supplementary.fls": b"INPUT ./supplementary.tex\n",
        "references.bib": b"% references\n",
        "sn-jnl.cls": b"% class\n",
        "sn-nature.bst": b"% style\n",
        "manuscript_npj.pdf": manuscript,
        "supplementary.pdf": supplementary,
        "supplementary_data_1_statistics.csv": b"metric\nrmse\n",
        "supplementary_data_1_statistics.manifest.json": b"{}\n",
        "supplementary_data_2_postselection_2022.json": b"{}\n",
        "supplementary_data_2_postselection_2022.manifest.json": b"{}\n",
    }
    for name, contents in required_contents.items():
        (paper / name).write_bytes(contents)
    statistics_sources = paper / "supplementary_data_1_sources"
    statistics_sources.mkdir()
    (statistics_sources / "source.json").write_text("{}\n")
    marker = {
        "status": "complete",
        "paper_sha256": hashlib.sha256(manuscript).hexdigest(),
    }
    (paper / ".journal_figures_verified").write_text(json.dumps(marker))
    output = tmp_path / "submission.zip"

    package_submission(paper, output)

    with zipfile.ZipFile(output) as archive:
        assert archive.read("manuscript_npj.pdf") == manuscript
        assert archive.read("supplementary.pdf") == supplementary
        assert archive.read("supplementary_data_1_sources/source.json") == b"{}\n"
        assert archive.read("supplementary_data_2_postselection_2022.json") == b"{}\n"
        manifest = json.loads(archive.read("submission_manifest.json"))
        assert manifest["supplementary_sha256"] == hashlib.sha256(
            supplementary
        ).hexdigest()
