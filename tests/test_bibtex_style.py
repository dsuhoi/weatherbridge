"""Conference references must compile without an editor or publisher field."""

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which('bibtex') is None, reason='BibTeX is not installed')
def test_conference_booktitle_without_editors(tmp_path):
    style = Path(__file__).resolve().parents[1] / 'paper' / 'sn-nature.bst'
    shutil.copy2(style, tmp_path / style.name)
    (tmp_path / 'check.aux').write_text(
        '\\relax\n\\citation{*}\n\\bibstyle{sn-nature}\n\\bibdata{check}\n'
    )
    (tmp_path / 'check.bib').write_text(
        '@inproceedings{plain, author={Smith, Alice}, title={A test}, '
        'booktitle={Conference Alpha}, year={2023}}\n'
        '@incollection{edited, author={Jones, Bob}, title={Another test}, '
        'editor={Green, Carol}, booktitle={Collection Beta}, year={2024}}\n'
    )
    result = subprocess.run(
        ['bibtex', 'check'], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'check.bbl').read_text()
    assert 'Conference Alpha' in output
    assert 'Collection Beta' in output
    assert '2023' in output and '2024' in output


@pytest.mark.skipif(shutil.which('bibtex') is None, reason='BibTeX is not installed')
def test_published_doi_links_take_priority_over_urls(tmp_path):
    style = Path(__file__).resolve().parents[1] / 'paper' / 'sn-nature.bst'
    shutil.copy2(style, tmp_path / style.name)
    (tmp_path / 'check.aux').write_text(
        '\\relax\n\\citation{*}\n\\bibstyle{sn-nature}\n\\bibdata{check}\n'
    )
    (tmp_path / 'check.bib').write_text(
        '@article{journal, author={Smith, Alice}, title={Journal test}, '
        'journal={Journal Alpha}, year={2023}, volume={1}, pages={1--2}, '
        'doi={10.1234/journal}, url={https://example.org/duplicate}}\n'
        '@inproceedings{conf, author={Jones, Bob}, title={Conference test}, '
        'booktitle={Conference Beta}, year={2024}, doi={10.1234/conf}}\n'
        '@inproceedings{url, author={Green, Carol}, title={URL test}, '
        'booktitle={Conference Gamma}, year={2025}, '
        'url={https://example.org/proceedings}}\n'
    )
    result = subprocess.run(
        ['bibtex', 'check'], cwd=tmp_path, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = (tmp_path / 'check.bbl').read_text()
    assert output.count('\\doi{10.1234/journal}') == 1
    assert output.count('\\doi{10.1234/conf}') == 1
    assert 'example.org/duplicate' not in output
    assert '\\url{https://example.org/proceedings}' in output


@pytest.mark.skipif(shutil.which('latexmk') is None, reason='latexmk is not installed')
def test_external_references_build_without_cached_aux(tmp_path):
    config = Path(__file__).resolve().parents[1] / 'paper' / 'latexmkrc'
    shutil.copy2(config, tmp_path / config.name)
    (tmp_path / 'main.tex').write_text(
        '\\documentclass{article}\n\\usepackage{xr}\n'
        '\\externaldocument[S-]{supplementary}\n'
        '\\begin{document}See section \\ref{S-check}.\\end{document}\n'
    )
    (tmp_path / 'supplementary.tex').write_text(
        '\\documentclass{article}\n\\begin{document}'
        '\\section{Check}\\label{check}Text.\\end{document}\n'
    )
    result = subprocess.run(
        ['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error', 'main.tex'],
        cwd=tmp_path, text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / 'supplementary.aux').is_file()
    assert 'undefined' not in (tmp_path / 'main.log').read_text()
