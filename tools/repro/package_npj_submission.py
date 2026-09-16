#!/usr/bin/env python3
"""Create a deterministic, evidence-bound npj submission source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

SOURCE_SUFFIXES = {
    ".bbl",
    ".bib",
    ".bst",
    ".cls",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".sty",
    ".tex",
}
REQUIRED_PAPER_FILES = (
    "main.tex",
    "main.bbl",
    "main.fls",
    "supplementary.tex",
    "supplementary.fls",
    "references.bib",
    "sn-jnl.cls",
    "sn-nature.bst",
    "manuscript_npj.pdf",
    "supplementary.pdf",
    "supplementary_data_1_statistics.csv",
    "supplementary_data_1_statistics.manifest.json",
    "supplementary_data_2_postselection_2022.json",
    "supplementary_data_2_postselection_2022.manifest.json",
    ".journal_figures_verified",
)


def _sha256_bytes(contents: bytes) -> str:
    return hashlib.sha256(contents).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_source_files(paper_dir: Path) -> list[Path]:
    paper_dir = paper_dir.resolve()
    paths: set[Path] = set()
    for recorder_name in ("main.fls", "supplementary.fls"):
        recorder = paper_dir / recorder_name
        if not recorder.is_file():
            raise FileNotFoundError(f"missing LaTeX recorder output: {recorder}")
        for line in recorder.read_text(errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            value = line[6:].strip()
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = paper_dir / candidate
            candidate = candidate.resolve()
            if (
                candidate.is_relative_to(paper_dir)
                and candidate.is_file()
                and candidate.suffix.lower() in SOURCE_SUFFIXES
                and candidate.name
                not in {"main.pdf", "manuscript_npj.pdf", "supplementary.pdf"}
            ):
                paths.add(candidate)
    for name in REQUIRED_PAPER_FILES:
        path = paper_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing submission dependency: {path}")
        if name not in {
            "main.fls",
            "supplementary.fls",
            "manuscript_npj.pdf",
            "supplementary.pdf",
        }:
            paths.add(path.resolve())
    return sorted(paths, key=lambda path: path.relative_to(paper_dir).as_posix())


def write_deterministic_zip(path: Path, entries: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(entries):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, entries[name], compress_type=zipfile.ZIP_DEFLATED)
    os.replace(temporary, path)


def package_submission(paper_dir: Path, output: Path) -> dict[str, object]:
    paper_dir = paper_dir.resolve()
    manuscript = paper_dir / "manuscript_npj.pdf"
    final_marker_path = paper_dir / ".journal_figures_verified"
    final_marker = json.loads(final_marker_path.read_text())
    if final_marker.get("status") != "complete":
        raise ValueError("final publication marker is not complete")
    manuscript_hash = _sha256(manuscript)
    if final_marker.get("paper_sha256") != manuscript_hash:
        raise ValueError("final publication marker does not bind the manuscript")

    entries: dict[str, bytes] = {
        "manuscript_npj.pdf": manuscript.read_bytes(),
        "supplementary.pdf": (paper_dir / "supplementary.pdf").read_bytes(),
        "supplementary_data_1_statistics.csv": (
            paper_dir / "supplementary_data_1_statistics.csv"
        ).read_bytes(),
        "supplementary_data_1_statistics.manifest.json": (
            paper_dir / "supplementary_data_1_statistics.manifest.json"
        ).read_bytes(),
        "supplementary_data_2_postselection_2022.json": (
            paper_dir / "supplementary_data_2_postselection_2022.json"
        ).read_bytes(),
        "supplementary_data_2_postselection_2022.manifest.json": (
            paper_dir / "supplementary_data_2_postselection_2022.manifest.json"
        ).read_bytes(),
    }
    statistics_sources = paper_dir / "supplementary_data_1_sources"
    if not statistics_sources.is_dir():
        raise FileNotFoundError(
            f"missing statistics source snapshots: {statistics_sources}"
        )
    for source in sorted(statistics_sources.iterdir()):
        if source.is_file():
            entries[f"supplementary_data_1_sources/{source.name}"] = (
                source.read_bytes()
            )
    for source in collect_source_files(paper_dir):
        relative = source.relative_to(paper_dir).as_posix()
        if relative.startswith("supplementary_data_1_statistics"):
            continue
        entries[f"source/{relative}"] = source.read_bytes()

    payload_files = {
        name: {"sha256": _sha256_bytes(contents), "size_bytes": len(contents)}
        for name, contents in sorted(entries.items())
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "venue": "npj Climate and Atmospheric Science",
        "article_type": "Article",
        "entrypoint": "source/main.tex",
        "build_command": "latexmk -pdf -halt-on-error main.tex",
        "manuscript_sha256": manuscript_hash,
        "supplementary_sha256": _sha256(paper_dir / "supplementary.pdf"),
        "final_marker_sha256": _sha256(final_marker_path),
        "champion_sha256": final_marker.get("champion_sha256"),
        "statistics_csv_sha256": final_marker.get("statistics_csv_sha256"),
        "postselection_data_sha256": final_marker.get(
            "postselection_data_sha256"
        ),
        "files": payload_files,
    }
    entries["submission_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    checksums = "".join(
        f"{_sha256_bytes(contents)}  {name}\n"
        for name, contents in sorted(entries.items())
    )
    entries["SHA256SUMS"] = checksums.encode()
    write_deterministic_zip(output, entries)
    archive_hash = _sha256(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{archive_hash}  {output.name}\n"
    )
    return {
        "path": str(output),
        "sha256": archive_hash,
        "entry_count": len(entries),
        "size_bytes": output.stat().st_size,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper/weatherbridge_npj_submission.zip"),
    )
    args = parser.parse_args()
    result = package_submission(args.paper_dir, args.out)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
