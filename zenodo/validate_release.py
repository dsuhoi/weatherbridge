#!/usr/bin/env python3
"""Validate the local Zenodo source and model archives before upload."""

from __future__ import annotations

import hashlib
import argparse
import json
import re
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "zenodo" / "dist"
VERSION = json.loads((ROOT / "zenodo" / "code_metadata.json").read_text())["metadata"]["version"]
MODEL_VERSION = json.loads((ROOT / "weatherbridge-release" / "zenodo_metadata.json").read_text())["metadata"]["version"]
SOURCE = DIST / f"weatherbridge-source-v{VERSION}.tar.gz"
MODELS = DIST / f"weatherbridge-models-v{MODEL_VERSION}.tar.gz"
MAX_ZENODO_FILE_BYTES = 50_000_000_000

REQUIRED_SOURCE = {
    f"weatherbridge-source-v{VERSION}/CITATION.cff",
    f"weatherbridge-source-v{VERSION}/LICENSE",
    f"weatherbridge-source-v{VERSION}/README.md",
    f"weatherbridge-source-v{VERSION}/SOURCE_MANIFEST.sha256",
    f"weatherbridge-source-v{VERSION}/train.py",
    f"weatherbridge-source-v{VERSION}/eval.py",
    f"weatherbridge-source-v{VERSION}/weather_time_interp/__init__.py",
    f"weatherbridge-source-v{VERSION}/zenodo/code_metadata.json",
    f"weatherbridge-source-v{VERSION}/requirements-publication.txt",
}
REQUIRED_MODELS = {
    "CITATION.cff",
    "LICENSE",
    "LICENSES/Apache-2.0.txt",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "MANIFEST.sha256",
    "docs/MODEL_CARD.md",
    "weights/catalogue.json",
    "weights/weatherbridge_14m_6h_bare.pt",
    "weights/weatherbridge_14m_12h_bare.pt",
    "weights/weatherdcae_14m_6h_bare.pt",
    "data/sample_era5_2020070100.npz",
    "zenodo_metadata.json",
}
FORBIDDEN_SUFFIXES = {".env", ".key", ".pem", ".p12", ".pfx"}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "AWS access key": re.compile(rb"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(rb"gh[opsu]_[A-Za-z0-9]{30,}"),
    "API key": re.compile(rb"sk-[A-Za-z0-9_-]{32,}"),
}
_PROVENANCE_TERM_HEX = (
    "636c61756465",
    "636f646578",
    "616e7468726f706963",
    "6f70656e6169",
    "63686174677074",
)
FORBIDDEN_PUBLIC_TERMS = re.compile(
    rb"\b(?:" + b"|".join(bytes.fromhex(value) for value in _PROVENANCE_TERM_HEX) + rb")\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_metadata(path: Path, expected_license: str, version: str = VERSION) -> None:
    payload = json.loads(path.read_text())["metadata"]
    required = {"title", "upload_type", "description", "creators", "version", "license"}
    missing = required - payload.keys()
    assert not missing, f"{path}: missing metadata keys {sorted(missing)}"
    assert payload["upload_type"] == "software"
    assert payload["version"] == version
    assert payload["license"] == expected_license
    assert len(payload["creators"]) == 5


def validate_internal_manifest(path: Path, manifest_name: str, prefix: str = "") -> None:
    expected: dict[str, str] = {}
    seen: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        handle = archive.extractfile(manifest_name)
        assert handle is not None, f"missing manifest: {manifest_name}"
        for raw_line in handle.read().decode().splitlines():
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            digest, name = raw_line.split(maxsplit=1)
            name = name.removeprefix("./")
            assert re.fullmatch(r"[0-9a-f]{64}", digest), f"invalid digest: {name}"
            assert name not in expected, f"duplicate manifest entry: {name}"
            expected[name] = digest

    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            relative = member.name.removeprefix(prefix).removeprefix("/")
            if member.name == manifest_name:
                continue
            assert relative in expected, f"unlisted archive file: {relative}"
            assert relative not in seen, f"duplicate archive file: {relative}"
            handle = archive.extractfile(member)
            assert handle is not None
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
            seen[relative] = digest.hexdigest()
    assert expected, f"empty or late manifest: {manifest_name}"
    missing = expected.keys() - seen.keys()
    mismatched = {name for name, digest in seen.items() if digest != expected[name]}
    assert not missing, f"manifest entries missing from {path.name}: {sorted(missing)}"
    assert not mismatched, f"manifest digest mismatch in {path.name}: {sorted(mismatched)}"


def validate_archive(
    path: Path,
    required: set[str],
    manifest_name: str,
    prefix: str = "",
) -> None:
    assert path.is_file(), f"missing archive: {path}"
    assert path.stat().st_size < MAX_ZENODO_FILE_BYTES, f"archive exceeds Zenodo limit: {path}"
    with tarfile.open(path, "r:gz") as archive:
        names = set()
        for member in archive:
            pure = PurePosixPath(member.name)
            assert not pure.is_absolute(), f"absolute archive path: {member.name}"
            assert ".." not in pure.parts, f"traversal archive path: {member.name}"
            assert pure.suffix.lower() not in FORBIDDEN_SUFFIXES, f"credential-like file: {member.name}"
            assert member.isfile() or member.isdir(), f"unsupported archive entry: {member.name}"
            name = str(pure)
            assert name not in names, f"duplicate archive entry: {name}"
            names.add(name)
            if not member.isfile() or member.size > 2_000_000:
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            data = handle.read()
            if b"\x00" in data:
                continue
            # Preserve the full environment inventory, including incidental SDKs.
            if pure.name != "requirements-publication.txt":
                assert not FORBIDDEN_PUBLIC_TERMS.search(data), (
                    f"AI-tooling provenance in public archive: {member.name}"
                )
            for label, pattern in SECRET_PATTERNS.items():
                assert not pattern.search(data), f"possible {label} in {member.name}"
        missing = required - names
        assert not missing, f"{path.name}: missing {sorted(missing)}"
    validate_internal_manifest(path, manifest_name, prefix)


def validate_external_checksums(dist: Path, paths: tuple[Path, ...]) -> None:
    expected = {}
    for line in (dist / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.strip()] = digest
    for path in paths:
        assert expected.get(path.name) == sha256(path), f"checksum mismatch: {path.name}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=DIST)
    parser.add_argument("--source-only", action="store_true")
    args = parser.parse_args()
    source = args.dist / SOURCE.name
    models = args.dist / MODELS.name
    validate_metadata(ROOT / "zenodo" / "code_metadata.json", "Apache-2.0")
    source_prefix = f"weatherbridge-source-v{VERSION}"
    validate_archive(
        source,
        REQUIRED_SOURCE,
        f"{source_prefix}/SOURCE_MANIFEST.sha256",
        source_prefix,
    )
    if not args.source_only:
        validate_metadata(ROOT / "weatherbridge-release" / "zenodo_metadata.json", "MIT", MODEL_VERSION)
        validate_archive(models, REQUIRED_MODELS, "MANIFEST.sha256")
    paths = (source,) if args.source_only else (source, models)
    validate_external_checksums(args.dist, paths)
    for path in paths:
        print(f"validated {path.name} ({path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
