import hashlib
import io
import tarfile

import pytest

from zenodo.validate_release import validate_archive


@pytest.mark.parametrize("fault", [None, "unlisted", "duplicate", "symlink", "digest"])
def test_archive_checks_contents_against_manifest(tmp_path, fault):
    path = tmp_path / "source.tar.gz"
    data = b"example\n"
    digest = hashlib.sha256(data if fault != "digest" else b"wrong").hexdigest()
    files = [("source.txt", data), ("MANIFEST.sha256", f"{digest}  source.txt\n".encode())]
    if fault in {"unlisted", "duplicate"}:
        files.append(("extra.txt" if fault == "unlisted" else "source.txt", data))
    with tarfile.open(path, "w:gz") as archive:
        for name, content in files:
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
        if fault == "symlink":
            member = tarfile.TarInfo("link")
            member.type = tarfile.SYMTYPE
            member.linkname = "/outside"
            archive.addfile(member)
    if fault is None:
        validate_archive(path, {"source.txt"}, "MANIFEST.sha256")
    else:
        with pytest.raises(AssertionError):
            validate_archive(path, {"source.txt"}, "MANIFEST.sha256")
