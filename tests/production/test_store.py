from __future__ import annotations

import json
from pathlib import Path

import pytest

from production.weatherbridge_app.store import BundleStore


def _bundle(root: Path, run_id: str = "20260731T120000Z-test") -> BundleStore:
    store = BundleStore(root)
    image = root / "runs" / run_id / "images" / "t2m" / "tau_3.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"RIFF-test-WEBP")
    manifest = {
        "run_id": run_id,
        "generated_at": "2026-07-31T12:00:00Z",
        "fields": {"t2m": {"images": {"3": "images/t2m/tau_3.webp"}}},
    }
    (image.parents[2] / "manifest.json").write_text(json.dumps(manifest))
    BundleStore._atomic_json(root / "latest.json", {"run_id": run_id})
    return store


def test_store_resolves_latest_manifest_and_image(tmp_path: Path) -> None:
    store = _bundle(tmp_path)
    assert store.latest_id() == "20260731T120000Z-test"
    assert store.load_manifest(store.latest_id())["run_id"] == store.latest_id()
    assert store.image_path(store.latest_id(), "t2m", 3).is_file()


def test_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = _bundle(tmp_path)
    with pytest.raises(ValueError, match="invalid run id"):
        store.load_manifest("../outside")
