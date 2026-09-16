from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from production.weatherbridge_app.api import create_app
from production.weatherbridge_app.store import BundleStore


def test_api_reports_no_forecast_separately_from_liveness(tmp_path: Path) -> None:
    empty_store = tmp_path / "not-created-by-api"
    client = TestClient(create_app(empty_store))
    assert not empty_store.exists()
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 503
    assert client.get("/api/v1/runs/latest").status_code == 404


def test_api_serves_immutable_bundle_image(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("WB_STALE_AFTER_SECONDS", "999999999")
    run_id = "20260731T120000Z-test"
    run = tmp_path / "runs" / run_id
    image = run / "images" / "t2m" / "tau_3.webp"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"RIFF-test-WEBP")
    manifest = {
        "run_id": run_id,
        "generated_at": "2026-07-31T12:00:00Z",
        "fields": {"t2m": {"images": {"3": "images/t2m/tau_3.webp"}}},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    BundleStore._atomic_json(tmp_path / "latest.json", {"run_id": run_id})
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/v1/runs/latest").json()["run_id"] == run_id
    response = client.get(f"/api/v1/runs/{run_id}/fields/t2m/3.webp")
    assert response.status_code == 200
    assert "immutable" in response.headers["cache-control"]
