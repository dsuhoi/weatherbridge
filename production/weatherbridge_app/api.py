"""FastAPI service for immutable WeatherBridge forecast bundles."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from production.weatherbridge_app.store import BundleStore

STATIC_DIR = Path(__file__).with_name("static")


def _parse_utc(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(
        dt.timezone.utc
    )


def create_app(store_root: str | Path | None = None) -> FastAPI:
    root = Path(store_root or os.getenv("WB_FORECAST_STORE", "/tmp/weatherbridge"))
    store = BundleStore(root)
    stale_after = int(os.getenv("WB_STALE_AFTER_SECONDS", "28800"))
    app = FastAPI(title="WeatherBridge realtime API", version="1.0.0")
    app.state.store = store
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready() -> JSONResponse:
        run_id = store.latest_id()
        if run_id is None:
            return JSONResponse(
                {"status": "not_ready", "reason": "no forecast"}, status_code=503
            )
        manifest = store.load_manifest(run_id)
        age = (
            dt.datetime.now(dt.timezone.utc) - _parse_utc(manifest["generated_at"])
        ).total_seconds()
        status_code = 200 if age <= stale_after else 503
        return JSONResponse(
            {
                "status": "ready" if status_code == 200 else "stale",
                "run_id": run_id,
                "age_seconds": age,
            },
            status_code=status_code,
        )

    @app.get("/api/v1/runs")
    async def runs(limit: int = 24) -> dict[str, object]:
        return {"runs": store.list_runs(max(1, min(limit, 100)))}

    @app.get("/api/v1/runs/latest")
    async def latest() -> dict[str, object]:
        run_id = store.latest_id()
        if run_id is None:
            raise HTTPException(
                status_code=404, detail="no forecast has been published"
            )
        return store.load_manifest(run_id)

    @app.get("/api/v1/runs/{run_id}")
    async def run_manifest(run_id: str) -> dict[str, object]:
        try:
            return store.load_manifest(run_id)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/v1/runs/{run_id}/fields/{field}/{tau}.webp")
    async def field_image(run_id: str, field: str, tau: int) -> FileResponse:
        try:
            path = store.image_path(run_id, field, tau)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(
            path,
            media_type="image/webp",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    @app.get("/api/v1/events")
    async def events(request: Request) -> StreamingResponse:
        async def stream():
            previous = None
            while not await request.is_disconnected():
                current = store.latest_id()
                if current != previous:
                    yield f"event: forecast\ndata: {json.dumps({'run_id': current})}\n\n"
                    previous = current
                else:
                    yield ": keepalive\n\n"
                await asyncio.sleep(5)

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


app = create_app()
