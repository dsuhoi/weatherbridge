# WeatherBridge Realtime Service

This deployment separates GPU inference from the public web process. The
worker watches an inbox for complete global anchor pairs, runs the retained
WeatherBridge checkpoint, renders all fields, and atomically advances
the latest pointer. The FastAPI service reads immutable bundles and serves the
responsive site, REST API, health checks and server-sent update events.

## Input contract

Each inbox file is an NPZ containing `x0`, `xT`, `valid_time_start`,
`valid_time_end`, and optionally `source` and `normalized`. Arrays must be
finite `(24, 360, 720)` states in the channel order defined by
`PAPER_CHANNELS_24`. Set `normalized=true` only when the arrays already use
the repository's fixed ERA5 statistics. Files appear in the inbox only after
the upstream writer has atomically renamed a completed temporary file.

The worker requires `weatherbridge_14m_6h_bare.pt` (SHA-256
`d3ba687b6309748925016019dc7a0dc1f923ef6f4207d365ccd9ade7f7e5eafc`),
`static_features_0p5.pt`, `json_stats_0p5.nc`, and
`surface_stats_0p5.json` in the mounted weight/data directories. Convert the
paper checkpoint with:

```bash
python tools/train/capmatched_to_bare.py \
  --ckpt /path/to/last.ckpt --arch weatherbridge \
  --out weights/weatherbridge_14m_6h_bare.pt
python tools/repro/check_weatherbridge_bare.py \
  weights/weatherbridge_14m_6h_bare.pt --device cuda
```

## Run

```bash
cd production
docker compose --profile gpu up --build -d
curl http://localhost:8080/healthz
curl http://localhost:8080/readyz
```

Use `--profile cpu` only for low-throughput validation. A synthetic and
prominently labelled UI smoke bundle can be published with
`docker compose --profile demo run --rm dev-seed`; it is never accepted as
scientific evidence. Production readiness remains false when no forecast is
present or the latest bundle is older than `WB_STALE_AFTER_SECONDS`.

API tests and pure rendering/store tests live under `tests/production/`.
