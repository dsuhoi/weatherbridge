from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPOSITORY_ROOT
    / "tools"
    / "train"
    / "sync_local_corr_inference_sources_cloudru.sh"
)
FILES = (
    "tools/eval/capmatched_loader.py",
    "weather_time_interp/model/weatherbridge_upr_lite_model.py",
    "weather_time_interp/model/weatherbridge_upr_scaled_model.py",
)
EXPERIMENTS = (
    "exp_upr_implicit_global_14m_hf_135_14m_6h_s202707_protocol_v2",
    "exp_upr_query_match_14m_hf_135_14m_6h_s202707_protocol_v1",
    (
        "exp_upr_spherical_implicit_global_14m_24ch_6h_2014_19_"
        "sparse135_lr1e4_eb16_s202707"
    ),
    (
        "exp_upr_endpoint_implicit_global_14m_24ch_6h_2014_19_"
        "sparse135_lr1e4_eb16_s202707"
    ),
    (
        "exp_upr_implicit_global_14m_nohf_24ch_6h_2014_19_"
        "sparse135_lr1e4_eb16_s202707"
    ),
    "exp_flow_pp3_hf_135_14m_6h_s202707_protocol_v2",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_syncs_only_frozen_files_after_terminal_markers(tmp_path: Path) -> None:
    target = tmp_path / "target"
    log_root = tmp_path / "logs"
    target.mkdir()
    log_root.mkdir()
    for relative in FILES:
        path = target / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stale\n")
    untouched = target / "weather_time_interp" / "memmap_dataset.py"
    untouched.write_text("keep\n")
    for experiment in EXPERIMENTS:
        (log_root / f"{experiment}.terminal").touch()

    manifest = log_root / "manifest.json"
    env = os.environ.copy()
    env.update(
        {
            "SOURCE_ROOT": str(REPOSITORY_ROOT),
            "TARGET_ROOT": str(target),
            "LOG_ROOT": str(log_root),
            "MANIFEST": str(manifest),
            "QUIET_SEC": "0",
            "WAIT_SEC": "0.01",
        }
    )
    subprocess.run(["bash", str(SCRIPT)], check=True, env=env)

    for relative in FILES:
        assert _sha256(target / relative) == _sha256(REPOSITORY_ROOT / relative)
    assert untouched.read_text() == "keep\n"
    payload = json.loads(manifest.read_text())
    assert payload["schema_version"] == 1
    assert set(payload["files"]) == set(FILES)
