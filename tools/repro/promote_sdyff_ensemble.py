#!/usr/bin/env python3
"""Publish the six completed N=21 cohorts without changing archived N=1 scores."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.eval.align_window_artifact import align_artifact, sha256_file
from tools.repro.promote_corrected_baselines import MODEL_ALIASES

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "metrics/sdyff_ensemble_n21"
TARGET = ROOT / "metrics/journal_unified"


def validate_ensemble(source: Path, reference: Path) -> None:
    data = json.loads(source.read_text())
    ref = json.loads(reference.read_text())
    summary = json.loads(source.with_name("ensemble_summary.json").read_text())
    if summary["source_sha256"] != sha256_file(source):
        raise ValueError("ensemble summary does not match source")
    protocol = data["ensemble_protocol"]
    if protocol["sizes"] != [1, 4, 16, 21] or protocol["method_sizes"]["model"] != 21:
        raise ValueError("expected the completed N=21 protocol")
    for key in ("channel_names", "years", "delta_t_hours", "n_per_tau", "seen_tau", "unseen_tau"):
        if data[key] != ref[key]:
            raise ValueError(f"ensemble/reference mismatch: {key}")
    if data["checkpoint_provenance"]["sha256"] != ref["checkpoint_provenance"]["sha256"]:
        raise ValueError("ensemble uses a different S-DYff checkpoint")
    for tau, values in data["per_tau"].items():
        for key, value in values["bilinear"].items():
            if not np.isclose(value, ref["per_tau"][tau]["bilinear"][key], rtol=2e-5, atol=2e-7):
                raise ValueError(f"Linear mismatch: {tau} {key}")
        for channel in data["channel_names"]:
            for prefix in ("rmse_norm_", "rmse_phys_", "acc_"):
                if not np.isfinite(values["model"][prefix + channel]):
                    raise ValueError(f"non-finite ensemble score: {tau} {channel}")


def main() -> None:
    records = []
    for horizon in (6, 12):
        if not (RAW / f"logs/{horizon}h_full.complete").is_file():
            raise ValueError(f"incomplete {horizon}h queue")
        for year in (2020, 2021, 2022):
            cohort = "frozen_extension" if year == 2022 else "full_year"
            source = RAW / f"metrics/{horizon}h/{year}/{cohort}/sdyff_ens.json"
            old = ROOT / f"metrics/corrected_baselines_v1/{horizon}h/{year}/{cohort}"
            validate_ensemble(source, old / "sdyff.json")
            target = TARGET / f"{horizon}h_{year}"
            if (horizon, year) in ((12, 2021), (6, 2022), (12, 2022)):
                core = ROOT / (f"metrics/postselection_2022_v1/{horizon}h" if year == 2022
                               else f"metrics/detailed_benchmark_v2/{horizon}h/{year}/full_year")
                aliases = dict(MODEL_ALIASES[horizon])
                aliases.update({
                    "flow_spectral": "weatherbridge_pp3_14m_6yr_ep8.json" if horizon == 6 else "weatherbridge_14m_3yr_ep10.json",
                    "weatherdcae_14m": "weatherdcae_14m_6yr_ep8_matched.json" if horizon == 6 else "weatherdcae_14m_3yr_ep10_matched.json",
                })
                for name, alias in aliases.items():
                    origin = (old if name in MODEL_ALIASES[horizon] else core) / f"{name}.json"
                    align_artifact(source_json=origin, reference_json=core / "flow_spectral.json",
                                   output_json=target / alias,
                                   output_window=target / "window_metrics" / f"{name}.npz")
            reference = target / MODEL_ALIASES[horizon]["sdyff"]
            reference_index = json.loads(reference.read_text())["evaluation_protocol"]["index_sha256"]
            for name, alias in MODEL_ALIASES[horizon].items():
                current = json.loads((target / alias).read_text())
                if current["evaluation_protocol"]["index_sha256"] != reference_index:
                    origin = old / f"{name}.json"
                    if current["per_tau"] != json.loads(origin.read_text())["per_tau"]:
                        raise ValueError(f"refusing to change archived {name} scores")
                    align_artifact(source_json=origin, reference_json=reference,
                                   output_json=target / alias,
                                   output_window=target / "window_metrics" / f"{name}.npz")
            published = target / "sdyff_ens21.json"
            align_artifact(source_json=source,
                           reference_json=reference,
                           output_json=published,
                           output_window=target / "window_metrics/sdyff_ens21.npz")
            records.append({"horizon": horizon, "year": year,
                            "source": str(source.relative_to(ROOT)), "source_sha256": sha256_file(source),
                            "published": str(published.relative_to(ROOT)), "published_sha256": sha256_file(published)})
    (TARGET / "sdyff_ens21.manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    previous = TARGET / "corrected_baselines_v1.manifest.json"
    manifest = json.loads(previous.read_text())
    for record in manifest["records"]:
        record["paper_alias_sha256"] = sha256_file(ROOT / record["paper_alias"])
    manifest["window_alignment_update"] = "sdyff_ens21.manifest.json"
    previous.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Published {len(records)} N=21 cohorts; original N=1 aliases retained.")


if __name__ == "__main__":
    main()
