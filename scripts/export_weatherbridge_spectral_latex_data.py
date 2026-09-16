#!/usr/bin/env python3
"""Export validated WeatherBridge spectral data for the paper PGFPlots figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPECTRA = ROOT / "metrics" / "journal_spectra_v6" / "6h_2020"
DEFAULT_OUT = ROOT / "paper" / "figs" / "data"
SPECTRAL_FILES = {
    "linear": "linear_tau{tau}.npz",
    "fuxi": "fuxi_tau{tau}.npz",
    "modafno": "modafno_tau{tau}.npz",
    "sdyff": "sdyff_tau{tau}.npz",
    "pixelattn_vfi": "pixelattn_vfi_tau{tau}.npz",
    "weatherdcae": "weatherdcae_14m_tau{tau}.npz",
    "flow_spectral": "flow_spectral_tau{tau}.npz",
}
SPECTRAL_FIELDS = ("Q850", "U850")
SPECTRAL_TAUS = (2, 3)


def _write_table(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("ell power_ratio signed_cospectrum\n")
        for row in rows:
            handle.write(" ".join(str(value) for value in row) + "\n")


def _validate_index(left: np.lib.npyio.NpzFile, right: np.lib.npyio.NpzFile) -> None:
    for key in ("window_year", "window_t0"):
        if not np.array_equal(left[key], right[key]):
            raise ValueError(f"spectral window index differs for {key}")


def export_spectra(spectra_dir: Path, out_dir: Path) -> None:
    for tau in SPECTRAL_TAUS:
        loaded = {
            model: np.load(spectra_dir / pattern.format(tau=tau), allow_pickle=False)
            for model, pattern in SPECTRAL_FILES.items()
        }
        reference = loaded["flow_spectral"]
        for value in loaded.values():
            _validate_index(reference, value)
            metadata = json.loads(str(value["metadata_json"].item()))
            if (
                metadata.get("schema_version") != 6
                or metadata.get("sht_grid")
                != "wb2_0p25_2x2_block_average_v1_latitude_strip_area_sht"
                or metadata.get("sample_strategy") != "all_valid_anchor_windows"
            ):
                raise ValueError("spectral figure requires dense schema-v6 metrics")
        for field in SPECTRAL_FIELDS:
            for model, arrays in loaded.items():
                channels = [str(value) for value in arrays["channel_names"]]
                index = channels.index(field)
                ell = np.asarray(arrays["ell"], dtype=np.int64)
                mask = (ell >= 80) & (ell <= 180)
                ratio = arrays["pred_El"][index] / np.clip(
                    arrays["gt_El"][index], 1.0e-30, None
                )
                signed_cospectrum = arrays["signed_cospectrum_l"][index]
                rows = [
                    [
                        int(degree),
                        f"{float(power):.9f}",
                        f"{float(cospectrum):.9f}",
                    ]
                    for degree, power, cospectrum in zip(
                        ell[mask], ratio[mask], signed_cospectrum[mask]
                    )
                ]
                _write_table(
                    out_dir / f"spectra_{field}_tau{tau}_{model}.dat",
                    rows,
                )
        for arrays in loaded.values():
            arrays.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra-dir", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    export_spectra(args.spectra_dir, args.out_dir)
    print(f"wrote WeatherBridge spectral data to {args.out_dir}")


if __name__ == "__main__":
    main()
