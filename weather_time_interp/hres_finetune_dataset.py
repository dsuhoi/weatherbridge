"""Training dataset for IFS HRES forecast anchors and ERA5 interior targets."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from tools.eval.batch_eval_forecast_anchor import (
    Era5Cache,
    _forecast_manifest_provenance,
    read_init,
)
from tools.train.training_protocol import memmap_dataset_provenance
from weather_time_interp.normalization import (
    PAPER_CHANNELS_24,
    load_channel_stats,
)


class HRESForecastAnchorDataset(Dataset):
    """Pair same-forecast HRES anchors with ERA5 at an interior valid time."""

    PL_NAMES = ["T", "U", "V", "Q", "Z"]
    PL_LEVELS = [1000, 925, 850, 700]
    SURF_NAMES = ["t2m", "u10", "v10", "mslp", "sst", "tcc", "tcwv"]

    def __init__(
        self,
        forecast_dir: str | Path,
        era5_memmap_dir: str | Path,
        years: list[int],
        *,
        max_tau_hours: int = 6,
        train: bool = True,
        train_hours: list[int] | None = None,
        eval_hours: list[int] | None = None,
        lead_stride_hours: int = 24,
        maximum_left_lead_hours: int = 120,
        forecast_error_augmentation_repeats: int = 1,
        forecast_error_scale_min: float = 1.0,
        forecast_error_scale_max: float = 1.0,
        forecast_error_augmentation_seed: int = 0,
        expose_anchor_error: bool = False,
        stats_path: str | Path,
        surface_stats_path: str | Path,
    ) -> None:
        if max_tau_hours != 6:
            raise ValueError("HRES weight fine-tuning is frozen to the 6 h task")
        if not years or len(years) != len(set(years)):
            raise ValueError("years must be a non-empty unique list")
        if lead_stride_hours <= 0 or maximum_left_lead_hours <= 0:
            raise ValueError("forecast lead bounds must be positive")
        if forecast_error_augmentation_repeats < 1:
            raise ValueError("forecast-error augmentation repeats must be positive")
        if not (
            np.isfinite(forecast_error_scale_min)
            and np.isfinite(forecast_error_scale_max)
            and 0.0 <= forecast_error_scale_min
            <= 1.0
            <= forecast_error_scale_max
        ):
            raise ValueError(
                "forecast-error scales must be finite and bracket 1.0"
            )
        if not train and forecast_error_augmentation_repeats != 1:
            raise ValueError("forecast-error augmentation is train-only")

        self.forecast_dir = Path(forecast_dir).resolve()
        self.era5_memmap_dir = Path(era5_memmap_dir).resolve()
        self.years = list(years)
        self.max_tau_hours = int(max_tau_hours)
        self.train = bool(train)
        self.lead_stride_hours = int(lead_stride_hours)
        self.maximum_left_lead_hours = int(maximum_left_lead_hours)
        self.forecast_error_augmentation_repeats = int(
            forecast_error_augmentation_repeats
        )
        self.forecast_error_scale_min = float(forecast_error_scale_min)
        self.forecast_error_scale_max = float(forecast_error_scale_max)
        self.forecast_error_augmentation_seed = int(
            forecast_error_augmentation_seed
        )
        # Degradation-aware supervision: return the measured anchor error field
        # (anchor minus the ERA5 analysis valid at the same time) so a model can
        # be trained to predict how degraded its own inputs are. Off by default
        # so existing frozen campaigns keep their exact batch contract.
        self.expose_anchor_error = bool(expose_anchor_error)
        self.channel_names = [
            f"{variable}{level}"
            for variable in self.PL_NAMES
            for level in self.PL_LEVELS
        ]
        self.surface_variables = list(self.SURF_NAMES)

        hours = train_hours if train else eval_hours
        if not hours:
            hours = list(range(1, self.max_tau_hours))
        self.tau_hours = [int(value) for value in hours]
        invalid = [
            value
            for value in self.tau_hours
            if value <= 0 or value >= self.max_tau_hours
        ]
        if invalid or len(self.tau_hours) != len(set(self.tau_hours)):
            raise ValueError(f"invalid or duplicate query hours: {self.tau_hours}")

        stats = load_channel_stats(stats_path, surface_stats_path)
        if tuple(stats.channel_names) != tuple(PAPER_CHANNELS_24):
            raise ValueError("normalization does not use the paper's 24 fields")
        mean = np.asarray(stats.mean, dtype=np.float32)
        std = np.asarray(stats.std, dtype=np.float32)
        self._mean = mean.reshape(-1, 1, 1)
        self._std = std.reshape(-1, 1, 1)
        self.mu = torch.from_numpy(mean[:20]).view(-1, 1, 1)
        self.sigma = torch.from_numpy(std[:20]).view(-1, 1, 1)
        self.surface_mu = torch.from_numpy(mean[20:]).view(-1, 1, 1)
        self.surface_sigma = torch.from_numpy(std[20:]).view(-1, 1, 1)

        self._manifest = self._load_manifest()
        self._records: list[dict[str, Any]] = []
        self.index: list[tuple[int, int, int, int]] = []
        target_years: set[int] = set()
        for binary_path in sorted(self.forecast_dir.glob("init_*.bin")):
            sidecar_path = binary_path.with_suffix(".json")
            metadata = json.loads(sidecar_path.read_text())
            init_time = np.datetime64(metadata["init_time"], "h")
            init_year = int(str(init_time)[:4])
            if init_year not in self.years:
                continue
            self._validate_record(binary_path, sidecar_path, metadata)
            leads = [int(value) for value in metadata["lead_hours"]]
            lead_to_index = {lead: index for index, lead in enumerate(leads)}
            record_index = len(self._records)
            self._records.append(
                {
                    "binary": binary_path,
                    "sidecar": sidecar_path,
                    "metadata": metadata,
                    "init_time": init_time,
                }
            )
            for left_index, left_lead in enumerate(leads):
                right_lead = left_lead + self.max_tau_hours
                if (
                    right_lead not in lead_to_index
                    or left_lead >= self.maximum_left_lead_hours
                    or left_lead % self.lead_stride_hours
                ):
                    continue
                right_index = lead_to_index[right_lead]
                for tau_hour in self.tau_hours:
                    valid_time = init_time + np.timedelta64(
                        left_lead + tau_hour,
                        "h",
                    )
                    target_years.add(int(str(valid_time)[:4]))
                    self.index.append(
                        (record_index, left_index, right_index, tau_hour)
                    )

        if not self._records:
            raise ValueError(
                f"{self.forecast_dir}: no HRES initialisations for {self.years}"
            )
        if not self.index:
            raise ValueError("HRES split has no eligible anchor/query samples")
        self._forecast_provenance = _forecast_manifest_provenance(
            self.forecast_dir,
            [record["binary"] for record in self._records],
        )
        for year in sorted(target_years):
            for suffix in ("bin", "json"):
                path = self.era5_memmap_dir / f"wb2_{year}.{suffix}"
                if not path.is_file():
                    raise ValueError(f"missing ERA5 target artifact: {path}")
        self.target_years = sorted(target_years)
        self._forecast_arrays: dict[int, np.ndarray] = {}
        self._era5_cache: Era5Cache | None = None

    def _load_manifest(self) -> dict[str, Any]:
        path = self.forecast_dir / "forecast_archive_manifest.json"
        payload = json.loads(path.read_text())
        if (
            payload.get("schema_version") != 2
            or payload.get("archive_kind")
            != "weatherbench2_ifs_hres_forecast_anchors"
        ):
            raise ValueError(f"{path}: unsupported forecast archive")
        return payload

    def _validate_record(
        self,
        binary_path: Path,
        sidecar_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        if metadata.get("channel_order") != list(PAPER_CHANNELS_24):
            raise ValueError(f"{sidecar_path}: non-canonical channel order")
        if metadata.get("dtype") != "float32":
            raise ValueError(f"{sidecar_path}: expected float32")
        shape = tuple(int(value) for value in metadata.get("shape", ()))
        if len(shape) != 4 or shape[1:] != (24, 360, 720):
            raise ValueError(f"{sidecar_path}: wrong HRES tensor shape {shape}")
        record = self._manifest.get("files", {}).get(binary_path.name)
        if record is None or record.get("sidecar") != sidecar_path.name:
            raise ValueError(f"{binary_path}: absent from archive manifest")
        stat = binary_path.stat()
        if (
            stat.st_size != int(record.get("binary_size_bytes", -1))
            or stat.st_mtime_ns != int(record.get("binary_mtime_ns", -1))
        ):
            raise ValueError(f"{binary_path}: binary identity changed")

    def __len__(self) -> int:
        return len(self.index) * self.forecast_error_augmentation_repeats

    def _forecast_error_scale(self, base_item: int, repeat: int) -> float:
        if repeat == 0:
            return 1.0
        token = (
            f"{self.forecast_error_augmentation_seed}:"
            f"{base_item}:{repeat}"
        ).encode("ascii")
        uniform = int.from_bytes(
            hashlib.sha256(token).digest()[:8],
            "big",
        ) / float(2**64 - 1)
        return self.forecast_error_scale_min + uniform * (
            self.forecast_error_scale_max - self.forecast_error_scale_min
        )

    def _forecast_array(self, record_index: int) -> np.ndarray:
        if record_index not in self._forecast_arrays:
            record = self._records[record_index]
            array, _, _ = read_init(record["binary"], record["sidecar"])
            self._forecast_arrays[record_index] = array
        return self._forecast_arrays[record_index]

    def _era5_hour(self, valid_time: np.datetime64) -> np.ndarray:
        if self._era5_cache is None:
            self._era5_cache = Era5Cache(self.era5_memmap_dir)
        target = self._era5_cache.hour(valid_time)
        if target is None:
            raise RuntimeError(f"ERA5 target is unavailable at {valid_time}")
        return target

    def _normalize(self, values: np.ndarray, label: str) -> torch.Tensor:
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (24, 360, 720) or not np.isfinite(values).all():
            raise RuntimeError(f"{label}: invalid 24x360x720 field")
        normalized = (values - self._mean) / self._std
        return torch.from_numpy(np.array(normalized, copy=True, order="C"))

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        base_count = len(self.index)
        repeat, base_item = divmod(item, base_count)
        record_index, left_index, right_index, tau_hour = self.index[base_item]
        record = self._records[record_index]
        array = self._forecast_array(record_index)
        leads = [int(value) for value in record["metadata"]["lead_hours"]]
        left_lead = leads[left_index]
        left_valid_time = record["init_time"] + np.timedelta64(
            left_lead,
            "h",
        )
        target_valid_time = left_valid_time + np.timedelta64(tau_hour, "h")
        timestamp = datetime.fromisoformat(str(left_valid_time))
        day_fraction = timestamp.timetuple().tm_yday / 365.0
        hour_fraction = timestamp.hour / 24.0
        x0 = self._normalize(array[left_index], "left HRES anchor")
        x1 = self._normalize(array[right_index], "right HRES anchor")
        error_scale = self._forecast_error_scale(base_item, repeat)
        clean_x0 = clean_x1 = None
        if error_scale != 1.0 or self.expose_anchor_error:
            clean_x0 = self._normalize(
                self._era5_hour(left_valid_time),
                "left ERA5 augmentation anchor",
            )
            clean_x1 = self._normalize(
                self._era5_hour(
                    left_valid_time + np.timedelta64(self.max_tau_hours, "h")
                ),
                "right ERA5 augmentation anchor",
            )
        if error_scale != 1.0:
            x0 = clean_x0 + error_scale * (x0 - clean_x0)
            x1 = clean_x1 + error_scale * (x1 - clean_x1)
        anchor_error = None
        if self.expose_anchor_error:
            # Measured after the augmentation rescale, so it is the error of the
            # anchors the model actually receives.
            anchor_error = 0.5 * (
                (x0 - clean_x0).abs() + (x1 - clean_x1).abs()
            )
        if anchor_error is not None:
            return {
                "x0": x0,
                "x1": x1,
                "target": self._normalize(
                    self._era5_hour(target_valid_time),
                    "ERA5 target",
                ),
                "anchor_error": anchor_error,
                "time_emb": torch.tensor(
                    [
                        np.sin(2 * np.pi * day_fraction),
                        np.cos(2 * np.pi * day_fraction),
                        np.sin(2 * np.pi * hour_fraction),
                        np.cos(2 * np.pi * hour_fraction),
                    ],
                    dtype=torch.float32,
                ),
                "tau": torch.tensor(
                    [tau_hour / self.max_tau_hours],
                    dtype=torch.float32,
                ),
                "tau_hour": torch.tensor([tau_hour], dtype=torch.long),
                "forecast_lead_hours": torch.tensor(
                    [left_lead],
                    dtype=torch.float32,
                ),
                "forecast_error_scale": torch.tensor(
                    [error_scale],
                    dtype=torch.float32,
                ),
            }
        return {
            "x0": x0,
            "x1": x1,
            "target": self._normalize(
                self._era5_hour(target_valid_time),
                "ERA5 target",
            ),
            "time_emb": torch.tensor(
                [
                    np.sin(2 * np.pi * day_fraction),
                    np.cos(2 * np.pi * day_fraction),
                    np.sin(2 * np.pi * hour_fraction),
                    np.cos(2 * np.pi * hour_fraction),
                ],
                dtype=torch.float32,
            ),
            "tau": torch.tensor(
                [tau_hour / self.max_tau_hours],
                dtype=torch.float32,
            ),
            "tau_hour": torch.tensor([tau_hour], dtype=torch.long),
            "forecast_lead_hours": torch.tensor(
                [left_lead],
                dtype=torch.float32,
            ),
            "forecast_error_scale": torch.tensor(
                [error_scale],
                dtype=torch.float32,
            ),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_forecast_arrays"] = {}
        state["_era5_cache"] = None
        return state


def hres_finetune_dataset_provenance(
    dataset: HRESForecastAnchorDataset,
) -> dict[str, Any]:
    """Bind a split to the archive manifest, selected inits and ERA5 targets."""
    manifest_path = dataset.forecast_dir / "forecast_archive_manifest.json"
    identity = {
        "forecast_root": str(dataset.forecast_dir),
        "forecast_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "forecast_archive": dataset._forecast_provenance,
        "selected_initialisations": sorted(
            record["binary"].name for record in dataset._records
        ),
        "years": dataset.years,
        "query_hours": dataset.tau_hours,
        "anchor_spacing_hours": dataset.max_tau_hours,
        "lead_stride_hours": dataset.lead_stride_hours,
        "expose_anchor_error": dataset.expose_anchor_error,
        "maximum_left_lead_hours": dataset.maximum_left_lead_hours,
        "base_sample_count": len(dataset.index),
        "sample_count": len(dataset),
        "forecast_error_augmentation": {
            "method": "paired_state_conditioned_HRES_minus_ERA5_scaling",
            "train_only": True,
            "repeats": dataset.forecast_error_augmentation_repeats,
            "scale_min": dataset.forecast_error_scale_min,
            "scale_max": dataset.forecast_error_scale_max,
            "seed": dataset.forecast_error_augmentation_seed,
            "paired_anchor_errors": True,
        },
        "target_source": "ERA5_at_intermediate_valid_time",
        "future_analysis_as_input": False,
        "era5_targets": memmap_dataset_provenance(
            dataset.era5_memmap_dir,
            dataset.target_years,
        ),
    }
    return {
        **identity,
        "identity_sha256": hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
