#!/usr/bin/env python3
"""Hydra 1.3 entry point for WTI evaluation.

Dispatches to one of the eval scripts based on `eval.kind`:

  - memmap_6h        → tools/eval/batch_eval_memmap.main()
  - memmap_12h       → tools/eval/batch_eval_12h_memmap.main()
  - crps_ensemble    → tools/eval/eval_ensemble_crps.main()
  - sh_spectra       → tools/eval/sh_spectra_ens_sdyff.main()
  - region_season_6h → tools/eval/region_season_eval.main()
  - region_season_12h→ tools/eval/region_season_12h_eval.main()
  - sh_spectra_12h   → tools/eval/sh_energy_spectra_12h.main()
  - sh_spectra_ens   → tools/eval/sh_spectra_ens_sdyff.main()
  - bootstrap_ci     → tools/eval/bootstrap_ci_{6h,12h}.main() | bootstrap_ci.main()

We forward the Hydra cfg to the underlying argparse-driven main() by
synthesising sys.argv from the cfg fields. This keeps the existing
implementations untouched while exposing a unified Hydra interface.

Examples:

    python eval.py eval=memmap_6h \\
        ++eval.models='dcae_skip:logs/.../last.ckpt,fuxi:logs/.../last.ckpt'
    python eval.py +legacy=eval_2020_paper_baseline_24ch
    python eval.py eval=crps_ensemble eval.fm_ckpt=... eval.base_ckpt=...
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))


def _apply_env_vars(env_vars: dict) -> None:
    if not env_vars:
        return
    for k, v in env_vars.items():
        os.environ[str(k)] = str(v)
        print(f"  [env] {k}={v}")


def _normalize_models(raw: Any) -> str:
    """Normalize Hydra-loaded models field to the comma-separated string
    that batch_eval_memmap.py expects."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        # Strip block-scalar newlines and whitespace.
        return ",".join(part.strip() for part in raw.replace("\n", "").split(",") if part.strip())
    if isinstance(raw, (list, tuple)):
        return ",".join(str(x).strip() for x in raw if str(x).strip())
    return str(raw)


def _build_argv(cfg: DictConfig) -> List[str]:
    """Build argv (positional [0] is program name) for the underlying eval script."""
    e = cfg.eval
    kind = str(e.kind)
    argv = [f"eval:{kind}"]

    def add(flag: str, val: Any) -> None:
        if val is None:
            return
        argv.extend([flag, str(val)])

    if kind in ("memmap_6h", "memmap_12h"):
        add("--memmap-dir", e.memmap_dir)
        add("--test-year", e.test_year)
        if not bool(e.no_acc) and e.climatology:
            add("--climatology", e.climatology)
        add("--stats-path", e.stats_path)
        add("--surface-stats-path", e.surface_stats_path)
        add("--static-path", e.static_path)
        models_raw = e.models
        if hasattr(models_raw, "_content"):  # DictConfig/ListConfig
            models_raw = OmegaConf.to_container(models_raw, resolve=True)
        models = _normalize_models(models_raw)
        if not models:
            raise ValueError(f"eval.models is empty for {kind} — set "
                             f"++eval.models='name1:ckpt1,name2:ckpt2'")
        add("--models", models)
        if kind == "memmap_6h":
            add("--out-acc-dir", e.out_acc_dir or e.out_dir)
            add("--out-rmse-dir", e.out_dir)
        else:
            add("--out-dir", e.out_dir)
            add("--paper-tag", e.paper_tag or "12h_eval")
            add("--max-tau-hours", e.max_tau_hours)
            add("--eval-hours", e.eval_hours or "1,2,3,4,5,6,7,8,9,10,11")
            add("--seen-tau", e.seen_tau or "1,2,3,5,7,9,10,11")
            add("--unseen-tau", e.unseen_tau or "4,6,8")
            if bool(e.no_acc):
                argv.append("--no-acc")
            if e.device:
                add("--device", e.device)
        add("--batch-size", e.batch_size)
        add("--num-workers", e.num_workers)
        add("--samples-per-date", e.samples_per_date)
        add("--eval-days-per-month", e.eval_days_per_month)
        if e.keep_n_channels is not None:
            add("--keep-n-channels", e.keep_n_channels)
    elif kind == "crps_ensemble":
        add("--mode", e.crps_mode)
        if e.ckpt:
            add("--ckpt", e.ckpt)
        if e.fm_ckpt:
            add("--fm_ckpt", e.fm_ckpt)
        if e.base_ckpt:
            add("--base_ckpt", e.base_ckpt)
        add("--n_ensemble", e.n_ensemble)
        add("--memmap_dir", e.memmap_dir)
        add("--data_root", str(REPO))
        add("--year", e.test_year)
        add("--batch_size", e.batch_size)
        add("--num_workers", e.num_workers)
        add("--samples_per_date", e.samples_per_date)
        add("--out_rmse", e.out_rmse)
        add("--out_crps", e.out_crps)
        add("--model_name", e.model_name)
        add("--max_tau_hours", e.max_tau_hours)
    elif kind in ("sh_spectra", "sh_spectra_ens"):
        add("--ckpt", e.ckpt)
        add("--memmap_dir", e.memmap_dir)
        add("--data_root", str(REPO))
        add("--year", e.test_year)
        add("--n_ensemble", e.n_ensemble)
        add("--n_dates", e.sh_n_dates)
        add("--tau_hours", e.sh_tau_hours)
        add("--channels", e.sh_channels)
        add("--lmax", e.sh_lmax)
        add("--batch_size", e.batch_size)
        add("--num_workers", e.num_workers)
        add("--samples_per_date", e.samples_per_date)
        add("--max_tau_hours", e.max_tau_hours)
        add("--out_dir", e.out_dir)
        add("--model_name", e.model_name)
        if e.sh_envs:
            add("--envs", e.sh_envs)
    elif kind in ("region_season_6h", "region_season_12h"):
        add("--memmap-dir", e.memmap_dir)
        add("--test-year", e.test_year)
        add("--stats-path", e.stats_path)
        add("--surface-stats-path", e.surface_stats_path)
        add("--static-path", e.static_path)
        models_raw = e.models
        if hasattr(models_raw, "_content"):
            models_raw = OmegaConf.to_container(models_raw, resolve=True)
        models = _normalize_models(models_raw)
        if not models:
            raise ValueError(f"eval.models is empty for {kind} — set "
                             f"++eval.models='name1:ckpt1,name2:ckpt2'")
        add("--models", models)
        add("--out-dir", e.out_dir)
        add("--batch-size", e.batch_size)
        add("--num-workers", e.num_workers)
        add("--samples-per-date", e.samples_per_date)
        add("--eval-days-per-month", e.eval_days_per_month)
        if kind == "region_season_12h":
            add("--max-tau-hours", e.max_tau_hours)
            add("--eval-hours", e.eval_hours or "1,2,3,4,5,6,7,8,9,10,11")
            if e.keep_n_channels is not None:
                add("--keep-n-channels", e.keep_n_channels)
            if e.device:
                add("--device", e.device)
    elif kind == "sh_spectra_12h":
        add("--memmap-dir", e.memmap_dir)
        add("--test-year", e.test_year)
        add("--stats-path", e.stats_path)
        add("--surface-stats-path", e.surface_stats_path)
        add("--static-path", e.static_path)
        add("--ckpt", e.ckpt)
        add("--model-name", e.model_name or "model")
        add("--model-kind", e.sh_model_kind or "hermite")
        if e.sh_envs:
            add("--envs", e.sh_envs)
        add("--out-dir", e.out_dir)
        add("--taus", e.sh_taus or "2,3,5,8")
        add("--channels", e.sh_channels)
        add("--lmax", e.sh_lmax)
        if e.keep_n_channels is not None:
            add("--keep-n-channels", e.keep_n_channels)
        add("--batch-size", e.batch_size)
        add("--samples-per-date", e.samples_per_date)
        add("--eval-days-per-month", e.eval_days_per_month)
        add("--max-tau-hours", e.max_tau_hours)
        if e.device:
            add("--device", e.device)
        if bool(getattr(e, "sh_skip_existing", False)):
            argv.append("--skip-existing")
    elif kind == "bootstrap_ci":
        # Dispatch to one of three legacy scripts based on `horizon`.
        horizon = str(getattr(e, "horizon", "6h"))
        if horizon == "6h":
            add("--metrics-dir", e.bs_metrics_dir or e.out_dir)
            if getattr(e, "bs_out_24ch", None):
                add("--out-24ch", e.bs_out_24ch)
            if getattr(e, "bs_out_27ch", None):
                add("--out-27ch", e.bs_out_27ch)
            add("--bootstraps", e.bs_bootstraps)
            add("--seed", e.bs_seed)
        elif horizon == "12h":
            add("--metrics-dir", e.bs_metrics_dir or e.out_dir)
            if getattr(e, "bs_out", None):
                add("--out", e.bs_out)
            add("--bootstraps", e.bs_bootstraps)
            add("--seed", e.bs_seed)
            if bool(getattr(e, "bs_write_tex", False)):
                argv.append("--write-tex")
                if getattr(e, "bs_tex_out", None):
                    add("--tex-out", e.bs_tex_out)
        elif horizon == "legacy":
            if getattr(e, "bs_out", None):
                add("--out", e.bs_out)
            add("--bootstraps", e.bs_bootstraps)
        else:
            raise ValueError(
                f"bootstrap_ci: unknown horizon={horizon!r} "
                f"(expected 6h | 12h | legacy)"
            )
    else:
        raise ValueError(f"Unknown eval.kind={kind}")

    return argv


def _dispatch(kind: str, cfg: DictConfig | None = None) -> None:
    """Invoke the appropriate eval main() with current sys.argv."""
    if kind == "memmap_6h":
        from tools.eval.batch_eval_memmap import main
        main()
    elif kind == "memmap_12h":
        from tools.eval.batch_eval_12h_memmap import main
        main()
    elif kind == "crps_ensemble":
        from tools.eval.eval_ensemble_crps import main
        main()
    elif kind in ("sh_spectra", "sh_spectra_ens"):
        from tools.eval.sh_spectra_ens_sdyff import main
        main()
    elif kind == "region_season_6h":
        from tools.eval.region_season_eval import main
        main()
    elif kind == "region_season_12h":
        from tools.eval.region_season_12h_eval import main
        main()
    elif kind == "sh_spectra_12h":
        from tools.eval.sh_energy_spectra_12h import main
        main()
    elif kind == "bootstrap_ci":
        horizon = str(getattr(cfg.eval, "horizon", "6h")) if cfg is not None else "6h"
        if horizon == "6h":
            from tools.eval.bootstrap_ci_6h import main
        elif horizon == "12h":
            from tools.eval.bootstrap_ci_12h import main
        elif horizon == "legacy":
            from tools.eval.bootstrap_ci import main
        else:
            raise ValueError(
                f"bootstrap_ci: unknown horizon={horizon!r} "
                f"(expected 6h | 12h | legacy)"
            )
        main()
    else:
        raise ValueError(f"Unknown eval.kind={kind}")


@hydra.main(config_path="conf", config_name="eval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print("=" * 70)
    print("WTI evaluation — Hydra entry point")
    print("=" * 70)
    print(OmegaConf.to_yaml(cfg, resolve=False))
    print("-" * 70)

    # Apply model-level env-vars from cfg.model.env_vars (e.g. SDYFF_NLAT).
    try:
        env_vars = OmegaConf.to_container(cfg.model.env_vars, resolve=True) or {}
        _apply_env_vars(env_vars)
    except Exception:
        pass

    argv = _build_argv(cfg)
    print(f"[dispatch] eval.kind={cfg.eval.kind}")
    print(f"[dispatch] argv: {' '.join(argv)}")
    sys.argv = argv

    _dispatch(str(cfg.eval.kind), cfg=cfg)


if __name__ == "__main__":
    main()
