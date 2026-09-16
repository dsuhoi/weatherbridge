#!/usr/bin/env python3
"""Convert a legacy Lightning checkpoint into a released bare blob.

The research repository's own converter routes through the production
``load_model_safe``, which imports the full training stack; that path is
currently broken and, more importantly, is not something a release artifact
should depend on. This converter needs only the vendored backbone:

1. read ``state_dict`` and ``hyper_parameters`` out of the checkpoint,
2. strip the Lightning ``model.`` / ``net.`` prefix,
3. keep only the hyper-parameters the backbone's ``__init__`` accepts,
4. build the backbone and load the weights **strictly**, so a mismatch is an
   error rather than a silently half-initialised model,
5. write ``{arch, kwargs, state_dict}``.

Usage:
    python scripts/lightning_to_bare.py \\
        --ckpt path/to/epoch=7-step=35032.ckpt \\
        --module modafno_baseline_model \\
        --class WeatherModAFNOResidualLinearModel \\
        --out weights/modafno_37m_6yr_bare.pt
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def strip_prefix(state: dict, prefixes=("model.", "net.")) -> dict:
    for prefix in prefixes:
        keys = [k for k in state if k.startswith(prefix)]
        if len(keys) == len(state) and keys:
            return {k[len(prefix):]: v for k, v in state.items()}
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--module", required=True)
    parser.add_argument("--class", dest="class_name", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--set", action="append", default=[],
        help=(
            "override a kwarg: --set depth=8, --set inp_shape=360,720 for a "
            "tuple, --set modulate_filter=true for a bool. Needed where the "
            "trainer renamed a hyper-parameter on its way into the backbone, "
            "for instance modafno_depth -> depth."
        ),
    )
    args = parser.parse_args()

    raw = torch.load(str(args.ckpt), map_location="cpu", weights_only=False)
    state = strip_prefix(dict(raw["state_dict"]))
    hparams = dict(raw.get("hyper_parameters", {}))

    module = importlib.import_module(f"weatherbridge.models.{args.module}")
    cls = getattr(module, args.class_name)
    accepted = set(inspect.signature(cls.__init__).parameters)
    kwargs = {k: v for k, v in hparams.items() if k in accepted}

    def _scalar(text: str):
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                continue
        lowered = text.lower()
        if lowered in ("true", "false"):
            return lowered == "true"
        if lowered == "none":
            return None
        return text

    for override in args.set:
        key, _, value = override.partition("=")
        if "," in value:
            kwargs[key] = tuple(_scalar(part) for part in value.split(","))
        else:
            kwargs[key] = _scalar(value)

    dropped = sorted(set(hparams) - set(kwargs))
    print(f"kept {len(kwargs)} kwargs, dropped {len(dropped)} trainer-only ones")

    net = cls(**kwargs)
    net.load_state_dict(state)  # strict: a mismatch must fail here
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"built {args.class_name}: {params:.2f} M parameters, strict load OK")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"arch": args.class_name, "kwargs": kwargs, "state_dict": state},
        str(args.out),
    )
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
