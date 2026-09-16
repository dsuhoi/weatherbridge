# Install jaxtyping shim before any vendored module imports it.
from . import _shim  # noqa: F401

"""Vendored ModAFNO from NVIDIA/physicsnemo (Apache-2.0).

Source commit: main branch as of 2026-05-08.
Files mirror physicsnemo paths:
  modafno.py, modembed.py        ← physicsnemo/models/afno/
  afno_layers.py, mlp_layers.py, fft.py, embedding_layers.py, activations.py, utils.py
                                  ← physicsnemo/nn/module/

Imports rewritten to relative; physicsnemo.core.{module,meta,version_check}
replaced by lightweight shims in physicsnemo_shim.py. The arithmetic code
inside the *Layer / *Mlp / Block classes is byte-identical to upstream.

WeatherModAFNOResidualLinearModel in modafno_baseline_model.py wraps
ModAFNO from this package for our pipeline.
"""

# Re-export the main user-facing classes to mirror physicsnemo.nn imports
from .afno_layers import (
    AFNO2DLayer,
    AFNOMlp,
    AFNOPatchEmbed,
    ModAFNO2DLayer,
    ModAFNOMlp,
    ScaleShiftMlp,
)
from .modembed import ModEmbedNet, SinusoidalTimestepEmbedding
from .embedding_layers import OneHotEmbedding
from .modafno import ModAFNO, Block

__all__ = [
    "AFNO2DLayer",
    "AFNOMlp",
    "AFNOPatchEmbed",
    "ModAFNO2DLayer",
    "ModAFNOMlp",
    "ScaleShiftMlp",
    "ModEmbedNet",
    "SinusoidalTimestepEmbedding",
    "OneHotEmbedding",
    "ModAFNO",
    "Block",
]
