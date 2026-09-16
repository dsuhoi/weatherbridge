"""Shim to satisfy physicsnemo.core.{module,meta,version_check} imports.

The vendored physicsnemo files import:
  - physicsnemo.core.module.Module       (their nn.Module subclass with metadata)
  - physicsnemo.core.meta.ModelMetaData  (dataclass with optimization hints)
  - physicsnemo.core.version_check.OptionalImport
We replace each with a minimal stub.

Also installs a minimal `jaxtyping` shim into sys.modules if the real package
isn't available (the vendored code uses Float[Tensor, "shape"] only as a type
annotation; at runtime we can ignore the shape spec).
"""
from __future__ import annotations

import sys as _sys

if "jaxtyping" not in _sys.modules:  # pragma: no cover
    try:
        import jaxtyping  # type: ignore  # noqa: F401
    except ImportError:
        import types as _types

        _jax = _types.ModuleType("jaxtyping")

        class _Annotated:
            """Minimal Float[T, "..."] stand-in: subscript returns the inner type."""

            def __init__(self, name: str = "Float") -> None:
                self.name = name

            def __getitem__(self, _key):
                # _key is (T, "shape spec") — return T so isinstance checks work.
                if isinstance(_key, tuple) and len(_key) >= 1:
                    return _key[0]
                return _key

        _jax.Float = _Annotated("Float")
        _jax.Int = _Annotated("Int")
        _jax.Bool = _Annotated("Bool")
        _jax.Array = _Annotated("Array")
        _sys.modules["jaxtyping"] = _jax

from dataclasses import dataclass, field
from typing import Any, Optional

import torch.nn as nn


class Module(nn.Module):
    """Drop-in replacement for `physicsnemo.core.module.Module`.

    The upstream class adds metadata, ONNX-export helpers and registry support.
    For our use we only need nn.Module: the constructor accepts an optional
    `meta=` kwarg (which physicsnemo subclasses pass through) and ignores it.
    """

    def __init__(self, *args: Any, meta: Any | None = None, **kwargs: Any) -> None:
        super().__init__()
        self._meta = meta


@dataclass
class ModelMetaData:
    """Drop-in replacement for `physicsnemo.core.meta.ModelMetaData` dataclass."""

    name: str = ""
    jit: bool = True
    cuda_graphs: bool = False
    amp: bool = True
    onnx_cpu: bool = True
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    var_dim: int = -1
    func_torch: bool = True
    auto_grad: bool = True


class OptionalImport:
    """Stub used by mlp_layers.py — upstream lazy-imports apex/transformer-engine.

    Returns a dummy object whose attributes are None so that any feature flags
    behind it default to "disabled" (good — we don't have those packages).
    """

    def __init__(self, package_name: str, *args: Any, **kwargs: Any) -> None:
        self.package_name = package_name

    def __bool__(self) -> bool:  # always falsy
        return False

    def __getattr__(self, name: str) -> Any:
        return None
