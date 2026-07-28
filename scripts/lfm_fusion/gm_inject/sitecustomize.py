"""Applies the Gemma-3 fusion patch at import time (loaded via PYTHONPATH).

Same mechanism as `lf_inject`, but targeting `sglang.srt.layers.layernorm`
rather than a model module: a `sys.meta_path` finder that defers to the normal
import machinery and patches the moment that specific module finishes
executing. A timer would be a race, because the module is imported lazily and
`Gemma3RMSNorm` instances are constructed immediately afterwards — the patch
has to be in place before the first constructor runs, since `MultiPlatformOp`
binds its dispatch target in `__init__`.

Does nothing unless `GEMMA_FUSION_PATCH` is set.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

TARGET = "sglang.srt.layers.layernorm"

if os.environ.get("GEMMA_FUSION_PATCH"):

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                import gemma_fusion_patch

                gemma_fusion_patch.apply()
            except Exception as e:
                print(f"[gemma_fusion_patch] FAILED to apply: {e!r}", flush=True)
                raise

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            saved = sys.meta_path
            sys.meta_path = [f for f in saved if not isinstance(f, _Finder)]
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path = saved
            if spec is None or spec.loader is None:
                return None
            spec.loader = _PatchingLoader(spec.loader)
            return spec

    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
