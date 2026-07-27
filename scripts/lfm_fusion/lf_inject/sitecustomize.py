"""Applies the LFM2.5 fusion patches at import time (loaded via PYTHONPATH).

The patch has to land on the class *before* the model is instantiated, and
`sglang.srt.models.lfm2_moe` is imported lazily by the model registry — long
after `sitecustomize` runs. A polling timer would be a race, so instead we
install a `sys.meta_path` finder that defers to the normal import machinery and
runs the patch as soon as that specific module finishes executing.

Does nothing at all unless `LFM_FUSION_PATCH` is set, so the stock serving path
is unchanged when the variable is absent.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys

TARGET = "sglang.srt.models.lfm2_moe"

if os.environ.get("LFM_FUSION_PATCH"):

    class _PatchingLoader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            try:
                import lfm_fusion_patch

                lfm_fusion_patch.apply()
            except Exception as e:  # never take the server down over this
                print(f"[lfm_fusion_patch] FAILED to apply: {e!r}", flush=True)
                raise

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != TARGET:
                return None
            # Re-run discovery with this finder removed to get the real spec.
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
