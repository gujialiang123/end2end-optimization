"""Both Falcon-H1 shims in one module.

Python imports exactly one sitecustomize, the first on sys.path, so putting
mamba_inject and ssd_inject on PYTHONPATH together silently loses whichever
comes second -- which cost one audit run. Anything needed at once lives here.

  MAMBA_SSU_AUTOINIT=triton   lazily initialise the selective_state_update
                              backend (bench_one_batch never builds a Scheduler)
  SSD_TILES=chunk_state:64,64,64;chunk_scan:64,64,64
                              override the SSD kernels' hardcoded 16x16x16 tiles
  FALCON_FUSION_PATCH=convtriton
                              route the causal conv through the Triton
                              implementation, which reads strides instead of
                              demanding a contiguous copy of a transposed view

Each is independent; set any combination.
"""
import os
import sys


def _init_mamba():
    from sglang.srt.layers.attention.mamba.ops import ssu_dispatch

    orig = ssu_dispatch.selective_state_update

    def wrapper(*args, **kwargs):
        if ssu_dispatch._mamba_ssu_backend is None:
            name = os.environ.get("MAMBA_SSU_AUTOINIT", "triton")
            ssu_dispatch._mamba_ssu_backend = ssu_dispatch._BACKEND_REGISTRY[name]()
            print(f"[fh_inject] selective_state_update backend '{name}' "
                  f"initialised lazily", flush=True)
        return orig(*args, **kwargs)

    ssu_dispatch.selective_state_update = wrapper
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "selective_state_update", None) is orig:
            mod.selective_state_update = wrapper


class _Launcher:
    """Carry extra constexpr kwargs into a Triton kernel launch."""

    def __init__(self, kernel, extra):
        self._kernel, self._extra = kernel, extra

    def __getitem__(self, grid):
        inner = self._kernel[grid]
        extra = self._extra

        def run(*args, **kwargs):
            kwargs.update(extra)
            return inner(*args, **kwargs)

        return run

    def __getattr__(self, name):
        return getattr(self._kernel, name)


def _init_tiles(spec):
    from sglang.srt.layers.attention.mamba.ops import (
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    targets = {
        "chunk_state": (ssd_chunk_state, "_chunk_state_fwd_kernel"),
        "chunk_scan": (ssd_chunk_scan, "_chunk_scan_fwd_kernel"),
        "state_passing": (ssd_state_passing, "_state_passing_fwd_kernel"),
    }
    applied = []
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        name, tiles = part.split(":")
        vals = [int(v) for v in tiles.split(",")]
        extra = ({"BLOCK_SIZE": vals[0]} if name == "state_passing"
                 else dict(zip(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"), vals)))
        mod, attr = targets[name]
        setattr(mod, attr, _Launcher(getattr(mod, attr), extra))
        applied.append(f"{name}={extra}")
    print(f"[ssd_inject] applied: {'; '.join(applied)}", flush=True)


def _init_falcon(spec):
    """Defer the conv patch until the backend module is imported."""
    import importlib.abc
    import importlib.machinery
    import importlib.util

    # convtriton patches the attention backend; foldmul patches the model
    # file. Hooking foldmul on the backend module fails, because that module is
    # imported from inside falcon_h1's own import and the class does not exist
    # yet -- a circular-import AttributeError. Each patch waits for its own
    # module.
    wants = {w.strip() for w in spec.split(",") if w.strip()}
    target = ("sglang.srt.models.falcon_h1" if "foldmul" in wants
              else "sglang.srt.layers.attention.hybrid_linear_attn_backend")

    class _Loader(importlib.abc.Loader):
        def __init__(self, inner):
            self._inner = inner

        def create_module(self, spec):
            return self._inner.create_module(spec)

        def exec_module(self, module):
            self._inner.exec_module(module)
            import falcon_fusion_patch

            falcon_fusion_patch.apply(module)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target_mod=None):
            if fullname != target:
                return None
            sys.meta_path.remove(self)
            try:
                found = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if found is None or found.loader is None:
                return None
            found.loader = _Loader(found.loader)
            return found

    sys.meta_path.insert(0, _Finder())


try:
    if os.environ.get("FALCON_FUSION_PATCH"):
        _init_falcon(os.environ["FALCON_FUSION_PATCH"])
    if os.environ.get("MAMBA_SSU_AUTOINIT"):
        _init_mamba()
    if os.environ.get("SSD_TILES"):
        _init_tiles(os.environ["SSD_TILES"])
except Exception as exc:  # pragma: no cover - diagnostic only
    print(f"[fh_inject] not applied: {exc}", flush=True)
