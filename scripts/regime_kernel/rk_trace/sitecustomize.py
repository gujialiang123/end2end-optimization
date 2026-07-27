"""Opt-in MoE kernel trace (loaded as sitecustomize when on PYTHONPATH).

Enabled by setting `RK_KERNEL_TRACE=/path/to/output.jsonl` before starting a
server. When the variable is unset this module does nothing, so the default
serving path is completely unchanged.

Records, per fused-MoE config lookup:
  timestamp, M as the kernel sees it, top_k, expert count, and the tile
  configuration actually selected for that M.

Why this matters: the regime -> M mapping must be measured, not assumed. CUDA
graph capture pads decode batches up to a captured batch size, so the M reaching
the kernel during "batch size 1" decode is not necessarily 1 x top_k. Everything
downstream (which profile bucket is used, and therefore whether specialization
helps) depends on the real distribution.
"""
from __future__ import annotations

import json
import os
import threading
import time

_PATH = os.environ.get("RK_KERNEL_TRACE")
_RATE = max(1, int(os.environ.get("RK_KERNEL_TRACE_SAMPLE", "1")))
_lock = threading.Lock()
_count = [0]
_fh = [None]
_installed = [False]


def _install():
    if _installed[0]:
        return True
    try:
        from sglang.srt.layers.moe.fused_moe_triton import fused_moe_triton_config as C
    except Exception:
        return False
    if getattr(C, "_rk_traced", False):
        _installed[0] = True
        return True
    try:
        _fh[0] = open(_PATH, "a", buffering=1)
    except Exception:
        return False
    orig = C.try_get_optimal_moe_config

    def wrapper(w1_shape, w2_shape, top_k, dtype, M, *a, **kw):
        out = orig(w1_shape, w2_shape, top_k, dtype, M, *a, **kw)
        with _lock:
            _count[0] += 1
            take = (_count[0] % _RATE) == 0
        if take:
            cfg = out[0] if isinstance(out, tuple) else out
            try:
                E, _, N = w2_shape
                _fh[0].write(json.dumps(dict(
                    ts=time.time(), M=int(M), top_k=int(top_k),
                    num_experts=int(E), N=int(N),
                    block_m=cfg.get("BLOCK_SIZE_M"),
                    block_n=cfg.get("BLOCK_SIZE_N"),
                    block_k=cfg.get("BLOCK_SIZE_K"),
                    group_m=cfg.get("GROUP_SIZE_M"),
                    num_warps=cfg.get("num_warps"),
                    num_stages=cfg.get("num_stages"))) + "\n")
            except Exception:
                pass
        return out

    C.try_get_optimal_moe_config = wrapper
    C._rk_traced = True
    _installed[0] = True
    return True


if _PATH:
    if not _install():
        # sglang is not imported yet in this process; retry shortly after start
        for delay in (5.0, 15.0, 40.0):
            threading.Timer(delay, _install).start()
    import atexit
    atexit.register(lambda: _fh[0] and _fh[0].close())
