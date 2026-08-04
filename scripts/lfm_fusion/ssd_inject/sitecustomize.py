"""Override the mamba SSD kernels' hardcoded tile sizes at launch time.

_chunk_state_fwd_kernel, _chunk_scan_fwd_kernel and _state_passing_fwd_kernel
declare BLOCK_SIZE_M/N/K as tl.constexpr defaulting to 16, carry no
@triton.autotune, and their call sites never override them. On Falcon-H1 those
three are 59 % of prefill kernel time, all running 16x16x16 on an H200.

Rather than reconstruct their thirty-odd stride arguments in a microbenchmark --
which is easy to get subtly wrong -- this wraps the kernel objects so the launch
carries whichever tiles we want. The grid lambdas already read
META["BLOCK_SIZE_M"], so they follow automatically.

    SSD_TILES=chunk_state:64,64,32 python -m sglang.bench_one_batch ...

Kernel names: chunk_state, chunk_scan, state_passing (the last takes a single
BLOCK_SIZE). Multiple entries separated by ';'. Absent variable = stock code.
"""
import os

_SPEC = os.environ.get("SSD_TILES", "")

if _SPEC:

    class _Launcher:
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

    def _parse(spec):
        out = {}
        for part in spec.split(";"):
            part = part.strip()
            if not part:
                continue
            name, tiles = part.split(":")
            vals = [int(v) for v in tiles.split(",")]
            if name == "state_passing":
                out[name] = {"BLOCK_SIZE": vals[0]}
            else:
                out[name] = dict(zip(("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K"),
                                     vals))
        return out

    try:
        from sglang.srt.layers.attention.mamba.ops import (
            ssd_chunk_scan,
            ssd_chunk_state,
            ssd_state_passing,
        )

        _TARGETS = {
            "chunk_state": (ssd_chunk_state, "_chunk_state_fwd_kernel"),
            "chunk_scan": (ssd_chunk_scan, "_chunk_scan_fwd_kernel"),
            "state_passing": (ssd_state_passing, "_state_passing_fwd_kernel"),
        }
        applied = []
        for name, extra in _parse(_SPEC).items():
            mod, attr = _TARGETS[name]
            setattr(mod, attr, _Launcher(getattr(mod, attr), extra))
            applied.append(f"{name}={extra}")
        print(f"[ssd_inject] applied: {'; '.join(applied)}", flush=True)
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[ssd_inject] not applied: {exc}", flush=True)
