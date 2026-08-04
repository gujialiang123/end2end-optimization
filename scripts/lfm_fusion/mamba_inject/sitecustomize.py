"""Make bench_one_batch usable on hybrid-mamba models.

`initialize_mamba_selective_state_update_backend` is only ever called from
Scheduler.__init__ (managers/scheduler.py:501). bench_one_batch does not build a
Scheduler, so every hybrid-mamba model prefills fine and then dies on the first
decode step with

    AssertionError: Mamba selective_state_update backend not initialized.

That makes the operator audit -- which is built on bench_one_batch with CUDA
graphs disabled -- unusable on Falcon-H1 and any other mamba hybrid. This shim
performs the same initialisation lazily, at the point of first use, so the audit
measures the same kernels the server would run.

It is a workaround for our tooling, not a proposed fix: the real fix belongs
upstream, either in bench_one_batch or by making the dispatch initialise itself.
"""
import os

if os.environ.get("MAMBA_SSU_AUTOINIT"):
    try:
        from sglang.srt.layers.attention.mamba.ops import ssu_dispatch

        _orig = ssu_dispatch.selective_state_update

        def _autoinit_selective_state_update(*args, **kwargs):
            if ssu_dispatch._mamba_ssu_backend is None:
                # Constructing a real ServerArgs would try to load a model
                # config; only the backend name is actually needed, so build the
                # registered class directly.
                backend = os.environ.get("MAMBA_SSU_AUTOINIT")
                cls = ssu_dispatch._BACKEND_REGISTRY[backend]
                ssu_dispatch._mamba_ssu_backend = cls()
                print(
                    f"[mamba_inject] initialised selective_state_update backend "
                    f"'{backend}' lazily (bench_one_batch does not build a Scheduler)",
                    flush=True,
                )
            return _orig(*args, **kwargs)

        ssu_dispatch.selective_state_update = _autoinit_selective_state_update

        # the symbol is re-exported and imported by name in mamba.py, so patch
        # every module that already bound it
        import sys

        for _name, _mod in list(sys.modules.items()):
            if _mod is not None and getattr(_mod, "selective_state_update", None) is _orig:
                _mod.selective_state_update = _autoinit_selective_state_update
    except Exception as exc:  # pragma: no cover - diagnostic only
        print(f"[mamba_inject] not applied: {exc}", flush=True)
