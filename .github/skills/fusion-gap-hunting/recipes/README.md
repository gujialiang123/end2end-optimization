# recipes

`example_candidates.json` — output of `impl/scan_fusion_gaps.py` against
sglang @ `17f7a1da1`, the run that reproduces the Gemma-3 finding.

28 candidates, **1 flagged `LIKELY REAL`**, and it is the correct one:

| candidate | siblings with a fused kernel | on the CUDA build | verdict |
|---|---|---|---|
| `Gemma3RMSNorm` | cpu `gemma3_rmsnorm_cpu`, npu `npu_gemma_rms_norm` | `gemma_rmsnorm`, `gemma_fused_add_rmsnorm` | **LIKELY REAL** → 2.13× e2e |
| `QuickGELU` | hip `gelu_quick`, npu `npu_fast_gelu` | — | rejected: no CUDA primitive |
| `NewGELU` | — | — | rejected: explicit `# TODO` upstream |

The `QuickGELU` row is the reason the per-platform check exists. It is
statically indistinguishable from the real finding until you check that
`gelu_quick` is imported only under `elif _is_hip` and is absent from the CUDA
build of `sgl_kernel`.

Reproduce:

```bash
python impl/scan_fusion_gaps.py --src /path/to/sglang \
    --python /path/to/env/bin/python --out candidates.json
```
