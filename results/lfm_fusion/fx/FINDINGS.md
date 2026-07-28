# LFM2.5 ShortConv / decoder-layer FX graph export and fusion mining

**Date:** 2026-07-27 · **GPU:** H200 (sm90, 132 SM, measured achievable HBM copy BW **4224 GB/s**), device 4 only
**Stack:** torch 2.9.1+cu128, triton 3.5.1, sglang @ `/home/t-jialianggu/work/sglang` (read-only)
**Model:** LFM2.5-8B-A1B — `hidden_size=2048`, 24 layers (**18 `conv` + 6 `full_attention`**), `conv_L_cache=3`,
`conv_bias=false`, `num_dense_layers=2` ⇒ **22 MoE layers**, E=32 / top-4, `moe_intermediate=1792`,
`routed_scaling_factor=1.0`, `use_expert_bias=true` (verified in `/data/hf/LFM2.5-8B-A1B/config.json`).

Companion to `docs/2026-07-27/lfm_fusion_results.md`. **Read §0 "Relationship to prior work" first** — the two prefill
chains this study re-derives are already implemented and shipped in this repo.

---

## 0. TL;DR

| # | Finding | Status |
|---|---|---|
| **b** | *"One Triton kernel can read `proj[T,3H]` and write `Bx` directly in `[H,T]`, fusing chunk + gate-mul + transpose."* | **CONFIRMED.** Bit-exact (26/26 checks incl. varlen), **6.16× faster** than eager at T=16000. **Inductor derives this kernel automatically** — and this repo already ships it (`LFM_FUSION_PATCH=conv`). |
| **c** | Inductor fuses **nothing** in ShortConv: `found 0 possible fusions`. `causal_conv1d_fwd`/`_update` are `ExternKernelSchedulerNode`s → hard barriers. The two gate-muls are on opposite sides of the conv → `cannot fuse op1 with op3: no shared data`. | Confirmed from scheduler log |
| **root cause** | **Two** mechanisms, not one. (i) *uncoalesced*: the transpose copy runs at **588 GB/s** and `C_gate*conv_out.T` at **874 GB/s** (14 % / 21 % of achievable) — already known. (ii) **new:** `B_gate*x` is *coalesced* yet runs at **2267 GB/s (54 %)**, 1.87× slower than an identical contiguous multiply, because strided **rows** defeat `TensorIterator` vectorization. Traffic reduction available is only 1.67×; the rest of the win is layout. | Measured + verified from CUDA kernel names in 3 traces |
| **a** | 6 chains enumerated (F1–F6). F1+F2 (**already shipped**, +2.33 % e2e prefill) = 4.2–5.0 % of prefill kernel time. **Still open:** F3+F4+F5 ⇒ **2.5–3.1 % of decode kernel time**, entirely unaddressed by the current patch. | Measured |
| **d** | `triton_poi_fused_copy__mul_sum_0` = Inductor's compile of `moe_sum_reduce_torch_compile` (`fused_moe.py:291-294`). Already traffic-optimal and equal to SGLang's hand-written CUDA version. **Not a fusion gap.** | Solved |
| **e** | Ranked below. #1 = hoist `req_pool_indices.to(torch.int32)` (3-line call-site fix, bit-exact, ~1.3 % decode kernel time). | — |

**Headline number:** in long prefill the ShortConv *glue* (gate muls + transpose copy) costs **9710.1 µs = 6.18 %** of
kernel time — **more than the `in_proj` GEMM it feeds (9516.5 µs)** and **8.9× the `causal_conv1d_fwd` kernel itself
(1086.5 µs)**.

### Relationship to prior work in this repo — read this before acting on anything below

**F1 and F2 are already implemented and already A/B'd.** `scripts/lfm_fusion/lf_triton_shortconv.py`
(`fused_gate_transpose`, `fused_transpose_gate`) wired in at
`scripts/lfm_fusion/lfm_fusion_patch.py::_patched_shortconv_forward` under `LFM_FUSION_PATCH=conv`, gated by
`CONV_FUSION_MIN_TOKENS=2048`. Measured end-to-end in
`docs/2026-07-27/lfm_fusion_results.md` §6: **+2.33 % req/s on long prefill (p<1e-4)**, +0.13 % / −0.03 % (n.s.) on the two
decode regimes. Nothing in §4–§5 below should be read as a new proposal for those two chains — this study
**independently re-derives and validates** them from the compiler's own output.

What is **new** in this study:

| | |
|---|---|
| **N1** | **Inductor derives F1 by itself**, emitting a kernel structurally equivalent to the hand-written one (§5). Independent confirmation that the shipped design is the one a compiler would choose. |
| **N2** | **The mechanism is two distinct effects, not one** (§3). The prior diagnosis (`docs/2026-07-27/lfm_fusion_results.md:185-188`: *"The defect is not the amount of traffic, it is that the traffic is uncoalesced"*) named the transpose copy and the transposed read in `C_gate * conv_out` — correct for those two ops. But **`B_gate * x` is coalesced and still 1.87× slower than an identical contiguous multiply**, because strided *rows* defeat `TensorIterator` vectorization. Measured, not inferred. |
| **N3** | **As of HEAD (`0574701`) the shipped patch leaves the whole decode path stock** (the `is_decode()` branch of `_patched_shortconv_forward`), and N2 says the batched-decode regime has the *same* non-vectorized multiply as prefill (88.5 µs, **1.92 %** of regime-B decode kernel time). `fused_gate_mul` already exists at `lf_triton_shortconv.py:179` and is currently used only by the benchmark. |
| **N4** | **F4: `req_pool_indices.to(torch.int32)` runs once per conv layer**, in both the stock *and* the patched path — 18 launches/forward, **1.27 %** of regime-A decode kernel time, for a kernel that moves 12 bytes. Never previously identified. Allocation site pinned to `mem_cache/common.py:354`. |
| **N5** | **`CONV_FUSION_MIN_TOKENS=2048` is a CPU-launch threshold, not a GPU one.** Under CUDA-graph timing the fused input kernel is **4.99× faster at T=512** and 5.87× at T=1000; the wall-clock ratios at those sizes are 1.11× and 1.15×, because Triton's python launch path pins wall time at ~19 µs. The prior work identified the ~30 µs floor; this quantifies that the *GPU* crossover is below T=512, so the guard is removable by making the launch cheaper rather than by making the kernel better. |
| **N6** | **`triton_poi_fused_copy__mul_sum_0` identified** (§6) and shown to have no headroom. |
| **N7** | **The `activation`-argument route for gating is a dead end** (§7), and the conv-epilogue fusion is feasible on decode but layout-hostile on prefill. |
| **N8** | **F6 (gate+transpose into the `in_proj` GEMM epilogue) quantified and rejected** (§4) — break-even at best. |

### Two corrections to `docs/2026-07-27/lfm_fusion_results.md` §9 "Next"

* **§9.1 said G3 "needs a layout change in the `causal_conv1d_fn` call rather than a call-site edit."**
  No change to the call is needed, and the shipped patch confirms this: the fused kernel *produces* `[H, T]`
  contiguous directly, which is exactly the layout `causal_conv1d_fn` already requires (`x.stride(-1) == 1`).
  G3 **is** a call-site edit plus one new Triton kernel. See §5.
* **§9.2 said "`causal_conv1d_update` already takes an `activation` argument, so a gating argument is a natural
  extension."** **The premise does not hold.** `activation` is collapsed to a `bool` at `causal_conv1d.py:113`
  before it reaches C++, and the epilogue is a hard-coded SiLU (`causal_conv1d.cu:637`). It cannot carry a tensor;
  a gating argument needs a new tensor parameter and a schema bump. The *fusion itself* is still clean **on the
  decode kernel** (§7) — but it is **layout-hostile on the prefill kernel** and must not be attempted there.

---

## 1. Method and reproduction

A full 8B checkpoint was **not** needed. I built op-for-op replicas of `Lfm2MoeShortConv.forward`
(`lfm2_moe.py:321-378`) and `Lfm2MoeDecoderLayer.forward` (`lfm2_moe.py:433`) with random weights, calling the
*real* `sgl_kernel` conv ops and the *real* sglang `RMSNorm`.

```bash
ENVDIR=/home/t-jialianggu/.conda/envs/sglang-dev
export CUDA_HOME=$ENVDIR PATH=$ENVDIR/bin:$PATH \
       HF_HOME=/home/t-jialianggu/work/EndtoEnd-auto-optimization/.hf_cache \
       TRITON_CACHE_DIR=/tmp/lfm_fx_triton_cache CUDA_VISIBLE_DEVICES=4 \
       TORCHINDUCTOR_CACHE_DIR=$PWD/results/lfm_fusion/fx/inductor_cache
cd /home/t-jialianggu/work/EndtoEnd-auto-optimization
python scripts/lfm_fusion/fx_export_graphs.py    # -> results/lfm_fusion/fx/graphs/  (31 files)
python scripts/lfm_fusion/fx_bench_fusions.py    # -> results/lfm_fusion/fx/bench_fusions.json
python scripts/lfm_fusion/fx_verify_fusion.py    # -> results/lfm_fusion/fx/verify_fusion.json  (26/26 PASS)
```

Four capture paths, all six module×mode combinations (`shortconv` × {decode, prefill}; `layer` × {stock, patched} ×
{decode, prefill}):

* `torch.fx.symbolic_trace` — python-level graph.
* `torch._dynamo.optimize(custom_backend)` — dynamo FX graph.
* `aot_module_simplified(..., decompositions=select_decomp_table())` — **aten-level** graph (the one that makes data
  dependencies explicit).
* `torch.compile(backend="inductor")` with the `output_code` / `fusion` / `schedule` artifact loggers captured to
  per-module log files → **Inductor's generated Triton code and its scheduler's fusion decisions**.

Timing: **`bench_graph()`** — capture *N* calls into a CUDA graph, replay, time with CUDA events. Wall-clock timing
is useless below T≈2048 here (Triton's python launch path is ~18 µs, a `torch.compile` guard ~50 µs), and CUDA-graph
replay is also the physically correct model because SGLang decode runs inside a CUDA graph.

Trace attribution: kernels were linked to CPU ops via `External id` and the enclosing CPU-op ancestor chain was
reconstructed by timestamp nesting, which is how the ambiguous "layout_copy" bucket in
`results/lfm_fusion/audit/*/audit.json` was resolved to exact source lines.

### Gotchas (recorded so nobody re-hits them)

* `aot_module_simplified` **must** be given `torch._inductor.decomposition.select_decomp_table()`, else Inductor
  asserts `both a fallback and a decomp for same op: aten.t.default`.
* `torch.fx.symbolic_trace` fails on sglang `RMSNorm.forward_cuda` (`if x.numel() == 0`, `layernorm.py:124`) →
  needs a `Tracer.is_leaf_module` override; and on `causal_conv1d_fn` (`if x.stride(-1) != 1`) → needs
  `torch.fx.wrap` leaf wrappers.
* `torch._inductor.config.trace` exposes no usable public attributes in 2.9. Use
  `torch._logging.set_logs(output_code=True, fusion=True, schedule=True)` + a `logging.FileHandler` on logger
  `"torch._inductor"`, plus `TORCHINDUCTOR_CACHE_DIR` to keep the generated `.py` files.

---

## 2. (c) What Inductor fuses, what it refuses, and why

### 2.1 Dynamo/AOT capture is clean

`torch._dynamo.explain`: **Graph Count 1, Graph Break Count 0** for every configuration (Op Count 8 for ShortConv,
36 for the stock layer, 31 for the patched layer). The `sgl_kernel` custom ops do **not** break the graph. AOT wraps
them as `torch.ops.higher_order.auto_functionalized_v2`. They have **no Meta/fake kernel registered**, which is
harmless here because their schemas declare `-> ()` with every output listed as a mutated input
(`Tensor($0! -> ) x, ...`), so there is nothing to infer.

`aot_shortconv_prefill_0.txt` gives the clean data-dependency chain:

```
t → mm → split → mul → transpose → clone → auto_functionalized_v2(sgl_kernel.causal_conv1d_fwd) → transpose → mul → t → mm
```

### 2.2 The scheduler refuses every fusion

From `results/lfm_fusion/fx/inductor_code/shortconv_prefill.fusion_decisions.txt`:

```
===== attempting fusion (1/10): 5 nodes =====
  ExternKernelSchedulerNode(name='op0')                       # in_proj mm
  SchedulerNode(name='op1'), Pointwise([2048, 2048], origins={causal_conv1d_fwd, clone, permute_1, mul, split})
  ExternKernelSchedulerNode(name='op2')                       # sgl_kernel.causal_conv1d_fwd
  SchedulerNode(name='op3'), Pointwise([2048, 2048], origins={mul_1, split, permute_3})
  ExternKernelSchedulerNode(name='op4')                       # out_proj mm
cannot fuse op1 with op3: no shared data
found 0 possible fusions
completed fusion round (1/10): fused 5 nodes into 5 nodes
```

Decode is identical with one extra node (`shortconv_decode.fusion_decisions.txt`): 6 nodes,
`cannot fuse op1 with op4: no shared data`, `found 0 possible fusions`.

**Three separate blockers, all confirmed:**

1. **`causal_conv1d_fwd` / `causal_conv1d_update` are `ExternKernelSchedulerNode`s.** Inductor never fuses a
   Pointwise node into an extern kernel — the C++ body is opaque. **The conv is a hard fusion barrier.**
   *Implication (this is the central structural fact): fusion can only happen **on each side** of the conv, or
   **inside** a hand-written / modified conv kernel.* Same for `flashinfer` `rmsnorm` / `fused_add_rmsnorm` in the
   full layer.
2. **op1 and op3 have no shared data.** `op1` reads `proj[:, 0:H]` and `proj[:, 2H:3H]`; `op3` reads
   `proj[:, H:2H]` and `conv_out`. Disjoint slices of the same buffer are, correctly, not "shared data" — and even
   if they were, blocker (1) makes them non-adjacent in the schedule. **The two gate multiplies can never be one
   kernel.** This is a genuine, permanent negative result.
3. Inductor still does the *intra-node* work correctly — see §5.

### 2.3 What Inductor does get right

* **`triton_poi_fused_causal_conv1d_fwd_clone_mul_split_transpose_0`** — it independently discovers exactly the
  hypothesised chunk+mul+transpose fusion (§5).
* **Stock decoder layer:** Inductor lowers `out_proj(...) + residual` into
  `extern_kernels.addmm(residual, ..., beta=1)` — the residual add becomes a **free cuBLAS epilogue**. In the
  *patched* layer the same add lives inside `fused_add_rmsnorm`. **Both are already optimal; there is nothing left
  on the residual path.** (`layer_stock_prefill.output_code.py` vs `layer_patched_prefill.output_code.py`.)
* **Decode ShortConv** lowers to 3 Triton pointwise + 2 `mm` + 1 extern conv, including a *standalone* kernel
  `triton_poi_fused__to_copy_as_strided_causal_conv1d_update_1` purely for the int32 cast of
  `req_pool_indices` — Inductor agrees this is a wasted kernel launch (see F4).

---

## 3. Root cause: **two** distinct effects, only one of which was previously identified

The prior campaign attributed the ShortConv glue cost to uncoalesced access
(`docs/2026-07-27/lfm_fusion_results.md:185-188`: *"The defect is not the amount of traffic, it is that the traffic is
uncoalesced. `Bx.transpose(0,1).contiguous()` and the transposed read inside `C_gate * conv_out` both walk memory
with a stride"*; also `lf_triton_shortconv.py:18-26`). That is correct for those two ops. It is **not** the
whole story: the third glue op, `B_gate * x`, is **perfectly coalesced** (both operands have innermost stride 1)
and is still far from peak.

Direct measurement (`scripts/lfm_fusion/fx_bench_fusions.py::bench_graph`, T=16000, H=2048, bf16, GPU 4;
`results/lfm_fusion/fx/bench_elementwise_paths.json`). Every row moves 3·T·H·2 = **196.61 MB** except the last,
which moves 2·T·H·2 = 131.07 MB:

| op | operand layout | µs | GB/s | % of 4224 GB/s achievable | vs contiguous |
|---|---|---|---|---|---|
| `a + b` (baseline) | both contiguous | 46.44 | 4234 | **100 %** | 1.00× |
| `a * b` (baseline) | both contiguous | 46.45 | 4233 | 100 % | 1.00× |
| **`B_gate * x`** | two `[T,H]` views of `[T,3H]`, innermost stride 1 | **86.73** | 2267 | **54 %** | **1.87× slower** |
| **`C_gate * conv_out.T`** | strided view × **transposed** view | **225.01** | 874 | **21 %** | **4.84× slower** |
| **`Bx.transpose(0,1).contiguous()`** | contiguous → transposed store | **222.84** | **588** | **14 %** | — |

*Validation against the real model:* the trace attributes `3974.0/18 = 220.8 µs` per layer to the transpose copy
(measured here: **222.84**, 0.9 % agreement) and `5736.1/18 = 318.7 µs` to the two muls combined (measured here:
`86.73 + 225.01 = 311.74`, 2.2 % agreement). The whole glue runs at **980 GB/s = 23 % of achievable**, matching the
prior campaign's "~0.83 TB/s, ~17 % of peak" headline.

**Two mechanisms, not one:**

1. **Uncoalesced** — the transpose copy (588 GB/s) and the transposed read in `C_gate * conv_out` (873 GB/s).
   Fixed by tiling. *This is what the prior work found.*
2. **Non-vectorizable** — `B_gate * x` at 2267 GB/s. Nothing is uncoalesced here; the operands simply have
   row-stride `3H`, so `TensorIterator` cannot collapse them to a 1-D contiguous problem, `can_vectorize_up_to`
   returns 1, and PyTorch dispatches the scalar generic kernel instead of the 8-wide vectorized one. **This is new.**

The kernel names in the profiler traces prove mechanism 2 directly:

| regime (audit) | total kernel time | gate muls (n=36) | CUDA template selected |
|---|---|---|---|
| C long prefill, 4×4000 = 16000 tok | 157029.2 µs | **5736.1 µs (3.65 %)** | `elementwise_kernel<128,4,gpu_kernel_impl_nocast<BinaryFunctor<bf16..>>>` — **scalar** |
| B concurrent decode, batch 32 | 4600.0 µs | **88.5 µs (1.92 %)** | `elementwise_kernel<128,4,gpu_kernel_impl_nocast<...>>` — **scalar** |
| A low-batch decode, batch 1 | 1990.6 µs | 35.9 µs (1.80 %) | `vectorized_elementwise_kernel<8, BinaryFunctor<bf16..>>` — **vectorized** |

and in the *same* prefill trace the residual adds — identical shape, identical 3-pass traffic, contiguous operands —
select `vectorized_elementwise_kernel<8, CUDAFunctor_add<bf16>>` and cost **45.8 µs per call** against the muls'
**159.3 µs** average.

At **batch 1** the tensor is `[1, 2048]`, the row stride is irrelevant, the problem collapses to contiguous, and the
vectorized kernel *is* selected — which is exactly why regime A shows a different kernel name from regimes B and C.

**Two consequences that matter for what to do next:**

* Most of F1's measured 6.16× is mechanism 1 + 2 together, **not** the 1.67× traffic reduction. Do not describe it
  as a pure fusion win.
* **Batched decode (regime B) suffers mechanism 2 exactly as prefill does** (`[32,2048]` views of `[32,6144]` are
  strided), yet the shipped `conv` patch routes all decode to the stock path. See N3 and F3.

The transposed copy tells the same story on the layout side: `Bx.transpose(0,1).contiguous()` appears as
`elementwise_kernel<128,4,gpu_kernel_impl_nocast<direct_copy_kernel_cuda...>>`, n=30 in prefill (**18** ShortConv
transposes + **12** attention q/k reshapes), 4581.7 µs total, of which the ShortConv share is **3974.0 µs (2.53 %)**
by ancestor-chain attribution (`aten::contiguous < aten::clone < aten::copy_`, n=18).

---

## 4. (a) Maximal fusable chains

Notation: `T` = tokens, `H = 2048`, bf16 = 2 B, so `T·H·2 = 4096·T` bytes. At T=16000 one `[T,H]` bf16 tensor is
**65.54 MB**.

### F1 — prefill input side: `chunk` + `B_gate*x` + `transpose` + `contiguous` → 1 kernel  ✅ **SHIPPED**

*Already implemented as `lf_triton_shortconv.fused_gate_transpose`, called from
`lfm_fusion_patch.py::_patched_shortconv_forward`. Re-derived here from Inductor and re-measured independently.*

| | ops | reads | writes | HBM | layout change |
|---|---|---|---|---|---|
| stock | `split`, `mul`, `permute`, `clone` | `proj[:,0:H]`, `proj[:,2H:3H]`, then `Bx` | `Bx`, then `Bx_t` | `2·T·H·2 + T·H·2` + `T·H·2 + T·H·2` = **5·T·H·2** | produces `[H,T]` |
| fused | one Triton kernel | `proj[:,0:H]`, `proj[:,2H:3H]` | `Bx_t` `[H,T]` | **3·T·H·2** | produces `[H,T]` |

Traffic ratio **1.667×** (327.68 → 196.61 MB at T=16000). Measured (CUDA-graph GPU time, GPU 4):

| T | eager | hand Triton | Inductor | speed-up | eager BW | Triton BW |
|---|---|---|---|---|---|---|
| 512 | 11.5 µs | 2.3 | 2.1 | 4.99× | 912 GB/s | 2732 GB/s |
| 2048 | 38.1 | 4.6 | 4.6 | 8.23× | 1100 | *(L2-resident)* |
| 4096 | 75.7 | 12.0 | 13.5 | 6.30× | 1108 | 4190 |
| 8192 | 149.1 | 24.5 | 27.3 | 6.08× | 1126 | **4105 (97 % of 4224)** |
| 16000 | 307.6 | 49.9 | 52.0 | **6.16×** | 1065 (25 %) | **3938 (93 %)** |

*(T ≤ 4096 has a working set ≤ 60 MB and is partly L2-resident on H200; T ≥ 8192 are the honest HBM numbers.)*
Correctness: **relerr 0.0** (bit-exact) at every T. `6.16× ≫ 1.67×` — see §3 for why.

### F2 — prefill output side: `chunk` + `C_gate * conv_out` + un-transpose → 1 kernel  ✅ **SHIPPED**

*Already implemented as `lf_triton_shortconv.fused_transpose_gate`, called from the same function.*

Traffic is **3·T·H·2 both before and after** (196.61 MB @ T=16000). This op is *already* a single kernel in eager.
**There is no fusion available here and no traffic to remove**; the win is purely escaping the non-vectorized
strided TensorIterator path. Reported honestly as a **retile, not a fusion**:

| T | eager | hand-tiled Triton | Inductor | speed-up |
|---|---|---|---|---|
| 4096 | 63.5 µs | 11.9 | 11.7 | 5.32× |
| 8192 | 122.4 | 24.4 | 25.2 | 5.01× |
| 16000 | **227.0** (866 GB/s, 21 %) | **50.4** (3898 GB/s, 92 %) | 48.7 | **4.50×** |

relerr 0.0. Inductor's own version (`triton_poi_fused_mul_split_transpose_1`, a Grid1D kernel with a
stride-2048 load carrying `eviction_policy='evict_last'`) matches the hand-tiled kernel — the strided read is
absorbed by L2 because 64 consecutive output blocks reuse each 128 B line.

Value: 18 × (227.0 − 50.4) = 3179 µs = **2.02 %** of prefill kernel time bottom-up, or 18 × (1153.2 − 999.4) =
2768 µs = **1.76 %** module-level ⇒ **1.8 – 2.0 %**.

### F1+F2 combined — whole ShortConv prefill module (both shipped)

| T | stock | +F1 | +F1+F2 | saved/layer | ×18 layers | % of 157029 µs prefill kernel time |
|---|---|---|---|---|---|---|
| 4096 | 345.0 µs | 285.7 | 247.6 | 97.4 µs | 1753 µs | 1.12 % |
| 8192 | 693.1 | 578.1 | 501.5 | 191.5 | 3448 | 2.19 % |
| 16000 | **1369.8** | 1153.2 | **999.4** | **370.4** | **6667** | **4.24 %** |

Bottom-up cross-check from the trace: `5736.1 (muls) + 3974.0 (transpose copies) = 9710.1 µs` replaced by
`18 × 100.3 = 1805 µs` ⇒ saving **7905 µs = 5.03 %**. **Report the range 4.2 – 5.0 % of prefill kernel time.**

*Harness validation:* audit says the ShortConv glue costs `9710.1/18 = 539.4 µs` per layer at T=16000; my
microbenchmarks independently give `307.6 + 227.0 = 534.6 µs`. **0.9 % agreement** against the real model trace.

### F3 — decode input side: `chunk` + `B_gate*x` → 1 Triton kernel  ⬜ **OPEN**

Traffic is `3·B·H·2` both ways (12.3 kB at B=1), so this is **not** a traffic play — it is mechanism 2 from §3.
At HEAD the `conv` patch deliberately routes all decode to the stock ops (`_patched_shortconv_forward`, comment:
*"T is the batch size, so the tiled kernel would only add overhead"*). That reasoning is right for a **tiled 64×64**
kernel, but a plain 1-D gate kernel — `lf_triton_shortconv.fused_gate_mul`, which already exists at
`lf_triton_shortconv.py:179` and is currently only exercised by the benchmark — is not tiled.

Measured (`bench_graph`): eager **2.14 µs** vs Triton **1.14 µs** at B=8, and 2.36 vs 1.30 at B=128. At B=1 they
tie at ~1.1 µs (the launch floor) because eager gets the vectorized path there (§3). So the win exists for
**B ≥ 4 only** — i.e. exactly regime B, where the audit shows 88.5 µs (1.92 %) on the scalar path.
Strictly dominated by F5, which removes the kernel entirely rather than making it faster; F3 is the
no-CUDA-rebuild fallback.

### F4 — hoist `req_pool_indices.to(torch.int32)` out of the per-layer loop  ⬜ **OPEN**

`lfm2_moe.py:344` (decode) and `:359/:363` (prefill) re-cast the *same* `[B]` tensor once **per conv layer**.
`ForwardBatch.req_pool_indices` is int64 — allocated as
`torch.tensor(req_pool_indices, dtype=torch.int64).to(device)` at `mem_cache/common.py:354-355` and documented at
`schedule_batch.py:1231` (`# shape: [b], int64`); the cast is real and
fires 18× per forward. It exists because `causal_conv1d.cu:270` does
`TORCH_CHECK(conv_state_indices.scalar_type() == torch::kInt32)`.

Trace evidence — `unrolled_elementwise_kernel<direct_copy_kernel_cuda>`, n=21 (18 conv layers + 3 elsewhere):

| regime | total | % of kernel time | per call |
|---|---|---|---|
| A low-batch decode | 25.2 µs | **1.27 %** | 1.20 µs |
| B concurrent decode | 26.2 µs | **0.57 %** | 1.25 µs |
| C prefill | 23.3 µs | 0.01 % | 1.17 µs |

**Reading 8 bytes and writing 4 costs 1.2 µs — it is 100 % launch overhead.** Confirmed independently:
`int32_cast_gpu_us` = 1.14–1.21 µs across B ∈ [1,128], flat. Ancestor-chain attribution proves **there is no layout
copy at all on the decode path** — the entire decode "layout_copy" bucket is this cast.
Fix: compute it once per forward (cache on `ForwardBatch`, or keep `req_pool_indices` as int32). ~**1.1–1.3 % of
decode kernel time**, bit-exact, no new kernel.

### F5 — push **both** gates into `causal_conv1d_update` (decode only)  ⬜ **OPEN**

Removes 2 kernel launches per conv layer. See §7 for the feasibility analysis. Worth the full 35.9 µs (1.80 %) in
regime A and 88.5 µs (1.92 %) in regime B. Requires a C++/CUDA change in `sgl-kernel`.

**Combined decode ceiling (F4+F5): 61.1 µs of 1990.6 = 3.07 % (regime A); 114.7 µs of 4600.0 = 2.49 % (regime B).**
Measured corroboration at module level (F4 + F3 only, i.e. *not* the full F5): 18 × (19.24 − 16.91) = 41.8 µs =
**2.10 %**.

### F6 — fuse the gate+transpose into the `in_proj` GEMM epilogue — ❌ **EVALUATED AND REJECTED**

Stock: GEMM stores `proj` (3·T·H·2 = 196.61 MB), F1 then reads 131.07 and writes 65.54 MB.
Epilogue-fused: GEMM stores only `Bx_t` (65.54) + `C_gate` (65.54) = 131.07 MB and F1 disappears.
Saving = 262.14 MB/layer @ T=16000 ⇒ 62.1 µs at 4224 GB/s.

But from the trace the `in_proj` GEMM is `nvjet_tst_192x192_64x4_2x1_v_bz_coopB_TNN`, n=18, 9516.5 µs ⇒
**528.7 µs/layer** *(identification: n=18 = the conv-layer count, and `2·16000·2048·6144 = 4.03e11` FLOP / 528.7 µs
= **761 TFLOP/s**, 77 % of H200 bf16 dense peak — consistent; if it were `out_proj` the implied rate would be an
implausible 254 TFLOP/s)*, and its 196.61 MB store is only 46.5 µs — **8.8 % of the GEMM, fully overlapped with
compute**.
So the real saving is just the elimination of the 49.9 µs F1 kernel, while a Triton GEMM that lands the typical
5–15 % behind nvjet on this shape costs **+26 to +79 µs**. Net **−29 to +24 µs**: **break-even at best, for a
complete replacement of the cuBLAS path.** Rejected.

---

## 5. (b) Verdict on the Bx-transpose-fusion hypothesis

> *"A single Triton kernel can read `proj` (`[T, 3H]`) and write `Bx` directly in `[H, T]` layout, fusing the chunk
> + gating multiply + transpose, thereby eliminating one full elementwise pass AND one full copy pass."*

### **VERDICT: CONFIRMED — Inductor derives it automatically without being asked, and this repo already ships it.**

Three independent confirmations: (i) Inductor's own generated kernel below; (ii) a hand-written Triton kernel
benchmarked here at **6.16×** with **relerr 0.0** on the real varlen path; (iii) `lf_triton_shortconv.fused_gate_transpose`,
written before this study, already A/B'd at **+2.33 % req/s e2e on long prefill**. The hypothesis is not merely
plausible — it is the design a compiler picks unprompted, and it is already in production behind
`LFM_FUSION_PATCH=conv`.

`results/lfm_fusion/fx/inductor_code/shortconv_prefill.output_code.py`, generated by Inductor from the unmodified
module (`TileHint.SQUARE`, `grid_type: 'Grid2D'`, `num_load: 2`):

```python
# Topologically Sorted Source Nodes: [chunk, Bx, transpose, Bx_t, causal_conv1d_fwd]
# Original ATen: [aten.split, aten.mul, aten.transpose, aten.clone, sgl_kernel.causal_conv1d_fwd]
@triton.jit
def triton_poi_fused_causal_conv1d_fwd_clone_mul_split_transpose_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK, XBLOCK):
    ...
    tmp0 = tl.load(in_ptr0 + (x1 + 6144*y0), xmask).to(tl.float32)   # B_gate = proj[:, 0:H]
    tmp1 = tl.load(in_ptr0 + (4096 + x1 + 6144*y0), xmask).to(tl.float32)  # x = proj[:, 2H:3H]
    tmp2 = tmp0 * tmp1                                                # the gate multiply
    tl.store(out_ptr0 + (y0 + 2048*x1), tmp2, xmask)                  # TRANSPOSED store into [H, T]
```

Two loads, zero intermediate stores, one transposed store. Exactly the hypothesised kernel. It eliminates
**one full elementwise pass and one full copy pass**, precisely as stated: 5·T·H·2 → 3·T·H·2.

### Constraint checks (all verified, not assumed)

* **`causal_conv1d_fn` requires `x.stride(-1) == 1`** — `causal_conv1d.py:59-60`: `if x.stride(-1) != 1: x = x.contiguous()`.
  A freshly allocated contiguous `[H, T]` buffer has `strides = (T, 1)`. **Asserted at runtime in
  `fx_verify_fusion.py`**: `shape=(2048, 777) strides=(777, 1)` ✓.
* **The `.contiguous()` in the model at `lfm2_moe.py:348` is redundant** with the wrapper's own contiguity fixup —
  removing it alone changes nothing (the wrapper would just do the same copy). The gain requires actually
  *producing* `[H,T]` in one pass.
* **`causal_conv1d_fwd` mutates `x` in place and returns it** (`causal_conv1d.py:63-74` returns `x`; `at::Tensor out = x;` at `causal_conv1d.cu:167` (fwd) and `:243` (update)). The fused kernel must therefore hand it a buffer it owns —
  it does.
* **Bit-exactness on the real varlen path**: `fx_verify_fusion.py` ran the *actual* `causal_conv1d_fn` with
  multi-sequence `query_start_loc`, scattered `cache_indices`, and non-zero initial `conv_states`, for
  `seqlens ∈ {[4000]*4, [1,137,999,2048,63], [3], [7,9]}` and decode `B ∈ {1,4,8,33,128}` with a random permutation
  of 256 state slots. **26/26 checks pass, `torch.equal` on both the output and the mutated `conv_state`.**

### Honest sizing

* Traffic reduction: **1.67×**. Measured speed-up: **6.16×**. The extra 3.7× comes from the two layout effects in
  §3 — the 588 GB/s transposed copy (uncoalesced) and the 2267 GB/s gate multiply (non-vectorizable) — *not* from
  fusion. Do not sell this as a pure fusion win.
* Value: F1 alone = 18 × (307.6 − 49.9) = 4639 µs = **2.95 %** of prefill kernel time bottom-up, or 18 × (1369.8 − 1153.2) = 3899 µs = **2.48 %** module-level ⇒ **2.5 – 3.0 %**; with F2, **4.2 – 5.0 %**.

### Output side: can `C_gate * conv_out` go into the conv epilogue or `out_proj`'s prologue?

* **Into the conv epilogue: NO on prefill.** `causal_conv1d_fwd_kernel` (`causal_conv1d.cu:318`) uses
  `blockIdx.y = channel_id` and cub `BlockLoad`/`BlockStore` with `kNElts = 8` **along seqlen** — this is precisely
  what forces the `[dim, seqlen]` contiguous layout. A thread holding 8 consecutive `t` for one channel would need
  `C_gate[t, H+channel_id]` for 8 values of `t` at stride `3H` ⇒ **8 fully uncoalesced loads per thread**.
  Layout-hostile; the separate tiled Triton kernel (F2) is strictly better. **On decode it is trivially possible** —
  see §7.
* **Into `out_proj`'s prologue: NO, in practice.** `out_proj` is a cuBLAS/nvjet GEMM; a prologue would mean replacing
  it with a Triton GEMM, which is the F6 trade, and F6 is break-even at best.
* Therefore F2 stays a standalone kernel. Its value is real (4.50×) but it is a **retile, not a fusion**.

---

## 6. (d) Identity of `triton_poi_fused_copy__mul_sum_0`

**It is Inductor's compilation of `moe_sum_reduce_torch_compile`**, at
`/home/t-jialianggu/work/sglang/python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py:291-294`:

```python
@torch.compile
def moe_sum_reduce_torch_compile(x, out, routed_scaling_factor):
    torch.sum(x, dim=1, out=out)
    out.mul_(routed_scaling_factor)
```

* Name decoding: `sum` ← `torch.sum`; `mul` ← `out.mul_(routed_scaling_factor)`; **`copy_` ← functionalization of
  the `out=` argument** (this is what made the name look mysterious). It is `poi` (**po**intwise, not a reduction)
  because the reduced dimension is only `top_k = 4`, so Inductor fully unrolls it.
* Dispatch site: `fused_moe.py:609-632`. On CUDA the `topk == 1` and `topk == 2` fast paths (lines 610, 612) are
  skipped because this model has `top_k = 4`, so control reaches `if tokens_in_chunk <= 32:` at **line 620** ⇒
  `moe_sum_reduce_torch_compile` (line 621); otherwise the CUDA `moe_sum_reduce` (line 627). That `<= 32` threshold
  explains the audit split exactly: **decode uses
  `triton_poi_fused_copy__mul_sum_0` (n=22); prefill uses the CUDA `moe_sum_reduce_warp_per_token_vec_kernel`
  (n=22, 1654.1 µs)**.
* **n = 22 = the number of MoE layers** (24 − `num_dense_layers=2`). ✓
* It is **NOT** the topk / expert-bias path. That path is a *separate*, already-fused CUDA kernel, present in
  the same trace as `void topkGatingSigmoid<__nv_bfloat16, 8, 32, 4, 16>(...)`, n=22, 45.6 µs (2.29 % of regime-A
  decode kernel time) — the `<..., 32, 4, ...>` template arguments are E=32 and top_k=4, and the `bool const*`
  second parameter is the `use_expert_bias` correction tensor.

**Verdict: already optimal; not a fusion gap.** It reads `T·top_k·H` and writes `T·H` — the information-theoretic
minimum for a top-k reduction. Measured head-to-head against SGLang's own hand-written CUDA `moe_sum_reduce`:

| T | compiled (µs, GPU) | SGLang CUDA `moe_sum_reduce` (µs) | traffic |
|---|---|---|---|
| 1 | 1.13 | 1.31 | 20 kB |
| 8 | 1.23 | 1.39 | 164 kB |
| 32 | 1.29 | 1.52 | 655 kB |
| 128 | 2.17 | 1.99 | 2.62 MB |

A wash — both are pure launch overhead at decode sizes (21.4 µs / 22 calls = 0.97 µs each in regime A = **1.08 %**
of decode kernel time; 26.8 µs = 0.58 % in regime B). **Eliminating it requires the MoE down-projection kernel to
perform the top-k reduction in-kernel** (an epilogue change inside `fused_moe_kernel`), not a call-site fusion.

⚠️ **One real, separate problem found:** the *wall* cost of the compiled callable is **48–53 µs per call** (pure
python `torch.compile` guard/dispatch, GPU idle), measured flat across T. Inside a CUDA graph this is free. In any
**non-CUDA-graph** decode path it is 22 × ~50 µs ≈ **1.1 ms of CPU per forward** — which is larger than the entire
1.99 ms of decode *kernel* time. The existing audit was run with `cuda_graph=false`, so its CPU-side numbers are
distorted by this. Not a GPU issue; flagged for the e2e harness.

---

## 7. (5) The decode path and the `activation` argument

**Can a gating multiply be pushed into `causal_conv1d_update` via `activation`? No — not via that argument.**

`causal_conv1d.py:109-113` reduces it to a **boolean** before it ever reaches C++:

```python
if activation not in [None, "silu", "swish"]:
    raise NotImplementedError(...)
activation_val = activation in ["silu", "swish"]
causal_conv1d_update_kernel(x, conv_state, weight, bias, activation_val, cache_seqlens, conv_state_indices, pad_slot_id)
```

and the kernel epilogue (`causal_conv1d.cu:637`; the prefill twin is at `:412`) is a hard-coded SiLU:

```cpp
if (params.silu_activation) { out_val = out_val / (1 + expf(-out_val)); }
out[i * params.out_l_stride] = input_t(out_val);
```

`activation` **cannot carry a tensor**. Fusing the gates needs new tensor arguments and a schema change.

**But the fusion itself is clean on decode — verdict: YES, both gates fit `causal_conv1d_update` with fully
coalesced access and no layout change.** From `causal_conv1d_update_kernel` (`causal_conv1d.cu:558`):

* grid = `(batch, ceil(dim / kNThreads))`; each thread owns one `(batch_id, channel_id)` pair and loops over seqlen.
* Input is read as `x[i * params.x_l_stride]`, and the read value is what gets cached: `conv_state[...] = x_val`.
* `batch_id` and `channel_id` are **exactly** the indices needed to address `proj[t, h]` and `proj[t, H + h]`.
  Consecutive threads have consecutive `channel_id`, and the gate slices have column-stride 1, so consecutive
  threads still touch consecutive addresses ⇒ **coalescing preserved**.
* Caching the *gated* value in `conv_state` is semantically correct because the current code already caches `Bx`
  (the gated value) — the model passes `Bx` as `x`.
* The output gate is a one-line change at the epilogue: `out_val *= float(Cgate[batch_id * 3H + H + channel_id])`.

Roughly a 5-line change per gate, plus schema/wrapper plumbing. It is a `sgl-kernel` C++/CUDA change, not a
call-site edit — hence its ranking below. On **prefill** the same idea fails on layout grounds (§5).

Measured decode module timings (CUDA-graph GPU time, T = decode batch):

| T | stock | + F4 (hoisted cast) | + F3 (Triton gate) | ×18 saving | % of regime-A decode kernel time |
|---|---|---|---|---|---|
| 1 | 18.25 µs | 17.04 | 16.97 | 23.0 µs | 1.15 % |
| 8 | 19.24 | 18.07 | 16.91 | 41.8 | 2.10 % |
| 32 | 20.82 | 19.66 | 18.48 | 42.0 | 2.11 % |
| 128 | 24.31 | 22.71 | 21.54 | 49.7 | 2.49 % |

Everything at these sizes is launch-latency-bound at a ~1.1 µs floor, so **each removed kernel launch is worth
~1.1–1.2 µs per layer**, i.e. ~0.9–1.1 % of decode kernel time per removed launch across 18 layers.

---

## 8. (e) Ranked implementation difficulty

Ranks 1–3 are the **remaining** work as of HEAD (`0574701`); the two prefill chains are already in the tree and are
listed at the bottom for completeness.

> ⚠️ **Concurrency note.** While this report was being written, `scripts/lfm_fusion/lfm_fusion_patch.py` acquired
> uncommitted working-tree changes (mtime 21:32, after this study's measurements) that implement **exactly F3 and
> F4** — a `gate` arm calling `K.fused_gate_mul(proj, H)` in the decode branch, and a `_cached_int32_indices(...)`
> helper replacing the three per-layer `req_pool_indices.to(torch.int32)` calls. Those edits were made by a
> concurrent process, not by this study, and are **not measured here**. Line-number references into that file are
> therefore given by symbol name. If those changes are the ones that land, ranks 1 and 2 below are already done and
> only need an `lf_e2e.py` A/B.

| Rank | Candidate | Where | Expected kernel-time saving | Risk | Effort |
|---|---|---|---|---|---|
| **1** | **F4** — hoist `req_pool_indices.to(torch.int32)` to once per forward | call site only: `lfm2_moe.py:344/359/363`, and the three `req_pool_indices.to(torch.int32)` sites in `lfm_fusion_patch.py` (cache on `ForwardBatch`) | **1.27 % decode-A, 0.57 % decode-B**, ~0 prefill | **none** — bit-exact by construction, no new kernel | ~3 lines. **Do this first.** Optionally a one-word `int64`→`int32` at `mem_cache/common.py:354` removes it at source (needs an audit of all consumers). |
| **2** | **F3** — route decode through the existing `fused_gate_mul` for `B ≥ 4` | the `is_decode()` branch of `_patched_shortconv_forward`; kernel already exists at `lf_triton_shortconv.py:179` | **~0.9 % decode-B** (half of the 1.92 % scalar-path cost) | low — bit-exact; needs the `B ≥ 4` guard or it regresses batch-1 | ~5 lines. The kernel is already written and correctness-gated. |
| **3** | **F5** — push both gates into `causal_conv1d_update` | **`sgl-kernel` C++/CUDA + op schema + python wrapper** | **1.80 % decode-A, 1.92 % decode-B** (supersedes F3) | medium — kernel + schema change and an `sgl-kernel` rebuild; must preserve `conv_state` semantics (it already caches the *gated* value, so this is safe) | ~2×5 lines of CUDA plus plumbing. Decode only — **must not** be attempted on `causal_conv1d_fwd` (§5). |
| **4** | **N5** — lower/remove `CONV_FUSION_MIN_TOKENS` by making the launch cheaper (CUDA-graph or C++ launcher for the prefill path) | `CONV_FUSION_MIN_TOKENS` and its guard in `lfm_fusion_patch.py` | up to **4.99×** on the input side at T=512, currently forfeited | medium — the guard exists for a real CPU-side reason; removing it without fixing the launch cost would regress | unknown; needs a prefill-side launch experiment |
| **5** | MoE top-k reduction as a `fused_moe_kernel` epilogue (removes `triton_poi_fused_copy__mul_sum_0` **and** `moe_sum_reduce`) | `fused_moe_triton` | 1.08 % decode-A / 1.05 % prefill | medium-high — touches the kernel that is 67 % of prefill time; any regression there dwarfs the gain | large |
| **6** | **F6** — gate+transpose in the `in_proj` GEMM epilogue | replace cuBLAS/nvjet with a Triton GEMM | **−29 to +24 µs/layer: break-even at best** | high | **Do not do.** Quantified and rejected in §4. |
| — | Hoist the per-layer `query_start_loc` construction (`lfm2_moe.py:356-362`; `aten::new_empty` n=18 and 41 pageable HtoD memcpys per prefill forward, 35.1 µs GPU but a possible CPU-side stall) | call site | ~0 GPU; CPU-side only | none | small; a CPU-latency item, not a kernel item |
| ✅ | **F1 + F2** — already shipped as `LFM_FUSION_PATCH=conv` | `lf_triton_shortconv.py`, `lfm_fusion_patch.py::_patched_shortconv_forward` | 4.2–5.0 % of prefill kernel time → **+2.33 % e2e** (measured, `docs/2026-07-27/lfm_fusion_results.md` §6) | — | done |

**Remaining upside if 1+2 are done (no `sgl-kernel` rebuild): ~1.3 % of regime-A and ~1.5 % of regime-B decode
kernel time.** Substituting 3 for 2 raises regime A to ~3.1 % and regime B to ~2.5 %. Prefill is already harvested.

---

## 9. Scope, caveats, and what was *not* shown

* **Kernel time ≠ end-to-end gain, and the conversion ratio is not 1:1 in either direction.** Every percentage
  above is a fraction of *summed CUDA kernel duration*. Three calibration points exist, all from this repo's own
  A/B runs (`docs/2026-07-27/lfm_fusion_results.md` §6) combined with the kernel-time shares I attributed from the traces:

  | arm | regime | kernel time removed | measured e2e | ratio |
  |---|---|---|---|---|
  | `norm+scale` | A low-batch decode | 48 adds (51.8 µs) + 22 muls (21.7 µs) = **3.69 %** | **+4.20 %** | **1.14 : 1** |
  | `norm+scale` | B concurrent decode | 48 adds (63.6 µs) + 22 muls (24.2 µs) = **1.91 %** | **+3.68 %** | **1.93 : 1** |
  | `conv` (= F1+F2) | C long prefill | **4.2 – 5.0 %** | **+2.33 %** | **≈ 0.5 : 1** |

  **On decode the e2e gain *exceeds* the kernel-time share** — because removing a launch also removes its CPU-side
  dispatch, and decode is frequently CPU-bound (the audit ran with `cuda_graph=false`). **On prefill it is roughly
  half**, because the GPU is saturated and the removed work partly overlapped. Projecting F3+F4+F5 at 2.5–3.1 % of
  decode kernel time gives a very wide **≈ +2.8 to +6 % e2e on regime A** at the 1.14–1.93 ratios — which is
  precisely why it must be measured, not asserted. Treat the ratio table as the calibration, not the projection.
* **No end-to-end serving A/B was run in *this* study.** All numbers here are module-level microbenchmarks or
  trace attribution against the *existing* audit. F1/F2 do have a prior e2e A/B (+2.33 % prefill,
  `docs/2026-07-27/lfm_fusion_results.md` §6); **F3, F4 and F5 have none** and must be A/B'd via `scripts/lfm_fusion/lf_e2e.py`
  before any claim is made for them.
* **F2 is not a fusion.** It has identical HBM traffic before and after (3·T·H·2 either way). Calling it a fusion
  win would be wrong; it is a retile of an op that was already a single kernel.
* **F1's 6.16× is not 6.16× of fusion.** Traffic only drops 1.67×; the rest is vectorization/coalescing (§3).
* **T ≤ 4096 microbenchmarks are partly L2-resident** on this H200 (60 MB L2); only T ≥ 8192 gives honest HBM
  numbers. The reported "% of peak" figures reflect this.
* **The 4224 GB/s reference** is a measured large-copy figure on this GPU, not the 4.8 TB/s spec number.
* **`req_pool_indices` dtype — resolved:** the allocation site is
  `sglang/srt/mem_cache/common.py:354-355`,
  `req_pool_indices_cpu = torch.tensor(req_pool_indices, dtype=torch.int64)` → `.to(device, non_blocking=True)`,
  reaching `ScheduleBatch.req_pool_indices` via `schedule_batch.py:1503,1626`. It is unambiguously **int64**
  (`schedule_batch.py:1231` documents it as such; the `torch.empty(0, dtype=torch.int32)` at line 1933 is only the
  empty-batch placeholder). **This makes F4 potentially a one-word deletion rather than a hoist** — flipping line
  354 to `torch.int32` removes the cast from every layer at the source. *Not verified end-to-end:* the tensor is
  used as an index into `req_to_token_pool` and friends in many places, and every one would need checking (torch
  advanced indexing accepts int32, but any `torch.cat` with an int64 tensor, e.g. `schedule_batch.py:2131`, or any
  C++ `TORCH_CHECK` on int64 would break). The safe first step is the per-forward hoist.
* **The audit traces were captured with `cuda_graph=false`**, so their CPU-side timings (notably §6's 50 µs
  `torch.compile` guard) do not reflect production CUDA-graph decode. GPU kernel times are unaffected.
* Nothing under `/home/t-jialianggu/work/sglang/` was modified. Ranks 1 (optional variant), 3 and 5 in §8 would
  require changes there; ranks 1 (default), 2 and 4 are confined to `scripts/lfm_fusion/`.

---

## 10. Artifact index

`results/lfm_fusion/fx/`

| Path | Contents |
|---|---|
| `FINDINGS.md` | this document |
| `graphs/symbolic_*.txt` | `torch.fx.symbolic_trace` graphs (`print_tabular` + `print_readable`) |
| `graphs/dynamo_*_0.txt` | dynamo-captured FX graphs |
| `graphs/aot_*_0.txt` | **aten-level AOT graphs** — the explicit data-dependency view |
| `graphs/explain_*.txt` | `torch._dynamo.explain` (graph count, break count, op count) |
| `graphs/inductor_*.log` | raw Inductor logs: scheduler fusion decisions + full output code |
| `graphs/summary.json` | index of all captures |
| `inductor_code/*.output_code.py` | **extracted Inductor Triton output code**, per module/mode |
| `inductor_code/*.fusion_decisions.txt` | **extracted scheduler fusion log** (`found 0 possible fusions`) |
| `inductor_cache/` | Inductor's generated per-kernel `.py` files |
| `bench_fusions.json` | all benchmark results (A–F), wall + CUDA-graph GPU times, relerr |
| `verify_fusion.json` | 26/26 bit-exactness checks incl. varlen prefill and scattered-slot decode |
| `bench_elementwise_paths.json` | §3 decomposition: each glue pass vs a contiguous baseline of identical traffic |

`scripts/lfm_fusion/`

| Path | Purpose |
|---|---|
| `fx_common.py` | config constants, `torch.fx.wrap` leaf conv wrappers, `ShortConvRepro`, `MoERepro`, input/weight factories |
| `fx_export_graphs.py` | all four capture paths; `LeafNormTracer`, `make_dump_backend`, `capture_inductor_logs`, `DecoderLayerStock/Patched` |
| `fx_bench_fusions.py` | hand-written Triton kernels (`gate_transpose`, `gate_only`, `out_gate_untranspose`) + benchmarks A–F with `bench_graph()` |
| `fx_verify_fusion.py` | bit-exactness verification against the real `sgl_kernel` conv ops on varlen/scattered inputs |
| `fx_bench_elementwise_paths.py` | §3: isolates the coalescing effect from the vectorization effect |

Pre-existing, referenced throughout: `lf_triton_shortconv.py` (the shipped fused kernels), `lfm_fusion_patch.py`
(the `norm` / `scale` / `conv` arms), `lf_e2e.py` (the serving A/B harness), `docs/2026-07-27/lfm_fusion_results.md`.
