#!/usr/bin/env python3
"""Fused-MoE kernel microbenchmark + correctness engine.

Runs one (model_shape, token_batch, routing) workload against a set of candidate
kernel configurations, gating every timing on a numerical comparison against the
default kernel path.

Correctness first, always: a configuration that fails the numerical check is
recorded in the failures file and is never emitted as a timing result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L


def build_workload(shape: dict, num_tokens: int, routing: str, torch, gen,
                   with_bias: bool = False):
    """Allocate the MoE inputs for one invocation."""
    E = shape["num_experts"]
    K = shape["hidden_size"]
    N = shape["moe_intermediate_size"]
    topk = shape["top_k"]
    dt = torch.bfloat16

    x = torch.randn(num_tokens, K, dtype=dt, device="cuda", generator=gen) / 10
    # Expert bias: LFM2.5 sets use_expert_bias=true, so the serving path runs the
    # WITH-BIAS kernel variant. Tuning the no-bias variant and deploying the
    # result is a real source of microbenchmark -> end-to-end mismatch.
    # w1 holds the fused gate+up projection => 2N rows
    w1 = torch.randn(E, 2 * N, K, dtype=dt, device="cuda", generator=gen) / 20
    w2 = torch.randn(E, K, N, dtype=dt, device="cuda", generator=gen) / 20
    gating = L.make_gating(num_tokens, E, routing, torch, gen)
    b1 = b2 = None
    if with_bias:
        b1 = torch.randn(E, 2 * N, dtype=dt, device="cuda", generator=gen) / 50
        b2 = torch.randn(E, K, dtype=dt, device="cuda", generator=gen) / 50
    return x, w1, w2, gating, topk, N, K, b1, b2


def make_topk(x, gating, topk):
    from sglang.srt.layers.moe.topk import TopKConfig, select_experts
    return select_experts(x, gating, TopKConfig(top_k=topk, renormalize=True))


def init_sglang_context(model_path: str):
    """Initialize SGLang's global server args.

    `fused_experts` consults `get_global_server_args()` (for the deterministic
    inference flag), which is only populated inside a real server process.
    Upstream's tuner runs under Ray workers that set it up; a standalone
    microbenchmark must set it explicitly or the very first kernel call raises
    "Global server args is not set yet!".
    """
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
    sa = ServerArgs(model_path=model_path)
    set_global_server_args_for_scheduler(sa)

    # main additionally reaches for the tensor-parallel group from inside the
    # MoE path (symmetric-memory allocation check), which a standalone process
    # never creates. Stand up a 1-rank group so the sweep can run outside a
    # server. Harmless on versions that do not need it.
    try:
        from sglang.srt.distributed import parallel_state as _ps

        if getattr(_ps, "_TP", None) is None:
            import os

            port = os.environ.get("RK_TP_PORT", "29591")
            _ps.init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=f"tcp://127.0.0.1:{port}",
                backend="nccl",
            )
            _ps.initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception as e:  # noqa: BLE001 - older layouts do not need this
        print(f"[rk] tp-group init skipped: {type(e).__name__}: {e}")
    return sa


def assert_clean_baseline(E: int, N: int, dtype=None) -> None:
    """Fail unless the un-overridden path really is the heuristic default."""
    try:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_moe_configs,
        )
    except ImportError:
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_config import (
            get_moe_configs,
        )
    cfg = get_moe_configs(E, N, dtype, 0, 0)
    if cfg:
        import triton

        raise SystemExit(
            f"baseline is NOT the heuristic default: a tuned config for "
            f"E={E},N={N} is reachable under triton {triton.__version__} "
            f"(buckets {sorted(cfg)[:6]}...). Every 'speedup over default' "
            f"here would be measured against an already-tuned baseline. "
            f"Point PYTHONPATH at a tree that does not contain the config "
            f"under test, or unset SGLANG_MOE_CONFIG_DIR."
        )
    print("[rk] baseline check: no tuned config reachable -> heuristic default",
          flush=True)


def run_once(fused_moe, override_config, x, w1, w2, topk_output, cfg, torch,
             b1=None, b2=None):
    from sglang.srt.layers.moe.moe_runner import MoeRunnerConfig
    mrc = MoeRunnerConfig(inplace=False)
    ctx = override_config(cfg) if cfg is not None else _null()
    with ctx:
        return fused_moe(x, w1, w2, topk_output, moe_runner_config=mrc,
                         b1=b1, b2=b2)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def time_config(fused_moe, override_config, x, w1, w2, topk_output, cfg, torch,
                warmup: int, iters: int, repeats: int, b1=None, b2=None):
    """CUDA-event timing: `repeats` independent rounds of `iters` iterations."""
    for _ in range(warmup):
        run_once(fused_moe, override_config, x, w1, w2, topk_output, cfg, torch, b1, b2)
    torch.cuda.synchronize()

    per_round = []
    for _ in range(repeats):
        st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            st[i].record()
            run_once(fused_moe, override_config, x, w1, w2, topk_output, cfg, torch, b1, b2)
            en[i].record()
        torch.cuda.synchronize()
        ts = [s.elapsed_time(e) for s, e in zip(st, en)]
        per_round.append(sorted(ts)[len(ts) // 2])   # median within the round
    return per_round


def correctness(ref, out, torch, rtol=2e-2, atol=2e-2):
    """Compare a candidate against the default kernel output.

    BF16 MoE accumulates in fp32 but reduces in bf16; different tile shapes
    change the reduction order, so exact equality is not expected. We require
    agreement within a BF16-appropriate tolerance and no NaN/Inf.
    """
    if out is None:
        return dict(ok=False, reason="no output")
    if torch.isnan(out).any().item() or torch.isinf(out).any().item():
        return dict(ok=False, reason="NaN/Inf in output", max_abs_err=float("nan"))
    diff = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp(min=1e-3)
    max_abs = diff.max().item()
    max_rel = (diff / denom).max().item()
    mean_abs = diff.mean().item()
    ok = bool(max_abs <= atol + rtol * ref.float().abs().max().item())
    return dict(ok=ok, max_abs_err=max_abs, max_rel_err=max_rel,
                mean_abs_err=mean_abs,
                reason="" if ok else f"max_abs_err={max_abs:.4g} exceeds tolerance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--routing", default="uniform", choices=["uniform", "skewed"])
    ap.add_argument("--configs", help="JSON file: list of configs, or a "
                                      "profile map M->config. Default: full sweep")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke)")
    ap.add_argument("--bias", action="store_true",
                    help="pass expert bias, matching models with use_expert_bias")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    shape = L.MODELS[a.model]
    E, K, N = shape["num_experts"], shape["hidden_size"], shape["moe_intermediate_size"]
    topk = shape["top_k"]
    # M is what fused_experts_impl computes: min(num_tokens, CHUNK_SIZE).
    # It is the token count, NOT tokens * top_k.
    M = a.tokens

    if a.configs:
        raw = json.loads(Path(a.configs).read_text())
        cands = raw if isinstance(raw, list) else list(raw.values())
    else:
        cands = L.build_search_space(M, N, K)
    if a.limit:
        cands = cands[:a.limit]

    plan = dict(model=a.model, tokens=a.tokens, top_k=topk, M=M, E=E, N=N, K=K,
                routing=a.routing, with_bias=a.bias, n_candidates=len(cands),
                warmup=a.warmup, iters=a.iters, repeats=a.repeats, out=a.out)
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        est = len(cands) * (a.warmup + a.iters * a.repeats) * 0.00025
        print(f"[dry-run] would benchmark {len(cands)} configs, "
              f"~{est:.0f}s estimated")
        return

    import torch

    # sglang moved the Triton MoE runner between 0.5.12 and main
    # (layers.moe.fused_moe_triton -> layers.moe.moe_runner.triton_utils).
    # Resolve either layout so the same sweep can run on both, which is what
    # lets us re-tune under a newer torch/Triton without a second harness.
    try:
        from sglang.srt.layers.moe.fused_moe_triton import override_config
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_moe
    except ImportError:
        from sglang.srt.layers.moe.moe_runner.triton_utils import override_config
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_moe

    init_sglang_context(shape["path"])
    gen = torch.Generator(device="cuda"); gen.manual_seed(L.SEED)
    x, w1, w2, gating, topk, N, K, b1, b2 = build_workload(
        shape, a.tokens, a.routing, torch, gen, with_bias=a.bias)
    topk_output = make_topk(x, gating, topk)
    routing_stats = L.expert_load_stats(topk_output.topk_ids, E, torch)

    # The baseline must actually be the heuristic default. If a tuned config
    # file is reachable -- e.g. because PYTHONPATH points at a tree that
    # contains the very config being evaluated -- get_moe_configs silently
    # returns it, including through the cross-version fallback, and every
    # "speedup over default" in the run is then a comparison against an
    # already-tuned baseline. That happened once (see
    # docs/2026-07-29/RETRACTION_triton36_baseline_contamination.md) and was
    # only caught days later, so fail loudly instead.
    assert_clean_baseline(E, N)

    # reference = the DEFAULT path, i.e. exactly what the server does today
    ref = run_once(fused_moe, override_config, x, w1, w2, topk_output, None, torch,
                   b1, b2)
    torch.cuda.synchronize()
    base_rounds = time_config(fused_moe, override_config, x, w1, w2, topk_output,
                              None, torch, a.warmup, a.iters, a.repeats, b1, b2)
    base = L.summarize(base_rounds)

    rows, failures = [], []
    t0 = time.time()
    for i, cfg in enumerate(cands):
        try:
            out = run_once(fused_moe, override_config, x, w1, w2, topk_output,
                           cfg, torch, b1, b2)
            torch.cuda.synchronize()
        except Exception as e:
            failures.append(dict(config=cfg, stage="runtime",
                                 kind="OOM" if "out of memory" in str(e).lower()
                                 else "runtime_failure", error=str(e)[:300]))
            continue
        corr = correctness(ref, out, torch)
        if not corr["ok"]:
            failures.append(dict(config=cfg, stage="correctness",
                                 kind="correctness_failure", **corr))
            continue
        try:
            rounds = time_config(fused_moe, override_config, x, w1, w2,
                                 topk_output, cfg, torch, a.warmup, a.iters,
                                 a.repeats, b1, b2)
        except Exception as e:
            failures.append(dict(config=cfg, stage="timing",
                                 kind="runtime_failure", error=str(e)[:300]))
            continue
        s = L.summarize(rounds)
        rows.append(dict(model=a.model, tokens=a.tokens, M=M, routing=a.routing,
                         config_key=L.config_key(cfg), **cfg, **s,
                         rounds_ms=rounds,
                         speedup_vs_default=base["median_ms"] / s["median_ms"],
                         correctness_ok=True,
                         max_abs_err=corr["max_abs_err"],
                         max_rel_err=corr["max_rel_err"]))
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(cands)} ok={len(rows)} fail={len(failures)} "
                  f"{time.time()-t0:.0f}s", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(plan=plan, environment=L.environment(),
                   routing_stats=routing_stats,
                   default_baseline=dict(**base, rounds_ms=base_rounds),
                   results=sorted(rows, key=lambda r: r["median_ms"]),
                   failures=failures,
                   wall_time_s=round(time.time() - t0, 1))
    out.write_text(json.dumps(payload, indent=2))
    best = payload["results"][0] if payload["results"] else None
    print(f"\ndefault median {base['median_ms']:.4f} ms")
    if best:
        print(f"best    median {best['median_ms']:.4f} ms  "
              f"({best['speedup_vs_default']:.3f}x)  {best['config_key']}")
    print(f"ok={len(rows)} failures={len(failures)} -> {out}")


if __name__ == "__main__":
    main()
