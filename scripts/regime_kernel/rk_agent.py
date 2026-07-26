#!/usr/bin/env python3
"""Minimal closed-loop kernel-specialization agent (RQ4).

This is deliberately NOT a wrapper around a parameter sweep. The controller owns
the decisions:

  1. it reads a measured workload/profiler summary,
  2. classifies the bottleneck from structured rules,
  3. chooses an ACTION CLASS based on that diagnosis,
  4. generates only the candidates that action implies,
  5. gates every candidate on correctness,
  6. benchmarks the survivors,
  7. accepts, rejects or rolls back against the incumbent,
  8. records the reason for every decision and stops when the budget is spent.

The search space it explores at each step is a consequence of the diagnosis, so
two different bottlenecks lead to two different sets of trials.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

HERE = Path(__file__).resolve().parent
PY = f"{L.ENVDIR}/bin/python"


# --------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------
@dataclass
class Diagnosis:
    bottleneck: str
    evidence: dict
    rationale: str


def diagnose(model: str, tokens: int, sweep: dict) -> Diagnosis:
    """Classify the kernel bottleneck from measured quantities.

    The signals are deliberately cheap and derived from data we already collect,
    so the loop does not depend on a full NCU pass at every iteration.
    """
    shape = L.MODELS[model]
    topk = shape["top_k"]
    M = tokens * topk
    N, K = shape["moe_intermediate_size"], shape["hidden_size"]
    base = sweep["default_baseline"]["median_ms"]
    results = sweep.get("results", [])
    spread = (max(r["median_ms"] for r in results) /
              min(r["median_ms"] for r in results)) if results else 1.0
    rs = sweep.get("routing_stats", {})
    cv = rs.get("cv_expert_load", 0.0)
    active = rs.get("active_experts", 0)
    E = shape["num_experts"]

    # arithmetic intensity of the grouped GEMM at this M
    flops = 2 * M * N * K * 2          # two projections
    bytes_w = E * (2 * N * K + K * N) * 2   # bf16 weights that must stream
    intensity = flops / max(bytes_w, 1)

    ev = dict(M=M, tokens=tokens, top_k=topk, default_ms=base,
              config_spread=round(spread, 3), arithmetic_intensity=round(intensity, 4),
              cv_expert_load=round(cv, 3), active_experts=active, num_experts=E)

    if M <= 16:
        return Diagnosis(
            "low_occupancy_launch_bound", ev,
            f"M={M} cannot fill the GPU: the grouped GEMM has far fewer rows "
            f"than a single tile, so latency is dominated by launch and weight "
            f"streaming rather than math (arithmetic intensity {intensity:.3f}).")
    if intensity < 1.0:
        return Diagnosis(
            "memory_bound_weight_streaming", ev,
            f"arithmetic intensity {intensity:.3f} FLOP/byte: every expert's "
            f"weights must be read for only {M} rows, so the kernel is bound by "
            f"weight movement.")
    if cv > 0.6:
        return Diagnosis(
            "routing_imbalance", ev,
            f"expert-load CV {cv:.2f} with {active}/{E} experts active: some "
            f"experts receive far more tokens, so tiles are unevenly filled.")
    return Diagnosis(
        "compute_bound", ev,
        f"M={M} with arithmetic intensity {intensity:.2f}: enough work per byte "
        f"to keep the Tensor Cores busy, so tile size and pipelining dominate.")


# --------------------------------------------------------------------------
# Action selection: the diagnosis decides WHICH candidates to try
# --------------------------------------------------------------------------
def actions_for(d: Diagnosis, model: str, tokens: int) -> list[dict]:
    shape = L.MODELS[model]
    N, K = shape["moe_intermediate_size"], shape["hidden_size"]
    M = tokens * shape["top_k"]
    space = L.build_search_space(M, N, K)

    def sel(pred, name, why, cap=24):
        cands = [c for c in space if pred(c)][:cap]
        return dict(action=name, rationale=why, candidates=cands)

    if d.bottleneck == "low_occupancy_launch_bound":
        return [
            sel(lambda c: c["BLOCK_SIZE_M"] == 16 and c["num_stages"] >= 4,
                "small_tile_deep_pipeline",
                "keep the M tile minimal to avoid wasted rows and raise "
                "num_stages so weight loads overlap the tiny amount of math"),
            sel(lambda c: c["BLOCK_SIZE_M"] == 16 and c["BLOCK_SIZE_K"] >= 128,
                "small_tile_wide_k",
                "widen the K tile so each launch streams more contiguous weight "
                "per unit of index overhead"),
        ]
    if d.bottleneck == "memory_bound_weight_streaming":
        return [
            sel(lambda c: c["BLOCK_SIZE_K"] >= 128 and c["num_stages"] >= 3,
                "increase_k_tile_and_stages",
                "improve weight reuse per load and deepen the software pipeline"),
            sel(lambda c: c["BLOCK_SIZE_N"] >= 128,
                "widen_n_tile",
                "amortise each weight tile over more output columns"),
        ]
    if d.bottleneck == "routing_imbalance":
        return [
            sel(lambda c: c["GROUP_SIZE_M"] > 1,
                "group_m_scheduling",
                "GROUP_SIZE_M changes block scheduling order, which is the knob "
                "that mitigates uneven per-expert tile occupancy"),
            sel(lambda c: c["BLOCK_SIZE_M"] <= 32,
                "smaller_m_tile_for_ragged_groups",
                "smaller M tiles waste fewer rows when expert groups are ragged"),
        ]
    return [
        sel(lambda c: c["BLOCK_SIZE_M"] >= 64 and c["num_warps"] == 8,
            "large_tile_more_warps",
            "at high arithmetic intensity, larger tiles and more warps raise "
            "Tensor-Core utilisation"),
        sel(lambda c: c["BLOCK_SIZE_M"] >= 64 and c["num_stages"] >= 4,
            "large_tile_deep_pipeline",
            "deep pipelining hides global-memory latency behind math"),
    ]


# --------------------------------------------------------------------------
# Loop
# --------------------------------------------------------------------------
def bench(model: str, tokens: int, configs: list[dict], out: Path, gpu: int,
          warmup: int, iters: int, repeats: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    cfgfile = out.with_suffix(".cands.json")
    cfgfile.write_text(json.dumps(configs))
    env = L.run_env(); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [PY, str(HERE / "rk_microbench.py"), "--model", model,
           "--tokens", str(tokens), "--configs", str(cfgfile), "--out", str(out),
           "--warmup", str(warmup), "--iters", str(iters), "--repeats", str(repeats)]
    log = out.with_suffix(".log")
    with open(log, "w") as lf:
        subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not out.exists():
        return dict(results=[], failures=[dict(kind="runtime_failure",
                                               error=log.read_text()[-300:])])
    return json.loads(out.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--tokens", type=int, required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--budget", type=int, default=4, help="max iterations")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--sweep", help="existing sweep json for the diagnosis")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    outdir = Path(a.out) if a.out else (L.RESULTS / "agent" / a.model / f"t{a.tokens}")
    outdir.mkdir(parents=True, exist_ok=True)

    sweep_path = Path(a.sweep) if a.sweep else (
        L.RESULTS / "raw" / "sweep" / a.model / f"full_t{a.tokens}_uniform.json")
    if not sweep_path.exists():
        raise SystemExit(f"need a sweep summary for the diagnosis: {sweep_path}")
    sweep = json.loads(sweep_path.read_text())

    d = diagnose(a.model, a.tokens, sweep)
    print(f"[diagnosis] {d.bottleneck}\n  {d.rationale}")

    incumbent_ms = sweep["default_baseline"]["median_ms"]
    incumbent = dict(name="default", config=None, median_ms=incumbent_ms)
    history, trace = [], []

    plans = actions_for(d, a.model, a.tokens)
    for it, plan in enumerate(plans[:a.budget]):
        t0 = time.time()
        print(f"\n[iter {it}] action={plan['action']} "
              f"({len(plan['candidates'])} candidates)\n  why: {plan['rationale']}")
        res = bench(a.model, a.tokens, plan["candidates"],
                    outdir / f"iter{it}_{plan['action']}.json", a.gpu,
                    a.warmup, a.iters, a.repeats)
        ok = res.get("results", [])
        fails = res.get("failures", [])
        if not ok:
            decision, reason, best = "reject", "no candidate passed correctness/runtime", None
            speedup = float("nan")
        else:
            best = ok[0]
            speedup = incumbent["median_ms"] / best["median_ms"]
            if speedup > 1.01:
                decision = "accept"
                reason = (f"{speedup:.3f}x faster than the incumbent "
                          f"({incumbent['name']}) and correctness passed")
                incumbent = dict(name=best["config_key"],
                                 config={k: best[k] for k in
                                         ("BLOCK_SIZE_M", "BLOCK_SIZE_N",
                                          "BLOCK_SIZE_K", "GROUP_SIZE_M",
                                          "num_warps", "num_stages")},
                                 median_ms=best["median_ms"])
            else:
                decision = "reject"
                reason = (f"best candidate only {speedup:.3f}x vs incumbent; "
                          f"below the 1.01x acceptance threshold")
        rec = dict(iteration=it, action=plan["action"],
                   rationale=plan["rationale"], bottleneck=d.bottleneck,
                   n_candidates=len(plan["candidates"]),
                   n_passed=len(ok), n_failed=len(fails),
                   best_key=best["config_key"] if best else "",
                   best_ms=best["median_ms"] if best else float("nan"),
                   incumbent_ms=incumbent["median_ms"],
                   speedup=speedup, decision=decision, reason=reason,
                   wall_s=round(time.time() - t0, 1))
        trace.append(rec)
        print(f"  -> {decision}: {reason}")
        history.append(dict(plan=plan["action"], failures=fails[:5]))

    final = dict(model=a.model, tokens=a.tokens,
                 M=a.tokens * L.MODELS[a.model]["top_k"],
                 diagnosis=asdict(d),
                 default_ms=sweep["default_baseline"]["median_ms"],
                 selected=incumbent,
                 total_speedup=sweep["default_baseline"]["median_ms"] / incumbent["median_ms"],
                 iterations=trace,
                 rejected=[t for t in trace if t["decision"] != "accept"],
                 environment=L.environment())
    (outdir / "agent_result.json").write_text(json.dumps(final, indent=2))
    print(f"\n[final] {incumbent['name']} @ {incumbent['median_ms']:.4f} ms "
          f"= {final['total_speedup']:.3f}x over default")
    print(f"wrote {outdir/'agent_result.json'}")


if __name__ == "__main__":
    main()
