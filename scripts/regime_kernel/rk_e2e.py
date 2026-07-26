#!/usr/bin/env python3
"""End-to-end stage: swap MoE kernel profiles with serving knobs frozen.

The only thing that changes between arms is `SGLANG_MOE_CONFIG_DIR`, which the
SGLang runtime already consults when loading fused-MoE configs. Serving knobs
are pinned to the validated per-regime winner from the serving campaign, so any
difference is attributable to the kernel profile alone.

Reuses the canonical serving harness (`scripts/serving_ceiling_lib.py`) for
server lifecycle, resolved-knob verification, per-workload warm-up and the
streaming benchmark client, so E2E numbers are directly comparable with the
existing campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import rk_lib as L
import serving_ceiling_lib as S
import run_serving_ceiling_campaign as C

OUT = L.RESULTS / "e2e"

# Regime -> (workload, serving knobs). Knobs come from the validated per-regime
# winners of the serving-ceiling campaign; they are held FIXED across arms.
REGIME_SERVING = {
    "A_low_batch_decode": dict(workload="R_short_decode",
                               cap=32, chunk=-1, policy="lpm", mem=0.85),
    "B_concurrent_decode": dict(workload="R_concurrent_decode",
                                cap=32, chunk=-1, policy="lpm", mem=0.85),
    "C_long_prefill": dict(workload="R_long_prefill",
                           cap=32, chunk=-1, policy="lpm", mem=0.85),
}


def arm_env(model: str, arm: str):
    """Environment for one arm; `default` leaves SGLANG_MOE_CONFIG_DIR unset."""
    env = L.run_env()
    if arm != "default":
        d = L.CONFIGS / "profiles" / f"{model}_{arm}"
        if not d.exists():
            raise SystemExit(f"profile dir missing: {d}")
        env["SGLANG_MOE_CONFIG_DIR"] = str(d)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(L.MODELS))
    ap.add_argument("--regime", required=True, choices=list(REGIME_SERVING))
    ap.add_argument("--arms", default="default,global_best,regime_aware")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, default=42000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    spec = REGIME_SERVING[a.regime]
    wl = spec["workload"]
    cfg = dict(cap=spec["cap"], chunk=spec["chunk"], policy=spec["policy"],
               mem=spec["mem"], config_id=-1,
               hash=f"cap{spec['cap']}_chunk{spec['chunk']}_pol{spec['policy']}_mem{spec['mem']}",
               is_cookbook=False)
    outdir = OUT / a.model / a.regime
    outdir.mkdir(parents=True, exist_ok=True)

    plan = dict(model=a.model, regime=a.regime, workload=wl,
                serving_knobs={k: spec[k] for k in ("cap", "chunk", "policy", "mem")},
                arms=a.arms.split(","), reps=a.reps, gpu=a.gpu)
    print(json.dumps(plan, indent=2))
    if a.dry_run:
        print("[dry-run] would run "
              f"{len(plan['arms'])} arms x {a.reps} reps on {wl}")
        return

    L.snapshot(outdir, "plan", dict(plan=plan, environment=L.environment()))
    rows = []
    for arm in plan["arms"]:
        log = outdir / f"server_{arm}.log"
        env = arm_env(a.model, arm)
        # launch through the canonical harness but with our env overlay
        old = dict(os.environ)
        os.environ.update({k: v for k, v in env.items()
                           if k in ("SGLANG_MOE_CONFIG_DIR",)})
        try:
            p, argv = S.launch_server(a.model, cfg, a.gpu, a.port, log)
            ok, info = S.wait_health(p, a.port, t=700)
            if not ok:
                S.kill_server(p)
                rows.append(dict(arm=arm, status="launch_failed", info=str(info)))
                continue
            resolved = S.parse_resolved(log)
            # warm-up, then measured repetitions
            for w in range(S.WARMUP_RUNS.get(wl, 1)):
                S.run_workload(a.model, wl, a.port, outdir / f"{arm}_warm{w}.jsonl")
            for rep in range(a.reps):
                tmp = outdir / f"{arm}_rep{rep}.jsonl"
                res, err, tail = S.run_workload(a.model, wl, a.port, tmp)
                if err:
                    rows.append(dict(arm=arm, rep=rep, status="bench_failed",
                                     info=str(err)[:200]))
                    continue
                row, _per_req = C.summarize(res, cfg, a.model, wl, rep, arm)
                row.update(arm=arm,
                           moe_config_dir=env.get("SGLANG_MOE_CONFIG_DIR", ""),
                           attention_backend=resolved.get("attention_backend"),
                           moe_runner_backend=resolved.get("moe_runner_backend"),
                           cuda_graph_captured=resolved.get("cuda_graph_captured"),
                           status="ok")
                rows.append(row)
            S.kill_server(p)
        finally:
            os.environ.clear(); os.environ.update(old)
        time.sleep(5)

    (outdir / "e2e_runs.json").write_text(json.dumps(rows, indent=2, default=str))
    print(f"wrote {outdir/'e2e_runs.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
