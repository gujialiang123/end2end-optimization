#!/usr/bin/env python3
"""2026-07-24 Qwen/LFM serving-ceiling campaign runner.

One server launch per (model, config) evaluates ALL six canonical workloads, so
the same configuration is directly comparable across regimes. Coordination and
resume-safety via a shared sqlite work-queue so up to N GPU workers run in
parallel with dynamic load balancing.

Usage:
  # 1. initialise the work queue (run once)
  python run_serving_ceiling_campaign.py --init --models qwen,lfm25 --outroot RESULTS

  # 2. launch one worker per free GPU (each pulls tasks until queue drains)
  GPU=4 PORT=33004 python run_serving_ceiling_campaign.py --gpu 4 --port 33004 --worker g4 --outroot RESULTS
  GPU=5 PORT=33005 python run_serving_ceiling_campaign.py --gpu 5 --port 33005 --worker g5 --outroot RESULTS
  GPU=6 PORT=33006 python run_serving_ceiling_campaign.py --gpu 6 --port 33006 --worker g6 --outroot RESULTS

  # smoke: only a few configs, tagged output
  python ... --smoke-configs 74,0,191
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sqlite3
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_ceiling_lib as L

WORKLOAD_ORDER = list(L.WORKLOADS)


def pctl(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k); c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def db_connect(dbpath):
    con = sqlite3.connect(dbpath, timeout=60)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA busy_timeout=60000;")
    return con


def db_init(dbpath, models, configs):
    con = db_connect(dbpath)
    con.execute("""CREATE TABLE IF NOT EXISTS tasks(
        model TEXT, config_id INTEGER, hash TEXT, is_cookbook INTEGER,
        status TEXT DEFAULT 'pending', worker TEXT, attempts INTEGER DEFAULT 0,
        started REAL, finished REAL, note TEXT,
        PRIMARY KEY(model, config_id))""")
    for model in models:
        for c in configs:
            con.execute("INSERT OR IGNORE INTO tasks(model,config_id,hash,is_cookbook) "
                        "VALUES(?,?,?,?)", (model, c["config_id"], c["hash"],
                                            int(c["is_cookbook"])))
    con.commit()
    n = con.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
    print("task queue:", dict(n), "total", con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
    con.close()


def claim_task(dbpath, worker):
    """Atomically claim one pending task (cookbook first, then by config_id)."""
    con = db_connect(dbpath)
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT model, config_id FROM tasks WHERE status='pending' "
            "ORDER BY is_cookbook DESC, config_id ASC LIMIT 1").fetchone()
        if row is None:
            con.execute("COMMIT")
            return None
        model, cid = row
        con.execute("UPDATE tasks SET status='running', worker=?, started=?, "
                    "attempts=attempts+1 WHERE model=? AND config_id=?",
                    (worker, time.time(), model, cid))
        con.execute("COMMIT")
        return model, cid
    finally:
        con.close()


def mark(dbpath, model, cid, status, note=""):
    con = db_connect(dbpath)
    con.execute("UPDATE tasks SET status=?, finished=?, note=? WHERE model=? AND config_id=?",
                (status, time.time(), note[:300], model, cid))
    con.commit(); con.close()


def append_csv(path, row, header):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if new:
            w.writeheader()
        w.writerow(row)


PER_RUN_HEADER = [
    "model", "config_id", "hash", "cap", "chunk", "policy", "mem", "is_cookbook",
    "workload", "rep", "completed", "num_prompts", "failed_reqs", "dur_s",
    "request_throughput", "input_throughput", "output_throughput", "total_throughput",
    "ttft_mean_ms", "ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms",
    "tpot_mean_ms", "tpot_p50_ms", "tpot_p95_ms", "tpot_p99_ms",
    "itl_mean_ms", "e2e_mean_ms", "e2e_p50_ms", "e2e_p95_ms", "e2e_p99_ms",
    "mean_in_len", "mean_out_len", "worker",
]


def next_free_port(base, span=40, tries=3):
    """Return a bindable port near `base`, rotating to avoid reuse of a port
    whose previous listener has not finished shutting down."""
    for _ in range(tries):
        for off in range(span):
            p = base + off
            if L.port_free(p):
                return p
        time.sleep(10)
    return None


def summarize(res, cfg, model, workload, rep, worker):
    """Compute per-run row + per-request records from bench_serving details."""
    ttfts = [t * 1000 for t in res.get("ttfts", [])]                 # ms
    itls_lists = res.get("itls", [])                                  # list[list] s
    out_lens = res.get("output_lens", [])
    in_lens = res.get("input_lens", [])
    errors = res.get("errors", [])
    per_req = []
    tpots, e2es = [], []
    for i in range(len(ttfts)):
        itl = itls_lists[i] if i < len(itls_lists) else []
        olen = out_lens[i] if i < len(out_lens) else 0
        ttft = ttfts[i]
        sum_itl_ms = sum(itl) * 1000
        tpot = (statistics.mean(itl) * 1000) if itl else None
        e2e = ttft + sum_itl_ms
        if tpot is not None:
            tpots.append(tpot)
        e2es.append(e2e)
        per_req.append(dict(
            model=model, config_id=cfg["config_id"], workload=workload, rep=rep,
            req_idx=i, in_len=in_lens[i] if i < len(in_lens) else None,
            out_len=olen, ttft_ms=ttft, tpot_ms=tpot, e2e_ms=e2e,
            err=(errors[i] if i < len(errors) and errors[i] else None)))
    failed = sum(1 for e in errors if e)
    row = dict(
        model=model, config_id=cfg["config_id"], hash=cfg["hash"],
        cap=cfg["cap"], chunk=cfg["chunk"], policy=cfg["policy"], mem=cfg["mem"],
        is_cookbook=int(cfg["is_cookbook"]), workload=workload, rep=rep,
        completed=res.get("completed"), num_prompts=len(ttfts), failed_reqs=failed,
        dur_s=res.get("duration"),
        request_throughput=res.get("request_throughput"),
        input_throughput=res.get("input_throughput"),
        output_throughput=res.get("output_throughput"),
        total_throughput=res.get("total_throughput"),
        ttft_mean_ms=(statistics.mean(ttfts) if ttfts else None),
        ttft_p50_ms=pctl(ttfts, 50), ttft_p95_ms=pctl(ttfts, 95), ttft_p99_ms=pctl(ttfts, 99),
        tpot_mean_ms=(statistics.mean(tpots) if tpots else None),
        tpot_p50_ms=pctl(tpots, 50), tpot_p95_ms=pctl(tpots, 95), tpot_p99_ms=pctl(tpots, 99),
        itl_mean_ms=(res.get("mean_itl_ms")),
        e2e_mean_ms=(statistics.mean(e2es) if e2es else None),
        e2e_p50_ms=pctl(e2es, 50), e2e_p95_ms=pctl(e2es, 95), e2e_p99_ms=pctl(e2es, 99),
        mean_in_len=(statistics.mean(in_lens) if in_lens else None),
        mean_out_len=(statistics.mean(out_lens) if out_lens else None),
        worker=worker,
    )
    return row, per_req


def process_task(model, cfg, gpu, port, outroot, reps, worker, dbpath, keep_raw):
    tag = f"{model} cfg{cfg['config_id']} {cfg['hash']}"
    raw_dir = outroot / "raw" / model / f"config_{cfg['config_id']:03d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / "server.log"
    per_run_csv = outroot / f"per_run_metrics_{worker}.csv"
    per_config_csv = outroot / f"per_config_log_{worker}.csv"
    failures_csv = outroot / f"failures_{worker}.csv"

    # Pick a port that we can actually bind right now. Rotating avoids reusing a
    # port whose previous listener is still being torn down (Errno 98).
    port = next_free_port(port)
    if port is None:
        return "failed", "no-free-port"
    if not L.wait_gpu_free(gpu, need_free_mib=110000, t=240):
        return "failed", "gpu-busy"

    print(f"[{worker}] launching {tag} on GPU{gpu}:{port}", flush=True)
    p, argv = L.launch_server(model, cfg, gpu, port, log_path)
    ok, info = L.wait_health(p, port, t=700)
    if not ok:
        # infrastructure failure (e.g. bind race, SIGKILL during load): retry once
        L.kill_server(p)
        L.wait_gpu_free(gpu, need_free_mib=110000, t=240)
        append_csv(failures_csv, dict(model=model, config_id=cfg["config_id"],
                   hash=cfg["hash"], stage="launch-attempt1", cause=str(info)),
                   ["model", "config_id", "hash", "stage", "cause"])
        print(f"[{worker}] launch retry for {tag} ({info})", flush=True)
        time.sleep(20)
        port = next_free_port(port)      # never reuse the port that just failed
        if port is None:
            return "failed", "no-free-port-on-retry"
        p, argv = L.launch_server(model, cfg, gpu, port, log_path)
        ok, info = L.wait_health(p, port, t=700)
    if not ok:
        L.kill_server(p)
        append_csv(failures_csv, dict(model=model, config_id=cfg["config_id"],
                   hash=cfg["hash"], stage="launch", cause=str(info)),
                   ["model", "config_id", "hash", "stage", "cause"])
        return "failed", f"launch:{info}"
    resolved = L.parse_resolved(log_path)
    append_csv(per_config_csv, dict(
        model=model, config_id=cfg["config_id"], hash=cfg["hash"],
        cap=cfg["cap"], chunk=cfg["chunk"], policy=cfg["policy"], mem=cfg["mem"],
        startup_s=info, launch_cmd=" ".join(argv), **resolved),
        ["model", "config_id", "hash", "cap", "chunk", "policy", "mem",
         "startup_s", "attention_backend", "moe_runner_backend",
         "disable_cuda_graph", "cuda_graph_captured", "max_running_requests",
         "chunked_prefill_size", "schedule_policy", "mem_fraction_static",
         "launch_cmd"])

    all_per_req = []
    n_fail = 0
    # randomize workload order per config to reduce ordering bias
    import random
    order = WORKLOAD_ORDER[:]
    random.Random(L.SEED + cfg["config_id"]).shuffle(order)
    for wl in order:
        for rep in range(reps):
            tmp = raw_dir / f"{wl}_rep{rep}.jsonl"
            if tmp.exists():
                tmp.unlink()
            res, err, tail = L.run_workload(model, wl, port, tmp)
            if err:
                # retry once for infra-style failures
                res, err, tail = L.run_workload(model, wl, port, tmp)
            if err:
                n_fail += 1
                append_csv(failures_csv, dict(model=model, config_id=cfg["config_id"],
                           hash=cfg["hash"], stage=f"{wl}_rep{rep}", cause=f"{err} :: {tail[:200]}"),
                           ["model", "config_id", "hash", "stage", "cause"])
                continue
            row, per_req = summarize(res, cfg, model, wl, rep, worker)
            append_csv(per_run_csv, row, PER_RUN_HEADER)
            all_per_req.extend(per_req)
            if not keep_raw:
                tmp.unlink(missing_ok=True)  # drop bulky raw jsonl in coverage
    L.kill_server(p)

    # write per-request parquet shard for this task
    if all_per_req:
        try:
            import pandas as pd
            pq = raw_dir / "per_request.parquet"
            pd.DataFrame(all_per_req).to_parquet(pq, index=False)
        except Exception as e:
            print(f"[{worker}] parquet write failed {tag}: {e}", flush=True)
    status = "done" if n_fail == 0 else ("done" if all_per_req else "failed")
    return status, f"fails={n_fail}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--models", default="qwen,lfm25")
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--smoke-configs", default="")
    ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--max-tasks", type=int, default=100000)
    args = ap.parse_args()

    outroot = Path(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)
    dbpath = str(outroot / "campaign.db")
    configs = L.build_configs()
    cfg_by_id = {c["config_id"]: c for c in configs}
    models = args.models.split(",")

    if args.init:
        sel = configs
        if args.smoke_configs:
            ids = [int(x) for x in args.smoke_configs.split(",")]
            sel = [cfg_by_id[i] for i in ids]
        db_init(dbpath, models, sel)
        return

    assert args.gpu is not None and args.port is not None, "worker needs --gpu --port"
    done = 0
    while done < args.max_tasks:
        claim = claim_task(dbpath, args.worker)
        if claim is None:
            print(f"[{args.worker}] queue drained; exiting", flush=True)
            break
        model, cid = claim
        cfg = cfg_by_id[cid]
        t0 = time.time()
        try:
            status, note = process_task(model, cfg, args.gpu, args.port, outroot,
                                        args.reps, args.worker, dbpath, args.keep_raw)
        except Exception as e:
            status, note = "failed", f"exc:{e}"
        mark(dbpath, model, cid, status, note)
        done += 1
        print(f"[{args.worker}] {model} cfg{cid} -> {status} ({note}) "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
