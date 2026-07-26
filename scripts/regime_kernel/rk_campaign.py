#!/usr/bin/env python3
"""P0 driver: tuning sweep, transfer matrix and routing control.

A sqlite work queue lets several GPU workers share the job list, and makes the
whole campaign resume-safe: a completed (model, tokens, routing, stage) job is
never re-run after an interruption.

Stages
  sweep    : full pruned search space at each token count -> per-M best config
  transfer : every tuned profile evaluated at every token count (RQ2)
  routing  : uniform vs skewed routing at fixed M (routing-control experiment)
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rk_lib as L

RAW = L.RESULTS / "raw"
DB = L.RESULTS / "campaign.db"
PY = f"{L.ENVDIR}/bin/python"
HERE = Path(__file__).resolve().parent

# token counts to sweep: the light crossover sweep plus prefill-sized chunks
SWEEP_TOKENS = L.TOKEN_SWEEP + L.PREFILL_TOKENS


def db():
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB), timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS jobs(
        stage TEXT, model TEXT, tokens INTEGER, routing TEXT, tag TEXT,
        status TEXT DEFAULT 'pending', worker TEXT, started REAL, finished REAL,
        note TEXT, PRIMARY KEY(stage, model, tokens, routing, tag))""")
    con.commit()
    return con


def claim(worker: str):
    con = db()
    try:
        con.execute("BEGIN IMMEDIATE")
        row = con.execute(
            "SELECT stage,model,tokens,routing,tag FROM jobs WHERE status='pending'"
            " ORDER BY rowid LIMIT 1").fetchone()
        if row is None:
            con.rollback()
            return None
        con.execute("UPDATE jobs SET status='running', worker=?, started=? "
                    "WHERE stage=? AND model=? AND tokens=? AND routing=? AND tag=?",
                    (worker, time.time(), *row))
        con.commit()
        return row
    finally:
        con.close()


def mark(row, status, note=""):
    con = db()
    con.execute("UPDATE jobs SET status=?, finished=?, note=? WHERE stage=? AND "
                "model=? AND tokens=? AND routing=? AND tag=?",
                (status, time.time(), note, *row))
    con.commit()
    con.close()


def out_path(stage, model, tokens, routing, tag):
    return RAW / stage / model / f"{tag}_t{tokens}_{routing}.json"


def init_queue(models, stages):
    con = db()
    n = 0
    for model in models:
        if "sweep" in stages:
            for t in SWEEP_TOKENS:
                con.execute("INSERT OR IGNORE INTO jobs(stage,model,tokens,routing,tag)"
                            " VALUES('sweep',?,?,'uniform','full')", (model, t))
                n += 1
        if "routing" in stages:
            # fixed M, different routing distributions
            for t in (8, 32, 64, 512):
                for r in ("uniform", "skewed"):
                    con.execute("INSERT OR IGNORE INTO jobs(stage,model,tokens,routing,tag)"
                                " VALUES('routing',?,?,?,'ctrl')", (model, t, r))
                    n += 1
    con.commit()
    print("queue:", dict(con.execute(
        "SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall()))
    con.close()


def init_transfer(models):
    """Queue the transfer matrix once the sweep has produced profiles."""
    con = db()
    for model in models:
        prof = L.CONFIGS / f"{model}_profiles.json"
        if not prof.exists():
            print(f"[skip] {prof} missing; run the sweep first")
            continue
        for t in SWEEP_TOKENS:
            con.execute("INSERT OR IGNORE INTO jobs(stage,model,tokens,routing,tag)"
                        " VALUES('transfer',?,?,'uniform','profiles')", (model, t))
    con.commit()
    print("queue:", dict(con.execute(
        "SELECT status,COUNT(*) FROM jobs GROUP BY status").fetchall()))
    con.close()


def run_job(row, gpu, warmup, iters, repeats, dry):
    stage, model, tokens, routing, tag = row
    out = out_path(stage, model, tokens, routing, tag)
    if out.exists() and out.stat().st_size > 0:
        return "done", "already present"
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [PY, str(HERE / "rk_microbench.py"), "--model", model,
           "--tokens", str(tokens), "--routing", routing, "--out", str(out),
           "--warmup", str(warmup), "--iters", str(iters),
           "--repeats", str(repeats)]
    if stage == "transfer":
        cmd += ["--configs", str(L.CONFIGS / f"{model}_profiles.json")]
    if dry:
        print("[dry-run]", " ".join(cmd))
        return "done", "dry-run"
    env = L.run_env()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = out.with_suffix(".log")
    with open(log, "w") as lf:
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if p.returncode != 0 or not out.exists():
        tail = log.read_text(errors="ignore")[-400:]
        return "failed", f"rc={p.returncode} :: {tail[-200:]}"
    return "done", ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--init-transfer", action="store_true")
    ap.add_argument("--models", default="lfm25,qwen")
    ap.add_argument("--stages", default="sweep,routing")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--worker", default="w0")
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    models = a.models.split(",")
    if a.init:
        init_queue(models, a.stages.split(","))
        L.snapshot(L.RESULTS, "environment", L.environment())
        return
    if a.init_transfer:
        init_transfer(models)
        return

    assert a.gpu is not None, "--gpu required for a worker"
    while True:
        row = claim(a.worker)
        if row is None:
            print(f"[{a.worker}] queue drained", flush=True)
            break
        t0 = time.time()
        status, note = run_job(row, a.gpu, a.warmup, a.iters, a.repeats, a.dry_run)
        mark(row, status, note)
        print(f"[{a.worker}] {row} -> {status} {note} {time.time()-t0:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
