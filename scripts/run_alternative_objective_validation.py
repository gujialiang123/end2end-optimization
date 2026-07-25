#!/usr/bin/env python3
"""Phase-4: run ONLY the missing targeted validations for alternative objectives.

Reads results/2026-07-26_alternative_objectives/validation_plan.csv, queues the
configurations that lack five valid repetitions, and runs them under the exact
canonical protocol (same harness, same six workloads, same warm-up, same seeds).

The cookbook (config_id 74) is queued FIRST for each model as an interleaved
anchor, so that the new run window can be compared against the original
validation baseline CI.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_ceiling_lib as L
import run_serving_ceiling_campaign as C

COOKBOOK_ID = 74


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="results/2026-07-26_alternative_objectives/validation_plan.csv")
    ap.add_argument("--outroot", default="results/2026-07-26_alternative_objectives")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--worker", default="a0")
    a = ap.parse_args()

    out = Path(a.outroot)
    out.mkdir(parents=True, exist_ok=True)
    dbp = str(out / "campaign.db")
    cfg_by_id = {c["config_id"]: c for c in L.build_configs()}

    if a.init:
        plan = pd.read_csv(a.plan)
        need = plan[plan.needs_run].copy()
        con = C.db_connect(dbp)
        con.execute("""CREATE TABLE IF NOT EXISTS tasks(
            model TEXT, config_id INTEGER, hash TEXT, is_cookbook INTEGER,
            status TEXT DEFAULT 'pending', worker TEXT, attempts INTEGER DEFAULT 0,
            started REAL, finished REAL, note TEXT, PRIMARY KEY(model, config_id))""")
        # anchor first so it is measured inside the same time window
        for model in sorted(need.model.unique()):
            c = cfg_by_id[COOKBOOK_ID]
            con.execute("INSERT OR IGNORE INTO tasks(model,config_id,hash,is_cookbook)"
                        " VALUES(?,?,?,1)", (model, COOKBOOK_ID, c["hash"]))
        for _, r in need.iterrows():
            c = cfg_by_id[int(r.config_id)]
            con.execute("INSERT OR IGNORE INTO tasks(model,config_id,hash,is_cookbook)"
                        " VALUES(?,?,?,?)",
                        (r.model, int(r.config_id), c["hash"], int(c["is_cookbook"])))
        con.commit()
        n = con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        print(f"queued {n} tasks ({len(need)} missing candidates + cookbook anchors)")
        con.close()
        return

    assert a.gpu is not None and a.port is not None
    while True:
        claim = C.claim_task(dbp, a.worker)
        if claim is None:
            print(f"[{a.worker}] queue drained", flush=True)
            break
        model, cid = claim
        t0 = time.time()
        try:
            status, note = C.process_task(model, cfg_by_id[cid], a.gpu, a.port,
                                          out, a.reps, a.worker, dbp, False)
        except Exception as e:
            status, note = "failed", f"exc:{e}"
        C.mark(dbp, model, cid, status, note)
        print(f"[{a.worker}] {model} cfg{cid} -> {status} ({note}) "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
