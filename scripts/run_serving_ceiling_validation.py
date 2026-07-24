#!/usr/bin/env python3
"""Phase-4 validation pass: re-run selected configurations with N repetitions.

Selection per (model, workload), from the coverage summary:
  cookbook baseline, highest request-throughput, highest output-throughput,
  lowest TTFT p95, lowest TPOT p95, lowest E2E p95, a sample of Pareto configs,
  and one or two clear regression configs.
Configs are de-duplicated across workloads (one server launch covers all six),
then re-measured with `--reps` repetitions so that final claims rest on repeated
means with 95 % confidence intervals rather than a single-run ranking.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import serving_ceiling_lib as L
import run_serving_ceiling_campaign as C

WORKLOADS = list(L.WORKLOADS)


def select(root: Path, model: str, n_pareto: int = 4):
    d = root / "analysis" / model
    sm = pd.read_csv(d / "summary_matrix.csv")
    deltas = pd.read_csv(d / "per_workload_deltas.csv")
    picked, why = set(), {}

    def add(cid, reason):
        cid = int(cid)
        picked.add(cid)
        why.setdefault(cid, []).append(reason)

    cb = int(sm.cookbook_config_id.iloc[0])
    add(cb, "cookbook baseline")
    for _, r in sm.iterrows():
        wl = r.workload
        add(r.best_throughput_config_id, f"{wl}: best request throughput")
        add(r.best_output_config_id, f"{wl}: best output throughput")
        add(r.lowest_ttft_p95_config_id, f"{wl}: lowest TTFT p95")
        add(r.lowest_tpot_p95_config_id, f"{wl}: lowest TPOT p95")
        add(r.lowest_e2e_p95_config_id, f"{wl}: lowest E2E p95")
        add(r.balanced_config_id, f"{wl}: balanced/Pareto pick")
        sub = deltas[deltas.workload == wl]
        par = sub[sub.pareto_ttft_outthr].sort_values("output_throughput",
                                                      ascending=False)
        for cid in par.config_id.head(n_pareto):
            add(cid, f"{wl}: Pareto point")
        # clear regressions: the two worst request-throughput configs
        for cid in sub.nsmallest(2, "request_throughput").config_id:
            add(cid, f"{wl}: clear regression")
    return sorted(picked), {k: sorted(set(v)) for k, v in why.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--valroot", required=True)
    ap.add_argument("--models", default="qwen,lfm25")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--port", type=int)
    ap.add_argument("--worker", default="v0")
    args = ap.parse_args()

    root, val = Path(args.outroot), Path(args.valroot)
    val.mkdir(parents=True, exist_ok=True)
    configs = L.build_configs()
    cfg_by_id = {c["config_id"]: c for c in configs}

    if args.init:
        sel_all, rationale = {}, {}
        for model in args.models.split(","):
            ids, why = select(root, model)
            sel_all[model] = ids
            rationale[model] = {str(k): v for k, v in why.items()}
            print(f"{model}: {len(ids)} unique configs selected for validation")
        (val / "validation_selection.json").write_text(
            json.dumps({"selected": sel_all, "rationale": rationale,
                        "reps": args.reps}, indent=2))
        dbp = str(val / "campaign.db")
        con = C.db_connect(dbp)
        con.execute("""CREATE TABLE IF NOT EXISTS tasks(
            model TEXT, config_id INTEGER, hash TEXT, is_cookbook INTEGER,
            status TEXT DEFAULT 'pending', worker TEXT, attempts INTEGER DEFAULT 0,
            started REAL, finished REAL, note TEXT, PRIMARY KEY(model, config_id))""")
        for model, ids in sel_all.items():
            for cid in ids:
                c = cfg_by_id[cid]
                con.execute("INSERT OR IGNORE INTO tasks(model,config_id,hash,is_cookbook)"
                            " VALUES(?,?,?,?)",
                            (model, cid, c["hash"], int(c["is_cookbook"])))
        con.commit()
        print("validation queue:",
              con.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], "tasks")
        con.close()
        return

    assert args.gpu is not None and args.port is not None
    dbp = str(val / "campaign.db")
    import time
    while True:
        claim = C.claim_task(dbp, args.worker)
        if claim is None:
            print(f"[{args.worker}] validation queue drained", flush=True)
            break
        model, cid = claim
        cfg = cfg_by_id[cid]
        t0 = time.time()
        try:
            status, note = C.process_task(model, cfg, args.gpu, args.port, val,
                                          args.reps, args.worker, dbp, False)
        except Exception as e:
            status, note = "failed", f"exc:{e}"
        C.mark(dbp, model, cid, status, note)
        print(f"[{args.worker}] {model} cfg{cid} -> {status} ({note}) "
              f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
