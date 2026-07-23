#!/usr/bin/env python3
"""v48 post-search validation — re-run top configs + cookbook, interleaved x5.

Selects cookbook + trial with highest raw objective + next 4 unique configs,
re-runs each 5 times in randomized/interleaved order, reports mean/std/95% CI,
speedup vs cookbook, and ranking stability. Writes validation_repeats.csv,
best_validated.json, best_raw.json.
"""
from __future__ import annotations
import argparse, csv, json, math, random, statistics, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_v48_lfm25_plateau as H

OUTDIR = H.OUTDIR


def load_trials():
    rows = list(csv.DictReader(open(OUTDIR / "per_trial_log.csv")))
    for r in rows:
        r["request_throughput"] = float(r["request_throughput"])
    return rows


def cfg_from_row(r):
    return dict(
        max_running_requests=int(r["max_running_requests"]),
        chunked_prefill_size=int(r["chunked_prefill_size"]),
        schedule_policy=r["schedule_policy"],
        mem_fraction_static=float(r["mem_fraction_static"]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--port", type=int, default=31702)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    rows = load_trials()
    rows_sorted = sorted(rows, key=lambda r: -r["request_throughput"])
    best_raw = rows_sorted[0]
    json.dump(dict(config=cfg_from_row(best_raw),
                   request_throughput=best_raw["request_throughput"],
                   completed_index=best_raw["completed_index"]),
              open(OUTDIR / "best_raw.json", "w"), indent=2)

    # candidates: cookbook + top-5 unique configs
    candidates = {"cookbook": H.COOKBOOK}
    for r in rows_sorted[:args.top]:
        candidates[f"trial_{r['completed_index']}"] = cfg_from_row(r)

    # interleaved schedule
    schedule = []
    for rep in range(args.repeats):
        names = list(candidates.keys())
        random.Random(20260722 + rep).shuffle(names)
        schedule += [(n, rep) for n in names]

    vdir = OUTDIR / "validation_runs"
    vdir.mkdir(parents=True, exist_ok=True)
    results = {n: [] for n in candidates}
    csv_path = OUTDIR / "validation_repeats.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["candidate", "repeat", "request_throughput", "status", "reason"])
        for name, rep in schedule:
            tdir = vdir / f"{name}_rep{rep}"
            m, status, reason, notes, startup = H.evaluate(candidates[name], args.port, args.gpu, tdir)
            rps = m["request_throughput"] if status == "ok" else None
            if rps:
                results[name].append(rps)
            w.writerow([name, rep, rps if rps else "", status, reason])
            f.flush()
            print(f"[validate] {name} rep{rep}: {rps if rps else 'FAIL('+reason+')'}", flush=True)

    def stats(xs):
        if len(xs) < 2:
            return dict(mean=(xs[0] if xs else None), std=None, ci95=None, n=len(xs))
        mean = statistics.mean(xs); std = statistics.stdev(xs)
        return dict(mean=mean, std=std, ci95=1.96 * std / math.sqrt(len(xs)), n=len(xs))

    cook_mean = stats(results["cookbook"])["mean"]
    summary = {}
    for name, xs in results.items():
        s = stats(xs)
        s["config"] = candidates[name]
        s["speedup_vs_cookbook"] = (s["mean"] / cook_mean) if (s["mean"] and cook_mean) else None
        summary[name] = s

    # validated best = highest validated mean among non-cookbook
    non_cook = {n: s for n, s in summary.items() if n != "cookbook" and s["mean"]}
    best_name = max(non_cook, key=lambda n: non_cook[n]["mean"])
    raw_rank = [f"trial_{r['completed_index']}" for r in rows_sorted[:args.top]]
    validated_rank = sorted(non_cook, key=lambda n: -non_cook[n]["mean"])
    out = dict(
        cookbook_validated=summary["cookbook"],
        best_validated_name=best_name,
        best_validated=summary[best_name],
        all_candidates=summary,
        raw_ranking=raw_rank,
        validated_ranking=validated_rank,
        ranking_stable=(raw_rank[0] == validated_rank[0]),
    )
    json.dump(out, open(OUTDIR / "best_validated.json", "w"), indent=2)
    print(f"\nBEST validated: {best_name} mean={summary[best_name]['mean']:.3f} "
          f"speedup={summary[best_name]['speedup_vs_cookbook']:.3f}x")
    print(f"ranking stable (raw#1==validated#1): {out['ranking_stable']}")
    print(f"saved {OUTDIR}/best_validated.json, validation_repeats.csv, best_raw.json")


if __name__ == "__main__":
    main()
