#!/usr/bin/env python3
"""v51 — high-concurrency TTFT rerun for the v4 slide data points.

Faithfully reproduces the ORIGINAL e2e-bench-runner protocol that generated
results/consolidated_v4_by_model_config.csv (200-word random prompts, seed=2026,
num_prompts=concurrency, ThreadPoolExecutor(concurrency), temperature=0,
ignore_eos, fixed max_new), but sends STREAMING /generate requests so we can
measure client-observed TTFT (submit -> first output token), which the original
non-streaming client could not capture.

Metrics per request: TTFT, per-token inter-arrival (ITL), TPOT (mean ITL),
E2E (submit -> last token), output_tokens. Aggregated to mean/p50/p95/p99.

Client-observed TTFT INCLUDES admission/scheduler queueing (that is the point).
"""
from __future__ import annotations
import argparse, concurrent.futures, json, random, time, statistics
from pathlib import Path
import requests

# ---- EXACT copy of the original e2e-bench-runner prompt generator ----
WORDS = ("machine learning artificial intelligence deep neural network "
         "training inference optimization performance benchmark profiling "
         "kernel implementation source code framework library backend "
         "transformer attention encoder decoder embedding tokenizer batch "
         "processing efficiency throughput latency memory bandwidth").split()
TIMEOUT_S = 600


def make_prompts(n: int, words: int, seed: int = 2026) -> list[str]:
    rng = random.Random(seed)
    return [" ".join(rng.choice(WORDS) for _ in range(words)) for _ in range(n)]


def send_sglang_stream(url: str, prompt: str, max_new: int) -> dict:
    """Streaming /generate. Returns per-request timing including TTFT."""
    t_submit = time.perf_counter()
    r = requests.post(f"{url}/generate", json={
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0,
                            "ignore_eos": True},
        "stream": True,
    }, timeout=TIMEOUT_S, stream=True)
    r.raise_for_status()
    ttft = None
    token_times = []
    last_len = 0
    completion_tokens = 0
    for line in r.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            now = time.perf_counter()
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            meta = obj.get("meta_info", {}) or {}
            ct = meta.get("completion_tokens")
            if ttft is None:
                ttft = now - t_submit
            token_times.append(now)
            if ct is not None:
                completion_tokens = ct
    t_end = time.perf_counter()
    if completion_tokens == 0:
        completion_tokens = len(token_times)
    # inter-token latencies from streaming chunk arrivals
    itls = [token_times[i] - token_times[i-1] for i in range(1, len(token_times))]
    return {
        "ttft_s": ttft,
        "e2e_s": t_end - t_submit,
        "itls": itls,
        "output_tokens": completion_tokens,
        "n_chunks": len(token_times),
    }


def run_regime_once(url: str, num_prompts: int, prompt_words: int, max_new: int,
                    concurrency: int, seed: int = 2026) -> dict:
    prompts = make_prompts(num_prompts, prompt_words, seed=seed)
    records, errors = [], []

    def worker(idx, prompt):
        try:
            res = send_sglang_stream(url, prompt, max_new)
            res["idx"] = idx; res["ok"] = True; res["error"] = None
            return res
        except Exception as e:
            return {"idx": idx, "ok": False, "error": f"{type(e).__name__}: {e}",
                    "ttft_s": None, "e2e_s": None, "itls": [], "output_tokens": 0}

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, i, p) for i, p in enumerate(prompts)]
        for f in concurrent.futures.as_completed(futures):
            rec = f.result()
            records.append(rec)
            if not rec["ok"]:
                errors.append(rec["error"])
    wall_s = time.perf_counter() - wall_start

    ok = [r for r in records if r["ok"]]
    total_out = sum(r["output_tokens"] for r in ok)
    return {
        "wall_s": wall_s,
        "req_per_s": len(ok) / wall_s if wall_s > 0 else 0.0,
        "output_tokens_per_s": total_out / wall_s if wall_s > 0 else 0.0,
        "completion_rate": len(ok) / max(1, len(records)),
        "num_ok": len(ok), "num_total": len(records),
        "errors": errors[:5],
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--prompt-words", type=int, default=200)
    ap.add_argument("--max-new", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    res = run_regime_once(args.url, args.num_prompts, args.prompt_words,
                          args.max_new, args.concurrency, args.seed)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"req/s={res['req_per_s']:.3f} num_ok={res['num_ok']}/{res['num_total']} "
          f"first_ttft={(res['records'][0].get('ttft_s') or 0)*1000:.1f}ms")


if __name__ == "__main__":
    main()
