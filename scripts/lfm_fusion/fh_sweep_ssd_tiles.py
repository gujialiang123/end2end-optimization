#!/usr/bin/env python3
"""Sweep the mamba SSD kernels' tile sizes on Falcon-H1 prefill.

The three SSD kernels run at a hardcoded 16x16x16 with no autotune and no way
for a caller to override them; on Falcon-H1 they are 59 % of prefill kernel
time. This drives real bench_one_batch runs through ssd_inject/sitecustomize.py
so the measurement includes every interaction, rather than a microbenchmark that
has to reconstruct thirty stride arguments correctly.

Each configuration is run several times and the median prefill latency is
reported, because the first run of a new tile pays Triton compilation.

  python scripts/lfm_fusion/fh_sweep_ssd_tiles.py --gpu 3 --reps 3
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L  # noqa: E402

OUT = L.RESULTS / "processed"
INJECT = Path(__file__).resolve().parent / "ssd_inject"
MAMBA = Path(__file__).resolve().parent / "mamba_inject"
PREFILL = re.compile(r"Prefill\. latency: ([\d.]+) s")

# (M, N, K) for chunk_state and chunk_scan. 16 is the stock value.
TILES = [
    (16, 16, 16),      # stock
    (32, 32, 16),
    (32, 64, 16),
    (64, 64, 16),
    (64, 128, 16),
    (32, 32, 32),
    (64, 64, 32),
    (64, 128, 32),
    (128, 128, 32),
    (64, 64, 64),
]


def run(model: str, gpu: int, spec: str, shape: dict) -> float | None:
    m = L.MODELS[model]
    argv = [
        L.PY, "-m", "sglang.bench_one_batch",
        "--model-path", m["path"],
        "--batch", str(shape["batch"]),
        "--input-len", str(shape["input_len"]),
        "--output-len", str(shape["output_len"]),
        "--tensor-parallel-size", "1",
        "--mem-fraction-static", "0.85",
        "--disable-cuda-graph",
        "--trust-remote-code",
    ]
    env = L.run_env({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MAMBA_SSU_AUTOINIT": "triton",
        "PYTHONPATH": f"{INJECT}:{MAMBA}",
    })
    if spec:
        env["SSD_TILES"] = spec
    else:
        env.pop("SSD_TILES", None)
    p = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=1800)
    hits = PREFILL.findall(p.stdout)
    if not hits:
        tail = (p.stdout + p.stderr)[-400:]
        print(f"    FAILED: {tail.strip()[:200]}")
        return None
    return float(hits[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="falconh1")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--reps", type=int, default=3,
                    help="scored runs; one extra warm-up run is always discarded")
    ap.add_argument("--regime", default="C_long_prefill")
    a = ap.parse_args()
    shape = L.REGIME_SHAPES[a.regime]

    results = {}
    for tiles in TILES:
        spec = "" if tiles == (16, 16, 16) else \
            f"chunk_state:{tiles[0]},{tiles[1]},{tiles[2]};" \
            f"chunk_scan:{tiles[0]},{tiles[1]},{tiles[2]}"
        # The first process to use a given tile pays Triton compilation, which
        # showed up as 1.1-1.8 s against a 0.68 s steady state, and the stock
        # tile was already cached from earlier audit runs -- so scoring the
        # first run compares a cold config against a warm one.
        run(a.model, a.gpu, spec, shape)
        lat = [run(a.model, a.gpu, spec, shape) for _ in range(a.reps)]
        lat = [x for x in lat if x is not None]
        if not lat:
            print(f"  {str(tiles):<18} failed")
            results[str(tiles)] = None
            continue
        med = st.median(lat)
        results[str(tiles)] = dict(median_s=med, runs=lat)
        base = results.get("(16, 16, 16)")
        rel = f"{base['median_s'] / med:.3f}x" if base else "--"
        print(f"  {str(tiles):<18} {med:.4f} s   {rel}"
              f"   (n={len(lat)}, spread {max(lat) - min(lat):.4f})")

    base = results.get("(16, 16, 16)")
    ok = {k: v for k, v in results.items() if v}
    if base and ok:
        best = min(ok, key=lambda k: ok[k]["median_s"])
        print(f"\nstock (16,16,16): {base['median_s']:.4f} s")
        print(f"best  {best}: {ok[best]['median_s']:.4f} s"
              f"  -> {base['median_s'] / ok[best]['median_s']:.3f}x")

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{a.model}_ssd_tile_sweep_{a.regime}.json"
    dest.write_text(json.dumps(
        {"model": a.model, "regime": a.regime, "shape": shape,
         "results": results}, indent=2))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
