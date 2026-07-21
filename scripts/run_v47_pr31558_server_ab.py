#!/usr/bin/env python3
"""v47: server-level e2e A/B for PR #31558 (avoid FLA l2-norm recompile by token
count) on Qwen3.6-35B-A3B-FP8 (hybrid linear-attention VLM).

Headline scenario (per the PR + our plan): a VLM served with a COLD Triton cache,
hit with images of VARYING resolution. Baseline compiles a fresh l2norm kernel for
each distinct token count (image size) -> compile stalls land on TTFT of the first
request at each new shape. Patched compiles once.

Design:
  - For each arm (baseline / patched l2norm.py in place), launch a FRESH server
    (cold Triton cache is the whole point). After warmup-ready, immediately fire a
    stream of image requests whose resolution cycles across a set, so several NEW
    token-counts are seen early. Measure per-request TTFT (first-token latency).
  - "dynamic" experiment: cycle resolutions {360p,720p,1080p,512x512,768x1024,...}
    -> baseline pays repeated compile stalls; patched pays one.
  - "fixed" control: single resolution, warm -> both compile once -> expect ~0.

This script drives one arm+mode per invocation (server lifecycle handled here), and
appends per-request rows to a jsonl. Uses the OpenAI /v1/chat/completions streaming
endpoint to capture true TTFT per request.
"""
import argparse, base64, io, json, os, subprocess, sys, time, signal, urllib.request, threading

MODEL = "/home/t-jialianggu/work/models/Qwen3.6-35B-A3B-FP8"
SGLANG = "/home/t-jialianggu/work/sglang-v515"
ENVDIR = "/home/t-jialianggu/.conda/envs/sglang-v515"
PY = f"{ENVDIR}/bin/python"
REPO = "/home/t-jialianggu/work/end2end-optimization"
PORT = int(os.environ.get("PORT", "31610"))
GPU = os.environ.get("GPU", "0")

RES_SET = ["360p", "720p", "1080p", "512x512", "640x800", "768x1024", "900x1200", "1024x1024"]

base_env = os.environ.copy()
base_env.update(dict(
    CUDA_HOME=ENVDIR, HF_HOME=f"{REPO}/../hf_cache",
    PYTHONPATH=f"{SGLANG}/python",
    PATH=f"{ENVDIR}/bin:" + base_env.get("PATH", ""),
    CUDA_VISIBLE_DEVICES=GPU,
))


def make_image_b64(w, h):
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8"))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return base64.b64encode(buf.getvalue()).decode()


def parse_res(r):
    presets = {"4k": (3840, 2160), "1080p": (1920, 1080), "720p": (1280, 720), "360p": (640, 360)}
    if r in presets:
        return presets[r]
    h, w = r.split("x"); return (int(w), int(h))


def wait_ready(timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return True
        except Exception:
            time.sleep(3)
    return False


def launch_server(arm, log_path):
    log = open(log_path, "w")
    argv = [PY, "-m", "sglang.launch_server", "--model-path", MODEL, "--trust-remote-code",
            "--host", "127.0.0.1", "--port", str(PORT), "--mem-fraction-static", "0.85"]
    p = subprocess.Popen(argv, env=base_env, stdout=log, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    return p, log


def one_request(w, h, max_tokens=8):
    """Streaming request; return (ttft_s, total_s, ok)."""
    b64 = make_image_b64(w, h)
    payload = {
        "model": "x", "stream": True, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe in one word."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line or not line.startswith("data:"):
                    continue
                body = line[len("data:"):].strip()
                if body == "[DONE]":
                    break
                try:
                    j = json.loads(body)
                except Exception:
                    continue
                delta = j.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
        return (ttft, time.time() - t0, ttft is not None)
    except Exception as e:
        return (None, time.time() - t0, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["baseline", "patched"])
    ap.add_argument("--mode", required=True, choices=["dynamic", "fixed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3, help="passes over the resolution set")
    ap.add_argument("--fixed-res", type=str, default="720p")
    args = ap.parse_args()

    # place the right l2norm variant
    src = f"{REPO}/patches/l2norm_v0.5.15.post1_" + ("pr31558" if args.arm == "patched" else "baseline") + ".py"
    dst = f"{SGLANG}/python/sglang/srt/layers/attention/fla/l2norm.py"
    subprocess.run(["cp", src, dst], check=True)

    logp = f"{REPO}/logs/v47_server_{args.arm}_{args.mode}.log"
    print(f"[{args.arm}/{args.mode}] launching fresh server (cold cache)...", flush=True)
    p, log = launch_server(args.arm, logp)
    if not wait_ready():
        print("server failed", flush=True)
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
        return
    print(f"[{args.arm}/{args.mode}] ready, firing requests", flush=True)

    if args.mode == "dynamic":
        seq = RES_SET
    else:
        seq = [args.fixed_res]

    idx = 0
    for rnd in range(args.rounds):
        for res in seq:
            w, h = parse_res(res)
            ttft, tot, ok = one_request(w, h)
            row = {"arm": args.arm, "mode": args.mode, "round": rnd, "res": res,
                   "w": w, "h": h, "seq_idx": idx, "ttft_s": ttft, "total_s": tot,
                   "ok": ok, "first_seen_res": (rnd == 0), "ts": time.strftime("%H:%M:%S")}
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
            ft = f"{ttft*1000:.1f}ms" if ttft else "FAIL"
            print(f"  r{rnd} {res:<9} idx={idx:<3} TTFT={ft}", flush=True)
            idx += 1

    try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception: pass
    log.close()
    print(f"[{args.arm}/{args.mode}] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
