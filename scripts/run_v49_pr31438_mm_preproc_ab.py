#!/usr/bin/env python3
"""v49: PR #31438 (parallelize multimodal preprocessing) e2e A/B on Qwen3.6-35B VLM.

The PR moves image I/O + HF-processor work off the tokenizer event loop onto
dedicated I/O + processor worker pools (default 2 processor workers, 16 I/O workers
for the patched arm; baseline is serial on the event loop). The benefit shows on
IMAGE BURSTS: many requests, each with several images, hitting the tokenizer at once.

Two things this script does:
  1. correctness probe: same fixed image+text prompt, temp 0, capture greedy output.
     Parallel preprocessing must not change results -> baseline and patched outputs
     should match token-for-token.
  2. burst A/B: fire N concurrent requests each with K images, measure per-request
     TTFT (server-side preprocessing + prefill). Report mean/median/p99 TTFT and
     wall time to drain the burst.

Server lifecycle handled by the caller. This drives requests against a running server.
"""
import argparse, base64, io, json, os, statistics, time, threading, urllib.request

PORT = int(os.environ.get("PORT", "31710"))


def make_image_b64(w, h, seed):
    from PIL import Image
    import numpy as np
    rng = np.random.RandomState(seed)
    img = Image.fromarray((rng.rand(h, w, 3) * 255).astype("uint8"))
    buf = io.BytesIO(); img.save(buf, format="JPEG"); return base64.b64encode(buf.getvalue()).decode()


def one_request(images_b64, prompt, max_tokens, temperature, stream=True):
    content = [{"type": "text", "text": prompt}]
    for b in images_b64:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    payload = {"model": "x", "stream": stream, "max_tokens": max_tokens,
               "temperature": temperature, "messages": [{"role": "user", "content": content}]}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=data, headers={"Content-Type": "application/json"})
    t0 = time.time(); ttft = None; text = ""
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if not stream:
                j = json.loads(resp.read())
                return (time.time() - t0, time.time() - t0,
                        j["choices"][0]["message"]["content"], True)
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    j = json.loads(body)
                except Exception:
                    continue
                d = j.get("choices", [{}])[0].get("delta", {})
                if d.get("content"):
                    if ttft is None:
                        ttft = time.time() - t0
                    text += d["content"]
        return (ttft, time.time() - t0, text, ttft is not None)
    except Exception as e:
        return (None, time.time() - t0, f"ERR:{e}", False)


def correctness_probe(arm, out, n_imgs=2):
    imgs = [make_image_b64(512, 512, seed=s) for s in range(n_imgs)]
    _, _, text, ok = one_request(imgs, "List what you see, deterministically.",
                                 max_tokens=48, temperature=0, stream=False)
    row = {"arm": arm, "kind": "correctness", "n_imgs": n_imgs, "ok": ok,
           "output": text, "ts": time.strftime("%H:%M:%S")}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{arm}] correctness: ok={ok} out={text[:100]!r}", flush=True)
    return text


def burst(arm, out, n_req, k_imgs, conc, max_tokens=8):
    """Fire n_req requests (k images each) with a concurrency cap; record TTFTs."""
    imgs_per_req = [[make_image_b64(640 + 32 * (j % 6), 480 + 32 * (j % 5), seed=i * 10 + j)
                     for j in range(k_imgs)] for i in range(n_req)]
    results = [None] * n_req
    sem = threading.Semaphore(conc)
    threads = []

    def worker(i):
        with sem:
            ttft, tot, _, ok = one_request(imgs_per_req[i], "Describe briefly.",
                                           max_tokens=max_tokens, temperature=0)
            results[i] = (ttft, tot, ok)

    t0 = time.time()
    for i in range(n_req):
        t = threading.Thread(target=worker, args=(i,)); t.start(); threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    ttfts = [r[0] * 1000 for r in results if r and r[0] is not None]
    oks = sum(1 for r in results if r and r[2])
    row = {"arm": arm, "kind": "burst", "n_req": n_req, "k_imgs": k_imgs, "conc": conc,
           "wall_s": round(wall, 2), "ok": oks, "n": len(ttfts),
           "ttft_mean_ms": round(statistics.mean(ttfts), 1) if ttfts else None,
           "ttft_median_ms": round(statistics.median(ttfts), 1) if ttfts else None,
           "ttft_p99_ms": round(sorted(ttfts)[int(len(ttfts) * 0.99) - 1], 1) if len(ttfts) > 1 else None,
           "req_throughput": round(oks / wall, 2), "ts": time.strftime("%H:%M:%S")}
    with open(out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[{arm}] burst n={n_req} k={k_imgs} c{conc}: wall={wall:.1f}s "
          f"TTFT mean={row['ttft_mean_ms']} p99={row['ttft_p99_ms']} "
          f"req_tput={row['req_throughput']} ok={oks}/{n_req}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    correctness_probe(args.arm, args.out)
    for rep in range(args.repeats):
        burst(args.arm, args.out, n_req=32, k_imgs=4, conc=16)
    for rep in range(args.repeats):
        burst(args.arm, args.out, n_req=16, k_imgs=4, conc=8)


if __name__ == "__main__":
    main()
