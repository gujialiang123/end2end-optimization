#!/usr/bin/env python3
"""Run the FX fusion scanner over real HF models.

Uses plain `transformers` rather than sglang on purpose: the point of this line
of work is that the analysis is framework- and hardware-agnostic. A model that
runs under torch.compile can be scanned, whatever the deployment stack.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fx_fusion_scanner import scan_module  # noqa: E402

MODELS = {
    "gemma3-1b": "/data/hf/models/gemma-3-1b-it",
    "qwen3-0.6b": "/data/hf/gujialiang123/models/Qwen3-0.6B",
    "olmo2-1b": "/data/hf/gujialiang123/models/OLMo-2-1B-Instruct",
    "granite-3.3-2b": "/data/hf/gujialiang123/models/granite-3.3-2b-instruct",
    "phi4-mini": "/data/hf/gujialiang123/models/Phi-4-mini-instruct",
    "exaone4-1.2b": "/data/hf/gujialiang123/models/EXAONE-4.0-1.2B",
    "falcon-h1-1.5b": "/data/hf/gujialiang123/models/Falcon-H1-1.5B-Instruct",
    "olmoe-1b-7b": "/data/hf/gujialiang123/models/OLMoE-1B-7B-Instruct",
}


def load(path: str, n_layers: int):
    """Load a model truncated to `n_layers` decoder layers.

    Truncating keeps compile time sane. It is sound for this purpose because
    decoder layers are structurally identical, so a chain that appears in layer
    0 appears in all of them -- we scale counts by the real depth instead of
    compiling all of it.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    text_cfg = getattr(cfg, "text_config", cfg)
    real_layers = getattr(text_cfg, "num_hidden_layers", None)
    if n_layers and real_layers:
        text_cfg.num_hidden_layers = min(n_layers, real_layers)

    model = AutoModelForCausalLM.from_config(
        cfg, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    return model.eval(), real_layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    path = MODELS[a.model]
    t0 = time.time()
    model, real_layers = load(path, a.layers)
    model = model.to(a.device)

    ids = torch.randint(0, 1000, (1, a.tokens), device=a.device)
    report = scan_module(model, (ids,))

    report["model"] = a.model
    report["model_path"] = path
    report["layers_compiled"] = a.layers
    report["layers_real"] = real_layers
    report["tokens"] = a.tokens
    report["wall_s"] = round(time.time() - t0, 1)
    report["torch"] = torch.__version__

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))

    print(f"\n===== {a.model} ({a.layers}/{real_layers} layers, {a.tokens} tokens) =====")
    print(f"graphs={report['n_graphs']}  chains={report['n_chains']}  "
          f"bytes_saved={report['total_bytes_saved']/1e6:.1f}MB  wall={report['wall_s']}s")
    print("top chain signatures:")
    for sig, n in list(report["by_signature"].items())[:10]:
        print(f"  x{n:<4} {sig}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
