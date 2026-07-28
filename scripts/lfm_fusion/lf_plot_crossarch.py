#!/usr/bin/env python3
"""Cross-architecture audit figure: the gap tracks model family, not novelty."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lf_lib as L

FAMILY = {"qwen": "Qwen", "qwen06": "Qwen", "qwen3next": "Qwen",
          "gemma3": "Google", "lfm25": "Liquid"}
LABEL = {"qwen": "Qwen3-30B\nMoE, mature", "qwen06": "Qwen3-0.6B\ndense, very mature",
         "qwen3next": "Qwen3-Next\nlinear-attn MoE, NEW",
         "gemma3": "Gemma-3-1B\ndense, mature", "lfm25": "LFM2.5-8B\nshort-conv MoE, new"}
COLOUR = {"Qwen": "#2c7fb8", "Google": "#c0392b", "Liquid": "#e07b39"}


def main():
    rows = []
    for f in sorted(glob.glob(str(L.RESULTS / "audit" / "*_A_low_batch_decode" / "audit.json"))):
        if "patched" in f:
            continue
        d = json.loads(Path(f).read_text())
        v = d["stages"].get("decode")
        if not v:
            continue
        g = {x["gap"]: x for x in v["fusion_gaps"]}
        rows.append(dict(model=d["model"],
                         total=sum(x["pct_of_kernel_time"] for x in g.values()),
                         eager=sum(g[k]["pct_of_kernel_time"] for k in
                                   ("eager_norm_decomp", "eager_norm_rsqrt",
                                    "eager_norm_pow") if k in g)))
    rows.sort(key=lambda r: -r["total"])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = range(len(rows))
    bars = ax.bar(x, [r["total"] for r in rows],
                  color=[COLOUR[FAMILY[r["model"]]] for r in rows])
    ax.bar(x, [r["eager"] for r in rows], color="black", alpha=0.28,
           label="of which: RMSNorm running as eager PyTorch")
    for i, r in enumerate(rows):
        ax.text(i, r["total"] + 1.0, f"{r['total']:.2f}%", ha="center",
                fontsize=10, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABEL[r["model"]] for r in rows], fontsize=8.5)
    ax.set_ylabel("fusion-gap kernels, % of decode CUDA kernel time")
    ax.set_ylim(top=max(r["total"] for r in rows) * 1.25)
    ax.set_title("The gap tracks MODEL FAMILY, not architecture novelty\n"
                 "three Qwen models span dense / MoE / linear attention and "
                 "mature / new — all clean", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOUR[f]) for f in
               ("Qwen", "Google", "Liquid")]
    ax.legend(handles + [plt.Rectangle((0, 0), 1, 1, color="black", alpha=0.28)],
              ["Qwen family", "Google (Gemma-3)", "Liquid (LFM2.5)",
               "of which: eager-PyTorch RMSNorm"], fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    out = L.RESULTS / "plots"
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out / f"cross_architecture_gaps.{ext}", dpi=150,
                    bbox_inches="tight")
    print(f"wrote {out/'cross_architecture_gaps.png'}")


if __name__ == "__main__":
    main()
