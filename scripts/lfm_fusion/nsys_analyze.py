#!/usr/bin/env python3
"""Derive LFM2.5 kernel timelines and fusion ceilings from nsys SQLite exports."""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results" / "lfm_fusion" / "nsys"

RUNS = {
    "decode_b1_nograph": {"stage": "decode", "cuda_graph": False, "tokens": 1},
    "decode_b1_graph": {"stage": "decode", "cuda_graph": True, "tokens": 1},
    "prefill_b4x4000_nograph": {
        "stage": "prefill",
        "cuda_graph": False,
        "tokens": 16_000,
    },
    "prefill_b4x4000_graph": {
        "stage": "prefill",
        "cuda_graph": True,
        "tokens": 16_000,
    },
}

LAYER_TYPES = [
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
    "attention",
    "conv",
    "conv",
]

HIDDEN = 2048
Q_SIZE = 2048
K_SIZE = 512
V_SIZE = 512
MOE_INTERMEDIATE = 1792
TOP_K = 4
BF16_BYTES = 2
MOE_LAYERS = 22
CONV_LAYERS = 18
ATTN_LAYERS = 6


@dataclass
class Kernel:
    index: int
    start: int
    end: int
    short_name: str
    demangled_name: str
    correlation_id: int | None
    graph_node_id: int | None
    grid_x: int
    block_x: int
    kind: str
    role: str = ""
    layer: int | None = None

    @property
    def duration_us(self) -> float:
        return (self.end - self.start) / 1_000


@dataclass
class Activity:
    start: int
    end: int
    activity: str
    name: str
    bytes: int = 0
    kernel_index: int | None = None


def classify(name: str) -> str:
    n = name.lower()
    if "fusedaddrmsnormkernel" in n:
        return "fused_add_rmsnorm"
    if "rmsnormkernel" in n:
        return "rmsnorm"
    if "causal_conv1d_update_kernel" in n:
        return "conv_update"
    if "causal_conv1d_fwd_kernel" in n:
        return "conv_fwd"
    if "direct_copy_kernel_cuda" in n:
        if "c10::bfloat16" in n:
            return "copy_bf16"
        if "lambda(int)" in n:
            return "copy_int"
        return "copy"
    if "binaryfunctor" in n and "mulfunctor" in n:
        return "mul"
    if "triton_poi_fused_copy__mul_sum_0" in n:
        return "moe_sum"
    if "moe_sum_reduce_warp_per_token_vec_kernel" in n:
        return "moe_sum"
    if "fused_moe_kernel" in n:
        return "fused_moe"
    if "moe_align_block_size_small_batch_expert_kernel" in n:
        return "moe_align"
    if "moe_align_block_size_kernel" in n:
        return "moe_align"
    if "count_and_sort_expert_tokens_kernel" in n:
        return "moe_count_sort"
    if "topkgatingsigmoid" in n:
        return "topk_sigmoid"
    if "act_and_mul_kernel" in n:
        return "act_and_mul"
    if "batchqkapplyrotary" in n:
        return "rope"
    if "store_kvcache" in n:
        return "store_kv"
    if "prepare_varlen_num_blocks_kernel" in n:
        return "attention_prepare"
    if "flashattnfwdcombine" in n:
        return "attention_combine"
    if "flashattnfwd" in n:
        return "attention"
    if "splitkreduce_kernel" in n:
        return "gemm_splitk_reduce"
    if "nvjet_" in n or "cublaslt::" in n:
        return "gemm"
    if "argmaxops" in n:
        return "argmax"
    return "other"


def load_run(stem: str):
    con = sqlite3.connect(RESULTS / f"{stem}.sqlite")
    rows = con.execute(
        """
        SELECT k.start, k.end, ss.value, sd.value, k.correlationId,
               k.graphNodeId, k.gridX, k.blockX
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds ss ON ss.id = k.shortName
        JOIN StringIds sd ON sd.id = k.demangledName
        ORDER BY k.start
        """
    ).fetchall()
    kernels = [
        Kernel(
            index=i,
            start=row[0],
            end=row[1],
            short_name=row[2],
            demangled_name=row[3],
            correlation_id=row[4],
            graph_node_id=row[5],
            grid_x=row[6],
            block_x=row[7],
            kind=classify(row[3]),
        )
        for i, row in enumerate(rows)
    ]

    activities = [
        Activity(
            start=k.start,
            end=k.end,
            activity="kernel",
            name=k.kind,
            kernel_index=k.index,
        )
        for k in kernels
    ]
    for start, end, byte_count, copy_kind in con.execute(
        "SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"
    ):
        activities.append(
            Activity(start, end, "memcpy", f"memcpy_kind_{copy_kind}", byte_count)
        )
    for start, end, byte_count in con.execute(
        "SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET"
    ):
        activities.append(Activity(start, end, "memset", "memset", byte_count))
    activities.sort(key=lambda x: (x.start, x.end))

    runtime = con.execute(
        """
        SELECT r.start, r.end, r.correlationId, s.value
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r
        JOIN StringIds s ON s.id = r.nameId
        """
    ).fetchall()
    con.close()
    return kernels, activities, runtime


def expect(kernels: list[Kernel], pos: int, kind: str, role: str, layer: int) -> int:
    got = kernels[pos].kind if pos < len(kernels) else "<eof>"
    if got != kind:
        raise RuntimeError(
            f"layer {layer}: expected {kind} for {role} at kernel {pos}, got {got}"
        )
    kernels[pos].role = role
    kernels[pos].layer = layer
    return pos + 1


def parse_layers(kernels: list[Kernel], stage: str):
    pos = next(i for i, k in enumerate(kernels) if k.kind == "rmsnorm")
    layer_ranges = []

    for layer, layer_type in enumerate(LAYER_TYPES):
        start = pos
        norm_kind = "rmsnorm" if layer == 0 else "fused_add_rmsnorm"
        pos = expect(kernels, pos, norm_kind, "operator_norm", layer)

        if layer_type == "conv":
            pos = expect(kernels, pos, "gemm", "conv_in_proj", layer)
            pos = expect(kernels, pos, "mul", "conv_b_gate_mul", layer)
            if stage == "prefill":
                pos = expect(
                    kernels, pos, "copy_bf16", "conv_layout_copy", layer
                )
            if kernels[pos].kind == "copy_int":
                pos = expect(
                    kernels, pos, "copy_int", "conv_cache_indices_cast", layer
                )
            conv_kind = "conv_fwd" if stage == "prefill" else "conv_update"
            pos = expect(kernels, pos, conv_kind, "conv_kernel", layer)
            pos = expect(kernels, pos, "mul", "conv_c_gate_mul", layer)
            pos = expect(kernels, pos, "gemm", "conv_out_proj", layer)
        else:
            pos = expect(kernels, pos, "gemm", "attn_qkv_proj", layer)
            if kernels[pos].kind == "copy_bf16":
                pos = expect(
                    kernels, pos, "copy_bf16", "attn_q_layout_copy", layer
                )
            pos = expect(kernels, pos, "rmsnorm", "attn_q_norm", layer)
            if kernels[pos].kind == "copy_bf16":
                pos = expect(
                    kernels, pos, "copy_bf16", "attn_k_layout_copy", layer
                )
            pos = expect(kernels, pos, "rmsnorm", "attn_k_norm", layer)
            pos = expect(kernels, pos, "rope", "attn_rope", layer)
            pos = expect(kernels, pos, "store_kv", "attn_store_kv", layer)
            pos = expect(
                kernels, pos, "attention_prepare", "attn_prepare", layer
            )
            pos = expect(kernels, pos, "attention", "attn_kernel", layer)
            if kernels[pos].kind == "attention_combine":
                pos = expect(
                    kernels, pos, "attention_combine", "attn_combine", layer
                )
            pos = expect(kernels, pos, "gemm", "attn_out_proj", layer)

        pos = expect(kernels, pos, "fused_add_rmsnorm", "ffn_norm", layer)
        if layer < 2:
            pos = expect(kernels, pos, "gemm", "dense_gate_up", layer)
            pos = expect(kernels, pos, "act_and_mul", "dense_activation", layer)
            pos = expect(kernels, pos, "gemm", "dense_down", layer)
            if kernels[pos].kind == "gemm_splitk_reduce":
                pos = expect(
                    kernels,
                    pos,
                    "gemm_splitk_reduce",
                    "dense_down_splitk_reduce",
                    layer,
                )
        else:
            pos = expect(kernels, pos, "gemm", "moe_router", layer)
            pos = expect(kernels, pos, "topk_sigmoid", "moe_topk", layer)
            pos = expect(kernels, pos, "moe_align", "moe_align", layer)
            if kernels[pos].kind == "moe_count_sort":
                pos = expect(
                    kernels, pos, "moe_count_sort", "moe_count_sort", layer
                )
            pos = expect(kernels, pos, "fused_moe", "moe_up", layer)
            pos = expect(
                kernels, pos, "act_and_mul", "moe_activation", layer
            )
            pos = expect(kernels, pos, "fused_moe", "moe_down", layer)
            pos = expect(kernels, pos, "moe_sum", "moe_sum", layer)

        layer_ranges.append((layer, layer_type, start, pos))

    pos = expect(kernels, pos, "fused_add_rmsnorm", "final_norm", 24)
    while kernels[pos].kind != "gemm":
        kernels[pos].role = "logits_prep"
        kernels[pos].layer = 24
        pos += 1
    pos = expect(kernels, pos, "gemm", "lm_head", 24)
    return layer_ranges, pos


def interval_union_ns(intervals):
    intervals = sorted(intervals)
    if not intervals:
        return 0
    total = 0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def wall_us(stem: str, stage: str) -> float:
    path = RESULTS / f"{stem}_result.jsonl"
    payload = json.loads(path.read_text().strip().splitlines()[-1])
    key = "median_decode_latency" if stage == "decode" else "prefill_latency"
    return payload[key] * 1_000_000


def role_us(kernels: list[Kernel], roles: set[str]) -> float:
    return sum(k.duration_us for k in kernels if k.role in roles)


def role_count(kernels: list[Kernel], roles: set[str]) -> int:
    return sum(1 for k in kernels if k.role in roles)


def candidate_rows(stem, meta, kernels, total_kernel_us, measured_wall_us):
    stage = meta["stage"]
    tokens = meta["tokens"]
    activation_bytes = tokens * HIDDEN * BF16_BYTES
    qk_bytes = tokens * (Q_SIZE + K_SIZE) * BF16_BYTES

    conv_roles = {"conv_b_gate_mul", "conv_c_gate_mul"}
    conv_saved = 4 * activation_bytes * CONV_LAYERS
    if stage == "prefill":
        conv_roles.add("conv_layout_copy")
        conv_saved = 6 * activation_bytes * CONV_LAYERS

    qk_roles = {"attn_q_norm", "attn_k_norm", "attn_rope"}
    qk_saved = 2 * qk_bytes * ATTN_LAYERS
    if stage == "prefill":
        qk_roles |= {"attn_q_layout_copy", "attn_k_layout_copy"}
        qk_saved = 4 * qk_bytes * ATTN_LAYERS

    next_norm_us = 0.0
    for layer in range(2, 24):
        if layer == 23:
            next_norm_us += role_us(
                kernels,
                {"final_norm"},
            )
        else:
            next_norm_us += sum(
                k.duration_us
                for k in kernels
                if k.layer == layer + 1 and k.role == "operator_norm"
            )

    definitions = [
        {
            "candidate": "shortconv_gate_layout_fusion",
            "kernels_removed": role_count(kernels, conv_roles),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": conv_saved,
            "measured_us": role_us(kernels, conv_roles),
            "status": "candidate",
            "assumption": "B*x, prefill layout copy, and C*conv_out move into causal_conv1d",
        },
        {
            "candidate": "fused_qk_layout_norm_rope",
            "kernels_removed": role_count(kernels, qk_roles)
            - ATTN_LAYERS
            - 1,
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": qk_saved - tokens * (8 + 4),
            "measured_us": role_us(kernels, qk_roles),
            "status": "candidate",
            "assumption": "reuse existing fused kernel plus one hoisted int64-to-int32 positions cast",
        },
        {
            "candidate": "moe_activation_in_down_gemm",
            "kernels_removed": role_count(kernels, {"moe_activation"}),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": (
                2
                * tokens
                * TOP_K
                * MOE_INTERMEDIATE
                * BF16_BYTES
                * MOE_LAYERS
            ),
            "measured_us": role_us(kernels, {"moe_activation"}),
            "status": "candidate",
            "assumption": "down GEMM loads gate/up and applies SiLU*up in its input prologue",
        },
        {
            "candidate": "moe_sum_plus_next_rmsnorm",
            "kernels_removed": role_count(kernels, {"moe_sum"}),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": 2 * activation_bytes * MOE_LAYERS,
            "measured_us": role_us(kernels, {"moe_sum"}) + next_norm_us,
            "status": "candidate",
            "assumption": "one kernel reduces top-4 outputs, adds residual, and computes RMSNorm",
        },
        {
            "candidate": "moe_sum_in_down_gemm",
            "kernels_removed": role_count(kernels, {"moe_sum"}),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": (
                2
                * tokens
                * TOP_K
                * HIDDEN
                * BF16_BYTES
                * MOE_LAYERS
            ),
            "measured_us": role_us(kernels, {"moe_sum"}),
            "status": "rejected_atomic_complexity",
            "assumption": "down GEMM atomically accumulates four weighted expert outputs per token",
        },
        {
            "candidate": "rope_plus_kv_store",
            "kernels_removed": role_count(kernels, {"attn_store_kv"}),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": (
                tokens * (K_SIZE + V_SIZE) * BF16_BYTES * ATTN_LAYERS
            ),
            "measured_us": role_us(kernels, {"attn_store_kv"}),
            "status": "low_ceiling",
            "assumption": "reuse existing fused_set_kv_buffer rotary call path",
        },
        {
            "candidate": "topk_plus_moe_alignment",
            "kernels_removed": role_count(
                kernels, {"moe_topk", "moe_align", "moe_count_sort"}
            )
            - MOE_LAYERS,
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": tokens * TOP_K * 4 * MOE_LAYERS,
            "measured_us": role_us(
                kernels, {"moe_topk", "moe_align", "moe_count_sort"}
            ),
            "status": "rejected_launch_only",
            "assumption": "top-k kernel also emits alignment metadata; topk_ids remain materialized",
        },
        {
            "candidate": "hoist_conv_metadata",
            "kernels_removed": max(
                0, role_count(kernels, {"conv_cache_indices_cast"}) - 1
            ),
            "extra_gpu_ops_removed": 34 if stage == "prefill" else 0,
            "hbm_bytes_saved": (
                17 * (20 + 12 * 4) if stage == "prefill" else 17 * 12
            ),
            "measured_us": role_us(kernels, {"conv_cache_indices_cast"}),
            "status": "rejected_default_graph_or_tiny",
            "assumption": "build query_start_loc and int32 cache indices once, not in 18 layers",
        },
        {
            "candidate": "main_rmsnorm_into_consumers",
            "kernels_removed": role_count(
                kernels, {"operator_norm", "ffn_norm", "final_norm"}
            ),
            "extra_gpu_ops_removed": 0,
            "hbm_bytes_saved": (
                2
                * activation_bytes
                * role_count(kernels, {"operator_norm", "ffn_norm", "final_norm"})
            ),
            "measured_us": role_us(
                kernels, {"operator_norm", "ffn_norm", "final_norm"}
            ),
            "status": "rejected_impractical",
            "assumption": "conservative two activation-tensor passes removed per norm",
        },
    ]
    for row in definitions:
        row.update(
            run=stem,
            stage=stage,
            cuda_graph=meta["cuda_graph"],
            total_kernel_us=round(total_kernel_us, 3),
            measured_chain_pct_kernel=round(
                100 * row["measured_us"] / total_kernel_us, 4
            ),
            wall_equivalent_hard_ceiling_pct=round(
                100 * row["measured_us"] / measured_wall_us, 4
            ),
        )
        row["measured_us"] = round(row["measured_us"], 3)
    return definitions


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_summary = []
    layer_summary = []
    role_summary = []
    representative = []
    representative_attention = []
    all_candidates = []

    for stem, meta in RUNS.items():
        kernels, activities, runtime = load_run(stem)
        layer_ranges, model_end = parse_layers(kernels, meta["stage"])
        total_kernel_us = sum(k.duration_us for k in kernels)
        measured_wall_us = wall_us(stem, meta["stage"])

        for i, kernel in enumerate(kernels):
            next_gap = (
                (kernels[i + 1].start - kernel.end) / 1_000
                if i + 1 < len(kernels)
                else ""
            )
            timeline_row = {
                "index": kernel.index,
                "start_us": round((kernel.start - kernels[0].start) / 1_000, 3),
                "duration_us": round(kernel.duration_us, 3),
                "gap_to_next_kernel_us": (
                    round(next_gap, 3) if next_gap != "" else ""
                ),
                "layer": "" if kernel.layer is None else kernel.layer,
                "role": kernel.role,
                "kind": kernel.kind,
                "short_name": kernel.short_name,
                "grid_x": kernel.grid_x,
                "block_x": kernel.block_x,
                "graph_node_id": (
                    "" if kernel.graph_node_id is None else kernel.graph_node_id
                ),
            }
            if kernel.layer == 3:
                representative.append({"run": stem, **timeline_row})
            if kernel.layer == 2:
                representative_attention.append({"run": stem, **timeline_row})

        write_csv(
            RESULTS / f"{stem}_timeline.csv",
            list(timeline_row),
            [
                {
                    "index": k.index,
                    "start_us": round((k.start - kernels[0].start) / 1_000, 3),
                    "duration_us": round(k.duration_us, 3),
                    "gap_to_next_kernel_us": (
                        round((kernels[i + 1].start - k.end) / 1_000, 3)
                        if i + 1 < len(kernels)
                        else ""
                    ),
                    "layer": "" if k.layer is None else k.layer,
                    "role": k.role,
                    "kind": k.kind,
                    "short_name": k.short_name,
                    "grid_x": k.grid_x,
                    "block_x": k.block_x,
                    "graph_node_id": (
                        "" if k.graph_node_id is None else k.graph_node_id
                    ),
                }
                for i, k in enumerate(kernels)
            ],
        )

        activity_rows = []
        for i, activity in enumerate(activities):
            activity_rows.append(
                {
                    "index": i,
                    "start_us": round(
                        (activity.start - activities[0].start) / 1_000, 3
                    ),
                    "duration_us": round((activity.end - activity.start) / 1_000, 3),
                    "gap_to_next_activity_us": (
                        round((activities[i + 1].start - activity.end) / 1_000, 3)
                        if i + 1 < len(activities)
                        else ""
                    ),
                    "activity": activity.activity,
                    "name": activity.name,
                    "bytes": activity.bytes,
                    "kernel_index": (
                        "" if activity.kernel_index is None else activity.kernel_index
                    ),
                }
            )
        write_csv(
            RESULTS / f"{stem}_activities.csv",
            list(activity_rows[0]),
            activity_rows,
        )

        by_role = {}
        for kernel in kernels:
            role = kernel.role or f"unassigned:{kernel.kind}"
            bucket = by_role.setdefault(role, [0, 0.0])
            bucket[0] += 1
            bucket[1] += kernel.duration_us
        for role, (calls, duration_us) in sorted(by_role.items()):
            role_summary.append(
                {
                    "run": stem,
                    "role": role,
                    "calls": calls,
                    "total_us": round(duration_us, 3),
                    "pct_kernel": round(100 * duration_us / total_kernel_us, 4),
                    "avg_us": round(duration_us / calls, 3),
                }
            )

        runtime_by_corr = {}
        launch_api_us = 0.0
        graph_launch_us = 0.0
        for start, end, corr, name in runtime:
            duration_us = (end - start) / 1_000
            if "LaunchKernel" in name:
                launch_api_us += duration_us
                runtime_by_corr.setdefault(corr, []).append(duration_us)
            if "cudaGraphLaunch" in name:
                graph_launch_us += duration_us

        busy_ns = interval_union_ns([(a.start, a.end) for a in activities])
        span_ns = activities[-1].end - activities[0].start
        run_summary.append(
            {
                "run": stem,
                "stage": meta["stage"],
                "cuda_graph": meta["cuda_graph"],
                "kernels": len(kernels),
                "kernel_us": round(total_kernel_us, 3),
                "gpu_activities": len(activities),
                "gpu_busy_union_us": round(busy_ns / 1_000, 3),
                "gpu_span_us": round(span_ns / 1_000, 3),
                "device_idle_between_activities_us": round(
                    (span_ns - busy_ns) / 1_000, 3
                ),
                "kernel_launch_api_us": round(launch_api_us, 3),
                "graph_launch_api_us": round(graph_launch_us, 3),
                "measured_wall_us": round(measured_wall_us, 3),
                "kernel_pct_wall": round(
                    100 * total_kernel_us / measured_wall_us, 3
                ),
            }
        )

        for layer, layer_type, start, end in layer_ranges:
            first = kernels[start].start
            last = kernels[end - 1].end
            layer_activities = [
                a for a in activities if a.start >= first and a.end <= last
            ]
            layer_busy_ns = interval_union_ns(
                [(a.start, a.end) for a in layer_activities]
            )
            layer_launch_us = 0.0
            for kernel in kernels[start:end]:
                layer_launch_us += sum(
                    runtime_by_corr.get(kernel.correlation_id, [])
                )
            layer_summary.append(
                {
                    "run": stem,
                    "layer": layer,
                    "layer_type": layer_type,
                    "kernels": end - start,
                    "kernel_us": round(
                        sum(k.duration_us for k in kernels[start:end]), 3
                    ),
                    "span_us": round((last - first) / 1_000, 3),
                    "device_idle_us": round(
                        ((last - first) - layer_busy_ns) / 1_000, 3
                    ),
                    "kernel_launch_api_us": round(layer_launch_us, 3),
                }
            )

        all_candidates.extend(
            candidate_rows(
                stem, meta, kernels, total_kernel_us, measured_wall_us
            )
        )

    write_csv(RESULTS / "run_summary.csv", list(run_summary[0]), run_summary)
    write_csv(RESULTS / "layer_summary.csv", list(layer_summary[0]), layer_summary)
    write_csv(RESULTS / "role_summary.csv", list(role_summary[0]), role_summary)
    launch_summary = []
    for run in RUNS:
        layers = [row for row in layer_summary if row["run"] == run]
        full = next(row for row in run_summary if row["run"] == run)
        launch_summary.append(
            {
                "run": run,
                "model_layer_kernels": sum(row["kernels"] for row in layers),
                "model_layer_kernel_us": round(
                    sum(row["kernel_us"] for row in layers), 3
                ),
                "model_layer_span_sum_us": round(
                    sum(row["span_us"] for row in layers), 3
                ),
                "model_layer_device_idle_sum_us": round(
                    sum(row["device_idle_us"] for row in layers), 3
                ),
                "device_idle_avg_per_layer_us": round(
                    sum(row["device_idle_us"] for row in layers) / len(layers), 3
                ),
                "model_layer_launch_api_sum_us": round(
                    sum(row["kernel_launch_api_us"] for row in layers), 3
                ),
                "launch_api_avg_per_layer_us": round(
                    sum(row["kernel_launch_api_us"] for row in layers)
                    / len(layers),
                    3,
                ),
                "full_capture_device_idle_us": full[
                    "device_idle_between_activities_us"
                ],
                "full_capture_kernel_launch_api_us": full[
                    "kernel_launch_api_us"
                ],
                "full_capture_graph_launch_api_us": full["graph_launch_api_us"],
            }
        )
    write_csv(
        RESULTS / "launch_overhead_summary.csv",
        list(launch_summary[0]),
        launch_summary,
    )
    write_csv(
        RESULTS / "representative_layer3_timeline.csv",
        list(representative[0]),
        representative,
    )
    write_csv(
        RESULTS / "representative_attention_layer2_timeline.csv",
        list(representative_attention[0]),
        representative_attention,
    )
    write_csv(
        RESULTS / "candidate_summary.csv",
        list(all_candidates[0]),
        all_candidates,
    )
    print(f"Wrote nsys analysis CSVs to {RESULTS}")


if __name__ == "__main__":
    main()
