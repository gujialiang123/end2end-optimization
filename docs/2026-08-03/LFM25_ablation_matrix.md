# LFM2.5-8B-A1B 优化消融矩阵（交付主表）

**写于**：2026-08-03 · **状态**：部分格子待补，见 §5
**配套文档**：`LFM25_FINAL_CASE_full_record.md`（详细过程）、`exp3_kernel_on_tuned_baseline.md`（L2×L3 实验）

---

## 0. 三个优化层

| 代号 | 名称 | 改什么 | 是否改代码 |
|---|---|---|:--:|
| **L1** | Serving config tuning | 4 个 serving 旋钮（`max_running_requests` / `chunked_prefill_size` / `schedule_policy` / `mem_fraction_static`） | ❌ |
| **L2** | Kernel config tuning | `fused_moe_kernel` 的 tile 参数（`BLOCK_SIZE_M/N/K`、`GROUP_SIZE_M`、`num_warps`、`num_stages`） | ❌ |
| **L3** | Kernel rewrite / fusion | 7 处模型代码改动，含 **4 个手写 Triton kernel** | ✅ |

---

## 1. 主表 A —— 单项效果（每个改动**单独**加在 cookbook 基线上）

单位：request throughput 的相对变化。`n.s.` = 不显著。

| regime | workload | baseline (req/s) | **L1** serving | **L2** kernel config | **L3** kernel rewrite |
|---|---|---:|---:|---:|---:|
| **A** 低批 decode | in=100, out=256, conc=1 | 1.688 | +0.38% ⚠️ | **+0.05%** (p=0.34 n.s.) | **+6.70%** (p=2.1e-41) |
| **B** 并发 decode | in=200, out=256, conc=32 | 21.67 | +1.11% ⚠️ | +0.5% (n=8, 未做顺序对照) | **+6.21%** (p=2.4e-08) ⚠️ |
| **C** 长 prefill | in=4000, out=32, conc=4 | 12.12 | **+56.94%** ⚠️ | **+23.26%** (p=1.1e-33) | **+6.18%** (p=4.5e-13) |

⚠️ **口径警告**：
- **L1 那一列来自另一个 campaign**（`2026-07-24_serving_ceiling_validation`），它自己的
  cookbook 基线是 A=1.681 / B=21.990 / C=12.604，与 L2/L3 那两列的基线（12.119）**不是同一次测量**。
  **只能比 ratio，不能比绝对值。** 完整内部一致的 C 数据要等实验 5（§5）。
- **B 的 L3 是脏基线上的旧数字**（n=6，未做顺序对照，7/27）。A 和 C 已用 n=16 counterbalanced 重测。

**L1 每个 regime 的最优旋钮**（验证 pass，35 配置 × 5 重复）：

| regime | 最优旋钮 | cookbook | ceiling | 提升 |
|---|---|---:|---:|---:|
| A 低批 decode | `cap8 · chunk−1 · fcfs · mem0.85` | 1.6814 ± 0.006 | 1.6878 ± 0.002 | +0.38% |
| B 并发 decode | `cap64 · chunk8192 · fcfs · mem0.75` | 21.990 ± 0.081 | 22.234 ± 0.166 | +1.11% |
| **C 长 prefill** | **`cap8 · chunk2048 · fcfs · mem0.90`** | 12.604 ± 0.382 | **19.781 ± 0.295** | **+56.94%** |
| （medium balanced） | `cap8 · chunk2048 · fcfs · mem0.90` | 7.108 | 7.235 | +1.79% |
| （shared prefix） | `cap96 · chunk2048 · lpm · mem0.90` | 14.081 | 27.262 | +93.61% |
| （tool agent） | `cap128 · chunk8192 · lpm · mem0.75` | 5.264 | 5.280 | +0.31% |

---

## 2. 主表 B —— 累积叠加（waterfall）

叠加顺序：**cookbook → L1 → L2 → L3**（部署时的自然顺序）。

### 2.1 regime C 长 prefill

| 阶段 | 配置 | req/s | vs 上一级 | vs cookbook |
|---|---|---:|---:|---:|
| S0 cookbook | `cap32/chunk−1/lpm/mem0.85`，无 tuned config，原版 kernel | 12.119 ± 0.116 | — | 1.000× |
| S1 + L1 | `cap8/chunk2048/fcfs/mem0.90` | ⬜ | ⬜ | ⬜ |
| S2 + L1 + L2 | 同上 + tuned MoE config | ⬜ | ⬜ | ⬜ |
| **S3 + L1 + L2 + L3** | 同上 + 7 项 kernel 改动 | ⬜ | ⬜ | ⬜ |

> **实验 5 正在跑这四格**（`C_long_prefill_tuned` 的 2×2）。

**已有的替代路径（跳过 L1）**——这一支是完整的：

| 阶段 | 配置 | req/s | vs 上一级 | vs cookbook |
|---|---|---:|---:|---:|
| S0 cookbook | `cap32/chunk−1/lpm/mem0.85` | **12.119 ± 0.116** | — | 1.000× |
| S2′ + L2 | + tuned MoE config | **14.939 ± 0.123** | **+23.26%** (p=1.1e-33) | 1.233× |
| **S3′ + L2 + L3** | + 7 项 kernel 改动 | **16.392 ± 0.200** | **+9.73%** (p=9.5e-19) | **1.352×** |
| （对照）S3″ 只有 L3 | 无 tuned config + 7 项 | 12.869 ± 0.182 | +6.18% (p=4.5e-13) | 1.062× |

**每格 n=16**（2 个顺序 × 8 重复，counterbalanced，2 个独立 server lifetime）。

### 2.2 regime A 低批 decode

| 阶段 | req/s | vs 上一级 | vs cookbook |
|---|---:|---:|---:|
| S0 cookbook | **1.6863 ± 0.0027** | — | 1.000× |
| S2′ + L2 | 1.6872 ± 0.0025 | **+0.05%** (p=0.34 n.s.) | 1.001× |
| **S3′ + L2 + L3** | **1.7944 ± 0.0037** | **+6.35%** (p=1.8e-34) | **1.064×** |
| （对照）只有 L3 | 1.7992 ± 0.0024 | +6.70% (p=2.1e-41) | 1.067× |
| S1 + L1 | ⬜ | ⬜ | ⬜ |
| S3 全三层 | ⬜ | ⬜ | ⬜ |

> L2 的 guarded 策略对 `M ≤ 32` 逐字段保持默认，CUDA graph 捕获的 decode batch 全落在那一段
> → **L2 在 decode 上按设计就该中性，实测证实。**

### 2.3 regime B 并发 decode

| 阶段 | req/s | vs 上一级 | vs cookbook |
|---|---:|---:|---:|
| S0 cookbook | 21.673 | — | 1.000× |
| S2′ + L2 | ⬜（07-26 研究测得 1.005×，n=8，未做顺序对照） | ⬜ | ⬜ |
| **S3′ + L2 + L3** | ⬜ | ⬜ | ⬜ |
| （对照）只有 L3 | 23.018 | +6.21% (p=2.4e-08) ⚠️ n=6 | 1.062× |
| S1 + L1 | ⬜ | ⬜ | ⬜ |
| S3 全三层 | ⬜ | ⬜ | ⬜ |

---

## 3. 主表 C —— L3 内部逐组件

### 3.1 在 cookbook 基线上（7/27，n=6，各自配对基线）

| 组件 | 类型 | A 低批 decode | B 并发 decode | C 长 prefill |
|---|---|---:|---:|---:|
| `norm` | 接线 | +2.35% | +2.89% | +1.42% (p=2e-04) |
| `scale` | 接线 | +1.40% | +1.02% (p=0.0048) | +0.73% (p=0.24 **n.s.**) |
| `norm+scale` | 接线 | +4.20% (p=2e-07) | +3.68% (p=4.7e-06) | +1.60% (p=0.009) |
| `conv` | **手写 Triton ×2** | +0.13% (p=0.22 **n.s.**) | −0.03% (p=0.95 **n.s.**) | **+2.33%** (p=0.0015) |
| `norm+scale+conv` | 混合 | +3.89% (p=2.5e-15) | +3.65% (p=6.1e-06) | +3.47% (p=9.5e-04) |
| `qkrope` | 接线 | +0.93% (p=7.2e-09) | **+5.42%** (p=1.6e-07) | +1.99% (p=0.018) |
| `gate+idx` | Triton×1 + 缓存 | −0.00% (p=0.97 **n.s.**) | +0.65% (p=0.12 **n.s.**) | +0.40% (p=0.54 **n.s.**) |
| `moesum` | **手写 Triton ×1** | **+4.55%** (p=1.5e-13) | +3.08% (p=0.0032) | ⬜ 未单独测 |
| **六项**（不含 `moesum`） | | +4.60% / +4.74% ¹ | +5.54% / +6.01% ¹ | +5.12% / +5.81% ¹ |
| **七项全开** | | **+6.57%** (p=4.6e-14) | **+6.21%** (p=2.4e-08) | **+5.30%** (p=1.2e-05) |

¹ 六项臂在两个 campaign 各测过一次，配对基线不同，两个值都列出。

**四种不同形状的收益**（这是「必须分 regime 测」的直接证据）：
- `norm+scale` 消除**每 forward 固定数量**的 kernel → decode 占比大，长 prefill 被稀释
- `conv` 消除**随 token 增长**的流量，需 T≥2048 → **只有长 prefill 够得到**
- `qkrope` 消除 6 个注意力层的工作 → **并发 decode 最受益**
- `moesum` 消除 launch + HBM 往返 → **小 T 最赚，低批 decode 最受益**

### 3.2 在 L2 tuned config 基线上（exp3，n=16，counterbalanced）

| 组件 | A 低批 decode | B 并发 decode | C 长 prefill |
|---|---:|---:|---:|
| 六项 | ⬜ | ⬜ | **+8.47%** (t=17.3) |
| `moesum` 边际 | ⬜ | ⬜ | **+1.69%** (p=2.8e-04) |
| **七项全开** | **+6.35%** (p=1.8e-34) | ⬜ | **+9.73%** (p=9.5e-19) |
| 其余 6 个单项 | ⬜ | ⬜ | ⬜ |

> ★ **`moesum` 的符号在换基线后翻了**：脏基线 −0.08%（p=0.88 不显著）→ 干净基线 **+1.69%（p=2.8e-04）**。
> **所以 §3.1 那张表的归因在干净基线上全部未经验证。**

---

## 4. 每个改动具体是什么

### L1 — Serving config tuning（不改代码）

只调 4 个启动参数。搜索方式是 **8×3×2×4 = 192 全网格穷举**（不是采样，无采样偏差），
再对前 35 个配置做 5 重复验证 pass。

| 旋钮 | 取值 |
|---|---|
| `max_running_requests` | 8, 16, 24, 32, 48, 64, 96, 128 |
| `chunked_prefill_size` | −1, 2048, 8192 |
| `schedule_policy` | lpm, fcfs |
| `mem_fraction_static` | 0.75, 0.80, 0.85, 0.90 |

**结论是分裂的**：3/6 regime 是 plateau（+0.3~1.1%），2/6 有断崖（长 prefill +56.9%、
shared-prefix +93.6%，但都是 TRADE-OFF——TTFT/TPOT 有代价）。
**下行风险比上行大一个数量级**（最坏 −64.9%）→ serving 旋钮是避坑杠杆不是提速杠杆。

### L2 — Kernel config tuning（不改代码）

给上游已有的 `fused_moe_kernel` 换 tile 参数。LFM2.5 的 MoE shape 是 **`E=32, N=1792`**；
上游 PR #22791 已为 LFM2 做过 H100/B200/MI325X，**唯独没有 H200**，
所以较大 prefill shape 全部落到两档启发式（server log 自己打
`Performance might be sub-optimal!`）。

- 每个 token-count 桶扫 **468–894 个候选**，19 个桶（与上游 H100/B200 文件对齐）
- 每个候选**先过正确性门禁再计时**：~9000 个配置，**0 次正确性失败**
- **guarded 策略**：`M ≤ 32` 逐字段等于默认 → decode 路径行为不变

### L3 — Kernel rewrite / fusion（改代码）

7 处改动，分两类：

#### 接线修复（融合原语**早已存在**，调用点没用）

| 组件 | 问题 | 修复 |
|---|---|---|
| **`norm`** | 层 forward 收了 `residual` 参数**第一行就覆盖掉**，导致 RMSNorm 走非融合分支，两个 add 各起一个 kernel | 改成 llama/qwen2 的 deferred-residual 写法，残差当"欠账"传给下一层由 norm kernel 顺手结清。**每层省 2 个 × 24 层 = 48 个 kernel** |
| **`qkrope`** | `sgl_kernel.fused_qk_norm_rope` 把 2 个 head-wise RMSNorm + RoPE 合成一个 in-place CUDA kernel，**Qwen3-MoE 已在调用**，LFM2.5 跑三个独立 kernel（decode 1.65% / prefill 3.61% kernel 时间） | 在 packed QKV 上直接调融合 kernel，守卫 bf16 + head_dim=64 + 无 rope scaling |
| **`scale`** | `config.json` 里 `routed_scaling_factor: 1.0`，但代码无条件乘 → 每 forward **22 个 kernel 把整个 `[T,2048]` 读一遍、乘 1、写回** | 等于 1.0 时跳过。**bit-exact** |
| **`idx`** | `req_pool_indices.to(int32)` 在**每个 conv 层**重算（18 次/forward），只搬 12 字节，纯 launch 开销，占低批 decode ~1.3% | 按 forward 缓存，用源张量 identity 作 key 防陈旧 |

#### 手写 Triton kernel

| 组件 | 问题 | 修复 | 隔离效果 |
|---|---|---|---|
| **`conv`**（2 个 kernel） | `causal_conv1d_fn` 要求 `[dim,seqlen]` 且末维 stride=1，**转置躲不掉只能被吸收**。`Bx.transpose().contiguous()` 和 `C_gate*conv_out` 的转置读**都不合并** → 18 层搬 8.79 GB 用 10.3 ms = **0.83 TB/s，仅峰值 17%** | conv 两侧各一个 tiled kernel，把 chunk + gating mul + transpose 折叠进一趟，转置用 `tl.trans` **在寄存器/共享内存里做** | T=16000 时 **5.93× / 4.33×**，**0.98 → 3.46 TB/s（17%→72% 峰值）**，全部 bit-exact。T<2048 门控回退 |
| **`moesum`**（1 个 kernel） | MoE top-k 归约把 `[T,H]` 写回 HBM，**下一层**的 `fused_add_rmsnorm` 立刻读回来，两者都是行方向 → **多跑一整趟 HBM 往返** | `FusedMoE` 返回 4 个未归约专家输出，一个 kernel 做完**归约 + 残差加 + RMSNorm** | T=1 **2.46×**、T=8 **2.68×**、T=16000 1.30×；但 T=128~1024 是 **0.72~0.74×**（输）→ 门控 `T≤32 或 T≥4096` |
| **`gate`**（1 个 kernel） | decode 的 `B_gate * x` 读 `proj` 的**跨步行**——是合并访问，但跨步行让 `TensorIterator` **无法向量化**，退化成标量 `elementwise_kernel`（由 trace 里的 kernel 名确认），只到 54% 峰值 | Triton kernel 直接读 `proj` 绕开 | bit-exact |

> **`conv` 和 `moesum` 的形状依赖正好相反**：`conv` 要大 T 才能摊掉 ~30µs 的 Triton launch 地板，
> `moesum` 省的就是 launch + 往返所以小 T 最赚。**两者合起来覆盖全范围。**

---

## 5. 实验配置

### 5.1 固定框架

```
模型      /data/hf/LFM2.5-8B-A1B   (bf16, TP=1)
硬件      1× NVIDIA H200, driver 580.105.08
软件      sglang 0.5.12.post1 @ 17f7a1da1
          torch 2.9.1+cu128 · Triton 3.5.1 · CUDA 12.8
conda     sglang-dev
客户端    sglang.bench_serving (streaming, --output-details)
```

### 5.2 cookbook 基线的完整启动命令

```bash
python -m sglang.launch_server \
    --model-path /data/hf/LFM2.5-8B-A1B \
    --served-model-name lfm2.5-8b-a1b \
    --host 127.0.0.1 --port <PORT> \
    --tensor-parallel-size 1 \
    --context-length 8192 \
    --schedule-conservativeness 1.0 \
    --trust-remote-code \
    --moe-runner-backend auto \
    --mem-fraction-static 0.85 \
    --max-running-requests 32 \
    --chunked-prefill-size -1 \
    --schedule-policy lpm \
    --max-prefill-tokens 16384
```

**从真实 server log 逐条核实的 resolved args**（不是从代码推断）：

| 参数 | 值 |
|---|---|
| **`disable_cuda_graph`** | **False** ← **CUDA graph 开着** |
| `cuda_graph_max_bs` | 256；实际捕获 `bs [1,2,4,8,12,16,24,32]` |
| `enable_torch_compile` | False |
| `enable_piecewise_cuda_graph` | False |
| `disable_radix_cache` | False（radix cache 开） |
| `disable_overlap_schedule` | False（overlap 调度开） |
| `enable_fused_qk_norm_rope` | **False** ← 上游后加的 server 级开关，默认关 |
| `attention_backend` | `fa3` |
| `moe_runner_backend` | `auto` |
| `dtype` / `kv_cache_dtype` | `auto` / `auto`（bf16） |
| `quantization` / `speculative_algorithm` | `None` / `None` |
| `page_size` | 1 |

> 捕获的 batch size 到 32 为止 = `max_running_requests`
> → **decode 路径全程 graph 重放，所有臂都是。**

### 5.3 A/B 方法

- **L3 切换**：`LFM_FUSION_PATCH` 环境变量。不设 = **逐字未改动的 sglang 原路径**，
  同一棵树、同一 commit、同一份 server 参数。
- **L2 切换**：`SGLANG_MOE_CONFIG_DIR` 环境变量。
- **顺序对照**：`lf_e2e.py` / `rk_e2e.py` 顺序执行 arm，**存在位置效应**
  （实测 regime C baseline 正序 12.020 / 逆序 12.219，**1.7%，比要测的效应一半还大**）。
  → exp3 之后所有实验都做 `{正序, 逆序}` counterbalance，合并后 n=16。
- **生效检查**：server log 必须出现 patch 标记 / config 加载行，否则静默失效的 patch
  会被误记为"与 baseline 相同"。
- **统计**：Welch t + **精确 Student-t 尾**（正态近似在 n=6 下 anti-conservative）。

### 5.4 正确性

- **token-identity 对这个模型结构性不可用**：top-4/32 路由，专家选择是离散 argmax，
  bf16 级扰动会翻转选中的专家 → 任何非 bit-identical 改动都触发。
- 改用 **GSM8K 全量 1319 题、贪心**。
- **用 bit-exact 的 `scale` 臂免费标定噪声底**：它数学上必然等于 baseline，
  却读数低 **0.8 点** → between-arm 系统噪声 ≥ 0.8 点。
  8 个臂跨度 2.5 点，在三个噪声度量下都在噪声内。
- **口径：未检测到质量回归。** 不是"质量提升"。

### 5.5 已知的实验瑕疵（必须披露）

1. **7/27 那批 e2e 全部带 `--skip-correctness`** —— `correctness.json` 里 `outputs: []`。
   正确性证据只来自**单独跑的 GSM8K campaign**（且那是 cookbook 基线，无 tuned config）。
2. **sglang 工作树有一处未提交改动**（`model_runner.py` 的 flashinfer_cutlass autotune
   allowlist 补丁，2026-06-11）。两臂共享**不影响 A/B**，但严格说 baseline 是
   "17f7a1da1 + 那一处补丁"。
3. **L2 自己没到 ceiling**：server log 明写
   `Config file not found ... E=32,N=1792,device_name=NVIDIA_H200_down.json`
   → **down projection 仍跑默认启发式。**
4. **7/27 与 exp3 对同一件事的读数不同**：C 长 prefill 上 L3 单独，
   7/27 = **+5.30%**（n=6），exp3 = **+6.18%**（n=16, counterbalanced, 修了泄漏 server 的坑）。
   **exp3 更可信，但报告里那张表还是 +5.30%**，交付前要统一。

---

## 6. ⬜ 空格清单 —— 请指示补哪些

| # | 空格 | 影响哪张表 | 预计 | 备注 |
|---|---|---|---|---|
| **1** | regime **B** 在 tuned config 上的 L2 / L3 | 表 2.3 整行 | ~30 min | 头条 3 个 regime 里唯一没有干净基线的 |
| **2** | L3 **逐组件**在 tuned config 上重测 | 表 3.2 整块 | ~2-3h / regime | ★ `moesum` 已证明会翻符号，其余 6 项归因**全部未验证** |
| **3** | `moesum` 在 **C 长 prefill** 单独测 | 表 3.1 那个 ⬜ | ~30 min | 7/27 从没单独测过 |
| **4** | **L1 串联**（S1 / S2 / S3 三格） | 表 2.1 上半、2.2、2.3 | 进行中 | **实验 5 正在 GPU 4 上跑 regime C** |
| **5** | 补 `_down.json` 后重测 L2 和 L3 增量 | 表 2.1 下半全部 | ~1-2h | ⚠️ **可能让 L3 的 +9.73% 缩水**，但不做则"超过 best autotuning"站不住 |
| **6** | 交付配置（L2+L3）的 **GSM8K** | §5.4 | ~2h | 现有正确性证据不覆盖 tuned config |
| **7** | `moesum` × tuned config 超可加的 **profile 证据** | §3.2 的解释 | ~1h | 现在只有推测机制。nsys 四格（`no_combine` 开关 × config 开关） |
| **8** | 最终 stack 上的 **NCU**（剩余 headroom） | 新增一节 | ~1h | Mason 证据链第 4 步，锦上添花 |

### 不需要补的

- **~~硬化 L1 的 autotuning ceiling~~** —— 已经是 **192/192 全网格穷举**
  （`docs/2026-07-24/qwen_serving_ceiling_methodology.md:36`，原文
  "the whole space is enumerated, so no sampling bias exists at all"）+ 35 配置 × 5 重复验证。
  "25 次 TPE 搜索失败"那个批评**只适用于已被取代的 06-30 研究**。

---

## 7. 数据来源

| 内容 | 路径 |
|---|---|
| L1 逐 regime ceiling | `results/2026-07-24_serving_ceiling_validation/analysis/lfm25/ceiling_per_regime.json` |
| L1 全网格（192 配置） | `results/2026-07-24_serving_ceiling/` |
| L1 收敛研究（100 trial 无热启动） | `results/2026-07-22_lfm25_plateau_100/` |
| L2 | `results/regime_kernel/` |
| L3 逐组件（配对基线） | `results/lfm_fusion/processed/fusion_ab*.csv` |
| L2×L3（exp3） | `results/lfm_fusion/e2e/exp3_layered_*_summary.json` |
| `moesum` 边际 | `results/lfm_fusion/e2e/exp3_moesum_marginal_C_long_prefill.json` |
| L1 串联（exp5，进行中） | `results/lfm_fusion/e2e/lfm25_exp3_l1_C_*/` |
| kernel 时间构成（nsys） | `results/lfm_fusion/nsys/FINDINGS.md` |
| NCU | `results/2026-07-10_v9_ncu_realworkload/` |
| GSM8K | `results/lfm_fusion/correctness/accuracy_*.json` |
