# Triton 3.6 重扫：编译器升级吃掉了我们 tuning 的全部收益

**日期**：2026-07-29（通宵跑）· **GPU**：H200 ×4（0/3/6/7）
**模型**：LFM2.5-8B-A1B（E=32, N=1792, top_k=4）· BF16 · TP1
**结论一句话**：**Triton 3.6 的默认路径（+29.8%）比我们在 3.5.1 上手工 tuning（+23.3%）更快。这个优化机会已经被上游编译器消化掉了。**

---

## 0. 为什么要做这件事

PR [#32687](https://github.com/sgl-project/sglang/pull/32687) 提交的是 **Triton 3.5.1** 上 tune 的 config，而上游 main 现在 pin `torch==2.11.0` → **Triton 3.6.0**。
上游源码里就写着这个风险：

> *updating the Triton version might cause all old configs to become suboptimal*

所以必须实测：在 3.6 上重扫，看结论是否还成立。

---

## 1. 主结果：3.6 上没有 tuning 空间

19 个桶全扫（每桶 468–906 个候选，`warmup=25 iters=100 repeats=5`），guarded 阈值 1.15×：

```
0/19 buckets specialised (threshold 1.15x)
```

**一个都没有。** 对照 3.5.1：

| M | 3.5.1 上 tuning 收益 | **3.6 上 tuning 收益** | 3.6默认 / 3.5.1默认 |
|---:|---:|---:|---:|
| 24 | 1.078× | 1.061× | 1.107× |
| 48 | **1.372×** | 1.062× | **1.314×** |
| 96 | **1.391×** | 1.004× | **1.409×** |
| 1536 | **1.561×** | 0.956× | **1.606×** |
| 3072 | **1.626×** | 0.997× | **1.699×** |

**最右列几乎精确等于中间列。** 也就是说：

> **Triton 3.6 编译器自己拿到的加速，正好就是我们在 3.5.1 上手工 tuning 拿到的那部分。**

原始数据：`results/regime_kernel/processed/triton_version_comparison.csv`、`raw/t36/`

---

## 2. 端到端确认

C 长 prefill（in=4000, out=32, conc=4），只有 `SGLANG_MOE_CONFIG_DIR` 在变：

| | default | 我们 tune 的 config | tuning 收益 |
|---|---:|---:|---:|
| **Triton 3.5.1** | 12.277 req/s | **15.142** | **+23.3%** |
| **Triton 3.6.0** | **15.932** | 15.885 | **−0.3%（中性）** |

三个关键比值：

- 升级 Triton（默认路径）：**+29.8%**
- 在 3.5.1 上 tuning：**+23.3%**
- **3.6 的 default 比 3.5.1 的 tuned 还快 5.2%**

**升级编译器比我们手工 tuning 更有效，而且是免费的。**

原始数据：`results/regime_kernel/processed/triton_version_e2e.csv`

---

## 3. 那 PR 还要不要？—— 要，但必须改口径

关键问题：3.6 用户会通过既有的跨版本 fallback 拿到我们的 3.5.1 config，**那会不会伤害他们？**

实测（在 3.6 上给它我们的 3.5.1 config）：

| M | 48 | 96 | 256 | 1024 | 2048 | 8192 |
|---|---:|---:|---:|---:|---:|---:|
| 相对 3.6 默认 | 1.014× | 0.999× | 0.996× | 1.010× | 1.000× | 0.991× |

端到端 **0.997×（−0.30%）**。**中性，不伤害。**

所以：

| 用户 | 本 PR 的价值 |
|---|---|
| Triton 3.5.1（仓库里还有 121 个该目录的文件） | **+23.3%**，真实收益 |
| Triton 3.6（= 当前 main） | **中性**，不受益也不受害；最佳行动是升级 Triton |

PR 保留，但正文必须如实写明这一点——**不能让维护者以为它对 main 的主流用户有 23% 的收益**。

---

## 4. 方法论：这条教训比数字本身更值钱

### 4.1 优化机会有保质期

我们在 2026-07-26 测到 1.44–1.64× 的 kernel 空间，那时是真的。三天后换个 Triton 版本，**同一个空间归零**。

这不是测错了，是**空间本身被上游消化了**。

对 final agent 的直接含义：

> **任何"发现的优化机会"都必须附带它被验证时的工具链版本。跨版本重新验证是必须的，不是可选的。**
> **而且要验证的不只是"我的改动还在不在"，还有"这个机会本身还在不在"。**

这和 Gemma-3 那次是同一类错误的两种形态：
- Gemma-3：**上游把我的修复做了**（#32383 抢先落地 2-D 路径）
- 本次：**上游把问题消灭了**（编译器改进让 tuning 无意义）

两次都是"我以为的空缺已经不在了"。

### 4.2 先查最便宜的杠杆

如果一个用户来问"我的 LFM2.5 在 H200 上慢"，正确的第一个回答是 **"升级 Triton"**（+29.8%，零风险，零维护），而不是 "我给你 tune 一个 config"（+23.3%，需要 2 小时 GPU，且会过期）。

**agent 的候选优化列表里应该把"升级依赖"排在"手工优化"前面。**

### 4.3 负面结果必须同等对待

这次的结论是"我们的工作在新版本上没价值了"。它照样被完整记录、数据照样上传、PR 照样如实更新。

如果只记录成功的实验，方法论就是被污染的。

---

## 5. 工程记录：为了在 3.6 上跑起来做了什么

三个都是真实障碍，记下来免得重踩：

1. **模块路径变了**。0.5.12 的 `layers.moe.fused_moe_triton` 在 main 上是 `layers.moe.moe_runner.triton_utils`。
   → `rk_microbench.py` 改成两种布局都试。

2. **main 需要 TP group**。standalone 进程不建 TP group，而 main 的 MoE 路径会去拿它（symmetric-memory 检查），直接 assert 失败。
   → 补 `init_distributed_environment(world_size=1)` + `initialize_model_parallel(1)`。

3. **CUDA 工具链不在 env 根目录**。`gemma-sglang` 是 torch **cu130**，但 env 根下没有 nvcc；用 `sglang-dev` 的 nvcc 12.8 去编，链接阶段 `cannot find -lcudart`。
   → 真正的 toolchain 在 `site-packages/nvidia/cu13/{bin,include,lib}`。把 `CUDA_HOME` 指到那里，并把 `lib` 加进 `LD_LIBRARY_PATH`/`LIBRARY_PATH`。
   → `serving_ceiling_lib.launch_server` 和 `rk_lib.run_env` 都加了「CUDA_HOME 可独立于 env 根设置」。

**这条 3 是之前"两 env 工具链混用导致 JIT 链接失败"的真正根因**，当时绕过去了，这次找到了。

---

## 6. 复现

```bash
ENVDIR=$HOME/.conda/envs/gemma-sglang           # torch 2.11 + triton 3.6
CU13=$ENVDIR/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CU13 PATH=$CU13/bin:$ENVDIR/bin:$PATH
export LD_LIBRARY_PATH=$CU13/lib LIBRARY_PATH=$CU13/lib
export PYTHONPATH=/tmp/sglang_lfm/python        # main 的源码
export RK_TP_PORT=29601

# 单桶扫描
python scripts/regime_kernel/rk_microbench.py --model lfm25 --tokens 3072 \
  --out results/regime_kernel/raw/t36/sweep_M3072.json --warmup 25 --iters 100 --repeats 5

# 从扫描结果构建 config（guarded 阈值）
python scripts/regime_kernel/rk_build_config.py \
  --sweeps results/regime_kernel/raw/t36 --out cfg.json \
  --report results/regime_kernel/processed/t36_buckets.csv
```

## 7. 产物

| 文件 | 内容 |
|---|---|
| `results/regime_kernel/raw/t36/` | 19 个桶的完整 3.6 扫描（每桶数百候选） |
| `results/regime_kernel/raw/t36_probe/` | 3.5.1 config 在 3.6 上的计时 |
| `results/regime_kernel/processed/t36_buckets.csv` | 每桶 default/best/speedup/是否特化 |
| `results/regime_kernel/processed/triton_version_comparison.csv` | 3.5.1 vs 3.6 kernel 级对照 |
| `results/regime_kernel/processed/triton_version_e2e.csv` | 端到端四格表 |
| `scripts/regime_kernel/rk_build_config.py` | 新增：从扫描结果按 guarded 策略生成 config |


---

## 8. 两个 PR 的当前状态（2026-07-29 06:30）

| PR | 标题 | 状态 |
|---|---|---|
| [#32670](https://github.com/sgl-project/sglang/pull/32670) | fix(gemma3): fuse high-rank RMSNorm and guard mixed-dtype weights | ready for review，2 文件 |
| [#32687](https://github.com/sgl-project/sglang/pull/32687) | [MoE] Add LFM2 MoE tuned config for H200 (Triton 3.5.1) | ready for review，1 文件 |

两个都由 @gujialiang123 在 07-28 22:39 手动转为 ready。

**CI 现状**：所有 check 显示 fail，但**不是代码失败**：

- 转正前：卡在 `Block draft PR`（draft gate，3–7 秒即挂，真正的测试从未运行）
- 转正后（我重推一次改变了 SHA 来触发）：gate 变成 **`Require run-ci label (optional)`**

`run-ci` 标签**只有维护者能加**（我们试过，403 Must have admin rights）。所以真正的 CPU/GPU 测试要等维护者打标签才会跑。这是仓库对外部贡献者的常规策略，不是 PR 有问题。

**#32687 需要你决定**：鉴于本文的结论（3.6 上收益归零、main 已经 pin 到 3.6），它现在只对 Triton 3.5.1 用户有价值。PR 标题和正文已经如实写明这一点，把判断权交给维护者。如果你觉得不值得占用 review 资源，关掉它是完全合理的。
