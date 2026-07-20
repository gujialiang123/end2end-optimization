# 换模型/换场景,kernel level 还有空间吗?—— 用别人的 sglang PR 作证据

**日期**：2026-07-20 · 目的:回答"如果换模型,kernel 层还有没有提升机会",用**已合并的 sglang PR / 官方博客**作为"这里有真实性能提升"的证据(不需我们自己找)。

## 一句话结论
**有,但空间集中在"sglang 优化还没追上的地方"**:① 新架构(线性注意力 / 混合 / GDN / 新 indexer)、② AMD/新硬件、③ 量化/新精度、④ MoE 路由 kernel。而我们的 **Qwen3-30B-A3B + H200 + bf16 是最成熟、被打磨得最干净的路径,所以我们才几乎找不到空间**——换到上面这些方向就有。

## 证据(来自真实 sglang PR / 官方博客,均为别人的工作)

### A. MoE 路由 kernel(bf16,H200 直接相关)★最硬的证据
- **MoE Align & Sort 重新设计**:实测 **A100 3× / H200 3× / MI100 10× / MI300X 7×**(HF 官方博客 by SGLang MoE 作者)。
- 意义:这正是我 M=1 胜利里"跳过"的 align/sort overhead kernel —— 它本身在 H200 上就有 **3× 空间**。说明**即使在成熟的 bf16/H200 路径,MoE 的"路由/对齐"这类非 GEMM kernel 仍有真实 kernel 空间**(只是占比小,端到端杠杆有限)。

### B. 新架构:线性注意力 / 混合 / 新 indexer ★最大的机会
(这些正是我们查到的 `fused_gdn_gating` 服务的 qwen3_next / qwen3_5 / jet_nemotron 一类)
- **Indexer Prologue Fusion**:kernel 数 **12 → 4**,**bs=1 decode 快 ~8%**(DeepSeek indexer)。
- **FlashKDA kernels**:为 KDA/线性注意力新写的 safe-gate kernel,改善 prefill+decode。
- **ReplaySSM**:线性注意力的 buffered decode 路径,提吞吐。
- **DeepSeek-V4 indexer / C128 state-pool**:长上下文 **>10% 吞吐**。
- 意义:**新架构的 kernel 路径还在建设中,headroom 明显比成熟的 Qwen3-MoE 大**。换到 Qwen3-Next / DeepSeek-indexer / Mamba-hybrid 这类,kernel 层有活干、且有 PR 证明能提升。

### C. AMD / 新硬件(headroom 更大,因路径更不成熟)
- **Kimi-K2.5 on MI300X(FlyDSL 融合 MoE,W4A16+BF16)**:吞吐 **+162%**,TTFT **−65%**(AMD 官方博客)。
- align/sort 在 AMD 上 **7–10×**(见 A)。

### D. 算法/调度层的 kernel 化(和纯 kernel 交叉)
- **Speculative Decoding V2**(CUDA-graphable、融合 metadata、去同步):**~11% e2e TPS**。
- **TopK V2**:融合 top-k 选择 + page-table 变换,支持 runtime k 到 2048。

## 和我们自己发现的对照(诚实)
| 我们的发现(Qwen3-30B/H200/bf16) | 别人 PR 证明的机会在哪 |
|---|---|
| decode 89% memory-bound,kernel 算力优化 e2e 只 1.5% | 一致:纯 GEMM kernel 难再压 |
| align/sort 是 overhead,我 M=1 跳过它拿到 1.23× | **别人把 align/sort 本身重写拿到 3×(H200)** → 同一处有空间,只是占比小 |
| Qwen3-30B 热路径已全 CUDA 融合,无空缺 | **新架构(GDN/线性注意力/indexer)还没融合完,headroom 大** |
| CPU-only 融合 gap 服务 qwen2_moe/qwen3_next | 对应 FlashKDA/indexer fusion 等**正在被别人补的 CUDA kernel** |

## 给团队的建议
1. **想证明"kernel 层有真实空间"**:直接引用上面 A/B 的 PR(MoE Align&Sort 3× on H200;Indexer Prologue Fusion 12→4 kernels、bs=1 +8%)——都是别人已合并、有实测的 bf16 kernel 提升。
2. **想让我们自己也拿到有分量的 kernel 提升**:**换到新架构**(Qwen3-Next / DeepSeek-indexer / 线性注意力混合模型),那里 kernel 路径不成熟、headroom 大;或换到 **AMD**。Qwen3-30B/H200/bf16 这条已被打磨干净,不是好的练兵场。
3. **但记住 e2e 杠杆**:很多 kernel 提升(如 align/sort 3×)只作用在小组件上,端到端可能只有个位数 %;真正端到端大杠杆仍是 spec decoding(+23%~30%,见 headroom 图)。

## 来源
- MoE Align & Sort:HF blog "Efficient MoE Align & Sort design in SGLang Fused MoE"(3×A100/H200,7–10×AMD)
- Kimi-K2.5 MI300X:AMD ROCm blog(FlyDSL 融合 MoE,+162% 吞吐)
- Indexer fusion / FlashKDA / ReplaySSM / Spec Decoding V2 / TopK V2 / DeepSeek-V4:sgl-project/sglang releases + PR notes
