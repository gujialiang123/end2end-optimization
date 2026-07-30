# SGLang 如何接入模型、如何选 backend —— 以及和 torch.compile/FX 体系的接口在哪

**日期**：2026-07-30
**背景**：mentor 提出的两个问题 ——
(1) Maya 那边拿到的是 PyTorch 模型，工作流是「导出 FX graph → 找可替换的子图 → 换成高效 kernel」；
(2) SGLang 内部到底怎么做这件事？它的 backend decision tree 长什么样？新模型怎么接进来？
**目的**：先把 SGLang 的真实机制查清楚（而不是猜），再据此设计能同时衔接两套体系的实验。

**所有结论都对应 `sgl-project/sglang` @ `35f2e6ab58`（2026-07-30）的具体文件和行号，可复核。**

---

## 0. 三句话结论

1. **SGLang 接入新模型不用 FX，是手写模型文件**（212 个 `models/*.py`），
   外加一条通用回退路径，那条路径用的是 **`nn.Module` 树遍历 + 类名字符串匹配**，仍然不是 FX。
2. **SGLang 的 backend 决策不看图**，是一棵**手写 if/elif 树**，输入是「硬件能力 + 模型 config 属性 + 功能开关」。
3. **SGLang 有 FX 基础设施，但只用来切图（piecewise CUDA graph），不做融合。**
   vLLM 的 `FusionPass`/`NoopEliminationPass` **没有移植过来**——
   `pass_manager.py:29` 的 docstring 提到它们，但代码里不存在。

**→ 这三点合起来正好是我们的机会：SGLang 已经有 FX pass 的挂载点，但那个位置是空的。**

---

## 1. 新模型是怎么接进来的

### 1.1 主路径：手写模型文件（212 个）

```bash
ls python/sglang/srt/models/*.py | wc -l   # 212
```

每个文件末尾注册自己：

```python
# models/gemma3_causal.py 最后一行
EntryClass = [Gemma3ForCausalLM, EmbeddingGemmaModel]
```

`models/registry.py:95` 的 `import_model_classes()` 扫描整个 package，
凡是有 `EntryClass` 属性的模块就注册进去。

**所以「接入一个新模型」= 有人手写一个 `.py`，把 HF 的实现按 SGLang 的层原语重写一遍。**
不是自动的，不涉及任何图分析。

工作量的实质：把 HF 的 `nn.Linear` 换成 SGLang 的 TP 感知线性层、
把 attention 换成 SGLang 的 `RadixAttention`（接 KV cache）、
把 norm 换成 `RMSNorm`（接融合 kernel）……**这就是「定制化」的来源。**

### 1.2 回退路径：`models/transformers.py`

有一条通用路径，能直接吃 HF 模型。核心机制在 `models/transformers.py:769` 起：

```python
def _recursive_replace(module: nn.Module, prefix: str):
    for child_name, child_module in module.named_children():
        if isinstance(child_module, nn.Linear):
            new_module = replace_linear_class(child_module, style, ...)   # 换成 TP 线性层
        elif child_module.__class__.__name__.endswith("RMSNorm"):          # <- 类名字符串匹配
            new_module = replace_rms_norm_class(child_module, hidden_size)
        else:
            _recursive_replace(child_module, prefix=qual_name)             # 递归下去
        if new_module is not child_module:
            setattr(module, child_name, new_module)
```

注意力则走 HF 自己的扩展点（`transformers.py:326`）：

```python
ALL_ATTENTION_FUNCTIONS["sglang"] = sglang_flash_attention_forward
...
self.text_config._attn_implementation = "sglang"    # :613
```

**这是 duck typing，不是图重写**：
- 认 `isinstance(x, nn.Linear)`
- 认 **类名以 "RMSNorm" 结尾**（`endswith("RMSNorm")`）
- attention 靠 HF 预留的 hook

**局限很明显**：只能替换「能被 `nn.Module` 边界框住」的东西。
写在 `forward` 里的散装算子（比如 `x = x + residual; x = norm(x)` 这种跨模块的融合机会）
它**完全看不见**——因为那不是一个 module，是两个 module 之间的胶水代码。

★ **这正是 Maya 那套 FX 方法的优势所在，也是两套体系最自然的接口点。**

---

## 2. Backend 决策树长什么样

### 2.1 Attention backend

`server_args.py:5507` 的 `_get_default_attn_backend()`，是一棵**纯手写的 if/elif**：

```python
if not use_mla_backend:                                    # <- 模型 config 属性
    if is_hopper_with_cuda_12_3() and is_no_spec_infer_or_topk_one(...):
        return "fa3"
    elif is_sm100_supported() and ...:
        return "trtllm_mha"
    elif is_hip():
        return "aiter"
    elif is_mps():
        return "torch_native"
    else:
        if is_flashinfer_available() and not model_config.has_attention_sinks:
            return "flashinfer"
        return "triton"
else:                                                      # MLA 架构
    if is_hopper_with_cuda_12_3():   return "fa3"
    elif is_sm100_supported():       return "flashinfer"
    ...
```

决策输入只有三类：

| 输入 | 例子 |
|---|---|
| **硬件能力** | `is_hopper_with_cuda_12_3()`, `is_sm100_supported()`, `is_hip()`, `is_mps()` |
| **模型 config 属性** | `use_mla_backend`, `has_attention_sinks`, `architectures` 里的字符串 |
| **功能开关** | 是否开投机解码、`topk > 1`、`page_size` |

**没有任何一项来自图分析。** 甚至有直接按模型名硬编码的（`"WhisperForConditionalGeneration" -> flashinfer`）。

### 2.2 MoE backend

同样是手写规则，散落在各个 quant 方法里。例如 `layers/quantization/fp8.py:1075`：

```python
if moe_runner_backend.is_auto():
    return deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM and (
        get_moe_a2a_backend().is_deepep() or ... )
```

判据是「这个 kernel 编译上可不可用 + 有没有配套的 all-to-all 后端」，
**不是「这个 shape 上哪个更快」**。

★ 这一点和我们的实测吻合：我们在 K1 实验里发现 **regime→backend 的最优选择不可迁移**，
用错 backend 最差 −34%。因为运行时压根没有基于形状/负载的选择逻辑。

---

## 3. torch.compile 在 SGLang 里的真实地位

### 3.1 默认关闭

```python
# server_args.py:1977
enable_torch_compile: A[bool, "Optimize the model with torch.compile. Experimental feature."] = False
torch_compile_max_bs: A[int, ...] = 32     # 只对 bs<=32 生效
```

**默认关，标注 experimental，且只覆盖小 batch。**

另有零星的函数级装饰器（`models/llama4.py:74`、`speculative/spec_utils.py:265` 等），
但那是**手工挑几个小函数编译**，不是全模型编译。

### 3.2 有 FX pass 基础设施，但融合 pass 是空的

`python/sglang/srt/compilation/` 是从 **vLLM 移植**过来的（文件头写着
`Adapted from https://github.com/vllm-project/vllm/blob/v0.10.0/vllm/compilation/`）。

`pass_manager.py:20` 的 `PostGradPassManager` docstring 说：

```
The order of the post-grad post-passes is:
1. passes (constructor parameter)
2. default passes (NoopEliminationPass, FusionPass)      <- 这两个
3. config["post_grad_custom_post_pass"]
4. fix_functionalization
```

**但 `FusionPass` 和 `NoopEliminationPass` 在 SGLang 里根本不存在**：

```bash
grep -rn "FusionPass\|NoopElimination" python/sglang/srt/ --include=*.py
# 只有 pass_manager.py:29 那行 docstring 命中，没有任何实现
```

`configure()` 里实际只挂了一个：

```python
def configure(self):
    self.pass_config = dict()
    self.fix_functionalization = FixFunctionalizationPass()    # 正确性修复，不是优化
```

**结论：docstring 是连同代码一起从 vLLM 抄来的，融合 pass 没有跟着移植。挂载点是空的。**

### 3.3 SGLang 用 FX 做的唯一一件事：切图

`compilation/backend.py:225` 的 `split_graph()`：

```python
def split_graph(graph: fx.GraphModule, ops: list[str]):
    for node in graph.graph.nodes:
        if node.op == "call_function" and str(node.target) in ops:
            subgraph_id += 1              # 在指定算子处切开
            ...
    split_gm = torch.fx.passes.split_module.split_module(
        graph, None, lambda node: node_to_subgraph_id[node], keep_original_order=True)
```

目的是 **piecewise CUDA graph**：在 attention 算子处把图切开，
attention 之外的部分用 CUDA graph 捕获（attention 因为要访问动态 KV cache，不能被静态捕获）。

**所以 SGLang 用 FX 是为了「切」，不是为了「融」。** 这个区分很重要。

---

## 4. 两套体系的对照

| | **Maya / torch.compile 路线** | **SGLang 现状** |
|---|---|---|
| 模型接入 | 导出 FX graph，自动分析 | **手写 212 个模型文件** |
| 找优化机会 | 图上做 pattern matching | **人工阅读代码 + profiler** |
| 替换 kernel | `subgraph_rewriter` 替换子图 | **手写调用点** |
| backend 选择 | 编译期基于图和形状 | **手写 if/elif，看硬件不看图** |
| 融合 | Inductor 自动 + 自定义 pass | **预先写好的融合 kernel，手工调用** |
| FX 的角色 | 核心 IR | **只用来切图给 CUDA graph** |

**两边的强弱正好互补：**

- SGLang 强在**手写 kernel 的质量**（fa3、trtllm、deep_gemm 这些是 FX/Inductor 生成不出来的）
- Maya 那套强在**自动发现和替换**（不用一个模型一个模型地手写）

**衔接点：让自动分析去找机会，让手写 kernel 去兑现。**

---

## 5. 这正好解释了我们已经做过的事

我们前几周的三个案例，事后看全都是「手工做了 FX 本该自动做的事」：

| 案例 | 我们怎么找到的 | FX 本可以怎么找 |
|---|---|---|
| **Gemma-3 q_norm 未融合** | 读代码发现 `if x.dim()==2` 守卫 | 图上匹配 `pow->mean->rsqrt->mul` 链，发现它没被替换成融合算子 |
| **LFM2.5 residual 未融合** | profiler 看到 48 个独立 add | 图上匹配 `add` 紧跟 `rms_norm` 的模式 |
| **LFM2.5 MoE config 缺失** | 查文件是否存在 | 图分析看不到（这是运行时查表，不在图里） |

前两个**完全可以用 FX pattern matching 自动化**，第三个不行。
**这个边界本身就是一个值得汇报的结论**：图能覆盖什么、不能覆盖什么。

而且我们已经写过 FX 工具（`scripts/lfm_fusion/fx_*.py`），
当时用它做候选筛选和数据依赖验证。**基础已经在了。**

---

## 6. 建议的实验设计

### 实验 A：量化「手写 kernel vs Inductor 自动生成」的差距（优先做）

**问题**：SGLang 坚持手写 kernel 是对的吗？torch.compile 能追平多少？

**做法**：同一个模型三个 arm，测端到端：

| arm | 配置 |
|---|---|
| 1 | SGLang 默认（手写融合 kernel，`enable_torch_compile=False`） |
| 2 | `--enable-torch-compile`（Inductor 接管） |
| 3 | 把手写融合 kernel 关掉 + 开 torch.compile（让 Inductor 从零融合） |

**为什么有价值**：这直接回答 mentor 的核心疑问——
「Maya 那套 torch.compile 体系，能不能达到 SGLang 手写的水平」。
答案不管是哪个方向都有用：
- 如果接近 -> 自动化路线可行，能省掉手写 212 个模型的成本
- 如果差很多 -> 说明手写 kernel 不可替代，衔接方式应该是「FX 找机会 + 手写 kernel 兑现」

**成本**：低。sglang 现成的 flag，不用改代码。**几小时。**

注意 `torch_compile_max_bs=32`，大 batch 下 arm 2 会退回 eager，测的时候要控制 batch。

### 实验 B：把我们的三个发现写成 FX pass，验证能否自动重现（最有价值）

**问题**：我们人工找到的优化点，FX pattern matching 能不能自动找出来？

**做法**：
1. 用 `torch.compile` 导出 gemma-3 的 post-grad FX graph
2. 写一个 pattern：`pow -> mean -> add -> rsqrt -> mul -> mul`（RMSNorm 的展开形式）
3. 在图上匹配，看能否命中我们手工发现的那 52 处 q_norm/k_norm
4. 用 `subgraph_rewriter` 替换成 `torch.ops.sgl_kernel.gemma_rmsnorm`
5. 验证数值等价 + 测性能

**为什么最有价值**：
- 它把我们**已经验证过收益的案例**（+36.6%）当作 ground truth，
  所以能干净地衡量「自动化方法的召回率」
- 产出物是**一个能挂进 SGLang `PostGradPassManager` 的真 pass**——
  而那个挂载点现在是空的（§3.2），我们填的是一个真实空白
- 直接对上 mentor 说的 "FX graph 导出 -> 替换成高效 kernel" 的工作流

**成本**：中。需要写 pass + 验证。**2–3 天。**

**关键设计**：先在**已知答案**的案例上做（Gemma-3），验证方法有效后再去扫新模型。
不要一上来就追求通用。

### 实验 C：跨模型扫描，衡量自动化的召回率

**问题**：这套 FX pass 放到没见过的模型上，能找到多少机会？

**做法**：把实验 B 的 pass 跑在我们审计过的 11 个模型上，
和我们**手工审计的结果**对比（`docs/2026-07-28/cross_architecture_audit.md` 里有完整数据）。

**输出一张表**：

| 模型 | 手工审计发现的空缺 | FX pass 自动发现 | 命中/漏报/误报 |
|---|---|---|---|

**为什么有价值**：这是**唯一能量化「自动化能替代多少人工」的实验**，
因为我们有人工基线。没有这个基线的团队做不了这个实验。

**成本**：低（pass 写好之后）。**1 天。**

### 实验 D：backend 决策能否基于图/形状而非硬件（备选）

**问题**：§2 说 backend 选择只看硬件。但我们 K1 实验证明**最优 backend 随 regime 变化**，
用错最差 −34%。能不能做一个基于形状的选择器？

**做法**：用我们已有的 backend 对比数据，做一个
`(模型架构, batch, seqlen) -> backend` 的规则，和 SGLang 的静态默认对比。

**为什么优先级低**：这更偏 serving 层而不是 kernel 层，
mentor 现在更想看 kernel level 的东西。**建议作为备选。**

---

## 7. 建议顺序：A -> B -> C

**理由**：

- **A 最便宜且能立刻回答 mentor 的核心疑问**（自动化 vs 手写差多少）。
  不管结果如何，都决定了后面的方向。
- **B 是真正的产出**——一个能填进 SGLang 空挂载点的 FX pass，
  而且用我们已验证的案例做 ground truth，可信度高。
- **C 让 B 的价值可量化**（召回率），并且我们有别人没有的人工基线。

**A 建议 GPU 一空出来就跑**，结果出来再决定 B 的具体形态。

### 预实验已做：风险排除，pattern 是干净的

原本担心 post-grad graph 里算子已被 Inductor 分解得面目全非，pattern 不好写。
**写文档时顺手验证了，结论是不用担心**（CPU 上跑的，几分钟）。

**Dynamo 层**（`torch.compile(backend=...)` 捕获）：

```
placeholder    l_x_
call_method    o        float          (l_x_,)
call_method    pow_1    pow            (o, 2)
call_method    mean     mean           (pow_1, -1)   {'keepdim': True}
call_method    add      add            (mean, 1e-06)
call_function  rsqrt    rsqrt          (add,)
call_method    o_1      mul            (o, rsqrt)
call_method    add_1    add            (float_2, 1.0)
call_method    o_2      mul            (o_1, add_1)
call_method    type_as  type_as        (o_2, l_x_)
```

**post-grad 层**（把 pass 挂到 `post_grad_custom_post_pass`，
也就是 SGLang `PostGradPassManager` 实际作用的位置）：

```
%convert_element_type   prims.convert_element_type.default   (bf16 -> fp32)
%pow_1                  aten.pow.Tensor_Scalar               (%convert_element_type, 2)
%mean                   aten.mean.dim                        (%pow_1, [-1], True)
%add                    aten.add.Tensor                      (%mean, 1e-06)
%rsqrt                  aten.rsqrt.default                   (%add,)
%mul                    aten.mul.Tensor                      (%convert_element_type, %rsqrt)
%convert_element_type_1 prims.convert_element_type.default   (weight -> fp32)
%add_1                  aten.add.Tensor                      (%convert_element_type_1, 1.0)
%mul_1                  aten.mul.Tensor                      (%mul, %add_1)
%convert_element_type_2 prims.convert_element_type.default   (-> bf16)
```

**两层都完全可匹配**，而且 post-grad 层全是规范的 `aten.*` / `prims.*` 算子，
比 Dynamo 层的 `call_method` 更好写 pattern。

额外收获：post-grad 图上每个节点都带 `num_users=N` 标注。
**这正好是融合的安全条件**——只有中间结果 `num_users=1`（没有别人在用）才能安全融掉。
不需要自己做活跃性分析，图里现成的。

**所以实验 B 的主要技术风险已经排除，可以直接动手。**

---

## 8. 复核方式

```bash
cd /tmp/sglang_lfm    # sglang main @ 35f2e6ab58

# §1.1 手写模型文件数量与注册机制
ls python/sglang/srt/models/*.py | wc -l
tail -3 python/sglang/srt/models/gemma3_causal.py
sed -n '95,115p' python/sglang/srt/models/registry.py

# §1.2 通用回退路径的 module 替换
sed -n '769,800p' python/sglang/srt/models/transformers.py

# §2.1 attention backend 决策树
sed -n '5507,5575p' python/sglang/srt/server_args.py

# §3.2 融合 pass 不存在（关键）
grep -rn "FusionPass\|NoopElimination" python/sglang/srt/ --include=*.py
sed -n '20,55p' python/sglang/srt/compilation/pass_manager.py

# §3.3 FX 只用来切图
sed -n '225,250p' python/sglang/srt/compilation/backend.py
```

---

## 9. 汇报时可以直接用的三句话

1. **「SGLang 没有 FX 驱动的模型接入流程。212 个模型是手写的，
   唯一的通用路径靠 `nn.Module` 树遍历和类名字符串匹配，看不见跨 module 的融合机会。」**

2. **「SGLang 的 backend 决策是一棵手写 if/elif 树，输入是硬件能力和 config 属性，
   完全不看图，也不看形状——这和我们实测『最优 backend 随 regime 变化、用错最差 −34%』正好对上。」**

3. **「SGLang 从 vLLM 移植了 FX pass 的基础设施，但融合 pass 没跟着移植过来——
   `PostGradPassManager` 的 docstring 写着 `FusionPass`，代码里不存在。
   这个挂载点是空的，正好是我们能填的地方。」**
