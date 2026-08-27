# Week 01｜LLM Systems 工程实践报告

> 主题：从 Transformer 结构走向 AI Infra 的资源核算与运行时观测  
> 对应路线：AI Infra 学习求职路线 V3 · Week 1  
> 核心目标：把“看懂 Transformer”进一步转化为“能够实现、核算、测量并解释 Transformer 的真实工程行为”。

---

# 0. 本周为什么要做这三个实验

Week 1 的理论部分解决的是：**Transformer 是什么、主要由哪些算子组成、模型资源应该如何估算。**

工程部分则要完成第一次完整闭环：

```text
Transformer 结构
    ↓
Tensor Shape
    ↓
参数量
    ↓
FLOPs
    ↓
理论显存
    ↓
PyTorch 实际执行
    ↓
GPU 时间与峰值显存
```

这三个实验分别负责其中不同的一段：

| 实验 | 核心问题 | 最终能力 |
|---|---|---|
| Experiment 1：Mini Transformer | Transformer 的一次 Forward 到底怎样发生？ | 能自己搭建最小 Decoder-only Transformer，并解释每个 Tensor 的 Shape |
| Experiment 2：Resource Accounting | 这个模型理论上需要多少参数、显存、FLOPs？ | 能从公式手算，并用 PyTorch 结果验证 |
| Experiment 3：Runtime Profiling | PyTorch 和 GPU 实际是怎样执行这段模型的？ | 会做基础 Benchmark、Profiler、Trace 和显存观测 |

AI Infra 的学习与普通“模型使用”最大的区别就在于：

> 不只要知道模型“能不能跑”，还要知道它**为什么这样跑、消耗了什么资源、瓶颈可能在哪里、如何证明自己的判断**。

因此，Week 1 的重点不是训练一个有能力的语言模型，而是建立后续所有 GPU / CUDA / Triton / Distributed / vLLM 学习都会反复使用的工程思维。

---

# 1. 实验环境与统一符号

本报告假设使用一个非常小的 Llama 风格 Decoder-only Transformer，主要参数如下：

```python
vocab_size = 10000
hidden_size = 256
num_layers = 4
num_heads = 8
ffn_hidden_size = 1024
max_seq_len = 128
```

统一使用以下数学符号：

| 符号 | 含义 |
|---|---|
| \(B\) | Batch Size |
| \(S\) | Sequence Length |
| \(D\) | Hidden Size / Model Dimension |
| \(H\) | Attention Head 数量 |
| \(D_h\) | 单个 Head 的维度，\(D_h=D/H\) |
| \(F\) | FFN Hidden Dimension |
| \(V\) | Vocabulary Size |
| \(L\) | Transformer Layer 数量 |

在默认配置中：

```text
B = 2
S = 128
D = 256
H = 8
Dh = 32
F = 1024
V = 10000
L = 4
```

工程目录建议保持如下结构：

```text
week01_llm_systems/
├── model/
│   ├── __init__.py
│   ├── config.py
│   ├── rmsnorm.py
│   ├── attention.py
│   ├── mlp.py
│   ├── block.py
│   └── transformer.py
├── tests/
│   └── test_model.py
├── resource_accounting.py
├── profile_forward.py
├── profiler_output/
│   └── forward_trace.json
└── week01_report.md
```

---

# 2. Experiment 1｜手写 Mini Transformer

## 2.1 实验目标

Experiment 1 的目标不是“重新发明 Transformer”，而是做到：

1. 能够自己实现一个最小 Decoder-only Transformer Forward；
2. 能够解释每一个主要 Tensor 的 Shape；
3. 能够解释 Attention 为什么会出现 \(S^2\)；
4. 能够理解 RMSNorm、Residual、MLP 在代码中的具体位置；
5. 能完成 Forward、Backward、Shape、数值有限性等最基本的工程 sanity check。

最终模型结构：

```text
input_ids [B,S]
    ↓
Embedding
    ↓ [B,S,D]
┌────────────────────────────┐
│ TransformerBlock × L       │
│                            │
│ RMSNorm                    │
│   ↓                        │
│ Causal Self-Attention      │
│   ↓                        │
│ Residual Add               │
│                            │
│ RMSNorm                    │
│   ↓                        │
│ MLP                        │
│   ↓                        │
│ Residual Add               │
└────────────────────────────┘
    ↓
Final RMSNorm
    ↓
LM Head
    ↓
logits [B,S,V]
```

当前 Week 1 **故意不实现**：

```text
KV Cache
RoPE
GQA
FlashAttention
Generation Loop
Distributed Parallelism
```

原因是先把最基础的数据流搞清楚，再逐层增加系统复杂度。

---

## 2.2 ModelConfig：把超参数工程化

建议使用 `dataclass` 管理模型配置：

```python
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 10000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    ffn_hidden_size: int = 1024
    max_seq_len: int = 128
    rms_norm_eps: float = 1e-6
```

### 为什么需要 Config？

因为后面的所有工程量几乎都由这些变量决定：

```text
hidden_size
num_heads
num_layers
ffn_hidden_size
sequence_length
vocab_size
```

它们同时决定：

- 参数量；
- Tensor Shape；
- Attention Matrix 大小；
- FLOPs；
- Weight Memory；
- Activation Memory；
- 后续多卡切分方式。

这也是为什么在 AI Infra 中，“模型超参数”不只是算法概念，而是直接决定资源需求的系统参数。

---

# 3. RMSNorm：从数学公式到张量实现

## 3.1 RMSNorm 的核心目标

神经网络经过很多层之后，hidden activation 的整体尺度可能发生明显变化。Norm 类操作的作用之一，就是帮助控制 activation 的数值尺度，让训练和深层网络计算更加稳定。

经典 LayerNorm 会进行：

\[
\mu=\frac{1}{D}\sum_i x_i
\]

\[
\sigma^2=\frac{1}{D}\sum_i(x_i-\mu)^2
\]

然后：

\[
\hat{x_i}=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}
\]

RMSNorm 做了简化：**不显式减去均值，只控制向量整体 magnitude / scale**。

RMS 定义：

\[
RMS(x)=\sqrt{\frac{1}{D}\sum_{i=1}^{D}x_i^2}
\]

RMSNorm：

\[
RMSNorm(x)=\gamma\odot\frac{x}{\sqrt{\frac{1}{D}\sum_i x_i^2+\epsilon}}
\]

其中 \(\gamma\) 是可训练 scale parameter。

---

## 3.2 RMSNorm 实现

```python
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(mean_square + self.eps)
        return x * self.weight
```

### Shape 分析

输入：

```text
x: [B,S,D]
```

执行：

```python
x.pow(2)
```

Shape 不变：

```text
[B,S,D]
```

然后：

```python
.mean(dim=-1, keepdim=True)
```

得到：

```text
[B,S,1]
```

这是因为 RMSNorm 是针对**每个 token 的 hidden dimension**做归一化：

```text
Token 1: [x1,x2,...,xD] → normalize
Token 2: [x1,x2,...,xD] → normalize
...
```

不是跨 token 做归一化。

---

## 3.3 `torch.rsqrt()` 是什么？

`torch.rsqrt(x)` 表示 reciprocal square root：

\[
rsqrt(x)=\frac{1}{\sqrt{x}}
\]

例如：

```python
x = torch.tensor([4.0, 9.0, 16.0])
print(torch.rsqrt(x))
```

得到：

```text
[0.5, 0.3333, 0.25]
```

因此：

```python
x * torch.rsqrt(mean_square + eps)
```

数学上等价于：

```python
x / torch.sqrt(mean_square + eps)
```

即：

\[
\frac{x}{\sqrt{mean(x^2)+\epsilon}}
\]

`rsqrt + multiply` 是 Norm 类算子里非常常见的基础计算模式。后续学习 CUDA/Triton、RMSNorm Fusion 时还会重新遇到。

> 注意：`mean(x^2)` 严格来说是 mean square，并不是传统意义上的 variance。变量命名成 `mean_square` 会比 `variance` 更准确。

---

## 3.4 为什么还有 `weight`？

归一化以后模型仍然需要一定的表示自由度，因此 RMSNorm 通常保留一个可训练 scale：

```python
self.weight = nn.Parameter(torch.ones(hidden_size))
```

Shape：

```text
[D]
```

最终：

\[
y_i=\gamma_i\frac{x_i}{RMS(x)}
\]

不同 hidden feature 可以学习不同的最佳 scale。

---

# 4. MLP：从 `[B,S,D]` 扩展到 `[B,S,F]`

实验使用最简单的两层 FFN：

```python
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, hidden_size, ffn_hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ffn_hidden_size, bias=False)
        self.fc2 = nn.Linear(ffn_hidden_size, hidden_size, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
```

数据流：

```text
[B,S,D]
   ↓ Linear D→F
[B,S,F]
   ↓ GELU
[B,S,F]
   ↓ Linear F→D
[B,S,D]
```

参数量：

第一层：

\[
DF
\]

第二层：

\[
FD
\]

所以：

\[
P_{MLP}=2DF
\]

默认配置：

\[
2\times256\times1024=524288
\]

可以直接用 PyTorch 验证：

```python
mlp = MLP(256, 1024)
params = sum(p.numel() for p in mlp.parameters())
print(params)
```

输出应为：

```text
524288
```

这里已经体现 Week 1 的一个重要学习方式：

> 每写一个模块，都同时问自己：**Shape 是什么？参数量是多少？主要 FLOPs 在哪里？**

---

# 5. Causal Self-Attention：Experiment 1 的核心

## 5.1 Attention 数据流

```text
                 X [B,S,D]
                 │
      ┌──────────┼──────────┐
      ↓          ↓          ↓
     Wq         Wk         Wv
      ↓          ↓          ↓
      Q          K          V
      │          │          │
      └──── split heads ────┘
             [B,H,S,Dh]
                  │
                QKᵀ
                  ↓
             [B,H,S,S]
                  ↓
          scale + causal mask
                  ↓
               softmax
                  ↓
             Attention Prob
                  ↓
                 × V
                  ↓
             [B,H,S,Dh]
                  ↓
             merge heads
                  ↓
              [B,S,D]
                  ↓
                 Wo
```

---

## 5.2 Q/K/V Projection

```python
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)
```

输入：

```text
x: [B,S,D]
```

三个 Projection 后仍然是：

```text
Q/K/V: [B,S,D]
```

Attention 有四个主要 Linear：

```text
Wq
Wk
Wv
Wo
```

每个都是 \(D\times D\)，所以：

\[
P_{Attention}=4D^2
\]

---

## 5.3 拆分 Multi-Head

```python
q = q.view(B, S, H, Dh)
k = k.view(B, S, H, Dh)
v = v.view(B, S, H, Dh)
```

其中：

\[
D_h=D/H
\]

默认：

\[
256/8=32
\]

因此：

```text
[B,S,D]
→ [B,S,H,Dh]
→ [2,128,8,32]
```

之后：

```python
q = q.transpose(1, 2)
k = k.transpose(1, 2)
v = v.transpose(1, 2)
```

得到：

```text
[B,H,S,Dh]
```

这样每个 batch 中的每个 Head 都可以被视为一个独立的 `[S,Dh]` 矩阵参与 Attention。

---

# 6. Tensor 的 Shape、Stride 与 `contiguous()`

这是 Experiment 1 中最值得建立的底层张量概念之一。

## 6.1 Shape 不等于物理内存布局

一个 PyTorch Tensor 除了 Shape，还有：

```text
Storage
Shape
Stride
```

例如：

```python
x = torch.randn(2, 3, 4)
print(x.shape)
print(x.stride())
```

常见结果：

```text
shape = (2,3,4)
stride = (12,4,1)
```

Stride 表示沿某个维度移动一个逻辑元素时，底层 Storage 需要跳过多少个元素。

---

## 6.2 `transpose()` 为什么通常很便宜？

执行：

```python
y = x.transpose(1, 2)
```

逻辑维度发生变化，但 PyTorch 通常不会立即重新复制整块数据，而只是改变 Shape / Stride 对 Storage 的解释方式。

因此 `transpose()` 通常是一个 View 操作。

例如原始矩阵：

```text
1 2 3
4 5 6
```

底层内存：

```text
1,2,3,4,5,6
```

Transpose 后逻辑上变成：

```text
1 4
2 5
3 6
```

但底层内存可以保持不变，仅通过 stride 表示新的访问顺序。

于是：

```python
y.is_contiguous()
```

通常会得到：

```text
False
```

---

## 6.3 `contiguous()` 的真实目的

`contiguous()` 的含义是：

> 按照 Tensor 当前的**逻辑顺序**重新整理一块物理连续的内存。

例如 Attention 得到：

```text
[B,H,S,Dh]
```

然后：

```python
out = out.transpose(1, 2)
```

变成：

```text
[B,S,H,Dh]
```

此时逻辑顺序变了，但内存布局通常还不是新的连续排列。

如果马上：

```python
out.view(B, S, D)
```

可能出现：

```text
view size is not compatible with input tensor's size and stride
```

因此常见写法：

```python
out = out.transpose(1, 2).contiguous().view(B, S, D)
```

步骤：

```text
[B,H,S,Dh]
   ↓ transpose
[B,S,H,Dh]   ← 逻辑顺序变了，但可能 non-contiguous
   ↓ contiguous
[B,S,H,Dh]   ← 重新整理成物理连续布局
   ↓ view
[B,S,D]
```

### `view()` 和 `reshape()` 的区别

`view()` 强调：

> 尽量不复制，只重新解释同一块内存。

因此它对 stride / contiguous 要求更严格。

`reshape()` 更灵活：如果无法直接 view，可能自动创建新的 contiguous copy。

学习底层系统时，显式写：

```python
contiguous().view(...)
```

更容易理解“这里发生了一次潜在的数据搬运”。

这非常重要，因为以后 GPU 性能优化中：

```text
transpose
contiguous
copy
layout conversion
```

都有可能带来额外 HBM memory traffic。

---

# 7. 为什么 Attention 出现 \(S^2\)

计算：

```python
scores = q @ k.transpose(-2, -1)
```

Shape：

```text
Q:  [B,H,S,Dh]
Kᵀ: [B,H,Dh,S]
```

结果：

```text
[B,H,S,S]
```

因此每个 Head 都需要构造一个：

\[
S\times S
\]

的 Attention Score Matrix。

序列长度增长时：

```text
S = 128    → 128 × 128
S = 4096   → 4096 × 4096
S = 32768  → 32768 × 32768
```

这就是标准 Attention 在长序列下计算和中间张量成本迅速增长的重要来源。

这也直接为 Week 2 的 Attention Alternatives，以及后面的 FlashAttention 做铺垫。

---

# 8. Causal Mask 与 Softmax

Decoder-only LM 不能看到未来 token，因此需要 causal mask。

示例：

```python
mask = torch.triu(
    torch.ones(S, S, device=x.device, dtype=torch.bool),
    diagonal=1,
)

scores = scores.masked_fill(mask, float('-inf'))
attn = torch.softmax(scores, dim=-1)
```

一个 4×4 的上三角 mask：

```text
0 1 1 1
0 0 1 1
0 0 0 1
0 0 0 0
```

被 mask 的位置填入 `-inf`，经过 softmax 后概率变成 0。

因此第 \(t\) 个 token 只能关注：

```text
0 ... t
```

不能关注：

```text
t+1 ... S-1
```

---

# 9. TransformerBlock 与 Residual

采用 Pre-Norm 结构：

```python
def forward(self, x):
    x = x + self.attn(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
```

结构：

```text
x
├─ RMSNorm → Attention ─┐
└───────────────────────+→ x'
                         │
x'                       │
├─ RMSNorm → MLP ────────┐
└────────────────────────+→ output
```

一个 Block 的参数量近似：

\[
P_{block}=4D^2+2DF+2D
\]

其中：

```text
4D²  → Q/K/V/O
2DF  → 两层 MLP
2D   → 两个 RMSNorm scale
```

---

# 10. Experiment 1 的工程验证

至少做以下 sanity check：

## 10.1 Shape

```python
logits = model(input_ids)
assert logits.shape == (B, S, vocab_size)
```

## 10.2 数值有限性

```python
assert torch.isfinite(logits).all()
```

## 10.3 Backward

```python
loss = logits.mean()
loss.backward()
assert model.embedding.weight.grad is not None
```

最终确认：

```text
Forward       ✅
Backward      ✅
Shape         ✅
Finite values ✅
Gradient      ✅
```

这些基础验证虽然简单，但代表了系统工程最重要的思想之一：

> **先证明 correctness，再讨论性能。**

---

# 11. Experiment 1 最终必须掌握的 Shape Trace

| Operation | Input | Output |
|---|---|---|
| Embedding | `[B,S]` | `[B,S,D]` |
| Q/K/V Projection | `[B,S,D]` | `[B,S,D]` |
| Split Heads | `[B,S,D]` | `[B,H,S,Dh]` |
| QKᵀ | `[B,H,S,Dh]` | `[B,H,S,S]` |
| Softmax | `[B,H,S,S]` | `[B,H,S,S]` |
| Attention × V | `[B,H,S,S]` | `[B,H,S,Dh]` |
| Merge Heads | `[B,H,S,Dh]` | `[B,S,D]` |
| Output Projection | `[B,S,D]` | `[B,S,D]` |
| MLP 1 | `[B,S,D]` | `[B,S,F]` |
| MLP 2 | `[B,S,F]` | `[B,S,D]` |
| LM Head | `[B,S,D]` | `[B,S,V]` |

如果这张表不能脱离代码独立画出来，说明 Transformer 的工程数据流还没有真正掌握。

---

# 12. Experiment 2｜Resource Accounting

## 12.1 实验目标

Experiment 2 将 Experiment 1 的代码结构转化为资源公式：

```text
模型结构
   ↓
参数量
   ↓
Weight Memory
   ↓
Training Model States
   ↓
Forward FLOPs
   ↓
Sequence Length Scaling
```

最终要求：

1. 手算参数量，并与 `sum(p.numel())` 完全一致；
2. 计算 FP32 / FP16 / BF16 / INT8 等理论 Weight Memory；
3. 估算 BF16 + Gradient + Adam States 的 model-state memory；
4. 估算 Attention / MLP / LM Head 的 Forward FLOPs；
5. 观察 Sequence Length 增长时 Attention 占比的变化。

---

# 13. 参数量核算

实验模型假设：

```text
Embedding
L × TransformerBlock
Final RMSNorm
LM Head
bias=False
Embedding 与 LM Head 不共享权重
```

---

## 13.1 Embedding

Embedding weight：

\[
[V,D]
\]

参数量：

\[
P_{embedding}=VD
\]

---

## 13.2 Attention

Q/K/V/O 四个 Linear：

\[
P_{attention}=D^2+D^2+D^2+D^2=4D^2
\]

所有层：

\[
4LD^2
\]

---

## 13.3 MLP

\[
D\rightarrow F\rightarrow D
\]

所以：

\[
P_{MLP}=DF+FD=2DF
\]

所有层：

\[
2LDF
\]

---

## 13.4 Norm

一个 RMSNorm 只有一个 `[D]` scale parameter。

一个 Block 两个：

\[
2D
\]

所有 Layer：

\[
2LD
\]

再加 Final RMSNorm：

\[
D
\]

---

## 13.5 LM Head

\[
[D,V]
\]

参数量：

\[
DV
\]

---

## 13.6 总参数量

\[
P=VD+L(4D^2+2DF+2D)+D+DV
\]

因为当前模型没有 weight tying：

\[
P=2VD+L(4D^2+2DF+2D)+D
\]

默认配置下：

```text
embedding   = 2,560,000
attention   = 1,048,576
mlp         = 2,097,152
block_norm  = 2,048
final_norm  = 256
lm_head     = 2,560,000
```

总计：

```text
8,268,032 parameters
```

---

# 14. Manual 参数量与 PyTorch 参数量验证

Manual：

```python
def count_parameters_manual(config):
    V = config.vocab_size
    D = config.hidden_size
    L = config.num_layers
    F = config.ffn_hidden_size

    embedding = V * D
    attention_per_layer = 4 * D * D
    mlp_per_layer = 2 * D * F
    norm_per_layer = 2 * D

    blocks = L * (
        attention_per_layer
        + mlp_per_layer
        + norm_per_layer
    )

    final_norm = D
    lm_head = D * V

    total = embedding + blocks + final_norm + lm_head

    return {
        "embedding": embedding,
        "attention": L * attention_per_layer,
        "mlp": L * mlp_per_layer,
        "block_norm": L * norm_per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }
```

PyTorch：

```python
def count_parameters_pytorch(model):
    return sum(p.numel() for p in model.parameters())
```

目标：

```text
Manual total : 8,268,032
PyTorch total: 8,268,032
Difference   : 0
```

如果不是 0，按下面顺序检查：

```text
Linear bias 是否存在？
Embedding 是否计算？
LM Head 是否计算？
RMSNorm weight 是否漏算？
Final Norm 是否漏算？
Embedding/LM Head 是否共享？
```

还可以：

```python
for name, p in model.named_parameters():
    print(name, tuple(p.shape), p.numel())
```

逐项与数学公式对应。

这一步非常重要，因为它训练的是：

> **公式模型必须能够和真实系统对象一一映射。**

---

# 15. Weight Memory

如果一个模型有 \(P\) 个参数，每个参数占 \(b\) Byte：

\[
M_{weights}=P\times b
\]

常见理论 dtype：

```python
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5,
}
```

例如 7B BF16：

\[
7\times10^9\times2=14\times10^9Bytes
\]

也就是：

```text
约 14 GB
约 13.04 GiB
```

必须区分：

\[
1GB=10^9Bytes
\]

\[
1GiB=1024^3Bytes
\]

因此：

```text
14 GB ≠ 14 GiB
```

---

# 16. Training Model-State Memory

训练时不只有 Weight。

最基础的模型状态可以拆成：

```text
Weights
Gradients
Optimizer m
Optimizer v
可能存在 FP32 Master Weights
```

一个常见的简化假设：

```text
BF16 parameter     2 Byte
BF16 gradient      2 Byte
FP32 Adam m        4 Byte
FP32 Adam v        4 Byte
```

所以：

\[
2+2+4+4=12Bytes/parameter
\]

则：

\[
M_{modelstate}\approx 12P
\]

但是必须明确：

> **12 Byte/parameter 不是普适定律。**

真实实现还与：

```text
Gradient dtype
Optimizer implementation
FP32 master weights
Mixed Precision strategy
ZeRO/FSDP sharding
CPU offload
```

有关。

因此 AI Infra 做资源估算时必须写清楚 assumption。

---

# 17. Model-State Memory 不等于训练 Peak Memory

这一点非常重要。

Experiment 2 估算的：

```text
Weights
Gradients
Optimizer States
```

属于模型状态。

真实训练 Peak Memory 还会包含：

```text
Activations
Autograd saved tensors
Temporary buffers
CUDA allocator overhead/cache
Kernel workspace
Logits
```

因此：

\[
TrainingPeakMemory \neq ModelStateMemory
\]

这个差异会在 Experiment 3 中直接测到。

---

# 18. FLOPs：先从 Linear 开始

对于矩阵：

\[
A_{m\times k}B_{k\times n}
\]

通常近似：

\[
2mkn
\]

FLOPs。

原因：一个输出元素大约包含：

```text
k 次乘法
+
k 次加法
```

因此 Linear：

输入：

```text
[B,S,Din]
```

权重：

```text
[Din,Dout]
```

FLOPs：

\[
2BSD_{in}D_{out}
\]

代码：

```python
def linear_flops(batch, seq, in_dim, out_dim):
    return 2 * batch * seq * in_dim * out_dim
```

---

# 19. Attention FLOPs

Attention 可以拆成三部分：

```text
Q/K/V/O Projection
QKᵀ
Attention Probability × V
```

## 19.1 Q/K/V/O Projection

四个 \(D\rightarrow D\) Linear：

\[
4\times 2BSD^2=8BSD^2
\]

## 19.2 QKᵀ

每个 Head：

\[
[S,D_h]\times[D_h,S]
\]

所有 Head：

\[
2BHS^2D_h
\]

因为：

\[
HD_h=D
\]

所以：

\[
2BS^2D
\]

## 19.3 Attention × V

同理：

\[
2BS^2D
\]

所以：

\[
\boxed{FLOPs_{Attention}\approx8BSD^2+4BS^2D}
\]

这里一个很重要的观察是：

> Head 数 \(H\) 在这个简化理论总 FLOPs 中被 \(HD_h=D\) 消掉了。

这不意味着 Head 数对实际性能没有影响。真实 GPU 性能还受：

```text
Head dimension
Kernel shape
Parallelism
Memory access pattern
```

影响。

---

# 20. MLP FLOPs

两个 Linear：

\[
D\rightarrow F
\]

和：

\[
F\rightarrow D
\]

因此：

\[
2BSDF+2BSFD=4BSDF
\]

所以：

\[
\boxed{FLOPs_{MLP}\approx4BSDF}
\]

暂时忽略 GELU 等 elementwise FLOPs，先抓主要 GEMM 成本。

---

# 21. 一个 Transformer Block 的主要 FLOPs

\[
\boxed{
FLOPs_{block}
\approx
8BSD^2+4BS^2D+4BSDF
}
\]

所有 Layer：

\[
L\times FLOPs_{block}
\]

LM Head：

\[
2BSDV
\]

因此完整模型还需要考虑 vocab projection。

对于 toy model，`V=10000`，LM Head 甚至可能是非常显眼的计算和内存来源。

---

# 22. Sequence Length Sweep：为什么要做

固定：

```text
B,D,H,F,L
```

改变：

```text
S=128,256,512,1024,2048,4096
```

观察 Attention 和 MLP FLOPs。

Attention：

\[
8BSD^2+4BS^2D
\]

其中包含 \(S^2\)。

MLP：

\[
4BSDF
\]

仅线性随 \(S\) 增长。

因此随着 sequence 变长：

```text
短序列：Projection / MLP 占比更明显
        ↓
长序列：QK / AV 的 S² 成本开始主导
        ↓
Attention 占比持续提高
```

这就是后续学习：

```text
Attention Alternatives
FlashAttention
Long Context Optimization
```

的直接动机。

---

# 23. Experiment 2 的核心工程思想

Experiment 2 最重要的不是背公式，而是形成下面的习惯：

```text
模型结构
  ↓
列出 Tensor / Weight Shape
  ↓
写参数公式
  ↓
写 FLOPs 公式
  ↓
声明 dtype / optimizer assumption
  ↓
得到 memory estimate
  ↓
用真实 PyTorch model 验证
```

这套方法以后会迁移到：

```text
Llama 7B / 70B
FSDP
Tensor Parallel
KV Cache
vLLM Serving Capacity
MoE
```

---

# 24. Experiment 3｜Runtime Profiling

## 24.1 实验目标

Experiment 2 回答：

> 理论上模型应该消耗多少资源？

Experiment 3 回答：

> PyTorch 和 GPU 实际上怎么执行？实际时间和峰值显存是多少？

核心链路：

```text
Transformer Python Module
    ↓
PyTorch Operator / ATen
    ↓
CUDA Kernel
    ↓
GPU Runtime
```

Week 1 暂时只要求：

```text
会测
会看
会解释
```

真正深入 Nsight / Roofline / SM throughput / DRAM throughput 放到后续 GPU 性能阶段。

---

# 25. GPU 异步执行：为什么不能直接 `time.time()`

CUDA 默认是异步执行模型。

CPU：

```text
launch kernel 1
launch kernel 2
launch kernel 3
继续执行 Python
```

GPU：

```text
kernel 1 → kernel 2 → kernel 3
```

CPU 发出 kernel launch 后不一定等待 GPU 真正执行完。

因此：

```python
start = time.time()
model(input_ids)
end = time.time()
```

测到的时间可能主要是 CPU 提交任务的时间，而不是 GPU 完整执行时间。

需要：

```python
torch.cuda.synchronize()
```

让 CPU 等 GPU 当前任务完成。

更推荐使用：

```python
torch.cuda.Event(enable_timing=True)
```

进行 CUDA 时间测量。

---

# 26. Warmup：为什么 Benchmark 不能测第一轮

第一轮 Forward 可能混入：

```text
CUDA Context 初始化
Library 初始化
Kernel 首次加载
Memory Allocation
Cache Cold Start
```

这些不属于模型 steady-state 性能。

因此 Benchmark 流程：

```text
Warmup
   ↓
Synchronize
   ↓
Measure
   ↓
Synchronize
```

示例：

```python
def warmup(model, input_ids, num_warmup=10):
    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()
```

---

# 27. Baseline Forward Timing

使用 CUDA Event：

```python
def benchmark_forward(
    model,
    input_ids,
    warmup_iters=10,
    measure_iters=50,
):
    warmup(model, input_ids, warmup_iters)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    with torch.inference_mode():
        for _ in range(measure_iters):
            _ = model(input_ids)

    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    return total_ms / measure_iters
```

为什么要测多次平均，而不是只测一次？

因为单次时间可能受到：

```text
OS 调度
GPU transient state
cache
background workload
```

影响。

后续更严谨的 benchmark 还会关注：

```text
P50
P95
P99
variance
```

Week 1 先建立稳定测量习惯。

---

# 28. Benchmark 与 Profiler 的区别

非常重要：

## Benchmark

目的：

> **尽可能准确地回答“它有多快？”**

使用：

```text
CUDA Event
warmup
多次测量
最小额外开销
```

## Profiler

目的：

> **回答“时间花在哪里？”**

Profiler 会增加额外开销，因此不应该直接把 profiler 的总时间当作最终性能结果。

可以简单记：

```text
Benchmark = how fast?
Profiler  = where time goes?
```

---

# 29. `torch.profiler`：第一次看 PyTorch 的真实 Operator

基本使用：

```python
from torch.profiler import profile, ProfilerActivity, record_function

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_flops=True,
) as prof:
    with torch.inference_mode():
        with record_function("mini_transformer_forward"):
            _ = model(input_ids)
```

输出：

```python
print(
    prof.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
)
```

---

# 30. 为什么 Profiler 里看到的是 `aten::xxx`

模型代码写的是：

```python
self.q_proj(x)
```

但底层执行链路可能是：

```text
nn.Linear
   ↓
aten::linear
   ↓
aten::mm / matmul
   ↓
CUDA GEMM Kernel
```

因此必须开始建立映射：

| 模型结构 | Profiler 中可能看到 |
|---|---|
| Embedding | `aten::embedding` |
| Q/K/V/O Linear | `aten::linear`, `aten::mm` |
| QKᵀ | `aten::matmul`, `aten::bmm` |
| Softmax | `aten::_softmax` |
| Attention × V | `aten::matmul`, `aten::bmm` |
| RMSNorm | `pow`, `mean`, `rsqrt`, `mul` |
| MLP | `linear`, `mm`, `gelu` |
| Residual | `add` |
| LM Head | `linear`, `mm` |

这就是第一次理解：

> **TransformerBlock 本身不是一个 GPU Kernel。**

它会被拆成大量 Operator，再最终映射到 CUDA kernels。

---

# 31. RMSNorm 为什么会拆成多个 Operator

我们的 RMSNorm：

```python
mean_square = x.pow(2).mean(dim=-1, keepdim=True)
x = x * torch.rsqrt(mean_square + eps)
return x * weight
```

Profiler 中可能看到：

```text
pow
mean
add
rsqrt
mul
mul
```

这意味着 eager PyTorch 的高层 RMSNorm 可能需要多个 operator / kernel 才完成。

这为后续学习 Kernel Fusion 提供非常直观的动机：

```text
多个小 Operator
   ↓
多次 Kernel Launch
   ↓
中间 Tensor 读写
   ↓
额外 HBM Traffic
```

如果写成一个 fused RMSNorm kernel，就可能减少：

```text
launch overhead
intermediate memory traffic
```

Week 1 暂时只观察，不优化。

---

# 32. `record_function()`：给 Profiler 插入高层路标

Profiler 默认显示大量 `aten::xxx`，很难快速知道它属于哪个模型模块。

因此可使用：

```python
with record_function("attention_qkv_projection"):
    q = self.q_proj(x)
    k = self.k_proj(x)
    v = self.v_proj(x)
```

类似：

```text
attention_qkv_projection
attention_qk_matmul
attention_softmax
attention_av_matmul
attention_output_projection
mlp
transformer_block_0
```

这样 Trace 会更清楚：

```text
transformer_block_0
├── attention_qkv_projection
├── attention_qk_matmul
├── attention_softmax
├── attention_av_matmul
├── attention_output_projection
└── mlp
```

这是一种非常实用的系统观测方法：

> **先给 workload 做语义分区，再看每个区间内部到底执行了什么。**

---

# 33. `Self CUDA` 与 `CUDA Total`

Profiler 中常见：

```text
Self CUDA
CUDA Total
```

可以理解为：

## CUDA Total

当前事件 + 所有子事件累计 CUDA 时间。

## Self CUDA

只属于当前事件本身的 CUDA 时间，不包括子事件。

例如：

```text
mini_transformer_forward
    ├── transformer_block_0
    ├── transformer_block_1
    └── lm_head
```

`mini_transformer_forward` 本身只是一个人为 marker，并没有直接执行某个 CUDA kernel，因此：

```text
CUDA Total 可能很大
Self CUDA 可能接近 0
```

这是正常的。

---

# 34. 导出 Trace

Profiler 可以导出时间线：

```python
prof.export_chrome_trace(
    "profiler_output/forward_trace.json"
)
```

可以使用兼容 Chrome Trace 的工具查看，例如 Perfetto。

你会看到类似：

```text
CPU timeline
  ├─ linear launch
  ├─ matmul launch
  ├─ softmax launch
  └─ ...

GPU timeline
  ├─ GEMM kernel
  ├─ bmm kernel
  ├─ softmax kernel
  └─ ...
```

这是第一次真正看到：

> **Python / PyTorch 是如何驱动 GPU 执行大量 Kernel 的。**

---

# 35. CUDA Memory：Allocated 与 Reserved

PyTorch 使用 CUDA caching allocator。

可以先粗略理解：

```text
GPU Memory
├── Reserved
│   PyTorch 从 CUDA 申请并纳入 allocator 管理的内存
│
└── Allocated
    当前真正分配给活跃 Tensor 的内存
```

所以通常：

\[
Reserved \ge Allocated
\]

测峰值：

```python
torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    _ = model(input_ids)

torch.cuda.synchronize()

peak_allocated = torch.cuda.max_memory_allocated()
peak_reserved = torch.cuda.max_memory_reserved()
```

---

# 36. Inference Peak Memory

示例：

```python
def measure_inference_memory(model, input_ids):
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        _ = model(input_ids)

    torch.cuda.synchronize()

    return {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }
```

Experiment 2 可能告诉你：

```text
BF16 Weight ≈ 15.77 MiB
```

但 Experiment 3 的 peak allocated 可能显著更高。

原因：真实 Forward 还包含：

```text
Input
Embedding output
Q/K/V
Attention scores
Softmax output
MLP intermediate
Logits
Temporary buffers
```

所以：

\[
PeakMemory \neq WeightMemory
\]

---

# 37. Toy Model 中 LM Head / Logits 的显存陷阱

当前模型：

```text
vocab_size = 10000
```

输出：

```text
[B,S,V]
```

例如：

```text
B = 2
S = 1024
V = 10000
FP32 logits
```

Logits 自身内存：

\[
2\times1024\times10000\times4Bytes
\]

约 81.9 MB。

所以看到 sequence 增长时 peak memory 增加，不能简单说：

> “都是 Attention 的 \(S^2\) 导致的。”

还必须检查：

```text
Logits
MLP Intermediate
Q/K/V
Attention Matrix
```

这是 AI Infra 非常重要的推理方式：

> **不能只看理论复杂度标签，必须结合实际 Tensor Shape。**

---

# 38. Training Peak Memory

示例：

```python
def measure_training_memory(model, input_ids):
    model.train()
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()

    torch.cuda.synchronize()

    result = {
        "allocated": torch.cuda.max_memory_allocated(),
        "reserved": torch.cuda.max_memory_reserved(),
    }

    model.zero_grad(set_to_none=True)
    return result
```

通常：

\[
TrainingPeak > InferencePeak
\]

因为训练要保存：

```text
Forward activations
Autograd saved tensors
Gradients
Backward temporary tensors
```

注意：如果没有真正创建 AdamW optimizer，那么这个实验**没有测 optimizer state memory**。

因此：

- Experiment 2 的 Adam state 是理论 model-state accounting；
- Experiment 3 的 training peak 是 Forward + Backward 的实际 peak。

两个概念不要混淆。

---

# 39. Sequence Length Sweep：理论 vs 实测

测试：

```text
S = 128
S = 256
S = 512
S = 1024
```

记录：

```text
Forward Latency
Peak Allocated Memory
Peak Reserved Memory
```

结果表：

| Seq | Forward Latency | Peak Allocated | Peak Reserved |
|---:|---:|---:|---:|
| 128 |  |  |  |
| 256 |  |  |  |
| 512 |  |  |  |
| 1024 |  |  |  |

Experiment 2 理论预测：

\[
Attention_{QK/AV}\propto S^2
\]

但真实 Forward Latency 不一定严格：

```text
S ×2 → latency ×4
```

原因包括：

```text
GPU utilization
Kernel launch overhead
GEMM efficiency
Tensor Core utilization
Memory traffic
LM Head
MLP
Kernel shape
```

因此：

\[
Theoretical FLOPs \neq Observed Latency
\]

这正是为什么 AI Infra 不能只停留在算法复杂度分析。

---

# 40. Experiment 2 与 Experiment 3 的对应关系

| 问题 | Experiment 2 | Experiment 3 |
|---|---|---|
| 参数量 | 数学公式 | `p.numel()` 验证 |
| Weight Memory | `P × dtype bytes` | GPU allocator 中的一部分 |
| Training State | 理论估算 | 不直接等于实际 peak |
| FLOPs | 理论估算 | Profiler 可辅助观察 |
| Runtime | 无法由 FLOPs 单独知道 | CUDA Event 实测 |
| Operator | 模型结构推断 | `torch.profiler` 直接观察 |
| Peak Memory | 粗略推断 | `max_memory_allocated()` 实测 |
| Bottleneck | 只能提出假设 | 开始从 Trace / Operator 找证据 |

真正的 AI Infra 学习必须同时具备：

\[
\boxed{Modeling + Measurement}
\]

---

# 41. Week 1 最终形成的完整系统链路

完成三个实验后，应该能够把下面这条链路讲清楚：

```text
Transformer
   ↓
Embedding / Attention / MLP / Norm
   ↓
Tensor Shape
   ↓
Parameter Count
   ↓
FLOPs / Memory Estimate
   ↓
PyTorch Module
   ↓
ATen Operator
   ↓
CUDA Kernel
   ↓
GPU Runtime / Memory
```

这就是 Week 1 最重要的学习成果。

---

# 42. 三个实验最重要的知识点总结

## Experiment 1：模型实现

必须真正掌握：

```text
[B,S,D]
[B,H,S,Dh]
[B,H,S,S]
```

之间如何变化。

重点知识：

- RMSNorm 的数学定义；
- `torch.rsqrt()`；
- Broadcasting；
- Multi-Head reshape；
- `transpose()` 与 stride；
- `contiguous()` 与物理内存；
- Causal Mask；
- Residual；
- Forward / Backward sanity check。

---

## Experiment 2：资源核算

必须掌握：

\[
P_{attention}=4D^2
\]

\[
P_{MLP}=2DF
\]

\[
FLOPs_{Attention}\approx8BSD^2+4BS^2D
\]

\[
FLOPs_{MLP}\approx4BSDF
\]

重点知识：

- 参数量和显存不是一回事；
- Weight Memory 和 Training Peak Memory 不是一回事；
- GB / GiB 的差异；
- 训练 model state 必须声明 optimizer / dtype assumption；
- Attention 中的 \(S^2\)；
- Manual estimate 必须用真实模型验证。

---

## Experiment 3：运行时观测

必须掌握：

- CUDA asynchronous execution；
- `torch.cuda.synchronize()`；
- Warmup；
- CUDA Event；
- Benchmark vs Profiler；
- PyTorch Module → ATen Operator → CUDA Kernel；
- `record_function()`；
- Allocated vs Reserved；
- Inference Peak vs Training Peak；
- 理论 FLOPs 和真实 latency 不完全等价。

---

# 43. 常见错误与排查方式

## 43.1 Manual 参数量和 PyTorch 不一致

排查：

```text
bias
Embedding
LM Head
RMSNorm scale
Final Norm
Weight Tying
```

---

## 43.2 `view()` 报 stride 错误

常见原因：前面做了 `transpose()`，Tensor non-contiguous。

解决：

```python
x = x.transpose(...).contiguous().view(...)
```

但要知道 `contiguous()` 可能引发真实内存 copy。

---

## 43.3 GPU 计时极小或不稳定

排查：

```text
是否 warmup？
是否 synchronize？
是否使用 CUDA Event？
是否只测 1 次？
GPU 上是否有其他任务？
```

---

## 43.4 Peak Memory 比理论 Weight 大很多

检查：

```text
logits
Q/K/V
attention scores
MLP intermediate
training activations
allocator reserved memory
```

不要把所有增长都归因于 Attention。

---

## 43.5 Profiler 很慢

这是正常的。Profiler 本身有开销。

特别是：

```text
record_shapes
profile_memory
with_stack
with_flops
```

都可能增加开销。

所以：

> **Profiler 用于定位，CUDA Event Benchmark 用于性能数字。**

---

# 44. Week 1 工程验收题

完成 Week 1 后，应能够不看笔记回答：

1. 为什么 7B BF16 权重约 14 GB？
2. 为什么简单 Attention 参数量是 \(4D^2\)？
3. 为什么简单 MLP 参数量是 \(2DF\)？
4. 为什么 Attention 出现 \(S^2\)？
5. 为什么理论 QK FLOPs 中 Head 数最终可以消掉？
6. 为什么 RMSNorm 对 `dim=-1` 做 mean？
7. `torch.rsqrt()` 做什么？
8. `transpose()` 为什么通常不复制数据？
9. `contiguous()` 的目的是什么？
10. 为什么 `.view()` 常常要求 contiguous-compatible stride？
11. 为什么训练显存远高于 Weight Memory？
12. 为什么 `12 Bytes/parameter` 不能当成永远成立的规则？
13. 为什么不能直接用 `time.time()` 测 CUDA Forward？
14. Benchmark 和 Profiler 的作用分别是什么？
15. `aten::mm` 与 `nn.Linear` 有什么关系？
16. 为什么一个 TransformerBlock 不等于一个 CUDA Kernel？
17. 为什么 `reserved >= allocated` 很常见？
18. 为什么理论 FLOPs 增长不一定等于 latency 同比例增长？
19. 如果 Manual 参数量与 PyTorch 不同，应如何排查？
20. 如果 Sequence Length 增长后显存异常升高，应检查哪些 Tensor？

如果至少 80% 能够用“机制 + 数据流 + 公式”回答，Week 1 工程部分就可以认为真正完成。

---

# 45. Week 1 最终工程产物 Checklist

```text
[ ] Mini Transformer 能 Forward
[ ] Mini Transformer 能 Backward
[ ] RMSNorm 实现正确
[ ] Attention Shape Trace 完整
[ ] causal mask 正确
[ ] Manual 参数 = PyTorch 参数
[ ] BF16 / FP32 Weight Memory 可计算
[ ] Training Model-State Memory 可计算
[ ] Attention / MLP / LM Head FLOPs 可计算
[ ] Sequence FLOPs Sweep 完成
[ ] CUDA Event Forward Benchmark 完成
[ ] torch.profiler Operator 表输出成功
[ ] Trace JSON 导出成功
[ ] Inference Peak Memory 完成
[ ] Training Peak Memory 完成
[ ] Sequence Runtime/Memory Sweep 完成
[ ] 能解释三组实验之间的关系
```

---

# 46. 从 Week 1 如何自然过渡到 Week 2

Week 1 已经暴露了两个非常重要的问题。

## 问题一：Attention 为什么是 \(S^2\)？

我们已经看到：

\[
QK^T\rightarrow[B,H,S,S]
\]

因此长 Context 会越来越贵。

Week 2 的 `Attention Alternatives` 会开始讨论：

> 标准 Attention 是否是唯一选择？有哪些架构试图改变这种计算方式？

---

## 问题二：推理时为什么要重复计算历史 Token？

当前 naive model 每次 Forward 都会重新构造全部 Q/K/V。

但 autoregressive generation 中，历史 token 的 K/V 实际上不会改变。

于是 Week 2 会引出：

```text
Autoregressive Generation
    ↓
Prefill
    ↓
KV Cache
    ↓
Decode
```

然后继续追问：

```text
为什么 Prefill 和 Decode 的性能特征不同？
为什么 Decode 经常更受显存带宽影响？
GQA 为什么可以降低 KV Cache？
```

因此 Week 1 不是孤立实验，而是整个 AI Infra 学习路线的第一层基础。

---

# 47. Week 1 最终结论

完成三个实验后，最重要的收获不是：

> “我写过一个 Transformer。”

而是建立了第一套 AI Infra 问题分析方法：

```text
先理解模型结构
    ↓
追踪 Tensor Shape
    ↓
建立资源公式
    ↓
声明 Assumption
    ↓
用真实实现验证
    ↓
用 Benchmark 测量
    ↓
用 Profiler 观察
    ↓
解释理论与实测差异
```

可以总结为：

\[
\boxed{
Problem
\rightarrow
Model
\rightarrow
Resource
\rightarrow
Measurement
\rightarrow
Evidence
}
\]

这就是后续 CUDA、Triton、Distributed Training、vLLM/SGLang 性能工程最核心的底层方法论。
