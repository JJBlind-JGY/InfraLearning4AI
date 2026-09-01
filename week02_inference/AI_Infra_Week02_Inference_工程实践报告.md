# Week 02｜LLM Inference Fundamentals 工程实践报告

> 主题：Autoregressive Generation、Prefill / Decode、KV Cache、KV Cache Memory Accounting、Generation Benchmark  
> 目标：从 Week 01 的“完整 Transformer Forward”继续向真实 LLM Inference 执行模式推进，建立 **模型执行 → KV Cache → GPU Memory → Runtime** 的完整系统认知。

---

# 0. 本周工程目标

Week 02 的三个实验不是互不相关的小作业，而是一条完整工程链路：

```text
Week 01 MiniTransformer
        ↓
Naive Autoregressive Generation
        ↓
发现历史 Token 被重复计算
        ↓
引入 KV Cache
        ↓
实现 Cached Generation
        ↓
计算 KV Cache 显存
        ↓
测量 Prefill / Decode 时间
        ↓
理解真实 LLM Inference 的资源特征
```

本周完成：

1. 在 Week 01 模型上实现 `naive generation` 与 `KV-cache generation`
2. 实现 `kv_cache_calculator.py`
3. 对 Prompt Length `128 / 512 / 2048` 测量 `Prefill / Decode` 时间趋势

最终建立：


\boxed{\text{Autoregressive Generation}\rightarrow\text{Prefill / Decode}\rightarrow\text{KV Cache}\rightarrow\text{GPU Memory / Bandwidth}}


---

# 1. 工程目录

```text
week02_inference/
├── kv_cache_demo.py
├── kv_cache_calculator.py
├── benchmark_generation.py
├── benchmark_results.csv
└── week02_report.md
```

Week 01 模型需要增加 KV Cache 支持：

```text
week01_llm_systems/model/
├── attention.py
├── block.py
└── transformer.py
```

职责：

- `attention.py`：保存、拼接和返回历史 K/V
- `block.py`：向本层 Attention 传递 Cache
- `transformer.py`：管理所有 Layer 的 Cache

---

# 2. Experiment 1｜Naive Generation 与 KV-Cache Generation

## 2.1 实验目标

Week 01 模型只实现：

```python
logits = model(input_ids)
```

输入：

```text
[B, S]
```

输出：

```text
[B, S, V]
```

Week 02 要让模型真正完成：

```text
Prompt → 预测 Token → 追加 Token → 再次预测 → ...
```

即 Autoregressive Generation。

---

# 3. Naive Autoregressive Generation

假设 Prompt：

```text
[A, B, C]
```

生成：

```text
[A, B, C] → Transformer → D
[A, B, C, D] → Transformer → E
[A, B, C, D, E] → Transformer → F
```

数学上：

\[
P(x_t|x_1,\dots,x_{t-1})
\]

标准实现：

```python
@torch.inference_mode()
def naive_generate(model, input_ids, max_new_tokens):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
    return generated
```

其中：

```python
logits[:, -1, :]
```

表示当前序列最后一个位置对“下一个 Token”的预测。

---

# 4. Naive Generation 的核心问题

假设：

```text
Prompt Length = 128
Generate = 4
```

计算过程：

```text
Step 1 → Forward 128 Tokens
Step 2 → Forward 129 Tokens
Step 3 → Forward 130 Tokens
Step 4 → Forward 131 Tokens
```

历史 Token 的 K/V 和其它中间计算不断被重复执行。

尤其历史：

\[
K_1,K_2,\dots,K_{128}
\]

\[
V_1,V_2,\dots,V_{128}
\]

不会因为新 Token 出现而改变。

因此最直接的优化是：

> **保存历史 K/V，后续 Decode 直接复用。**

---

# 5. 为什么缓存 K/V，而不是 Q？

当前 Token \(t\) 产生：

\[
Q_t,K_t,V_t
\]

下一步 \(t+1\) 需要新的：

\[
Q_{t+1}
\]

去查询历史：

\[
K_1,\dots,K_t
\]

并读取：

\[
V_1,\dots,V_t
\]

历史 Query 完成当前 Attention 后不会被未来 Token 再使用，因此：

\[
\boxed{\text{Cache K/V，而不是 Q}}
\]

系统直觉：

```text
Historical K/V → 被未来 Query 重复访问
Historical Q   → 当前 Step 结束后失去用途
```

---

# 6. KV Cache 的数据结构

一个 Layer 中：

\[
K,V\in\mathbb{R}^{B\times H_{KV}\times T\times d_h}
\]

当前实验使用 MHA，因此：

\[
H_{KV}=H
\]

代码可表示为：

```python
past_key_value = (past_k, past_v)
```

其中：

```text
past_k: [B, H, PAST_LEN, Dh]
past_v: [B, H, PAST_LEN, Dh]
```

---

# 7. 为什么每一层都必须有独立 KV Cache？

不同层：

\[
K^{(l)}=H^{(l)}W_K^{(l)}
\]

不同 Layer 的：

- Hidden State 不同
- \(W_K/W_V\) 不同

所以：

\[
K^{(0)}\neq K^{(1)}
\]

因此实际 Cache 是：

```text
Layer 0 → K0,V0
Layer 1 → K1,V1
Layer 2 → K2,V2
...
```

这也是后面 KV Cache Memory 公式出现 \(L\) 的原因。

---

# 8. 修改 Attention 接口

Week 01：

```python
def forward(self, x):
```

Week 02：

```python
def forward(self, x, past_key_value=None, use_cache=False):
```

含义：

- `past_key_value=None`：没有历史 Cache，例如 Prefill
- `use_cache=True`：本次 Forward 返回更新后的 Cache
- `use_cache=False`：保持普通 Forward 行为

---

# 9. Q/K/V Projection 与 Head 拆分

输入：

```text
x: [B, Q_LEN, D]
```

Projection：

```python
q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
```

拆 Head：

```python
q = q.view(B, Q_LEN, self.num_heads, self.head_dim).transpose(1, 2)
k = k.view(B, Q_LEN, self.num_heads, self.head_dim).transpose(1, 2)
v = v.view(B, Q_LEN, self.num_heads, self.head_dim).transpose(1, 2)
```

得到：

```text
Q: [B, H, Q_LEN, Dh]
K: [B, H, Q_LEN, Dh]
V: [B, H, Q_LEN, Dh]
```

---

# 10. 追加历史 Cache

```python
if past_key_value is not None:
    past_k, past_v = past_key_value
    k = torch.cat([past_k, k], dim=2)
    v = torch.cat([past_v, v], dim=2)
```

Shape：

```text
past_k: [B,H,PAST_LEN,Dh]
new_k : [B,H,Q_LEN,Dh]
```

合并：

```text
k: [B,H,KV_LEN,Dh]
```

其中：

\[
KV\_LEN=PAST\_LEN+Q\_LEN
\]

---

# 11. Prefill 与 Decode Shape

## Prefill

```text
PAST_LEN = 0
Q_LEN = Prompt Length
KV_LEN = Prompt Length
```

例如：

```text
Q: [1,8,128,32]
K: [1,8,128,32]
V: [1,8,128,32]
```

## Decode

若已有 128 个 Token：

```text
PAST_LEN = 128
Q_LEN = 1
KV_LEN = 129
```

那么：

```text
Q: [1,8,1,32]
K: [1,8,129,32]
V: [1,8,129,32]
```

于是：

\[
QK^T:[1,8,1,129]
\]

这就是 Prefill 和 Cached Decode 最核心的 Shape 差异。

---

# 12. 最容易写错的工程细节：Cached Causal Mask

Week 01 的普通 Mask：

```python
torch.triu(torch.ones(S, S), diagonal=1)
```

只适合：

```text
Q_LEN = KV_LEN = S
```

Decode 时：

```text
Q_LEN = 1
KV_LEN = PAST_LEN + 1
```

当前 Query 的绝对位置是：

```text
position = PAST_LEN
```

因此必须基于绝对位置生成 Mask。

---

# 13. 正确的 Cached Causal Mask

```python
past_len = KV_LEN - Q_LEN
q_positions = torch.arange(past_len, past_len + Q_LEN, device=x.device)
k_positions = torch.arange(KV_LEN, device=x.device)
causal_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))
```

核心规则：

\[
Mask(q,k)=k>q
\]

即：Key 的绝对位置晚于 Query 时禁止访问。

---

# 14. 为什么普通上三角 Mask 会错？

假设：

```text
PAST_LEN = 4
Q_LEN = 1
KV_LEN = 5
```

当前 Query 实际是 position 4，应当看到：

```text
K0 K1 K2 K3 K4
✓  ✓  ✓  ✓  ✓
```

如果错误地把 Query 当作 position 0，就会把合法历史 Token 屏蔽掉。

因此：

\[
\boxed{\text{Cached Attention 的位置必须考虑 past\_len}}
\]

---

# 15. Attention Output

```python
scores = q @ k.transpose(-2, -1)
scores = scores / math.sqrt(self.head_dim)
attn = torch.softmax(scores, dim=-1)
out = attn @ v
```

Shape：

```text
scores: [B,H,Q_LEN,KV_LEN]
attn  : [B,H,Q_LEN,KV_LEN]
out   : [B,H,Q_LEN,Dh]
```

恢复：

```python
out = out.transpose(1, 2).contiguous().view(B, Q_LEN, D)
out = self.o_proj(out)
```

---

# 16. 返回 Cache

```python
return (out, (k, v)) if use_cache else out
```

注意返回的是已经合并完成的全部 K/V，而不是仅当前 Token 的 K/V。

---

# 17. 修改 Transformer Block

Block 的职责很简单：把本层 Cache 传给本层 Attention，并把更新后的 Cache 返回。

核心代码：

```python
normed_x = self.norm1(x)
if use_cache:
    attn_out, present_key_value = self.attn(normed_x, past_key_value=past_key_value, use_cache=True)
else:
    attn_out = self.attn(normed_x, past_key_value=past_key_value, use_cache=False)
    present_key_value = None
x = x + attn_out
x = x + self.mlp(self.norm2(x))
return (x, present_key_value) if use_cache else x
```

---

# 18. 修改 MiniTransformer

若没有历史 Cache：

```python
if past_key_values is None:
    past_key_values = [None] * len(self.layers)
```

逐层执行：

```python
for layer, layer_past in zip(self.layers, past_key_values):
```

Cached 模式：

```python
x, layer_present = layer(x, past_key_value=layer_past, use_cache=True)
present_key_values.append(layer_present)
```

最终：

```python
return (logits, tuple(present_key_values)) if use_cache else logits
```

---

# 19. 两种 Forward 模式

普通：

```python
logits = model(input_ids)
```

Cached：

```python
logits, past_key_values = model(input_ids, use_cache=True)
```

要求：

```python
assert len(past_key_values) == config.num_layers
```

---

# 20. KV-Cache Generation

必须显式分成：

```text
Prefill → Decode
```

Prefill：

```python
logits, past_key_values = model(input_ids, use_cache=True)
```

Decode：

```python
logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
```

每个 Decode Step 只输入：

```text
[B,1]
```

---

# 21. Cached Generation 标准实现

```python
@torch.inference_mode()
def kv_cache_generate(model, input_ids, max_new_tokens):
    generated = input_ids.clone()
    logits, past_key_values = model(input_ids, use_cache=True)
    for step in range(max_new_tokens):
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if step == max_new_tokens - 1: break
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)
    return generated
```

---

# 22. Naive 与 Cached 的计算差异

Naive：

```text
Step 1 → 128 tokens
Step 2 → 129 tokens
Step 3 → 130 tokens
...
```

Cached：

```text
Prefill → 128 tokens
Decode 1 → 1 token + cached 128 KV
Decode 2 → 1 token + cached 129 KV
Decode 3 → 1 token + cached 130 KV
...
```

因此：

\[
\boxed{KV\ Cache\ 用显存换取历史 K/V 重复计算的消除}
\]

---

# 23. Correctness Test：Token 一致性

```python
naive_output = naive_generate(model, prompt, max_new_tokens=8)
cached_output = kv_cache_generate(model, prompt, max_new_tokens=8)
assert torch.equal(naive_output, cached_output)
```

优化后首先必须保证功能一致。

---

# 24. 更严格的 Correctness Test：Logits

比较：

```text
Naive：完整 Prompt + 新 Token
Cached：新 Token + 历史 Cache
```

使用：

```python
max_diff = (naive_next_logits - cached_next_logits).abs().max()
same = torch.allclose(naive_next_logits, cached_next_logits, atol=1e-5, rtol=1e-4)
```

---

# 25. 为什么推荐 `torch.allclose()`？

GPU 浮点运算并不满足严格结合律：

\[
(a+b)+c\neq a+(b+c)
\]

Full Forward 与 Incremental Decode 的矩阵 Shape 不完全相同，底层运算顺序可能产生极小误差，因此数值正确性更适合使用容差比较。

---

# 26. Toy Model 的限制

当前 MiniTransformer：

```text
随机初始化
没有训练
没有 Tokenizer
没有 RoPE
```

所以生成 Token 没有语言语义是正常的。

本实验研究的是：

\[
\boxed{\text{Inference Execution Pattern}}
\]

而不是模型质量。

---

# 27. Experiment 2｜KV Cache Calculator

## 27.1 目标

把 KV Cache 从抽象概念变成可以准确估算的 GPU Memory Resource。

一层：

\[
K,V\in\mathbb{R}^{B\times H_{KV}\times T\times d_h}
\]

所有 Layer 总元素：

\[
N_{KV}=2LBTH_{KV}d_h
\]

显存：

\[
\boxed{M_{KV}=2LBTH_{KV}d_h\times Bytes_{dtype}}
\]

---

# 28. 参数含义

| 变量 | 含义 |
|---|---|
| 2 | K + V |
| L | Transformer Layer 数 |
| B | Batch / Concurrent Requests |
| T | Context Length |
| Hkv | KV Head 数 |
| Dh | Head Dimension |
| Bytes | DType 字节数 |

---

# 29. DType 映射

```python
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}
```

`INT4 = 0.5 Byte` 是理论 packing 值，真实系统还可能包含 scale、metadata、alignment 等额外开销。

---

# 30. Calculator 核心实现

```python
def kv_cache_bytes(batch, seq, layers, kv_heads, head_dim, dtype): return 2 * batch * seq * layers * kv_heads * head_dim * DTYPE_BYTES[dtype]
```

每 Token：

```python
def kv_cache_bytes_per_token(batch, layers, kv_heads, head_dim, dtype): return kv_cache_bytes(batch, 1, layers, kv_heads, head_dim, dtype)
```

---

# 31. Week 01 Toy Model 标准计算

配置：

```text
B = 1
L = 4
Hkv = 8
Dh = 32
dtype = BF16
```

因为：

\[
D=256,H=8,d_h=32
\]

Prompt 128：

\[
2\times1\times128\times4\times8\times32\times2=524288Bytes
\]

所以：

\[
\boxed{512KiB=0.5MiB}
\]

---

# 32. 128 / 512 / 2048 标准答案

| Seq | BF16 KV Cache |
|---:|---:|
| 128 | 0.5 MiB |
| 512 | 2 MiB |
| 2048 | 8 MiB |

因此：

\[
\boxed{KVCache\ Capacity\propto T}
\]

即：

\[
O(T)
\]

而不是：

\[
O(T^2)
\]

---

# 33. Attention Complexity 与 KV Capacity 的区别

Dense Prefill Attention 的核心矩阵：

\[
QK^T\in\mathbb{R}^{S\times S}
\]

因此计算存在：

\[
O(S^2)
\]

KV Cache 只保存每个 Token 的 K/V，因此容量是：

\[
O(S)
\]

必须明确区分：

```text
Attention Compute ≠ KV Cache Capacity
```

---

# 34. MHA / GQA / MQA 的 KV 对比

Toy Model：

\[
H_Q=8
\]

2K Context：

| Architecture | Hq | Hkv | KV Cache |
|---|---:|---:|---:|
| MHA | 8 | 8 | 8 MiB |
| GQA 示例 | 8 | 2 | 2 MiB |
| MQA | 8 | 1 | 1 MiB |

因为：

\[
\boxed{M_{KV}\propto H_{KV}}
\]

所以 GQA / MQA 的系统价值之一就是显著减少 KV Cache。

---

# 35. Calculator 与真实 Cache Tensor 交叉验证

```python
def cache_nbytes(past_key_values):
    total = 0
    for k, v in past_key_values:
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total
```

运行：

```python
logits, cache = model(prompt, use_cache=True)
actual_bytes = cache_nbytes(cache)
```

理论值应与 Tensor 实际容量一致。

如果模型是 FP32：

```text
element_size() = 4
```

因此实际 Cache 是同 Shape BF16 Cache 的 2 倍。

---

# 36. Experiment 3｜Prefill / Decode Benchmark

## 36.1 实验目标

验证：

1. Prompt 越长，Prefill 时间如何变化
2. Context 越长，Decode ms/token 如何变化
3. Naive 与 KV-Cache Generation 的性能差异如何变化

---

# 37. 固定实验条件

建议固定：

```text
Batch Size = 1
Generated Tokens = 固定
Model = 同一个 MiniTransformer
Device = 同一 GPU
DType = 相同
Sampling = Greedy
Prompt = Random Tokens
Warmup = 相同
Repeats = 相同
```

否则数据不能直接比较。

---

# 38. Prefill Time 的实验定义

```text
Prompt
↓
model(prompt, use_cache=True)
↓
建立所有 Layer KV Cache
↓
获得最后位置 Logits
```

测得：

\[
PrefillLatency
\]

它可以看作：

```text
Toy TTFT Proxy
```

但真实线上 TTFT 还包含 Queue、Scheduler、Tokenizer、Network、Sampling 等因素。

---

# 39. Decode Time 的实验定义

Prefill 后，每一步：

```text
[B,1] Current Token
+
Past KV
↓
Model
↓
Next Token
```

统计：

\[
Average\ Decode\ Time/Token
\]

可以视为：

```text
Toy TPOT Proxy
```

---

# 40. CUDA Benchmark 为什么需要同步？

CUDA 默认异步。

如果直接：

```python
start = time.time(); model(x); end = time.time()
```

CPU 可能只统计 Kernel Launch，而 GPU 尚未执行结束。

因此需要：

```python
torch.cuda.synchronize()
```

或 CUDA Event。

---

# 41. 推荐 Timing 函数

```python
def timed_call(fn, device):
    if device.type == "cuda":
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); start.record(); result = fn(); end.record(); torch.cuda.synchronize()
        return result, start.elapsed_time(end)
    start = time.perf_counter(); result = fn(); return result, (time.perf_counter() - start) * 1000
```

---

# 42. 为什么必须 Warmup？

第一次运行可能包含：

```text
CUDA Context 初始化
Kernel 加载
Allocator 初始化
Cache Cold Start
```

这些不是 steady-state 性能。

正确 Benchmark：

```text
Warmup
↓
重复正式测量
↓
取 Median / Average
```

---

# 43. 为什么推荐 Median？

Benchmark 可能受到：

```text
OS Scheduling
Background Process
Clock Variation
Runtime Noise
```

影响。

Median 对偶发离群值更稳健：

```python
statistics.median(results)
```

---

# 44. Naive Benchmark 的实际工作

```python
for _ in range(max_new_tokens):
    logits = model(generated)
    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
    generated = torch.cat([generated, next_token], dim=1)
```

每一步 Context 都增长，且整个历史重新计算。

---

# 45. Cached Benchmark 的实际工作

Prefill：

```python
logits, cache = model(prompt, use_cache=True)
```

Decode：

```python
logits, cache = model(current_token, past_key_values=cache, use_cache=True)
```

每次只处理一个新 Token，因此可以单独统计：

```text
Prefill ms
Decode total ms
Decode ms/token
```

---

# 46. 推荐 Benchmark 输出

```text
prompt_len
prefill_ms
decode_ms_per_token
cached_total_ms
naive_total_ms
speedup
```

其中：

\[
Speedup=\frac{NaiveTotal}{CachedTotal}
\]

---

# 47. Prompt Sweep

本周固定：

```python
prompt_lengths = [128, 512, 2048]
```

第一次调试建议先：

```text
128 / 512
```

确认正确后再加入 2048。

---

# 48. Benchmark 的标准答案不是固定毫秒值

实际值取决于：

```text
GPU
PyTorch Version
CUDA Version
DType
GPU Clock
Model Shape
```

所以没有固定的“128 Prompt 必须 X ms”。

真正有标准的是：

\[
\boxed{\text{趋势与解释}}
\]

---

# 49. 标准趋势一：Prefill 随 Prompt 增长

一般：

\[
Prefill_{2048}>Prefill_{512}>Prefill_{128}
\]

因为：

Projection / MLP 等大致有：

\[
O(S)
\]

Dense Attention 的 QK/AV 有：

\[
O(S^2)
\]

因此 Prompt 越长，Prefill 工作量越大。

---

# 50. 为什么实际延迟不是严格按照 \(S^2\) 增长？

因为完整 Transformer 还包含：

```text
Q/K/V/O Projection
MLP
LM Head
Norm
```

大量操作为线性复杂度，同时 GPU 对不同矩阵 Shape 的利用率也不同。

所以：

\[
\boxed{Theoretical\ Complexity\neq Observed\ Latency\ Ratio}
\]

---

# 51. 标准趋势二：Decode ms/token 随 Context 增长

Cached Decode：

\[
Q:[1,D]
\]

历史：

\[
K:[T,D]
\]

因此：

\[
QK^T:[1,T]
\]

单步 Attention 的工作和 KV Read 随 \(T\) 增长，因此通常：

\[
T\uparrow\Rightarrow DecodeTime/token\uparrow
\]

---

# 52. 为什么 Toy Model 可能不完全单调？

模型太小时，以下固定成本可能占比明显：

```text
Kernel Launch
CPU/Python Overhead
GPU Under-utilization
```

因此短序列下结果可能有噪声，不必强行要求严格单调。

---

# 53. 标准趋势三：KV Cache 优于 Naive

Naive：

```text
每一步重新处理完整 Context
```

Cached：

```text
历史 K/V 只计算一次
```

通常：

\[
CachedTotal<NaiveTotal
\]

而且随着 Prompt Length 或 Output Length 增长，Cache 复用收益更明显。

---

# 54. 为什么 Output Length 越长，Cache 越有价值？

若只生成 1 个 Token，基本只有 Prefill，Cache 很少被复用。

若生成 100 个 Token：

```text
Naive → 重复完整历史 Forward 100 次
Cached → 历史只算一次，后续仅增量 Decode
```

因此：

\[
GeneratedLength\uparrow\Rightarrow CacheReuseBenefit\uparrow
\]

---

# 55. 推荐结果表

| Prompt | Prefill ms | Decode ms/token | KV Total | Naive Total | Speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 实测 | 实测 | 实测 | 实测 | 实测 |
| 512 | 实测 | 实测 | 实测 | 实测 | 实测 |
| 2048 | 实测 | 实测 | 实测 | 实测 | 实测 |

---

# 56. 推荐结果分析标准表达

## Prefill

> 随 Prompt Length 从 128 增长到 2048，Prefill latency 整体上升。理论上，Projection、MLP 与 LM Head 主要随序列长度线性增长，而 Dense Attention 中 QKᵀ 和 Attention×V 包含 \(O(S^2D)\) 项，因此长 Prompt 会显著增加 Prefill 计算。实际延迟不会严格按照 \(S^2\) 比例增长，因为 GPU 利用率、Kernel Shape 和其它线性复杂度算子都会影响总运行时间。

## Decode

> 在 KV Cache 模式下，每一步仅对最新 Token 计算新的 Q/K/V，但当前 Query 仍然需要访问全部历史 K/V，因此单步 Decode 的 Attention 工作量和 KV Memory Traffic 随 Context Length 增长。随着 Prompt Length 增大，Decode latency/token 通常呈上升趋势。

## Naive vs Cached

> Naive Generation 在每个 Decode Step 都重新对完整历史 Context 执行 Forward，导致历史 K/V 被重复计算；KV-Cache Generation 则只在 Prefill 阶段计算一次历史 K/V，并在后续 Decode 中持续复用，因此显著减少重复计算。随着 Prompt Length 或 Generated Length 增大，两者性能差距通常进一步扩大。

---

# 57. 三个实验的完整逻辑

Experiment 1：

```text
Naive Generation
↓
Historical Computation Repeated
↓
KV Cache
```

Experiment 2：

```text
KV Cache 减少计算
↓
但增加 HBM 占用
```

公式：

\[
M_{KV}=2LBTH_{KV}d_hBytes
\]

Experiment 3：

```text
Prompt ↑ → Prefill ↑
Context ↑ → Decode KV Read ↑
Output ↑ → KV Cache Reuse Benefit ↑
```

最终：

\[
\boxed{Algorithm\rightarrow Memory\rightarrow Runtime}
\]

---

# 58. Week 01 与 Week 02 的连接

Week 01：

\[
[B,S,D]\rightarrow Q,K,V
\]

Week 02：

\[
K,V\rightarrow Cache
\]

Week 01：

\[
Attention\sim O(S^2)
\]

Week 02：

```text
Prefill → Dense Attention S²
```

Week 01：

\[
ArithmeticIntensity=\frac{FLOPs}{Bytes}
\]

Week 02：

```text
Decode
→ 1 Token
→ Weights + KV Read
→ Lower Arithmetic Intensity
→ Memory-Bandwidth Pressure
```

---

# 59. 常见工程错误

## 59.1 Cached Decode 仍传完整 Context

错误：

```python
model(generated, past_key_values=cache, use_cache=True)
```

正确：

```python
model(next_token, past_key_values=cache, use_cache=True)
```

---

## 59.2 Causal Mask 没考虑 `past_len`

普通 `[S,S]` Mask 不能直接用于：

```text
Q_LEN = 1
KV_LEN = T+1
```

必须基于绝对 Query Position。

---

## 59.3 所有 Layer 共用一份 Cache

错误。每一层 K/V 都不同，必须按 Layer 管理。

---

## 59.4 Calculator 忘记 K + V 的 `2`

必须：

\[
2\times
\]

---

## 59.5 把 KV Cache 说成 \(O(S^2)\)

错误。

Cache Capacity：

\[
O(S)
\]

Dense Prefill Attention Compute：

\[
O(S^2)
\]

---

## 59.6 没 Warmup 就记录性能

正确 Benchmark 至少需要：

```text
Warmup + Repeat + Median/Average
```

---

## 59.7 GPU 直接使用 `time.time()` 且不同步

CUDA 异步会产生错误时间，优先使用 CUDA Event。

---

## 59.8 只看输出 Token，不比较 Logits

Token 一致是基本测试，更严格应该增加：

```python
torch.allclose(...)
```

---

# 60. 必须会回答的 10 个验收问题

## Q1：为什么 Naive Generation 浪费？

每个新 Token 都重新对完整历史序列 Forward，历史 K/V 与其它计算被重复执行。

## Q2：为什么缓存 K/V，不缓存 Q？

历史 K/V 会被未来 Query 重复访问，历史 Q 完成当前 Step 后不再使用。

## Q3：为什么每层都有独立 KV Cache？

因为每层 Hidden State 和 \(W_K/W_V\) 都不同。

## Q4：KV Cache Shape？

\[
[B,H_{KV},T,d_h]
\]

每层 K/V 各一份。

## Q5：KV Cache Memory？

\[
\boxed{2LBTH_{KV}d_hBytes}
\]

## Q6：KV Cache 随 Sequence 是平方增长吗？

不是。Cache 容量是 \(O(T)\)，Dense Prefill Attention 才包含 \(O(T^2)\) 计算。

## Q7：为什么 Cached Decode 只输入 `[B,1]`？

历史 K/V 已经缓存，只需要对新 Token 计算新的 Q/K/V。

## Q8：Context 越长，Decode 为什么仍变慢？

新 Query 仍需访问所有历史 K/V，所以计算量和 Memory Traffic 随 \(T\) 增长。

## Q9：为什么 GQA 减少 KV Cache？

\[
M_{KV}\propto H_{KV}
\]

GQA 减少 KV Head 数量。

## Q10：为什么 KV Cache 是 Compute-Memory Trade-off？

不用 Cache：Memory 少但重复计算多；使用 Cache：显存占用增加但历史计算显著减少。

---

# 61. Week 02 实验验收 Checklist

## Experiment 1

- [ ] `naive_generate()` 可以运行
- [ ] Attention 支持 `past_key_value`
- [ ] Block 支持 KV Cache
- [ ] MiniTransformer 管理每层 Cache
- [ ] Prefill 可以建立 Cache
- [ ] Decode 只输入 `[B,1]`
- [ ] Naive / Cached Token 一致
- [ ] Naive / Cached Logits `allclose`
- [ ] 能解释 Cached Causal Mask

## Experiment 2

- [ ] 实现 `kv_cache_calculator.py`
- [ ] 输入 batch / seq / layers / kv_heads / head_dim / dtype
- [ ] 输出 Total KV Cache
- [ ] 输出 per-token / per-layer / per-request memory
- [ ] 128 / 512 / 2048 结果正确
- [ ] MHA / GQA / MQA 对比正确
- [ ] 理论值与真实 Tensor nbytes 对齐

## Experiment 3

- [ ] Prompt 128 Benchmark
- [ ] Prompt 512 Benchmark
- [ ] Prompt 2048 Benchmark
- [ ] Warmup
- [ ] Repeats
- [ ] CUDA Event / Synchronization
- [ ] Prefill latency
- [ ] Decode ms/token
- [ ] Naive total
- [ ] KV total
- [ ] Speedup
- [ ] 输出 CSV
- [ ] 完成趋势分析

---

# 62. Week 02 最终知识闭环

完整推理数据流：

```text
Prompt
 ↓
Prefill
 ↓
一次处理完整 Prompt
 ↓
建立每层 KV Cache
 ↓
生成 First Token
 ↓
Decode
 ↓
每次只输入一个新 Token
 ↓
读取历史 K/V
 ↓
追加新的 K/V
 ↓
生成 Next Token
```

资源关系：

```text
Long Prompt
↓
Prefill Compute ↑

Long Context
↓
KV Cache ↑
Decode KV Read ↑

High Concurrency
↓
KV Cache Memory ↑

Fewer KV Heads
↓
KV Cache ↓
```

最终：

\[
\boxed{\text{Model Architecture}\rightarrow\text{Tensor Shape}\rightarrow\text{Cache}\rightarrow\text{Memory}\rightarrow\text{Runtime}}
\]

---

# 63. 后续知识连接

本周 KV Cache 是后面这些系统技术的共同基础：

```text
Paged KV / PagedAttention
Continuous Batching
Prefix Cache
Chunked Prefill
vLLM Scheduler
SGLang Radix Cache
KV Cache Quantization
PD Disaggregation
KV Transfer
Cache Pool / Mooncake
```

这些技术都继续回答同一个问题：

> 当大量请求都维护不断增长的 KV Cache 时，如何更高效地管理 GPU Memory、调度计算和组织数据移动？

因此本周真正应该记住的不是 `past_key_values` API，而是：

\[
\boxed{KV\ Cache=LLM\ Serving\ 中最核心的状态资源之一}
\]
