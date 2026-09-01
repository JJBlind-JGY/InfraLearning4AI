import torch

from week01_llm_systems.model.config import ModelConfig
from week02_inference.model.transformer import MiniTransformer


@torch.inference_mode()
def naive_generate(model, input_ids, max_new_tokens):
    """
    完成整体的token循环推理的流程，基本上模型推理走的KV Cache的流程就是如此所示

    :param model: 模型参数
    :param input_ids: 输入的token索引
    :param max_new_tokens: 最大生成的token数量
    :return: 最后的文本
    """
    generated = input_ids.clone()  # [B, V]

    for _ in range(max_new_tokens):
        # 每一次都把前一个历史序列重新送入模型
        logits, _ = model(generated)
        # 取最后一个位置，对下一个token进行预测
        next_token_logits = logits[:, -1, :]

        # greedy decoding
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # 将新token添加到历史序列
        generated = torch.cat([generated, next_token], dim=-1)

    return generated


@torch.inference_mode()
def kv_cache_generate(model, input_ids, max_new_tokens, temperature=1.0, top_k=None):
    """
    整个 KV Cache 过程
    :param model: Transformer
    :param input_ids: 输入的token序列
    :param max_new_tokens: 最大生成token数
    :param temperature: 活动参数
    :param top_k: 获取top_k数量的tokens
    :return: 生成的数据
    """
    model.eval()  # 避免Drop_out, BN的影响
    generated = input_ids.clone()
    batch_size, seq_len = input_ids.shape

    # 上下文参数的指定
    max_context_len = getattr(model.config, "max_position_embeddings", None)
    if max_context_len is not None and seq_len + max_new_tokens > max_context_len:
        raise ValueError("Requested generation length exceeds model's max context length")

    ### Prefill
    logits, past_key_values = model(input_ids, past_key_values=None, use_cache=True)

    ## Decode
    for step in range(max_new_tokens):
        next_token_logits = logits[:, -1, :]  # [B, vocab]

        if temperature != 1.0:
            """
            操作：将所有的 logits 除以一个超参数 temperature（温度）。
            数学效果：
                温度 = 1：不变。
                温度 < 1（如 0.8）：logits 的绝对值被放大（差距拉大）。经过后续 Softmax 后，概率分布更尖锐（高概率的更高，低概率的更低），模型更倾向于选最高分，随机性降低。            
                温度 > 1（如 1.2）：logits 的绝对值被缩小（差距缩小）。经过 Softmax 后，概率分布更平滑，低概率的词也有更多机会被选中，随机性增加。
        
                极端情况：温度趋近于 0 时，等价于 Argmax（贪心）；温度趋近于无穷大时，趋近于均匀分布（完全随机乱说）
            """
            next_token_logits = next_token_logits / temperature

        if top_k is not None:
            # 仅保留 top_k 个概率最高的 logits
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float("-inf")

        probs = torch.softmax(next_token_logits, dim=-1)
        """
        torch.multinomial 根据 probs 中给定的概率权重，进行随机抽取。
        num_samples=1 表示抽取 1 个样本。
     
        与 Argmax 的本质区别：
            Argmax：永远选概率最大的那个（比如概率 90% 的词 A，10% 的词 B，永远选 A）。        
            Multinomial：按概率随机抽。词 A 有 90% 的概率被抽中，词 B 有 10% 的概率被抽中。这使得模型偶尔会“冒险”选一些次优词，极大地增加了文本的多样性。
        """
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        generated = torch.cat([generated, next_token], dim=1)

        if step == max_new_tokens - 1:
            break

        # Decode 仅输入最新的 token
        logits, past_key_values = model(next_token, past_key_values=past_key_values, use_cache=True)

    return generated


# 做一些基础测试
def test_generation_equivalence(model, input_ids):
    # 1. 调用“朴素生成”（即每次重新计算全部序列，不用缓存）
    naive_output = naive_generate(model, input_ids=input_ids, max_new_tokens=8)
    # 2. 调用“缓存生成”（即你刚写好的 kv_cache_generate）
    cached_output = kv_cache_generate(model, input_ids, 8)

    print("Naive:", naive_output)
    print("Cached:", cached_output)

    same = torch.equal(naive_output, cached_output)  # 判断两个张量在元素级别上是否完全相等（包括 dtype）
    print("Same tokens:", same)
    return same


@torch.inference_mode()
def test_cache_logits(model, prompt):
    logits, cache = model(prompt, use_cache=True)

    first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    # 把 prompt 和新生成的 token 拼成完整序列 [prompt + token]
    full_sequence = torch.cat([prompt, first_token], dim=1)
    # 重新传入模型（注意这里没有传 cache，use_cache 默认为 False）
    naive_logits, _ = model(full_sequence)
    # 取新序列最后一个位置的 logits（即模型对“下一个”token 的预测）
    naive_new_logits = naive_logits[:, -1, :]

    # 只传入最新生成的 1 个 token，并传入之前存好的 cache
    cached_logits, _ = model(first_token, past_key_values=cache, use_cache=True)
    # 取当前输出的 logits（因为输入只有 1 个 token，最后一个位置就是它）
    cached_new_logits = cached_logits[:, -1, :]

    max_diff = (naive_new_logits - cached_new_logits).abs().max()
    print("Max logits difference:", max_diff.item())
    print("All close:", torch.allclose(naive_new_logits, cached_new_logits, atol=1e-5, rtol=1e-4))


def cache_nbytes(past_key_values):
    """
    计算 past_key_values 元组中所有的 K/V 张量占用的字节数
    :param past_key_values: 一个元组，每层为 (k,v) 或者 None
    """
    total = 0
    if past_key_values is None:
        return 0
    for layer_cache in past_key_values:
        if layer_cache is not None:
            k, v = layer_cache
            total += k.numel() * k.element_size()
            total += v.numel() * v.element_size()
    return total


@torch.inference_mode()
def test_cache_memory(model, prompt):
    logits, cache = model(prompt, use_cache=True)
    actual_bytes = cache_nbytes(cache)
    print(f"Actual KV Cache size: {actual_bytes / 1024 ** 2:.2f} MiB")

    # 计算理论值
    batch = prompt.shape[0]
    seq = prompt.shape[1]
    layers = 4
    kv_heads = 8
    head_dim = 32
    dtype = "fp32"

    from week02_inference.kv_cache_calculator import kv_cache_bytes
    theoretical_bytes = kv_cache_bytes(batch, seq, layers, kv_heads, head_dim, dtype)
    print(f"Theoretical size: {theoretical_bytes / 1024**2:.2f} MiB")
    print(f"Difference: {abs(actual_bytes - theoretical_bytes) / 1024**2:.6f} MiB")


if __name__ == '__main__':
    config = ModelConfig()
    transformer = MiniTransformer(config)

    input_ids = torch.randint(low=0, high=config.vocab_size, size=(2, 128))

    test_generation_equivalence(transformer, input_ids)
    test_cache_logits(transformer, input_ids)
    test_cache_memory(transformer, input_ids)
    