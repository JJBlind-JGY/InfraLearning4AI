import argparse, csv, statistics, time
import torch

from week01_llm_systems.model.config import ModelConfig
from week02_inference.model.transformer import MiniTransformer


# 时刻记录
def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def timed_call(fn, device):
    if device.type == "cuda":
        # 使用 CUDA Event 记录 GPU 时间（以毫秒为单位），比 time.time() 更精确。
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        synchronize(device)
        start.record()
        result = fn()
        end.record()
        synchronize(device)
        elapsed_ms = start.elapsed_time(end)
        return result, elapsed_ms
    else:
        # CPU 回退使用 perf_counter
        start_time = time.perf_counter()
        result = fn()
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return result, elapsed_ms


@torch.inference_mode()
def benchmark_naive_once(model, prompt, max_new_tokens, device):
    """
    每次迭代都重新计算整个序列（包括历史 token），复杂度 O(N²)。
    返回总时间（包含所有 decode 步骤）。
    """

    def generation():
        generated = prompt.clone()
        for _ in range(max_new_tokens):
            logits, _ = model(generated)  # 每一次都是传输完整的序列
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    _, total_ms = timed_call(generation, device)
    return total_ms


# benchmark_cached版本测试时间消耗
@torch.inference_mode()
def benchmark_cached_once(model, prompt, max_new_tokens, device):
    """
    这是重点，分为 Prefill 和 Decode 两个阶段
    """

    """
    一次性传入 prompt，use_cache=True 让模型返回所有层的 K/V 缓存。
    直接从 logits 的最后一个位置取 argmax，得到第一个生成 token。
    prefill_ms 即为 Prefill 延迟。
    """

    def prefill():
        logits, cache = model(prompt, use_cache=True)
        first_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        return first_token, cache

    (first_token, cache), prefill_ms = timed_call(prefill, device)

    if max_new_tokens <= 1:
        return {
            "prefill_ms": prefill_ms,
            "decode_total_ms": 0.0,
            "decode_ms_per_token": 0.0,
            "total_ms": prefill_ms
        }

    def decode():
        current_token = first_token
        past_key_values = cache

        for _ in range(max_new_tokens - 1):
            logits, past_key_values = model(current_token, past_key_values=past_key_values,
                                            use_cache=True)
            current_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

        return current_token

    _, decode_total_ms = timed_call(decode, device)
    decode_steps = max_new_tokens - 1
    decode_ms_per_token = decode_total_ms / decode_steps

    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "total_ms": prefill_ms + decode_total_ms
    }


# 测试脚本
def benchmark_prompt_length(model, config, prompt_len, max_new_tokens, repeats,
                            warmups, device):
    prompt = torch.randint(low=0, high=config.vocab_size, size=(1, prompt_len), device=device)

    # 基础的warmup，不计时
    for _ in range(warmups):
        benchmark_cached_once(model, prompt, 3, device)
        benchmark_naive_once(model, prompt, 2, device)

    # 正式测试计时
    cached_results = []
    naive_results = []
    for _ in range(repeats):
        cached = benchmark_cached_once(model, prompt, max_new_tokens, device)
        naive_ms = benchmark_naive_once(model, prompt, max_new_tokens, device)
        cached_results.append(cached)
        naive_results.append(naive_ms)

    # 使用中位数避免噪音数据影响
    prefill_ms = statistics.median([r["prefill_ms"] for r in cached_results])
    decode_ms_per_token = statistics.median([r["decode_ms_per_token"] for r in cached_results])
    cached_total_ms = statistics.median([r["total_ms"] for r in cached_results])
    naive_total_ms = statistics.median(naive_results)

    speedup = naive_total_ms / cached_total_ms

    return {
        "prompt_len": prompt_len,
        "prefill_ms": prefill_ms,
        "decode_ms_per_token": decode_ms_per_token,
        "cached_total_ms": cached_total_ms,
        "naive_total_ms": naive_total_ms,
        "speedup": speedup
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--new-tokens", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--csv", type=str, default="benchmark_results.csv")

    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    config = ModelConfig()
    transformer = MiniTransformer(config).to(device)

    transformer.eval()
    results = []
    print("\n===== Generation Benchmark =====")
    print(
        f"{'Prompt':>8}"
        f"{'Prefill(ms)':>15}"
        f"{'Decode/token':>16}"
        f"{'KV Total':>14}"
        f"{'Naive Total':>16}"
        f"{'Speedup':>12}"
    )

    for prompt_len in args.prompt_lengths:
        result = benchmark_prompt_length(model=transformer, config=config, prompt_len=prompt_len,
                                         max_new_tokens=args.new_tokens,
                                         repeats=args.repeats, warmups=args.warmups,
                                         device=device)
        results.append(result)
        print(
            f"{result['prompt_len']:8d}"
            f"{result['prefill_ms']:15.3f}"
            f"{result['decode_ms_per_token']:16.3f}"
            f"{result['cached_total_ms']:14.3f}"
            f"{result['naive_total_ms']:16.3f}"
            f"{result['speedup']:11.2f}x"
        )

    with open(args.csv, "w", newline="", encoding="utf-8", ) as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_len", "prefill_ms", "decode_ms_per_token", "cached_total_ms",
                                               "naive_total_ms", "speedup"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Results saved to. {args.csv}")
