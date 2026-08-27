import os
import torch
from torch.profiler import profile, ProfilerActivity, record_function

from week01_llm_systems.model.config import ModelConfig
from week01_llm_systems.model.transformer import MiniTransformer


def print_environment():
    print("===== Environment =====")
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        raise RuntimeError("Experiment 3 requires a CUDA GPU.")
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"GPU             : "f"{torch.cuda.get_device_name(0)}")


def build_model_and_input(batch_size=2, seq_len=128,):
    config = ModelConfig()
    device = torch.device("cuda")
    model = MiniTransformer(config).to(device)
    model.eval()    ## 移除 DropOut、BatchNorm的影响
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(batch_size, seq_len), device=device)
    return config, model, input_ids


# Warmup避免 CUDA Context初始化、 Kernel加载、 Allocator分配、 Cache冷启动、 Library初始化 的影响
def warmup(model, input_ids, num_warmup=10):
    with torch.inference_mode():        ## 明确关闭梯度计算，和 no_grad() 效果相同，但是明确纯推理的过程
        for _ in range(num_warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()


# 基准测试计算
def benchmark_forward(model, input_ids, warmup_iters=10, measure_iters=50):
    warmup(model, input_ids, warmup_iters)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()        # 避免异步cuda的影响  需要做一个cuda同步
    start_event.record()
    with torch.inference_mode():
        for _ in range(measure_iters):
            _ = model(input_ids)
    end_event.record()
    torch.cuda.synchronize()        # 避免异步cuda的影响  需要做一个cuda同步

    total_ms = start_event.elapsed_time(end_event)
    avg_ms = total_ms / measure_iters
    return avg_ms


# 性能测试
def run_profiler(model, input_ids, output_dir="logs"):
    os.makedirs(output_dir, exist_ok=True)
    warmup(model, input_ids, num_warmup=10)

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

    torch.cuda.synchronize()

    print("\n===== PyTorch Profiler =====")
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=30))

    trace_path = os.path.join(output_dir, "forward_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nTrace saved to: {trace_path}")
    return prof


# 实际推理开销
def bytes_to_mib(num_bytes):
    return num_bytes / (1024 ** 2)


def measure_inference_memory(model, input_ids,):
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()  # reset_peak_memory_stats() 用来重置 peak memory tracking 起点
    with torch.inference_mode():
        _ = model(input_ids)
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return {
        "allocated": peak_allocated,
        "reserved": peak_reserved,
    }


# 实际训练开销
def measure_training_memory(model, input_ids):
    model.train()
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    logits = model(input_ids)
    loss = logits.float().mean()
    loss.backward()
    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.zero_grad(set_to_none=True)
    return {
        "allocated": peak_allocated,
        "reserved": peak_reserved,
    }




if __name__ == "__main__":
    print_environment()
    config, model, input_ids = build_model_and_input(batch_size=2, seq_len=128)
    avg_ms = benchmark_forward(model, input_ids,)
    print("\n===== Baseline Timing =====")
    print(f"Average forward latency: {avg_ms:.4f} ms")

    prof = run_profiler(model, input_ids)

    inference_memory = measure_inference_memory(model, input_ids)
    print("\n===== Inference Memory =====")
    print(f"Peak allocated: {bytes_to_mib(inference_memory['allocated']):.2f} MiB")
    print(f"Peak reserved : {bytes_to_mib(inference_memory['reserved']):.2f} MiB")

    training_memory = measure_training_memory(model, input_ids,)
    print("\n===== Training Memory =====")
    print(f"Peak allocated: {bytes_to_mib(training_memory['allocated']):.2f} MiB")
    print(f"Peak reserved : {bytes_to_mib(training_memory['reserved']):.2f} MiB")
