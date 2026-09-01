"""
    此脚本主要用于计算 KV Cache 有关的内存、计算量等等内容
"""

import argparse

# 数据类型对应的每个元素的字节数
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5
}


# 单位转换
class BytesTrans:

    @staticmethod
    def bytes_to_kib(num_bytes):
        return num_bytes / 1024

    @staticmethod
    def bytes_to_gib(num_bytes):
        return num_bytes / (1024 ** 3)

    @staticmethod
    def bytes_to_mib(num_bytes):
        return num_bytes / (1024 ** 2)


# 核心计算函数
def kv_cache_bytes(batch, seq, layers, kv_heads, head_dim, dtype):
    """
    计算整个 KV Cache 占用的字节数。
    公式：2 * batch * seq * layers * kv_heads * head_dim * bytes_per_element
    """
    if dtype not in DTYPE_BYTES:
        raise ValueError(f"Unsupported dtype: {dtype}")
    bytes_per_element = DTYPE_BYTES[dtype]

    total_element = 2 * batch * seq * layers * kv_heads * head_dim
    total_bytes = total_element * bytes_per_element
    return total_bytes


def kv_cache_bytes_per_token(batch, layers, kv_heads, head_dim, dtype):
    """
    计算每个 token 新增的 KV Cache 字节数（即 seq=1 时的值）
    """
    return kv_cache_bytes(batch, 1, layers, kv_heads, head_dim, dtype)


# 打印函数
def report(batch, seq, layers, kv_heads, head_dim, dtype):
    total_bytes = kv_cache_bytes(batch, seq, layers, kv_heads, head_dim, dtype)
    per_token_bytes = kv_cache_bytes_per_token(batch, layers, kv_heads, head_dim, dtype)
    per_layer_bytes = total_bytes / layers
    per_request_bytes = total_bytes / batch

    print("===== KV Cache Calculator =====")
    print(f"Batch       : {batch}")
    print(f"Sequence    : {seq}")
    print(f"Layers      : {layers}")
    print(f"KV Heads    : {kv_heads}")
    print(f"Head Dim    : {head_dim}")
    print(f"DType       : {dtype}")
    print()
    print("KV Cache shape per layer:")
    print(f"K: [{batch}, {kv_heads}, {seq}, {head_dim}]")
    print(f"V: [{batch}, {kv_heads}, {seq}, {head_dim}]")
    print()
    print(f"Per token   : {BytesTrans.bytes_to_kib(per_token_bytes):.2f} KiB")
    print(f"Per layer   : {BytesTrans.bytes_to_mib(per_layer_bytes):.6f} MiB")
    print(f"Per request : {BytesTrans.bytes_to_mib(per_request_bytes):.6f} MiB")
    print(f"Total       : {BytesTrans.bytes_to_mib(total_bytes):.6f} MiB")
    print(f"Total       : {BytesTrans.bytes_to_gib(total_bytes):.4f} GiB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate LLM KV Cache memory.")
    parser.add_argument("--batch", type=int, required=True, help="Batch size")
    parser.add_argument("--seq", type=int, required=True, help="Sequence length")
    parser.add_argument("--layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--kv-heads", type=int, required=True, help="Number of KV heads")
    parser.add_argument("--head-dim", type=int, required=True, help="Dimension per head")
    parser.add_argument("--dtype", type=str, default="bf16",
                        choices=list(DTYPE_BYTES.keys()), help="Data type")

    args = parser.parse_args()
    report(batch=args.batch, seq=args.seq, layers=args.layers,
           kv_heads=args.kv_heads, head_dim=args.head_dim, dtype=args.dtype)

