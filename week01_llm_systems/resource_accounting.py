from week01_llm_systems.model.config import ModelConfig
from week01_llm_systems.model.transformer import MiniTransformer

# 模型参数量计算
def count_parameters_manual(config):
    V = config.vocab_size
    D = config.hidden_size
    L = config.num_layers
    F = config.ffn_hidden_size

    embedding = V * D
    attention_per_layer = 4 * D * D
    mlp_per_layer = 2 * D * F
    norm_per_layer = 2 * D

    blocks = L * (attention_per_layer + mlp_per_layer + norm_per_layer)
    final_norm = D
    lm_head = D * V

    total = (
            embedding
            + blocks
            + final_norm
            + lm_head
    )

    return {
        "embedding": embedding,
        "attention": L * attention_per_layer,
        "mlp": L * mlp_per_layer,
        "block_norm": L * norm_per_layer,
        "final_norm": final_norm,
        "lm_head": lm_head,
        "total": total,
    }


def count_parameters_pytorch(model):
    return sum(p.numel() for p in model.parameters())



## 内存的核算
DTYPE_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
    "int4": 0.5
}

def bytes_to_gib(num_bytes):
    return num_bytes / (1024**3)

def weight_memory(num_params, dtype='bf16'):
    bytes_per_param = DTYPE_BYTES[dtype]
    total_bytes = num_params * bytes_per_param
    return total_bytes

def training_model_state_memory(num_params, param_bytes=2, grad_bytes=2, adam_m_bytes=4,
                                adam_v_bytes=4, master_weight_bytes=0):
    return num_params * (param_bytes + grad_bytes + adam_m_bytes + adam_v_bytes + master_weight_bytes)

def linear_flops(batch, seq, in_dim, out_dim):
    return 2 * batch * seq * in_dim * out_dim

def attention_flops(config, batch, seq):
    D = config.hidden_size
    # Q, K, V, O 四个 D -> D Linear
    projection_flops = 4 * linear_flops(batch=batch, seq=seq, in_dim=D, out_dim=D)
    # QK^T
    qk_flops = 2 * batch * seq * seq * D
    # Attention Probability × V
    av_flops = 2 * batch * seq * seq * D
    total = projection_flops + qk_flops + av_flops
    return {
        "projection": projection_flops,
        "qk": qk_flops,
        "av": av_flops,
        "total": total,
    }

def mlp_flops(config, batch, seq):
    D = config.hidden_size
    F = config.ffn_hidden_size
    fc1_flops = linear_flops(batch=batch, seq=seq, in_dim=D, out_dim=F)
    fc2_flops = linear_flops(batch=batch, seq=seq, in_dim=F, out_dim=D)
    total = fc1_flops + fc2_flops
    return {
        "fc1": fc1_flops,
        "fc2": fc2_flops,
        "total": total,
    }


def transformer_block_flops(config, batch, seq):
    attention = attention_flops(config, batch, seq)
    mlp = mlp_flops(config, batch, seq)
    total = attention["total"] + mlp["total"]
    return {
        "attention": attention,
        "mlp": mlp,
        "total": total,
    }

def model_forward_flops(config, batch, seq,):
    embed_head = linear_flops(batch, seq, config.vocab_size, config.hidden_size)
    block = transformer_block_flops(config, batch, seq)
    all_blocks = config.num_layers * block["total"]
    lm_head = linear_flops(
        batch=batch,
        seq=seq,
        in_dim=config.hidden_size,
        out_dim=config.vocab_size,
    )
    total = all_blocks + lm_head + embed_head
    return {
        "attention_per_layer":
            block["attention"]["total"],
        "mlp_per_layer":
            block["mlp"]["total"],
        "block_per_layer":
            block["total"],
        "all_blocks":
            all_blocks,
        "lm_head":
            lm_head,
        "total":
            total,
    }


if __name__ == '__main__':
    config = ModelConfig()
    model = MiniTransformer(config)
    manual = count_parameters_manual(config)
    actual = count_parameters_pytorch(model)

    print("===== Parameter Accounting =====")

    for name, value in manual.items():
        print(f"{name:15s}: {value:,}")
    print(f"\nManual total : {manual['total']:,}")
    print(f"PyTorch total: {actual:,}")
    print(f"Difference   : {manual['total'] - actual:,}")
    for name, param in model.named_parameters():
        print(
            f"{name:50s} "
            f"shape={str(tuple(param.shape)):20s} "
            f"params={param.numel():,}"
        )

