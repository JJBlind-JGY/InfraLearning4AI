import torch

from week01_llm_systems.model.config import ModelConfig
from week01_llm_systems.model.transformer import MiniTransformer

if __name__ == '__main__':
    config = ModelConfig()
    model = MiniTransformer(config)

    # 测试数据
    B = 2
    S = 128
    input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, S))

    logits = model(input_ids)
    print("input shape: ", input_ids.shape)
    print("output shape: ", logits.shape)
    print(f"input: {input_ids}")
    print(f"output: {logits}")

    params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {params}")

    loss = logits.mean()
    loss.backward()
    print(model.embedding.weight.grad is not None)
