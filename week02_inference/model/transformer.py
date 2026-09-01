import torch.nn as nn

from week02_inference.model.block import TransformerBlock
from week01_llm_systems.model.rmsnorm import RMSNorm


class MiniTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_heads=config.num_heads,
                ffn_hidden_size=config.ffn_hidden_size,
                eps=config.rms_norm_eps
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        """
        相较于week1，这一版transformer适配了KV Cache的输入 符合自回归式生成(AutoregressiveGeneration)
        :param input_ids: Query索引
        :param past_key_values: 过去的 所有层次的 kv 缓存
        :param use_cache: yes or not
        :return: new token
        """
        x = self.embedding(input_ids)

        if past_key_values is None:
            ### 说明当前是Prefill阶段
            past_key_values = [None] * len(self.layers)
        elif len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values length must equal num_layers.")

        present_key_values = [] if use_cache else None

        for layer, layer_past in zip(self.layers, past_key_values):
            if use_cache:
                x, layer_present = layer(x, past_key_value=layer_past, use_cache=True)
                present_key_values.append(layer_present)
            else:
                x = layer(x, past_key_value=None, use_cache=False)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        if use_cache:
            return logits, tuple(present_key_values)
        else:
            return logits, None
