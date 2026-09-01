import torch.nn as nn

from week01_llm_systems.model.rmsnorm import RMSNorm
from week01_llm_systems.model.mlp import MLP
from week02_inference.model.attention import CasualAttention


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, ffn_hidden_size, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps)
        self.attn = CasualAttention(hidden_size, num_heads)
        self.norm2 = RMSNorm(hidden_size, eps)
        self.mlp = MLP(hidden_size, ffn_hidden_size)

    def forward(self, x, past_key_value=None, use_cache=False):
        """
        同样，和Week1的区别就在于引入了 KV Cache来帮助完成推理模式

        :param x: inputs_matrix
        :param past_key_value: 过去的 kv cache
        :param use_cache: yes or not
        :return: activations
        """
        normed_x = self.norm1(x)
        if use_cache:
            attn_out, present_key_value = self.attn(normed_x, past_key_value=past_key_value, use_cache=use_cache)
        else:
            attn_out = self.attn(normed_x, past_key_value=None, use_cache=False)
            present_key_value = None

        x += attn_out
        normed_x = self.norm2(x)
        mlp_out = self.mlp(normed_x)
        x += mlp_out

        if use_cache:
            return x, present_key_value
        return x
