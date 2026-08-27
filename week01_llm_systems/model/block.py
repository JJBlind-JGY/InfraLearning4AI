import torch.nn as nn

from week01_llm_systems.model.mlp import MLP
from week01_llm_systems.model.rmsnorm import RMSNorm
from week01_llm_systems.model.attention import CasualSelfAttention


class TransformerBlock(nn.Module):

    def __init__(self, hidden_size, num_heads, ffn_hidden_size, eps=1e-6):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps)
        self.norm2 = RMSNorm(hidden_size, eps)
        self.attn = CasualSelfAttention(hidden_size, num_heads)
        self.mlp = MLP(hidden_size, ffn_hidden_size)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
