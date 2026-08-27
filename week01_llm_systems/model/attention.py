import math

import torch
import torch.nn as nn


class CasualSelfAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        B, S, D = x.shape

        q = self.q_proj(x)      # [B, S, D]
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, S, self.num_heads, self.head_dim)     # [B, S, H, D_h]
        k = k.view(B, S, self.num_heads, self.head_dim)
        v = v.view(B, S, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)       # [B, H, S, D_h]

        # attention calculate
        scores = q @ k.transpose(-2, -1)        # k -> [B, H, D_h, S]  scores -> [B, H, S, S]
        scores = scores / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)    # [B, H, S, S]

        out = attn @ v      # [B, H, S, D_h]
        out = out.transpose(1, 2)   # [B, S, H, D_h]
        out = out.contiguous().view(B, S, D)
        return self.o_proj(out)
