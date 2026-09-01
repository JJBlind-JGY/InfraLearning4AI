import math

import torch
import torch.nn as nn


class CasualAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.hidden_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, past_key_value=None, use_cache=False):
        """
        这是week2的attention与week1的attention的不同之处，主要是涉及到一个 KV Cache 的引入，降低推理难度

        :param x:   [B, Q_len, D]
        :param past_key_value:  (past_k: [B, H, PAST_LEN, Dh], past_v: [B, H, PAST_LEN, Dh])
        :param use_cache:   yes or no
        :return: activations
        """
        B, Q_LEN, D = x.shape

        # Q K V projection
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, Q_LEN, D] --> [B, H, Q_LEN, Dh]
        q = q.view(B, Q_LEN, self.num_heads, self.hidden_dim)
        k = k.view(B, Q_LEN, self.num_heads, self.hidden_dim)
        v = v.view(B, Q_LEN, self.num_heads, self.hidden_dim)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # Append historical KV Cache
        if past_key_value is not None:
            past_k, past_v = past_key_value

            # historical: [B, H, PAST_LEN, Dh]  current: [B, H, Q_LEN, Dh]
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # q: [B, H, Q_LEN, Dh], k/v: [B, H, KV_LEN, Dh]
        KV_LEN = k.size(2)
        past_len = KV_LEN - Q_LEN

        # continue progress
        scores = q @ k.transpose(-2, -1)
        scores /= math.sqrt(self.hidden_dim)    # [B, H, Q_LEN, KV_LEN]

        # MASK - Modified
        # 当前 Query 的绝对位置
        # prefill: past_len = 0     q_position = [0,1,2,...]
        # Decode: past_len = 128    q_position = [128]
        q_positions = torch.arange(past_len, past_len + Q_LEN, device=x.device)
        k_positions = torch.arange(KV_LEN, device=x.device)
        casual_mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        scores = scores.masked_fill(casual_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # softmax & activations
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v      # [B, H, Q_LEN, Dh]
        out = out.transpose(1, 2).contiguous().view(B, Q_LEN, D)

        # Return KV Cache
        if use_cache:
            present_key_value = (k, v)
            return out, present_key_value
        return out

