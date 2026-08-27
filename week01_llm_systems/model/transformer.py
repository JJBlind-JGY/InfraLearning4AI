import torch.nn as nn

from week01_llm_systems.model.rmsnorm import RMSNorm
from week01_llm_systems.model.block import TransformerBlock


class MiniTransformer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_heads,
                ffn_hidden_size=self.config.ffn_hidden_size,
                eps=self.config.rms_norm_eps
            )
            for _ in range(self.config.num_layers)
        ])

        self.final_norm = RMSNorm(hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits
