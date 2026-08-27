from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 10000

    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8

    ffn_hidden_size: int = 1024

    max_seq_len: int = 128

    rms_norm_eps: float = 1e-6
