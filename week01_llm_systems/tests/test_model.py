import torch

from week01_llm_systems.model.rmsnorm import RMSNorm
from week01_llm_systems.model.mlp import MLP


def test_rmsnorm_shape():

    B=2
    S=128
    D=256

    x = torch.randn(B, S, D)

    norm = RMSNorm(D)

    y = norm(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


if __name__ == '__main__':
    test_rmsnorm_shape()
    print("Test Passed")

    mlp = MLP(256, 1024)
    params = sum(p.numel() for p in mlp.parameters())
    print(params)
