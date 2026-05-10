import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.normalized_shape = (dim,)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast weight to input dtype so F.rms_norm can dispatch to fused kernel.
        # Under bfloat16 autocast, x is bf16 but self.weight stays fp32; mismatch
        # triggers a slow-path fallback otherwise. The cast is on a small (D,) tensor.
        return F.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), self.eps)
