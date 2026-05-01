"""Loop Index Embedding — ported from CART (Model_Paper_1).

Sinusoidal encoding of the current loop iteration r (0-based),
projected to model_dim and added to h_input before each block pass.
Gives the shared-weight block explicit awareness of loop depth.

lie_dim=32 fixed (not swept). Parameter cost: 32 × model_dim ≈ 25K at d=768.
"""
import math
import torch
import torch.nn as nn


class LoopIndexEmbedding(nn.Module):
    def __init__(self, model_dim: int, lie_dim: int = 32, max_loops: int = 16):
        super().__init__()
        self.proj = nn.Linear(lie_dim, model_dim, bias=False)
        pe = self._build_sinusoidal(max_loops, lie_dim)
        self.register_buffer("pe", pe)  # [max_loops, lie_dim]

    def _build_sinusoidal(self, max_loops: int, lie_dim: int) -> torch.Tensor:
        pe = torch.zeros(max_loops, lie_dim)
        pos = torch.arange(max_loops).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, lie_dim, 2).float() * -(math.log(10000.0) / lie_dim)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

    def forward(self, h: torch.Tensor, r: int) -> torch.Tensor:
        signal = self.proj(self.pe[r])  # [model_dim]
        return h + signal               # broadcast: [B, T, model_dim] + [model_dim]
