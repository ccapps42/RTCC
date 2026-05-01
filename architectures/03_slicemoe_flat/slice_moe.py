"""SliceMoE — position-fixed, non-overlapping dimensional partitioning.

Each expert owns a contiguous slice of the embedding dimension.
No routing, no overlap. Vectorized over experts with einsum.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SliceMoE(nn.Module):
    """Position-fixed experts on non-overlapping slices of the embedding.

    Expert i processes dims [i*slice_dim : (i+1)*slice_dim].
    Batched SwiGLU weights — no Python loop over experts.
    """

    def __init__(self, model_dim: int, slice_dim: int, expert_hidden: int):
        super().__init__()
        assert model_dim % slice_dim == 0
        self.n_experts = model_dim // slice_dim
        self.slice_dim = slice_dim

        # Batched weights: [n_experts, slice_dim, expert_hidden]
        self.gate = nn.Parameter(torch.empty(self.n_experts, slice_dim, expert_hidden))
        self.up   = nn.Parameter(torch.empty(self.n_experts, slice_dim, expert_hidden))
        self.down = nn.Parameter(torch.empty(self.n_experts, expert_hidden, slice_dim))
        self._init_weights()

    def _init_weights(self):
        for p in [self.gate, self.up, self.down]:
            nn.init.normal_(p, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        E, Sd = self.n_experts, self.slice_dim

        # Partition x into slices: [B, S, E, slice_dim]
        slices = x.view(B, S, E, Sd)

        # Vectorized SwiGLU over all expert slices simultaneously
        # [B, S, E, Sd] x [E, Sd, H] -> [B, S, E, H]
        g = torch.einsum("bsei,eih->bseh", slices, self.gate)
        u = torch.einsum("bsei,eih->bseh", slices, self.up)
        h = F.silu(g) * u
        # [B, S, E, H] x [E, H, Sd] -> [B, S, E, Sd]
        out_slices = torch.einsum("bseh,ehi->bsei", h, self.down)

        # Reassemble: [B, S, D] — contiguous needed after einsum
        return out_slices.reshape(B, S, D)
