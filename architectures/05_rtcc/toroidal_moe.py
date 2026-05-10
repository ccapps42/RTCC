"""ToroidalMoE — the core RTCC innovation.

Forward pass:
  1. Reshape embedding to 2D grid [BS, 1, H, W] and circular-pad
  2. Unfold into overlapping patches — each expert sees a patch_size x patch_size window
  3. Apply position-fixed expert MLPs (all active, no routing)
  4. Overlap-add fold back to grid
  5. Reshape back to flat embedding [B, S, D]

Expert MLPs are vectorized — stored as batched [n_experts, ...] parameters.
gate and up projections are stored fused as [n_experts, patch_dims, 2*hidden]
to halve GEMM launches for the first projection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ToroidalMoE(nn.Module):
    def __init__(self, grid_rows: int, grid_cols: int, patch_size: int, stride: int,
                 expert_hidden: int):
        super().__init__()
        self.H = grid_rows
        self.W = grid_cols
        self.patch_size = patch_size
        self.stride = stride
        self.overlap = patch_size - stride

        self.n_er = grid_rows // stride
        self.n_ec = grid_cols // stride
        self.n_experts = self.n_er * self.n_ec
        self.patch_dims = patch_size * patch_size

        # Fused gate+up: [n_experts, patch_dims, 2*hidden] — halves first-projection GEMM launches
        self.gate_up = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, 2 * expert_hidden))
        self.down    = nn.Parameter(torch.empty(self.n_experts, expert_hidden, self.patch_dims))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate_up, std=0.02)
        nn.init.normal_(self.down,    std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        p = self.overlap

        # 1. Reshape to grid and circular-pad in one shot
        x_grid = x.view(B * S, 1, self.H, self.W)
        x_pad  = F.pad(x_grid, (p, p, p, p), mode='circular')  # [BS, 1, H+2p, W+2p]

        # 2. Unfold patches
        patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.stride)
        # patches: [BS, patch_dims, n_experts]
        patches = patches.permute(0, 2, 1).contiguous()         # [BS, n_experts, patch_dims]

        # 3. Fused gate+up projection then SwiGLU
        gu = torch.einsum("bep,eph->beh", patches, self.gate_up)  # [BS, n_experts, 2*hidden]
        g, u = gu.chunk(2, dim=-1)                                 # each [BS, n_experts, hidden]
        h = F.silu(g) * u

        # 4. Down projection
        out_patches = torch.einsum("beh,ehp->bep", h, self.down)   # [BS, n_experts, patch_dims]

        # 5. Overlap-add fold
        Hp, Wp = self.H + 2 * p, self.W + 2 * p
        folded = F.fold(
            out_patches.permute(0, 2, 1),                          # [BS, patch_dims, n_experts]
            output_size=(Hp, Wp),
            kernel_size=self.patch_size,
            stride=self.stride,
        )                                                            # [BS, 1, Hp, Wp]

        # 6. Trim padding and reshape
        return folded[:, 0, p:p + self.H, p:p + self.W].contiguous().view(B, S, D)
