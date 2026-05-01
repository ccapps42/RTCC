"""FlatGridMoE — overlapping 2D patch experts with ZERO-PADDING boundary.

Identical to ToroidalMoE except _toroidal_pad is replaced by zero-padding.
This isolates the boundary condition contribution: flat vs toroidal.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlatGridMoE(nn.Module):
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

        self.gate = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, expert_hidden))
        self.up   = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, expert_hidden))
        self.down = nn.Parameter(torch.empty(self.n_experts, expert_hidden, self.patch_dims))
        self._init_weights()

    def _init_weights(self):
        for p in [self.gate, self.up, self.down]:
            nn.init.normal_(p, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        p = self.overlap
        Hp = self.H + 2 * p
        Wp = self.W + 2 * p

        # Reshape to grid and zero-pad (NOT circular)
        x_grid = x.view(B, S, self.H, self.W)
        # F.pad: (left, right, top, bottom) for last two dims
        x_pad = F.pad(x_grid, (p, p, p, p), mode='constant', value=0.0)

        # Unfold
        xp = x_pad.view(B * S, 1, Hp, Wp)
        patches = F.unfold(xp, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.permute(0, 2, 1)     # [BS, n_experts, patch_dims]

        # Vectorized SwiGLU
        g = torch.einsum("bep,eph->beh", patches, self.gate)
        u = torch.einsum("bep,eph->beh", patches, self.up)
        h = F.silu(g) * u
        out_patches = torch.einsum("beh,ehp->bep", h, self.down)

        # Overlap-add fold
        out_patches = out_patches.permute(0, 2, 1)
        folded = F.fold(out_patches, output_size=(Hp, Wp),
                        kernel_size=self.patch_size, stride=self.stride)
        out_grid = folded[:, 0, p:p + self.H, p:p + self.W]

        return out_grid.reshape(B, S, D)
