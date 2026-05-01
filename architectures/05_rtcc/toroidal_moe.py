"""ToroidalMoE — the core RTCC innovation.

Forward pass:
  1. Reshape embedding to 2D grid [B, S, H, W]
  2. Toroidal circular padding by `overlap` cells on all sides
  3. Unfold into overlapping patches — each expert sees a patch_size x patch_size window
  4. Apply position-fixed expert MLPs (all active, no routing)
  5. Overlap-add fold back to grid (shared border dims receive gradient from multiple experts)
  6. Reshape back to flat embedding [B, S, D]

Expert MLPs are vectorized — stored as batched [n_experts, ...] parameters.
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

        # Batched SwiGLU: [n_experts, patch_dims, hidden]
        self.gate = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, expert_hidden))
        self.up   = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, expert_hidden))
        self.down = nn.Parameter(torch.empty(self.n_experts, expert_hidden, self.patch_dims))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gate, std=0.02)
        nn.init.normal_(self.up, std=0.02)
        nn.init.normal_(self.down, std=0.02)

    def _toroidal_pad(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """Circular pad [B, S, H, W] by p cells on each side of H and W."""
        # Row axis
        x = torch.cat([x[:, :, -p:, :], x, x[:, :, :p, :]], dim=2)
        # Col axis
        x = torch.cat([x[:, :, :, -p:], x, x[:, :, :, :p]], dim=3)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        p = self.overlap

        # 1. Reshape to grid
        x_grid = x.view(B, S, self.H, self.W)             # [B, S, H, W]

        # 2. Toroidal pad
        x_pad = self._toroidal_pad(x_grid, p)             # [B, S, H+2p, W+2p]

        # 3. Unfold patches — treat S as batch dim alongside B for unfold
        # Merge B and S for efficient 2D unfold
        Hp, Wp = self.H + 2 * p, self.W + 2 * p
        xp = x_pad.view(B * S, 1, Hp, Wp)                 # [BS, 1, Hp, Wp]
        # unfold: [BS, patch_dims, n_er, n_ec]
        patches = F.unfold(xp, kernel_size=self.patch_size, stride=self.stride)
        # patches: [BS, patch_dims, n_er*n_ec]
        patches = patches.permute(0, 2, 1)                # [BS, n_experts, patch_dims]

        # 4. Vectorized SwiGLU for all experts simultaneously
        # gate/up: [BS, n_experts, patch_dims] x [n_experts, patch_dims, hidden] -> [BS, n_experts, hidden]
        g = torch.einsum("bep,eph->beh", patches, self.gate)
        u = torch.einsum("bep,eph->beh", patches, self.up)
        h = F.silu(g) * u
        # down: [BS, n_experts, hidden] x [n_experts, hidden, patch_dims] -> [BS, n_experts, patch_dims]
        out_patches = torch.einsum("beh,ehp->bep", h, self.down)  # [BS, n_experts, patch_dims]

        # 5. Overlap-add fold back to grid (with padding)
        out_patches = out_patches.permute(0, 2, 1)       # [BS, patch_dims, n_experts]
        folded = F.fold(
            out_patches,
            output_size=(Hp, Wp),
            kernel_size=self.patch_size,
            stride=self.stride,
        )                                                  # [BS, 1, Hp, Wp]

        # 6. Remove circular padding — trim back to [H, W]
        out_grid = folded[:, 0, p:p + self.H, p:p + self.W]  # [BS, H, W]

        # 7. Reshape to flat embedding (contiguous needed after slicing fold output)
        return out_grid.reshape(B, S, D)
