"""TopKToroidMoE — the RTCC Paper 2 contribution.

Treats the up-projected residual stream [B, S, grid_side**2] as a
grid_side x grid_side toroidal sheet. A linear router on the flat input
picks top-K of (grid_side/stride)**2 experts per token. Each expert
processes its fixed patch_size x patch_size patch from the toroidally-padded
sheet via a small SwiGLU MLP. Selected expert outputs are routing-weighted,
scattered to per-expert positions, and overlap-add folded back to the sheet.

Contributions in the wrap-zone of the padded grid are discarded by the
final crop (matches the original ToroidalMoE convention in 05_rtcc).

Returns (output, aux_loss). aux_loss uses the DeepSeek-MoE formulation:
    L_aux = n_experts * sum_i (frac_i * mean_prob_i)
where frac_i is the top-1 routing fraction and mean_prob_i is mean softmax
probability for expert i across the batch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKToroidMoE(nn.Module):
    def __init__(self, grid_side: int, patch_size: int, stride: int,
                 expert_hidden: int, top_k: int, padding_mode: str = "circular"):
        super().__init__()
        self.H = self.W = grid_side
        self.patch_size = patch_size
        self.stride = stride
        self.overlap = patch_size - stride
        self.top_k = top_k
        self.padding_mode = padding_mode

        assert padding_mode in ("circular", "zeros"), padding_mode
        assert self.overlap >= 1
        assert grid_side % stride == 0

        self.n_er = grid_side // stride
        self.n_ec = grid_side // stride
        self.n_experts = self.n_er * self.n_ec
        self.patch_dims = patch_size * patch_size
        self.expert_hidden = expert_hidden

        assert top_k <= self.n_experts

        in_dim = grid_side * grid_side
        self.router = nn.Linear(in_dim, self.n_experts, bias=False)

        # Fused gate+up: [n_experts, patch_dims, 2*expert_hidden] halves first GEMM launches
        self.gate_up = nn.Parameter(torch.empty(self.n_experts, self.patch_dims, 2 * expert_hidden))
        self.down    = nn.Parameter(torch.empty(self.n_experts, expert_hidden, self.patch_dims))
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.router.weight, std=0.02)
        nn.init.normal_(self.gate_up, std=0.02)
        nn.init.normal_(self.down,    std=0.02)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        N = B * S
        p = self.overlap

        # 1. Reshape to grid; pad according to padding_mode.
        # Circular (toroidal): wrap-around via torch.cat (faster than F.pad mode='circular').
        # Zeros (B4 flat-grid): standard zero pad; isolates the toroidal-boundary contribution.
        x_grid = x.view(N, 1, self.H, self.W)
        if self.padding_mode == "circular":
            x_pad = torch.cat([x_grid[:, :, -p:, :], x_grid, x_grid[:, :, :p, :]], dim=2)
            x_pad = torch.cat([x_pad[:, :, :, -p:], x_pad, x_pad[:, :, :, :p]], dim=3)
        else:  # "zeros"
            x_pad = F.pad(x_grid, (p, p, p, p), mode="constant", value=0.0)

        # 2. Unfold all n_experts patches at once -> [N, n_experts, patch_dims]
        patches = F.unfold(x_pad, kernel_size=self.patch_size, stride=self.stride)
        patches = patches.permute(0, 2, 1)

        # 3. Route on the flat input
        xf = x.view(N, D)
        logits = self.router(xf)                                       # [N, n_experts]
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)          # [N, K]
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # 4. Load-balance aux loss (DeepSeek-style)
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device)
            counts.scatter_add_(0, topk_idx[:, 0],
                                torch.ones(N, device=x.device))
            frac = counts / N
        mean_prob = scores.mean(dim=0)
        aux_loss = (frac * mean_prob).sum() * self.n_experts

        # 5. Gather top-K patches and per-token expert weights
        idx_p = topk_idx.unsqueeze(-1).expand(-1, -1, self.patch_dims)
        selected_patches = patches.gather(dim=1, index=idx_p)          # [N, K, patch_dims]
        g_w = self.gate_up[topk_idx]                                   # [N, K, patch_dims, 2*hidden]
        d_w = self.down[topk_idx]                                      # [N, K, hidden, patch_dims]

        # 6. SwiGLU on selected patches
        gu = torch.einsum("nkp,nkph->nkh", selected_patches, g_w)
        g, u = gu.chunk(2, dim=-1)
        h = F.silu(g) * u
        out_patches = torch.einsum("nkh,nkhp->nkp", h, d_w)            # [N, K, patch_dims]

        # 7. Apply routing weights
        out_patches = out_patches * topk_w.unsqueeze(-1)

        # 8. Scatter back to a full [N, n_experts, patch_dims] tensor (zeros elsewhere)
        out_all = torch.zeros(N, self.n_experts, self.patch_dims,
                              device=x.device, dtype=out_patches.dtype)
        out_all.scatter_add_(dim=1, index=idx_p, src=out_patches)

        # 9. Overlap-add fold back to padded grid
        Hp, Wp = self.H + 2 * p, self.W + 2 * p
        folded = F.fold(
            out_all.permute(0, 2, 1),
            output_size=(Hp, Wp),
            kernel_size=self.patch_size,
            stride=self.stride,
        )

        # 10. Crop wrap-zone, reshape back to [B, S, D]
        return folded[:, 0, p:p + self.H, p:p + self.W].reshape(B, S, D), aux_loss
