"""MLA (Multi-head Latent Attention) — adapted from CART (Model_Paper_1).

KV is compressed through a latent bottleneck (mla_latent_dim) then expanded.
RoPE is applied to full head_dim on Q and K (CART convention, not decoupled).
Flash attention via scaled_dot_product_attention.

Different locations use different head counts (prelude=16, recurrent=12, coda=8)
but share the same model_dim, head_dim, and mla_latent_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding


class MLASelfAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, head_dim: int,
                 mla_latent_dim: int, rope_base: float = 10_000.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj  = nn.Linear(model_dim, n_heads * head_dim, bias=False)
        self.kv_down = nn.Linear(model_dim, mla_latent_dim, bias=False)
        self.k_up    = nn.Linear(mla_latent_dim, n_heads * head_dim, bias=False)
        self.v_up    = nn.Linear(mla_latent_dim, n_heads * head_dim, bias=False)
        self.o_proj  = nn.Linear(n_heads * head_dim, model_dim, bias=False)

        self.rope = RotaryEmbedding(head_dim, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)   # [B, H, T, D]
        latent = self.kv_down(x)
        K = self.k_up(latent).view(B, T, H, D).transpose(1, 2)
        V = self.v_up(latent).view(B, T, H, D).transpose(1, 2)

        Q = self.rope(Q, T)
        K = self.rope(K, T)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)
