"""MLA (Multi-head Latent Attention) — adapted from CART (Model_Paper_1).

KV is compressed through a latent bottleneck (mla_latent_dim) then expanded.
K and V up-projections are fused into a single linear (one kernel launch).
RoPE is applied to full head_dim on Q and K (CART convention, not decoupled).
Flash attention via scaled_dot_product_attention.

Different locations use different head counts (prelude=16, recurrent=12, coda=8)
but share the same model_dim, head_dim, and mla_latent_dim.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding


class MLACrossAttention(nn.Module):
    """MLA cross-attention for the recurrent core.

    Q from h (current hidden state). K and V are pre-computed from prelude output e
    and passed in — computed once before the loop, reused every iteration.
    No RoPE (h has no direct token-position correspondence). Attention is causal
    to prevent h[t] from attending to e[t+1], which encodes future token t+1.
    """
    def __init__(self, model_dim: int, n_heads: int, head_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(model_dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, model_dim, bias=False)

    def forward(self, h: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        H, D = self.n_heads, self.head_dim
        Q = self.q_proj(h).view(B, T, H, D).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)


class MLAKVProjection(nn.Module):
    """Computes K and V from prelude output e — called once before the loop.
    K and V up-projections fused: single [latent, 2*H*D] linear then chunk."""
    def __init__(self, model_dim: int, n_heads: int, head_dim: int, mla_latent_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.kv_down = nn.Linear(model_dim, mla_latent_dim, bias=False)
        self.kv_up   = nn.Linear(mla_latent_dim, 2 * n_heads * head_dim, bias=False)

    def forward(self, e: torch.Tensor):
        B, T, _ = e.shape
        H, D = self.n_heads, self.head_dim
        latent = self.kv_down(e)
        kv = self.kv_up(latent)                               # [B, T, 2*H*D]
        K, V = kv.chunk(2, dim=-1)                            # each [B, T, H*D]
        K = K.view(B, T, H, D).transpose(1, 2)                # [B, H, T, D]
        V = V.view(B, T, H, D).transpose(1, 2)
        return K, V


class MLASelfAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, head_dim: int,
                 mla_latent_dim: int, rope_base: float = 10_000.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim

        self.q_proj  = nn.Linear(model_dim, n_heads * head_dim, bias=False)
        self.kv_down = nn.Linear(model_dim, mla_latent_dim, bias=False)
        self.kv_up   = nn.Linear(mla_latent_dim, 2 * n_heads * head_dim, bias=False)
        self.o_proj  = nn.Linear(n_heads * head_dim, model_dim, bias=False)

        self.rope = RotaryEmbedding(head_dim, rope_base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        H, D = self.n_heads, self.head_dim

        Q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        latent = self.kv_down(x)
        kv = self.kv_up(latent)
        K, V = kv.chunk(2, dim=-1)
        K = K.view(B, T, H, D).transpose(1, 2)
        V = V.view(B, T, H, D).transpose(1, 2)

        Q = self.rope(Q, T)
        K = self.rope(K, T)

        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, H * D)
        return self.o_proj(out)
