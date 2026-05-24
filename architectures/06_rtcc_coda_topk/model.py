"""TopKToroidCodaModel — RTCC Paper 2.

CART-identical backbone (P=6 dense prelude + R=6 dense recurrent core) with
the single coda FFN replaced by up_proj -> top-K toroid MoE -> down_proj.

Recurrent loop order matches CART exactly:
    hyper.combine(buffer) -> LIE -> recurrent(h_input, K, V) -> LTI -> update_buffer

Forward returns plain logits by default (so eval works). Set return_aux=True
during training to get (logits, aux_loss) — the trainer subclass does this.
"""
import sys
import math
import importlib.util
from pathlib import Path

_here = Path(__file__).parent
_root = _here.parent.parent
sys.path.insert(0, str(_root))


def _load_sibling(name, filename):
    spec = importlib.util.spec_from_file_location(name, _here / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_cfg_mod = _load_sibling("_topk_toroid_coda_config", "config.py")
TopKToroidCodaConfig = _cfg_mod.TopKToroidCodaConfig

_toroid_mod = _load_sibling("_topk_toroid_module", "top_k_toroid.py")
TopKToroidMoE = _toroid_mod.TopKToroidMoE

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.components.norm import RMSNorm
from shared.components.attention import MLASelfAttention, MLACrossAttention, MLAKVProjection
from shared.components.lti import LTIInjection
from shared.components.hyper import HyperConnection
from shared.components.lie import LoopIndexEmbedding


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseBlock(nn.Module):
    """Prelude block: MLA self-attn + SwiGLU FFN. Matches CART prelude."""
    def __init__(self, model_dim, n_heads, head_dim, mla_latent_dim, ffn_hidden):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLASelfAttention(model_dim, n_heads, head_dim, mla_latent_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = SwiGLU(model_dim, ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class RecurrentBlock(nn.Module):
    """CART recurrent block: MLA cross-attn (K/V from prelude e) + SwiGLU FFN."""
    def __init__(self, model_dim, n_heads, head_dim, ffn_hidden):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLACrossAttention(model_dim, n_heads, head_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = SwiGLU(model_dim, ffn_hidden)

    def forward(self, x: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), K, V)
        x = x + self.ffn(self.norm2(x))
        return x


class TopKToroidCodaBlock(nn.Module):
    """Coda: MLA self-attn + (up_proj -> TopKToroidMoE -> down_proj). Returns (out, aux)."""

    def __init__(self, model_dim, n_heads, head_dim, mla_latent_dim,
                 up_dim, grid_side, patch_size, stride, expert_hidden, top_k,
                 padding_mode="circular"):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLASelfAttention(model_dim, n_heads, head_dim, mla_latent_dim)
        self.norm2 = RMSNorm(model_dim)
        self.up_proj   = nn.Linear(model_dim, up_dim, bias=False)
        self.moe       = TopKToroidMoE(grid_side, patch_size, stride, expert_hidden, top_k,
                                       padding_mode=padding_mode)
        self.down_proj = nn.Linear(up_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x))
        x_norm = self.norm2(x)
        x_up = self.up_proj(x_norm)
        x_moe, aux = self.moe(x_up)
        x_down = self.down_proj(x_moe)
        return x + x_down, aux


class TopKToroidCodaModel(nn.Module):
    def __init__(self, cfg: TopKToroidCodaConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        ffn_hidden = cfg.ffn_hidden

        self.embedding = nn.Embedding(cfg.vocab_size, d)

        self.prelude = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_prelude, cfg.head_dim, cfg.mla_latent_dim, ffn_hidden)
            for _ in range(cfg.prelude_layers)
        ])

        self.recurrent = RecurrentBlock(d, cfg.n_heads_recurrent, cfg.head_dim, ffn_hidden)
        self.kv_proj   = MLAKVProjection(d, cfg.n_heads_recurrent, cfg.head_dim, cfg.mla_latent_dim)

        assert cfg.coda_layers == 1, "RTCC coda is locked at 1 layer"
        self.coda = TopKToroidCodaBlock(
            d, cfg.n_heads_coda, cfg.head_dim, cfg.mla_latent_dim,
            cfg.up_dim, cfg.grid_side, cfg.patch_size, cfg.stride,
            cfg.expert_hidden, cfg.top_k,
            padding_mode=cfg.padding_mode,
        )

        self.hyper = HyperConnection(cfg.n_hyper)
        self.lie   = LoopIndexEmbedding(d, cfg.lie_dim)
        self.lti   = LTIInjection(d, cfg.lti_init_value)
        self.norm  = RMSNorm(d)

        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

        compression = cfg.expert_hidden / cfg.patch_dims
        print(
            f"RTCC coda top-K: K={cfg.top_k} of {cfg.n_experts} experts "
            f"({cfg.n_experts_per_side}x{cfg.n_experts_per_side}) | "
            f"patch={cfg.patch_size}x{cfg.patch_size} stride={cfg.stride} "
            f"overlap={cfg.overlap} | sheet={cfg.grid_side}x{cfg.grid_side} "
            f"(up_dim={cfg.up_dim}) | expert_hidden={cfg.expert_hidden} "
            f"(compression={compression:.2f}) | padding={cfg.padding_mode} | "
            f"aux_coef={cfg.moe_aux_loss_coef}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, n_loops: int | None = None,
                return_aux: bool = False):
        if n_loops is None:
            n_loops = self.cfg.max_loop_iters

        h = self.embedding(x)
        for block in self.prelude:
            h = block(h)
        e = h
        K, V = self.kv_proj(e)

        h = e.clone()
        buffer = self.hyper.init_buffer(h)

        for r in range(n_loops):
            h_input = self.hyper.combine(buffer)
            h_input = self.lie(h_input, r)
            block_out = self.recurrent(h_input, K, V)
            h = self.lti(h_input, block_out)
            buffer = self.hyper.update_buffer(buffer, h)

        h, aux = self.coda(h)
        h = self.norm(h)
        logits = self.lm_head(h)

        if return_aux:
            return logits, aux
        return logits
