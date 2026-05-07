"""Flat Grid Overlap — 2D overlapping patches with flat (zero-pad) boundary.

Isolates the toroidal boundary contribution: identical to RTCC except for padding.
Loop order matches CART/RTCC exactly.
"""
import sys
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


_cfg = _load_sibling("_flat_grid_config", "config.py")
FlatGridConfig = _cfg.FlatGridConfig

_fg = _load_sibling("_flat_grid_moe", "flat_grid_moe.py")
FlatGridMoE = _fg.FlatGridMoE

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.components.norm import RMSNorm
from shared.components.attention import MLASelfAttention, MLACrossAttention, MLAKVProjection
from shared.components.lti import LTIInjection
from shared.components.hyper import HyperConnection
from shared.components.lie import LoopIndexEmbedding


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DenseBlock(nn.Module):
    def __init__(self, model_dim, n_heads, head_dim, mla_latent_dim, ffn_hidden):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLASelfAttention(model_dim, n_heads, head_dim, mla_latent_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = SwiGLU(model_dim, ffn_hidden)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class FlatGridBlock(nn.Module):
    """Recurrent block: MLA cross-attn (K/V from prelude e) + FlatGridMoE FFN."""
    def __init__(self, model_dim, n_heads, head_dim,
                 grid_rows, grid_cols, patch_size, stride, expert_hidden):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLACrossAttention(model_dim, n_heads, head_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = FlatGridMoE(grid_rows, grid_cols, patch_size, stride, expert_hidden)

    def forward(self, x, K, V):
        x = x + self.attn(self.norm1(x), K, V)
        x = x + self.ffn(self.norm2(x))
        return x


class FlatGridModel(nn.Module):
    def __init__(self, cfg: FlatGridConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim

        self.embedding = nn.Embedding(cfg.vocab_size, d)

        self.prelude = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_prelude, cfg.head_dim, cfg.mla_latent_dim, cfg.ffn_hidden)
            for _ in range(cfg.prelude_layers)
        ])

        self.recurrent = FlatGridBlock(
            d, cfg.n_heads_recurrent, cfg.head_dim,
            cfg.grid_rows, cfg.grid_cols, cfg.patch_size, cfg.stride, cfg.expert_hidden,
        )
        self.kv_proj   = MLAKVProjection(d, cfg.n_heads_recurrent, cfg.head_dim, cfg.mla_latent_dim)

        self.coda = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_coda, cfg.head_dim, cfg.mla_latent_dim, cfg.ffn_hidden)
            for _ in range(cfg.coda_layers)
        ])

        self.hyper = HyperConnection(cfg.n_hyper)
        self.lie   = LoopIndexEmbedding(d, cfg.lie_dim)
        self.lti   = LTIInjection(d, cfg.lti_init_value)
        self.norm  = RMSNorm(d)

        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, n_loops: int | None = None) -> torch.Tensor:
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

        for block in self.coda:
            h = block(h)

        h = self.norm(h)
        return self.lm_head(h)
