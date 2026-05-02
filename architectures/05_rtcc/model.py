"""RTCC — Recurrent Toroidal Cortical Columns.

Recurrent block replaces SwiGLU FFN with ToroidalMoE.
Loop order matches CART exactly:
    hyper.combine(buffer) -> LIE -> block(h_input) -> LTI(h_input, e, block_out) -> update_buffer

All experts are always active (position-fixed, no routing).
No aux loss — ToroidalMoE is parameter-efficient by design.
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


_cfg = _load_sibling("_rtcc_config", "config.py")
RTCCConfig = _cfg.RTCCConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.components.norm import RMSNorm
from shared.components.attention import MLASelfAttention
from shared.components.lti import LTIInjection
from shared.components.hyper import HyperConnection
from shared.components.lie import LoopIndexEmbedding


def _load_toroidal_moe():
    """Load toroidal_moe via file path (digit-prefix dir can't be a package)."""
    spec = importlib.util.spec_from_file_location(
        "toroidal_moe", _here / "toroidal_moe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ToroidalMoE


ToroidalMoE = _load_toroidal_moe()


# ---------------------------------------------------------------------------
# Dense SwiGLU (prelude / coda)
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up   = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

class DenseBlock(nn.Module):
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


class RTCCBlock(nn.Module):
    """Recurrent block: MLA attention + ToroidalMoE FFN."""

    def __init__(self, model_dim, n_heads, head_dim, mla_latent_dim,
                 grid_rows, grid_cols, patch_size, stride, expert_hidden):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLASelfAttention(model_dim, n_heads, head_dim, mla_latent_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = ToroidalMoE(grid_rows, grid_cols, patch_size, stride, expert_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class RTCCModel(nn.Module):
    def __init__(self, cfg: RTCCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim

        self.embedding = nn.Embedding(cfg.vocab_size, d)

        self.prelude = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_prelude, cfg.head_dim,
                       cfg.mla_latent_dim, cfg.ffn_hidden)
            for _ in range(cfg.prelude_layers)
        ])

        self.recurrent = RTCCBlock(
            d, cfg.n_heads_recurrent, cfg.head_dim, cfg.mla_latent_dim,
            cfg.grid_rows, cfg.grid_cols, cfg.patch_size, cfg.stride, cfg.expert_hidden,
        )

        self.coda = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_coda, cfg.head_dim,
                       cfg.mla_latent_dim, cfg.ffn_hidden)
            for _ in range(cfg.coda_layers)
        ])

        self.hyper = HyperConnection(cfg.n_hyper)
        self.lie   = LoopIndexEmbedding(d, cfg.lie_dim)
        self.lti   = LTIInjection(d, cfg.lti_init_value)
        self.norm  = RMSNorm(d)

        self.lm_head = nn.Linear(d, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self._init_weights()

        print(
            f"RTCC config: {cfg.grid_rows}x{cfg.grid_cols} grid | "
            f"patch={cfg.patch_size}x{cfg.patch_size} stride={cfg.stride} "
            f"overlap={cfg.overlap} | {cfg.n_experts} experts | "
            f"privacy={cfg.privacy_pct:.0%}"
        )

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

        h = e.clone()
        buffer = self.hyper.init_buffer(h)

        for r in range(n_loops):
            h_input = self.hyper.combine(buffer)
            h_input = self.lie(h_input, r)
            block_out = self.recurrent(h_input)
            h = self.lti(h_input, e, block_out)
            buffer = self.hyper.update_buffer(buffer, h)

        for block in self.coda:
            h = block(h)

        h = self.norm(h)
        return self.lm_head(h)
