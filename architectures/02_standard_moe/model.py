"""Standard MoE baseline — top-k gated MoE FFN with shared experts.

Recurrent block uses vectorized einsum dispatch (no per-expert Python loop).
Shared experts (always active) follow DeepSeek-MoE convention.
Recurrent loop order matches CART exactly:
    hyper.combine(buffer) -> LIE -> block(h_input) -> LTI(h_input, e, block_out) -> update_buffer
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


_cfg = _load_sibling("_std_moe_config", "config.py")
StandardMoEConfig = _cfg.StandardMoEConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.components.norm import RMSNorm
from shared.components.attention import MLASelfAttention
from shared.components.lti import LTIInjection
from shared.components.hyper import HyperConnection
from shared.components.lie import LoopIndexEmbedding


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
# Vectorized MoE FFN
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """Top-k gated MoE with shared experts and vectorized einsum dispatch.

    Returns (output, aux_loss).
    Expert weights stored as batched parameters — no Python loop over experts.
    """

    def __init__(self, dim: int, expert_dim: int, n_experts: int, n_shared: int, top_k: int):
        super().__init__()
        self.n_experts = n_experts
        self.n_shared  = n_shared
        self.top_k     = top_k

        self.router = nn.Linear(dim, n_experts, bias=False)

        # Routed expert weights — batched [n_experts, dim, expert_dim]
        self.r_gate = nn.Parameter(torch.empty(n_experts, dim, expert_dim))
        self.r_up   = nn.Parameter(torch.empty(n_experts, dim, expert_dim))
        self.r_down = nn.Parameter(torch.empty(n_experts, expert_dim, dim))

        # Shared expert weights — batched [n_shared, dim, expert_dim]
        if n_shared > 0:
            self.s_gate = nn.Parameter(torch.empty(n_shared, dim, expert_dim))
            self.s_up   = nn.Parameter(torch.empty(n_shared, dim, expert_dim))
            self.s_down = nn.Parameter(torch.empty(n_shared, expert_dim, dim))
        else:
            self.s_gate = self.s_up = self.s_down = None

        self._init_weights()

    def _init_weights(self):
        for p in [self.r_gate, self.r_up, self.r_down]:
            nn.init.normal_(p, std=0.02)
        if self.n_shared > 0:
            for p in [self.s_gate, self.s_up, self.s_down]:
                nn.init.normal_(p, std=0.02)

    def _swiglu_batched(self, x, gate_w, up_w, down_w):
        """x: [N, k, D], weights: [N, k, D, E] or [N, k, E, D] — returns [N, k, D]."""
        gate = torch.einsum("nkd,nkde->nke", x, gate_w)
        up   = torch.einsum("nkd,nkde->nke", x, up_w)
        h    = F.silu(gate) * up
        return torch.einsum("nke,nked->nkd", h, down_w)

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        N = B * T
        xf = x.view(N, D)                          # [N, D]

        # ----- Routing -----
        logits = self.router(xf)                   # [N, E]
        scores = F.softmax(logits, dim=-1)
        topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)   # [N, k]
        topk_w = topk_vals / topk_vals.sum(dim=-1, keepdim=True) # [N, k] normalized

        # ----- Aux loss (load balancing) -----
        with torch.no_grad():
            counts = torch.zeros(self.n_experts, device=x.device)
            counts.scatter_add_(0, topk_idx[:, 0],
                                torch.ones(N, device=x.device))
            frac = counts / N
        mean_prob = scores.mean(dim=0)
        aux_loss = (frac * mean_prob).sum() * self.n_experts

        # ----- Vectorized routed expert dispatch -----
        # Gather weight slices for selected experts: [N, k, D, E_dim]
        xf_exp = xf.unsqueeze(1).expand(-1, self.top_k, -1)     # [N, k, D]
        g_w = self.r_gate[topk_idx]          # [N, k, D, E_dim]
        u_w = self.r_up[topk_idx]            # [N, k, D, E_dim]
        d_w = self.r_down[topk_idx]          # [N, k, E_dim, D]
        routed_out = self._swiglu_batched(xf_exp, g_w, u_w, d_w)  # [N, k, D]
        out = (routed_out * topk_w.unsqueeze(-1)).sum(dim=1)       # [N, D]

        # ----- Shared experts (always active) -----
        if self.n_shared > 0:
            # Expand x for all shared experts: [N, S, D]
            xf_s = xf.unsqueeze(1).expand(-1, self.n_shared, -1)
            # Expand weights to match [N, S, D, E_dim] (broadcast from [S, D, E_dim])
            s_g = self.s_gate.unsqueeze(0).expand(N, -1, -1, -1)
            s_u = self.s_up.unsqueeze(0).expand(N, -1, -1, -1)
            s_d = self.s_down.unsqueeze(0).expand(N, -1, -1, -1)
            shared_out = self._swiglu_batched(xf_s, s_g, s_u, s_d)  # [N, S, D]
            out = out + shared_out.sum(dim=1)

        return out.view(B, T, D), aux_loss


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

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


class MoEBlock(nn.Module):
    def __init__(self, model_dim, n_heads, head_dim, mla_latent_dim,
                 expert_dim, n_experts, n_shared, top_k):
        super().__init__()
        self.norm1 = RMSNorm(model_dim)
        self.attn  = MLASelfAttention(model_dim, n_heads, head_dim, mla_latent_dim)
        self.norm2 = RMSNorm(model_dim)
        self.ffn   = MoEFFN(model_dim, expert_dim, n_experts, n_shared, top_k)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        moe_out, aux = self.ffn(self.norm2(x))
        x = x + moe_out
        return x, aux


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class StandardMoEModel(nn.Module):
    def __init__(self, cfg: StandardMoEConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.model_dim
        import math
        ffn_hidden = math.ceil(int(8 / 3 * d) / 256) * 256

        self.embedding = nn.Embedding(cfg.vocab_size, d)

        self.prelude = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_prelude, cfg.head_dim, cfg.mla_latent_dim, ffn_hidden)
            for _ in range(cfg.prelude_layers)
        ])

        self.recurrent = MoEBlock(
            d, cfg.n_heads_recurrent, cfg.head_dim, cfg.mla_latent_dim,
            cfg.expert_dim, cfg.n_experts, cfg.n_shared, cfg.top_k,
        )

        self.coda = nn.ModuleList([
            DenseBlock(d, cfg.n_heads_coda, cfg.head_dim, cfg.mla_latent_dim, ffn_hidden)
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
        # MoEFFN has its own _init_weights called in __init__

    def forward(self, x: torch.Tensor, n_loops: int | None = None):
        if n_loops is None:
            n_loops = self.cfg.max_loop_iters

        h = self.embedding(x)
        for block in self.prelude:
            h = block(h)
        e = h

        h = e.clone()
        buffer = self.hyper.init_buffer(h)
        total_aux = torch.tensor(0.0, device=x.device)

        for r in range(n_loops):
            h_input = self.hyper.combine(buffer)
            h_input = self.lie(h_input, r)
            block_out, aux = self.recurrent(h_input)
            total_aux = total_aux + aux
            h = self.lti(h_input, e, block_out)
            buffer = self.hyper.update_buffer(buffer, h)

        for block in self.coda:
            h = block(h)

        h = self.norm(h)
        return self.lm_head(h), total_aux / n_loops
