"""TopKToroidCodaConfig — RTCC Paper 2 architecture.

CART d=1024 backbone (locked from CART Stage 2: P=6, R=6, coda=1) with the
single coda FFN replaced by up_proj 1024->4096 -> top-K toroid MoE on a 64x64
toroidal sheet -> down_proj 4096->1024.

expert_hidden auto-derives from patch_size at compression ratio 0.4 (rounded
to multiples of 8) so the K x overlap ablation does not conflate geometry
with per-expert capacity. Set expert_hidden explicitly in YAML to override.
"""
import sys
import math
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class TopKToroidCodaConfig(BaseConfig):
    architecture: str = "rtcc_coda_topk"

    # ----- CART backbone (locked) -----
    model_dim: int = 1024
    prelude_layers: int = 6
    coda_layers: int = 1
    max_loop_iters: int = 6

    # ----- Toroid coda -----
    up_dim: int = 4096            # = grid_side ** 2
    grid_side: int = 64
    patch_size: int = 9           # overlap = patch_size - stride; ablation axis
    stride: int = 8               # fixed -> 8x8 = 64 experts on a 64x64 sheet
    expert_hidden: int = 0        # 0 = auto-derive from patch_size at ratio 0.4
    top_k: int = 2                # ablation axis
    moe_aux_loss_coef: float = 0.01
    padding_mode: str = "circular"  # "circular" = toroidal (default); "zeros" = B4 flat-grid

    def __post_init__(self):
        assert self.padding_mode in ("circular", "zeros"), (
            f"padding_mode must be 'circular' or 'zeros', got {self.padding_mode!r}"
        )
        assert self.up_dim == self.grid_side ** 2, (
            f"up_dim {self.up_dim} != grid_side**2 ({self.grid_side}**2 = {self.grid_side**2})"
        )
        self.overlap = self.patch_size - self.stride
        assert self.overlap >= 1, (
            f"patch_size ({self.patch_size}) must exceed stride ({self.stride}) for overlap >= 1"
        )
        assert self.grid_side % self.stride == 0, (
            f"stride {self.stride} does not tile grid_side {self.grid_side}"
        )
        self.n_experts_per_side = self.grid_side // self.stride
        self.n_experts = self.n_experts_per_side ** 2
        self.patch_dims = self.patch_size ** 2
        assert self.top_k <= self.n_experts, (
            f"top_k {self.top_k} > n_experts {self.n_experts}"
        )

        if self.expert_hidden == 0:
            # Compression ratio 0.4, rounded to nearest multiple of 8 (Tensor Core alignment)
            nominal = self.patch_dims * 0.4
            self.expert_hidden = max(8, int(round(nominal / 8)) * 8)

    @property
    def ffn_hidden(self) -> int:
        """Dense SwiGLU hidden dim used in prelude blocks AND in the recurrent core
        FFN (matches CART convention: 8/3 * d, rounded up to 256)."""
        nominal = int(8 / 3 * self.model_dim)
        return math.ceil(nominal / 256) * 256
