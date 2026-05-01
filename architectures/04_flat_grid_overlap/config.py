"""FlatGridConfig — same geometry as RTCC but flat (zero) boundary padding."""
import sys
import math
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class FlatGridConfig(BaseConfig):
    architecture: str = "flat_grid_overlap"

    grid_rows: int = 32
    grid_cols: int = 24
    patch_size: int = 10
    stride: int = 8
    expert_hidden: int = 256

    def __post_init__(self):
        assert self.grid_rows * self.grid_cols == self.model_dim, (
            f"grid {self.grid_rows}x{self.grid_cols}={self.grid_rows*self.grid_cols} "
            f"!= model_dim {self.model_dim}"
        )
        self.overlap = self.patch_size - self.stride
        assert self.overlap >= 0
        self.private_side = self.patch_size - 2 * self.overlap
        assert self.private_side >= 1
        self.private_dims = self.private_side ** 2
        self.privacy_pct = self.private_dims / (self.patch_size ** 2)

        assert self.grid_rows % self.stride == 0
        assert self.grid_cols % self.stride == 0
        self.n_expert_rows = self.grid_rows // self.stride
        self.n_expert_cols = self.grid_cols // self.stride
        assert self.n_expert_rows >= 3
        assert self.n_expert_cols >= 3
        self.n_experts = self.n_expert_rows * self.n_expert_cols
        self.patch_dims = self.patch_size ** 2

    @property
    def ffn_hidden(self) -> int:
        nominal = int(8 / 3 * self.model_dim)
        return math.ceil(nominal / 256) * 256
