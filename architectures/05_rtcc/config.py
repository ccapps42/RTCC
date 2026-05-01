"""RTCCConfig — computes privacy% from first principles and asserts all constraints."""
import sys
import math
from dataclasses import dataclass, field
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class RTCCConfig(BaseConfig):
    architecture: str = "rtcc"

    # Grid shape (must satisfy model_dim == grid_rows * grid_cols)
    grid_rows: int = 32
    grid_cols: int = 24   # 32*24 = 768

    # Patch / expert layout
    patch_size: int = 10
    stride: int = 8
    expert_hidden: int = 256   # hidden dim inside each expert MLP

    def __post_init__(self):
        # Verify grid matches model_dim
        assert self.grid_rows * self.grid_cols == self.model_dim, (
            f"grid {self.grid_rows}x{self.grid_cols}={self.grid_rows*self.grid_cols} "
            f"!= model_dim {self.model_dim}"
        )
        # Compute derived geometry
        self.overlap = self.patch_size - self.stride
        assert self.overlap >= 0, "patch_size must be >= stride"
        self.private_side = self.patch_size - 2 * self.overlap
        assert self.private_side >= 1, (
            f"No private dims — private_side={self.private_side}. "
            "Reduce overlap (increase stride or decrease patch_size)."
        )
        self.private_dims = self.private_side ** 2
        self.privacy_pct = self.private_dims / (self.patch_size ** 2)

        assert self.grid_rows % self.stride == 0, (
            f"stride {self.stride} does not tile row axis ({self.grid_rows} rows)"
        )
        assert self.grid_cols % self.stride == 0, (
            f"stride {self.stride} does not tile col axis ({self.grid_cols} cols)"
        )
        self.n_expert_rows = self.grid_rows // self.stride
        self.n_expert_cols = self.grid_cols // self.stride
        # Verify F.unfold with 2*overlap padding gives exactly n_expert_rows/cols patches
        _ner = (self.grid_rows + 2*self.overlap - self.patch_size) // self.stride + 1
        _nec = (self.grid_cols + 2*self.overlap - self.patch_size) // self.stride + 1
        assert _ner == self.n_expert_rows, (
            f"Toroidal unfold gives {_ner} expert-rows, expected {self.n_expert_rows}. "
            f"Check patch_size={self.patch_size}/stride={self.stride}/grid_rows={self.grid_rows}."
        )
        assert _nec == self.n_expert_cols, (
            f"Toroidal unfold gives {_nec} expert-cols, expected {self.n_expert_cols}. "
            f"Check patch_size={self.patch_size}/stride={self.stride}/grid_cols={self.grid_cols}."
        )
        assert self.n_expert_rows >= 3, (
            f"Too few experts on row axis: {self.n_expert_rows} (need >=3)"
        )
        assert self.n_expert_cols >= 3, (
            f"Too few experts on col axis: {self.n_expert_cols} (need >=3)"
        )
        self.n_experts = self.n_expert_rows * self.n_expert_cols
        self.patch_dims = self.patch_size ** 2

    @property
    def ffn_hidden(self) -> int:
        nominal = int(8 / 3 * self.model_dim)
        return math.ceil(nominal / 256) * 256
