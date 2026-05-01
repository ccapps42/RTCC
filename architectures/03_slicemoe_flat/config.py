"""SliceMoE Flat — non-overlapping dimensional partition, position-fixed experts."""
import sys
import math
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class SliceMoeFlatConfig(BaseConfig):
    architecture: str = "slicemoe_flat"

    # Slice config — must divide model_dim evenly
    slice_dim: int = 64     # dims per expert (768/64=12 experts at paper scale)
    expert_hidden: int = 256  # hidden dim inside each expert MLP

    def __post_init__(self):
        assert self.model_dim % self.slice_dim == 0, (
            f"model_dim {self.model_dim} not divisible by slice_dim {self.slice_dim}"
        )
        self.n_experts = self.model_dim // self.slice_dim

    @property
    def ffn_hidden(self) -> int:
        nominal = int(8 / 3 * self.model_dim)
        return math.ceil(nominal / 256) * 256
