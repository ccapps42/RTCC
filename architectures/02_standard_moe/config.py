import sys
import math
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class StandardMoEConfig(BaseConfig):
    architecture: str = "standard_moe"

    # MoE routing
    n_experts: int = 16       # routed experts
    n_shared: int = 2         # always-active shared experts (DeepSeek-style)
    top_k: int = 2
    expert_dim: int = 256     # hidden dim inside each expert (not ffn_hidden)
    moe_aux_loss_coef: float = 0.01
