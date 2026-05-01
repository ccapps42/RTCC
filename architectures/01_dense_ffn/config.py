import sys
import math
from dataclasses import dataclass
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import BaseConfig


@dataclass
class DenseFfnConfig(BaseConfig):
    architecture: str = "dense_ffn"

    @property
    def ffn_hidden(self) -> int:
        nominal = int(8 / 3 * self.model_dim)
        return math.ceil(nominal / 256) * 256
