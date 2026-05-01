"""MoE trainer subclass — unpacks (logits, aux_loss) from model.forward."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from shared.training.trainer import Trainer


class MoETrainer(Trainer):
    def _forward(self, x, y, n_loops):
        logits, aux = self.model(x, n_loops=n_loops)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        # Scale aux by config coefficient
        scaled_aux = aux * self.cfg.moe_aux_loss_coef
        return loss, scaled_aux
