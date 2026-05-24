"""Trainer subclass for the top-K toroid coda arch.

Calls forward with return_aux=True so the aux loss is captured for the
load-balance objective. The model's default forward returns plain logits
so evaluate_perplexity_bin keeps working unchanged.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from shared.training.trainer import Trainer


class TopKToroidTrainer(Trainer):
    def _forward(self, x, y, n_loops):
        logits, aux = self.model(x, n_loops=n_loops, return_aux=True)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )
        scaled_aux = aux * self.cfg.moe_aux_loss_coef
        return loss, scaled_aux
