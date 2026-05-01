"""AdamW optimizer with cosine LR schedule and linear warmup."""
import math
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, lr_max: float, weight_decay: float) -> AdamW:
    # Don't apply weight decay to biases and norm params
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.dim() < 2]
    return AdamW([
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=lr_max, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer: AdamW, warmup_steps: int, total_steps: int,
                    lr_max: float, lr_min: float) -> LambdaLR:
    min_ratio = lr_min / lr_max

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
