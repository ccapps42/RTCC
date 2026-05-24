"""AdamW optimizer with cosine LR schedule and linear warmup."""
import math
import torch
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(model: torch.nn.Module, lr_max: float, weight_decay: float):
    # Three groups:
    #   decay      : 2D weights (Linear, embedding tables), get the full weight_decay
    #   no_decay   : 1D params (RMSNorm gain, biases, LTI a_param, etc.)
    #   no_decay   : routers — Linear weights that we exclude from decay because
    #                in top-K MoE the router has weak learning signal (token loss
    #                only flows through routing weights via the K-normalized
    #                weighting). Weight decay + DeepSeek-style aux loss together
    #                squeeze the router toward uniform output (~0.5/0.5 at K=2),
    #                killing differentiation. See router-collapse diagnostic
    #                2026-05-24 in CLAUDE.md / project memory.
    def _is_router(name: str) -> bool:
        return "router" in name

    decay_params = []
    no_decay_params = []
    n_router = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _is_router(n):
            no_decay_params.append(p)
            n_router += 1
        elif p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    print(f"Optimizer param groups: "
          f"{len(decay_params)} decay (wd={weight_decay}), "
          f"{len(no_decay_params)} no-decay (incl. {n_router} router tensor(s))")
    try:
        from bitsandbytes.optim import AdamW8bit
        optimizer = AdamW8bit(groups, lr=lr_max, betas=(0.9, 0.95), eps=1e-8)
        print("Optimizer: AdamW8bit (bitsandbytes)")
    except ImportError:
        from torch.optim import AdamW
        optimizer = AdamW(groups, lr=lr_max, betas=(0.9, 0.95), eps=1e-8)
        print("Optimizer: AdamW (bitsandbytes not available)")
    return optimizer


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
