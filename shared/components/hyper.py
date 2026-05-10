"""Hyper-connections — ported from CART (Model_Paper_1).

Maintains a ring buffer of the last n_hyper hidden states.
Combines them with learned scalar weights (softmax-normalized).
Initialized to residual baseline: weights = [1, 0, 0, ...].

Placement matches CART exactly: combine BEFORE the block, update AFTER LTI.
"""
import torch
import torch.nn as nn


class HyperConnection(nn.Module):
    def __init__(self, n_hyper: int):
        super().__init__()
        init = torch.zeros(n_hyper)
        init[0] = 1.0
        self.weights = nn.Parameter(init)
        self.n_hyper = n_hyper

    def init_buffer(self, h: torch.Tensor) -> list[torch.Tensor]:
        # All slots reference the same h; safe because LTI builds fresh tensors
        # and we never mutate buffer entries in place.
        return [h] * self.n_hyper

    def combine(self, buffer: list[torch.Tensor]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        stacked = torch.stack(buffer, dim=0)             # [n_hyper, B, T, D]
        return torch.einsum("h,hbtd->btd", w, stacked)

    def update_buffer(self, buffer: list[torch.Tensor],
                      h_new: torch.Tensor) -> list[torch.Tensor]:
        return [h_new] + buffer[:-1]
