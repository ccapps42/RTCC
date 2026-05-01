"""LTI-stable recurrent injection — full formulation with prelude anchor.

h = sigmoid(A) * h_input + B * e + transformer_out

A: learnable diagonal, sigmoid-parameterized → spectral radius < 1 guaranteed.
B: learnable diagonal scale for prelude output injection.
e: prelude output (fixed across all loop iterations, passed from model.forward).

Initialized so A ≈ 0.9 (near-standard residual), B ≈ 0 (anchor inactive at start).
"""
import math
import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    def __init__(self, dim: int, lti_init_value: float = 0.9):
        super().__init__()
        # sigmoid_inverse(v) = log(v / (1 - v))
        init_a = math.log(lti_init_value / (1.0 - lti_init_value))
        self.a_param = nn.Parameter(torch.full((dim,), init_a))
        self.b_param = nn.Parameter(torch.zeros(dim))

    def forward(self, h_input: torch.Tensor, e: torch.Tensor,
                transformer_out: torch.Tensor) -> torch.Tensor:
        A = torch.sigmoid(self.a_param)   # [dim], all in (0, 1)
        return A * h_input + self.b_param * e + transformer_out

    def spectral_radius(self) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.a_param).max().item()
