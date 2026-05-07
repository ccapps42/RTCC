"""LTI-stable recurrent injection — matches CART exactly.

h = sigmoid(A) * h_input + transformer_out

A: learnable diagonal, sigmoid-parameterized → spectral radius < 1 guaranteed.
Initialized so A ≈ lti_init_value (near-standard residual).
"""
import math
import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    def __init__(self, dim: int, lti_init_value: float = 0.9):
        super().__init__()
        init_a = math.log(lti_init_value / (1.0 - lti_init_value))
        self.a_param = nn.Parameter(torch.full((dim,), init_a))

    def forward(self, h_input: torch.Tensor,
                transformer_out: torch.Tensor) -> torch.Tensor:
        A = torch.sigmoid(self.a_param)
        return A * h_input + transformer_out

    def spectral_radius(self) -> float:
        with torch.no_grad():
            return torch.sigmoid(self.a_param).max().item()
