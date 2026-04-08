from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def build_feedforward(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation_cls: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(activation_cls())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dims: tuple[int, ...] = (128, 128, 64, 32),
        activation_cls: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.network = build_feedforward(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation_cls=activation_cls,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

