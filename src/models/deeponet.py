from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from .mlp import build_feedforward


class VanillaDeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim: int = 10,
        trunk_input_dim: int = 1,
        latent_dim: int = 64,
        branch_hidden_dims: Iterable[int] = (128, 128),
        trunk_hidden_dims: Iterable[int] = (64, 64),
        activation_cls: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.branch_net = build_feedforward(
            input_dim=branch_input_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.trunk_net = build_feedforward(
            input_dim=trunk_input_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        branch_inputs: torch.Tensor,
        trunk_inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if trunk_inputs is None:
            if branch_inputs.shape[-1] != self.branch_input_dim + self.trunk_input_dim:
                raise ValueError(
                    "Pointwise DeepONet input must contain branch and trunk features concatenated."
                )
            branch_x = branch_inputs[:, : self.branch_input_dim]
            trunk_x = branch_inputs[:, self.branch_input_dim :]
            branch_features = self.branch_net(branch_x)
            trunk_features = self.trunk_net(trunk_x)
            outputs = torch.sum(branch_features * trunk_features, dim=1, keepdim=True) + self.bias
            return outputs

        if branch_inputs.dim() != 2 or trunk_inputs.dim() != 2:
            raise ValueError("branch_inputs and trunk_inputs must both be rank-2 tensors.")

        branch_features = self.branch_net(branch_inputs)
        trunk_features = self.trunk_net(trunk_inputs)
        outputs = branch_features @ trunk_features.T
        outputs = outputs.unsqueeze(-1) + self.bias
        return outputs

