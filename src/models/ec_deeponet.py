from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .mlp import build_feedforward


class FourierFeatures(nn.Module):
    def __init__(self, input_dim: int = 1, num_frequencies: int = 16, max_frequency: float = 8.0) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be non-negative.")

        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.register_buffer(
            "frequencies",
            torch.linspace(1.0, max_frequency, num_frequencies, dtype=torch.float32).reshape(1, 1, -1),
        )

    @property
    def output_dim(self) -> int:
        return self.input_dim * (2 * self.num_frequencies + 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.num_frequencies == 0:
            return inputs
        expanded = inputs.unsqueeze(-1) * self.frequencies
        sin_features = torch.sin(2.0 * math.pi * expanded)
        cos_features = torch.cos(2.0 * math.pi * expanded)
        combined = torch.cat([inputs.unsqueeze(-1), sin_features, cos_features], dim=-1)
        return combined.reshape(inputs.shape[0], -1)


class ECDeepONet(nn.Module):
    def __init__(
        self,
        branch_input_dim: int = 10,
        trunk_input_dim: int = 1,
        latent_dim: int = 96,
        branch_hidden_dims: Iterable[int] = (192, 192),
        trunk_hidden_dims: Iterable[int] = (128, 128),
        refine_hidden_dims: Iterable[int] = (128, 128),
        num_frequencies: int = 16,
        max_frequency: float = 8.0,
        activation_cls: type[nn.Module] = nn.GELU,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        self.branch_input_dim = branch_input_dim
        self.trunk_input_dim = trunk_input_dim
        self.latent_dim = latent_dim

        self.time_encoder = FourierFeatures(
            input_dim=trunk_input_dim,
            num_frequencies=num_frequencies,
            max_frequency=max_frequency,
        )

        self.branch_pre = build_feedforward(
            input_dim=branch_input_dim,
            hidden_dims=branch_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.trunk_pre = build_feedforward(
            input_dim=self.time_encoder.output_dim,
            hidden_dims=trunk_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.branch_refine = build_feedforward(
            input_dim=latent_dim,
            hidden_dims=refine_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.trunk_refine = build_feedforward(
            input_dim=latent_dim,
            hidden_dims=refine_hidden_dims,
            output_dim=latent_dim,
            activation_cls=activation_cls,
        )
        self.context_gate = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            activation_cls(),
            nn.Linear(latent_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.mix_norm = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()
        self.branch_norm = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()
        self.trunk_norm = nn.LayerNorm(latent_dim) if use_layernorm else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_inputs: torch.Tensor, trunk_inputs: torch.Tensor) -> torch.Tensor:
        if branch_inputs.dim() != 2 or trunk_inputs.dim() != 2:
            raise ValueError("ECDeepONet expects branch_inputs and trunk_inputs to be rank-2 tensors.")

        branch_features = self.branch_norm(self.branch_pre(branch_inputs))  # [B, H]
        trunk_features = self.trunk_norm(self.trunk_pre(self.time_encoder(trunk_inputs)))  # [T, H]

        base_mix = branch_features.unsqueeze(1) * trunk_features.unsqueeze(0)  # [B, T, H]
        base_mix = self.mix_norm(base_mix)
        context = base_mix.mean(dim=1)  # [B, H]

        refined_branch = self.branch_refine(context)  # [B, H]
        gate = self.context_gate(context).unsqueeze(1)  # [B, 1, H]

        refined_trunk = self.trunk_refine(base_mix.reshape(-1, self.latent_dim)).reshape(base_mix.shape)
        refined_trunk = self.dropout(refined_trunk)

        base_output = base_mix.sum(dim=-1, keepdim=True)
        refined_output = (refined_branch.unsqueeze(1) * refined_trunk * gate).sum(dim=-1, keepdim=True)
        return base_output + refined_output + self.bias

