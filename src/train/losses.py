from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.data.schema import (
    CP_WATER_J_PER_KG_K,
    DEFAULT_INLET_TEMPERATURE_C,
    DEFAULT_MASS_FLOW_RATE_KG_PER_S,
)


@dataclass
class LossBreakdown:
    total: torch.Tensor
    temp: torch.Tensor
    energy: torch.Tensor
    smooth: torch.Tensor


class EnergyConsistencyLoss(nn.Module):
    def __init__(
        self,
        trunk_times_hours: torch.Tensor,
        temperature_mean: float,
        temperature_scale: float,
        temperature_weight: float = 1.0,
        energy_weight: float = 0.1,
        smoothness_weight: float = 0.0,
        inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
        mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
        cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
        temperature_loss: str = "mse",
        energy_loss: str = "mse",
        energy_scale_mj: float | None = None,
        relative_energy_epsilon_mj: float = 1.0,
    ) -> None:
        super().__init__()
        trunk_times_hours = trunk_times_hours.reshape(-1).float()
        if len(trunk_times_hours) < 2:
            raise ValueError("At least two trunk time points are required.")
        dt_seconds = (trunk_times_hours[1:] - trunk_times_hours[:-1]) * 3600.0
        self.register_buffer("dt_seconds", dt_seconds)
        self.temperature_mean = float(temperature_mean)
        self.temperature_scale = float(temperature_scale)
        self.temperature_weight = float(temperature_weight)
        self.energy_weight = float(energy_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.inlet_temperature_c = float(inlet_temperature_c)
        self.mass_flow_rate_kg_per_s = float(mass_flow_rate_kg_per_s)
        self.cp_water_j_per_kg_k = float(cp_water_j_per_kg_k)
        self.energy_loss = energy_loss
        self.energy_scale_mj = max(float(energy_scale_mj), 1e-6) if energy_scale_mj is not None else 1.0
        self.relative_energy_epsilon_mj = float(relative_energy_epsilon_mj)
        if temperature_loss == "huber":
            self.temp_loss_fn = nn.SmoothL1Loss()
        elif temperature_loss == "mse":
            self.temp_loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported temperature_loss: {temperature_loss}")

    def _inverse_temperature(self, scaled_temperatures: torch.Tensor) -> torch.Tensor:
        return scaled_temperatures * self.temperature_scale + self.temperature_mean

    def _energy_from_temperature(self, temperatures_c: torch.Tensor) -> torch.Tensor:
        heat_rate = (
            self.mass_flow_rate_kg_per_s
            * self.cp_water_j_per_kg_k
            * (self.inlet_temperature_c - temperatures_c)
        )
        trapezoids = 0.5 * (heat_rate[:, 1:, :] + heat_rate[:, :-1, :]) * self.dt_seconds.view(1, -1, 1)
        energy_joules = trapezoids.sum(dim=1)
        return energy_joules / 1e6

    def _smoothness_penalty(self, temperatures_c: torch.Tensor) -> torch.Tensor:
        if temperatures_c.shape[1] < 3:
            return torch.zeros((), device=temperatures_c.device)
        second_diff = temperatures_c[:, 2:, :] - 2.0 * temperatures_c[:, 1:-1, :] + temperatures_c[:, :-2, :]
        return (second_diff**2).mean()

    def _energy_loss(self, pred_energy_mj: torch.Tensor, true_energy_mj: torch.Tensor) -> torch.Tensor:
        if self.energy_loss == "mse":
            return F.mse_loss(pred_energy_mj, true_energy_mj)
        if self.energy_loss == "huber":
            return F.smooth_l1_loss(pred_energy_mj, true_energy_mj)
        if self.energy_loss == "scaled_mse":
            residual = (pred_energy_mj - true_energy_mj) / self.energy_scale_mj
            return torch.mean(residual**2)
        if self.energy_loss == "relative_mse":
            residual = (pred_energy_mj - true_energy_mj) / (true_energy_mj.abs() + self.relative_energy_epsilon_mj)
            return torch.mean(residual**2)
        raise ValueError(f"Unsupported energy_loss: {self.energy_loss}")

    def forward(
        self,
        pred_scaled: torch.Tensor,
        target_scaled: torch.Tensor,
        true_energy_mj: torch.Tensor,
    ) -> LossBreakdown:
        temp_loss = self.temp_loss_fn(pred_scaled, target_scaled)
        pred_temp_c = self._inverse_temperature(pred_scaled)
        pred_energy_mj = self._energy_from_temperature(pred_temp_c)
        energy_loss = self._energy_loss(pred_energy_mj, true_energy_mj)
        smoothness_loss = self._smoothness_penalty(pred_temp_c)
        total_loss = (
            self.temperature_weight * temp_loss
            + self.energy_weight * energy_loss
            + self.smoothness_weight * smoothness_loss
        )
        return LossBreakdown(
            total=total_loss,
            temp=temp_loss,
            energy=energy_loss,
            smooth=smoothness_loss,
        )
