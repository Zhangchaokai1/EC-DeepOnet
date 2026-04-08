from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    train_temp_losses: list[float] = field(default_factory=list)
    val_temp_losses: list[float] = field(default_factory=list)
    train_energy_losses: list[float] = field(default_factory=list)
    val_energy_losses: list[float] = field(default_factory=list)
    train_smooth_losses: list[float] = field(default_factory=list)
    val_smooth_losses: list[float] = field(default_factory=list)


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def _select_monitor_value(
    metric_name: str,
    *,
    total: float,
    temp: float,
    energy: float,
    smooth: float,
) -> float:
    metric_map = {
        "total": total,
        "temp": temp,
        "energy": energy,
        "smooth": smooth,
    }
    if metric_name not in metric_map:
        raise ValueError(f"Unsupported monitor metric: {metric_name}")
    return float(metric_map[metric_name])


def train_pointwise_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    criterion: nn.Module | None = None,
) -> TrainingHistory:
    criterion = criterion or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory()

    for _ in range(epochs):
        model.train()
        train_batch_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(float(loss.item()))

        model.eval()
        val_batch_losses: list[float] = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_batch_losses.append(float(loss.item()))

        history.train_losses.append(_mean_or_nan(train_batch_losses))
        history.val_losses.append(_mean_or_nan(val_batch_losses))

    return history


def train_casewise_deeponet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trunk_inputs: torch.Tensor,
    device: torch.device,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    criterion: nn.Module | None = None,
) -> TrainingHistory:
    criterion = criterion or nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = TrainingHistory()
    trunk_inputs = trunk_inputs.to(device)

    for _ in range(epochs):
        model.train()
        train_batch_losses: list[float] = []
        for branch_inputs, targets, _ in train_loader:
            branch_inputs = branch_inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(branch_inputs, trunk_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_batch_losses.append(float(loss.item()))

        model.eval()
        val_batch_losses: list[float] = []
        with torch.no_grad():
            for branch_inputs, targets, _ in val_loader:
                branch_inputs = branch_inputs.to(device)
                targets = targets.to(device)
                outputs = model(branch_inputs, trunk_inputs)
                loss = criterion(outputs, targets)
                val_batch_losses.append(float(loss.item()))

        history.train_losses.append(_mean_or_nan(train_batch_losses))
        history.val_losses.append(_mean_or_nan(val_batch_losses))

    return history


def train_casewise_model_with_custom_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trunk_inputs: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    epochs: int = 500,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    scheduler_patience: int = 20,
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 60,
    min_delta: float = 1e-5,
    monitor_metric: str = "total",
) -> tuple[TrainingHistory, dict[str, torch.Tensor] | None]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = loss_fn.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
    )
    history = TrainingHistory()
    trunk_inputs = trunk_inputs.to(device)

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_monitor_value = float("inf")
    patience_counter = 0

    for _ in range(epochs):
        model.train()
        train_total_losses: list[float] = []
        train_temp_losses: list[float] = []
        train_energy_losses: list[float] = []
        train_smooth_losses: list[float] = []
        for branch_inputs, targets, true_energies in train_loader:
            branch_inputs = branch_inputs.to(device)
            targets = targets.to(device)
            true_energies = true_energies.to(device)

            optimizer.zero_grad()
            outputs = model(branch_inputs, trunk_inputs)
            breakdown = loss_fn(outputs, targets, true_energies)
            breakdown.total.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_total_losses.append(float(breakdown.total.item()))
            train_temp_losses.append(float(breakdown.temp.item()))
            train_energy_losses.append(float(breakdown.energy.item()))
            train_smooth_losses.append(float(breakdown.smooth.item()))

        model.eval()
        val_total_losses: list[float] = []
        val_temp_losses: list[float] = []
        val_energy_losses: list[float] = []
        val_smooth_losses: list[float] = []
        with torch.no_grad():
            for branch_inputs, targets, true_energies in val_loader:
                branch_inputs = branch_inputs.to(device)
                targets = targets.to(device)
                true_energies = true_energies.to(device)
                outputs = model(branch_inputs, trunk_inputs)
                breakdown = loss_fn(outputs, targets, true_energies)
                val_total_losses.append(float(breakdown.total.item()))
                val_temp_losses.append(float(breakdown.temp.item()))
                val_energy_losses.append(float(breakdown.energy.item()))
                val_smooth_losses.append(float(breakdown.smooth.item()))

        train_total = _mean_or_nan(train_total_losses)
        val_total = _mean_or_nan(val_total_losses)
        history.train_losses.append(train_total)
        history.val_losses.append(val_total)
        history.train_temp_losses.append(_mean_or_nan(train_temp_losses))
        history.val_temp_losses.append(_mean_or_nan(val_temp_losses))
        history.train_energy_losses.append(_mean_or_nan(train_energy_losses))
        history.val_energy_losses.append(_mean_or_nan(val_energy_losses))
        history.train_smooth_losses.append(_mean_or_nan(train_smooth_losses))
        history.val_smooth_losses.append(_mean_or_nan(val_smooth_losses))

        monitor_value = _select_monitor_value(
            monitor_metric,
            total=val_total,
            temp=history.val_temp_losses[-1],
            energy=history.val_energy_losses[-1],
            smooth=history.val_smooth_losses[-1],
        )

        scheduler.step(monitor_value)

        if monitor_value + min_delta < best_monitor_value:
            best_monitor_value = monitor_value
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return history, best_state_dict
