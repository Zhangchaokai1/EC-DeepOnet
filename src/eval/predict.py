from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

from src.data.datasets import CaseDatasetArrays, DatasetScalers, PointDatasetArrays
from src.data.energy import compute_standardized_energy_mj
from src.eval.metrics import regression_metrics


def inverse_transform_temperatures(
    temperatures_scaled: np.ndarray,
    scalers: DatasetScalers,
) -> np.ndarray:
    original_shape = temperatures_scaled.shape
    restored = scalers.temperature_scaler.inverse_transform(
        np.asarray(temperatures_scaled).reshape(-1, 1)
    )
    return restored.reshape(original_shape)


def predict_pointwise_temperatures(
    model: torch.nn.Module,
    point_dataset: PointDatasetArrays,
    scalers: DatasetScalers,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(point_dataset.inputs), batch_size):
            end = min(start + batch_size, len(point_dataset.inputs))
            batch_inputs = torch.from_numpy(point_dataset.inputs[start:end]).float().to(device)
            batch_outputs = model(batch_inputs).cpu().numpy()
            predictions.append(batch_outputs)
    predictions_scaled = np.concatenate(predictions, axis=0)
    return inverse_transform_temperatures(predictions_scaled, scalers).reshape(-1, 1)


def predict_casewise_temperatures(
    model: torch.nn.Module,
    case_dataset: CaseDatasetArrays,
    scalers: DatasetScalers,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    trunk_inputs = torch.from_numpy(case_dataset.trunk_inputs).float().to(device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(case_dataset.branch_inputs), batch_size):
            end = min(start + batch_size, len(case_dataset.branch_inputs))
            branch_inputs = torch.from_numpy(case_dataset.branch_inputs[start:end]).float().to(device)
            batch_outputs = model(branch_inputs, trunk_inputs).cpu().numpy()
            predictions.append(batch_outputs)
    predictions_scaled = np.concatenate(predictions, axis=0)
    return inverse_transform_temperatures(predictions_scaled, scalers)


def predict_casewise_temperatures_from_point_model(
    model: torch.nn.Module,
    case_dataset_scaled: CaseDatasetArrays,
    scalers: DatasetScalers,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    n_cases = len(case_dataset_scaled.branch_inputs)
    n_points = len(case_dataset_scaled.trunk_inputs)
    repeated_branch = np.repeat(case_dataset_scaled.branch_inputs, n_points, axis=0)
    tiled_trunk = np.tile(case_dataset_scaled.trunk_inputs, (n_cases, 1))
    point_inputs = np.concatenate([repeated_branch, tiled_trunk], axis=1)

    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(point_inputs), batch_size):
            end = min(start + batch_size, len(point_inputs))
            batch_inputs = torch.from_numpy(point_inputs[start:end]).float().to(device)
            batch_outputs = model(batch_inputs).cpu().numpy()
            predictions.append(batch_outputs)
    predictions_scaled = np.concatenate(predictions, axis=0).reshape(n_cases, n_points, 1)
    return inverse_transform_temperatures(predictions_scaled, scalers)


def evaluate_pointwise_predictions(
    true_temperatures_c: np.ndarray,
    pred_temperatures_c: np.ndarray,
) -> dict[str, float]:
    return regression_metrics(true_temperatures_c, pred_temperatures_c)


def evaluate_case_energy_predictions(
    case_dataset: CaseDatasetArrays,
    pred_temperatures_c: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    predicted_energies: list[float] = []
    trunk_times = case_dataset.trunk_inputs.reshape(-1)
    for curve in pred_temperatures_c:
        predicted_energies.append(
            compute_standardized_energy_mj(
                times_hours=trunk_times,
                temperatures_c=np.asarray(curve).reshape(-1),
                t_start_hours=float(trunk_times[0]),
                t_end_hours=float(trunk_times[-1]),
                n_points=len(trunk_times),
            )
        )
    predicted_energies_array = np.asarray(predicted_energies, dtype=np.float64).reshape(-1, 1)
    metrics = regression_metrics(case_dataset.energies_mj, predicted_energies_array)
    return metrics, predicted_energies_array


def ranking_metrics(
    true_values: np.ndarray,
    pred_values: np.ndarray,
) -> dict[str, float]:
    true_values = np.asarray(true_values).reshape(-1)
    pred_values = np.asarray(pred_values).reshape(-1)
    spearman = spearmanr(true_values, pred_values)
    kendall = kendalltau(true_values, pred_values)
    return {
        "SpearmanR": float(spearman.statistic),
        "SpearmanP": float(spearman.pvalue),
        "KendallTau": float(kendall.statistic),
        "KendallP": float(kendall.pvalue),
    }
