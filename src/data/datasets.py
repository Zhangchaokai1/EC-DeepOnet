from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from .energy import compute_standardized_energy_curve
from .io import iter_case_series
from .schema import (
    DEFAULT_INTEGRATION_POINTS,
    DEFAULT_T_END_HOURS,
    DEFAULT_T_START_HOURS,
    DEFAULT_TIME_SERIES_DIR,
    INPUT_COLUMNS,
)


@dataclass(frozen=True)
class CaseDatasetArrays:
    branch_inputs: np.ndarray
    trunk_inputs: np.ndarray
    temperatures: np.ndarray
    energies_mj: np.ndarray
    case_ids: np.ndarray


@dataclass(frozen=True)
class PointDatasetArrays:
    inputs: np.ndarray
    temperatures: np.ndarray
    case_ids: np.ndarray


@dataclass(frozen=True)
class DatasetScalers:
    branch_scaler: StandardScaler
    trunk_scaler: StandardScaler
    temperature_scaler: StandardScaler


class PointwiseTemperatureDataset(Dataset):
    def __init__(self, inputs: np.ndarray, temperatures: np.ndarray):
        self.inputs = torch.from_numpy(inputs.astype(np.float32))
        self.temperatures = torch.from_numpy(temperatures.astype(np.float32))

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.temperatures[idx]


class CasewiseBranchDataset(Dataset):
    def __init__(self, branch_inputs: np.ndarray, temperatures: np.ndarray, energies_mj: np.ndarray):
        self.branch_inputs = torch.from_numpy(branch_inputs.astype(np.float32))
        self.temperatures = torch.from_numpy(temperatures.astype(np.float32))
        self.energies_mj = torch.from_numpy(energies_mj.astype(np.float32))

    def __len__(self) -> int:
        return len(self.branch_inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.branch_inputs[idx], self.temperatures[idx], self.energies_mj[idx]


def split_case_dataframe(
    df: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 123,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_val = train_test_split(df, test_size=test_size, random_state=random_state)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def build_case_dataset(
    df: pd.DataFrame,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
    input_columns: list[str] = INPUT_COLUMNS,
    t_start_hours: float = DEFAULT_T_START_HOURS,
    t_end_hours: float = DEFAULT_T_END_HOURS,
    n_points: int = DEFAULT_INTEGRATION_POINTS,
    id_column: str = "ID",
) -> CaseDatasetArrays:
    branch_rows: list[np.ndarray] = []
    temperature_rows: list[np.ndarray] = []
    energy_rows: list[float] = []
    case_ids: list[int] = []
    trunk_inputs: np.ndarray | None = None

    for _, row, times_hours, temperatures_c in iter_case_series(df, ts_dir=ts_dir):
        curve = compute_standardized_energy_curve(
            times_hours=times_hours,
            temperatures_c=temperatures_c,
            t_start_hours=t_start_hours,
            t_end_hours=t_end_hours,
            n_points=n_points,
        )
        branch_rows.append(row[input_columns].to_numpy(dtype=np.float64))
        temperature_rows.append(curve.temperatures_c.reshape(-1, 1))
        energy_rows.append(float(curve.cumulative_energy_mj[-1]))
        case_ids.append(int(row[id_column]) if id_column in row else len(case_ids))
        if trunk_inputs is None:
            trunk_inputs = curve.times_hours.reshape(-1, 1)

    if trunk_inputs is None:
        raise ValueError("No valid cases were found to build the case dataset.")

    return CaseDatasetArrays(
        branch_inputs=np.asarray(branch_rows, dtype=np.float64),
        trunk_inputs=trunk_inputs.astype(np.float64),
        temperatures=np.asarray(temperature_rows, dtype=np.float64),
        energies_mj=np.asarray(energy_rows, dtype=np.float64).reshape(-1, 1),
        case_ids=np.asarray(case_ids, dtype=np.int64),
    )


def flatten_case_dataset(case_dataset: CaseDatasetArrays) -> PointDatasetArrays:
    n_cases, n_points, _ = case_dataset.temperatures.shape
    repeated_branch = np.repeat(case_dataset.branch_inputs, n_points, axis=0)
    tiled_trunk = np.tile(case_dataset.trunk_inputs, (n_cases, 1))
    point_inputs = np.concatenate([repeated_branch, tiled_trunk], axis=1)
    temperatures = case_dataset.temperatures.reshape(n_cases * n_points, 1)
    case_ids = np.repeat(case_dataset.case_ids, n_points)
    return PointDatasetArrays(
        inputs=point_inputs.astype(np.float64),
        temperatures=temperatures.astype(np.float64),
        case_ids=case_ids.astype(np.int64),
    )


def fit_case_scalers(train_case_dataset: CaseDatasetArrays) -> DatasetScalers:
    branch_scaler = StandardScaler()
    trunk_scaler = StandardScaler()
    temperature_scaler = StandardScaler()

    branch_scaler.fit(train_case_dataset.branch_inputs)
    trunk_scaler.fit(train_case_dataset.trunk_inputs)
    temperature_scaler.fit(train_case_dataset.temperatures.reshape(-1, 1))

    return DatasetScalers(
        branch_scaler=branch_scaler,
        trunk_scaler=trunk_scaler,
        temperature_scaler=temperature_scaler,
    )


def transform_case_dataset(case_dataset: CaseDatasetArrays, scalers: DatasetScalers) -> CaseDatasetArrays:
    branch_scaled = scalers.branch_scaler.transform(case_dataset.branch_inputs)
    trunk_scaled = scalers.trunk_scaler.transform(case_dataset.trunk_inputs)
    temperatures_scaled = scalers.temperature_scaler.transform(
        case_dataset.temperatures.reshape(-1, 1)
    ).reshape(case_dataset.temperatures.shape)

    return CaseDatasetArrays(
        branch_inputs=branch_scaled.astype(np.float64),
        trunk_inputs=trunk_scaled.astype(np.float64),
        temperatures=temperatures_scaled.astype(np.float64),
        energies_mj=case_dataset.energies_mj.astype(np.float64),
        case_ids=case_dataset.case_ids,
    )


def build_point_dataset_from_cases(
    case_dataset: CaseDatasetArrays,
    scalers: DatasetScalers,
) -> PointDatasetArrays:
    point_dataset = flatten_case_dataset(case_dataset)
    branch_scaled = scalers.branch_scaler.transform(point_dataset.inputs[:, : len(INPUT_COLUMNS)])
    trunk_scaled = scalers.trunk_scaler.transform(point_dataset.inputs[:, len(INPUT_COLUMNS) :])
    point_inputs = np.concatenate([branch_scaled, trunk_scaled], axis=1)
    temperatures_scaled = scalers.temperature_scaler.transform(point_dataset.temperatures)

    return PointDatasetArrays(
        inputs=point_inputs.astype(np.float64),
        temperatures=temperatures_scaled.astype(np.float64),
        case_ids=point_dataset.case_ids,
    )

