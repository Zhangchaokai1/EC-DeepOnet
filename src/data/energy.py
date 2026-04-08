from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .io import iter_case_series
from .schema import (
    CP_WATER_J_PER_KG_K,
    DEFAULT_INLET_TEMPERATURE_C,
    DEFAULT_INTEGRATION_POINTS,
    DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    DEFAULT_T_END_HOURS,
    DEFAULT_T_START_HOURS,
    DEFAULT_TIME_SERIES_DIR,
    TARGET_ENERGY_COLUMN,
)


@dataclass(frozen=True)
class EnergyCurve:
    times_hours: np.ndarray
    temperatures_c: np.ndarray
    heat_rate_watts: np.ndarray
    cumulative_energy_mj: np.ndarray


def build_standard_time_grid(
    t_start_hours: float = DEFAULT_T_START_HOURS,
    t_end_hours: float = DEFAULT_T_END_HOURS,
    n_points: int = DEFAULT_INTEGRATION_POINTS,
) -> np.ndarray:
    if n_points < 2:
        raise ValueError("n_points must be at least 2.")
    if t_end_hours <= t_start_hours:
        raise ValueError("t_end_hours must be larger than t_start_hours.")
    return np.linspace(t_start_hours, t_end_hours, n_points, dtype=np.float64)


def _sort_series(times_hours: np.ndarray, temperatures_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if times_hours.shape != temperatures_c.shape:
        raise ValueError("times_hours and temperatures_c must have the same shape.")
    order = np.argsort(times_hours)
    return times_hours[order], temperatures_c[order]


def interpolate_temperature(
    times_hours: np.ndarray,
    temperatures_c: np.ndarray,
    target_times_hours: np.ndarray,
) -> np.ndarray:
    times_hours, temperatures_c = _sort_series(
        np.asarray(times_hours, dtype=np.float64).reshape(-1),
        np.asarray(temperatures_c, dtype=np.float64).reshape(-1),
    )
    if len(times_hours) < 2:
        raise ValueError("At least two time points are required for interpolation.")
    if target_times_hours.min() < times_hours.min() or target_times_hours.max() > times_hours.max():
        raise ValueError("target_times_hours must lie within the source time range.")
    return np.interp(target_times_hours, times_hours, temperatures_c)


def compute_heat_rate_watts(
    temperatures_c: np.ndarray,
    inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
    mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
) -> np.ndarray:
    temperatures_c = np.asarray(temperatures_c, dtype=np.float64)
    return mass_flow_rate_kg_per_s * cp_water_j_per_kg_k * (inlet_temperature_c - temperatures_c)


def compute_cumulative_energy_mj(
    times_hours: np.ndarray,
    temperatures_c: np.ndarray,
    inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
    mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
) -> np.ndarray:
    times_hours, temperatures_c = _sort_series(
        np.asarray(times_hours, dtype=np.float64).reshape(-1),
        np.asarray(temperatures_c, dtype=np.float64).reshape(-1),
    )
    heat_rate_watts = compute_heat_rate_watts(
        temperatures_c,
        inlet_temperature_c=inlet_temperature_c,
        mass_flow_rate_kg_per_s=mass_flow_rate_kg_per_s,
        cp_water_j_per_kg_k=cp_water_j_per_kg_k,
    )
    times_seconds = times_hours * 3600.0
    delta_t = np.diff(times_seconds)
    trapezoids = 0.5 * (heat_rate_watts[1:] + heat_rate_watts[:-1]) * delta_t
    cumulative_joules = np.concatenate(([0.0], np.cumsum(trapezoids)))
    return cumulative_joules / 1e6


def compute_standardized_energy_curve(
    times_hours: np.ndarray,
    temperatures_c: np.ndarray,
    t_start_hours: float = DEFAULT_T_START_HOURS,
    t_end_hours: float = DEFAULT_T_END_HOURS,
    n_points: int = DEFAULT_INTEGRATION_POINTS,
    inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
    mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
) -> EnergyCurve:
    target_grid = build_standard_time_grid(
        t_start_hours=t_start_hours,
        t_end_hours=t_end_hours,
        n_points=n_points,
    )
    standardized_temperatures = interpolate_temperature(
        times_hours,
        temperatures_c,
        target_grid,
    )
    heat_rate_watts = compute_heat_rate_watts(
        standardized_temperatures,
        inlet_temperature_c=inlet_temperature_c,
        mass_flow_rate_kg_per_s=mass_flow_rate_kg_per_s,
        cp_water_j_per_kg_k=cp_water_j_per_kg_k,
    )
    cumulative_energy_mj = compute_cumulative_energy_mj(
        target_grid,
        standardized_temperatures,
        inlet_temperature_c=inlet_temperature_c,
        mass_flow_rate_kg_per_s=mass_flow_rate_kg_per_s,
        cp_water_j_per_kg_k=cp_water_j_per_kg_k,
    )
    return EnergyCurve(
        times_hours=target_grid,
        temperatures_c=standardized_temperatures,
        heat_rate_watts=heat_rate_watts,
        cumulative_energy_mj=cumulative_energy_mj,
    )


def compute_standardized_energy_mj(
    times_hours: np.ndarray,
    temperatures_c: np.ndarray,
    t_start_hours: float = DEFAULT_T_START_HOURS,
    t_end_hours: float = DEFAULT_T_END_HOURS,
    n_points: int = DEFAULT_INTEGRATION_POINTS,
    inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
    mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
) -> float:
    curve = compute_standardized_energy_curve(
        times_hours=times_hours,
        temperatures_c=temperatures_c,
        t_start_hours=t_start_hours,
        t_end_hours=t_end_hours,
        n_points=n_points,
        inlet_temperature_c=inlet_temperature_c,
        mass_flow_rate_kg_per_s=mass_flow_rate_kg_per_s,
        cp_water_j_per_kg_k=cp_water_j_per_kg_k,
    )
    return float(curve.cumulative_energy_mj[-1])


def add_energy_column(
    df: pd.DataFrame,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
    output_column: str = TARGET_ENERGY_COLUMN,
    t_start_hours: float = DEFAULT_T_START_HOURS,
    t_end_hours: float = DEFAULT_T_END_HOURS,
    n_points: int = DEFAULT_INTEGRATION_POINTS,
    inlet_temperature_c: float = DEFAULT_INLET_TEMPERATURE_C,
    mass_flow_rate_kg_per_s: float = DEFAULT_MASS_FLOW_RATE_KG_PER_S,
    cp_water_j_per_kg_k: float = CP_WATER_J_PER_KG_K,
) -> pd.DataFrame:
    df = df.copy()
    energies: list[float] = []
    for _, _, times_hours, temperatures_c in iter_case_series(df, ts_dir=ts_dir):
        energies.append(
            compute_standardized_energy_mj(
                times_hours=times_hours,
                temperatures_c=temperatures_c,
                t_start_hours=t_start_hours,
                t_end_hours=t_end_hours,
                n_points=n_points,
                inlet_temperature_c=inlet_temperature_c,
                mass_flow_rate_kg_per_s=mass_flow_rate_kg_per_s,
                cp_water_j_per_kg_k=cp_water_j_per_kg_k,
            )
        )
    df[output_column] = energies
    return df

