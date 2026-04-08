from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import (
    DEFAULT_CSV_PATH,
    DEFAULT_TIME_SERIES_DIR,
    STATUS_COLUMN,
    SUCCESS_STATUS,
    TIME_SERIES_FILE_COLUMN,
    TIME_SERIES_TEMPERATURE_KEY,
    TIME_SERIES_TIME_KEY,
)


def load_main_dataframe(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    success_only: bool = True,
    reset_index: bool = True,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if success_only and STATUS_COLUMN in df.columns:
        df = df[df[STATUS_COLUMN] == SUCCESS_STATUS].copy()
    if reset_index:
        df = df.reset_index(drop=True)
    return df


def resolve_timeseries_path(
    time_series_file: Path | str,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
) -> Path:
    candidate = Path(time_series_file)
    if candidate.is_absolute():
        return candidate
    return Path(ts_dir) / candidate


def load_timeseries_npz(
    time_series_file: Path | str,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    npz_path = resolve_timeseries_path(time_series_file, ts_dir=ts_dir)
    with np.load(npz_path, allow_pickle=True) as data:
        times = np.asarray(data[TIME_SERIES_TIME_KEY]).reshape(-1)
        temperatures = np.asarray(data[TIME_SERIES_TEMPERATURE_KEY]).reshape(-1)
    return times.astype(np.float64), temperatures.astype(np.float64)


def iter_case_series(
    df: pd.DataFrame,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
) -> Iterable[tuple[int, pd.Series, np.ndarray, np.ndarray]]:
    for idx, row in df.iterrows():
        times, temperatures = load_timeseries_npz(row[TIME_SERIES_FILE_COLUMN], ts_dir=ts_dir)
        yield idx, row, times, temperatures


def collect_missing_timeseries_files(
    df: pd.DataFrame,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
) -> list[Path]:
    missing: list[Path] = []
    for _, row in df.iterrows():
        npz_path = resolve_timeseries_path(row[TIME_SERIES_FILE_COLUMN], ts_dir=ts_dir)
        if not npz_path.exists():
            missing.append(npz_path)
    return missing

