from __future__ import annotations

from pathlib import Path

import pandas as pd

from .energy import add_energy_column
from .io import load_main_dataframe
from .schema import DEFAULT_CSV_PATH, DEFAULT_TIME_SERIES_DIR, TARGET_ENERGY_COLUMN


def build_energy_augmented_dataframe(
    csv_path: Path | str = DEFAULT_CSV_PATH,
    ts_dir: Path | str = DEFAULT_TIME_SERIES_DIR,
    success_only: bool = True,
) -> pd.DataFrame:
    df = load_main_dataframe(csv_path=csv_path, success_only=success_only, reset_index=True)
    return add_energy_column(df, ts_dir=ts_dir, output_column=TARGET_ENERGY_COLUMN)

