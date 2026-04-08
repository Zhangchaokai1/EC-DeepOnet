from __future__ import annotations

import argparse
from pathlib import Path

from src.data.prepare import build_energy_augmented_dataframe
from src.data.schema import DEFAULT_CSV_PATH, DEFAULT_TIME_SERIES_DIR, TARGET_ENERGY_COLUMN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a canonical CSV with recomputed E_{1-100h} values."
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the original simulation summary CSV.",
    )
    parser.add_argument(
        "--ts-dir",
        type=Path,
        default=DEFAULT_TIME_SERIES_DIR,
        help="Directory containing exported time-series NPZ files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("processed_energy_dataset.csv"),
        help="Destination CSV path for the recomputed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = build_energy_augmented_dataframe(csv_path=args.csv_path, ts_dir=args.ts_dir, success_only=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False, encoding="utf-8")

    print(f"Saved processed dataset to: {args.output_path.resolve()}")
    print(f"Rows: {len(df)}")
    print(
        f"{TARGET_ENERGY_COLUMN} range (MJ): "
        f"{df[TARGET_ENERGY_COLUMN].min():.6f} - {df[TARGET_ENERGY_COLUMN].max():.6f}"
    )


if __name__ == "__main__":
    main()

