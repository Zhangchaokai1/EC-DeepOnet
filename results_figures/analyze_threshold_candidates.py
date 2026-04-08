import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures.common import (
    create_threshold_candidate_violin_plot,
    summarize_candidates_above_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize candidate_pool samples whose pred_energy_mean_mj is above a threshold."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optimization run directory, e.g. outputs/optimization_case/20260314_123439",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=426.90,
        help="Filter threshold for pred_energy_mean_mj",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="table_fig07_candidates_above_threshold.csv",
        help="CSV filename saved under outputs/results_figures_v1/",
    )
    parser.add_argument(
        "--plot-name",
        type=str,
        default="fig07_candidates_above_threshold_violin.png",
        help="Plot filename saved under outputs/results_figures_v1/",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = summarize_candidates_above_threshold(
        threshold=args.threshold,
        optimization_dir=args.run_dir,
        output_filename=args.output_name,
    )

    print(f"optimization_dir: {result['optimization_dir']}")
    print(f"candidate_path: {result['candidate_path']}")
    print(f"threshold: {result['threshold']:.2f}")
    print(f"total_count: {result['total_count']}")
    print(f"selected_count: {result['selected_count']}")
    print(f"selected_ratio: {result['selected_ratio']:.6f}")
    print()
    print(result["summary_df"].to_string(index=False))
    plot_path = create_threshold_candidate_violin_plot(
        threshold=args.threshold,
        optimization_dir=args.run_dir,
        filename=args.plot_name,
    )
    svg_path = plot_path.with_suffix(".svg")
    print()
    print(f"plot_path: {plot_path}")
    if svg_path.exists():
        print(f"svg_path: {svg_path}")
