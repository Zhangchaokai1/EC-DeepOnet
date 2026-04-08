import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures.common import (
    create_fig01_correlation_heatmap,
    create_fig01_feature_temperature_energy_boxplot,
)


if __name__ == "__main__":
    create_fig01_correlation_heatmap()
    create_fig01_feature_temperature_energy_boxplot()
