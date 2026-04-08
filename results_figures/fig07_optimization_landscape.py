import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures.common import create_fig07_optimization_landscape


if __name__ == "__main__":
    create_fig07_optimization_landscape()
