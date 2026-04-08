import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures.common import (
    create_fig12_temperature_parity,
    create_fig12_temperature_parity_train,
)


if __name__ == "__main__":
    create_fig12_temperature_parity()
    create_fig12_temperature_parity_train()
