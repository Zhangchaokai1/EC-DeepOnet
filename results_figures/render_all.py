import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from results_figures.common import (
    create_fig01_correlation_heatmap,
    create_fig02_model_comparison,
    create_fig03_energy_parity,
    create_fig03_energy_parity_train,
    create_fig04_ablation,
    create_fig05_expert_loss_band,
    create_fig06_case_overview,
    create_fig07_optimization_landscape,
    create_fig08_optimization_benefit,
    create_fig09_expert_boxplot,
    create_fig10_parameter_shift,
    create_fig11_training_loss,
    create_fig12_temperature_parity,
    create_fig12_temperature_parity_train,
    create_fig13_parameter_response_network,
    save_results_tables,
)


if __name__ == "__main__":
    create_fig01_correlation_heatmap()
    create_fig02_model_comparison()
    create_fig03_energy_parity()
    create_fig03_energy_parity_train()
    create_fig04_ablation()
    create_fig05_expert_loss_band()
    create_fig06_case_overview()
    create_fig07_optimization_landscape()
    create_fig08_optimization_benefit()
    create_fig09_expert_boxplot()
    create_fig10_parameter_shift()
    create_fig11_training_loss()
    create_fig12_temperature_parity()
    create_fig12_temperature_parity_train()
    create_fig13_parameter_response_network()
    save_results_tables()
    print("已全部重绘")
