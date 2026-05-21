# Surrogate Modeling and Material Optimization for PCM-Enhanced Energy Piles

[中文说明](README_ZH.md) | [English](README.md)

This repository contains the code, data, and model files used for the EC-DeepONet study on PCM-enhanced energy piles.

## Contents

- data preparation
- COMSOL batch simulation and time-series export
- training scripts for MLP, DeepONet, and EC-DeepONet
- EC-DeepONet seed search and checkpoint refinement
- PCM optimization with the ensemble surrogate
- `Data_10Params.csv`, `processed_energy_dataset.csv`, `time_series_data/`, and `DeepOnet.mph`

## Requirements

- Python 3.10+
- PyTorch
- NumPy, Pandas, SciPy, scikit-learn
- Matplotlib, Seaborn
- COMSOL Multiphysics and `MPh` for COMSOL-related scripts

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Common scripts

```bash
python run_baselines.py
python run_ec_focus_search.py
python run_ec_checkpoint_refine.py
python run_ec_best_seed_search.py
python prepare_energy_dataset.py
python simulation.py
python run_pcm_optimization_case.py
```

## Notes

- Script details are in `docs/script_guide_en.md` and `docs/script_guide_zh.md`.
- `results_figures/` contains the figure rendering scripts.
- `time_series_data/` keeps the `.npz` files needed to rebuild `processed_energy_dataset.csv`.
