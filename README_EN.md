# Surrogate Modeling and Material Optimization for PCM-Enhanced Energy Piles

[中文说明](README.md) | [English](README_EN.md)

This repository contains the core code and required data for the paper on EC-DeepONet-based surrogate modeling and material optimization of PCM-enhanced energy piles. The release is intentionally kept clean and focused on reproducing the main workflow.

The repository currently includes:

- data-preparation scripts
- COMSOL batch-simulation and time-series export scripts
- training scripts for MLP, DeepONet, and EC-DeepONet
- random-seed screening and ensemble selection scripts for EC-DeepONet
- ensemble-surrogate-driven PCM optimization scripts
- processed datasets required for training
- the summary CSV, time-series data, and the COMSOL model file

The following non-essential items were intentionally removed:

- manuscript-writing scripts
- literature-management scripts
- notebook drafts
- manuscript-figure helper scripts and intermediate figure artifacts
- historical experiment output folders

## 1. Repository structure

```text
.
├── README.md
├── README_EN.md
├── requirements.txt
├── .gitignore
├── DeepOnet.mph
├── Data_10Params.csv
├── processed_energy_dataset.csv
├── time_series_data/
├── prepare_energy_dataset.py
├── simulation.py
├── txt2csv.py
├── run_baselines.py
├── run_ec_search.py
├── run_ec_focus_search.py
├── run_ec_checkpoint_refine.py
├── run_ec_best_seed_search.py
├── run_pcm_optimization_case.py
├── results_figures/
├── docs/
│   ├── script_guide_zh.md
│   └── script_guide_en.md
└── src/
    ├── data/
    ├── eval/
    ├── models/
    └── train/
```

## 2. Environment requirements

- Python 3.10 or later
- PyTorch
- NumPy / Pandas / SciPy / scikit-learn
- Matplotlib / Seaborn
- To run COMSOL-related scripts, you also need:
  - COMSOL Multiphysics
  - the Python `MPh` interface

Install dependencies with:

```bash
pip install -r requirements.txt
```

## 3. Recommended workflow

### 3.1 Reproduce surrogate-model training only

Use the prepared dataset directly:

```bash
python run_baselines.py
python run_ec_focus_search.py
python run_ec_checkpoint_refine.py
python run_ec_best_seed_search.py
```

### 3.2 Rebuild the processed dataset from raw time-series files

```bash
python prepare_energy_dataset.py
```

This script reads:

- `Data_10Params.csv`
- `time_series_data/`

and generates:

- `processed_energy_dataset.csv`

### 3.3 Re-run COMSOL batch simulations

```bash
python simulation.py
```

This script calls `DeepOnet.mph`, samples PCM parameters in batch, and exports the corresponding time-series results.

### 3.4 Run ensemble-surrogate-driven PCM optimization

First complete random-seed screening so that `outputs/ec_best_seed_search/...` is available, then run:

```bash
python run_pcm_optimization_case.py
```

### 3.5 Re-render the manuscript figures

After the main workflow outputs are available, run:

```bash
python results_figures/render_all.py
```

To export SVG figures as well:

```bash
python results_figures/render_all_svg.py
```

By default, the figure scripts automatically read the latest results under `outputs/` for:

- `baseline_runs`
- `ec_focus_search`
- `ec_checkpoint_refine`
- `ec_best_seed_search`
- `optimization_case`

If you want to specify the best-seed directory and COMSOL model path explicitly, use:

```bash
python run_pcm_optimization_case.py ^
  --best-seed-dir outputs/ec_best_seed_search/20260404_120000 ^
  --model-path DeepOnet.mph
```

## 4. Script guide

Script-by-script descriptions, inputs, outputs, and recommended usage are provided in:

- [English guide](docs/script_guide_en.md)
- [中文说明](docs/script_guide_zh.md)

## 5. Notes

- This release is intended for reproducing the main workflow and does not include manuscript-layout utilities or figure-production helper scripts outside the retained `results_figures/` module.
- The released dataset has been reindexed to consecutive successful cases `1..134` to avoid non-contiguous sample IDs in the public dataset. The original COMSOL batch index is retained in the `Original_ID` column.
- `Data_10Params.csv`, `processed_energy_dataset.csv`, and `time_series_data/` in this release all correspond to the reindexed successful samples.
- `time_series_data/` is the released time-series directory and is sufficient to rebuild the processed dataset.
- Only the `.npz` files required for training and dataset reconstruction are kept in `time_series_data/`. Intermediate COMSOL `.txt`, `.csv`, and timeout logs were removed.
- `run_pcm_optimization_case.py` now prefers the explicitly provided `--best-seed-dir`; otherwise, it automatically selects the latest run under `outputs/ec_best_seed_search/`.
- `run_ec_checkpoint_refine.py` now prefers the explicitly provided `--focus-run-dir`; otherwise, it automatically selects the latest run under `outputs/ec_focus_search/`.
- `results_figures/` keeps the figure re-rendering scripts, but not historical figure outputs. The corresponding experiment result folders must exist before rendering.
