# Script Guide

[中文说明](script_guide_zh.md) | [English](script_guide_en.md)

This document provides an English description of the retained scripts in this repository so that the code entry points can be understood quickly after release.

## 1. Data-related scripts

### `txt2csv.py`

- Purpose: clean raw TXT tables exported by COMSOL and convert them into standard CSV files.
- Typical use: called by `simulation.py` after exporting the full temperature field of the pipe domain.
- Input: TXT files exported from COMSOL.
- Output: structured CSV files.

### `simulation.py`

- Purpose: sample PCM parameters in batch, call the COMSOL model for high-fidelity simulation, and export the time-series result for each sample.
- Typical use: generate the raw dataset from scratch.
- Input:
  - `DeepOnet.mph`
  - built-in parameter sampling ranges
- Output:
  - `Data_10Params.csv`
  - `time_series_data/*.npz`
  - corresponding intermediate TXT / CSV export files
- Note: this is the most external-environment-dependent script in the project because it requires both COMSOL and `MPh`.

### `prepare_energy_dataset.py`

- Purpose: recompute cumulative heat exchange under the standardized definition from the summary CSV and NPZ time-series files, then build the dataset used for training.
- Typical use: transform raw high-fidelity simulation outputs into surrogate-model training inputs.
- Input:
  - `Data_10Params.csv`
  - `time_series_data/`
- Output:
  - `processed_energy_dataset.csv`
- Note: the released dataset keeps only the `.npz` time-series files, which is sufficient for this script.
- Note: in the GitHub release, successful samples have been reindexed consecutively as `1..134`; use the `Original_ID` column if you need the original COMSOL batch index.

## 2. Training-related scripts

### `run_baselines.py`

- Purpose: train and evaluate two baseline models, MLP and standard DeepONet.
- Typical use: generate baseline performance results for comparison with EC-DeepONet.
- Input:
  - `processed_energy_dataset.csv`
- Output:
  - `outputs/baseline_runs/...`
  - loss-curve CSV files
  - temperature and energy prediction results
  - summary metrics

### `run_ec_search.py`

- Purpose: perform the first-round hyperparameter search for EC-DeepONet.
- Typical use: coarsely screen candidate network structures and training configurations.
- Input:
  - `processed_energy_dataset.csv`
- Output:
  - `outputs/ec_search/...`
  - training logs and validation metrics for each configuration

### `run_ec_focus_search.py`

- Purpose: run a second, more focused EC-DeepONet search around promising configurations.
- Typical use: further determine the main architecture and training strategy from the initial search results.
- Input:
  - `processed_energy_dataset.csv`
- Output:
  - `outputs/ec_focus_search/...`
  - stage-wise training results
  - validation metrics and configuration summaries

### `run_ec_best_seed_search.py`

- Purpose: repeatedly train the final EC-DeepONet structure with multiple random seeds and screen the most stable and best-performing models.
- Typical use: provide candidate expert models for the later ensemble surrogate.
- Input:
  - `processed_energy_dataset.csv`
- Output:
  - `outputs/ec_best_seed_search/...`
  - `summary_metrics.csv`
  - stage-I / stage-II model weights and metrics for each seed
- Note: this version supports specifying the dataset path, output directory, and random split seed through command-line arguments.

### `run_ec_checkpoint_refine.py`

- Purpose: continue refinement and local fine-tuning from a better checkpoint produced by `run_ec_focus_search.py`.
- Typical use: generate the `ec_checkpoint_refine` outputs required by the figure scripts and final result analysis.
- Input:
  - `processed_energy_dataset.csv`
  - `outputs/ec_focus_search/...`
- Output:
  - `outputs/ec_checkpoint_refine/...`
  - refined model weights
  - validation metrics
  - best-configuration summaries
- Notes:
  - If `--focus-run-dir` is not provided explicitly, the script automatically selects the latest run under `outputs/ec_focus_search/`.
  - By default, `s1_wide` and `s1_wide__ft_refine_scaled_005` are used as the reference checkpoints.

## 3. Optimization-related scripts

### `run_pcm_optimization_case.py`

- Purpose: load multiple trained EC-DeepONet expert models, build an ensemble surrogate, score and screen large candidate PCM parameter combinations, and call COMSOL for limited high-fidelity re-evaluation.
- Typical use: reproduce the ensemble-surrogate-driven material optimization experiment reported in the paper.
- Input:
  - `processed_energy_dataset.csv`
  - `outputs/ec_best_seed_search/...`
  - `DeepOnet.mph`
- Output:
  - `outputs/optimization_case/...`
  - candidate-pool prediction results
  - shortlisted candidates
  - high-fidelity verification results
  - summary JSON files
- Notes:
  - If `--best-seed-dir` is not provided explicitly, the script automatically selects the latest run under `outputs/ec_best_seed_search/`.
  - High-fidelity re-evaluation still requires COMSOL and `MPh`.

## 4. Figure-related scripts

### `results_figures/render_all.py`

- Purpose: re-render all main result figures at once and save them to `outputs/results_figures_v1/`.
- Typical use: generate all manuscript figures after the main workflow has finished.
- Required output folders:
  - `outputs/baseline_runs/...`
  - `outputs/ec_focus_search/...`
  - `outputs/ec_checkpoint_refine/...`
  - `outputs/ec_best_seed_search/...`
  - `outputs/optimization_case/...`

### `results_figures/render_all_svg.py`

- Purpose: export SVG versions of the result figures in batch.
- Typical use: use when high-quality vector figures are needed.

### `results_figures/common.py`

- Purpose: centralize shared figure logic, data loading, model restoration, and result aggregation functions.
- Typical use: imported by both single-figure scripts and batch figure-rendering scripts.

### `results_figures/common_svg.py`

- Purpose: similar to `common.py`, but organized for the SVG export workflow.

### `results_figures/fig01_*.py` to `fig13_*.py`

- Purpose: generate individual result figures.
- Typical use: use when only one specific figure needs to be regenerated.

### `results_figures/analyze_threshold_candidates.py`

- Purpose: perform supplementary analysis of the candidate-screening threshold and high-score region.

### `results_figures/export_fig07_word_png.py`

- Purpose: export figure variants suitable for Word insertion.

## 5. Recommended execution order

If you only want to reproduce the main workflow, the recommended order is:

1. `prepare_energy_dataset.py`
2. `run_baselines.py`
3. `run_ec_focus_search.py`
4. `run_ec_checkpoint_refine.py`
5. `run_ec_best_seed_search.py`
6. `run_pcm_optimization_case.py`
7. `results_figures/render_all.py`

If you need to regenerate the raw high-fidelity dataset from scratch, run these additionally:

1. `simulation.py`
2. `prepare_energy_dataset.py`
