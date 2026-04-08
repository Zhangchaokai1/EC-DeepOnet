# PCM 增强型能量桩代理建模与材料优化代码仓库

[中文说明](README_ZH.md) | [English](README.md)

本仓库整理了论文对应的核心代码与必要数据，目标是提供一个适合发布到 GitHub 的干净版本，便于复现实验主流程。

当前仓库保留了以下内容：

- 数据准备代码
- COMSOL 批量仿真与时间序列导出代码
- MLP / DeepONet / EC-DeepONet 训练代码
- EC-DeepONet 随机种子筛选与集成优选代码
- 基于集成代理的 PCM 材料优化代码
- 训练所需的处理后数据集
- 原始汇总 CSV、时间序列数据与 COMSOL 模型文件

当前仓库刻意移除了以下非核心内容：

- 论文写作脚本
- 文献整理脚本
- Notebook 草稿
- 论文绘图脚本与中间产物
- 历次实验输出结果目录

## 1. 仓库结构

```text
.
├── README.md
├── README_ZH.md
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

## 2. 环境要求

- Python 3.10 或更高版本
- PyTorch
- NumPy / Pandas / SciPy / scikit-learn
- Matplotlib / Seaborn
- 若需要运行 COMSOL 仿真相关脚本，还需要：
  - COMSOL Multiphysics
  - Python 的 `MPh` 接口

安装依赖：

```bash
pip install -r requirements.txt
```

## 3. 推荐使用流程

### 3.1 如果你只想复现代理模型训练

直接使用已经整理好的 `processed_energy_dataset.csv`：

```bash
python run_baselines.py
python run_ec_focus_search.py
python run_ec_checkpoint_refine.py
python run_ec_best_seed_search.py
```

### 3.2 如果你想从原始时间序列重新构建处理后数据集

```bash
python prepare_energy_dataset.py
```

该脚本会读取：

- `Data_10Params.csv`
- `time_series_data/`

并生成：

- `processed_energy_dataset.csv`

### 3.3 如果你想重新运行 COMSOL 批量仿真

```bash
python simulation.py
```

该脚本会调用 `DeepOnet.mph`，批量采样 PCM 参数，并导出时间序列结果。

### 3.4 如果你想执行基于集成代理的材料优选

先完成随机种子筛选，得到 `outputs/ec_best_seed_search/...` 目录，然后运行：

```bash
python run_pcm_optimization_case.py
```

### 3.5 如果你想重绘论文结果图

在主流程输出目录已经生成之后，可以运行：

```bash
python results_figures/render_all.py
```

如果需要同时导出 SVG：

```bash
python results_figures/render_all_svg.py
```

绘图脚本默认会自动读取 `outputs/` 下最新一次的：

- `baseline_runs`
- `ec_focus_search`
- `ec_checkpoint_refine`
- `ec_best_seed_search`
- `optimization_case`

如果需要显式指定最优模型目录和 COMSOL 模型路径，可以使用：

```bash
python run_pcm_optimization_case.py ^
  --best-seed-dir outputs/ec_best_seed_search/20260404_120000 ^
  --model-path DeepOnet.mph
```

## 4. 脚本说明

每个脚本的作用、输入、输出和推荐使用时机，见：

- [docs/script_guide_zh.md](docs/script_guide_zh.md)
- [docs/script_guide_en.md](docs/script_guide_en.md)

## 5. 说明

- 该版本以“复现主流程”为目标，不包含论文排版和绘图相关辅助脚本。
- 发布版数据已按 134 个成功样本重新连续编号为 `1..134`，以避免论文与公开数据中出现跳号；原始 COMSOL 批量编号保留在 `Original_ID` 列中。
- `Data_10Params.csv`、`processed_energy_dataset.csv` 和 `time_series_data/` 在发布版中默认对应上述重排后的成功样本数据。
- `time_series_data/` 为发布版时间序列数据目录，保留后可直接重建处理后数据集。
- 当前发布版中，`time_series_data/` 仅保留训练与数据重建真正需要的 `.npz` 文件；COMSOL 导出的中间 `.txt`、`.csv` 和超时日志已清理。
- `run_pcm_optimization_case.py` 已改为优先读取你指定的 `--best-seed-dir`；若未指定，则自动选择 `outputs/ec_best_seed_search/` 下最新一次运行结果。
- `run_ec_checkpoint_refine.py` 已改为优先读取你指定的 `--focus-run-dir`；若未指定，则自动选择 `outputs/ec_focus_search/` 下最新一次运行结果。
- `results_figures/` 中保留了结果图重绘脚本，但不包含历史图片产物；运行前需要先有对应的实验输出目录。
