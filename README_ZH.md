# PCM 增强型能量桩代理建模与材料优化代码仓库

[中文说明](README_ZH.md) | [English](README.md)

本仓库提供 EC-DeepONet 相关的代码、数据和模型文件，用于复现实验主流程。

## 内容

- 数据准备
- COMSOL 批量仿真与时间序列导出
- MLP / DeepONet / EC-DeepONet 训练
- EC-DeepONet 随机种子筛选与 checkpoint 优选
- 基于集成代理的 PCM 优化
- `Data_10Params.csv`、`processed_energy_dataset.csv`、`time_series_data/`、`DeepOnet.mph`

## 环境

- Python 3.10+
- PyTorch
- NumPy / Pandas / SciPy / scikit-learn
- Matplotlib / Seaborn
- COMSOL Multiphysics 与 `MPh`（仅仿真相关脚本需要）

安装依赖：

```bash
pip install -r requirements.txt
```

## 常用脚本

```bash
python run_baselines.py
python run_ec_focus_search.py
python run_ec_checkpoint_refine.py
python run_ec_best_seed_search.py
python prepare_energy_dataset.py
python simulation.py
python run_pcm_optimization_case.py
```

## 说明

- 各脚本的输入、输出和用法见 `docs/script_guide_zh.md` 与 `docs/script_guide_en.md`。
- `results_figures/` 保留结果图重绘脚本。
- `time_series_data/` 保留重建 `processed_energy_dataset.csv` 所需的 `.npz` 文件。
