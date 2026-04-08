# 脚本说明

本文档使用中文说明当前仓库中每个保留脚本的作用，便于后续发布到 GitHub 后快速理解代码入口。

## 1. 数据相关脚本

### `txt2csv.py`

- 作用：将 COMSOL 导出的原始 TXT 表格清洗并转换为标准 CSV。
- 典型用途：供 `simulation.py` 在导出整个管道温度场后调用。
- 输入：COMSOL 导出的 TXT 文件。
- 输出：结构化 CSV 文件。

### `simulation.py`

- 作用：批量采样 PCM 参数，调用 COMSOL 模型执行高保真仿真，并导出每个样本的时间序列结果。
- 典型用途：从头生成原始数据集。
- 输入：
  - `DeepOnet.mph`
  - 内置的参数采样范围
- 输出：
  - `Data_10Params.csv`
  - `time_series_data/*.npz`
  - 对应的中间 TXT / CSV 导出文件
- 说明：该脚本依赖 COMSOL 和 `MPh`，是整个项目中最依赖外部软件环境的脚本。

### `prepare_energy_dataset.py`

- 作用：根据原始汇总 CSV 和时间序列 NPZ 文件，重新计算标准化口径下的累计换热量，并构建训练所用数据集。
- 典型用途：将高保真原始数据整理成代理模型训练输入。
- 输入：
  - `Data_10Params.csv`
  - `time_series_data/`
- 输出：
  - `processed_energy_dataset.csv`
- 说明：发布版数据目录中仅保留 `.npz` 时间序列文件，这已经足够支持该脚本运行。
- 说明：当前 GitHub 发布版已将成功样本重新连续编号为 `1..134`；若需要追溯原始 COMSOL 批量编号，请查看 `Original_ID` 列。

## 2. 训练相关脚本

### `run_baselines.py`

- 作用：训练并评估两个基线模型，即 MLP 和标准 DeepONet。
- 典型用途：生成基线性能结果，作为 EC-DeepONet 的对照。
- 输入：
  - `processed_energy_dataset.csv`
- 输出：
  - `outputs/baseline_runs/...`
  - loss 曲线 CSV
  - 温度与能量预测结果
  - 评价指标汇总

### `run_ec_search.py`

- 作用：执行第一轮 EC-DeepONet 超参数搜索。
- 典型用途：粗筛一组候选网络结构和训练配置。
- 输入：
  - `processed_energy_dataset.csv`
- 输出：
  - `outputs/ec_search/...`
  - 每组配置的训练记录和验证指标

### `run_ec_focus_search.py`

- 作用：在较优配置附近做第二轮更聚焦的 EC-DeepONet 搜索。
- 典型用途：从初步搜索结果中进一步确定主干结构和训练策略。
- 输入：
  - `processed_energy_dataset.csv`
- 输出：
  - `outputs/ec_focus_search/...`
  - 各阶段训练结果
  - 验证指标与配置摘要

### `run_ec_best_seed_search.py`

- 作用：在固定最终结构下，对多个随机种子重复训练，筛选最稳定、性能最好的 EC-DeepONet 模型。
- 典型用途：为后续集成代理提供候选专家模型。
- 输入：
  - `processed_energy_dataset.csv`
- 输出：
  - `outputs/ec_best_seed_search/...`
  - `summary_metrics.csv`
  - 每个 seed 的阶段一、阶段二模型权重和指标
- 说明：当前版本已支持通过命令行指定数据集路径、输出目录和数据划分随机种子。

### `run_ec_checkpoint_refine.py`

- 作用：基于 `run_ec_focus_search.py` 产出的较优 checkpoint，继续做第二轮精修和小范围微调。
- 典型用途：生成后续绘图脚本和最终结果分析所需的 `ec_checkpoint_refine` 输出。
- 输入：
  - `processed_energy_dataset.csv`
  - `outputs/ec_focus_search/...`
- 输出：
  - `outputs/ec_checkpoint_refine/...`
  - 精修阶段模型权重
  - 验证指标
  - 最优配置摘要
- 说明：
  - 若不显式传入 `--focus-run-dir`，脚本会自动选择 `outputs/ec_focus_search/` 下最新一次运行目录。
  - 默认使用 `s1_wide` 与 `s1_wide__ft_refine_scaled_005` 作为基准 checkpoint。

## 3. 优化相关脚本

### `run_pcm_optimization_case.py`

- 作用：读取多个训练好的 EC-DeepONet 专家模型，构建集成代理，对大规模候选 PCM 参数组合进行评分、筛选，并调用 COMSOL 做少量高保真回代验证。
- 典型用途：执行论文中的“集成代理驱动材料优选”实验。
- 输入：
  - `processed_energy_dataset.csv`
  - `outputs/ec_best_seed_search/...`
  - `DeepOnet.mph`
- 输出：
  - `outputs/optimization_case/...`
  - 候选池预测结果
  - shortlist 候选
  - 高保真验证结果
  - summary JSON
- 说明：
  - 若不显式传入 `--best-seed-dir`，脚本会自动选择 `outputs/ec_best_seed_search/` 下最新一次运行目录。
  - 若需要回代验证，仍依赖 COMSOL 和 `MPh`。

## 4. 绘图相关脚本

### `results_figures/render_all.py`

- 作用：一次性重绘所有主要结果图，并保存到 `outputs/results_figures_v1/`。
- 典型用途：在主流程实验完成后，批量生成论文所需结果图。
- 依赖输出目录：
  - `outputs/baseline_runs/...`
  - `outputs/ec_focus_search/...`
  - `outputs/ec_checkpoint_refine/...`
  - `outputs/ec_best_seed_search/...`
  - `outputs/optimization_case/...`

### `results_figures/render_all_svg.py`

- 作用：批量导出 SVG 版本结果图。
- 典型用途：需要高质量矢量图时使用。

### `results_figures/common.py`

- 作用：集中封装绘图公共逻辑、数据加载、模型恢复和结果整理函数。
- 典型用途：被各单图脚本和批量绘图脚本调用。

### `results_figures/common_svg.py`

- 作用：与 `common.py` 类似，但针对 SVG 输出流程组织。

### `results_figures/fig01_*.py` 到 `fig13_*.py`

- 作用：分别绘制单个结果图。
- 典型用途：只想重绘某一幅图时使用。

### `results_figures/analyze_threshold_candidates.py`

- 作用：对候选筛选阈值和高分区域进行补充分析。

### `results_figures/export_fig07_word_png.py`

- 作用：导出适合 Word 使用的图像版本。

## 5. 推荐运行顺序

如果只是复现主流程，建议按照下面顺序使用：

1. `prepare_energy_dataset.py`
2. `run_baselines.py`
3. `run_ec_focus_search.py`
4. `run_ec_checkpoint_refine.py`
5. `run_ec_best_seed_search.py`
6. `run_pcm_optimization_case.py`
7. `results_figures/render_all.py`

如果需要从头重新生成高保真原始数据，再额外运行：

1. `simulation.py`
2. `prepare_energy_dataset.py`
