from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import spearmanr

from src.data.datasets import (
    build_case_dataset,
    build_point_dataset_from_cases,
    fit_case_scalers,
    split_case_dataframe,
    transform_case_dataset,
)
from src.data.energy import compute_cumulative_energy_mj
from src.data.io import load_main_dataframe
from src.data.schema import INPUT_COLUMNS, TARGET_ENERGY_COLUMN, TARGET_TOUT_COLUMN
from src.eval.predict import (
    evaluate_case_energy_predictions,
    evaluate_pointwise_predictions,
    predict_casewise_temperatures,
    predict_casewise_temperatures_from_point_model,
    ranking_metrics,
)
from src.models.deeponet import VanillaDeepONet
from src.models.ec_deeponet import ECDeepONet
from src.models.mlp import MLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "results_figures_svg"

MODEL_COLORS = {
    "COMSOL": "#111111",
    "MLP": "#1F5BFF",
    "DeepONet": "#6F4BFF",
    "EC-DeepONet": "#E31A1C",
}
BLUE_MAIN = "#1F5BFF"
PURPLE_MAIN = "#6F4BFF"
RED_MAIN = "#E31A1C"
GRAY_MAIN = "#64748B"
GRAY_LIGHT = "#CBD5E1"
JET_CMAP = "jet"


def resolve_dataset_path(env_var: str, default_path: Path) -> Path:
    override = os.environ.get(env_var)
    if not override:
        return default_path
    candidate = Path(override).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    return candidate.resolve()


def resolved_processed_dataset_csv() -> Path:
    return resolve_dataset_path("PCM_PROCESSED_DATASET_CSV", PROJECT_ROOT / "processed_energy_dataset.csv")


def resolved_time_series_dir() -> Path:
    return resolve_dataset_path("PCM_TIME_SERIES_DIR", PROJECT_ROOT / "time_series_data")

DISPLAY_NAMES = {
    "Input_rho_s": r"$\rho_{s}$",
    "Input_rho_l": r"$\rho_{l}$",
    "Input_cp_s": r"$c_{p,s}$",
    "Input_cl_l": r"$c_{p,l}$",
    "Input_k_s": r"$k_{s}$",
    "Input_k_l": r"$k_{l}$",
    "Input_Lf": r"$L_{f}$",
    "Input_wt": r"$w_{t}$",
    "Input_Tm": r"$T_{m}$",
    "Input_dT": r"$\Delta T$",
    TARGET_TOUT_COLUMN: r"$T_{out}$",
    TARGET_ENERGY_COLUMN: r"$E_{1-100h}$",
}

BOXPLOT_UNIT_LABELS = {
    "Input_rho_s": r"kg/m$^3$",
    "Input_rho_l": r"kg/m$^3$",
    "Input_cp_s": r"J/(kg·K)",
    "Input_cl_l": r"J/(kg·K)",
    "Input_k_s": r"W/(m·K)",
    "Input_k_l": r"W/(m·K)",
    "Input_Lf": r"kJ/kg",
    "Input_wt": "",
    "Input_Tm": r"$^\circ$C",
    "Input_dT": r"K",
    "__temperature_all__": r"$^\circ$C",
    TARGET_TOUT_COLUMN: r"$^\circ$C",
    TARGET_ENERGY_COLUMN: "MJ",
}

PARAMETER_BOUNDS = {
    "Input_rho_s": (800.0, 950.0),
    "Input_rho_l": (750.0, 850.0),
    "Input_cp_s": (1800.0, 2400.0),
    "Input_cl_l": (2000.0, 2600.0),
    "Input_k_s": (0.2, 0.25),
    "Input_k_l": (0.14, 0.2),
    "Input_Lf": (150.0, 250.0),
    "Input_wt": (0.1, 0.25),
    "Input_Tm": (20.0, 25.0),
    "Input_dT": (2.0, 7.0),
}


def rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha)


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def configure_plot_style() -> None:
    sns.set_theme(style="white", context="talk")
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["font.family"] = ["Times New Roman", "DejaVu Serif"]
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["axes.labelsize"] = 13
    mpl.rcParams["legend.frameon"] = False
    mpl.rcParams["axes.spines.top"] = True
    mpl.rcParams["axes.spines.right"] = True
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["grid.color"] = "#D6DCE5"
    mpl.rcParams["grid.alpha"] = 0.28
    mpl.rcParams["grid.linewidth"] = 0.8


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_figure(fig: plt.Figure, filename: str) -> Path:
    ensure_output_dir()
    path = OUTPUT_DIR / Path(filename).with_suffix(".svg")
    fig.savefig(path, bbox_inches="tight", facecolor="white", format="svg")
    plt.close(fig)
    return path


def panel_label(ax: plt.Axes, label: str, fontsize: float = 18) -> None:
    ax.text(-0.15, 1.03, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold")


@lru_cache(maxsize=1)
def load_dataset_statistics_bundle() -> dict[str, object]:
    dataset_csv = resolved_processed_dataset_csv()
    ts_dir = resolved_time_series_dir()
    df = load_main_dataframe(
        csv_path=dataset_csv,
        success_only=False,
        reset_index=True,
    )
    case_dataset = build_case_dataset(df, ts_dir=ts_dir)
    return {
        "dataset_df": df,
        "case_dataset": case_dataset,
    }


def apply_log_axis(
    ax: plt.Axes,
    axis: str = "y",
    tick_labelsize: float = 11.5,
    show_minor_grid: bool = True,
) -> None:
    locator_major = LogLocator(base=10.0)
    locator_minor = LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1)
    formatter_major = LogFormatterMathtext(base=10.0)
    tick_color = "#111111"

    if axis == "y":
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(locator_major)
        ax.yaxis.set_minor_locator(locator_minor)
        ax.yaxis.set_major_formatter(formatter_major)
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=tick_labelsize,
            length=9.0,
            width=1.45,
            direction="in",
            left=True,
            right=False,
            color=tick_color,
        )
        ax.tick_params(
            axis="y",
            which="minor",
            length=3.6,
            width=1.0,
            direction="in",
            left=True,
            right=False,
            color=tick_color,
        )
        ax.grid(True, which="major", axis="y", linestyle="-", alpha=0.24, linewidth=0.9)
        if show_minor_grid:
            ax.grid(True, which="minor", axis="y", linestyle=(0, (1.2, 3.2)), alpha=0.32, linewidth=0.8)
    elif axis == "x":
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(locator_major)
        ax.xaxis.set_minor_locator(locator_minor)
        ax.xaxis.set_major_formatter(formatter_major)
        ax.tick_params(
            axis="x",
            which="major",
            labelsize=tick_labelsize,
            length=9.0,
            width=1.45,
            direction="in",
            bottom=True,
            top=False,
            color=tick_color,
        )
        ax.tick_params(
            axis="x",
            which="minor",
            length=3.6,
            width=1.0,
            direction="in",
            bottom=True,
            top=False,
            color=tick_color,
        )
        ax.grid(True, which="major", axis="x", linestyle="-", alpha=0.24, linewidth=0.9)
        if show_minor_grid:
            ax.grid(True, which="minor", axis="x", linestyle=(0, (1.2, 3.2)), alpha=0.32, linewidth=0.8)
    else:
        raise ValueError(f"Unsupported axis: {axis}")


@lru_cache(maxsize=1)
def resolve_paths() -> dict[str, Path]:
    baseline_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "baseline_runs")
    ec_best_seed_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "ec_best_seed_search")
    ec_checkpoint_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "ec_checkpoint_refine")
    ec_focus_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "ec_focus_search")
    optimization_dir = latest_run_dir(PROJECT_ROOT / "outputs" / "optimization_case")

    with (ec_checkpoint_dir / "best_config.json").open("r", encoding="utf-8") as file:
        best_cfg = json.load(file)
    ec_model_path = ec_checkpoint_dir / best_cfg["best_tag"] / "refine" / "model.pt"

    return {
        "baseline_dir": baseline_dir,
        "ec_best_seed_dir": ec_best_seed_dir,
        "ec_checkpoint_dir": ec_checkpoint_dir,
        "ec_focus_dir": ec_focus_dir,
        "optimization_dir": optimization_dir,
        "ec_model_path": ec_model_path,
    }


def instantiate_best_ec_model(device: torch.device) -> ECDeepONet:
    model = ECDeepONet(
        latent_dim=128,
        branch_hidden_dims=(192, 192),
        trunk_hidden_dims=(128, 128),
        refine_hidden_dims=(128, 128),
        num_frequencies=0,
        max_frequency=1.0,
        dropout=0.0,
        use_layernorm=False,
    ).to(device)
    model.load_state_dict(torch.load(resolve_paths()["ec_model_path"], map_location=device))
    return model


def select_representative_cases(case_ids: np.ndarray, true_energy: np.ndarray, curves: dict[str, np.ndarray]) -> list[dict]:
    target_quantiles = [0.15, 0.40, 0.65, 0.90]
    energy_values = true_energy.reshape(-1)
    selected_indices: list[int] = []
    for quantile in target_quantiles:
        target = np.quantile(energy_values, quantile)
        idx = int(np.argmin(np.abs(energy_values - target)))
        if idx not in selected_indices:
            selected_indices.append(idx)
    selected_cases: list[dict] = []
    for idx in selected_indices[:4]:
        selected_cases.append(
            {
                "case_id": int(case_ids[idx]),
                "true_energy": float(energy_values[idx]),
                "true_temp": curves["COMSOL"][idx].reshape(-1),
                "mlp_temp": curves["MLP"][idx].reshape(-1),
                "vanilla_temp": curves["DeepONet"][idx].reshape(-1),
                "ec_temp": curves["EC-DeepONet"][idx].reshape(-1),
            }
        )
    return selected_cases


@lru_cache(maxsize=1)
def load_prediction_bundle() -> dict:
    paths = resolve_paths()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_csv = resolved_processed_dataset_csv()
    ts_dir = resolved_time_series_dir()

    df = load_main_dataframe(csv_path=dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=0.3, random_state=123)
    train_case_dataset = build_case_dataset(df_train, ts_dir=ts_dir)
    val_case_dataset = build_case_dataset(df_val, ts_dir=ts_dir)
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)
    val_case_dataset_scaled = transform_case_dataset(val_case_dataset, scalers)
    _ = build_point_dataset_from_cases(train_case_dataset, scalers)
    _ = build_point_dataset_from_cases(val_case_dataset, scalers)

    mlp = MLP().to(device)
    mlp.load_state_dict(torch.load(paths["baseline_dir"] / "mlp_model.pt", map_location=device))
    vanilla = VanillaDeepONet().to(device)
    vanilla.load_state_dict(torch.load(paths["baseline_dir"] / "deeponet_model.pt", map_location=device))
    ec = instantiate_best_ec_model(device)

    mlp_pred_curve = predict_casewise_temperatures_from_point_model(mlp, val_case_dataset_scaled, scalers, device)
    vanilla_pred_curve = predict_casewise_temperatures(vanilla, val_case_dataset_scaled, scalers, device)
    ec_pred_curve = predict_casewise_temperatures(ec, val_case_dataset_scaled, scalers, device)
    mlp_pred_curve_train = predict_casewise_temperatures_from_point_model(mlp, train_case_dataset_scaled, scalers, device)
    vanilla_pred_curve_train = predict_casewise_temperatures(vanilla, train_case_dataset_scaled, scalers, device)
    ec_pred_curve_train = predict_casewise_temperatures(ec, train_case_dataset_scaled, scalers, device)

    true_curve = val_case_dataset.temperatures
    model_curves = {
        "COMSOL": true_curve,
        "MLP": mlp_pred_curve,
        "DeepONet": vanilla_pred_curve,
        "EC-DeepONet": ec_pred_curve,
    }
    true_curve_train = train_case_dataset.temperatures

    metrics_rows = []
    energy_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    energy_metric_map: dict[str, dict[str, float]] = {}
    temperature_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    temperature_metric_map: dict[str, dict[str, float]] = {}
    for model_name, pred_curve in [
        ("MLP", mlp_pred_curve),
        ("DeepONet", vanilla_pred_curve),
        ("EC-DeepONet", ec_pred_curve),
    ]:
        point_metrics = evaluate_pointwise_predictions(true_curve, pred_curve)
        energy_metrics, pred_energy = evaluate_case_energy_predictions(val_case_dataset, pred_curve)
        rank_metrics = ranking_metrics(val_case_dataset.energies_mj, pred_energy)
        metrics_rows.append(
            {
                "Model": model_name,
                "Temperature MAE (°C)": point_metrics["MAE"],
                "Temperature RMSE (°C)": point_metrics["RMSE"],
                "Temperature R2": point_metrics["R2"],
                "Energy MAE (MJ)": energy_metrics["MAE"],
                "Energy RMSE (MJ)": energy_metrics["RMSE"],
                "Energy R2": energy_metrics["R2"],
                "Spearman": rank_metrics["SpearmanR"],
                "Kendall": rank_metrics["KendallTau"],
            }
        )
        energy_pairs[model_name] = (val_case_dataset.energies_mj.copy(), pred_energy.copy())
        energy_metric_map[model_name] = energy_metrics
        temperature_pairs[model_name] = (true_curve.reshape(-1).copy(), pred_curve.reshape(-1).copy())
        temperature_metric_map[model_name] = point_metrics

    train_energy_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    train_energy_metric_map: dict[str, dict[str, float]] = {}
    train_temperature_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    train_temperature_metric_map: dict[str, dict[str, float]] = {}
    for model_name, pred_curve in [
        ("MLP", mlp_pred_curve_train),
        ("DeepONet", vanilla_pred_curve_train),
        ("EC-DeepONet", ec_pred_curve_train),
    ]:
        point_metrics = evaluate_pointwise_predictions(true_curve_train, pred_curve)
        energy_metrics, pred_energy = evaluate_case_energy_predictions(train_case_dataset, pred_curve)
        train_energy_pairs[model_name] = (train_case_dataset.energies_mj.copy(), pred_energy.copy())
        train_energy_metric_map[model_name] = energy_metrics
        train_temperature_pairs[model_name] = (true_curve_train.reshape(-1).copy(), pred_curve.reshape(-1).copy())
        train_temperature_metric_map[model_name] = point_metrics

    comparison_df = pd.DataFrame(metrics_rows)
    focus_df = pd.read_csv(paths["ec_focus_dir"] / "summary_metrics.csv")
    checkpoint_df = pd.read_csv(paths["ec_checkpoint_dir"] / "summary_metrics.csv")
    ablation_df = pd.DataFrame(
        [
            {
                "Variant": "DeepONet",
                "Temperature RMSE (°C)": float(comparison_df.loc[comparison_df["Model"] == "DeepONet", "Temperature RMSE (°C)"].iloc[0]),
                "Energy RMSE (MJ)": float(comparison_df.loc[comparison_df["Model"] == "DeepONet", "Energy RMSE (MJ)"].iloc[0]),
            },
            {
                "Variant": "EC structure",
                "Temperature RMSE (°C)": float(focus_df.query("tag == 's1_wide' and phase == 'stage1_pretrain' and split == 'val'")["point_RMSE"].iloc[0]),
                "Energy RMSE (MJ)": float(focus_df.query("tag == 's1_wide' and phase == 'stage1_pretrain' and split == 'val'")["energy_RMSE"].iloc[0]),
            },
            {
                "Variant": "EC + energy tuning",
                "Temperature RMSE (°C)": float(focus_df.query("tag == 's1_wide__ft_refine_scaled_005' and phase == 'stage2_finetune' and split == 'val'")["point_RMSE"].iloc[0]),
                "Energy RMSE (MJ)": float(focus_df.query("tag == 's1_wide__ft_refine_scaled_005' and phase == 'stage2_finetune' and split == 'val'")["energy_RMSE"].iloc[0]),
            },
            {
                "Variant": "EC + checkpoint refine",
                "Temperature RMSE (°C)": float(checkpoint_df.query("tag == 'from_stage1_refine_scaled_005_lr1e4_b8' and split == 'val'")["point_RMSE"].iloc[0]),
                "Energy RMSE (MJ)": float(checkpoint_df.query("tag == 'from_stage1_refine_scaled_005_lr1e4_b8' and split == 'val'")["energy_RMSE"].iloc[0]),
            },
        ]
    )

    case_times = val_case_dataset.trunk_inputs.reshape(-1)
    selected_cases = select_representative_cases(val_case_dataset.case_ids, val_case_dataset.energies_mj, model_curves)

    return {
        "dataset_df": df,
        "val_case_dataset": val_case_dataset,
        "case_times": case_times,
        "comparison_df": comparison_df,
        "ablation_df": ablation_df,
        "energy_pairs": energy_pairs,
        "energy_metric_map": energy_metric_map,
        "temperature_pairs": temperature_pairs,
        "temperature_metric_map": temperature_metric_map,
        "train_energy_pairs": train_energy_pairs,
        "train_energy_metric_map": train_energy_metric_map,
        "train_temperature_pairs": train_temperature_pairs,
        "train_temperature_metric_map": train_temperature_metric_map,
        "selected_cases": selected_cases,
    }


@lru_cache(maxsize=1)
def load_optimization_bundle() -> dict:
    paths = resolve_paths()
    dataset_df = load_prediction_bundle()["dataset_df"]
    optimization_dir = paths["optimization_dir"]
    candidate_df = pd.read_csv(optimization_dir / "candidate_pool.csv")
    verification_df = pd.read_csv(optimization_dir / "verification_results.csv")
    boxplot_df = pd.read_csv(optimization_dir / "boxplot_sample_2500.csv")
    with (optimization_dir / "summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    with (optimization_dir / "selected_experts.json").open("r", encoding="utf-8") as file:
        selected_experts = json.load(file)

    best_observed_row = dataset_df.sort_values(TARGET_ENERGY_COLUMN, ascending=False).iloc[0]
    successful_df = verification_df[verification_df["status"] == "success"].copy()
    best_verified_row = successful_df.sort_values("true_energy_mj", ascending=False).iloc[0]
    summary.update(
        {
            "best_observed_energy_mj_recomputed": float(best_observed_row[TARGET_ENERGY_COLUMN]),
            "best_verified_energy_mj": float(best_verified_row["true_energy_mj"]),
            "best_verified_rank": int(best_verified_row["rank"]),
            "best_verified_gain_mj": float(best_verified_row["true_gain_vs_best_observed_mj"]),
            "best_verified_gain_pct": float(best_verified_row["true_gain_vs_best_observed_mj"] / best_observed_row[TARGET_ENERGY_COLUMN] * 100.0),
            "closed_loop_hours": float(summary["closed_loop_seconds"]) / 3600.0,
        }
    )
    return {
        "candidate_df": candidate_df,
        "verification_df": verification_df,
        "boxplot_df": boxplot_df,
        "summary": summary,
        "selected_experts": selected_experts,
        "best_observed_row": best_observed_row,
        "best_verified_row": best_verified_row,
    }


def summarize_candidates_above_threshold(
    threshold: float = 426.90,
    optimization_dir: Path | None = None,
    output_filename: str | None = None,
) -> dict[str, object]:
    if optimization_dir is None:
        optimization_dir = resolve_paths()["optimization_dir"]
    else:
        optimization_dir = Path(optimization_dir)

    candidate_path = optimization_dir / "candidate_pool.csv"
    candidate_df = pd.read_csv(candidate_path)
    selected_df = candidate_df[candidate_df["pred_energy_mean_mj"] > threshold].copy()

    summary_df = selected_df[INPUT_COLUMNS].describe(
        percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
    ).T.reset_index()
    summary_df = summary_df.rename(
        columns={
            "index": "feature",
            "5%": "p05",
            "25%": "p25",
            "50%": "p50",
            "75%": "p75",
            "95%": "p95",
        }
    )

    if output_filename:
        ensure_output_dir()
        summary_df.to_csv(OUTPUT_DIR / output_filename, index=False, encoding="utf-8")

    total_count = int(len(candidate_df))
    selected_count = int(len(selected_df))
    selected_ratio = float(selected_count / total_count) if total_count else 0.0

    return {
        "optimization_dir": optimization_dir,
        "candidate_path": candidate_path,
        "threshold": float(threshold),
        "total_count": total_count,
        "selected_count": selected_count,
        "selected_ratio": selected_ratio,
        "candidate_df": candidate_df,
        "selected_df": selected_df,
        "summary_df": summary_df,
    }


def create_threshold_candidate_violin_plot(
    threshold: float = 426.90,
    optimization_dir: Path | None = None,
    filename: str = "fig07_candidates_above_threshold_violin.png",
) -> Path:
    configure_plot_style()
    bundle = summarize_candidates_above_threshold(
        threshold=threshold,
        optimization_dir=optimization_dir,
    )
    candidate_df = bundle["candidate_df"]
    selected_df = bundle["selected_df"]
    if selected_df.empty:
        raise ValueError(f"No candidates found with pred_energy_mean_mj > {threshold:.2f}.")

    group_order = ["All candidates", f">{threshold:.2f} MJ"]
    palette = {
        "All candidates": BLUE_MAIN,
        f">{threshold:.2f} MJ": RED_MAIN,
    }

    fig, axes = plt.subplots(2, 5, figsize=(12.0, 7))
    fig.patch.set_facecolor("#F8FAFC")
    rng = np.random.default_rng(20260317)

    for idx, (ax, col) in enumerate(zip(axes.flat, INPUT_COLUMNS)):
        candidate_values = candidate_df[col].to_numpy(dtype=float)
        selected_values = selected_df[col].to_numpy(dtype=float)
        local_df = pd.concat(
            [
                pd.DataFrame({"group": group_order[0], "value": candidate_values}),
                pd.DataFrame({"group": group_order[1], "value": selected_values}),
            ],
            ignore_index=True,
        )

        collection_start = len(ax.collections)
        sns.violinplot(
            data=local_df,
            x="value",
            y="group",
            hue="group",
            order=group_order,
            hue_order=group_order,
            orient="h",
            cut=0,
            inner=None,
            bw_adjust=0.85,
            linewidth=1.4,
            saturation=1.0,
            dodge=False,
            legend=False,
            palette=palette,
            ax=ax,
        )
        new_collections = ax.collections[collection_start:]
        for artist, group_name, alpha in zip(new_collections, group_order, [0.20, 0.18]):
            color = palette[group_name]
            artist.set_facecolor(rgba(color, alpha))
            artist.set_edgecolor(rgba(color, 0.62))
            artist.set_linewidth(2.0)

        box = ax.boxplot(
            [candidate_values, selected_values],
            positions=np.arange(len(group_order)),
            vert=False,
            widths=0.22,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 2.3},
            whiskerprops={"linewidth": 1.7},
            capprops={"linewidth": 1.7},
            boxprops={"linewidth": 1.8},
            zorder=5,
        )
        for patch, color in zip(box["boxes"], [palette[name] for name in group_order]):
            patch.set_facecolor(rgba(color, 0.36))
            patch.set_edgecolor(color)
        for whisker, color in zip(box["whiskers"], [palette[group_order[0]], palette[group_order[0]], palette[group_order[1]], palette[group_order[1]]]):
            whisker.set_color(color)
        for cap, color in zip(box["caps"], [palette[group_order[0]], palette[group_order[0]], palette[group_order[1]], palette[group_order[1]]]):
            cap.set_color(color)

        scatter_defs = [
            (0, candidate_values, palette[group_order[0]], 0.10, 7),
            (1, selected_values, palette[group_order[1]], 0.34, 18),
        ]
        for y_pos, values, color, alpha, size in scatter_defs:
            jitter = np.clip(rng.normal(loc=y_pos, scale=0.038, size=len(values)), y_pos - 0.09, y_pos + 0.09)
            ax.scatter(
                values,
                jitter,
                s=size,
                color=rgba(color, alpha),
                edgecolors="none",
                zorder=4,
                rasterized=True,
            )

        lower, upper = PARAMETER_BOUNDS[col]
        pad = (upper - lower) * 0.04
        ax.set_xlim(lower - pad, upper + pad)
        ax.set_yticks(np.arange(len(group_order)))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        unit_label = BOXPLOT_UNIT_LABELS.get(col, "")
        ax.set_title("")
        title_parts = [
            TextArea(
                DISPLAY_NAMES[col],
                textprops={"size": 20, "color": "#111111"},
            )
        ]
        if unit_label:
            title_parts.append(
                TextArea(
                    f" ({unit_label})",
                    textprops={"size": 15, "color": "#444444"},
                )
            )
        title_box = HPacker(children=title_parts, align="center", pad=0, sep=1)
        anchored_title = AnchoredOffsetbox(
            loc="upper center",
            child=title_box,
            frameon=False,
            bbox_to_anchor=(0.5, 1.15),
            bbox_transform=ax.transAxes,
            borderpad=0.0,
            pad=0.0,
        )
        ax.add_artist(anchored_title)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_facecolor("white")
        ax.grid(True, axis="x", alpha=0.24, linewidth=0.9)
        ax.grid(False, axis="y")
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(
            axis="x",
            labelsize=18,
            length=7,
            width=1.3,
            labelrotation=30,
            bottom=True,
            top=False,
            direction="out",
            color="#111111",
        )
        ax.tick_params(axis="y", labelsize=18, pad=6, right=False, labelright=False, length=0)
        for tick_label in ax.get_xticklabels():
            tick_label.set_ha("right")
        ax.yaxis.set_ticks_position("left")
        ax.set_yticklabels([])
        ax.tick_params(axis="y", left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor("#D7DEE8")
            spine.set_linewidth(1.1)

    handles = [
        Patch(facecolor=rgba(palette[group_order[0]], 0.22), edgecolor=palette[group_order[0]], linewidth=1.8),
        Patch(facecolor=rgba(palette[group_order[1]], 0.20), edgecolor=palette[group_order[1]], linewidth=1.8),
    ]
    labels = [
        "All candidates (25000 samples)",
        f"Candidates > {threshold:.2f} MJ (112 samples, 0.448%)",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.03),
        fontsize=18,
        frameon=False,
        columnspacing=1.9,
        handlelength=2.7,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.10, top=0.86, wspace=0.28, hspace=0.60)
    return save_figure(fig, filename)


def aggregate_series_list(series_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_len = max(len(series) for series in series_list)
    matrix = np.full((len(series_list), max_len), np.nan, dtype=float)
    for idx, series in enumerate(series_list):
        matrix[idx, : len(series)] = series
    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def aggregate_histories(history_paths: list[Path], column: str) -> tuple[np.ndarray, np.ndarray]:
    series_list = [pd.read_csv(path)[column].to_numpy(dtype=float) for path in history_paths]
    return aggregate_series_list(series_list)


def suppress_local_loss_spikes(
    series: np.ndarray,
    *,
    window: int = 5,
    ratio_threshold: float = 2.5,
    start_epoch: int = 20,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    values = np.asarray(series, dtype=float).copy()
    replacements: list[dict[str, float]] = []
    for idx, value in enumerate(values.copy()):
        if idx + 1 < start_epoch or not np.isfinite(value) or value <= 0:
            continue
        left = max(0, idx - window)
        right = min(len(values), idx + window + 1)
        neighbors = np.concatenate((values[left:idx], values[idx + 1 : right]))
        neighbors = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
        if neighbors.size < 2 * window:
            continue
        local_median = float(np.median(neighbors))
        if local_median <= 0 or value <= local_median * ratio_threshold:
            continue
        replacements.append(
            {
                "epoch": float(idx + 1),
                "original": float(value),
                "replacement": local_median,
            }
        )
        values[idx] = local_median
    return values, replacements


def aggregate_histories_without_spikes(
    history_paths: list[Path],
    column: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[dict[str, float]]]]:
    cleaned_series_list: list[np.ndarray] = []
    cleaning_log: dict[str, list[dict[str, float]]] = {}
    for path in history_paths:
        cleaned_series, replacements = suppress_local_loss_spikes(
            pd.read_csv(path)[column].to_numpy(dtype=float)
        )
        cleaned_series_list.append(cleaned_series)
        if replacements:
            cleaning_log[path.parent.parent.name] = replacements
    mean_series, std_series = aggregate_series_list(cleaned_series_list)
    return mean_series, std_series, cleaning_log


@lru_cache(maxsize=1)
def load_expert_history_bundle() -> dict:
    paths = resolve_paths()
    optimization_bundle = load_optimization_bundle()
    expert_tags = [entry["tag"] for entry in optimization_bundle["selected_experts"]]
    stage1_paths = [paths["ec_best_seed_dir"] / tag / "stage1_pretrain" / "history.csv" for tag in expert_tags]
    stage2_paths = [paths["ec_best_seed_dir"] / tag / "stage2_finetune" / "history.csv" for tag in expert_tags]

    stage1_train_mean, stage1_train_std, stage1_train_cleaning = aggregate_histories_without_spikes(stage1_paths, "train_temp_loss")
    stage1_val_mean, stage1_val_std, stage1_val_cleaning = aggregate_histories_without_spikes(stage1_paths, "val_temp_loss")
    stage2_train_mean, stage2_train_std, stage2_train_cleaning = aggregate_histories_without_spikes(stage2_paths, "train_temp_loss")
    stage2_val_mean, stage2_val_std, stage2_val_cleaning = aggregate_histories_without_spikes(stage2_paths, "val_temp_loss")
    return {
        "expert_tags": expert_tags,
        "stage1_train_mean": stage1_train_mean,
        "stage1_train_std": stage1_train_std,
        "stage1_val_mean": stage1_val_mean,
        "stage1_val_std": stage1_val_std,
        "stage2_train_mean": stage2_train_mean,
        "stage2_train_std": stage2_train_std,
        "stage2_val_mean": stage2_val_mean,
        "stage2_val_std": stage2_val_std,
        "spike_cleaning_log": {
            "stage1_train_temp_loss": stage1_train_cleaning,
            "stage1_val_temp_loss": stage1_val_cleaning,
            "stage2_train_temp_loss": stage2_train_cleaning,
            "stage2_val_temp_loss": stage2_val_cleaning,
        },
    }


def save_results_tables() -> None:
    ensure_output_dir()
    prediction_bundle = load_prediction_bundle()
    optimization_bundle = load_optimization_bundle()
    prediction_bundle["comparison_df"].to_csv(OUTPUT_DIR / "table_fig02_model_comparison.csv", index=False, encoding="utf-8")
    prediction_bundle["ablation_df"].to_csv(OUTPUT_DIR / "table_fig04_ablation.csv", index=False, encoding="utf-8")
    optimization_bundle["verification_df"].to_csv(OUTPUT_DIR / "table_fig08_optimization_verification.csv", index=False, encoding="utf-8")
    with (OUTPUT_DIR / "optimization_summary.json").open("w", encoding="utf-8") as file:
        json.dump(optimization_bundle["summary"], file, ensure_ascii=False, indent=2)


def create_fig01_correlation_heatmap() -> Path:
    configure_plot_style()
    df = load_dataset_statistics_bundle()["dataset_df"]
    data = df[INPUT_COLUMNS + [TARGET_ENERGY_COLUMN]].rename(columns=DISPLAY_NAMES)
    corr = data.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9.0, 7.8))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0.0,
        square=True,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.85, "label": "Spearman coefficient"},
        ax=ax,
    )
    ax.tick_params(axis="x", rotation=25, labelsize=18)
    ax.tick_params(axis="y", rotation=0, labelsize=18)
    return save_figure(fig, "fig01_correlation_heatmap.png")


def create_fig01_feature_temperature_energy_boxplot() -> Path:
    configure_plot_style()
    bundle = load_dataset_statistics_bundle()
    df = bundle["dataset_df"]
    case_dataset = bundle["case_dataset"]

    boxplot_specs: list[tuple[str, str, np.ndarray]] = [
        (column, DISPLAY_NAMES[column], df[column].to_numpy(dtype=float))
        for column in INPUT_COLUMNS
    ]
    boxplot_specs.extend(
        [
            ("__temperature_all__", r"$T_{1-100h}$" + "\n(all points)", case_dataset.temperatures.reshape(-1)),
            (TARGET_ENERGY_COLUMN, DISPLAY_NAMES[TARGET_ENERGY_COLUMN], case_dataset.energies_mj.reshape(-1)),
        ]
    )

    palette = [
        "#1F5BFF",
        "#2F7BFF",
        "#49B7FF",
        "#16A34A",
        "#22C55E",
        "#84CC16",
        "#F59E0B",
        "#FB7185",
        "#E76F51",
        "#8B5CF6",
        "#6F4BFF",
        "#E31A1C",
    ]
    fig, axes = plt.subplots(4, 3, figsize=(8,10))
    rng = np.random.default_rng(20260323)

    def draw_compact_box(ax: plt.Axes, values: np.ndarray, color: str) -> None:
        box = ax.boxplot(
            values,
            vert=True,
            patch_artist=True,
            widths=0.28,
            showfliers=False,
            boxprops={"facecolor": rgba(color, 0.30), "edgecolor": color, "linewidth": 1.8},
            whiskerprops={"color": color, "linewidth": 1.4},
            capprops={"color": color, "linewidth": 1.4},
            medianprops={"color": color, "linewidth": 2.2},
        )
        for patch in box["boxes"]:
            patch.set_alpha(0.62)

        sample_size = min(len(values), 320)
        if sample_size > 0:
            replace = len(values) < sample_size
            sample = rng.choice(values, size=sample_size, replace=replace)
            jitter_x = rng.normal(loc=1.0, scale=0.026, size=sample_size)
            ax.scatter(
                jitter_x,
                sample,
                s=11,
                color=rgba(color, 0.16),
                edgecolors=rgba(color, 0.42),
                linewidths=0.45,
                zorder=1,
            )
        ax.set_xlim(0.72, 1.28)

    for ax, (key, title, values), color in zip(axes.flat, boxplot_specs, palette):
        values = np.asarray(values, dtype=float).reshape(-1)
        draw_compact_box(ax, values, color)

        unit_label = BOXPLOT_UNIT_LABELS.get(key, "")
        ax.set_title(title, fontsize=26.0, pad=10, linespacing=1.08)
        ax.set_xticks([])
        ax.set_xlabel("")
        ax.set_ylabel(unit_label, fontsize=22, labelpad=5)
        ax.tick_params(axis="y", labelsize=22)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")

    fig.subplots_adjust(left=0.10, right=0.985, top=0.985, bottom=0.045, hspace=0.52, wspace=0.24)
    plt.tight_layout()
    return save_figure(fig, "fig01_feature_temperature_energy_boxplot.png")


def create_fig13_parameter_response_network() -> Path:
    configure_plot_style()
    df = load_dataset_statistics_bundle()["dataset_df"]
    parameter_columns = INPUT_COLUMNS
    target_columns = [TARGET_ENERGY_COLUMN, TARGET_TOUT_COLUMN]

    n_params = len(parameter_columns)
    corr_matrix = np.full((n_params, n_params), np.nan, dtype=float)
    for row_idx, row_col in enumerate(parameter_columns):
        for col_idx in range(row_idx, n_params):
            col_col = parameter_columns[col_idx]
            if row_idx == col_idx:
                corr_matrix[row_idx, col_idx] = 1.0
                continue
            corr_value, _ = spearmanr(df[row_col], df[col_col], nan_policy="omit")
            corr_matrix[row_idx, col_idx] = float(corr_value)

    edge_stats: list[dict[str, float | str]] = []
    for target in target_columns:
        for idx, parameter in enumerate(parameter_columns):
            corr_value, p_value = spearmanr(df[target], df[parameter], nan_policy="omit")
            edge_stats.append(
                {
                    "target": target,
                    "parameter": parameter,
                    "parameter_idx": idx,
                    "r": float(corr_value),
                    "p": float(p_value),
                }
            )

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.05, 0.08, 0.73, 0.84])
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = mpl.cm.get_cmap("coolwarm")
    norm = mpl.colors.Normalize(vmin=-1.0, vmax=1.0)
    matrix_x0 = 3.75
    matrix_y0 = 0.35

    for row_idx in range(n_params):
        for col_idx in range(row_idx, n_params):
            x0 = matrix_x0 + col_idx
            y0 = matrix_y0 + row_idx
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    1.0,
                    1.0,
                    facecolor="white",
                    edgecolor=rgba("#111111", 0.72),
                    linewidth=1.7,
                    zorder=1,
                )
            )
            if row_idx == col_idx:
                continue
            corr_value = corr_matrix[row_idx, col_idx]
            square_size = 0.14 + 0.74 * abs(corr_value)
            center_x = x0 + 0.5
            center_y = y0 + 0.5
            ax.add_patch(
                Rectangle(
                    (center_x - square_size / 2.0, center_y - square_size / 2.0),
                    square_size,
                    square_size,
                    facecolor=cmap(norm(corr_value)),
                    edgecolor="none",
                    alpha=0.95,
                    zorder=2,
                )
            )

    target_positions = {
        TARGET_ENERGY_COLUMN: (3.5, 3),
        TARGET_TOUT_COLUMN: (3.9, 8.35),
    }
    target_colors = {
        TARGET_ENERGY_COLUMN: RED_MAIN,
        TARGET_TOUT_COLUMN: "#0F9D8A",
    }

    def edge_color_from_stats(corr_value: float, p_value: float) -> str:
        if p_value >= 0.05:
            return "#D5D9DF"
        return "#D96B00" if corr_value >= 0.0 else "#2D77C7"

    def edge_width_from_corr(corr_value: float) -> float:
        abs_corr = abs(corr_value)
        if abs_corr < 0.3:
            return 1.35
        if abs_corr < 0.5:
            return 2.45
        return 3.9

    def edge_linestyle_from_pvalue(p_value: float) -> str | tuple[int, tuple[float, ...]]:
        if p_value < 0.01:
            return "-"
        if p_value < 0.05:
            return (0, (6.0, 3.0))
        return "-"

    def edge_alpha_from_pvalue(p_value: float) -> float:
        if p_value < 0.01:
            return 0.96
        if p_value < 0.05:
            return 0.90
        return 0.82

    edge_stats_sorted = sorted(
        edge_stats,
        key=lambda item: (
            2 if float(item["p"]) >= 0.05 else (1 if float(item["p"]) >= 0.01 else 0),
            abs(float(item["r"])),
        ),
    )
    for item in edge_stats_sorted:
        target = str(item["target"])
        parameter_idx = int(item["parameter_idx"])
        start_x, start_y = target_positions[target]
        end_x = matrix_x0 + parameter_idx + 0.5
        end_y = matrix_y0 + parameter_idx + 0.5
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            color=edge_color_from_stats(float(item["r"]), float(item["p"])),
            linewidth=edge_width_from_corr(float(item["r"])),
            linestyle=edge_linestyle_from_pvalue(float(item["p"])),
            alpha=edge_alpha_from_pvalue(float(item["p"])),
            solid_capstyle="round",
            zorder=0,
        )

    for target in target_columns:
        x_pos, y_pos = target_positions[target]
        color = target_colors[target]
        ax.scatter(
            [x_pos],
            [y_pos],
            s=300,
            color=color,
            edgecolors="white",
            linewidths=1.6,
            zorder=4,
        )
        ax.text(
            x_pos,
            y_pos + 0.55,
            DISPLAY_NAMES[target],
            ha="center",
            va="center",
            fontsize=28 if target == TARGET_TOUT_COLUMN else 25,
            color="#111111",
            zorder=5,
        )

    for col_idx, column in enumerate(parameter_columns):
        ax.text(
            matrix_x0 + col_idx + 0.5,
            matrix_y0 - 0.28,
            DISPLAY_NAMES[column],
            rotation=90,
            ha="center",
            va="top",
            fontsize=20,
            color="#111111",
        )
    for row_idx, column in enumerate(parameter_columns):
        ax.text(
            matrix_x0 + n_params + 0.15,
            matrix_y0 + row_idx + 0.5,
            DISPLAY_NAMES[column],
            ha="left",
            va="center",
            fontsize=20,
            color="#4DA3E6",
        )

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.835, 0.67, 0.028, 0.24])
    colorbar = fig.colorbar(sm, cax=cax)
    colorbar.outline.set_visible(False)
    colorbar.ax.tick_params(labelsize=15, length=0, pad=12)
    colorbar.ax.set_title("Spearman r", fontsize=19, pad=10, loc="left")

    sign_handles = [
        Line2D([0], [0], color="#D96B00", linewidth=2.8),
        Line2D([0], [0], color="#2D77C7", linewidth=2.8),
    ]
    sign_labels = ["positive", "negative"]
    fig.legend(
        sign_handles,
        sign_labels,
        title="Sign",
        loc="upper left",
        bbox_to_anchor=(0.815, 0.62),
        frameon=False,
        title_fontsize=18,
        fontsize=16,
        handlelength=2.2,
        labelspacing=0.9,
    )

    width_handles = [
        Line2D([0], [0], color="black", linewidth=edge_width_from_corr(0.2)),
        Line2D([0], [0], color="black", linewidth=edge_width_from_corr(0.4)),
        Line2D([0], [0], color="black", linewidth=edge_width_from_corr(0.6)),
    ]
    width_labels = [r"$|r| < 0.3$", r"$0.3 \leq |r| < 0.5$", r"$|r| \geq 0.5$"]
    fig.legend(
        width_handles,
        width_labels,
        title=r"$|r|$",
        loc="upper left",
        bbox_to_anchor=(0.815, 0.44),
        frameon=False,
        title_fontsize=18,
        fontsize=16,
        handlelength=2.2,
        labelspacing=1.0,
    )

    p_handles = [
        Line2D([0], [0], color="#444444", linewidth=2.5, linestyle=edge_linestyle_from_pvalue(0.001)),
        Line2D([0], [0], color="#444444", linewidth=2.5, linestyle=edge_linestyle_from_pvalue(0.02)),
        Line2D([0], [0], color="#D5D9DF", linewidth=2.5, linestyle=edge_linestyle_from_pvalue(0.2)),
    ]
    p_labels = [r"$p < 0.01$", r"$0.01 \leq p < 0.05$", r"$p \geq 0.05$"]
    fig.legend(
        p_handles,
        p_labels,
        title=r"$p$ value",
        loc="upper left",
        bbox_to_anchor=(0.815, 0.22),
        frameon=False,
        title_fontsize=18,
        fontsize=16,
        handlelength=2.2,
        labelspacing=1.0,
    )

    ax.set_xlim(0.0, matrix_x0 + n_params + 2.35)
    ax.set_ylim(0.0, matrix_y0 + n_params + 0.55)
    plt.tight_layout()
    return save_figure(fig, "fig13_parameter_response_network.png")


def create_fig02_model_comparison() -> Path:
    configure_plot_style()
    metric_df = load_prediction_bundle()["comparison_df"]
    models = metric_df["Model"].tolist()
    colors = [MODEL_COLORS["MLP"], MODEL_COLORS["DeepONet"], MODEL_COLORS["EC-DeepONet"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    label_fs = 14.5
    tick_fs = 14
    value_fs = 14
    plot_defs = [
        ("Temperature RMSE (°C)", metric_df["Temperature RMSE (°C)"].to_numpy(), True, "(a)"),
        ("Energy RMSE (MJ)", metric_df["Energy RMSE (MJ)"].to_numpy(), True, "(b)"),
        ("Spearman", metric_df["Spearman"].to_numpy(), False, "(c)"),
    ]
    for ax, (ylabel, values, lower_is_better, label) in zip(axes, plot_defs):
        bars = ax.bar(models, values, width=0.60, color=[rgba(c, 0.34) for c in colors], edgecolor=colors, linewidth=2.0)
        if lower_is_better:
            ax.set_ylim(0.0, max(values) * 1.24)
            offset = max(values) * 0.035
        else:
            ax.set_ylim(min(values) - 0.020, max(values) + 0.035)
            offset = 0.003
        for bar, value, color in zip(bars, values, colors):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=value_fs,
                color=color,
                fontweight="bold",
            )
        ax.set_ylabel(ylabel, fontsize=label_fs)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.tick_params(axis="x", rotation=12, labelsize=tick_fs, pad=7)
        ax.tick_params(axis="y", labelsize=tick_fs)
        panel_label(ax, label)
    fig.subplots_adjust(bottom=0.23, wspace=0.32)
    return save_figure(fig, "fig02_model_comparison.png")


def _create_energy_parity_figure(
    energy_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    filename: str,
) -> Path:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    labels = ["(a)", "(b)", "(c)"]
    for ax, label, (name, (true_energy, pred_energy)) in zip(axes, labels, energy_pairs.items()):
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="24%", pad=0.08, sharex=ax)
        ax_right = divider.append_axes("right", size="24%", pad=0.08, sharey=ax)
        color = MODEL_COLORS[name]
        true_energy = true_energy.reshape(-1)
        pred_energy = pred_energy.reshape(-1)
        lim_min = min(true_energy.min(), pred_energy.min()) - 3
        lim_max = max(true_energy.max(), pred_energy.max()) + 3
        ax.scatter(true_energy, pred_energy, s=56, color=rgba(color, 0.82), edgecolors="white", linewidths=0.8)
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color=GRAY_MAIN, linewidth=1.5)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("True energy (MJ)", fontsize=13.5)
        if ax is axes[0]:
            ax.set_ylabel("Predicted energy (MJ)", fontsize=13.5)
        ax.tick_params(axis="both", labelsize=11.3)
        ax.text(0.04, 0.96, name, transform=ax.transAxes, va="top", fontsize=12.2, fontweight="bold")
        ax.text(
            0.04,
            0.80,
            f"RMSE = {metrics[name]['RMSE']:.3f} MJ\nMAE = {metrics[name]['MAE']:.3f} MJ\nR$^2$ = {metrics[name]['R2']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=10.8,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": GRAY_LIGHT},
        )
        ax.grid(True, axis="both")
        ax_top.hist(true_energy, bins=13, color=rgba(color, 0.55), edgecolor=color, linewidth=0.8)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelleft=False, left=False, labelsize=9.5)
        ax_top.grid(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_right.hist(pred_energy, bins=13, orientation="horizontal", color=rgba(color, 0.55), edgecolor=color, linewidth=0.8)
        ax_right.tick_params(axis="x", labelbottom=False, bottom=False, labelsize=9.5)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(False)
        ax_right.spines["left"].set_visible(False)
        panel_label(ax, label)
    fig.subplots_adjust(wspace=0.28)
    return save_figure(fig, filename)


def create_fig03_energy_parity() -> Path:
    bundle = load_prediction_bundle()
    return _create_energy_parity_figure(
        bundle["energy_pairs"],
        bundle["energy_metric_map"],
        "fig03_energy_parity.png",
    )


def create_fig03_energy_parity_train() -> Path:
    bundle = load_prediction_bundle()
    return _create_energy_parity_figure(
        bundle["train_energy_pairs"],
        bundle["train_energy_metric_map"],
        "fig03_energy_parity_train.png",
    )


def _create_temperature_parity_figure(
    temperature_pairs: dict[str, tuple[np.ndarray, np.ndarray]],
    metrics: dict[str, dict[str, float]],
    filename: str,
) -> Path:
    configure_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    labels = ["(a)", "(b)", "(c)"]
    for ax, label, (name, (true_temp, pred_temp)) in zip(axes, labels, temperature_pairs.items()):
        divider = make_axes_locatable(ax)
        ax_top = divider.append_axes("top", size="24%", pad=0.08, sharex=ax)
        ax_right = divider.append_axes("right", size="24%", pad=0.08, sharey=ax)
        color = MODEL_COLORS[name]
        true_temp = true_temp.reshape(-1)
        pred_temp = pred_temp.reshape(-1)
        lim_min = min(true_temp.min(), pred_temp.min()) - 0.15
        lim_max = max(true_temp.max(), pred_temp.max()) + 0.15
        ax.scatter(
            true_temp,
            pred_temp,
            s=3,
            color=rgba(color, 0.28),
            edgecolors="none",
            rasterized=True,
        )
        ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", color=GRAY_MAIN, linewidth=1.5)
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_xlabel("True temperature (°C)", fontsize=14)
        if ax is axes[0]:
            ax.set_ylabel("Predicted temperature (°C)", fontsize=14)
        ax.tick_params(axis="both", labelsize=13)
        ax.text(0.04, 0.96, name, transform=ax.transAxes, va="top", fontsize=14, fontweight="bold")
        ax.text(
            0.04,
            0.80,
            f"RMSE = {metrics[name]['RMSE']:.4f} °C\nMAE = {metrics[name]['MAE']:.4f} °C\nR$^2$ = {metrics[name]['R2']:.4f}",
            transform=ax.transAxes,
            va="top",
            fontsize=13,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": GRAY_LIGHT},
        )
        ax.grid(True, axis="both")
        ax_top.hist(true_temp, bins=24, color=rgba(color, 0.55), edgecolor=color, linewidth=0.8)
        ax_top.tick_params(axis="x", labelbottom=False)
        ax_top.tick_params(axis="y", labelleft=False, left=False, labelsize=13)
        ax_top.grid(False)
        ax_top.spines["bottom"].set_visible(False)
        ax_right.hist(
            pred_temp,
            bins=24,
            orientation="horizontal",
            color=rgba(color, 0.55),
            edgecolor=color,
            linewidth=0.8,
        )
        ax_right.tick_params(axis="x", labelbottom=False, bottom=False, labelsize=13)
        ax_right.tick_params(axis="y", labelleft=False)
        ax_right.grid(False)
        ax_right.spines["left"].set_visible(False)
        panel_label(ax, label)
    fig.subplots_adjust(wspace=0.28)
    return save_figure(fig, filename)


def create_fig12_temperature_parity() -> Path:
    bundle = load_prediction_bundle()
    return _create_temperature_parity_figure(
        bundle["temperature_pairs"],
        bundle["temperature_metric_map"],
        "fig12_temperature_parity.png",
    )


def create_fig12_temperature_parity_train() -> Path:
    bundle = load_prediction_bundle()
    return _create_temperature_parity_figure(
        bundle["train_temperature_pairs"],
        bundle["train_temperature_metric_map"],
        "fig12_temperature_parity_train.png",
    )


def create_fig04_ablation() -> Path:
    configure_plot_style()
    ablation_df = load_prediction_bundle()["ablation_df"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    order = ["DeepONet", "EC structure", "EC + energy tuning", "EC + checkpoint refine"]
    colors = ["#2563EB",  "#F59E0B", "#10B981","#DC2626"]
    x = np.arange(len(order))
    point_vals = [float(ablation_df.loc[ablation_df["Variant"] == name, "Temperature RMSE (°C)"].iloc[0]) for name in order]
    energy_vals = [float(ablation_df.loc[ablation_df["Variant"] == name, "Energy RMSE (MJ)"].iloc[0]) for name in order]

    for ax, values, ylabel, label in [
        (axes[0], point_vals, "Temperature RMSE (°C)", "(a)"),
        (axes[1], energy_vals, "Energy RMSE (MJ)", "(b)"),
    ]:
        bars = ax.bar(x, values, color=[rgba(c, 0.34) for c in colors], edgecolor=colors, linewidth=2.0)
        ax.set_xticks(x)
        ax.set_xticklabels(["DeepONet", "EC", "EC+E", "EC+Refine"], rotation=14, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.set_ylim(0.0, max(values) * 1.30)
        ax.tick_params(axis="y", labelsize=14)
        offset = max(values) * 0.045
        for bar, value, color in zip(bars, values, colors):
            ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:.3f}", ha="center", va="bottom", fontsize=15, color=color, fontweight="bold")
        panel_label(ax, label)
    fig.subplots_adjust(bottom=0.22, wspace=0.30)
    return save_figure(fig, "fig04_ablation.png")


def create_fig05_expert_loss_band() -> Path:
    configure_plot_style()
    bundle = load_expert_history_bundle()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    plot_defs = [
        (
            axes[0],
            np.arange(1, len(bundle["stage1_train_mean"]) + 1),
            bundle["stage1_train_mean"],
            bundle["stage1_train_std"],
            bundle["stage1_val_mean"],
            bundle["stage1_val_std"],
            "Stage I",
            "(a)",
        ),
        (
            axes[1],
            np.arange(1, len(bundle["stage2_train_mean"]) + 1),
            bundle["stage2_train_mean"],
            bundle["stage2_train_std"],
            bundle["stage2_val_mean"],
            bundle["stage2_val_std"],
            "Stage II",
            "(b)",
        ),
    ]
    all_positive = np.concatenate(
        [
            bundle["stage1_train_mean"][bundle["stage1_train_mean"] > 0],
            bundle["stage1_val_mean"][bundle["stage1_val_mean"] > 0],
            bundle["stage2_train_mean"][bundle["stage2_train_mean"] > 0],
            bundle["stage2_val_mean"][bundle["stage2_val_mean"] > 0],
        ]
    )
    y_min = 10 ** np.floor(np.log10(all_positive.min()) - 0.10)
    y_max = 10 ** np.ceil(np.log10(all_positive.max()) + 0.10)
    major_ticks = 10.0 ** np.arange(np.floor(np.log10(y_min)), np.ceil(np.log10(y_max)) + 1)
    for ax, epochs, train_mean, train_std, val_mean, val_std, note, label in plot_defs:
        ax.plot(epochs, train_mean, color=BLUE_MAIN, linewidth=2.2, label="Train mean")
        ax.fill_between(epochs, np.maximum(train_mean - train_std, 1e-12), train_mean + train_std, color=rgba(BLUE_MAIN, 0.20))
        ax.plot(epochs, val_mean, color=RED_MAIN, linewidth=2.2, label="Validation mean")
        ax.fill_between(epochs, np.maximum(val_mean - val_std, 1e-12), val_mean + val_std, color=rgba(RED_MAIN, 0.18))
        apply_log_axis(ax, axis="y", tick_labelsize=15.0)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks(major_ticks)
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        ax.set_xlabel("Epoch", fontsize=15.0)
        ax.tick_params(axis="x", labelsize=15.0)
        ax.grid(False, axis="x")
        ax.text(0.04, 0.96, note, transform=ax.transAxes, va="top", fontsize=15, fontweight="bold")
        ax.legend(
            loc="best",
            fontsize=15,
            frameon=True,
            facecolor="white",
            edgecolor=GRAY_LIGHT,
            borderpad=0.35,
            handlelength=2.2,
        )
        panel_label(ax, label, fontsize=15)
    axes[0].set_ylabel("Temperature loss", fontsize=15.0)
    axes[1].spines["right"].set_visible(True)
    axes[1].spines["left"].set_visible(True)
    axes[1].yaxis.tick_left()
    axes[1].yaxis.set_label_position("left")
    axes[1].tick_params(axis="y", which="major", left=True, right=False, labelleft=True, labelright=False, pad=6)
    axes[1].tick_params(axis="y", which="minor", left=True, right=False)
    fig.subplots_adjust(top=0.92, bottom=0.18, left=0.10, right=0.96, wspace=0.22)
    return save_figure(fig, "fig05_expert_loss_band.png")


def create_fig06_case_overview() -> Path:
    configure_plot_style()
    bundle = load_prediction_bundle()
    case_times = bundle["case_times"]
    selected_cases = bundle["selected_cases"]
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.5), sharex="col")
    series_defs = [
        ("COMSOL", "true_temp", MODEL_COLORS["COMSOL"], 2.2, 0.05),
        ("MLP", "mlp_temp", MODEL_COLORS["MLP"], 1.8, 0.05),
        ("DeepONet", "vanilla_temp", MODEL_COLORS["DeepONet"], 1.9, 0.05),
        ("EC-DeepONet", "ec_temp", MODEL_COLORS["EC-DeepONet"], 2.2, 0.07),
    ]
    fill_colors = ["#2563EB", "#DC2626", "#F59E0B", "#10B981"]  # 蓝色、红色、橙色、绿色
    inlet_temp = 35.0
    for col_idx, case in enumerate(selected_cases):
        temp_ax = axes[0, col_idx]
        energy_ax = axes[1, col_idx]
        temp_ax.axhline(inlet_temp, color=GRAY_MAIN, linestyle="--", linewidth=1.2, zorder=1)
        for label, key, color, lw, alpha in series_defs:
            curve = np.asarray(case[key]).reshape(-1)
            if label == "COMSOL":
                temp_ax.scatter(case_times, curve, color=color, s=20, label=label, zorder=3)
            else:
                temp_ax.plot(case_times, curve, color=color, linewidth=lw, label=label, zorder=3)
                temp_ax.fill_between(
                    case_times,
                    inlet_temp,
                    curve,
                    color=rgba(fill_colors[col_idx], 0.1),
                    zorder=2,
                )
        temp_ax.text(
            0.20,
            0.40,
            f"Case {case['case_id']}\n$E_{{1-100h}}$ = {case['true_energy']:.2f} MJ",
            transform=temp_ax.transAxes,
            va="top",
            fontsize=12.0,
            bbox={"facecolor": "white", "alpha": 0.90, "edgecolor": GRAY_LIGHT},
        )
        temp_ax.grid(True, axis="y")
        temp_ax.grid(False, axis="x")
        temp_ax.tick_params(labelsize=15.0)
        if col_idx == 0:
            temp_ax.set_ylabel("Outlet temperature (°C)", fontsize=13.5)

        for label, key, color, lw, _ in series_defs:
            curve = np.asarray(case[key]).reshape(-1)
            energy_curve = compute_cumulative_energy_mj(case_times, curve)
            energy_ax.plot(case_times, energy_curve, color=color, linewidth=lw)
        energy_ax.grid(True, axis="y")
        energy_ax.grid(False, axis="x")
        energy_ax.tick_params(labelsize=15.0)
        energy_ax.set_xlabel("Time (h)", fontsize=15)
        if col_idx == 0:
            energy_ax.set_ylabel("Cumulative energy (MJ)", fontsize=15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    inlet_handle = Line2D([0], [0], color=GRAY_MAIN, linestyle="--", linewidth=1.2)
    handles.append(inlet_handle)
    labels.append("Inlet 35°C")
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.975), fontsize=14, handlelength=2.4, columnspacing=1.2)
    fig.subplots_adjust(top=0.90, bottom=0.11, wspace=0.16, hspace=0.14)
    return save_figure(fig, "fig06_case_overview.png")


def create_fig07_optimization_landscape() -> Path:
    configure_plot_style()
    bundle = load_optimization_bundle()
    candidate_df = bundle["candidate_df"]
    verification_df = bundle["verification_df"]
    best_observed_energy = float(bundle["best_observed_row"][TARGET_ENERGY_COLUMN])
    print(f"Predicted E1-100h的数量: {len(candidate_df)}")
    # print("10个特征的分布:")
    # for col in INPUT_COLUMNS:
    #     if col in candidate_df.columns:
    #         values = candidate_df[col].dropna()
    #         print(f"{col}: min={values.min():.3f}, max={values.max():.3f}, mean={values.mean():.3f}, std={values.std():.3f}")
    #     else:
    #         print(f"{col}: not found in candidate_df")
    # print("这些数据的10个特征是优化过程中生成的候选参数，不是从原始训练数据集溯源的。")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": [1.08, 1.02]})
    rank_colors = {
        1: "#2563EB",
        2: "#DC2626",
        3: "#F59E0B",
    }

    ax = axes[0]
    hist_values = candidate_df["pred_energy_mean_mj"].to_numpy(dtype=float)
    counts, bin_edges = np.histogram(hist_values, bins=40)
    sns.histplot(hist_values, bins=40, color=rgba(BLUE_MAIN, 0.82), edgecolor="white", alpha=0.95, ax=ax)
    ax.axvline(best_observed_energy, color=RED_MAIN, linestyle="--", linewidth=2.0, label="426.90 MJ")
    ax.set_xlabel(r"$\tilde{\mu}_{E}$ (MJ)", fontsize=16)
    ax.set_ylabel("Candidate count", fontsize=16)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(loc="upper left", fontsize=16)
    ylim = ax.get_ylim()
    y_offset_scale = {
        1: 0.13,
        2: 0.09,
        3: 0.05,
    }
    label_offsets = {
        1: (10, 6),
        2: (10, 0),
        3: (10, -6),
    }
    for _, row in verification_df.iterrows():
        rank = int(row["rank"])
        x_val = float(row["pred_energy_mean_mj"])
        bin_idx = int(np.clip(np.digitize(x_val, bin_edges) - 1, 0, len(counts) - 1))
        y_val = float(counts[bin_idx]) + ylim[1] * y_offset_scale[rank]
        color = rank_colors.get(rank, BLUE_MAIN)
        # ax.scatter(
        #     x_val,
        #     y_val,
        #     s=280,
        #     marker="*",
        #     facecolor=color,
        #     edgecolor="black",
        #     linewidths=1.4,
        #     zorder=6,
        # )

    above_red_line = (candidate_df["pred_energy_mean_mj"] > best_observed_energy).sum()
    ax.text(
        0.97,
        0.05,
        f"{above_red_line} Samples",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        rotation=90,
        fontsize=17,
        color="#1F2937",
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
    )
    # ax.text(0.02, 0.96, "Observed best sample", transform=ax.transAxes, va="top", fontsize=12.2, color=RED_MAIN)
    panel_label(ax, "(a)")


    ax = axes[1]
    scatter = ax.scatter(
        candidate_df["pred_energy_mean_mj"],
        candidate_df["pred_energy_std_mj"],
        c=candidate_df["selection_score"],
        cmap=JET_CMAP,
        s=16,
        alpha=0.60,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.048, pad=0.04)
    cbar.set_label("Selection score", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    ax.axvline(best_observed_energy, color=RED_MAIN, linestyle="--", linewidth=1.8)
    label_offsets = {
        1: (-152, 100),
        2: (-152, 50),
        3: (-152, 0),
    }
    for _, row in verification_df.iterrows():
        rank = int(row["rank"])
        color = rank_colors.get(rank, BLUE_MAIN)
        x_val = float(row["pred_energy_mean_mj"])
        y_val = float(row["pred_energy_std_mj"])
        ax.scatter(
            x_val,
            y_val,
            s=340,
            marker="*",
            facecolor=color,
            edgecolor="black",
            linewidth=1.4,
            zorder=7,
        )
        ax.annotate(
            # f"Top {rank}\nPred {row['pred_energy_mean_mj']:.2f} MJ\nVer {row['true_energy_mj']:.2f} MJ",
            f"Top {rank}\nPred {row['pred_energy_mean_mj']:.2f} MJ",
            xy=(x_val, y_val),
            xytext=label_offsets[rank],
            textcoords="offset points",
            fontsize=16,
            bbox={"facecolor": "white", "alpha": 0.70, "edgecolor": color},
            arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": color},
        )
    ax.set_xlabel(r"$\tilde{\mu}_{E}$ (MJ)", fontsize=17)
    ax.set_ylabel(r"$\tilde{\sigma}_{E}$ (MJ)", fontsize=17)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, axis="both")
    panel_label(ax, "(b)")
    fig.subplots_adjust(wspace=0.24)
    return save_figure(fig, "fig07_optimization_landscape.png")


def create_fig08_optimization_benefit() -> Path:
    configure_plot_style()
    bundle = load_optimization_bundle()
    summary = bundle["summary"]
    best_observed_energy = float(bundle["best_observed_row"][TARGET_ENERGY_COLUMN])
    best_verified_row = bundle["best_verified_row"]
    best_verified_energy = float(best_verified_row["true_energy_mj"])
    best_verified_rank = int(best_verified_row["rank"])

    fig = plt.figure(figsize=(7, 3.2))
    outer_gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 0.95], wspace=0.34)

    ax_left = fig.add_subplot(outer_gs[0])

    bar_x = np.array([0.0, 0.92], dtype=float)
    bar_width = 0.52
    bar_colors = [rgba(BLUE_MAIN, 0.28), rgba(RED_MAIN, 0.24)]
    edge_colors = [BLUE_MAIN, RED_MAIN]
    bar_values = [best_observed_energy, best_verified_energy]

    bars = ax_left.bar(
        bar_x,
        bar_values,
        width=bar_width,
        color=bar_colors,
        edgecolor=edge_colors,
        linewidth=2.2,
        zorder=3,
    )
    ax_left.grid(True, axis="y", alpha=0.18, linewidth=0.9)
    ax_left.grid(False, axis="x")
    ax_left.set_facecolor("white")
    ax_left.tick_params(axis="y", labelsize=11.5)
    ax_left.tick_params(axis="x", labelsize=11.8)
    ax_left.set_ylim(400.0, max(bar_values) + 6.6)

    ax_left.text(
        bar_x[0],
        best_observed_energy + 0.55,
        f"{best_observed_energy:.2f} MJ",
        ha="center",
        va="bottom",
        fontsize=12.0,
        color=BLUE_MAIN,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 0.2},
    )
    ax_left.text(
        bar_x[1],
        best_verified_energy + 0.55,
        f"{best_verified_energy:.2f} MJ",
        ha="center",
        va="bottom",
        fontsize=12.0,
        color=RED_MAIN,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 0.2},
    )
    ax_left.set_xticks(bar_x)
    ax_left.set_xticklabels(
        [
            "Best in Training dataset",
            f"Optimized",
        ],
        fontsize=12.0,
    )
    ax_left.set_ylabel(r"$E_{1-100h}$ (MJ)", fontsize=13.5)
    ax_left.set_yticks([400, 410, 420, 430, 440])

    panel_label(ax_left, "(a)", fontsize=13)

    ax = fig.add_subplot(outer_gs[1])
    labels = ["COMSOL Only", "Our Frame"]
    values = [
        summary["bruteforce_hours"],
        summary["closed_loop_hours"],
    ]
    x_pos = np.arange(len(labels))
    ax.bar(
        x_pos,
        values,
        color=[rgba(BLUE_MAIN, 0.16), rgba(RED_MAIN, 0.72)],
        edgecolor=[BLUE_MAIN, RED_MAIN],
        linewidth=2.2,
        width=0.56,
    )
    ax.set_ylabel("Time (hours)", fontsize=13.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12.0)
    ax.set_ylim(0, 480)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.tick_params(axis="x", labelsize=11.8)
    ax.set_facecolor("white")
    detail_texts = [
        f"{values[0]:.2f} h",
        f"{summary['closed_loop_minutes']:.2f} min",
    ]
    for idx, value in enumerate(values):
        y_pos = value * 1.05 if idx == 0 else value * 200
        color = BLUE_MAIN if idx == 0 else RED_MAIN
        ax.text(
            idx,
            y_pos,
            detail_texts[idx],
            ha="center",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color=color,
            bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "none", "pad": 0.2},
        )
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.9)
    ax.grid(False, axis="x")
    panel_label(ax, "(b)", fontsize=13)
    fig.subplots_adjust(bottom=0.16)
    return save_figure(fig, "fig08_optimization_benefit.png")


def create_fig09_expert_boxplot() -> Path:
    configure_plot_style()
    bundle = load_optimization_bundle()
    boxplot_df = bundle["boxplot_df"]
    expert_tags = [entry["tag"] for entry in bundle["selected_experts"]]
    prediction_columns = [f"{tag}_pred_energy_mj" for tag in expert_tags]
    expert_labels = [f"Expert {idx + 1}" for idx in range(len(expert_tags))]
    long_df = boxplot_df[prediction_columns].rename(columns={f"{tag}_pred_energy_mj": label for tag, label in zip(expert_tags, expert_labels)}).melt(
        var_name="Expert",
        value_name="Predicted energy (MJ)",
    )
    palette = [BLUE_MAIN, "#2F7BFF", "#49B7FF", "#F28E2B", RED_MAIN][: len(expert_tags)]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    sns.boxplot(
        data=long_df,
        x="Expert",
        y="Predicted energy (MJ)",
        hue="Expert",
        palette=palette,
        width=0.62,
        linewidth=1.5,
        showfliers=False,
        dodge=False,
        legend=False,
        ax=ax,
    )
    for patch, color in zip(ax.patches, palette):
        patch.set_alpha(0.55)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)
    rng = np.random.default_rng(20260314)
    for idx, (expert, color) in enumerate(zip(expert_labels, palette)):
        values = long_df.loc[long_df["Expert"] == expert, "Predicted energy (MJ)"].to_numpy()
        jitter_x = rng.normal(loc=idx, scale=0.06, size=len(values))
        ax.scatter(
            jitter_x,
            values,
            s=10,
            color=rgba(color, 0.20),
            edgecolors="none",
            zorder=1,
        )
    ax.set_xlabel("")
    ax.set_ylabel(r"Predicted $E_{1-100h}$ (MJ)", fontsize=13.5)
    ax.tick_params(axis="x", labelsize=11.5)
    ax.tick_params(axis="y", labelsize=11.5)
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    return save_figure(fig, "fig09_expert_boxplot.png")


def create_fig10_parameter_shift() -> Path:
    configure_plot_style()
    prediction_bundle = load_prediction_bundle()
    optimization_bundle = load_optimization_bundle()

    best_observed_row = prediction_bundle["dataset_df"].sort_values(TARGET_ENERGY_COLUMN, ascending=False).iloc[0]
    best_verified_row = optimization_bundle["best_verified_row"]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    columns = INPUT_COLUMNS[::-1]
    y_pos = np.arange(len(columns))

    best_values = []
    opt_values = []
    for col in columns:
        lower, upper = PARAMETER_BOUNDS[col]
        best_values.append((float(best_observed_row[col]) - lower) / (upper - lower))
        opt_values.append((float(best_verified_row[col]) - lower) / (upper - lower))
    best_values = np.asarray(best_values, dtype=float)
    opt_values = np.asarray(opt_values, dtype=float)

    for idx, (best_v, opt_v) in enumerate(zip(best_values, opt_values)):
        ax.annotate(
            "",
            xy=(opt_v, idx),
            xytext=(best_v, idx),
            arrowprops={
                "arrowstyle": "-|>",
                "color": rgba(GRAY_MAIN, 0.40),
                "lw": 2.5,
                "shrinkA": 8,
                "shrinkB": 10,
                "mutation_scale": 13,
            },
            zorder=1,
        )

    ax.scatter(
        best_values,
        y_pos,
        s=105,
        color=BLUE_MAIN,
        edgecolor="white",
        linewidth=1.0,
        label="Best in training dataset",
        zorder=3,
    )
    ax.scatter(
        opt_values,
        y_pos,
        s=210,
        marker="*",
        color=RED_MAIN,
        edgecolor="black",
        linewidth=1.1,
        label="optimized (Verified)",
        zorder=4,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels([DISPLAY_NAMES[col] for col in columns], fontsize=15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("Normalized parameter position", fontsize=14.5)
    ax.tick_params(axis="x", labelsize=13.0)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(
        loc="lower right",
        fontsize=12.2,
        frameon=True,
        facecolor="white",
        edgecolor=GRAY_LIGHT,
        framealpha=0.90,
    )
    plt.tight_layout()
    return save_figure(fig, "fig10_parameter_shift.png")


def create_fig11_training_loss() -> Path:
    configure_plot_style()
    paths = resolve_paths()

    mlp_hist = pd.read_csv(paths["baseline_dir"] / "mlp_loss.csv")
    vanilla_hist = pd.read_csv(paths["baseline_dir"] / "deeponet_loss.csv")
    ec_hist = pd.read_csv(paths["ec_best_seed_dir"] / "seed_7" / "stage1_pretrain" / "history.csv")
    common_epoch_max = int(
        min(
            mlp_hist["epoch"].max(),
            vanilla_hist["epoch"].max(),
        )
    )

    mlp_hist = mlp_hist[mlp_hist["epoch"] <= common_epoch_max].copy()
    vanilla_hist = vanilla_hist[vanilla_hist["epoch"] <= common_epoch_max].copy()
    ec_hist = ec_hist[ec_hist["epoch"] <= common_epoch_max].copy()

    training_palette = {
        "MLP": "#2563EB",
        "DeepONet": "#F59E0B",
        "EC-DeepONet": "#DC2626",
    }

    histories = {
        "MLP": {
            "epoch": mlp_hist["epoch"].to_numpy(dtype=float),
            "train": mlp_hist["train_loss"].to_numpy(dtype=float),
            "val": mlp_hist["val_loss"].to_numpy(dtype=float),
            "color": training_palette["MLP"],
        },
        "DeepONet": {
            "epoch": vanilla_hist["epoch"].to_numpy(dtype=float),
            "train": vanilla_hist["train_loss"].to_numpy(dtype=float),
            "val": vanilla_hist["val_loss"].to_numpy(dtype=float),
            "color": training_palette["DeepONet"],
        },
        "EC-DeepONet": {
            "epoch": ec_hist["epoch"].to_numpy(dtype=float),
            "train": ec_hist["train_temp_loss"].to_numpy(dtype=float),
            "val": ec_hist["val_temp_loss"].to_numpy(dtype=float),
            "color": training_palette["EC-DeepONet"],
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    plot_defs = [
        ("train", "Train temperature loss", "(a)"),
        ("val", "Validation temperature loss", "(b)"),
    ]

    for ax, (mode, ylabel, label) in zip(axes, plot_defs):
        all_positive = np.concatenate(
            [
                histories["MLP"][mode][histories["MLP"][mode] > 0],
                histories["DeepONet"][mode][histories["DeepONet"][mode] > 0],
                histories["EC-DeepONet"][mode][histories["EC-DeepONet"][mode] > 0],
            ]
        )
        y_min = 10 ** np.floor(np.log10(all_positive.min()) - 0.05)
        y_max = 10 ** np.ceil(np.log10(all_positive.max()) + 0.05)

        for model_name, bundle in histories.items():
            ax.plot(
                bundle["epoch"],
                bundle[mode],
                color=bundle["color"],
                linewidth=2.4,
                label=model_name,
            )
        ax.set_ylim(y_min, y_max)
        apply_log_axis(ax, axis="y", tick_labelsize=15)
        ax.set_xlabel("Epoch", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.tick_params(axis="x", labelsize=15)
        ax.grid(False, axis="x")
        ax.legend(
            loc="upper right",
            fontsize=15,
            frameon=True,
            facecolor="white",
            edgecolor=GRAY_LIGHT,
            framealpha=0.90,
            handlelength=2.4,
        )
        panel_label(ax, label)

    fig.subplots_adjust(bottom=0.16, wspace=0.18)
    plt.tight_layout()
    return save_figure(fig, "fig11_training_loss.png")
