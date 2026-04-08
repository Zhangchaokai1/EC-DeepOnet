from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    CasewiseBranchDataset,
    build_case_dataset,
    fit_case_scalers,
    split_case_dataframe,
    transform_case_dataset,
)
from src.data.io import load_main_dataframe
from src.eval.predict import (
    evaluate_case_energy_predictions,
    evaluate_pointwise_predictions,
    predict_casewise_temperatures,
    ranking_metrics,
)
from src.models.ec_deeponet import ECDeepONet
from src.train.losses import EnergyConsistencyLoss
from src.train.trainer import TrainingHistory, train_casewise_model_with_custom_loss


BASELINE_POINT_RMSE = 0.06394333678442818
BASELINE_ENERGY_RMSE = 12.005381096909694


@dataclass(frozen=True)
class StageConfig:
    name: str
    temperature_weight: float
    energy_weight: float
    smoothness_weight: float
    temperature_loss: str
    energy_loss: str
    learning_rate: float
    weight_decay: float
    batch_size_cases: int
    epochs: int
    grad_clip_norm: float
    scheduler_patience: int
    scheduler_factor: float
    early_stopping_patience: int
    min_delta: float
    monitor_metric: str
    trainable_prefixes: tuple[str, ...] | None = None
    relative_energy_epsilon_mj: float = 1.0


@dataclass(frozen=True)
class Stage1Config:
    name: str
    latent_dim: int
    branch_hidden_dims: tuple[int, ...]
    trunk_hidden_dims: tuple[int, ...]
    refine_hidden_dims: tuple[int, ...]
    num_frequencies: int
    max_frequency: float
    dropout: float
    use_layernorm: bool
    stage: StageConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Focused search for EC-DeepONet.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("processed_energy_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ec_focus_search")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--top-k-stage1", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_output_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history(history: TrainingHistory, path: Path) -> None:
    data = {
        "epoch": list(range(1, len(history.train_losses) + 1)),
        "train_loss": history.train_losses,
        "val_loss": history.val_losses,
        "train_temp_loss": history.train_temp_losses,
        "val_temp_loss": history.val_temp_losses,
        "train_energy_loss": history.train_energy_losses,
        "val_energy_loss": history.val_energy_losses,
        "train_smooth_loss": history.train_smooth_losses,
        "val_smooth_loss": history.val_smooth_losses,
    }
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")


def metrics_row(tag: str, phase: str, split: str, point_metrics: dict, energy_metrics: dict, rank_metrics: dict) -> dict:
    row = {"tag": tag, "phase": phase, "split": split}
    row.update({f"point_{k}": v for k, v in point_metrics.items()})
    row.update({f"energy_{k}": v for k, v in energy_metrics.items()})
    row.update({f"rank_{k}": v for k, v in rank_metrics.items()})
    return row


def overall_score(val_row: dict) -> float:
    point_ratio = float(val_row["point_RMSE"]) / BASELINE_POINT_RMSE
    energy_ratio = float(val_row["energy_RMSE"]) / BASELINE_ENERGY_RMSE
    return 0.35 * point_ratio + 0.65 * energy_ratio


def trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def set_trainable_prefixes(model: torch.nn.Module, trainable_prefixes: tuple[str, ...] | None) -> None:
    if trainable_prefixes is None:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for name, parameter in model.named_parameters():
        parameter.requires_grad = False
        for prefix in trainable_prefixes:
            if name == prefix or name.startswith(f"{prefix}."):
                parameter.requires_grad = True
                break


def build_loader(case_dataset_scaled, batch_size_cases: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        CasewiseBranchDataset(
            case_dataset_scaled.branch_inputs,
            case_dataset_scaled.temperatures,
            case_dataset_scaled.energies_mj,
        ),
        batch_size=batch_size_cases,
        shuffle=shuffle,
    )


def evaluate_model(
    model: torch.nn.Module,
    train_case_dataset,
    val_case_dataset,
    train_case_dataset_scaled,
    val_case_dataset_scaled,
    scalers,
    device: torch.device,
) -> tuple[dict, dict, np.ndarray]:
    train_pred_curve = predict_casewise_temperatures(model, train_case_dataset_scaled, scalers, device)
    val_pred_curve = predict_casewise_temperatures(model, val_case_dataset_scaled, scalers, device)

    train_point_metrics = evaluate_pointwise_predictions(train_case_dataset.temperatures, train_pred_curve)
    val_point_metrics = evaluate_pointwise_predictions(val_case_dataset.temperatures, val_pred_curve)
    train_energy_metrics, train_pred_energy = evaluate_case_energy_predictions(train_case_dataset, train_pred_curve)
    val_energy_metrics, val_pred_energy = evaluate_case_energy_predictions(val_case_dataset, val_pred_curve)
    train_rank_metrics = ranking_metrics(train_case_dataset.energies_mj, train_pred_energy)
    val_rank_metrics = ranking_metrics(val_case_dataset.energies_mj, val_pred_energy)

    train_row = metrics_row("unused", "unused", "train", train_point_metrics, train_energy_metrics, train_rank_metrics)
    val_row = metrics_row("unused", "unused", "val", val_point_metrics, val_energy_metrics, val_rank_metrics)
    return train_row, val_row, val_pred_energy


def build_loss(
    stage_cfg: StageConfig,
    trunk_times_hours: torch.Tensor,
    scalers,
    energy_scale_mj: float,
) -> EnergyConsistencyLoss:
    return EnergyConsistencyLoss(
        trunk_times_hours=trunk_times_hours,
        temperature_mean=float(scalers.temperature_scaler.mean_[0]),
        temperature_scale=float(scalers.temperature_scaler.scale_[0]),
        temperature_weight=stage_cfg.temperature_weight,
        energy_weight=stage_cfg.energy_weight,
        smoothness_weight=stage_cfg.smoothness_weight,
        temperature_loss=stage_cfg.temperature_loss,
        energy_loss=stage_cfg.energy_loss,
        energy_scale_mj=energy_scale_mj,
        relative_energy_epsilon_mj=stage_cfg.relative_energy_epsilon_mj,
    )


def run_stage(
    *,
    model: ECDeepONet,
    stage_cfg: StageConfig,
    stage_dir: Path,
    train_case_dataset,
    val_case_dataset,
    train_case_dataset_scaled,
    val_case_dataset_scaled,
    scalers,
    trunk_inputs_scaled: torch.Tensor,
    trunk_times_hours: torch.Tensor,
    energy_scale_mj: float,
    device: torch.device,
) -> tuple[TrainingHistory, dict, dict, np.ndarray]:
    set_trainable_prefixes(model, stage_cfg.trainable_prefixes)
    loss_fn = build_loss(stage_cfg, trunk_times_hours, scalers, energy_scale_mj)
    train_loader = build_loader(train_case_dataset_scaled, stage_cfg.batch_size_cases, shuffle=True)
    val_loader = build_loader(val_case_dataset_scaled, stage_cfg.batch_size_cases, shuffle=False)

    history, _ = train_casewise_model_with_custom_loss(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        trunk_inputs=trunk_inputs_scaled,
        loss_fn=loss_fn,
        device=device,
        epochs=stage_cfg.epochs,
        learning_rate=stage_cfg.learning_rate,
        weight_decay=stage_cfg.weight_decay,
        grad_clip_norm=stage_cfg.grad_clip_norm,
        scheduler_patience=stage_cfg.scheduler_patience,
        scheduler_factor=stage_cfg.scheduler_factor,
        early_stopping_patience=stage_cfg.early_stopping_patience,
        min_delta=stage_cfg.min_delta,
        monitor_metric=stage_cfg.monitor_metric,
    )

    save_history(history, stage_dir / "history.csv")
    torch.save(model.state_dict(), stage_dir / "model.pt")

    train_row, val_row, val_pred_energy = evaluate_model(
        model=model,
        train_case_dataset=train_case_dataset,
        val_case_dataset=val_case_dataset,
        train_case_dataset_scaled=train_case_dataset_scaled,
        val_case_dataset_scaled=val_case_dataset_scaled,
        scalers=scalers,
        device=device,
    )
    return history, train_row, val_row, val_pred_energy


def default_stage1_configs() -> list[Stage1Config]:
    return [
        Stage1Config(
            name="s1_base",
            latent_dim=96,
            branch_hidden_dims=(160, 160),
            trunk_hidden_dims=(128, 96),
            refine_hidden_dims=(128, 96),
            num_frequencies=0,
            max_frequency=1.0,
            dropout=0.0,
            use_layernorm=False,
            stage=StageConfig(
                name="pretrain",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=8e-4,
                weight_decay=1e-5,
                batch_size_cases=12,
                epochs=320,
                grad_clip_norm=1.0,
                scheduler_patience=20,
                scheduler_factor=0.5,
                early_stopping_patience=70,
                min_delta=1e-5,
                monitor_metric="temp",
            ),
        ),
        Stage1Config(
            name="s1_base_batch8",
            latent_dim=96,
            branch_hidden_dims=(160, 160),
            trunk_hidden_dims=(128, 96),
            refine_hidden_dims=(128, 96),
            num_frequencies=0,
            max_frequency=1.0,
            dropout=0.0,
            use_layernorm=False,
            stage=StageConfig(
                name="pretrain",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=6e-4,
                weight_decay=1e-5,
                batch_size_cases=8,
                epochs=360,
                grad_clip_norm=1.0,
                scheduler_patience=24,
                scheduler_factor=0.5,
                early_stopping_patience=80,
                min_delta=1e-5,
                monitor_metric="temp",
            ),
        ),
        Stage1Config(
            name="s1_wide",
            latent_dim=128,
            branch_hidden_dims=(192, 192),
            trunk_hidden_dims=(128, 128),
            refine_hidden_dims=(128, 128),
            num_frequencies=0,
            max_frequency=1.0,
            dropout=0.0,
            use_layernorm=False,
            stage=StageConfig(
                name="pretrain",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=7e-4,
                weight_decay=1e-5,
                batch_size_cases=12,
                epochs=360,
                grad_clip_norm=1.0,
                scheduler_patience=24,
                scheduler_factor=0.5,
                early_stopping_patience=80,
                min_delta=1e-5,
                monitor_metric="temp",
            ),
        ),
        Stage1Config(
            name="s1_layernorm",
            latent_dim=96,
            branch_hidden_dims=(160, 160),
            trunk_hidden_dims=(128, 96),
            refine_hidden_dims=(128, 96),
            num_frequencies=0,
            max_frequency=1.0,
            dropout=0.0,
            use_layernorm=True,
            stage=StageConfig(
                name="pretrain",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=8e-4,
                weight_decay=1e-5,
                batch_size_cases=12,
                epochs=320,
                grad_clip_norm=1.0,
                scheduler_patience=20,
                scheduler_factor=0.5,
                early_stopping_patience=70,
                min_delta=1e-5,
                monitor_metric="temp",
            ),
        ),
        Stage1Config(
            name="s1_soft_fourier",
            latent_dim=96,
            branch_hidden_dims=(160, 160),
            trunk_hidden_dims=(128, 96),
            refine_hidden_dims=(128, 96),
            num_frequencies=4,
            max_frequency=4.0,
            dropout=0.0,
            use_layernorm=False,
            stage=StageConfig(
                name="pretrain",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=8e-4,
                weight_decay=1e-5,
                batch_size_cases=12,
                epochs=320,
                grad_clip_norm=1.0,
                scheduler_patience=20,
                scheduler_factor=0.5,
                early_stopping_patience=70,
                min_delta=1e-5,
                monitor_metric="temp",
            ),
        ),
    ]


def default_stage2_configs() -> list[StageConfig]:
    return [
        StageConfig(
            name="ft_refine_scaled_002",
            temperature_weight=1.0,
            energy_weight=0.02,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="scaled_mse",
            learning_rate=2e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=140,
            grad_clip_norm=0.8,
            scheduler_patience=12,
            scheduler_factor=0.5,
            early_stopping_patience=30,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
        ),
        StageConfig(
            name="ft_refine_scaled_005",
            temperature_weight=1.0,
            energy_weight=0.05,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="scaled_mse",
            learning_rate=2e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=140,
            grad_clip_norm=0.8,
            scheduler_patience=12,
            scheduler_factor=0.5,
            early_stopping_patience=30,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
        ),
        StageConfig(
            name="ft_refine_relative_005",
            temperature_weight=1.0,
            energy_weight=0.05,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="relative_mse",
            learning_rate=2e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=140,
            grad_clip_norm=0.8,
            scheduler_patience=12,
            scheduler_factor=0.5,
            early_stopping_patience=30,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
        ),
        StageConfig(
            name="ft_all_scaled_001",
            temperature_weight=1.0,
            energy_weight=0.01,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="scaled_mse",
            learning_rate=1.5e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=120,
            grad_clip_norm=0.8,
            scheduler_patience=10,
            scheduler_factor=0.5,
            early_stopping_patience=24,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=None,
        ),
        StageConfig(
            name="ft_all_scaled_002",
            temperature_weight=1.0,
            energy_weight=0.02,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="scaled_mse",
            learning_rate=1.5e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=120,
            grad_clip_norm=0.8,
            scheduler_patience=10,
            scheduler_factor=0.5,
            early_stopping_patience=24,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=None,
        ),
        StageConfig(
            name="ft_all_relative_002",
            temperature_weight=1.0,
            energy_weight=0.02,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="relative_mse",
            learning_rate=1.5e-4,
            weight_decay=0.0,
            batch_size_cases=12,
            epochs=120,
            grad_clip_norm=0.8,
            scheduler_patience=10,
            scheduler_factor=0.5,
            early_stopping_patience=24,
            min_delta=1e-5,
            monitor_metric="total",
            trainable_prefixes=None,
        ),
    ]


def instantiate_model(cfg: Stage1Config, device: torch.device) -> ECDeepONet:
    return ECDeepONet(
        latent_dim=cfg.latent_dim,
        branch_hidden_dims=cfg.branch_hidden_dims,
        trunk_hidden_dims=cfg.trunk_hidden_dims,
        refine_hidden_dims=cfg.refine_hidden_dims,
        num_frequencies=cfg.num_frequencies,
        max_frequency=cfg.max_frequency,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
    ).to(device)


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    run_dir = ensure_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_main_dataframe(csv_path=args.dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=args.test_size, random_state=args.random_state)
    train_case_dataset = build_case_dataset(df_train)
    val_case_dataset = build_case_dataset(df_val)
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)
    val_case_dataset_scaled = transform_case_dataset(val_case_dataset, scalers)

    trunk_inputs_scaled = torch.from_numpy(train_case_dataset_scaled.trunk_inputs).float()
    trunk_times_hours = torch.from_numpy(train_case_dataset.trunk_inputs).float()
    energy_scale_mj = float(np.std(train_case_dataset.energies_mj.reshape(-1), ddof=0))

    stage1_configs = default_stage1_configs()
    stage2_configs = default_stage2_configs()

    write_json(run_dir / "stage1_configs.json", [asdict(cfg) for cfg in stage1_configs])
    write_json(run_dir / "stage2_configs.json", [asdict(cfg) for cfg in stage2_configs])

    all_rows: list[dict] = []
    experiment_details: list[dict] = []
    stage1_records: list[dict] = []

    for stage1_cfg in stage1_configs:
        exp_tag = stage1_cfg.name
        exp_dir = run_dir / exp_tag
        stage1_dir = exp_dir / "stage1_pretrain"
        stage1_dir.mkdir(parents=True, exist_ok=True)

        model = instantiate_model(stage1_cfg, device)
        history, train_row, val_row, val_pred_energy = run_stage(
            model=model,
            stage_cfg=stage1_cfg.stage,
            stage_dir=stage1_dir,
            train_case_dataset=train_case_dataset,
            val_case_dataset=val_case_dataset,
            train_case_dataset_scaled=train_case_dataset_scaled,
            val_case_dataset_scaled=val_case_dataset_scaled,
            scalers=scalers,
            trunk_inputs_scaled=trunk_inputs_scaled,
            trunk_times_hours=trunk_times_hours,
            energy_scale_mj=energy_scale_mj,
            device=device,
        )

        train_row.update({"tag": exp_tag, "phase": "stage1_pretrain", "score": overall_score(val_row)})
        val_row.update({"tag": exp_tag, "phase": "stage1_pretrain", "score": overall_score(val_row)})
        all_rows.extend([train_row, val_row])

        pd.DataFrame(
            {
                "case_id": val_case_dataset.case_ids,
                "true_energy_mj": val_case_dataset.energies_mj.reshape(-1),
                "pred_energy_mj": val_pred_energy.reshape(-1),
            }
        ).to_csv(stage1_dir / "val_energy.csv", index=False, encoding="utf-8")

        state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stage1_records.append(
            {
                "config": stage1_cfg,
                "state_dict": state_dict,
                "val_row": copy.deepcopy(val_row),
                "param_count": trainable_parameter_count(model),
                "trained_epochs": len(history.train_losses),
            }
        )

        experiment_details.append(
            {
                "tag": exp_tag,
                "phase": "stage1_pretrain",
                "config": asdict(stage1_cfg),
                "trained_epochs": len(history.train_losses),
                "param_count": trainable_parameter_count(model),
                "val_metrics": val_row,
            }
        )

    stage1_records.sort(key=lambda record: (record["val_row"]["score"], record["val_row"]["point_RMSE"], record["val_row"]["energy_RMSE"]))
    selected_stage1 = stage1_records[: max(1, args.top_k_stage1)]
    write_json(
        run_dir / "selected_stage1.json",
        [
            {
                "tag": record["config"].name,
                "val_metrics": record["val_row"],
                "trained_epochs": record["trained_epochs"],
            }
            for record in selected_stage1
        ],
    )

    best_final_val: dict | None = None
    best_final_tag: str | None = None

    for selected in selected_stage1:
        base_cfg = selected["config"]
        for stage2_cfg in stage2_configs:
            exp_tag = f"{base_cfg.name}__{stage2_cfg.name}"
            exp_dir = run_dir / exp_tag
            stage2_dir = exp_dir / "stage2_finetune"
            stage2_dir.mkdir(parents=True, exist_ok=True)

            model = instantiate_model(base_cfg, device)
            model.load_state_dict(selected["state_dict"])

            history, train_row, val_row, val_pred_energy = run_stage(
                model=model,
                stage_cfg=stage2_cfg,
                stage_dir=stage2_dir,
                train_case_dataset=train_case_dataset,
                val_case_dataset=val_case_dataset,
                train_case_dataset_scaled=train_case_dataset_scaled,
                val_case_dataset_scaled=val_case_dataset_scaled,
                scalers=scalers,
                trunk_inputs_scaled=trunk_inputs_scaled,
                trunk_times_hours=trunk_times_hours,
                energy_scale_mj=energy_scale_mj,
                device=device,
            )

            train_row.update({"tag": exp_tag, "phase": "stage2_finetune", "score": overall_score(val_row)})
            val_row.update({"tag": exp_tag, "phase": "stage2_finetune", "score": overall_score(val_row)})
            all_rows.extend([train_row, val_row])

            pd.DataFrame(
                {
                    "case_id": val_case_dataset.case_ids,
                    "true_energy_mj": val_case_dataset.energies_mj.reshape(-1),
                    "pred_energy_mj": val_pred_energy.reshape(-1),
                }
            ).to_csv(stage2_dir / "val_energy.csv", index=False, encoding="utf-8")

            detail = {
                "tag": exp_tag,
                "phase": "stage2_finetune",
                "base_stage1": base_cfg.name,
                "stage2_config": asdict(stage2_cfg),
                "trainable_prefixes": stage2_cfg.trainable_prefixes,
                "trainable_param_count": trainable_parameter_count(model),
                "trained_epochs": len(history.train_losses),
                "val_metrics": val_row,
            }
            experiment_details.append(detail)

            if best_final_val is None or overall_score(val_row) < overall_score(best_final_val):
                best_final_val = copy.deepcopy(val_row)
                best_final_tag = exp_tag

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    write_json(run_dir / "summary.json", experiment_details)
    write_json(
        run_dir / "best_config.json",
        {
            "best_tag": best_final_tag,
            "best_val": best_final_val,
        },
    )

    print(f"Run directory: {run_dir.resolve()}")
    print(summary_df.to_string(index=False))
    if best_final_val is not None:
        print("\nBest final configuration:")
        print(json.dumps({"best_tag": best_final_tag, "best_val": best_final_val}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
