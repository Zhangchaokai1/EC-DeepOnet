from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

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


@dataclass(frozen=True)
class ECConfig:
    name: str
    latent_dim: int
    branch_hidden_dims: tuple[int, ...]
    trunk_hidden_dims: tuple[int, ...]
    refine_hidden_dims: tuple[int, ...]
    num_frequencies: int
    max_frequency: float
    dropout: float
    use_layernorm: bool
    temperature_weight: float
    energy_weight: float
    smoothness_weight: float
    temperature_loss: str
    learning_rate: float
    weight_decay: float
    batch_size_cases: int
    epochs: int
    grad_clip_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search EC-DeepONet configurations.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("processed_energy_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ec_search")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.3)
    return parser.parse_args()


def ensure_output_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history(history: TrainingHistory, path: Path) -> None:
    data = {
        "epoch": list(range(1, len(history.train_losses) + 1)),
        "train_loss": history.train_losses,
        "val_loss": history.val_losses,
    }
    if history.train_temp_losses:
        data["train_temp_loss"] = history.train_temp_losses
    if history.val_temp_losses:
        data["val_temp_loss"] = history.val_temp_losses
    if history.train_energy_losses:
        data["train_energy_loss"] = history.train_energy_losses
    if history.val_energy_losses:
        data["val_energy_loss"] = history.val_energy_losses
    if history.train_smooth_losses:
        data["train_smooth_loss"] = history.train_smooth_losses
    if history.val_smooth_losses:
        data["val_smooth_loss"] = history.val_smooth_losses
    pd.DataFrame(data).to_csv(path, index=False, encoding="utf-8")


def default_search_configs() -> list[ECConfig]:
    return [
        ECConfig(
            name="ec_cfg_01",
            latent_dim=96,
            branch_hidden_dims=(192, 192),
            trunk_hidden_dims=(128, 128),
            refine_hidden_dims=(128, 128),
            num_frequencies=16,
            max_frequency=8.0,
            dropout=0.0,
            use_layernorm=True,
            temperature_weight=1.0,
            energy_weight=0.10,
            smoothness_weight=0.0,
            temperature_loss="mse",
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size_cases=16,
            epochs=300,
            grad_clip_norm=1.0,
        ),
        ECConfig(
            name="ec_cfg_02",
            latent_dim=96,
            branch_hidden_dims=(192, 192),
            trunk_hidden_dims=(128, 128),
            refine_hidden_dims=(128, 128),
            num_frequencies=16,
            max_frequency=10.0,
            dropout=0.0,
            use_layernorm=True,
            temperature_weight=1.0,
            energy_weight=0.20,
            smoothness_weight=1e-5,
            temperature_loss="mse",
            learning_rate=8e-4,
            weight_decay=1e-5,
            batch_size_cases=16,
            epochs=320,
            grad_clip_norm=1.0,
        ),
        ECConfig(
            name="ec_cfg_03",
            latent_dim=128,
            branch_hidden_dims=(256, 256),
            trunk_hidden_dims=(160, 160),
            refine_hidden_dims=(160, 160),
            num_frequencies=24,
            max_frequency=10.0,
            dropout=0.0,
            use_layernorm=True,
            temperature_weight=1.0,
            energy_weight=0.20,
            smoothness_weight=1e-5,
            temperature_loss="mse",
            learning_rate=8e-4,
            weight_decay=1e-5,
            batch_size_cases=12,
            epochs=360,
            grad_clip_norm=1.0,
        ),
        ECConfig(
            name="ec_cfg_04",
            latent_dim=96,
            branch_hidden_dims=(192, 192),
            trunk_hidden_dims=(128, 128),
            refine_hidden_dims=(128, 128),
            num_frequencies=32,
            max_frequency=12.0,
            dropout=0.05,
            use_layernorm=True,
            temperature_weight=1.0,
            energy_weight=0.30,
            smoothness_weight=1e-5,
            temperature_loss="huber",
            learning_rate=7e-4,
            weight_decay=2e-5,
            batch_size_cases=16,
            epochs=360,
            grad_clip_norm=0.8,
        ),
        ECConfig(
            name="ec_cfg_05",
            latent_dim=128,
            branch_hidden_dims=(256, 192),
            trunk_hidden_dims=(160, 128),
            refine_hidden_dims=(160, 128),
            num_frequencies=16,
            max_frequency=6.0,
            dropout=0.0,
            use_layernorm=False,
            temperature_weight=1.0,
            energy_weight=0.50,
            smoothness_weight=5e-5,
            temperature_loss="mse",
            learning_rate=6e-4,
            weight_decay=2e-5,
            batch_size_cases=12,
            epochs=360,
            grad_clip_norm=0.8,
        ),
        ECConfig(
            name="ec_cfg_06",
            latent_dim=96,
            branch_hidden_dims=(160, 160),
            trunk_hidden_dims=(128, 96),
            refine_hidden_dims=(128, 96),
            num_frequencies=0,
            max_frequency=1.0,
            dropout=0.0,
            use_layernorm=True,
            temperature_weight=1.0,
            energy_weight=0.20,
            smoothness_weight=1e-5,
            temperature_loss="mse",
            learning_rate=1e-3,
            weight_decay=1e-5,
            batch_size_cases=16,
            epochs=300,
            grad_clip_norm=1.0,
        ),
    ]


def metrics_row(model_name: str, split: str, point_metrics: dict, energy_metrics: dict, rank_metrics: dict) -> dict:
    row = {"model_name": model_name, "split": split}
    row.update({f"point_{k}": v for k, v in point_metrics.items()})
    row.update({f"energy_{k}": v for k, v in energy_metrics.items()})
    row.update({f"rank_{k}": v for k, v in rank_metrics.items()})
    return row


def score_result(val_row: dict) -> tuple[float, float, float]:
    return (val_row["energy_RMSE"], val_row["point_RMSE"], -val_row["rank_SpearmanR"])


def main() -> None:
    args = parse_args()
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

    all_rows: list[dict] = []
    all_details: list[dict] = []
    best_val_row: dict | None = None
    best_config_name: str | None = None

    search_configs = default_search_configs()
    with (run_dir / "search_configs.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(cfg) for cfg in search_configs], f, ensure_ascii=False, indent=2)

    for cfg in search_configs:
        cfg_dir = run_dir / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        model = ECDeepONet(
            latent_dim=cfg.latent_dim,
            branch_hidden_dims=cfg.branch_hidden_dims,
            trunk_hidden_dims=cfg.trunk_hidden_dims,
            refine_hidden_dims=cfg.refine_hidden_dims,
            num_frequencies=cfg.num_frequencies,
            max_frequency=cfg.max_frequency,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        ).to(device)

        loss_fn = EnergyConsistencyLoss(
            trunk_times_hours=trunk_times_hours,
            temperature_mean=float(scalers.temperature_scaler.mean_[0]),
            temperature_scale=float(scalers.temperature_scaler.scale_[0]),
            temperature_weight=cfg.temperature_weight,
            energy_weight=cfg.energy_weight,
            smoothness_weight=cfg.smoothness_weight,
            temperature_loss=cfg.temperature_loss,
        )

        train_loader = DataLoader(
            CasewiseBranchDataset(
                train_case_dataset_scaled.branch_inputs,
                train_case_dataset_scaled.temperatures,
                train_case_dataset_scaled.energies_mj,
            ),
            batch_size=cfg.batch_size_cases,
            shuffle=True,
        )
        val_loader = DataLoader(
            CasewiseBranchDataset(
                val_case_dataset_scaled.branch_inputs,
                val_case_dataset_scaled.temperatures,
                val_case_dataset_scaled.energies_mj,
            ),
            batch_size=cfg.batch_size_cases,
            shuffle=False,
        )

        history, _ = train_casewise_model_with_custom_loss(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            trunk_inputs=trunk_inputs_scaled,
            loss_fn=loss_fn,
            device=device,
            epochs=cfg.epochs,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            grad_clip_norm=cfg.grad_clip_norm,
            scheduler_patience=20,
            scheduler_factor=0.5,
            early_stopping_patience=60,
            min_delta=1e-5,
        )
        save_history(history, cfg_dir / "history.csv")
        torch.save(model.state_dict(), cfg_dir / "model.pt")
        with (cfg_dir / "config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, ensure_ascii=False, indent=2)

        train_pred_curve = predict_casewise_temperatures(model, train_case_dataset_scaled, scalers, device)
        val_pred_curve = predict_casewise_temperatures(model, val_case_dataset_scaled, scalers, device)

        train_point_metrics = evaluate_pointwise_predictions(train_case_dataset.temperatures, train_pred_curve)
        val_point_metrics = evaluate_pointwise_predictions(val_case_dataset.temperatures, val_pred_curve)
        train_energy_metrics, train_pred_energy = evaluate_case_energy_predictions(train_case_dataset, train_pred_curve)
        val_energy_metrics, val_pred_energy = evaluate_case_energy_predictions(val_case_dataset, val_pred_curve)
        train_rank_metrics = ranking_metrics(train_case_dataset.energies_mj, train_pred_energy)
        val_rank_metrics = ranking_metrics(val_case_dataset.energies_mj, val_pred_energy)

        pd.DataFrame(
            {
                "case_id": val_case_dataset.case_ids,
                "true_energy_mj": val_case_dataset.energies_mj.reshape(-1),
                "pred_energy_mj": val_pred_energy.reshape(-1),
            }
        ).to_csv(cfg_dir / "val_energy.csv", index=False, encoding="utf-8")

        train_row = metrics_row(cfg.name, "train", train_point_metrics, train_energy_metrics, train_rank_metrics)
        val_row = metrics_row(cfg.name, "val", val_point_metrics, val_energy_metrics, val_rank_metrics)
        all_rows.extend([train_row, val_row])
        all_details.append(
            {
                "config": asdict(cfg),
                "train": train_row,
                "val": val_row,
                "best_epoch_count": len(history.train_losses),
            }
        )

        if best_val_row is None or score_result(val_row) < score_result(best_val_row):
            best_val_row = val_row
            best_config_name = cfg.name

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(run_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_details, f, ensure_ascii=False, indent=2)
    with (run_dir / "best_config.json").open("w", encoding="utf-8") as f:
        json.dump({"best_config_name": best_config_name, "best_val": best_val_row}, f, ensure_ascii=False, indent=2)

    print(f"Run directory: {run_dir.resolve()}")
    print(summary_df.to_string(index=False))
    print(f"\nBest config: {best_config_name}")
    if best_val_row is not None:
        print(json.dumps(best_val_row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

