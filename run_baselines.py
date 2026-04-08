from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.datasets import (
    CasewiseBranchDataset,
    PointwiseTemperatureDataset,
    build_case_dataset,
    build_point_dataset_from_cases,
    fit_case_scalers,
    split_case_dataframe,
    transform_case_dataset,
)
from src.data.io import load_main_dataframe
from src.eval.predict import (
    evaluate_case_energy_predictions,
    evaluate_pointwise_predictions,
    predict_casewise_temperatures,
    predict_casewise_temperatures_from_point_model,
    predict_pointwise_temperatures,
    ranking_metrics,
)
from src.models.deeponet import VanillaDeepONet
from src.models.mlp import MLP
from src.train.trainer import TrainingHistory, train_casewise_deeponet, train_pointwise_model


@dataclass
class BaselineRunConfig:
    dataset_csv: str
    output_dir: str
    epochs_mlp: int
    epochs_deeponet: int
    batch_size_points: int
    batch_size_cases: int
    random_state: int
    test_size: float
    learning_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline experiments for MLP and Vanilla DeepONet.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("processed_energy_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "baseline_runs")
    parser.add_argument("--epochs-mlp", type=int, default=300)
    parser.add_argument("--epochs-deeponet", type=int, default=300)
    parser.add_argument("--batch-size-points", type=int, default=1024)
    parser.add_argument("--batch-size-cases", type=int, default=16)
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def ensure_output_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_history(history: TrainingHistory, path: Path) -> None:
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_losses) + 1),
            "train_loss": history.train_losses,
            "val_loss": history.val_losses,
        }
    ).to_csv(path, index=False, encoding="utf-8")


def save_curve_predictions(
    case_dataset: object,
    pred_temperatures: np.ndarray,
    path: Path,
) -> None:
    rows = []
    for case_id, curve in zip(case_dataset.case_ids, pred_temperatures):
        for t, temp in zip(case_dataset.trunk_inputs.reshape(-1), np.asarray(curve).reshape(-1)):
            rows.append(
                {
                    "case_id": int(case_id),
                    "time_h": float(t),
                    "pred_temperature_c": float(temp),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def save_energy_predictions(
    case_dataset: object,
    pred_energies: np.ndarray,
    path: Path,
) -> None:
    pd.DataFrame(
        {
            "case_id": case_dataset.case_ids,
            "true_energy_mj": case_dataset.energies_mj.reshape(-1),
            "pred_energy_mj": pred_energies.reshape(-1),
        }
    ).to_csv(path, index=False, encoding="utf-8")


def run_mlp(
    train_points,
    val_points,
    train_case_dataset,
    val_case_dataset,
    train_case_dataset_scaled,
    val_case_dataset_scaled,
    scalers,
    device: torch.device,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    model = MLP().to(device)
    train_loader = DataLoader(
        PointwiseTemperatureDataset(train_points.inputs, train_points.temperatures),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        PointwiseTemperatureDataset(val_points.inputs, val_points.temperatures),
        batch_size=batch_size,
        shuffle=False,
    )
    history = train_pointwise_model(
        model,
        train_loader,
        val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    save_history(history, run_dir / "mlp_loss.csv")

    train_pred_c = predict_pointwise_temperatures(model, train_points, scalers, device)
    val_pred_c = predict_pointwise_temperatures(model, val_points, scalers, device)
    train_true_c = scalers.temperature_scaler.inverse_transform(train_points.temperatures)
    val_true_c = scalers.temperature_scaler.inverse_transform(val_points.temperatures)

    point_metrics_train = evaluate_pointwise_predictions(train_true_c, train_pred_c)
    point_metrics_val = evaluate_pointwise_predictions(val_true_c, val_pred_c)

    train_curve_pred_c = predict_casewise_temperatures_from_point_model(
        model, train_case_dataset_scaled, scalers, device
    )
    val_curve_pred_c = predict_casewise_temperatures_from_point_model(
        model, val_case_dataset_scaled, scalers, device
    )
    train_energy_metrics, train_pred_energy = evaluate_case_energy_predictions(train_case_dataset, train_curve_pred_c)
    val_energy_metrics, val_pred_energy = evaluate_case_energy_predictions(val_case_dataset, val_curve_pred_c)
    train_rank_metrics = ranking_metrics(train_case_dataset.energies_mj, train_pred_energy)
    val_rank_metrics = ranking_metrics(val_case_dataset.energies_mj, val_pred_energy)

    torch.save(model.state_dict(), run_dir / "mlp_model.pt")
    save_curve_predictions(val_case_dataset, val_curve_pred_c, run_dir / "mlp_val_curves.csv")
    save_energy_predictions(val_case_dataset, val_pred_energy, run_dir / "mlp_val_energy.csv")

    return {
        "model_name": "MLP",
        "point_metrics_train": point_metrics_train,
        "point_metrics_val": point_metrics_val,
        "energy_metrics_train": train_energy_metrics,
        "energy_metrics_val": val_energy_metrics,
        "ranking_metrics_train": train_rank_metrics,
        "ranking_metrics_val": val_rank_metrics,
    }


def run_deeponet(
    train_case_dataset,
    val_case_dataset,
    train_case_dataset_scaled,
    val_case_dataset_scaled,
    scalers,
    device: torch.device,
    run_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    model = VanillaDeepONet().to(device)
    train_loader = DataLoader(
        CasewiseBranchDataset(
            train_case_dataset_scaled.branch_inputs,
            train_case_dataset_scaled.temperatures,
            train_case_dataset_scaled.energies_mj,
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        CasewiseBranchDataset(
            val_case_dataset_scaled.branch_inputs,
            val_case_dataset_scaled.temperatures,
            val_case_dataset_scaled.energies_mj,
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    trunk_inputs = torch.from_numpy(train_case_dataset_scaled.trunk_inputs).float()
    history = train_casewise_deeponet(
        model,
        train_loader,
        val_loader,
        trunk_inputs=trunk_inputs,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    save_history(history, run_dir / "deeponet_loss.csv")

    train_curve_pred_c = predict_casewise_temperatures(model, train_case_dataset_scaled, scalers, device)
    val_curve_pred_c = predict_casewise_temperatures(model, val_case_dataset_scaled, scalers, device)
    train_true_curve_c = train_case_dataset.temperatures
    val_true_curve_c = val_case_dataset.temperatures

    point_metrics_train = evaluate_pointwise_predictions(train_true_curve_c, train_curve_pred_c)
    point_metrics_val = evaluate_pointwise_predictions(val_true_curve_c, val_curve_pred_c)

    train_energy_metrics, train_pred_energy = evaluate_case_energy_predictions(train_case_dataset, train_curve_pred_c)
    val_energy_metrics, val_pred_energy = evaluate_case_energy_predictions(val_case_dataset, val_curve_pred_c)
    train_rank_metrics = ranking_metrics(train_case_dataset.energies_mj, train_pred_energy)
    val_rank_metrics = ranking_metrics(val_case_dataset.energies_mj, val_pred_energy)

    torch.save(model.state_dict(), run_dir / "deeponet_model.pt")
    save_curve_predictions(val_case_dataset, val_curve_pred_c, run_dir / "deeponet_val_curves.csv")
    save_energy_predictions(val_case_dataset, val_pred_energy, run_dir / "deeponet_val_energy.csv")

    return {
        "model_name": "VanillaDeepONet",
        "point_metrics_train": point_metrics_train,
        "point_metrics_val": point_metrics_val,
        "energy_metrics_train": train_energy_metrics,
        "energy_metrics_val": val_energy_metrics,
        "ranking_metrics_train": train_rank_metrics,
        "ranking_metrics_val": val_rank_metrics,
    }


def main() -> None:
    args = parse_args()
    run_dir = ensure_output_dir(args.output_dir)
    config = BaselineRunConfig(
        dataset_csv=str(args.dataset_csv),
        output_dir=str(run_dir),
        epochs_mlp=args.epochs_mlp,
        epochs_deeponet=args.epochs_deeponet,
        batch_size_points=args.batch_size_points,
        batch_size_cases=args.batch_size_cases,
        random_state=args.random_state,
        test_size=args.test_size,
        learning_rate=args.learning_rate,
    )
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(config), f, ensure_ascii=False, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_main_dataframe(csv_path=args.dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=args.test_size, random_state=args.random_state)
    train_case_dataset = build_case_dataset(df_train)
    val_case_dataset = build_case_dataset(df_val)
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)
    val_case_dataset_scaled = transform_case_dataset(val_case_dataset, scalers)
    train_points_scaled = build_point_dataset_from_cases(train_case_dataset, scalers)
    val_points_scaled = build_point_dataset_from_cases(val_case_dataset, scalers)

    mlp_results = run_mlp(
        train_points=train_points_scaled,
        val_points=val_points_scaled,
        train_case_dataset=train_case_dataset,
        val_case_dataset=val_case_dataset,
        train_case_dataset_scaled=train_case_dataset_scaled,
        val_case_dataset_scaled=val_case_dataset_scaled,
        scalers=scalers,
        device=device,
        run_dir=run_dir,
        epochs=args.epochs_mlp,
        batch_size=args.batch_size_points,
        learning_rate=args.learning_rate,
    )
    deeponet_results = run_deeponet(
        train_case_dataset=train_case_dataset,
        val_case_dataset=val_case_dataset,
        train_case_dataset_scaled=train_case_dataset_scaled,
        val_case_dataset_scaled=val_case_dataset_scaled,
        scalers=scalers,
        device=device,
        run_dir=run_dir,
        epochs=args.epochs_deeponet,
        batch_size=args.batch_size_cases,
        learning_rate=args.learning_rate,
    )

    all_results = [mlp_results, deeponet_results]
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    rows = []
    for result in all_results:
        for split in ["train", "val"]:
            row = {"model_name": result["model_name"], "split": split}
            row.update({f"point_{k}": v for k, v in result[f"point_metrics_{split}"].items()})
            row.update({f"energy_{k}": v for k, v in result[f"energy_metrics_{split}"].items()})
            row.update({f"rank_{k}": v for k, v in result[f"ranking_metrics_{split}"].items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(run_dir / "summary_metrics.csv", index=False, encoding="utf-8")

    print(f"Run directory: {run_dir.resolve()}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
