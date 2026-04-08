from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_ec_focus_search import (
    Stage1Config,
    StageConfig,
    ensure_output_dir,
    instantiate_model,
    overall_score,
    run_stage,
    set_seed,
    trainable_parameter_count,
    write_json,
)
from src.data.datasets import build_case_dataset, fit_case_scalers, split_case_dataframe, transform_case_dataset
from src.data.io import load_main_dataframe


SEEDS = [7, 11, 23, 37, 123, 2026, 3407, 7777]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search the best random seed for the final EC-DeepONet setting.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("processed_energy_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ec_best_seed_search")
    parser.add_argument("--split-seed", type=int, default=123)
    parser.add_argument("--test-size", type=float, default=0.3)
    return parser.parse_args()


def build_stage1_config() -> Stage1Config:
    return Stage1Config(
        name="seed_s1_wide",
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
    )


def build_stage2_config() -> StageConfig:
    return StageConfig(
        name="seed_ft_refine_scaled_005",
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
    )


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    dataset_csv = args.dataset_csv
    split_seed = args.split_seed

    df = load_main_dataframe(csv_path=dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=args.test_size, random_state=split_seed)
    train_case_dataset = build_case_dataset(df_train)
    val_case_dataset = build_case_dataset(df_val)
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)
    val_case_dataset_scaled = transform_case_dataset(val_case_dataset, scalers)
    trunk_inputs_scaled = torch.from_numpy(train_case_dataset_scaled.trunk_inputs).float()
    trunk_times_hours = torch.from_numpy(train_case_dataset.trunk_inputs).float()
    energy_scale_mj = float(np.std(train_case_dataset.energies_mj.reshape(-1), ddof=0))

    stage1_cfg = build_stage1_config()
    stage2_cfg = build_stage2_config()
    write_json(output_dir / "stage1_config.json", asdict(stage1_cfg))
    write_json(output_dir / "stage2_config.json", asdict(stage2_cfg))

    summary_rows: list[dict] = []
    details: list[dict] = []
    best_final: dict | None = None
    best_tag: str | None = None

    for seed in SEEDS:
        set_seed(seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tag = f"seed_{seed}"

        stage1_dir = output_dir / tag / "stage1_pretrain"
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
        stage1_score = overall_score(val_row)
        train_row.update({"tag": tag, "phase": "stage1_pretrain", "score": stage1_score, "seed": seed})
        val_row.update({"tag": tag, "phase": "stage1_pretrain", "score": stage1_score, "seed": seed})
        summary_rows.extend([train_row, val_row])
        pd.DataFrame(
            {
                "case_id": val_case_dataset.case_ids,
                "true_energy_mj": val_case_dataset.energies_mj.reshape(-1),
                "pred_energy_mj": val_pred_energy.reshape(-1),
            }
        ).to_csv(stage1_dir / "val_energy.csv", index=False, encoding="utf-8")

        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stage2_dir = output_dir / tag / "stage2_finetune"
        stage2_dir.mkdir(parents=True, exist_ok=True)
        model = instantiate_model(stage1_cfg, device)
        model.load_state_dict(best_state_dict)
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
        final_score = overall_score(val_row)
        train_row.update({"tag": tag, "phase": "stage2_finetune", "score": final_score, "seed": seed})
        val_row.update({"tag": tag, "phase": "stage2_finetune", "score": final_score, "seed": seed})
        summary_rows.extend([train_row, val_row])
        details.append(
            {
                "tag": tag,
                "seed": seed,
                "stage2_config": asdict(stage2_cfg),
                "trainable_param_count": trainable_parameter_count(model),
                "val_metrics": val_row,
            }
        )
        pd.DataFrame(
            {
                "case_id": val_case_dataset.case_ids,
                "true_energy_mj": val_case_dataset.energies_mj.reshape(-1),
                "pred_energy_mj": val_pred_energy.reshape(-1),
            }
        ).to_csv(stage2_dir / "val_energy.csv", index=False, encoding="utf-8")

        if best_final is None or final_score < best_final["score"]:
            best_final = copy.deepcopy(val_row)
            best_tag = tag

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False, encoding="utf-8")
    write_json(output_dir / "summary.json", details)
    write_json(output_dir / "best_config.json", {"best_tag": best_tag, "best_val": best_final})

    print(f"Run directory: {output_dir.resolve()}")
    print(summary_df.to_string(index=False))
    print("\nBest final configuration:")
    print(json.dumps({"best_tag": best_tag, "best_val": best_final}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
