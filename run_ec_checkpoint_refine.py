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
    trainable_parameter_count,
    write_json,
)
from src.data.datasets import build_case_dataset, fit_case_scalers, split_case_dataframe, transform_case_dataset
from src.data.io import load_main_dataframe


DEFAULT_FOCUS_BASE = Path("outputs") / "ec_focus_search"
DEFAULT_STAGE1_TAG = "s1_wide"
DEFAULT_STAGE2_TAG = "s1_wide__ft_refine_scaled_005"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine EC-DeepONet checkpoints based on focused search outputs.")
    parser.add_argument("--dataset-csv", type=Path, default=Path("processed_energy_dataset.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs") / "ec_checkpoint_refine")
    parser.add_argument("--focus-run-dir", type=Path, default=None, help="Directory produced by run_ec_focus_search.py.")
    parser.add_argument("--base-stage1-tag", type=str, default=DEFAULT_STAGE1_TAG)
    parser.add_argument("--base-stage2-tag", type=str, default=DEFAULT_STAGE2_TAG)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--split-seed", type=int, default=123)
    return parser.parse_args()


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_base_checkpoints(
    focus_run_dir: Path,
    base_stage1_tag: str,
    base_stage2_tag: str,
) -> tuple[Path, Path]:
    stage1_ckpt = focus_run_dir / base_stage1_tag / "stage1_pretrain" / "model.pt"
    stage2_ckpt = focus_run_dir / base_stage2_tag / "stage2_finetune" / "model.pt"
    if not stage1_ckpt.exists():
        raise FileNotFoundError(f"Missing stage1 checkpoint: {stage1_ckpt}")
    if not stage2_ckpt.exists():
        raise FileNotFoundError(f"Missing stage2 checkpoint: {stage2_ckpt}")
    return stage1_ckpt, stage2_ckpt


def build_model_config() -> Stage1Config:
    return Stage1Config(
        name="ckpt_s1_wide",
        latent_dim=128,
        branch_hidden_dims=(192, 192),
        trunk_hidden_dims=(128, 128),
        refine_hidden_dims=(128, 128),
        num_frequencies=0,
        max_frequency=1.0,
        dropout=0.0,
        use_layernorm=False,
        stage=StageConfig(
            name="unused",
            temperature_weight=1.0,
            energy_weight=0.0,
            smoothness_weight=0.0,
            temperature_loss="mse",
            energy_loss="scaled_mse",
            learning_rate=7e-4,
            weight_decay=1e-5,
            batch_size_cases=12,
            epochs=1,
            grad_clip_norm=1.0,
            scheduler_patience=1,
            scheduler_factor=0.5,
            early_stopping_patience=1,
            min_delta=1e-5,
            monitor_metric="temp",
        ),
    )


def build_stage_configs(base_stage1_ckpt: Path, base_stage2_ckpt: Path) -> list[tuple[str, Path, StageConfig]]:
    return [
        (
            "from_stage1_refine_scaled_005_lr1e4_b8",
            base_stage1_ckpt,
            StageConfig(
                name="from_stage1_refine_scaled_005_lr1e4_b8",
                temperature_weight=1.0,
                energy_weight=0.05,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=1e-4,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=260,
                grad_clip_norm=0.8,
                scheduler_patience=18,
                scheduler_factor=0.5,
                early_stopping_patience=52,
                min_delta=1e-5,
                monitor_metric="total",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
        (
            "from_stage1_refine_scaled_006_lr1e4_b8",
            base_stage1_ckpt,
            StageConfig(
                name="from_stage1_refine_scaled_006_lr1e4_b8",
                temperature_weight=1.0,
                energy_weight=0.06,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=1e-4,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=260,
                grad_clip_norm=0.8,
                scheduler_patience=18,
                scheduler_factor=0.5,
                early_stopping_patience=52,
                min_delta=1e-5,
                monitor_metric="total",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
        (
            "from_stage2_temp_polish_lr5e5_b8",
            base_stage2_ckpt,
            StageConfig(
                name="from_stage2_temp_polish_lr5e5_b8",
                temperature_weight=1.0,
                energy_weight=0.0,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=5e-5,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=220,
                grad_clip_norm=0.8,
                scheduler_patience=16,
                scheduler_factor=0.5,
                early_stopping_patience=44,
                min_delta=1e-5,
                monitor_metric="temp",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
        (
            "from_stage2_mixed_scaled_002_lr5e5_b8",
            base_stage2_ckpt,
            StageConfig(
                name="from_stage2_mixed_scaled_002_lr5e5_b8",
                temperature_weight=1.0,
                energy_weight=0.02,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=5e-5,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=220,
                grad_clip_norm=0.8,
                scheduler_patience=16,
                scheduler_factor=0.5,
                early_stopping_patience=44,
                min_delta=1e-5,
                monitor_metric="total",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
        (
            "from_stage2_mixed_scaled_005_lr5e5_b8",
            base_stage2_ckpt,
            StageConfig(
                name="from_stage2_mixed_scaled_005_lr5e5_b8",
                temperature_weight=1.0,
                energy_weight=0.05,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="scaled_mse",
                learning_rate=5e-5,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=220,
                grad_clip_norm=0.8,
                scheduler_patience=16,
                scheduler_factor=0.5,
                early_stopping_patience=44,
                min_delta=1e-5,
                monitor_metric="total",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
        (
            "from_stage2_mixed_relative_002_lr5e5_b8",
            base_stage2_ckpt,
            StageConfig(
                name="from_stage2_mixed_relative_002_lr5e5_b8",
                temperature_weight=1.0,
                energy_weight=0.02,
                smoothness_weight=0.0,
                temperature_loss="mse",
                energy_loss="relative_mse",
                learning_rate=5e-5,
                weight_decay=0.0,
                batch_size_cases=8,
                epochs=220,
                grad_clip_norm=0.8,
                scheduler_patience=16,
                scheduler_factor=0.5,
                early_stopping_patience=44,
                min_delta=1e-5,
                monitor_metric="total",
                trainable_prefixes=("branch_refine", "trunk_refine", "context_gate", "bias"),
            ),
        ),
    ]


def main() -> None:
    args = parse_args()
    focus_run_dir = args.focus_run_dir or latest_run_dir(DEFAULT_FOCUS_BASE)
    base_stage1_ckpt, base_stage2_ckpt = resolve_base_checkpoints(
        focus_run_dir=focus_run_dir,
        base_stage1_tag=args.base_stage1_tag,
        base_stage2_tag=args.base_stage2_tag,
    )

    output_dir = ensure_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_main_dataframe(csv_path=args.dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=args.test_size, random_state=args.split_seed)
    train_case_dataset = build_case_dataset(df_train)
    val_case_dataset = build_case_dataset(df_val)
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)
    val_case_dataset_scaled = transform_case_dataset(val_case_dataset, scalers)
    trunk_inputs_scaled = torch.from_numpy(train_case_dataset_scaled.trunk_inputs).float()
    trunk_times_hours = torch.from_numpy(train_case_dataset.trunk_inputs).float()
    energy_scale_mj = float(np.std(train_case_dataset.energies_mj.reshape(-1), ddof=0))

    model_cfg = build_model_config()
    stage_entries = build_stage_configs(base_stage1_ckpt=base_stage1_ckpt, base_stage2_ckpt=base_stage2_ckpt)
    write_json(output_dir / "model_config.json", asdict(model_cfg))
    write_json(
        output_dir / "stage_configs.json",
        [
            {
                "tag": tag,
                "base_checkpoint": str(base_checkpoint),
                "config": asdict(stage_cfg),
            }
            for tag, base_checkpoint, stage_cfg in stage_entries
        ],
    )
    write_json(
        output_dir / "input_checkpoints.json",
        {
            "focus_run_dir": str(focus_run_dir),
            "base_stage1_checkpoint": str(base_stage1_ckpt),
            "base_stage2_checkpoint": str(base_stage2_ckpt),
        },
    )

    summary_rows: list[dict] = []
    details: list[dict] = []
    best_final: dict | None = None
    best_tag: str | None = None

    for tag, base_checkpoint, stage_cfg in stage_entries:
        run_dir = output_dir / tag / "refine"
        run_dir.mkdir(parents=True, exist_ok=True)
        model = instantiate_model(model_cfg, device)
        model.load_state_dict(torch.load(base_checkpoint, map_location=device))
        history, train_row, val_row, val_pred_energy = run_stage(
            model=model,
            stage_cfg=stage_cfg,
            stage_dir=run_dir,
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
        score = overall_score(val_row)
        train_row.update({"tag": tag, "phase": "refine", "score": score})
        val_row.update({"tag": tag, "phase": "refine", "score": score})
        summary_rows.extend([train_row, val_row])
        details.append(
            {
                "tag": tag,
                "base_checkpoint": str(base_checkpoint),
                "config": asdict(stage_cfg),
                "epochs": len(history.train_losses),
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
        ).to_csv(run_dir / "val_energy.csv", index=False, encoding="utf-8")
        if best_final is None or score < best_final["score"]:
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
