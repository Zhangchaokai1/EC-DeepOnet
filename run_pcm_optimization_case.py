from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import qmc
from sklearn.metrics import pairwise_distances

from simulation import solve_with_timeout
from src.data.datasets import build_case_dataset, fit_case_scalers, split_case_dataframe, transform_case_dataset
from src.data.energy import compute_standardized_energy_mj
from src.data.io import load_main_dataframe, load_timeseries_npz
from src.data.schema import DEFAULT_CSV_PATH, DEFAULT_TIME_SERIES_DIR, INPUT_COLUMNS
from src.models.ec_deeponet import ECDeepONet


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_BASE = PROJECT_ROOT / "outputs" / "optimization_case"
BEST_SEED_BASE = PROJECT_ROOT / "outputs" / "ec_best_seed_search"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "DeepOnet.mph"


@dataclass
class OptimizationConfig:
    dataset_csv: str = str(PROJECT_ROOT / "processed_energy_dataset.csv")
    best_seed_dir: str = ""
    model_path: str = str(DEFAULT_MODEL_PATH)
    expert_count: int = 5
    candidate_count: int = 25000
    boxplot_sample_count: int = 2500
    verify_top_k: int = 3
    random_state: int = 20260314
    test_size: float = 0.3
    timeout_seconds: int = 180
    diversity_distance_threshold: float = 0.85
    use_conservative_score: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ensemble-assisted PCM material optimization.")
    parser.add_argument("--dataset-csv", type=Path, default=PROJECT_ROOT / "processed_energy_dataset.csv")
    parser.add_argument("--best-seed-dir", type=Path, default=None, help="Directory produced by run_ec_best_seed_search.py.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the COMSOL .mph model.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--expert-count", type=int, default=5)
    parser.add_argument("--candidate-count", type=int, default=25000)
    parser.add_argument("--boxplot-sample-count", type=int, default=2500)
    parser.add_argument("--verify-top-k", type=int, default=3)
    parser.add_argument("--random-state", type=int, default=20260314)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--diversity-distance-threshold", type=float, default=0.85)
    parser.add_argument("--disable-conservative-score", action="store_true")
    return parser.parse_args()


def ensure_output_dir(base_dir: Path) -> Path:
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under: {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def build_config(args: argparse.Namespace) -> OptimizationConfig:
    best_seed_dir = args.best_seed_dir
    if best_seed_dir is None:
        best_seed_dir = latest_run_dir(BEST_SEED_BASE)
    return OptimizationConfig(
        dataset_csv=str(args.dataset_csv),
        best_seed_dir=str(best_seed_dir),
        model_path=str(args.model_path),
        expert_count=args.expert_count,
        candidate_count=args.candidate_count,
        boxplot_sample_count=args.boxplot_sample_count,
        verify_top_k=args.verify_top_k,
        random_state=args.random_state,
        test_size=args.test_size,
        timeout_seconds=args.timeout_seconds,
        diversity_distance_threshold=args.diversity_distance_threshold,
        use_conservative_score=not args.disable_conservative_score,
    )


def instantiate_model(device: torch.device) -> ECDeepONet:
    return ECDeepONet(
        latent_dim=128,
        branch_hidden_dims=(192, 192),
        trunk_hidden_dims=(128, 128),
        refine_hidden_dims=(128, 128),
        num_frequencies=0,
        max_frequency=1.0,
        dropout=0.0,
        use_layernorm=False,
    ).to(device)


def select_expert_models(best_seed_dir: Path, expert_count: int) -> list[tuple[str, Path]]:
    best_seed_summary = best_seed_dir / "summary_metrics.csv"
    summary_df = pd.read_csv(best_seed_summary)
    val_df = summary_df[(summary_df["phase"] == "stage2_finetune") & (summary_df["split"] == "val")].copy()
    if val_df.empty:
        raise RuntimeError(f"No stage2 validation rows found in {best_seed_summary}")
    selected = val_df.sort_values("score").head(expert_count)
    seed_models: list[tuple[str, Path]] = []
    for _, row in selected.iterrows():
        tag = str(row["tag"])
        model_path = best_seed_dir / tag / "stage2_finetune" / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing expert model checkpoint: {model_path}")
        seed_models.append((tag, model_path))
    return seed_models


def sample_candidates_lhs(n_samples: int, random_state: int) -> pd.DataFrame:
    sampler = qmc.LatinHypercube(d=10, seed=random_state)
    u = sampler.random(n_samples)

    rho_l = 750.0 + u[:, 0] * 100.0
    rho_s_min = np.maximum(rho_l + 0.1, 800.0)
    rho_s = rho_s_min + u[:, 1] * (950.0 - rho_s_min)

    cp_s = 1800.0 + u[:, 2] * 600.0
    cl_l_min = np.maximum(cp_s + 0.1, 2000.0)
    cl_l = cl_l_min + u[:, 3] * (2600.0 - cl_l_min)

    k_l = 0.140 + u[:, 4] * 0.060
    k_s_min = np.maximum(k_l + 1e-4, 0.200)
    k_s = k_s_min + u[:, 5] * (0.250 - k_s_min)

    lf = 150.0 + u[:, 6] * 100.0
    wt = 0.10 + u[:, 7] * 0.15
    tm = 20.0 + u[:, 8] * 5.0
    dt = 2.0 + u[:, 9] * 5.0

    return pd.DataFrame(
        {
            "Input_rho_s": rho_s,
            "Input_rho_l": rho_l,
            "Input_cp_s": cp_s,
            "Input_cl_l": cl_l,
            "Input_k_s": k_s,
            "Input_k_l": k_l,
            "Input_Lf": lf,
            "Input_wt": wt,
            "Input_Tm": tm,
            "Input_dT": dt,
        }
    )


def predict_energy_curve_ensemble(
    candidate_df: pd.DataFrame,
    trunk_inputs_scaled: np.ndarray,
    time_scaler_mean: float,
    time_scaler_scale: float,
    temp_scaler_mean: float,
    temp_scaler_scale: float,
    branch_scaler,
    seed_models: list[tuple[str, Path]],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    branch_scaled = branch_scaler.transform(candidate_df[INPUT_COLUMNS].to_numpy(dtype=np.float64)).astype(np.float32)
    trunk_scaled = trunk_inputs_scaled.astype(np.float32)
    trunk_tensor = torch.from_numpy(trunk_scaled).float().to(device)
    trunk_times = trunk_scaled.reshape(-1) * time_scaler_scale + time_scaler_mean

    ensemble_energy = []
    expert_tags = [tag for tag, _ in seed_models]
    with torch.no_grad():
        for _, model_path in seed_models:
            model = instantiate_model(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            energy_list = []
            for start in range(0, len(branch_scaled), batch_size):
                end = min(start + batch_size, len(branch_scaled))
                branch_tensor = torch.from_numpy(branch_scaled[start:end]).float().to(device)
                pred_scaled = model(branch_tensor, trunk_tensor).cpu().numpy()
                pred_temp = pred_scaled * temp_scaler_scale + temp_scaler_mean
                for curve in pred_temp:
                    energy_list.append(
                        compute_standardized_energy_mj(
                            times_hours=trunk_times,
                            temperatures_c=np.asarray(curve).reshape(-1),
                            t_start_hours=float(trunk_times[0]),
                            t_end_hours=float(trunk_times[-1]),
                            n_points=len(trunk_times),
                        )
                    )
            ensemble_energy.append(np.asarray(energy_list, dtype=np.float64))
    energies = np.stack(ensemble_energy, axis=1)
    return energies.mean(axis=1), energies.std(axis=1, ddof=0), energies, expert_tags


def select_diverse_candidates(
    candidate_df: pd.DataFrame,
    train_branch_scaled: np.ndarray,
    top_k: int,
    distance_threshold: float,
) -> pd.DataFrame:
    candidate_scaled = candidate_df[[f"{col}_scaled" for col in INPUT_COLUMNS]].to_numpy(dtype=np.float64)
    chosen_indices: list[int] = []
    for idx in candidate_df.sort_values("selection_score", ascending=False).index:
        if not chosen_indices:
            chosen_indices.append(int(idx))
            continue
        selected = candidate_scaled[chosen_indices]
        dist = np.linalg.norm(candidate_scaled[idx] - selected, axis=1)
        if float(dist.min()) >= distance_threshold:
            chosen_indices.append(int(idx))
        if len(chosen_indices) >= top_k:
            break

    if len(chosen_indices) < top_k:
        for idx in candidate_df.sort_values("selection_score", ascending=False).index:
            if int(idx) not in chosen_indices:
                chosen_indices.append(int(idx))
            if len(chosen_indices) >= top_k:
                break
    return candidate_df.loc[chosen_indices].copy().reset_index(drop=True)


def simulate_candidate(
    candidate: pd.Series,
    output_dir: Path,
    timeout_seconds: int,
    model_path: Path,
) -> dict:
    ts_name = f"candidate_{int(candidate['rank']):02d}.npz"
    ts_path = output_dir / ts_name
    csv_path = output_dir / ts_name.replace(".npz", ".csv")
    params = {
        "rho_s": f"{candidate['Input_rho_s']}[kg/m^3]",
        "rho_l": f"{candidate['Input_rho_l']}[kg/m^3]",
        "cp_s": f"{candidate['Input_cp_s']}[J/(kg*K)]",
        "cl_l": f"{candidate['Input_cl_l']}[J/(kg*K)]",
        "k_s": f"{candidate['Input_k_s']}[W/(m*K)]",
        "k_l": f"{candidate['Input_k_l']}[W/(m*K)]",
        "Lf": f"{candidate['Input_Lf']}[kJ/kg]",
        "wt": f"{candidate['Input_wt']}",
        "Tm": f"{candidate['Input_Tm']}[degC]",
        "dT": f"{candidate['Input_dT']}[K]",
    }
    result = solve_with_timeout(
        model_path=str(model_path),
        params=params,
        ts_path=str(ts_path),
        csv_path=str(csv_path),
        timeout_seconds=timeout_seconds,
    )
    if result.get("status") != "success":
        return {
            "status": result.get("status"),
            "elapsed": result.get("elapsed"),
            "true_energy_mj": np.nan,
            "true_tout_c": np.nan,
            "pred_error_mj": np.nan,
        }

    times, temps = load_timeseries_npz(ts_path)
    true_energy = compute_standardized_energy_mj(
        times_hours=times,
        temperatures_c=temps,
        t_start_hours=1.0,
        t_end_hours=100.0,
        n_points=500,
    )
    return {
        "status": "success",
        "elapsed": result.get("elapsed"),
        "true_energy_mj": float(true_energy),
        "true_tout_c": float(result["tout"]),
    }


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    run_dir = ensure_output_dir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_models = select_expert_models(Path(cfg.best_seed_dir), cfg.expert_count)

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(cfg), file, ensure_ascii=False, indent=2)
    with (run_dir / "selected_experts.json").open("w", encoding="utf-8") as file:
        json.dump(
            [
                {"tag": tag, "model_path": str(model_path)}
                for tag, model_path in seed_models
            ],
            file,
            ensure_ascii=False,
            indent=2,
        )

    df = load_main_dataframe(csv_path=cfg.dataset_csv, success_only=False, reset_index=True)
    df_train, df_val = split_case_dataframe(df, test_size=cfg.test_size, random_state=123)
    train_case_dataset = build_case_dataset(df_train)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, fit_case_scalers(train_case_dataset))
    scalers = fit_case_scalers(train_case_dataset)
    train_case_dataset_scaled = transform_case_dataset(train_case_dataset, scalers)

    candidate_df = sample_candidates_lhs(cfg.candidate_count, cfg.random_state)
    screening_start = time.perf_counter()
    pred_mean, pred_std, expert_energies, expert_tags = predict_energy_curve_ensemble(
        candidate_df=candidate_df,
        trunk_inputs_scaled=train_case_dataset_scaled.trunk_inputs,
        time_scaler_mean=float(scalers.trunk_scaler.mean_[0]),
        time_scaler_scale=float(scalers.trunk_scaler.scale_[0]),
        temp_scaler_mean=float(scalers.temperature_scaler.mean_[0]),
        temp_scaler_scale=float(scalers.temperature_scaler.scale_[0]),
        branch_scaler=scalers.branch_scaler,
        seed_models=seed_models,
        device=device,
    )
    screening_seconds = time.perf_counter() - screening_start
    candidate_df["pred_energy_mean_mj"] = pred_mean
    candidate_df["pred_energy_std_mj"] = pred_std
    for expert_idx, expert_tag in enumerate(expert_tags):
        candidate_df[f"{expert_tag}_pred_energy_mj"] = expert_energies[:, expert_idx]

    candidate_scaled = scalers.branch_scaler.transform(candidate_df[INPUT_COLUMNS])
    for col_idx, col in enumerate(INPUT_COLUMNS):
        candidate_df[f"{col}_scaled"] = candidate_scaled[:, col_idx]
    nn_dist = pairwise_distances(candidate_scaled, train_case_dataset_scaled.branch_inputs).min(axis=1)
    candidate_df["nearest_train_distance"] = nn_dist

    mean_norm = (candidate_df["pred_energy_mean_mj"] - candidate_df["pred_energy_mean_mj"].min()) / (
        candidate_df["pred_energy_mean_mj"].max() - candidate_df["pred_energy_mean_mj"].min() + 1e-12
    )
    std_norm = (candidate_df["pred_energy_std_mj"] - candidate_df["pred_energy_std_mj"].min()) / (
        candidate_df["pred_energy_std_mj"].max() - candidate_df["pred_energy_std_mj"].min() + 1e-12
    )
    dist_norm = (candidate_df["nearest_train_distance"] - candidate_df["nearest_train_distance"].min()) / (
        candidate_df["nearest_train_distance"].max() - candidate_df["nearest_train_distance"].min() + 1e-12
    )
    if cfg.use_conservative_score:
        candidate_df["selection_score"] = mean_norm - 0.18 * std_norm - 0.12 * dist_norm
    else:
        candidate_df["selection_score"] = mean_norm

    candidate_df = candidate_df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    top_candidates = select_diverse_candidates(
        candidate_df=candidate_df,
        train_branch_scaled=train_case_dataset_scaled.branch_inputs,
        top_k=cfg.verify_top_k,
        distance_threshold=cfg.diversity_distance_threshold,
    )
    top_candidates.insert(0, "rank", np.arange(1, len(top_candidates) + 1))

    best_observed = pd.read_csv(cfg.dataset_csv).sort_values("Output_TotalEnergy_MJ", ascending=False).iloc[0]
    best_observed_energy = float(best_observed["Output_TotalEnergy_MJ"])

    verification_rows = []
    verify_dir = run_dir / "verification_runs"
    verify_dir.mkdir(parents=True, exist_ok=True)
    for _, row in top_candidates.iterrows():
        result = simulate_candidate(row, verify_dir, cfg.timeout_seconds, Path(cfg.model_path))
        result["rank"] = int(row["rank"])
        result["pred_energy_mean_mj"] = float(row["pred_energy_mean_mj"])
        result["pred_energy_std_mj"] = float(row["pred_energy_std_mj"])
        result["pred_gain_vs_best_observed_mj"] = float(row["pred_energy_mean_mj"] - best_observed_energy)
        if not math.isnan(result["true_energy_mj"]):
            result["true_gain_vs_best_observed_mj"] = float(result["true_energy_mj"] - best_observed_energy)
            result["pred_error_mj"] = float(result["pred_energy_mean_mj"] - result["true_energy_mj"])
        else:
            result["true_gain_vs_best_observed_mj"] = np.nan
            result["pred_error_mj"] = np.nan
        for col in INPUT_COLUMNS:
            result[col] = float(row[col])
        verification_rows.append(result)

    candidate_df.to_csv(run_dir / "candidate_pool.csv", index=False, encoding="utf-8")
    boxplot_sample_df = candidate_df.sample(
        n=min(cfg.boxplot_sample_count, len(candidate_df)),
        random_state=cfg.random_state,
    ).copy()
    boxplot_sample_df.to_csv(run_dir / "boxplot_sample_2500.csv", index=False, encoding="utf-8")
    top_candidates.to_csv(run_dir / "selected_candidates.csv", index=False, encoding="utf-8")
    verification_df = pd.DataFrame(verification_rows)
    verification_df.to_csv(run_dir / "verification_results.csv", index=False, encoding="utf-8")

    brute_force_seconds = cfg.candidate_count * 51.931895154625622
    verification_seconds = float(sum(v.get("elapsed", 0.0) or 0.0 for v in verification_rows))
    closed_loop_seconds = screening_seconds + verification_seconds
    summary = {
        "best_observed_case_id": int(best_observed["ID"]),
        "best_observed_energy_mj": best_observed_energy,
        "expert_count": cfg.expert_count,
        "selected_experts": expert_tags,
        "candidate_count": cfg.candidate_count,
        "selected_candidate_count": len(top_candidates),
        "verified_success_count": int((verification_df["status"] == "success").sum()) if not verification_df.empty else 0,
        "screening_seconds": screening_seconds,
        "closed_loop_seconds": closed_loop_seconds,
        "closed_loop_minutes": closed_loop_seconds / 60.0,
        "bruteforce_seconds": brute_force_seconds,
        "bruteforce_hours": brute_force_seconds / 3600.0,
        "screening_speedup_vs_bruteforce": brute_force_seconds / screening_seconds if screening_seconds > 0 else float("inf"),
        "closed_loop_speedup_vs_bruteforce": brute_force_seconds / closed_loop_seconds if closed_loop_seconds > 0 else float("inf"),
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"Run directory: {run_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not verification_df.empty:
        print(verification_df.to_string(index=False))


if __name__ == "__main__":
    main()
