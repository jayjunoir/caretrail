"""Experiment A for CareTrail Phase 1.

Compares:
(a) Naive random train/test split vs.
(b) Entity-resolution group-aware split

Evaluates data leakage and model generalization using LogisticRegression across 20 seeds (0 to 19).
Saves:
- results/experiment_a_results.csv
- results/er_failures.csv
"""

from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.data_prep import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    impute_missing_per_node,
    load_all_clinics,
    scale_features,
)
from src.duplicates import inject_cross_clinic_duplicates
from src.entity_resolution import resolve_entities
from src.federated import run_experiment_b
from src.identities import generate_patient_profiles


def count_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame) -> int:
    """Count number of test records whose true entity (ground_truth_patient_id) is in train."""
    train_entities = set(train_df["ground_truth_patient_id"].unique())
    leakage = test_df["ground_truth_patient_id"].isin(train_entities).sum()
    return int(leakage)


def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
) -> Tuple[float, float, float]:
    """Preprocess data strictly on train statistics, fit LogisticRegression, evaluate AUROC.

    Returns:
        Tuple of (auroc_train, auroc_test, generalization_gap).
    """
    # 1. Impute missing values with per-node medians computed strictly on train split
    clean_train, clean_test = impute_missing_per_node(train_df, test_df)

    # 2. Fit StandardScaler strictly on train statistics
    scaled_train, scaled_test, _ = scale_features(clean_train, clean_test)

    X_train = scaled_train[FEATURE_COLUMNS].values
    y_train = scaled_train[TARGET_COLUMN].values
    X_test = scaled_test[FEATURE_COLUMNS].values
    y_test = scaled_test[TARGET_COLUMN].values

    # 3. Fit Logistic Regression
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed,
    )
    clf.fit(X_train, y_train)

    train_probs = clf.predict_proba(X_train)[:, 1]
    test_probs = clf.predict_proba(X_test)[:, 1]

    auroc_train = float(roc_auc_score(y_train, train_probs))
    auroc_test = float(roc_auc_score(y_test, test_probs))
    gen_gap = float(auroc_train - auroc_test)

    return auroc_train, auroc_test, gen_gap


def run_experiment_a(
    data_dir: str = "data",
    results_dir: str = "results",
    num_seeds: int = 20,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Execute Experiment A comparing naive random vs ER group-aware split across seeds.

    Args:
        data_dir: Directory containing processed UCI datasets.
        results_dir: Directory to store output CSVs.
        num_seeds: Number of random seeds to evaluate (default 20: 0 to 19).
        test_size: Holdout fraction (default 0.20).

    Returns:
        Tuple of (results_df, summary_df, failures_df).
    """
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CareTrail Phase 1: Running Experiment A")
    print("=" * 70)

    # Step 1: Load raw datasets
    print("Loading raw clinical datasets...")
    raw_df = load_all_clinics(data_dir=data_dir)
    print(f"Loaded {len(raw_df)} total records across {raw_df['clinic_id'].nunique()} clinics.")

    # Step 2: Generate synthetic patient identities
    print("Generating synthetic patient profiles...")
    cohort_df = generate_patient_profiles(raw_df, seed=42)

    # Step 3: Inject ~8% cross-clinic duplicates
    print("Injecting ~8% cross-clinic duplicates with perturbations...")
    augmented_df, gt_pairs = inject_cross_clinic_duplicates(
        cohort_df, duplicate_fraction=0.08, seed=42
    )
    print(
        f"Augmented cohort: {len(augmented_df)} records "
        f"({len(augmented_df) - len(cohort_df)} duplicates injected)."
    )

    # Step 4: Run Entity Resolution
    print("Executing Entity Resolution (blocking, RapidFuzz scoring, NetworkX clustering)...")
    clustered_df, er_metrics, failures_df = resolve_entities(augmented_df)
    print(f"ER Evaluation Metrics against Ground Truth:")
    print(f"  Precision: {er_metrics['precision']:.4f}")
    print(f"  Recall:    {er_metrics['recall']:.4f}")
    print(f"  F1-Score:  {er_metrics['f1']:.4f}")
    print(f"  TP: {er_metrics['tp']}, FP: {er_metrics['fp']}, FN: {er_metrics['fn']}")

    # Save ER failures
    failures_file = res_path / "er_failures.csv"
    failures_df.to_csv(failures_file, index=False)
    print(f"Saved ER failure cases ({len(failures_df)} rows) to {failures_file}")

    # Step 5: Run experiments across seeds
    print(f"\nRunning 20-seed evaluation (seeds 0 to {num_seeds - 1})...")
    records: List[Dict] = []

    for seed in range(num_seeds):
        # (a) Naive Random Split
        train_naive, test_naive = train_test_split(
            clustered_df,
            test_size=test_size,
            random_state=seed,
        )
        naive_leakage = count_leakage(train_naive, test_naive)
        naive_train_auc, naive_test_auc, naive_gap = train_and_evaluate(
            train_naive, test_naive, seed=seed
        )

        records.append({
            "seed": seed,
            "split_type": "naive_random",
            "auroc_train": naive_train_auc,
            "auroc_test": naive_test_auc,
            "generalization_gap": naive_gap,
            "leakage_count": naive_leakage,
        })

        # (b) Entity-Resolution Group-Aware Split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(
            gss.split(clustered_df, groups=clustered_df["cluster_id"])
        )
        train_er = clustered_df.iloc[train_idx].copy()
        test_er = clustered_df.iloc[test_idx].copy()

        er_leakage = count_leakage(train_er, test_er)
        er_train_auc, er_test_auc, er_gap = train_and_evaluate(
            train_er, test_er, seed=seed
        )

        records.append({
            "seed": seed,
            "split_type": "er_group_aware",
            "auroc_train": er_train_auc,
            "auroc_test": er_test_auc,
            "generalization_gap": er_gap,
            "leakage_count": er_leakage,
        })

    results_df = pd.DataFrame(records)
    results_file = res_path / "experiment_a_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved experiment results to {results_file}")

    # Compute summary statistics
    summary_df = (
        results_df.groupby("split_type")
        .agg(
            auroc_train_mean=("auroc_train", "mean"),
            auroc_train_std=("auroc_train", "std"),
            auroc_test_mean=("auroc_test", "mean"),
            auroc_test_std=("auroc_test", "std"),
            gap_mean=("generalization_gap", "mean"),
            gap_std=("generalization_gap", "std"),
            leakage_mean=("leakage_count", "mean"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT A SUMMARY TABLE (20 SEEDS)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    return results_df, summary_df, failures_df


def run_all_experiments(
    data_dir: str = "data",
    results_dir: str = "results",
    num_seeds: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute both Experiment A (leakage evaluation) and Experiment B (federation evaluation)."""
    # 1. Run Experiment A
    res_a, summary_a, _ = run_experiment_a(
        data_dir=data_dir, results_dir=results_dir, num_seeds=num_seeds
    )

    # 2. Run Experiment B
    res_b, summary_b = run_experiment_b(
        data_dir=data_dir, results_dir=results_dir, num_seeds=num_seeds
    )

    return summary_a, summary_b


if __name__ == "__main__":
    run_all_experiments()
