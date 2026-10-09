"""Federated Learning (FedAvg) and baseline evaluations for CareTrail Phase 2.

Implements:
1. Hand-rolled FedAvg across the 4 clinical nodes (Cleveland, Hungarian, Switzerland, VA)
   using PyTorch CPU with zero external federated frameworks.
2. Weighted parameter aggregation strictly based on node training dataset size:
   w_global = sum((n_k / n_total) * w_k)
3. Three experimental baselines:
   (a) Local-only: train and evaluate on each clinic independently.
   (b) Centralized: train on pooled data from all 4 clinics.
   (c) Federated: FedAvg across N rounds.
4. Experiment B runner executing across 20 seeds (0 to 19) with group-aware ER splits,
   logging training events to the audit ledger and saving summary to results/experiment_b_results.csv.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn

from src.consent import ConsentRegistry
from src.data_prep import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    impute_missing_per_node,
    load_all_clinics,
    scale_features,
)
from src.duplicates import inject_cross_clinic_duplicates
from src.entity_resolution import resolve_entities
from src.identities import generate_patient_profiles
from src.ledger import (
    EMPTY_HASH,
    AuditLedger,
    compute_merkle_root,
    hash_model_delta,
    hash_model_weights,
)
from src.model import LogisticRegressionModel


CLINIC_NODES: List[str] = ["cleveland", "hungarian", "switzerland", "va"]


def train_node_local(
    initial_weights: Dict[str, torch.Tensor],
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 5,
    lr: float = 0.02,
    optimizer_name: str = "adam",
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Train a local model starting from initial_weights on node dataset (X, y).

    Args:
        initial_weights: Model parameters to initialize from.
        X: Feature matrix of shape (N, 13).
        y: Labels of shape (N,).
        epochs: Number of local training epochs.
        lr: Learning rate.
        optimizer_name: 'adam' or 'sgd'.
        seed: Random seed for deterministic reproducibility.

    Returns:
        Updated weights dictionary.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model = LogisticRegressionModel(in_features=len(FEATURE_COLUMNS))
    model.set_weights(initial_weights)
    model.train()

    criterion = nn.BCELoss()
    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()

    return model.get_weights()


def aggregate_fedavg(
    node_weights: Dict[str, Dict[str, torch.Tensor]],
    node_sample_sizes: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Perform FedAvg weighted parameter aggregation strictly based on node training size.

    w_global = sum((n_k / n_total) * w_k)

    Args:
        node_weights: Dict mapping clinic_id to parameter weights dict.
        node_sample_sizes: Dict mapping clinic_id to number of local training samples n_k.

    Returns:
        Aggregated global model weights.
    """
    total_samples = sum(node_sample_sizes.values())
    if total_samples == 0:
        raise ValueError("Total training samples across nodes must be > 0")

    first_node_weights = next(iter(node_weights.values()))
    aggregated: Dict[str, torch.Tensor] = {}

    for param_name in first_node_weights.keys():
        weighted_sum = sum(
            (node_sample_sizes[node] / total_samples) * node_weights[node][param_name]
            for node in node_weights.keys()
        )
        aggregated[param_name] = weighted_sum

    return aggregated


def run_centralized(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 50,
    lr: float = 0.02,
    optimizer_name: str = "adam",
    seed: int = 42,
) -> Tuple[LogisticRegressionModel, Dict[str, float]]:
    """Train centralized baseline on pooled data from all 4 clinics."""
    torch.manual_seed(seed)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[TARGET_COLUMN].values

    model = LogisticRegressionModel(in_features=len(FEATURE_COLUMNS))
    model.train()
    criterion = nn.BCELoss()

    if optimizer_name.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_t), y_t)
        loss.backward()
        optimizer.step()

    auroc_train, acc_train = model.evaluate(X_train, y_train)
    auroc_test, acc_test = model.evaluate(X_test, y_test)

    metrics = {
        "auroc_train": auroc_train,
        "auroc_test": auroc_test,
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
    }
    return model, metrics


def run_local_only(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    epochs: int = 50,
    lr: float = 0.02,
    optimizer_name: str = "adam",
    seed: int = 42,
) -> Tuple[Dict[str, LogisticRegressionModel], Dict[str, float]]:
    """Train local-only baseline for each clinic independently.

    Evaluates each clinic's model on its local test set and computes macro-average metrics.
    """
    models: Dict[str, LogisticRegressionModel] = {}
    auroc_train_list, auroc_test_list = [], []
    acc_train_list, acc_test_list = [], []

    clinics = sorted(train_df["clinic_id"].unique())

    for idx, clinic in enumerate(clinics):
        c_train = train_df[train_df["clinic_id"] == clinic]
        c_test = test_df[test_df["clinic_id"] == clinic]

        X_k_train = c_train[FEATURE_COLUMNS].values
        y_k_train = c_train[TARGET_COLUMN].values
        X_k_test = c_test[FEATURE_COLUMNS].values
        y_k_test = c_test[TARGET_COLUMN].values

        torch.manual_seed(seed + idx)
        model = LogisticRegressionModel(in_features=len(FEATURE_COLUMNS))
        criterion = nn.BCELoss()
        if optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        X_t = torch.tensor(X_k_train, dtype=torch.float32)
        y_t = torch.tensor(y_k_train, dtype=torch.float32).unsqueeze(1)

        for _ in range(epochs):
            optimizer.zero_grad()
            loss = criterion(model(X_t), y_t)
            loss.backward()
            optimizer.step()

        models[clinic] = model

        tr_auc, tr_acc = model.evaluate(X_k_train, y_k_train)
        te_auc, te_acc = model.evaluate(X_k_test, y_k_test)

        auroc_train_list.append(tr_auc)
        auroc_test_list.append(te_auc)
        acc_train_list.append(tr_acc)
        acc_test_list.append(te_acc)

    metrics = {
        "auroc_train": float(np.mean(auroc_train_list)),
        "auroc_test": float(np.mean(auroc_test_list)),
        "accuracy_train": float(np.mean(acc_train_list)),
        "accuracy_test": float(np.mean(acc_test_list)),
    }
    return models, metrics


def run_fedavg(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_rounds: int = 10,
    local_epochs: int = 5,
    lr: float = 0.02,
    optimizer_name: str = "adam",
    seed: int = 42,
    ledger: Optional[AuditLedger] = None,
    consent_registry: Optional[ConsentRegistry] = None,
) -> Tuple[LogisticRegressionModel, Dict[str, float]]:
    """Execute federated learning via FedAvg across clinical nodes."""
    torch.manual_seed(seed)
    global_model = LogisticRegressionModel(in_features=len(FEATURE_COLUMNS))

    clinics = sorted(train_df["clinic_id"].unique())

    for round_idx in range(num_rounds):
        current_global_weights = global_model.get_weights()
        current_model_hash = hash_model_weights(current_global_weights)

        node_weights: Dict[str, Dict[str, torch.Tensor]] = {}
        node_sizes: Dict[str, int] = {}

        for c_idx, clinic in enumerate(clinics):
            node_df = train_df[train_df["clinic_id"] == clinic]

            # Filter active consent if registry provided
            if consent_registry is not None:
                node_df = consent_registry.filter_active_records(node_df)
                pids = node_df["patient_id"].astype(str).tolist()
                manifest_root, _ = consent_registry.record_manifest(round_idx, clinic, pids)
            else:
                pids = node_df["patient_id"].astype(str).tolist() if "patient_id" in node_df.columns else []
                commitments = [hash_model_weights({"t": torch.tensor([float(i)])}) for i in range(len(pids))]
                manifest_root = compute_merkle_root(commitments)

            X_k = node_df[FEATURE_COLUMNS].values
            y_k = node_df[TARGET_COLUMN].values
            n_k = len(node_df)

            node_sizes[clinic] = n_k

            # Local node training
            node_seed = seed + round_idx * 100 + c_idx
            updated_w = train_node_local(
                current_global_weights,
                X_k,
                y_k,
                epochs=local_epochs,
                lr=lr,
                optimizer_name=optimizer_name,
                seed=node_seed,
            )
            node_weights[clinic] = updated_w

            # Audit ledger commit
            if ledger is not None:
                update_hash = hash_model_delta(updated_w, current_global_weights)
                ledger.append_entry(
                    event_type="round_commit",
                    node_id=clinic,
                    round_idx=round_idx,
                    manifest_root=manifest_root,
                    update_hash=update_hash,
                    model_hash=current_model_hash,
                )

        # Weighted FedAvg aggregation
        new_global_weights = aggregate_fedavg(node_weights, node_sizes)
        global_model.set_weights(new_global_weights)

    # Final release entry to ledger
    if ledger is not None:
        final_weights = global_model.get_weights()
        final_model_hash = hash_model_weights(final_weights)
        ledger.append_entry(
            event_type="model_release",
            node_id="coordinator",
            round_idx=num_rounds,
            manifest_root=EMPTY_HASH,
            update_hash=EMPTY_HASH,
            model_hash=final_model_hash,
        )

    X_train_all = train_df[FEATURE_COLUMNS].values
    y_train_all = train_df[TARGET_COLUMN].values
    X_test_all = test_df[FEATURE_COLUMNS].values
    y_test_all = test_df[TARGET_COLUMN].values

    auroc_train, acc_train = global_model.evaluate(X_train_all, y_train_all)
    auroc_test, acc_test = global_model.evaluate(X_test_all, y_test_all)

    metrics = {
        "auroc_train": auroc_train,
        "auroc_test": auroc_test,
        "accuracy_train": acc_train,
        "accuracy_test": acc_test,
    }
    return global_model, metrics


def run_experiment_b(
    data_dir: str = "data",
    results_dir: str = "results",
    num_seeds: int = 20,
    test_size: float = 0.20,
    num_rounds: int = 10,
    local_epochs: int = 5,
    lr: float = 0.02,
    ledger_path: Optional[str] = "results/audit_ledger.jsonl",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Execute Experiment B comparing Local, Centralized, and FedAvg across 20 seeds."""
    res_path = Path(results_dir)
    res_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CareTrail Phase 2: Running Experiment B (Federated vs Centralized vs Local)")
    print("=" * 70)

    # Step 1: Load and prepare cohort with ER from Phase 1
    print("Loading raw clinical datasets...")
    raw_df = load_all_clinics(data_dir=data_dir)
    print(f"Loaded {len(raw_df)} records across {raw_df['clinic_id'].nunique()} clinics.")

    print("Generating synthetic patient profiles and injecting duplicates...")
    cohort_df = generate_patient_profiles(raw_df, seed=42)
    augmented_df, _ = inject_cross_clinic_duplicates(cohort_df, duplicate_fraction=0.08, seed=42)

    print("Resolving entities to assign cluster IDs...")
    clustered_df, _, _ = resolve_entities(augmented_df, threshold=0.83)

    # Initialize Ledger and Consent Registry if path given
    ledger = AuditLedger(ledger_path) if ledger_path else None
    consent_registry = ConsentRegistry(ledger=ledger)
    for pid in clustered_df["patient_id"]:
        consent_registry.register_patient(str(pid))

    records: List[Dict[str, Any]] = []

    print(f"\nRunning 20-seed evaluation (seeds 0 to {num_seeds - 1})...")

    for seed in range(num_seeds):
        # Entity-Resolution Group-Aware Split
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(clustered_df, groups=clustered_df["cluster_id"]))

        train_er = clustered_df.iloc[train_idx].copy()
        test_er = clustered_df.iloc[test_idx].copy()

        # Preprocessing strictly on train split
        clean_train, clean_test = impute_missing_per_node(train_er, test_er)
        scaled_train, scaled_test, _ = scale_features(clean_train, clean_test)

        # 1. Centralized baseline
        _, cent_metrics = run_centralized(
            scaled_train, scaled_test, epochs=num_rounds * local_epochs, lr=lr, seed=seed
        )
        records.append({
            "seed": seed,
            "strategy": "centralized",
            "auroc_train": cent_metrics["auroc_train"],
            "auroc_test": cent_metrics["auroc_test"],
            "accuracy_train": cent_metrics["accuracy_train"],
            "accuracy_test": cent_metrics["accuracy_test"],
        })

        # 2. Local-only baseline
        _, loc_metrics = run_local_only(
            scaled_train, scaled_test, epochs=num_rounds * local_epochs, lr=lr, seed=seed
        )
        records.append({
            "seed": seed,
            "strategy": "local",
            "auroc_train": loc_metrics["auroc_train"],
            "auroc_test": loc_metrics["auroc_test"],
            "accuracy_train": loc_metrics["accuracy_train"],
            "accuracy_test": loc_metrics["accuracy_test"],
        })

        # 3. Federated FedAvg
        # For ledger logging, only log seed 0 to avoid overwhelming ledger with 20 runs
        seed_ledger = ledger if seed == 0 else None
        _, fed_metrics = run_fedavg(
            scaled_train,
            scaled_test,
            num_rounds=num_rounds,
            local_epochs=local_epochs,
            lr=lr,
            seed=seed,
            ledger=seed_ledger,
            consent_registry=consent_registry if seed == 0 else None,
        )
        records.append({
            "seed": seed,
            "strategy": "federated",
            "auroc_train": fed_metrics["auroc_train"],
            "auroc_test": fed_metrics["auroc_test"],
            "accuracy_train": fed_metrics["accuracy_train"],
            "accuracy_test": fed_metrics["accuracy_test"],
        })

    results_df = pd.DataFrame(records)
    results_file = res_path / "experiment_b_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved experiment results to {results_file}")

    # Summary table
    summary_df = (
        results_df.groupby("strategy")
        .agg(
            auroc_train_mean=("auroc_train", "mean"),
            auroc_train_std=("auroc_train", "std"),
            auroc_test_mean=("auroc_test", "mean"),
            auroc_test_std=("auroc_test", "std"),
            accuracy_train_mean=("accuracy_train", "mean"),
            accuracy_train_std=("accuracy_train", "std"),
            accuracy_test_mean=("accuracy_test", "mean"),
            accuracy_test_std=("accuracy_test", "std"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT B SUMMARY TABLE (20 SEEDS)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print("=" * 70 + "\n")

    return results_df, summary_df


if __name__ == "__main__":
    run_experiment_b()
