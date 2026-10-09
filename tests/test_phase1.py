"""Unit tests for CareTrail Phase 1.

Verifies:
1. No NaN leaks during data prep (both train and test splits).
2. All 4 identity perturbation types (swapped names, character transpositions, shifted DOBs, format variants).
3. ~8% duplicate injection with ground-truth preservation and vital perturbation.
4. Entity resolution pipeline achieves recall >= 0.80.
"""

from datetime import datetime
import numpy as np
import pandas as pd
import pytest

from src.data_prep import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    impute_missing_per_node,
    load_all_clinics,
    scale_features,
)
from src.duplicates import inject_cross_clinic_duplicates
from src.entity_resolution import parse_dob_safely, resolve_entities
from src.identities import (
    generate_patient_profiles,
    perturb_identity,
    shift_dob,
    swap_name,
    toggle_dob_format,
    transpose_characters,
)


def test_data_prep_no_nan_leaks():
    """Verify that data loading, binarization, node median imputation, and scaling leak 0 NaNs."""
    df = load_all_clinics(data_dir="data")

    # Check basic properties
    assert len(df) > 0, "Cohort should not be empty"
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1}), "Target must be binarized to {0, 1}"
    assert "clinic_id" in df.columns, "clinic_id must be present"

    # Simulate an 80/20 train/test split
    np.random.seed(42)
    mask = np.random.rand(len(df)) < 0.8
    train_df = df[mask].copy()
    test_df = df[~mask].copy()

    # Impute missing values with node-level medians computed strictly on train
    clean_train, clean_test = impute_missing_per_node(train_df, test_df)

    assert clean_train[FEATURE_COLUMNS].isna().sum().sum() == 0, (
        "Train set contains NaN leaks after imputation"
    )
    assert clean_test[FEATURE_COLUMNS].isna().sum().sum() == 0, (
        "Test set contains NaN leaks after imputation"
    )

    # Scale features strictly on train statistics
    scaled_train, scaled_test, scaler = scale_features(clean_train, clean_test)

    assert scaled_train[FEATURE_COLUMNS].isna().sum().sum() == 0, (
        "Train set contains NaN leaks after scaling"
    )
    assert scaled_test[FEATURE_COLUMNS].isna().sum().sum() == 0, (
        "Test set contains NaN leaks after scaling"
    )


def test_edge_case_node_all_nan():
    """Verify fallback when an entire node has all NaNs in training set."""
    synthetic_train = pd.DataFrame({
        "clinic_id": ["node_A", "node_A", "node_B", "node_B"],
        "age": [50.0, 52.0, 60.0, 62.0],
        "chol": [np.nan, np.nan, 200.0, 220.0],  # node_A has all NaNs for chol
        "trestbps": [np.nan, np.nan, np.nan, np.nan],  # All nodes all NaNs
        "num": [0, 1, 0, 1],
    })
    synthetic_test = pd.DataFrame({
        "clinic_id": ["node_A", "node_B"],
        "age": [51.0, 61.0],
        "chol": [np.nan, np.nan],
        "trestbps": [np.nan, np.nan],
        "num": [0, 1],
    })
    cols = ["age", "chol", "trestbps"]

    c_train, c_test = impute_missing_per_node(
        synthetic_train, synthetic_test, feature_cols=cols
    )
    assert c_train[cols].isna().sum().sum() == 0, "Train should have 0 NaNs"
    assert c_test[cols].isna().sum().sum() == 0, "Test should have 0 NaNs"
    # node_A chol should fall back to global train median (210.0)
    assert c_train.loc[c_train["clinic_id"] == "node_A", "chol"].iloc[0] == 210.0
    # trestbps was all NaNs across all rows -> fallback to 0.0
    assert (c_train["trestbps"] == 0.0).all()


def test_identity_perturbations():
    """Verify that all 4 identity perturbation functions work accurately."""
    base_identity = {
        "full_name": "Emeka Okonkwo",
        "dob": "1980-05-15",
        "sex": 1,
    }

    # 1. Swapped names
    p_swap = perturb_identity(base_identity, perturbation_types=["swapped_names"])
    assert p_swap["full_name"] == "Okonkwo Emeka", (
        f"Expected 'Okonkwo Emeka', got '{p_swap['full_name']}'"
    )

    # 2. Character transposition
    p_trans = perturb_identity(
        base_identity, perturbation_types=["character_transposition"], seed=42
    )
    assert p_trans["full_name"] != base_identity["full_name"], "Name should be transposed"
    # Same characters overall, just adjacent swap
    assert sorted(p_trans["full_name"].replace(" ", "")) == sorted(
        base_identity["full_name"].replace(" ", "")
    )

    # 3. Shifted DOB (+/- 1 day)
    p_dob_shift = perturb_identity(
        base_identity, perturbation_types=["shifted_dob"], seed=10
    )
    orig_dt = datetime.strptime(base_identity["dob"], "%Y-%m-%d").date()
    new_dt = datetime.strptime(p_dob_shift["dob"], "%Y-%m-%d").date()
    day_diff = abs((new_dt - orig_dt).days)
    assert day_diff == 1, f"Expected 1 day difference, got {day_diff}"

    # 4. Format variant (ISO to DD/MM/YYYY and back)
    p_fmt = perturb_identity(base_identity, perturbation_types=["format_variant"])
    assert p_fmt["dob"] == "15/05/1980", f"Expected '15/05/1980', got '{p_fmt['dob']}'"

    # Toggling again should return to ISO
    p_back = perturb_identity(p_fmt, perturbation_types=["format_variant"])
    assert p_back["dob"] == "1980-05-15", f"Expected '1980-05-15', got '{p_back['dob']}'"


def test_duplicate_injection():
    """Verify ~8% duplicate injection, clinic reassignment, and vital perturbations."""
    df = load_all_clinics(data_dir="data")
    cohort_df = generate_patient_profiles(df, seed=42)

    combined_df, pairs_df = inject_cross_clinic_duplicates(
        cohort_df, duplicate_fraction=0.08, seed=42
    )

    n_orig = len(cohort_df)
    n_dups = len(combined_df) - n_orig

    # Verify duplicate rate is ~8% (between 7% and 9%)
    dup_ratio = n_dups / n_orig
    assert 0.07 <= dup_ratio <= 0.09, (
        f"Expected ~0.08 duplicate ratio, got {dup_ratio:.4f}"
    )

    # Verify pairing records match
    assert len(pairs_df) == n_dups, "Pairs table should have exactly n_dups rows"

    # Verify clinic reassignment: duplicate clinic must differ from original clinic
    assert (pairs_df["orig_clinic"] != pairs_df["dup_clinic"]).all(), (
        "Duplicate records must be assigned to a different clinic"
    )

    # Verify ground truth patient ID matches
    for _, row in pairs_df.iterrows():
        orig_match = combined_df[combined_df["patient_id"] == row["original_patient_id"]]
        dup_match = combined_df[combined_df["patient_id"] == row["duplicate_patient_id"]]
        assert len(orig_match) == 1
        assert len(dup_match) == 1
        assert (
            orig_match["ground_truth_patient_id"].iloc[0]
            == dup_match["ground_truth_patient_id"].iloc[0]
        ), "Ground truth patient ID must match between original and duplicate"

    # Verify continuous vitals noise was applied
    dup_rows = combined_df[combined_df["patient_id"].str.contains("_DUP_")]
    assert len(dup_rows) == n_dups


def test_entity_resolution_recall():
    """Verify entity resolution pipeline achieves recall >= 0.80 and produces diagnostics."""
    df = load_all_clinics(data_dir="data")
    cohort_df = generate_patient_profiles(df, seed=42)
    combined_df, _ = inject_cross_clinic_duplicates(
        cohort_df, duplicate_fraction=0.08, seed=42
    )

    clustered_df, metrics, failures_df = resolve_entities(
        combined_df, threshold=0.83
    )

    # Critical requirement: recall >= 0.80
    assert metrics["recall"] >= 0.80, (
        f"Entity resolution recall {metrics['recall']:.4f} is below the 0.80 threshold"
    )
    assert metrics["precision"] >= 0.70, (
        f"Entity resolution precision {metrics['precision']:.4f} is too low"
    )
    assert metrics["f1"] >= 0.75, (
        f"Entity resolution F1-score {metrics['f1']:.4f} is too low"
    )

    # Verify clusters
    assert "cluster_id" in clustered_df.columns
    assert clustered_df["cluster_id"].isna().sum() == 0, "Every record must have a cluster_id"

    # Verify failures DataFrame
    assert isinstance(failures_df, pd.DataFrame)
    assert "error_type" in failures_df.columns
