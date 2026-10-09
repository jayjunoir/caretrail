"""Data preparation pipeline for CareTrail Phase 1.

Loads the 4 UCI Heart Disease datasets (Cleveland, Hungarian, Switzerland, VA),
binarizes the target variable, and provides strict training-only node-specific
median imputation and feature scaling.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

FEATURE_COLUMNS: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

TARGET_COLUMN: str = "num"

CLINIC_FILE_MAP: Dict[str, str] = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

NA_VALUES: List[Union[str, int, float]] = ["?", "-9", "-9.0", -9, -9.0]


def load_clinic_data(file_path: Union[str, Path], clinic_id: str) -> pd.DataFrame:
    """Load and format a single clinic's raw data file.

    Args:
        file_path: Path to the processed.<clinic>.data file.
        clinic_id: Identifier for the clinic (e.g., 'cleveland').

    Returns:
        DataFrame with standardized columns, numeric types, binarized target,
        and clinic_id.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found at {path}")

    df = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values=NA_VALUES,
        skipinitialspace=True,
    )

    # Convert all feature columns to float
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize target: num > 0 -> 1, else 0
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce").fillna(0)
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

    # Add clinic_id
    df["clinic_id"] = clinic_id

    return df


def load_all_clinics(data_dir: Union[str, Path] = "data") -> pd.DataFrame:
    """Load all 4 clinic datasets and combine into a single DataFrame.

    Args:
        data_dir: Directory containing the processed.*.data files.

    Returns:
        Combined DataFrame of all 4 clinics.
    """
    base_dir = Path(data_dir)
    dfs: List[pd.DataFrame] = []

    for clinic_id, filename in CLINIC_FILE_MAP.items():
        file_path = base_dir / filename
        clinic_df = load_clinic_data(file_path, clinic_id)
        dfs.append(clinic_df)

    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def compute_node_medians(
    train_df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    node_col: str = "clinic_id",
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
    """Compute per-node medians and global medians strictly on a training set.

    Args:
        train_df: Training DataFrame.
        feature_cols: Feature columns to compute medians for.
        node_col: Column indicating the node/clinic.

    Returns:
        Tuple of (per_node_medians_dict, global_medians_dict).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Global medians across all training rows
    global_medians: Dict[str, float] = {}
    for col in feature_cols:
        med = train_df[col].median(skipna=True)
        # If entire column is NaN across all training rows, fallback to 0.0
        global_medians[col] = 0.0 if pd.isna(med) else float(med)

    # Per-node medians
    node_medians: Dict[str, Dict[str, float]] = {}
    for node, group in train_df.groupby(node_col):
        node_medians[str(node)] = {}
        for col in feature_cols:
            n_med = group[col].median(skipna=True)
            if pd.isna(n_med):
                # Fallback to global median if node has all NaNs for this column
                node_medians[str(node)][col] = global_medians[col]
            else:
                node_medians[str(node)][col] = float(n_med)

    return node_medians, global_medians


def impute_missing_per_node(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    node_col: str = "clinic_id",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Impute missing values with per-node medians computed strictly on train split.

    Args:
        train_df: Training DataFrame.
        test_df: Optional testing DataFrame.
        feature_cols: Columns to impute.
        node_col: Column indicating node/clinic.

    Returns:
        Tuple of (imputed_train_df, imputed_test_df).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    node_medians, global_medians = compute_node_medians(
        train_df, feature_cols=feature_cols, node_col=node_col
    )

    clean_train = train_df.copy()

    # Impute train
    for node, group in clean_train.groupby(node_col):
        node_key = str(node)
        medians = node_medians.get(node_key, global_medians)
        for col in feature_cols:
            val = medians.get(col, global_medians[col])
            clean_train.loc[clean_train[node_col] == node, col] = clean_train.loc[
                clean_train[node_col] == node, col
            ].fillna(val)

    # Secondary check: fill any remaining NaNs with global fallback
    for col in feature_cols:
        clean_train[col] = clean_train[col].fillna(global_medians[col])

    clean_test: Optional[pd.DataFrame] = None
    if test_df is not None:
        clean_test = test_df.copy()
        for node, group in clean_test.groupby(node_col):
            node_key = str(node)
            medians = node_medians.get(node_key, global_medians)
            for col in feature_cols:
                val = medians.get(col, global_medians[col])
                clean_test.loc[clean_test[node_col] == node, col] = clean_test.loc[
                    clean_test[node_col] == node, col
                ].fillna(val)

        # Secondary check for test
        for col in feature_cols:
            clean_test[col] = clean_test[col].fillna(global_medians[col])

    return clean_train, clean_test


def scale_features(
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], StandardScaler]:
    """Fit StandardScaler strictly on training set statistics and scale features.

    Args:
        train_df: Training DataFrame.
        test_df: Optional testing DataFrame.
        feature_cols: List of feature columns to scale.

    Returns:
        Tuple of (scaled_train_df, scaled_test_df, fitted_scaler).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    scaler = StandardScaler()
    scaled_train = train_df.copy()
    scaled_train[feature_cols] = scaler.fit_transform(scaled_train[feature_cols])

    scaled_test: Optional[pd.DataFrame] = None
    if test_df is not None:
        scaled_test = test_df.copy()
        scaled_test[feature_cols] = scaler.transform(scaled_test[feature_cols])

    return scaled_train, scaled_test, scaler
