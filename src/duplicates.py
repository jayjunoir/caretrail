"""Cross-clinic duplicate injection for CareTrail Phase 1.

Duplicates approximately 8% of the cohort into other clinics with perturbed identities
and +/- 2% Gaussian noise on continuous vitals. Returns the combined cohort DataFrame
and ground-truth pairing records.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.identities import perturb_identity

CONTINUOUS_VITALS: List[str] = ["trestbps", "chol", "thalach", "oldpeak"]


def apply_vital_noise(
    val: float,
    col_name: str,
    rng: np.random.RandomState,
    noise_std: float = 0.02,
) -> float:
    """Apply +/- 2% Gaussian noise to a continuous vital sign value.

    Args:
        val: Original numeric vital value.
        col_name: Name of the column.
        rng: Random state instance.
        noise_std: Standard deviation of relative Gaussian noise (default 0.02 = 2%).

    Returns:
        Perturbed vital sign.
    """
    if pd.isna(val):
        return val

    val_f = float(val)
    if val_f == 0.0:
        # For values at zero (e.g. oldpeak), add absolute Gaussian noise of scale noise_std
        noise = rng.normal(0.0, noise_std)
        new_val = max(0.0, val_f + noise)
    else:
        # Relative Gaussian noise +/- 2%
        noise_factor = rng.normal(1.0, noise_std)
        new_val = max(0.0, val_f * noise_factor)

    if col_name in ("trestbps", "chol", "thalach"):
        return float(round(new_val))
    return float(round(new_val, 2))


def inject_cross_clinic_duplicates(
    df: pd.DataFrame,
    duplicate_fraction: float = 0.08,
    seed: int = 42,
    continuous_vitals: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sample ~8% of the cohort, duplicate across different clinics with noise.

    Args:
        df: Input DataFrame with clinical and identity attributes.
        duplicate_fraction: Fraction of cohort to duplicate (~0.08).
        seed: Random seed for reproducibility.
        continuous_vitals: List of continuous vitals to perturb.

    Returns:
        Tuple of (combined_df, ground_truth_pairs_df):
            - combined_df: Original cohort concatenated with perturbed duplicates.
            - ground_truth_pairs_df: Table tracking (original_patient_id,
              duplicate_patient_id, ground_truth_patient_id, etc.).
    """
    if continuous_vitals is None:
        continuous_vitals = CONTINUOUS_VITALS

    rng = np.random.RandomState(seed)
    n_cohort = len(df)
    n_duplicates = int(round(duplicate_fraction * n_cohort))

    # All unique clinics present in cohort
    all_clinics = sorted(df["clinic_id"].unique())
    if len(all_clinics) < 2:
        raise ValueError("At least 2 clinics required for cross-clinic duplicate injection.")

    # Sample rows to duplicate without replacement
    sampled_indices = rng.choice(n_cohort, size=n_duplicates, replace=False)

    dup_rows: List[Dict[str, Any]] = []
    pairing_records: List[Dict[str, Any]] = []

    for i, idx in enumerate(sampled_indices):
        orig_row = df.iloc[idx].to_dict()
        orig_clinic = orig_row["clinic_id"]

        # Choose a different clinic
        candidate_clinics = [c for c in all_clinics if c != orig_clinic]
        dest_clinic = rng.choice(candidate_clinics)

        # Perturb identity attributes
        orig_identity = {
            "full_name": orig_row["full_name"],
            "dob": orig_row["dob"],
            "sex": orig_row["sex"],
        }
        # Use derived seed for deterministic identity perturbation
        row_seed = int(seed * 1000 + i)
        perturbed_id = perturb_identity(orig_identity, seed=row_seed)

        dup_record = dict(orig_row)
        dup_pid = f"{orig_row['patient_id']}_DUP_{i:04d}"

        dup_record["patient_id"] = dup_pid
        dup_record["clinic_id"] = dest_clinic
        dup_record["full_name"] = perturbed_id["full_name"]
        dup_record["dob"] = perturbed_id["dob"]
        # Ground truth patient ID matches the original patient!
        dup_record["ground_truth_patient_id"] = orig_row["ground_truth_patient_id"]

        # Perturb continuous vitals with +/- 2% Gaussian noise
        for col in continuous_vitals:
            if col in dup_record:
                dup_record[col] = apply_vital_noise(dup_record[col], col, rng=rng)

        dup_rows.append(dup_record)

        pairing_records.append({
            "original_patient_id": orig_row["patient_id"],
            "duplicate_patient_id": dup_pid,
            "ground_truth_patient_id": orig_row["ground_truth_patient_id"],
            "orig_clinic": orig_clinic,
            "dup_clinic": dest_clinic,
            "orig_name": orig_row["full_name"],
            "dup_name": perturbed_id["full_name"],
            "orig_dob": orig_row["dob"],
            "dup_dob": perturbed_id["dob"],
        })

    dup_df = pd.DataFrame(dup_rows)
    combined_df = pd.concat([df, dup_df], ignore_index=True)
    pairs_df = pd.DataFrame(pairing_records)

    return combined_df, pairs_df
