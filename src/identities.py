"""Synthetic identity generation and perturbation for CareTrail Phase 1.

Generates realistic patient demographic profiles with common Nigerian and international
names, calculated birth dates based on age (base year 2026), and realistic identity
perturbations (swapped names, character transpositions, shifted DOBs, format variants).
"""

from datetime import date, datetime, timedelta
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Pools of realistic names
NIGERIAN_MALE_FIRST = [
    "Emeka", "Babatunde", "Oluwaseun", "Chukwudi", "Adebayo",
    "Ibrahim", "Musa", "Chinedu", "Olumide", "Tunde",
    "Damilola", "Segun", "Obinna", "Femi", "Kelechi",
]

NIGERIAN_FEMALE_FIRST = [
    "Chioma", "Amina", "Ngozi", "Aisha", "Funmilayo",
    "Zainab", "Blessing", "Ifeoma", "Folake", "Halima",
    "Amaka", "Chiamaka", "Temitope", "Yetunde", "Fatima",
]

INTL_MALE_FIRST = [
    "John", "Michael", "David", "James", "Robert",
    "Daniel", "Thomas", "William", "Alexander", "Lucas",
    "Oliver", "Benjamin", "Ethan", "Henry", "Samuel",
]

INTL_FEMALE_FIRST = [
    "Mary", "Sarah", "Emma", "Olivia", "Sophia",
    "Emily", "Grace", "Elizabeth", "Chloe", "Mia",
    "Ava", "Isabella", "Charlotte", "Harper", "Evelyn",
]

NIGERIAN_SURNAMES = [
    "Okonkwo", "Adeleke", "Balogun", "Eze", "Nwosu",
    "Danjuma", "Bello", "Abubakar", "Oladipo", "Okafor",
    "Alabi", "Lawal", "Soyinka", "Achebe", "Sanusi",
    "Ogundipe", "Bamidele", "Chukwu", "Mohammed", "Yakubu",
]

INTL_SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Miller", "Davis", "Wilson", "Taylor",
    "Anderson", "Thomas", "Martinez", "Clark", "Robinson",
    "White", "Hall", "Harris", "Martin", "Thompson",
]

BASE_YEAR: int = 2026


def parse_dob(dob_str: str) -> Tuple[date, str]:
    """Parse DOB from string in ISO (YYYY-MM-DD) or European (DD/MM/YYYY) format.

    Returns:
        Tuple of (parsed_date, original_format_type: 'ISO' | 'DMY').
    """
    dob_str = dob_str.strip()
    if "/" in dob_str:
        dt = datetime.strptime(dob_str, "%d/%m/%Y").date()
        return dt, "DMY"
    else:
        dt = datetime.strptime(dob_str, "%Y-%m-%d").date()
        return dt, "ISO"


def format_dob(d: date, fmt: str = "ISO") -> str:
    """Format date to string in specified format ('ISO' or 'DMY')."""
    if fmt == "DMY":
        return d.strftime("%d/%m/%Y")
    return d.strftime("%Y-%m-%d")


def swap_name(full_name: str) -> str:
    """Swap the components of a full name (e.g., 'First Last' -> 'Last First')."""
    parts = full_name.strip().split()
    if len(parts) >= 2:
        return f"{parts[-1]} {' '.join(parts[:-1])}"
    return full_name


def transpose_characters(name: str, rng: Optional[random.Random] = None) -> str:
    """Transpose two adjacent characters within a name."""
    r = rng if rng is not None else random
    words = name.split()
    valid_word_indices = [i for i, w in enumerate(words) if len(w) >= 3]
    if not valid_word_indices:
        return name

    chosen_word_idx = r.choice(valid_word_indices)
    word = list(words[chosen_word_idx])
    # Pick adjacent index to swap
    idx = r.randint(0, len(word) - 2)
    word[idx], word[idx + 1] = word[idx + 1], word[idx]
    words[chosen_word_idx] = "".join(word)
    return " ".join(words)


def shift_dob(
    dob_str: str,
    days: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> str:
    """Shift DOB by +/- 1 day while preserving format."""
    r = rng if rng is not None else random
    dt, fmt = parse_dob(dob_str)
    if days is None:
        days = r.choice([-1, 1])
    new_dt = dt + timedelta(days=days)
    return format_dob(new_dt, fmt)


def toggle_dob_format(dob_str: str) -> str:
    """Toggle date format between YYYY-MM-DD and DD/MM/YYYY."""
    dt, fmt = parse_dob(dob_str)
    target_fmt = "DMY" if fmt == "ISO" else "ISO"
    return format_dob(dt, target_fmt)


def perturb_identity(
    identity: Dict[str, Any],
    perturbation_types: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply identity perturbations to a patient profile.

    Perturbation types supported:
        - 'swapped_names': 'First Last' -> 'Last First'
        - 'character_transposition': Swap adjacent characters in name
        - 'shifted_dob': Shift birthdate by +/- 1 day
        - 'format_variant': Toggle between ISO (YYYY-MM-DD) and DD/MM/YYYY

    Args:
        identity: Patient identity dictionary.
        perturbation_types: Specific list of perturbations to apply.
            If None, randomly chooses 1 or 2 perturbation types.
        seed: Random seed for deterministic perturbation.

    Returns:
        Perturbed copy of identity dictionary.
    """
    rng = random.Random(seed) if seed is not None else random
    perturbed = dict(identity)

    available_types = [
        "swapped_names",
        "character_transposition",
        "shifted_dob",
        "format_variant",
    ]

    if perturbation_types is None:
        # Pick 1 or 2 distinct perturbations
        k = rng.choice([1, 2])
        chosen_types = rng.sample(available_types, k=k)
    else:
        chosen_types = list(perturbation_types)

    name = perturbed.get("full_name", "")
    dob = perturbed.get("dob", "")

    for p_type in chosen_types:
        if p_type in ("swapped_names", "swapped_name"):
            name = swap_name(name)
        elif p_type in ("character_transposition", "transposition"):
            name = transpose_characters(name, rng=rng)
        elif p_type in ("shifted_dob", "shift_dob"):
            dob = shift_dob(dob, rng=rng)
        elif p_type in ("format_variant", "format_variants"):
            dob = toggle_dob_format(dob)

    perturbed["full_name"] = name
    perturbed["dob"] = dob
    return perturbed


def generate_patient_profiles(
    df: pd.DataFrame,
    seed: int = 42,
    base_year: int = BASE_YEAR,
) -> pd.DataFrame:
    """Generate synthetic patient profiles for all rows in a clinical DataFrame.

    Fields generated:
        - patient_id: Unique record ID (e.g. 'PID_cleveland_0001')
        - full_name: Realistic name sampled from Nigerian and international pools
        - dob: Date of birth (YYYY-MM-DD) calculated from age assuming base_year (2026)
        - sex: Preserved from clinical data (1=Male, 0=Female)
        - clinic_id: Node indicator
        - ground_truth_patient_id: Identifier tracking true entity across records

    Args:
        df: Input DataFrame containing 'age', 'sex', and 'clinic_id'.
        seed: Random seed for reproducible generation.
        base_year: Reference year for age calculation (default 2026).

    Returns:
        DataFrame with identity columns added.
    """
    rng = np.random.RandomState(seed)
    py_rng = random.Random(seed)

    out_df = df.copy()
    n = len(out_df)

    patient_ids: List[str] = []
    full_names: List[str] = []
    dobs: List[str] = []
    gt_patient_ids: List[str] = []

    for idx, row in out_df.iterrows():
        clinic = str(row.get("clinic_id", "clinic"))
        pid = f"PID_{clinic}_{idx:04d}"
        patient_ids.append(pid)
        gt_patient_ids.append(pid)

        # Sex: 1 is male, 0 is female
        is_male = bool(row.get("sex", 1) == 1)

        # 50% Nigerian, 50% International name distribution
        use_nigerian = py_rng.random() < 0.5
        if use_nigerian:
            first = py_rng.choice(NIGERIAN_MALE_FIRST if is_male else NIGERIAN_FEMALE_FIRST)
            last = py_rng.choice(NIGERIAN_SURNAMES)
        else:
            first = py_rng.choice(INTL_MALE_FIRST if is_male else INTL_FEMALE_FIRST)
            last = py_rng.choice(INTL_SURNAMES)

        full_names.append(f"{first} {last}")

        # Calculate DOB from age
        age = float(row.get("age", 50.0))
        if np.isnan(age):
            age = 50.0
        birth_year = base_year - int(round(age))
        birth_month = py_rng.randint(1, 12)
        # Using 1..28 ensures valid dates across all months
        birth_day = py_rng.randint(1, 28)
        dob = date(birth_year, birth_month, birth_day)
        dobs.append(dob.strftime("%Y-%m-%d"))

    out_df["patient_id"] = patient_ids
    out_df["full_name"] = full_names
    out_df["dob"] = dobs
    out_df["ground_truth_patient_id"] = gt_patient_ids

    return out_df
