"""Entity Resolution module for CareTrail Phase 1.

Performs cross-clinic patient identity resolution:
1. Blocks on sex + birth year window (+/- 1 year).
2. Scores candidate pairs using RapidFuzz (Jaro-Winkler + token sort) and DOB matching.
3. Builds connected components using NetworkX to form patient clusters.
4. Evaluates Precision, Recall, F1 against ground-truth and exports failure cases (FP, FN).
"""

from collections import defaultdict
from datetime import date, datetime
import itertools
from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def parse_dob_safely(dob_str: Any) -> Optional[Tuple[int, int, int]]:
    """Parse date string in either ISO (YYYY-MM-DD) or DMY (DD/MM/YYYY) format.

    Returns:
        Tuple of (year, month, day) or None if unparseable.
    """
    if pd.isna(dob_str):
        return None
    s = str(dob_str).strip()
    try:
        if "/" in s:
            dt = datetime.strptime(s, "%d/%m/%Y").date()
        else:
            dt = datetime.strptime(s, "%Y-%m-%d").date()
        return (dt.year, dt.month, dt.day)
    except Exception:
        return None


def calculate_dob_similarity(dob1: Optional[Tuple[int, int, int]], dob2: Optional[Tuple[int, int, int]]) -> float:
    """Compute similarity between two birth dates."""
    if dob1 is None or dob2 is None:
        return 0.0

    y1, m1, d1 = dob1
    y2, m2, d2 = dob2

    if (y1, m1, d1) == (y2, m2, d2):
        return 1.0

    # Check day difference
    try:
        dt1 = date(y1, m1, d1)
        dt2 = date(y2, m2, d2)
        diff_days = abs((dt1 - dt2).days)
        if diff_days == 1:
            return 0.95
        if diff_days == 2:
            return 0.88
    except Exception:
        pass

    # Transposed day and month (e.g. 05/10 vs 10/05)
    if y1 == y2 and m1 == d2 and d1 == m2:
        return 0.90

    # Same year and month
    if y1 == y2 and m1 == m2:
        return 0.75

    # Same year, different month
    if y1 == y2:
        return 0.50

    # Birth year off by 1 year, same month/day
    if abs(y1 - y2) == 1 and m1 == m2 and d1 == d2:
        return 0.25

    return 0.10


def calculate_name_similarity(name1: str, name2: str) -> float:
    """Compute name similarity combining RapidFuzz token sort and Jaro-Winkler.

    - Token sort handles swapped names ('Emeka Okonkwo' vs 'Okonkwo Emeka').
    - Jaro-Winkler handles character transpositions and minor typos ('Emeka' vs 'Eemka').
    - Non-matching tokens or low token sort scores receive a stronger penalty
      to prevent common first/last names from inflating similarity.
    """
    n1 = str(name1).strip().lower()
    n2 = str(name2).strip().lower()

    if n1 == n2:
        return 1.0

    t1 = n1.split()
    t2 = n2.split()
    if not t1 or not t2:
        return 0.0

    # Pairwise token alignment: find best match for each token
    forward_sims = [
        max(max(JaroWinkler.similarity(w1, w2), fuzz.ratio(w1, w2) / 100.0) for w2 in t2)
        for w1 in t1
    ]
    backward_sims = [
        max(max(JaroWinkler.similarity(w2, w1), fuzz.ratio(w2, w1) / 100.0) for w1 in t1)
        for w2 in t2
    ]
    all_sims = forward_sims + backward_sims
    min_token_sim = min(all_sims)
    avg_token_sim = sum(all_sims) / len(all_sims)

    jw = JaroWinkler.similarity(n1, n2)
    token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0

    # Stronger penalty if token sort / token match is low (e.g. shared first name only)
    if min_token_sim < 0.65:
        return float(min(token_sort, min_token_sim * 0.5 + avg_token_sim * 0.35))

    return float(max(token_sort, jw, avg_token_sim))


def score_candidate_pair(
    row1: Dict[str, Any],
    row2: Dict[str, Any],
    dob1: Optional[Tuple[int, int, int]],
    dob2: Optional[Tuple[int, int, int]],
    name_weight: float = 0.65,
    dob_weight: float = 0.35,
) -> float:
    """Calculate composite similarity score for two patient profiles."""
    name_sim = calculate_name_similarity(row1.get("full_name", ""), row2.get("full_name", ""))
    dob_sim = calculate_dob_similarity(dob1, dob2)

    composite = (name_weight * name_sim) + (dob_weight * dob_sim)
    return float(composite)


def generate_candidate_blocks(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """Generate candidate pairs across clinics blocked on sex + birth year window (+/- 1 year).

    Returns:
        List of index pairs (i, j) with i < j, clinic_id[i] != clinic_id[j],
        same sex, and abs(birth_year[i] - birth_year[j]) <= 1.
    """
    parsed_dobs: List[Optional[Tuple[int, int, int]]] = [
        parse_dob_safely(d) for d in df["dob"]
    ]

    # Index rows by (sex, birth_year)
    blocks: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for idx, (sex, parsed_dob) in enumerate(zip(df["sex"], parsed_dobs)):
        if parsed_dob is not None and not pd.isna(sex):
            s = int(sex)
            y = parsed_dob[0]
            blocks[(s, y)].append(idx)

    clinics = df["clinic_id"].values
    candidate_pairs: Set[Tuple[int, int]] = set()

    for (sex, year), idx_list in blocks.items():
        # Compare with year - 1, year, year + 1
        for comp_year in [year - 1, year, year + 1]:
            other_list = blocks.get((sex, comp_year), [])
            for i in idx_list:
                for j in other_list:
                    if i < j and clinics[i] != clinics[j]:
                        candidate_pairs.add((i, j))

    return sorted(list(candidate_pairs))


def resolve_entities(
    df: pd.DataFrame,
    threshold: float = 0.83,
    name_weight: float = 0.65,
    dob_weight: float = 0.35,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """Execute the full entity resolution pipeline on a cohort DataFrame.

    Pipeline:
    1. Blocking: sex + birth year window (+/- 1 year).
    2. Scoring: RapidFuzz name similarity + DOB similarity.
    3. Graph clustering: NetworkX connected components.
    4. Evaluation: Precision, Recall, F1 against ground_truth_patient_id.
    5. Failure case extraction: Table of False Positives and False Negatives.

    Args:
        df: Input DataFrame containing patient_id, full_name, dob, sex, clinic_id,
            and ground_truth_patient_id.
        threshold: Match score threshold for linking candidate pairs (default 0.83).
        name_weight: Weight for name similarity (default 0.65).
        dob_weight: Weight for DOB similarity (default 0.35).

    Returns:
        Tuple of:
            - clustered_df: Original DataFrame with added 'cluster_id' column.
            - metrics: Dict with 'precision', 'recall', 'f1', 'tp', 'fp', 'fn'.
            - failures_df: DataFrame documenting FP and FN cases.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)

    parsed_dobs = [parse_dob_safely(d) for d in df["dob"]]
    records = df.to_dict(orient="records")

    candidate_pairs = generate_candidate_blocks(df)

    # Graph for clustering
    G = nx.Graph()
    # Add all patient_ids as nodes
    for pid in df["patient_id"]:
        G.add_node(pid)

    scored_pairs: Dict[Tuple[str, str], float] = {}

    for i, j in candidate_pairs:
        rec1 = records[i]
        rec2 = records[j]
        score = score_candidate_pair(
            rec1, rec2, parsed_dobs[i], parsed_dobs[j],
            name_weight=name_weight, dob_weight=dob_weight
        )
        pid1 = rec1["patient_id"]
        pid2 = rec2["patient_id"]
        pair_key = (min(pid1, pid2), max(pid1, pid2))
        scored_pairs[pair_key] = score

        if score >= threshold:
            G.add_edge(pid1, pid2, weight=score)

    # Form clusters via connected components
    components = list(nx.connected_components(G))
    patient_to_cluster: Dict[str, str] = {}
    for cluster_idx, comp in enumerate(components):
        cluster_id = f"cluster_{cluster_idx:05d}"
        for pid in comp:
            patient_to_cluster[pid] = cluster_id

    df["cluster_id"] = df["patient_id"].map(patient_to_cluster)

    # --- Evaluation against Ground Truth ---
    # Find all ground truth duplicate pairs: (pid1, pid2) where gt_id is identical
    gt_map: Dict[str, List[str]] = defaultdict(list)
    for pid, gt_id in zip(df["patient_id"], df["ground_truth_patient_id"]):
        gt_map[gt_id].append(pid)

    gt_pairs: Set[Tuple[str, str]] = set()
    for gt_id, pids in gt_map.items():
        if len(pids) > 1:
            for p1, p2 in itertools.combinations(sorted(pids), 2):
                gt_pairs.add((p1, p2))

    # Predicted pairs: pairs in the same cluster
    cluster_map: Dict[str, List[str]] = defaultdict(list)
    for pid, c_id in zip(df["patient_id"], df["cluster_id"]):
        cluster_map[c_id].append(pid)

    pred_pairs: Set[Tuple[str, str]] = set()
    for c_id, pids in cluster_map.items():
        if len(pids) > 1:
            for p1, p2 in itertools.combinations(sorted(pids), 2):
                pred_pairs.add((p1, p2))

    tp_pairs = pred_pairs.intersection(gt_pairs)
    fp_pairs = pred_pairs - gt_pairs
    fn_pairs = gt_pairs - pred_pairs

    tp = len(tp_pairs)
    fp = len(fp_pairs)
    fn = len(fn_pairs)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "total_gt_pairs": len(gt_pairs),
        "total_pred_pairs": len(pred_pairs),
    }

    # Extract failure cases table
    failure_records: List[Dict[str, Any]] = []
    pid_to_row = {r["patient_id"]: r for r in records}

    # FP cases
    for p1, p2 in sorted(fp_pairs):
        r1 = pid_to_row[p1]
        r2 = pid_to_row[p2]
        pair_key = (min(p1, p2), max(p1, p2))
        sc = scored_pairs.get(pair_key, 0.0)
        failure_records.append({
            "error_type": "FP",
            "patient_id_1": p1,
            "patient_id_2": p2,
            "gt_id_1": r1["ground_truth_patient_id"],
            "gt_id_2": r2["ground_truth_patient_id"],
            "name_1": r1["full_name"],
            "name_2": r2["full_name"],
            "dob_1": r1["dob"],
            "dob_2": r2["dob"],
            "clinic_1": r1["clinic_id"],
            "clinic_2": r2["clinic_id"],
            "score": round(sc, 4),
        })

    # FN cases
    for p1, p2 in sorted(fn_pairs):
        r1 = pid_to_row[p1]
        r2 = pid_to_row[p2]
        pair_key = (min(p1, p2), max(p1, p2))
        sc = scored_pairs.get(pair_key, 0.0)
        failure_records.append({
            "error_type": "FN",
            "patient_id_1": p1,
            "patient_id_2": p2,
            "gt_id_1": r1["ground_truth_patient_id"],
            "gt_id_2": r2["ground_truth_patient_id"],
            "name_1": r1["full_name"],
            "name_2": r2["full_name"],
            "dob_1": r1["dob"],
            "dob_2": r2["dob"],
            "clinic_1": r1["clinic_id"],
            "clinic_2": r2["clinic_id"],
            "score": round(sc, 4),
        })

    failures_df = pd.DataFrame(failure_records)

    return df, metrics, failures_df
