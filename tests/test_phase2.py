"""Unit tests for CareTrail Phase 2.

Verifies:
1. Tamper test: modifying 1 byte in a ledger entry or manifest root causes verify_chain() to fail.
2. Merkle proof test: inclusion verified with valid proof, rejected with invalid proof or tampered leaf.
3. Consent revocation test: revoked patient never appears in any subsequent training manifest.
4. Determinism test: fixed random seeds produce identical FedAvg weights and hashes.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from src.consent import ConsentRegistry, ConsentState
from src.federated import (
    aggregate_fedavg,
    run_fedavg,
    train_node_local,
)
from src.ledger import (
    EMPTY_HASH,
    GENESIS_PREV_HASH,
    AuditLedger,
    canonical_hash_entry,
    compute_merkle_root,
    compute_patient_commitment,
    get_merkle_proof,
    hash_model_weights,
    verify_chain,
    verify_merkle_proof,
)
from src.model import LogisticRegressionModel


def test_tamper_ledger_verification(tmp_path: Path):
    """Test 1: Verify that tampering 1 byte in a ledger entry causes verify_chain() to fail."""
    ledger_file = tmp_path / "test_audit_ledger.jsonl"
    ledger = AuditLedger(ledger_path=ledger_file)

    # 1. Append valid entries
    e0 = ledger.append_entry(
        event_type="round_commit",
        node_id="cleveland",
        round_idx=0,
        manifest_root="a" * 64,
        update_hash="b" * 64,
        model_hash="c" * 64,
    )
    e1 = ledger.append_entry(
        event_type="round_commit",
        node_id="hungarian",
        round_idx=0,
        manifest_root="d" * 64,
        update_hash="e" * 64,
        model_hash="f" * 64,
    )
    e2 = ledger.append_entry(
        event_type="model_release",
        node_id="coordinator",
        round_idx=1,
        manifest_root=EMPTY_HASH,
        update_hash=EMPTY_HASH,
        model_hash="1" * 64,
    )

    # Chain should be valid
    assert verify_chain(ledger_file) is True, "Original ledger chain must be valid"

    # 2. Tamper manifest_root in entry 1 by altering 1 byte
    lines = ledger_file.read_text(encoding="utf-8").strip().split("\n")
    entry_to_tamper = json.loads(lines[1])
    # Alter the last character of manifest_root
    original_root = entry_to_tamper["manifest_root"]
    tampered_root = original_root[:-1] + ("0" if original_root[-1] != "0" else "1")
    entry_to_tamper["manifest_root"] = tampered_root
    lines[1] = json.dumps(entry_to_tamper, sort_keys=True)

    tampered_ledger_file = tmp_path / "tampered_ledger.jsonl"
    tampered_ledger_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # verify_chain MUST fail on tampered ledger
    assert verify_chain(tampered_ledger_file) is False, (
        "Tampering manifest_root must cause verify_chain() to fail"
    )

    # 3. Tamper previous hash continuity
    entry_prev_tampered = json.loads(lines[2])
    entry_prev_tampered["prev_hash"] = "9" * 64
    lines[2] = json.dumps(entry_prev_tampered, sort_keys=True)
    tampered_prev_file = tmp_path / "tampered_prev_ledger.jsonl"
    tampered_prev_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert verify_chain(tampered_prev_file) is False, (
        "Tampering prev_hash continuity must cause verify_chain() to fail"
    )


def test_merkle_proof_verification():
    """Test 2: Verify inclusion with valid Merkle proof, reject with invalid/tampered proof."""
    patient_ids = [f"PID_{i:04d}" for i in range(11)]
    salts = [f"salt_{i:04d}" for i in range(11)]
    leaves = [compute_patient_commitment(pid, s) for pid, s in zip(patient_ids, salts)]

    root = compute_merkle_root(leaves)
    assert len(root) == 64, "Root must be a 64-character SHA-256 hex string"

    # Verify every leaf can be proven and verified
    for leaf in leaves:
        proof = get_merkle_proof(leaf, leaves)
        assert verify_merkle_proof(leaf, proof, root) is True, (
            f"Valid proof for leaf {leaf[:8]} failed verification"
        )

    # Negative test 1: Tampered leaf hash
    tampered_leaf = compute_patient_commitment("PID_FAKE", "salt_FAKE")
    proof_sample = get_merkle_proof(leaves[0], leaves)
    assert verify_merkle_proof(tampered_leaf, proof_sample, root) is False, (
        "Tampered leaf must fail verification against legitimate proof"
    )

    # Negative test 2: Tampered proof step
    tampered_proof = [dict(step) for step in proof_sample]
    tampered_proof[0]["hash"] = "f" * 64
    assert verify_merkle_proof(leaves[0], tampered_proof, root) is False, (
        "Tampered proof step must fail verification"
    )

    # Negative test 3: Non-existent leaf lookup raises ValueError
    with pytest.raises(ValueError):
        get_merkle_proof("non_existent_leaf_hash", leaves)


def test_consent_revocation_exclusion(tmp_path: Path):
    """Test 3: Revoked patient never appears in any subsequent training manifest."""
    ledger_file = tmp_path / "consent_audit_ledger.jsonl"
    ledger = AuditLedger(ledger_path=ledger_file)
    registry = ConsentRegistry(ledger=ledger)

    all_pids = [f"PID_nodeA_{i:03d}" for i in range(10)]
    for pid in all_pids:
        registry.register_patient(pid, state=ConsentState.ACTIVE)

    # Synthetic node cohort
    df = pd.DataFrame({
        "patient_id": all_pids,
        "clinic_id": ["nodeA"] * len(all_pids),
        "target": [0, 1] * 5,
    })

    # Round 0: All 10 active
    active_df_r0 = registry.filter_active_records(df)
    assert len(active_df_r0) == 10
    root_r0, leaves_r0 = registry.record_manifest(
        round_idx=0, node_id="nodeA", patient_ids=active_df_r0["patient_id"].tolist()
    )
    assert len(leaves_r0) == 10

    # Audit PID_nodeA_003 before revocation: must be included
    audit_pre = registry.audit("PID_nodeA_003", round_idx=0, node_id="nodeA")
    assert audit_pre["included"] is True
    assert audit_pre["verified"] is True

    # Patient PID_nodeA_003 revokes consent
    revocation_entry = registry.revoke_consent(
        patient_id="PID_nodeA_003",
        round_idx=0,
        node_id="nodeA",
    )
    assert registry.is_active("PID_nodeA_003") is False
    assert revocation_entry["event_type"] == "consent_revoked"
    assert revocation_entry["node_id"] == "nodeA"

    # Round 1: strictly exclude revoked patient PID_nodeA_003
    active_df_r1 = registry.filter_active_records(df)
    assert len(active_df_r1) == 9
    assert "PID_nodeA_003" not in active_df_r1["patient_id"].values

    root_r1, leaves_r1 = registry.record_manifest(
        round_idx=1, node_id="nodeA", patient_ids=active_df_r1["patient_id"].tolist()
    )
    assert len(leaves_r1) == 9

    # Verify revoked patient's commitment is completely absent from round 1 manifest
    revoked_commitment = registry.get_commitment("PID_nodeA_003")
    assert revoked_commitment not in leaves_r1

    # Audit PID_nodeA_003 in Round 1: must report NOT included
    audit_post = registry.audit("PID_nodeA_003", round_idx=1, node_id="nodeA")
    assert audit_post["included"] is False
    assert audit_post["verified"] is True  # Non-inclusion confirmed

    # Audit an active patient PID_nodeA_000 in Round 1: must be included and verified
    audit_active = registry.audit("PID_nodeA_000", round_idx=1, node_id="nodeA")
    assert audit_active["included"] is True
    assert audit_active["verified"] is True


def test_fedavg_determinism():
    """Test 4: Fixed random seeds produce identical FedAvg weights and cryptographic hashes."""
    def run_simulation(seed: int):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 4 mock clinical nodes with 13 features
        clinics = ["cleveland", "hungarian", "switzerland", "va"]
        node_data = {}
        for c_idx, c in enumerate(clinics):
            rng = np.random.RandomState(seed + c_idx)
            X = rng.randn(40, 13).astype(np.float32)
            y = rng.randint(0, 2, size=(40,)).astype(np.float32)
            node_data[c] = (X, y)

        global_model = LogisticRegressionModel(in_features=13)

        for r in range(3):
            curr_w = global_model.get_weights()
            node_weights = {}
            node_sizes = {}
            for c_idx, c in enumerate(clinics):
                X_k, y_k = node_data[c]
                node_sizes[c] = len(X_k)
                node_seed = seed + r * 100 + c_idx
                w_k = train_node_local(
                    curr_w,
                    X_k,
                    y_k,
                    epochs=2,
                    lr=0.05,
                    optimizer_name="adam",
                    seed=node_seed,
                )
                node_weights[c] = w_k

            new_w = aggregate_fedavg(node_weights, node_sizes)
            global_model.set_weights(new_w)

        final_weights = global_model.get_weights()
        model_hash = hash_model_weights(final_weights)
        return final_weights, model_hash

    # Run twice with the exact same seed
    w1, h1 = run_simulation(seed=123)
    w2, h2 = run_simulation(seed=123)

    assert h1 == h2, f"Model hashes must match under identical seeds: {h1} vs {h2}"

    for param_name in w1.keys():
        assert torch.equal(w1[param_name], w2[param_name]), (
            f"Weights for {param_name} differ across deterministic runs"
        )
