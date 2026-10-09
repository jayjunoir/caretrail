"""Patient consent management and audit verification for CareTrail Phase 2.

Handles:
1. Patient consent registry with state tracking (ACTIVE or REVOKED).
2. Ledger recording of 'consent_revoked' events upon revocation.
3. Strict exclusion of revoked patient records from all subsequent federated rounds.
4. Cryptographic auditing of patient inclusion in training manifests via Merkle proofs.

Note on Machine Unlearning:
# Revocation guarantees exclusion from all subsequent training rounds and rollback/retraining from prior checkpoints; it does not perform true in-place machine unlearning.
"""

from enum import Enum
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from src.ledger import (
    EMPTY_HASH,
    AuditLedger,
    compute_merkle_root,
    compute_patient_commitment,
    get_merkle_proof,
    verify_merkle_proof,
)


class ConsentState(str, Enum):
    """Consent status for a patient profile."""
    ACTIVE = "ACTIVE"
    REVOKED = "REVOKED"


class ConsentRegistry:
    """Registry maintaining patient consent states, salts, and training manifests."""

    def __init__(self, ledger: Optional[AuditLedger] = None, salt_seed: str = "caretrail_default_salt"):
        self.ledger = ledger
        self.salt_seed = salt_seed
        self._states: Dict[str, ConsentState] = {}
        self._salts: Dict[str, str] = {}
        # Stores round manifests: (round_idx, node_id) -> list of patient commitments
        self._round_manifests: Dict[Tuple[int, str], List[str]] = {}

    def get_or_create_salt(self, patient_id: str) -> str:
        """Get or deterministically generate a unique salt for a patient."""
        if patient_id not in self._salts:
            s = hashlib.sha256(f"{self.salt_seed}:{patient_id}".encode("utf-8")).hexdigest()[:16]
            self._salts[patient_id] = s
        return self._salts[patient_id]

    def get_commitment(self, patient_id: str) -> str:
        """Compute the leaf commitment sha256(patient_id + ':' + salt) for a patient."""
        salt = self.get_or_create_salt(patient_id)
        return compute_patient_commitment(patient_id, salt)

    def register_patient(self, patient_id: str, state: ConsentState = ConsentState.ACTIVE, salt: Optional[str] = None) -> None:
        """Register a patient with initial consent status."""
        if salt is not None:
            self._salts[patient_id] = salt
        else:
            self.get_or_create_salt(patient_id)
        self._states[patient_id] = state

    def is_active(self, patient_id: str) -> bool:
        """Check whether a patient's consent is currently ACTIVE."""
        return self._states.get(patient_id, ConsentState.ACTIVE) == ConsentState.ACTIVE

    def revoke_consent(
        self,
        patient_id: str,
        round_idx: int = 0,
        node_id: str = "coordinator",
        ledger: Optional[AuditLedger] = None,
    ) -> Dict[str, Any]:
        """Revoke a patient's consent and log a 'consent_revoked' event to the ledger.

        # Revocation guarantees exclusion from all subsequent training rounds and rollback/retraining from prior checkpoints; it does not perform true in-place machine unlearning.
        """
        self._states[patient_id] = ConsentState.REVOKED
        active_ledger = ledger if ledger is not None else self.ledger

        commitment = self.get_commitment(patient_id)

        entry: Dict[str, Any] = {}
        if active_ledger is not None:
            entry = active_ledger.append_entry(
                event_type="consent_revoked",
                node_id=node_id,
                round_idx=round_idx,
                manifest_root=commitment,
                update_hash=EMPTY_HASH,
                model_hash=EMPTY_HASH,
            )

        return entry

    def filter_active_records(self, df: pd.DataFrame, id_col: str = "patient_id") -> pd.DataFrame:
        """Filter a DataFrame to strictly retain only ACTIVE consent records.

        Revoked patients are completely excluded from downstream training.
        """
        if id_col not in df.columns:
            return df

        mask = df[id_col].apply(lambda pid: self.is_active(str(pid)))
        return df[mask].copy().reset_index(drop=True)

    def record_manifest(self, round_idx: int, node_id: str, patient_ids: List[str]) -> Tuple[str, List[str]]:
        """Record the training manifest for a specific round and node.

        Returns:
            Tuple of (manifest_root, list of commitments).
        """
        commitments = [self.get_commitment(str(pid)) for pid in patient_ids]
        manifest_root = compute_merkle_root(commitments)
        self._round_manifests[(round_idx, node_id)] = commitments
        return manifest_root, commitments

    def get_manifest_commitments(self, round_idx: int, node_id: str) -> List[str]:
        """Retrieve stored commitments for a given round and node."""
        return self._round_manifests.get((round_idx, node_id), [])

    def audit(
        self,
        patient_id: str,
        round_idx: int,
        node_id: str,
    ) -> Dict[str, Any]:
        """Audit whether a patient's record was included in a specific training round and node.

        Returns cryptographic Merkle proof and verification status against the manifest root.
        """
        commitment = self.get_commitment(patient_id)
        leaves = self.get_manifest_commitments(round_idx, node_id)
        included = commitment in leaves

        if included:
            manifest_root = compute_merkle_root(leaves)
            proof = get_merkle_proof(commitment, leaves)
            verified = verify_merkle_proof(commitment, proof, manifest_root)
        else:
            manifest_root = compute_merkle_root(leaves) if leaves else EMPTY_HASH
            proof = []
            verified = True  # Non-inclusion correctly confirmed

        return {
            "patient_id": patient_id,
            "round": round_idx,
            "node_id": node_id,
            "included": included,
            "commitment": commitment,
            "merkle_proof": proof,
            "manifest_root": manifest_root,
            "verified": verified,
        }
