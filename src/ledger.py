"""Audit Ledger module for CareTrail Phase 2.

Implements:
1. SHA-256 hash-chained append-only ledger stored as JSONL.
2. Merkle tree over salted commitments of patient IDs for data minimization:
   - Leaf commitment: sha256(patient_id + ":" + salt).
   - Stores ONLY cryptographic hashes and commitments, NEVER raw health data or identities.
3. Cryptographic Merkle inclusion proofs (generation and verification).
4. Ledger chain verification and tamper detection.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch


GENESIS_PREV_HASH: str = "0" * 64
EMPTY_HASH: str = hashlib.sha256(b"").hexdigest()


def compute_patient_commitment(patient_id: str, salt: str) -> str:
    """Compute cryptographic commitment for a patient record: sha256(patient_id + ':' + salt).

    Ensures no raw identities or patient health records are written to the ledger.
    """
    token = f"{patient_id}:{salt}".encode("utf-8")
    return hashlib.sha256(token).hexdigest()


def compute_merkle_root(leaves: List[str]) -> str:
    """Compute the Merkle root over a list of leaf hash commitments.

    If the number of nodes at any level is odd, the last node is duplicated.
    """
    if not leaves:
        return EMPTY_HASH

    current_level = list(leaves)
    while len(current_level) > 1:
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])
        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            combined = (current_level[i] + current_level[i + 1]).encode("utf-8")
            next_level.append(hashlib.sha256(combined).hexdigest())
        current_level = next_level

    return current_level[0]


def get_merkle_proof(record_commitment: str, leaves: List[str]) -> List[Dict[str, str]]:
    """Generate a Merkle inclusion proof for a given leaf commitment.

    Args:
        record_commitment: The target leaf commitment to prove.
        leaves: The full list of leaves in the tree.

    Returns:
        List of dicts representing the proof path:
        [{"position": "left" | "right", "hash": sibling_hash}, ...]

    Raises:
        ValueError: If record_commitment is not found in leaves.
    """
    if record_commitment not in leaves:
        raise ValueError(f"Leaf commitment {record_commitment} not present in leaves list")

    idx = leaves.index(record_commitment)
    current_level = list(leaves)
    proof: List[Dict[str, str]] = []

    while len(current_level) > 1:
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])

        if idx % 2 == 0:
            sibling = current_level[idx + 1]
            proof.append({"position": "right", "hash": sibling})
        else:
            sibling = current_level[idx - 1]
            proof.append({"position": "left", "hash": sibling})

        next_level: List[str] = []
        for i in range(0, len(current_level), 2):
            combined = (current_level[i] + current_level[i + 1]).encode("utf-8")
            next_level.append(hashlib.sha256(combined).hexdigest())

        idx = idx // 2
        current_level = next_level

    return proof


def verify_merkle_proof(
    leaf: str,
    proof: List[Union[Dict[str, str], Tuple[str, str]]],
    root: str,
) -> bool:
    """Verify that a given leaf belongs to the Merkle tree with the given root.

    Args:
        leaf: Leaf commitment hash.
        proof: Inclusion proof steps.
        root: Expected Merkle root hash.

    Returns:
        True if the proof validly evaluates to the root, False otherwise.
    """
    curr = leaf
    for step in proof:
        if isinstance(step, dict):
            pos = step.get("position")
            sibling = step.get("hash", "")
        elif isinstance(step, (tuple, list)) and len(step) == 2:
            pos, sibling = step[0], step[1]
        else:
            return False

        if pos == "right":
            combined = (curr + sibling).encode("utf-8")
            curr = hashlib.sha256(combined).hexdigest()
        elif pos == "left":
            combined = (sibling + curr).encode("utf-8")
            curr = hashlib.sha256(combined).hexdigest()
        else:
            return False

    return curr == root


def canonical_hash_entry(entry: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of an entry's content excluding 'entry_hash'."""
    content = {k: v for k, v in entry.items() if k != "entry_hash"}
    canonical_json = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def hash_model_weights(weights: Dict[str, torch.Tensor]) -> str:
    """Compute deterministic SHA-256 hash of PyTorch model parameters."""
    hasher = hashlib.sha256()
    for name in sorted(weights.keys()):
        tensor_bytes = weights[name].detach().cpu().numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(tensor_bytes)
    return hasher.hexdigest()


def hash_model_delta(
    w_new: Dict[str, torch.Tensor],
    w_old: Optional[Dict[str, torch.Tensor]] = None,
) -> str:
    """Compute deterministic SHA-256 hash of model update delta."""
    hasher = hashlib.sha256()
    for name in sorted(w_new.keys()):
        if w_old is not None and name in w_old:
            delta = (w_new[name] - w_old[name]).detach().cpu().numpy().tobytes()
        else:
            delta = w_new[name].detach().cpu().numpy().tobytes()
        hasher.update(name.encode("utf-8"))
        hasher.update(delta)
    return hasher.hexdigest()


def verify_chain(ledger_path: Union[str, Path] = "results/audit_ledger.jsonl") -> bool:
    """Validate ledger chain continuity and hash integrity from genesis.

    Args:
        ledger_path: Path to the JSONL ledger file.

    Returns:
        True if all entries are authentic and chain is intact, False otherwise.
    """
    path = Path(ledger_path)
    if not path.exists():
        return True

    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                entries.append(json.loads(line_str))

    if not entries:
        return True

    for idx, entry in enumerate(entries):
        # 1. Index check
        if entry.get("index") != idx:
            return False

        # 2. Previous hash check
        expected_prev = GENESIS_PREV_HASH if idx == 0 else entries[idx - 1]["entry_hash"]
        if entry.get("prev_hash") != expected_prev:
            return False

        # 3. Canonical entry hash check
        computed_hash = canonical_hash_entry(entry)
        if entry.get("entry_hash") != computed_hash:
            return False

    return True


class AuditLedger:
    """Append-only, SHA-256 hash-chained audit ledger stored in JSONL format."""

    def __init__(self, ledger_path: Union[str, Path] = "results/audit_ledger.jsonl"):
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def get_entries(self) -> List[Dict[str, Any]]:
        """Read all entries from the ledger."""
        if not self.ledger_path.exists():
            return []

        entries: List[Dict[str, Any]] = []
        with open(self.ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    entries.append(json.loads(s))
        return entries

    def get_last_entry(self) -> Optional[Dict[str, Any]]:
        """Retrieve the most recent entry in the ledger."""
        entries = self.get_entries()
        return entries[-1] if entries else None

    def append_entry(
        self,
        event_type: str,
        node_id: str,
        round_idx: int,
        manifest_root: str,
        update_hash: str,
        model_hash: str,
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Append a new validated event entry to the hash-chained ledger.

        Args:
            event_type: "round_commit" | "consent_revoked" | "model_release"
            node_id: Identifier of clinic or coordinator
            round_idx: Federated learning round index
            manifest_root: Merkle root over salted patient commitments
            update_hash: SHA-256 hash of serialized model delta/weights
            model_hash: SHA-256 hash of global model
            timestamp: Optional ISO-8601 timestamp string

        Returns:
            The complete appended entry dictionary.
        """
        last_entry = self.get_last_entry()
        next_index = (last_entry["index"] + 1) if last_entry else 0
        prev_hash = last_entry["entry_hash"] if last_entry else GENESIS_PREV_HASH

        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()

        entry: Dict[str, Any] = {
            "index": next_index,
            "timestamp": timestamp,
            "event_type": event_type,
            "node_id": node_id,
            "round": round_idx,
            "manifest_root": manifest_root,
            "update_hash": update_hash,
            "model_hash": model_hash,
            "prev_hash": prev_hash,
        }
        entry["entry_hash"] = canonical_hash_entry(entry)

        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")

        return entry

    def verify(self) -> bool:
        """Verify the integrity of this ledger file."""
        return verify_chain(self.ledger_path)
