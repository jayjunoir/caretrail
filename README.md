# CareTrail: Decentralized, Leakage-Free Healthcare Intelligence with Verifiable Audit Trails

> **ForgeHacks Submission** | Track: AI + Healthcare  
> *Making healthcare information clearer, more accessible, and verifiable.*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/reproducible-20--seed-green.svg)]()

---

## ⚠️ Synthetic Data & Educational Disclosure
**All patient profiles, names, and demographic records presented in this project are 100% synthetic.** Clinical vital measurements are adapted from the standard, anonymized UCI Heart Disease benchmark across 4 clinical nodes (Cleveland, Hungarian, Switzerland, Long Beach VA). 

*Disclaimer: CareTrail is an educational technology demonstration for ForgeHacks and does not provide clinical diagnosis or medical advice.*

---

## Problem Overview
When patients switch employers or health insurance providers (HMOs), their medical files fragment across disparate hospital systems. In collaborative healthcare AI, this introduces two major failures:
1. **Cross-Silo Leakage:** Duplicate records of the same patient (under slightly differing names or typos) appear in both training and evaluation sets, artificially inflating benchmark metrics.
2. **Loss of Patient Agency:** Centralizing raw health records exposes sensitive data to third-party brokers. Patients lack visibility into whether their records trained a given model, and have no mechanism to enforce consent revocation.

---

## Technical Solution

1. **Entity Resolution Without Middlemen:** Soft-blocking demographic matching (`RapidFuzz` + `NetworkX`) resolves cross-clinic patient visits into unified clusters with >85% precision and >80% recall without central database aggregation.
2. **Leakage-Free Group Splitting:** Guarantees that patient clusters are never split across train and test partitions, preventing optimistic evaluation bias.
3. **Decentralized PyTorch FedAvg:** Federated Logistic Regression trained across 4 clinical nodes. Raw clinical data stays strictly behind local firewalls; only model parameter updates travel between nodes.
4. **Salted Merkle Commitments & Audit Ledger:** Every federated training round logs a salted Merkle root of included patient IDs to a SHA-256 append-only JSONL ledger (`results/audit_ledger.jsonl`). Zero health data or identities are published to the ledger.
5. **Verifiable Consent Revocation:** Patients can audit model inclusion via cryptographic Merkle proofs and revoke consent at any time, guaranteeing exclusion from all subsequent training rounds.
6. **Deterministic Risk Attribution:** Explains clinical risk directly from model weights ($w^T \cdot x + b$) with physiological units, preventing LLM hallucinations.

---

## Empirical Benchmark (20 Random Seeds)

### Experiment A: Leakage Elimination
* **Naive Random Split:** Leaked an average of **23.3** duplicate patient records across splits, artificially boosting test AUROC.
* **Group-Aware Split:** **0.0** leaked records across all 20 seeds.

### Experiment B: Decentralized Federated Learning
| Strategy | Test AUROC (mean ± std) | Test Accuracy (mean ± std) | Data Movement |
| :--- | :---: | :---: | :---: |
| **Centralized (Pooled)** | 0.8806 ± 0.0309 | 0.8123 ± 0.0292 | Centralized |
| **Federated (FedAvg)** | **0.8740 ± 0.0340** | **0.8000 ± 0.0345** | **Zero Raw Data Transfer** |
| **Isolated Local Nodes** | 0.8263 ± 0.0554 | 0.8180 ± 0.0310 | Local Only |

*FedAvg captures **99.2%** of centralized performance while outperforming isolated local training by **+0.048 AUROC**.*

---

## Quickstart & Reproducibility

### 1. Installation
```powershell
# Clone repository
git clone [https://github.com/](https://github.com/)<your-username>/caretrail.git
cd caretrail

# Create virtual environment and install dependencies
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt