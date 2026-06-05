# Project Context

## 1. Current Project Direction

This project uses `DiffSBDD` for 3D molecular diffusion generation and `DDPO/PPO` for reinforcement learning optimization.

The current direction has evolved from simple docking + drug-likeness rewards to a **high-dimensional feature classifier-based reward system**:

- **DDP entrypoint**: `ddpo/main_diffsbdd_ddp.py` (PyTorch native DistributedDataParallel, launched via `torchrun`)
- **Launch configs**: `launch/*.yaml` + `launch/*.sh` (6 different configurations for different reward combinations)
- **Core reward component**: `ChemblClassifierReward` — a 2220-dimension MLP classifier trained on ChEMBL positive/negative molecules, used as a PPO reward signal
- **Supporting rewards**: fingerprint similarity priors (ECFP4/FCFP4), logP distribution matching, MolSkill human preference, diversity penalties, ligand efficiency
- **Normalization**: dual-layer — per-component EMA z-score (`ComponentNormalizer`) + global z-score (`RewardNormalizer`)
- **Monitoring**: 22 molecular descriptors tracked per-epoch (LogP/TPSA/HBA/HBD/carbon fraction/heteroatom count/aromatic rings/fingerprint diversity/MMD)
- **Smoke testing**: `launch/smoke.sh` (1 GPU / 1 epoch / batch=4, auto-cleanup)
- **Classifier pipeline**: standalone `chembl_classifier/` (data extraction → training → evaluation → integration to reward)
- **Docking backend**: `unidock` for the current 8GJL target

Key changes from earlier project state:
- ❌ No longer Lightning DDP → ✅ PyTorch native DDP
- ❌ No longer just QED/SA/docking → ✅ Multi-component classifier-centered reward system
- ❌ `docs/` previously described a different project state → ✅ Now updated to reflect current state (2026-05-27)

## 2. Unified Molecule Semantics (Current Decision)

This document records the current agreed molecule semantics for the project.

### 2.1 Single source of truth

The single primary molecule object used downstream is:
- `rdmol`: the RDKit `Mol` converted from the OpenBabel-reconstructed structure of one diffusion sample after removing padding according to `nodesxsample`

A diffusion sample itself is **not yet** the final chemistry object.
The sample consists of:
- coordinates `x`
- atom-type tensor `categorical`
- valid node count `nodesxsample`

The canonical reconstruction rule is:
- first use **OpenBabel** to reconstruct structure / perceive bonds from the diffusion geometry
- then convert that reconstructed structure into one canonical **RDKit `Mol`**
- all downstream chemistry logic must use that converted `rdmol`

OpenBabel is the structure reconstruction backend.
RDKit `Mol` is the single downstream chemistry object.

### 2.2 Canonical pipeline

The canonical molecule pipeline is:

1. Diffusion model generates a `sample`
2. Remove padding according to `nodesxsample`
3. Use **OpenBabel** to reconstruct structure / infer bonds from the diffusion geometry
4. Convert the reconstructed structure into one unified RDKit `Mol` (`rdmol`)
5. Run RDKit sanitize / legality check on that `rdmol`
6. Compute QED / SA directly from that same `rdmol`
7. Export SDF / PDBQT for docking from that same `rdmol`
8. Use SMILES only for logging / deduplication / novelty checks
9. Combine component rewards explicitly into final total reward

### 2.3 Required rules

#### Rule A — single molecule object
All downstream chemistry logic must use the same converted `rdmol`.
Do not let filter, drug-likeness reward, and docking each reconstruct molecules independently with diverging semantics.

#### Rule B — OpenBabel reconstruction, RDKit downstream
OpenBabel is responsible for structure reconstruction / bond perception from diffusion geometry.
RDKit `Mol` is responsible for all downstream chemistry semantics.
Do not allow separate downstream branches that continue using OpenBabel objects as primary chemistry objects.

#### Rule C — OpenBabel-invalid means invalid molecule
If OpenBabel reconstruction judges a sample unreasonable / invalid, that sample is an invalid molecule.
It must receive the maximum failure penalty and skip all downstream stages.
That means:
- do not continue to RDKit sanitize
- do not compute QED / SA
- do not prepare docking input
- do not run docking

#### Rule D — RDKit validity as docking gate
`rdkit_valid` is the required gate before docking.
If the converted RDKit molecule fails RDKit sanitize, it must not proceed to docking.

#### Rule E — QED / SA input
QED and SA must be computed directly from the unified `rdmol`.
Do **not** use SMILES as the primary input path for QED / SA.
Do **not** round-trip through `Mol -> SMILES -> Mol` unless there is a documented exceptional reason.

#### Rule F — docking input source
Docking input files must be exported from the unified `rdmol`.
Do not create a separate "raw diffusion output -> SDF/PDBQT" path that bypasses the canonical RDKit molecule.

#### Rule G — SMILES role
SMILES is an auxiliary representation only.
SMILES may be used for:
- logging
- CSV export
- duplicate filtering
- novelty checks against training data

SMILES must not become the primary chemistry object for reward calculation.

## 3. Canonical Per-Sample Fields

Each generated sample should conceptually be tracked with the following layers.

### 3.1 Sample layer
- `x`
- `categorical`
- `nodesxsample`
- `sample_id`

### 3.2 Reconstruction layer
- `obabel_reconstruct_success`
- `rdmol`
- `build_success`

### 3.3 RDKit legality layer
- `sanitize_success`
- `rdkit_valid`

### 3.4 Representation layer
- `smiles`
- `canonical_smiles`

### 3.5 Drug-likeness layer
- `qed`
- `sa_raw`
- `sa_norm`
- `druglikeness_reward`
- `druglikeness_success`

### 3.6 Docking layer
- `docking_input_success`
- `dock_success`
- `dock_score_raw`
- `dock_score_norm`
- `docking_reward`

### 3.7 Final reward layer
- `total_reward`
- `failure_type`

Recommended `failure_type` values:
- `build_failed`
- `sanitize_failed`
- `druglikeness_failed`
- `docking_input_failed`
- `docking_backend_failed`
- `success`

## 4. Reward Semantics（2026-05 更新：高维特征分类器体系）

The project has evolved from simple QED/SA/docking rewards to a **multi-component reward system**
centered around a ChEMBL classifier and fingerprint priors.

### 4.1 Current reward components

| Component | YAML Key | Type | Description |
|-----------|----------|------|-------------|
| QED | `qed` | Drug-likeness | Quantitative Estimate of Drug-likeness |
| SA | `sa` | Drug-likeness | Synthetic Accessibility |
| Docking | `docking` | Structure-based | UniDock docking score (kcal/mol) |
| Ligand Efficiency | `ligand_efficiency` | Efficiency | dock_score / heavy_atoms |
| ChEMBL Prior ECFP4 | `chembl_prior_ecfp4` | Distribution prior | ECFP4 fingerprint similarity to ChEMBL |
| ChEMBL Prior FCFP4 | `chembl_prior_fcfp4` | Distribution prior | FCFP4 fingerprint similarity to ChEMBL |
| **ChEMBL Classifier** | `chembl_classifier` | **MLP classifier** | **2220D MLP (Morgan FP + RDKit 2D descriptors)** |
| MolSkill | `molskill` | Human preference | MolSkill human-judgment model |
| LogP Distribution | `logp_distribution` | Distribution matching | LogP distribution vs ChEMBL baseline |
| Diversity ECFP4 | `diversity_ecfp4` | Batch penalty | Within-batch ECFP4 similarity penalty |
| Diversity FCFP4 | `diversity_fcfp4` | Batch penalty | Within-batch FCFP4 similarity penalty |
| Connectivity | `connectivity` | Validity | Fragment / disconnected structure penalty |
| Component Normalizer | (built-in) | Normalization | Per-component EMA z-score (optional) |
| Stability | `stability` | Wrapper | Injects RDKit validity / stability fields |

### 4.2 ChEMBL Classifier reward（核心新增）

```python
# chembl_classifier_reward.py → ChemblClassifierReward(BaseReward)
# Featurizer: MolSkill "morgan_count_rdkit_2d" (2048 Morgan FP + 172 RDKit 2D = 2220D)
# Model: 3-layer MLP (2220 → 512 → 128 → 1, BatchNorm + ReLU + Dropout)
# Output: P(ChEMBL-like) ∈ [0, 1]
# Reward mapping:
#   prob ≥ good_threshold → linear map to [0, max_reward], saturates at cap_high
#   prob < bad_threshold  → linear map to [penalty, 0], floors at cap_low
#   middle region → neutral 0
# Invalid molecules → failure_penalty (default: -1.0)
```

### 4.3 Training configurations

```yaml
# Example: 7gpu_classifier_only.yaml
reward:
  weights:
    chembl_classifier: 1.0
    qed: 0
    sa: 0
    docking: 0
  per_component_normalization:
    enabled: true
    momentum: 0.01
  normalization:
    enabled: true
    method: "zscore_total"
    clamp: { enabled: true, min: -2.0, max: 2.0 }
  chembl_classifier:
    good_threshold: 0.5
    cap_high: 0.8
    cap_low: 0.2
    max_reward: 1.0
    penalty: -1.0
```

### 4.4 Reward assembly flow

```
YAML Config (launch/*.yaml)
    ↓
factory.build_rewarder(config, dataset_info, logger)
    ├── get_enabled_rewards(config)  →  weight > 0 的组件列表
    ├── 逐组件实例化 (QED/SA/Docking/Classifier/Prior/Diversity/...)
    ├── 可选 ComponentNormalizer (EMA z-score)
    ├── CompositeReward(rewards, weights, component_normalizer)
    └── StabilityAwareReward(composite_reward, dataset_info)
        ↓
训练循环: rewarder.calculate_rewards(data)  →  batch['reward_total']
```

### 4.5 Naming consistency chain

YAML key → `reward_prefix` → batch field → CSV column → terminal display must be consistent.
See `.codebuddy/rules/项目规则.mdc` §七 for the full checklist.

## 5. Naming Guidance

To reduce ambiguity, prefer these names:
- use `rdkit_valid` instead of bare `valid`
- avoid using `stability` if it really means RDKit sanitize success
- if needed, separate:
  - `build_success`
  - `sanitize_success`
  - `rdkit_valid`
  - `dock_success`

Do not use one overloaded field to mean all of:
- RDKit legality
- reward success
- docking success

## 6. Implementation Guidance

When updating code related to molecule reconstruction, filtering, reward, logging, or docking:
- preserve diffusion trajectory semantics
- preserve timestep semantics
- preserve mask semantics
- preserve batch semantics
- do not silently change reward meaning
- do not silently mix different molecule-building paths

Preferred implementation direction:
- factor molecule reconstruction into one shared canonical path
- let filter, drug-likeness reward, and docking consume the same reconstructed `rdmol`
- make failure boundaries explicit
- keep per-component logging explicit
