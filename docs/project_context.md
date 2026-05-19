# Project Context

## 1. Current Project Direction

This project uses `DiffSBDD` for 3D molecular diffusion generation and `DDPO/PPO` for reinforcement learning optimization.

Current verified code path in this repository snapshot:
- active, user-maintained multi-GPU entrypoint: `ddpo/main_diffsbdd_ddp.py` via launchers under `scripts/`
- active training loop ownership: custom DDP/runtime path (`ddpo/runtime/*` + `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`)
- rollout implementation in use: `ddpo/runtime/rollout.py`
- model adapter in use: `ddpo/integration/diffsbdd_model.py`
- docking backend: `unidock` for the current 8GJL / 6-GPU workflow
- reward components: docking reward + ChEMBL prior and/or QED/SA depending on `reward.weights`

Important repository/documentation mismatch:
- some older project docs still describe `ddpo/main_diffsbdd_lightning.py` + Lightning DDP as the preferred active path
- in the current repository tree, `ddpo/main_diffsbdd_lightning.py` is not present
- in the current repository tree, `ddpo/trainer/lightning_ddpo_module.py` is not present
- therefore, Lightning DDP should currently be treated as a historical / intended direction, not the verified active mainline in this checkout

The repository still contains historical Ray-based and legacy docking paths, but the intended active direction remains:
- keep diffusion semantics explicit
- keep mask semantics explicit
- keep reward semantics explicit
- avoid introducing new Ray-based orchestration
- use GPU-based docking/scoring
- prefer native PyTorch DDP for future multi-GPU refactors over reintroducing Ray

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

## 4. Reward Semantics

The final reward must remain explicit and inspectable.
Recommended structure:
- keep each component score visible
- keep each component weight in config
- separate invalid molecule failures from docking failures
- log component means and failure counts

Suggested interpretation:
- invalid molecule failure: reconstruction or RDKit legality failed
- docking failure: docking input creation or docking backend failed after RDKit-valid gating

A typical total reward should be an explicit weighted sum of component scores, for example:
- drug-likeness component from QED + normalized SA
- docking component from normalized docking score

The precise weights must live in config and be logged.

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
