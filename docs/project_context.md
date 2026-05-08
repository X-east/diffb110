# Project Context

**最后更新**: 2026-05-08

## 1. Current Project Direction

This project uses `DiffSBDD` for 3D molecular diffusion generation and `DDPO/PPO` for reinforcement learning optimization.

Current verified code paths in this repository:
- **8-GPU DDP production**: `ddpo/main_diffsbdd_ddp.py` launched by `scripts/run_rl_qed_sa_docking_8gjl_8gpu.sh`
- **Single-GPU debugging**: `ddpo/main_diffsbdd_single_gpu.py`
- **Policy update**: `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`
- **Rollout**: `ddpo/runtime/rollout.py`
- **Model adapter**: `ddpo/integration/diffsbdd_model.py`
- **Current docking backend**: UniDock (8GJL 8-GPU config)
- **Reward components**: UniDock docking + QED + SA, combined with explicit weights
- **KL constraint**: integrated into EDMActor, kl_coef configurable (currently default 0.0)
- **Reward normalization**: z-score via RewardNormalizer
- **Component factory**: `ddpo/runtime/build.py` builds all components from YAML

### Architecture highlights

- **Lightning is deprecated** — training uses `torchrun` + custom DDP runtime, not PyTorch Lightning
- **Ray is deprecated** — GPU-direct docking/scoring, no Ray orchestration
- **Docking module is 4-layer decoupled** — docking → preparation → config → backends → registry
- **Molecule semantics is unified** — OpenBabel → RDKit single pipeline with `molecule_semantics.py`
- **prepare_mol_for_docking()** — deep copy + AddHs + Gasteiger charges from canonical mol
- **ref_model** — `copy.deepcopy(model_impl)` for KL, independent parameters, `torch.no_grad()` forward

### Key design decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Multi-GPU training | torchrun + DDP | Lightning deprecated; direct DDP is more controllable |
| Docking backend | UniDock (primary) | GPU-accelerated, verified with 8GJL config |
| Molecule pipeline | OpenBabel→RDKit single path | Avoid semantic divergence; all downstream shares same rdmol |
| Docking input prep | AddH + Gasteiger from canonical mol | Improve docking accuracy; single preparation path |
| KL constraint | deepcopy ref_model | Independent params, no sharing, no_grad forward |
| Reward normalization | Global z-score | group_index deprecated; unified RewardNormalizer |
| Component construction | runtime/build.py factory | Config-driven: YAML → all components |

## 2. Unified Molecule Semantics (Current Decision)

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

### 2.2 Canonical pipeline

The canonical molecule pipeline is:

1. Diffusion model generates a `sample`
2. Remove padding according to `nodesxsample`
3. Use **OpenBabel** to reconstruct structure / infer bonds from the diffusion geometry
4. Convert the reconstructed structure into one unified RDKit `Mol` (`rdmol`)
5. Run RDKit sanitize / legality check on that `rdmol`
6. Compute QED / SA directly from that same `rdmol`
7. Prepare docking input from that same `rdmol` via `prepare_mol_for_docking()`:
   - Deep copy mol (`Chem.Mol(mol)`)
   - Add hydrogens (`Chem.AddHs(mol, addCoords=True)`)
   - Assign Gasteiger charges (`AllChem.ComputeGasteigerCharges(mol)`)
   - Export SDF / PDBQT from the prepared mol
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

#### Rule D — RDKit validity as docking gate
`rdkit_valid` is the required gate before docking.
If the converted RDKit molecule fails RDKit sanitize, it must not proceed to docking.

#### Rule E — QED / SA input
QED and SA must be computed directly from the unified `rdmol`.
Do **not** use SMILES as the primary input path for QED / SA.
Do **not** round-trip through `Mol -> SMILES -> Mol` unless there is a documented exceptional reason.

#### Rule F — docking input source
Docking input files must be exported from the unified `rdmol` via `prepare_mol_for_docking()`.
Do not create a separate "raw diffusion output -> SDF/PDBQT" path that bypasses the canonical RDKit molecule.

#### Rule G — SMILES role
SMILES is an auxiliary representation only.
SMILES may be used for: logging, CSV export, duplicate filtering, novelty checks.
SMILES must not become the primary chemistry object for reward calculation.

## 3. KL Divergence Constraint

### 3.1 Implementation
- `ddpo/kl/gaussian_kl.py` — pure tensor function `compute_gaussian_kl(mu_ref, sigma_ref, mu_new, sigma_new, mask)`
- Integrated in `edm_actor.py` → `calculate_loss()`: `loss = ppo_loss + kl_coef * kl`
- `ref_model` created by `copy.deepcopy(model_impl)`, forward in `torch.no_grad()`
- KL formula: `KL[N(μ_ref, σ_ref²) || N(μ_new, σ_new²)]`

### 3.2 Current status
- `kl_coef = 0.0` in current configs — KL is disabled
- Known issues:
  - No NaN/Inf protection in `compute_gaussian_kl()` — extreme σ can cause training explosion
  - No separate KL logging — only mixed loss visible, cannot monitor KL contribution
- Enable with `kl_coef > 0` in YAML config after adding protections

## 4. Reward System Architecture

### 4.1 Docking module (4-layer decoupled)

```
docking.py (语义层)
  → docking_preparation.py (准备层: SDF + AddH + Gasteiger)
    → docking_config.py (配置层: 常量/默认值)
      → docking_registry.py → docking_backends/ (后端层)
        ├── unidock.py       (主后端)
        ├── autodock_gpu.py  (备选)
        └── smina.py         (CPU备用)
```

### 4.2 Reward normalization
- `normalization.py` — `RewardNormalizer`
- Global z-score: `normalized = (reward - running_mean) / (running_std + eps)`
- Configurable: method, clamp range, failure handling
- **`group_index` is deprecated** — always use global statistics

### 4.3 Composite reward
- `composite.py` — explicit weighted sum of component rewards
- Weights in YAML config
- Each component score logged separately

## 5. Canonical Per-Sample Fields

Each generated sample should conceptually be tracked with the following layers.

### 5.1 Sample layer
- `x`, `categorical`, `nodesxsample`, `sample_id`

### 5.2 Reconstruction layer
- `obabel_reconstruct_success`, `rdmol`, `build_success`

### 5.3 RDKit legality layer
- `sanitize_success`, `rdkit_valid`

### 5.4 Representation layer
- `smiles`, `canonical_smiles`

### 5.5 Drug-likeness layer
- `qed`, `sa_raw`, `sa_norm`, `druglikeness_reward`, `druglikeness_success`

### 5.6 Docking layer
- `docking_input_success`, `dock_success`, `dock_score_raw`, `dock_score_norm`, `docking_reward`

### 5.7 Final reward layer
- `total_reward`, `failure_type`

Recommended `failure_type` values:
- `build_failed`, `sanitize_failed`, `druglikeness_failed`, `docking_input_failed`, `docking_backend_failed`, `success`

## 6. Reward Semantics

The final reward must remain explicit and inspectable.
- keep each component score visible
- keep each component weight in config
- separate invalid molecule failures from docking failures
- log component means and failure counts

## 7. Naming Guidance

- use `rdkit_valid` instead of bare `valid`
- avoid using `stability` if it really means RDKit sanitize success
- separate: `build_success`, `sanitize_success`, `rdkit_valid`, `dock_success`

## 8. Implementation Guidance

When updating code related to molecule reconstruction, filtering, reward, logging, or docking:
- preserve diffusion trajectory semantics
- preserve timestep semantics
- preserve mask semantics
- preserve batch semantics
- do not silently change reward meaning
- do not silently mix different molecule-building paths
- factor molecule reconstruction into one shared canonical path
- let filter, drug-likeness reward, and docking consume the same reconstructed `rdmol`
- make failure boundaries explicit
- keep per-component logging explicit
