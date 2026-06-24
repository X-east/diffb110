# Project Context

> **最后更新**: 2026-06-18

This document records the current project direction, molecule semantics, reward semantics, and route-level decisions for DiffB110.

## 1. Current Project Direction

DiffB110 uses DiffGui and DiffSBDD for pocket-aware 3D molecular generation, with DDPO/PPO for reinforcement learning optimization and local analysis artifacts for route decisions. As of 2026-06-24, the current research main line is DiffGui reinforcement learning on the same target/task; DiffSBDD RL is retained as a baseline and backup path.

The project has evolved from simple docking + QED/SA optimization into a multi-objective molecular-quality control system:

- **DiffGui PPO entrypoint**: `ddpo/main_diffgui_docking_ddp.py`
- **DiffSBDD retained DDPO entrypoint**: `ddpo/main_diffsbdd_ddp.py`
- **Launch configs**: `launch/*.yaml` and `launch/*.sh`
- **Main outputs**: `OUT/`
- **Analysis evidence**: `analysis/`
- **Task records**: `work_records/`
- **Core rewards**: docking, ligand efficiency, QED, SA, ChEMBL priors, ChEMBL MLP, ChEMBL D-MPNN, ring/pathology facts, chemistry guards, distribution matching, and diversity penalties
- **Monitoring**: validity, docking success, reward components, descriptor drift, carbon fraction, hetero atoms, TPSA, HBA/HBD, logP, ring count, aromatic rings, three-member rings, fused/complex rings, fingerprint similarity, and MMD
- **Docking backend**: UniDock in the `dif` environment

Key changes from earlier project states:

- No longer Lightning DDP; current mainline uses PyTorch native DDP.
- No longer just QED/SA/docking; current reward work is multi-component and descriptor-aware.
- No longer only a 2220D MLP classifier; D-MPNN and explicit structural facts are now part of the reward/assessment design.
- DiffGui is no longer only a reference sampler; it is the current environment-isolated PPO/DDPO main line.
- Symmetric dual-R sidechain generation has been investigated deeply and is not a current near-term main method.

## 2. Current Research Lines

| Line | Main Code / Evidence | Current Status |
|---|---|---|
| DiffGui RL | `ddpo/main_diffgui_docking_ddp.py`, `ddpo/diffgui/runtime/`, `analysis/diffgui_rl/`, `OUT/diffgui-*` | Current main RL platform; focus on same-target reward optimization, ring/pathology control, top-N reranking, and molecule-quality retention |
| DiffSBDD RL | `ddpo/main_diffsbdd_ddp.py`, `analysis/diffsbdd_rl/`, `OUT/diffsbdd-*` | Retained DDPO baseline and backup path; reward drift and structural guards remain useful comparison evidence |
| ChEMBL / D-MPNN | `chembl_classifier/`, `ddpo/rewards/chembl_*` | MLP exists; D-MPNN is the stronger learned-prior candidate but must be calibrated |
| DiffSBDD small mods | `DiffSBDD/equivariant_diffusion/`, `analysis/diffsbdd_small_mod/` | Local quality improvements remain possible |
| DiffSBDD x/h/e | `DiffSBDD/edge_diffusion/`, `analysis/diffsbdd_xhe/` | Scientifically meaningful but high-risk and not yet the stable mainline |
| Symmetric dual-R | `ddpo/constraints/`, `analysis/dual_sidechain_honokiol_9pc_*`, `work_records/26-6/260617/` | Explored deeply; current hard/soft constraint route is not promising as a main method |

## 3. Unified Molecule Semantics

The downstream molecule object should have one source of truth:

- `rdmol`: the RDKit `Mol` converted from the reconstructed structure of one diffusion sample after removing padding according to `nodesxsample`.

A diffusion sample itself is not yet the final chemistry object. A sample usually contains:

- coordinates `x`
- atom-type tensor `categorical`
- valid node count `nodesxsample`
- optional trajectory/logprob metadata

The canonical reconstruction rule is:

1. Diffusion model generates a sample.
2. Remove padding according to `nodesxsample`.
3. Reconstruct or perceive bonds from geometry using the configured reconstruction path.
4. Convert the reconstructed structure into one unified RDKit molecule.
5. Run RDKit legality/sanitize checks.
6. Compute rewards and descriptors from that same molecule object.
7. Export docking inputs from that same molecule object.
8. Use SMILES only for logging, deduplication, novelty, and reporting.

### Required Rules

| Rule | Meaning |
|---|---|
| Single molecule object | Filter, drug-likeness reward, docking, descriptor extraction, and SDF export should not independently reconstruct divergent molecules |
| RDKit validity as gate | `rdkit_valid` is a required gate before docking and most reward calculations |
| Finite failure penalties | Failed molecules receive finite penalties, not `NaN` or `Inf` |
| SMILES is auxiliary | SMILES should not become the primary reward calculation object unless explicitly justified |
| Failure reasons are explicit | Reconstruction, sanitize, docking, classifier, and reward failures should be separately recorded |

## 4. Canonical Per-Sample Fields

Each generated sample should conceptually be tracked with these layers:

| Layer | Fields |
|---|---|
| Sample | `x`, `categorical`, `nodesxsample`, `sample_id` |
| Reconstruction | `reconstruction_success`, `rdmol`, `build_success`, `failure_reason` |
| RDKit legality | `sanitize_success`, `rdkit_valid` |
| Representation | `smiles`, `canonical_smiles`, `inchikey`, `murcko_scaffold` |
| Drug-likeness | `qed`, `sa_raw`, `sa_norm` |
| Docking | `docking_input_success`, `docking_success`, `docking_raw`, `docking_reward` |
| Structure facts | `ring_count`, `aromatic_ring_count`, `three_member_ring_count`, `fragment_count`, `spiro_atom_count`, `bridgehead_atom_count` |
| Reward | component rewards, `reward_total`, `reward_for_advantage`, `advantage` |
| Training selection | `training_keep`, `training_filter_reason`, replay/good-pool flags when enabled |

Recommended failure types:

- `reconstruction_failed`
- `sanitize_failed`
- `rdkit_invalid`
- `docking_input_failed`
- `docking_backend_failed`
- `classifier_failed`
- `reward_missing`
- `success`

## 5. Reward Semantics

The project uses a multi-component reward system. The exact active components are determined by `launch/*.yaml`.

| Component | YAML Key | Type | Notes |
|---|---|---|---|
| Docking | `docking` | Structure-based | UniDock score normalized into reward |
| Ligand Efficiency | `ligand_efficiency` | Structure-based efficiency | Docking score divided by molecule size |
| QED | `qed` | Drug-likeness | RDKit QED |
| SA | `sa` | Synthesis proxy | Synthetic accessibility |
| ChEMBL Prior ECFP4 | `chembl_prior_ecfp4` | Distribution prior | Fingerprint similarity |
| ChEMBL Prior FCFP4 | `chembl_prior_fcfp4` | Distribution prior | Feature fingerprint similarity |
| ChEMBL MLP | `chembl_classifier` | Learned prior | 2220D Morgan/RDKit descriptor classifier |
| ChEMBL D-MPNN | `chembl_dmpnn` | Learned graph prior | Message-passing classifier reward |
| MolSkill | `molskill` | Preference model | Historical/optional reward |
| LogP Distribution | `logp_distribution` | Distribution matching | Baseline distribution control |
| TPSA Distribution | `tpsa_distribution` | Distribution matching | Baseline distribution control |
| Diversity ECFP4 | `diversity_ecfp4` | Batch penalty | Within-batch similarity penalty |
| Diversity FCFP4 | `diversity_fcfp4` | Batch penalty | Within-batch similarity penalty |
| Ring Complexity | `ring_complexity` | Structural rule | Coarse ring/fused/spiro/bridgehead control |
| Ring Pathology | `ring_pathology` | Structural rule | Large rings, fused large rings, abnormal small rings |
| Aromatic Scaffold | `aromatic_scaffold` | Structural rule | Preserve reasonable aromatic scaffold |
| Chemistry Guard | `chemistry_guard` | Structural guard | Carbon/hetero/fragment/bond-length/halogen checks |
| Stability wrapper | `stability` | Wrapper | Injects validity/failure fields |

Reward assembly flow:

```text
YAML config
  -> ddpo.rewards.factory
  -> component reward instances
  -> optional ComponentNormalizer
  -> CompositeReward
  -> optional StabilityAwareReward
  -> batch["reward_total"]
  -> RewardNormalizer / advantage construction
  -> PPO update
```

Naming consistency must hold across:

```text
YAML key -> reward_prefix -> batch field -> dataclass/schema -> CSV column -> terminal display
```

## 6. ChEMBL MLP And D-MPNN

### MLP Classifier

The MLP classifier uses 2220 features:

- 2048 Morgan fingerprint dimensions
- RDKit 2D descriptors

It is useful as a learned soft prior, but recent experiments show it should not be the only quality criterion.

Key files:

- `chembl_classifier/model.py`
- `chembl_classifier/train.py`
- `chembl_classifier/evaluate.py`
- `ddpo/rewards/chembl_classifier_reward.py`

### D-MPNN

D-MPNN is the current graph-based learned-prior candidate.

Key files:

- `chembl_classifier/dmpnn/`
- `chembl_classifier/config_dmpnn_v0_1_scaffold_strongpoly.yaml`
- `ddpo/rewards/chembl_dmpnn_reward.py`
- `test/test_chembl_dmpnn_data.py`
- `test/test_chembl_dmpnn_training.py`
- `test/test_chembl_dmpnn_reward.py`

D-MPNN must be calibrated against docking, explicit structural facts, descriptor drift, top-molecule review, and diversity.

## 7. DiffGui-Specific Context

DiffGui is handled as an environment-isolated path:

- Reference project: `reference project/DiffGui/`
- Main PPO entrypoint: `ddpo/main_diffgui_docking_ddp.py`
- Runtime modules: `ddpo/diffgui/runtime/`
- Outputs: `OUT/diffgui-*`
- Analyses: `analysis/diffgui_rl/`

Current DiffGui conclusion:

- DiffGui RL is the current main reinforcement-learning route.
- Its routine evaluation is same-target/same-task optimization, not cross-target generalization.
- It is useful for testing distribution shaping, especially ring/pathology control.
- DiffGui should not currently be used as the main hard symmetric dual-R sidechain platform.
- Good-molecule replay is implemented but should only be used with a short window, small ratio, and KL/clip gate after good-pool auditing.

## 8. Symmetric Dual-R Context

The dual-R / symmetric sidechain route was analyzed in detail in June 2026.

Current conclusion:

- DiffSBDD fixed-core/inpaint can run, but generated sidechains were often poor or degenerate.
- DiffGui has explicit edge/halfedge diffusion, but the hard dual-R mechanism remains unclear and high risk.
- The current route should remain documented as evidence, not treated as the near-term main method.

Key evidence:

- `analysis/dual_sidechain_honokiol_9pc_20260617/`
- `analysis/dual_sidechain_honokiol_9pc_20260618/`
- `work_records/26-6/260617/`
- `work_records/26-6/260618/260618-01 DiffGui双R侧链解耦推进分析.md`

## 9. Naming Guidance

Prefer explicit field names:

- `rdkit_valid`, not bare `valid`
- `reconstruction_success`
- `sanitize_success`
- `docking_success`
- `reward_found`
- `training_keep`

Do not use one overloaded field to mean all of:

- RDKit legality
- reward success
- docking success
- training participation

## 10. Implementation Guidance

When updating reconstruction, filtering, reward, logging, docking, or replay code:

- preserve diffusion trajectory semantics
- preserve timestep semantics
- preserve mask semantics
- preserve batch semantics
- do not silently change reward meaning
- do not silently mix molecule-building paths
- make failure boundaries explicit
- keep per-component logging explicit
- write local artifacts under `OUT/` or `analysis/` as appropriate

## 11. Evidence Rule

For project-level conclusions, inspect the relevant artifacts:

- raw run outputs: `OUT/`
- analysis reports and plots: `analysis/`
- task records: `work_records/`
- daily/weekly summaries: `work_days/`
- launch configs: `launch/`
- resolved run configs: `OUT/<run>/config/resolved_config.yaml`

Do not rely on old README text or memory when run artifacts are available.
