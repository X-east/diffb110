# Project Context

> **最后更新**: 2026-09-02

This document records the current project direction, molecule semantics, reward semantics, and route-level decisions for DiffB110.

## 1. Current Project Direction

DiffB110 的 DiffSBDD 保留线已确认**最终主线方向为 E2PO**（structure-matched, x-only v=0, lr=1e-5, 1-epoch global-e2po DPO）。该决策由 2026-09-02 的 1000-attempt 生成与对比证据支撑（见 `work_records/26-9/260902/260902-02 lr1e-5 1000分子生成对比Original.md`）：宽松口径 sanitize 通过率 81.9%，高于标准 DiffSBDD（~66.9%）与 GCDM-SBDD（~51.9%）；环系/多环病理下降 ~36pp；对接中位 −7.93（弱于 Original −9.51，但结构合理性显著占优）。

其他 DiffSBDD 支线作为**对照或并行实验**保留，但不作为 DiffSBDD 保留线主线。各路线仍共享模型横评、固定尝试次数、分子重建、奖励事实、结构/对接筛选和 MD 候选交接规范。

当前保留/并行路线包括：

- **【主线】DiffSBDD E2PO**：structure-matched, x-only（v=0）, lr=1e-5, 1-epoch global-e2po DPO；训练入口 `ddpo/main_diffsbdd_global_dpo.py`，规范配置 `launch/diffsbdd_global_e2po_x_only_lr1e5_mainline.yaml`，生成脚本 `launch/run_diffsbdd_global_e2po_x_only_lr1e5_generation.sh`。
- DiffSBDD 原始模型、完整去噪轨迹和终局筛选（对照基线）；
- DiffSBDD 轨迹上的 BC-all、filtered BC、AWR、offline IQL 等离线学习实验（对照）；
- AliDiff 风格 Global DPO 非 x-only 变体（对照）；
- DecompDPO/DecompDiff 风格分解式偏好优化（对照）；
- 在线 DiffSBDD DDPO、DiffGui PPO/DDPO（对照/其他模型）。

主线选定不撤销其他路线的历史证据；新训练/生成默认走 E2PO 主线配置，除非实验记录明确标注为对照实验。

共享的证据链是：

```text
扩散模型横评（保留起点证据）
  -> DiffSBDD 完整去噪轨迹与 reward sidecar
  -> 按路线构造冻结数据、偏好对或完整轨迹
  -> 训练/采样各自的候选 checkpoint
  -> 多指标筛选与重打分
  -> 选择最优分子进入分子动力学模拟
```

该共享流程可用于报告、答辩和新任务摘要。目的不是只比较训练 loss，而是对每条路线都产出可审计的候选分子集合，并判断其是否值得进入后续 MD 验证。

各路线都遵循以下实践阶段：

1. **Cross-model diffusion benchmark**: compare DiffGui, DiffSBDD, PMDM, GCDM-SBDD, TargetDiff and related baselines under the same RE-CmeB / 8GJL-C1101 task, using a fixed attempt count rather than refilling invalid generations. This supplies baseline evidence; it does not by itself select the current offline-RL actor.
2. **Route-specific data construction**: collect DiffSBDD complete trajectories when needed, attach finite reward sidecars, construct preference pairs or decomposition labels when needed, verify manifest/SHA/sample identity, and freeze a no-leakage split.
3. **Parallel training/evaluation**: run each selected route with its own base-model, loss, data, and checkpoint semantics. For IQL, recorded behavior scores are not safe old-policy log probabilities; for preference methods, distinguish endpoint denoising preference loss from complete trajectory log-probability objectives.
4. **Independent generation and screening**: sample independently from each checkpoint, then filter and rerank by RDKit and PB legality, uniqueness, docking, ligand efficiency, drug-like descriptors, ring/pathology facts, ChEMBL/D-MPNN-like priors, and manual 3D interaction review.
5. **Molecular dynamics preparation**: select candidates only after route-matched evidence is available. MD protocol, engine, force field, simulation time, and acceptance criteria should be documented separately when that stage starts.

The project has evolved from simple docking + QED/SA optimization into a multi-objective molecular-quality control system:

### Long-running job execution rule

Formal generation, analysis, and training jobs must run independently of a
foreground Codex/tool session. Launch them with a detached process, record the
launcher and child PIDs, exact command, output/log/status paths, and resume
checkpoint. After launch, verify that the process is re-parented to PID 1 and
that durable logs or metrics continue to advance. A frontend command timeout is
not evidence of a scientific or training failure; inspect the detached job and
resume from the latest checkpoint when needed.

- **DiffSBDD complete-trajectory collector**: `ddpo/main_diffsbdd_collect_ddp.py`
- **Offline-IQL implementation record**: `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md`
- **Launch configs**: `launch/*.yaml` and `launch/*.sh`
- **Main outputs**: `OUT/`
- **Analysis evidence**: `analysis/`
- **Task records**: `work_records/`
- **Core rewards**: docking, ligand efficiency, QED, SA, ChEMBL priors, ChEMBL MLP, ChEMBL D-MPNN, ring/pathology facts, chemistry guards, distribution matching, and diversity penalties
- **Monitoring**: validity, docking success, reward components, descriptor drift, carbon fraction, hetero atoms, TPSA, HBA/HBD, logP, ring count, aromatic rings, three-member rings, fused/complex rings, fingerprint similarity, and MMD
- **Docking backend**: UniDock in the `dif` environment

Key changes from earlier project states:

- No longer Lightning DDP; active training branches use their documented runtime, with PyTorch native DDP where applicable.
- No longer just QED/SA/docking; current reward work is multi-component and descriptor-aware.
- No longer only a 2220D MLP classifier; D-MPNN and explicit structural facts are now part of the reward/assessment design.
- DiffGui, online DiffSBDD DDPO, offline IQL, AliDiff-style Global DPO, and DecompDPO-style local preference optimization are kept as parallel routes or comparisons according to each experiment record.
- No formal RL or preference-optimization mainline is currently confirmed; route selection remains conditional on matched evidence and an explicit project decision.
- Symmetric dual-R sidechain generation has been investigated deeply and is not a current near-term main method.
- The project narrative is no longer a four-line method exploration summary; it is now organized as parallel route work packages with no confirmed single mainline.

## 2. Parallel Project Parts And Work Packages

| Parallel work package | Main Code / Evidence | Current Status |
|---|---|---|
| Benchmark evidence | `analysis/benchmark/`, `work_records/26-7/260730/` | Preserve common fixed-attempt starting-point evidence; do not equate one benchmark ranking with route suitability |
| DiffSBDD trajectory/offline branch | `ddpo/main_diffsbdd_collect_ddp.py`, `work_records/26-7/260731/`, independent IQL worktree | Parallel branch containing Original, BC, filtered BC, AWR and IQL experiments with route-specific gates |
| AliDiff-style global preference branch | `reference project/AliDiff-main/`, `analysis/diffsbdd_decompdpo_adaptation_audit_260822/`, current DiffSBDD DPO pilots | Parallel whole-molecule preference branch; must use DiffSBDD-native state/loss semantics when adapted |
| DecompDPO-style decomposed branch | `reference project/DecompDPO/`, `work_records/26-8/260822/` | Parallel local/multi-granularity branch; decomposition and bond mechanisms are not assumed portable |
| Online/legacy RL comparisons | `ddpo/main_diffgui_docking_ddp.py`, `ddpo/main_diffsbdd_ddp.py`, `analysis/diffgui_rl/`, `analysis/diffsbdd_rl/` | Run or cite according to the individual experiment record; not globally classified as the sole mainline |
| Screening, reranking and MD handoff | `scripts/docking/`, `chembl_classifier/`, `ddpo/utils/molecular_descriptors.py`, `analysis/` | Shared downstream evidence layer for every route |

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

## 7. DiffSBDD Offline Trajectory/RL Branch

One parallel branch uses the regular `dif` environment and treats the DiffSBDD reverse-diffusion chain as a finite-horizon offline MDP:

- Collector: `ddpo/main_diffsbdd_collect_ddp.py`
- Dataset/algorithm record: `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md`
- State/action semantics: `state=(z_t,t,pocket,node_count)`, `action=z_{t-1}`
- Required controls: Original, BC-all, filtered BC, episode-return AWR, trajectory IQL

Branch-level conclusion:

- DiffSBDD offline IQL is one experimental route, not the currently confirmed formal route.
- It is not valid to claim actor improvement until the relevant episode/manifest, reward-sidecar, actual-good/scaffold, held-out critic, and independent-generation gates pass.
- Pure offline PPO is excluded because stored behavior scores cannot safely serve as PPO old-policy log probabilities.
- DiffGui PPO/DDPO and online DiffSBDD DDPO may be run as parallel or comparison routes, but their checkpoints and trajectory semantics must not be mixed with DiffSBDD offline branches.

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
- write local artifacts under `runs/` or `analysis/` as appropriate

## 11. Evidence Rule

All proposed model, training, reward, sampler, and reinforcement-learning
directions must be grounded in existing project code or a mature public
implementation, paper, repository, or official documentation. Do not treat
intuition-only ideas as recommended methods or selected project routes. An
unsupported idea is permitted only as a clearly labelled isolated prototype
with a documented reference gap, risk, and acceptance ablation.

For project-level conclusions, inspect the relevant artifacts:

- raw run outputs: `runs/`
- analysis reports and plots: `analysis/`
- task records: `work_records/`
- daily/weekly summaries: `work_days/`
- launch configs: `launch/`
- resolved run configs: `runs/<run>/config/resolved_config.yaml`

Do not rely on old README text or memory when run artifacts are available.
