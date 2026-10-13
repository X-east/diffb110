# Project Context

> **最后更新**: 2026-09-29（删去与 `管线总览.md` 重复的路线清单/路径清单与过时表述；语义与规范内容未变）
> **定位：L1 按需查阅文档 —— 不要通读。** 唯一必读文档是 [`管线总览.md`](管线总览.md)（<80 行）。
> 本文记录分子语义、规范字段、reward 语义、命名/实现/证据规范；管线状态与代码入口见 `管线总览.md`。

## 章节导航（只读需要的那一节）

| 节 | 内容 | 什么时候查 |
|---|---|---|
| 1 | 方向性结论（主线 ESRPO 及 ChEMBL 版、共享流程） | 想确认主流向 |
| 2 | 路线决策规则（不排名、不混语义、门槛） | 判断某条路线定位 |
| 3 | 分子语义（下游分子对象的唯一真相源） | 处理分子对象 / 重建前 |
| 4 | 每个样本的规范字段 | 写数据、manifest、sidecar 前 |
| 5 | reward 语义 | 改 reward / 解释 reward 前 |
| 6 | ChEMBL MLP / D-MPNN 分类器 | 用 `chembl_prior` 前 |
| 7 | 离线轨迹 / RL 支线（IQL） | 碰 IQL 支线前 |
| 8 | Symmetric Dual-R（历史，非近期主方法） | 想重复调查前 |
| 9 | 命名规范 | 起名 / 改名时 |
| 10 | 实现规范 | 写代码前 |
| 11 | 证据规则 | 下任何结论前 |

## 1. Current Project Direction

DiffB110 的 DiffSBDD 保留线**最终主线是 ESRPO**（structure-matched, x-only v=0, lr=1e-5,
1-epoch global-esrpo DPO），依据 2026-09-02 的 1000-attempt 生成对比
（[260902-02](../work_records/26-9/260902/260902-02 lr1e-5 1000分子生成对比Original.md)）。其 **ChEMBL 版本**（跨源偏好对，winner 取自
ChEMBL 真实分子）进行中。

其余 DiffSBDD 支线（原始模型完整轨迹、离线 BC/AWR/IQL、AliDiff 风格 Global DPO、
DecompDPO 风格分解偏好、在线 DDPO、DiffGui RL）保留为**对照或并行实验**，各线仍共享
模型横评、固定尝试次数、分子重建、奖励事实、结构/对接筛选和 MD 候选交接规范。

> **管线状态、代码入口、各线权威文档指针：`docs/管线总览.md`（唯一总览入口）。**
> 本节只记录不随实验推进而变的方向性结论，不重复路线清单。

新训练/生成默认走 ESRPO 主线配置，除非实验记录明确标注为对照实验。

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

各路线共享同一套实践阶段（口径见 `docs/管线总览.md` 第一节）：

1. **Cross-model benchmark** — 固定尝试数横评，提供起点证据，不单独决定路线；
2. **Route-specific data construction** — 轨迹/reward sidecar/偏好对，manifest+SHA 校验、冻结无泄漏 split；
3. **Parallel training/evaluation** — 各线用各自 base 模型与 loss 语义（IQL 的 behavior score ≠ old-policy logp；偏好方法须区分 endpoint 去噪偏好与完整轨迹 log-prob）；
4. **Independent generation and screening** — 独立生成后按合法性/唯一性/对接/类药描述子/环病理/ChEMBL 先验筛选；
5. **MD preparation** — 仅在路线匹配证据齐备后选候选，MD 协议单独文档化。

The project has evolved from simple docking + QED/SA optimization into a multi-objective
molecular-quality control system.

Key paths (entry points and per-pipeline pointers: `docs/管线总览.md`):
collector `ddpo/main_diffsbdd_collect_ddp.py` · launch configs `launch/*.yaml|*.sh` ·
outputs `OUT/` · evidence `analysis/` · records `work_records/`.
Core rewards: docking, ligand efficiency, QED, SA, ChEMBL priors/MLP/D-MPNN, ring/pathology
facts, chemistry guards, distribution matching, diversity penalties.
Monitoring: validity, docking success, reward components, descriptor drift, carbon fraction,
hetero atoms, TPSA, HBA/HBD, logP, ring counts, ring pathology, fingerprint similarity, MMD.
Docking backend: UniDock in the `dif` environment.

### Long-running job execution rule

Formal generation, analysis, and training jobs must run independently of a
foreground session. Launch them detached; record launcher/child PIDs, exact
command, output/log/status paths and resume checkpoint; verify the process is
re-parented to PID 1 and that durable logs/metrics keep advancing. A frontend
command timeout is not evidence of a scientific failure — inspect the detached
job and resume from the latest checkpoint.

Key changes from earlier project states:

- No longer Lightning DDP; active training branches use their documented runtime, with PyTorch native DDP where applicable.
- No longer just QED/SA/docking; current reward work is multi-component and descriptor-aware.
- No longer only a 2220D MLP classifier; D-MPNN and explicit structural facts are now part of the reward/assessment design.
- DiffGui, online DiffSBDD DDPO, offline IQL, AliDiff-style Global DPO, and DecompDPO-style local preference optimization are kept as parallel routes or comparisons according to each experiment record.
- Symmetric dual-R sidechain generation has been investigated deeply and is not a current near-term main method.

## 2. Parallel Project Parts And Work Packages

**See `docs/管线总览.md`** — it holds the single-source-of-truth table of every pipeline
(status / code entry point / authoritative doc). Do not duplicate the route list here;
only route-level *decision rules* belong in this file.

Route-level decision rules that stay here:

- One benchmark ranking does **not** by itself select a route's base actor.
- Each route must use its own base-model, loss, data and checkpoint semantics; never mix
  DiffGui and DiffSBDD trajectory/checkpoint semantics.
- A route is only "selected" after matched independent generation evidence and an explicit
  project decision; parallel routes are not ranked globally.

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

Parallel branch (main `dif` env): the DiffSBDD reverse-diffusion chain as a finite-horizon
offline MDP; state `(z_t, t, pocket, node_count)` → action `z_{t-1}`;
collector `ddpo/main_diffsbdd_collect_ddp.py`.

- Controls: Original, BC-all, filtered BC, episode-return AWR, trajectory IQL.
- **Not** the confirmed route; no improvement claim until episode/manifest, reward-sidecar,
  actual-good/scaffold, held-out critic and independent-generation gates pass.
- Pure offline PPO is excluded (stored behavior scores ≠ PPO old-policy log probabilities).
- DiffGui / online DDPO checkpoints and trajectory semantics must never be mixed into it.

Details: `work_records/26-7/260731/260731-01 …全过程记录.md`,
`docs/overview/DiffSBDD-IQL中期整体梳理.md`.

## 8. Symmetric Dual-R Context (June 2026 — not near-term)

- DiffSBDD fixed-core/inpaint runs, but generated sidechains were often poor or degenerate.
- DiffGui has explicit edge/halfedge diffusion, yet the hard dual-R mechanism stays unclear / high risk.
- Keep as documented evidence only, not a near-term main method.

Evidence: `analysis/dual_sidechain_honokiol_9pc_2026061{7,8}/`, `work_records/26-6/260617/`,
[260618-01](../work_records/26-6/260618/260618-01 DiffGui双R侧链解耦推进分析.md).

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
