# DiffSBDD-IQL 模型、训练与使用说明

> 最后更新：2026-09-29。本文是 RE-CmeB / 8GJL-C1101 上 **DiffSBDD 完整去噪轨迹离线 IQL 并行支线** 的说明，不代表当前已确认的项目正式主线。AliDiff/Global DPO、DecompDPO/Local DPO、BC/AWR 和在线 PPO/DDPO 与本支线并行记录。本文只陈述已在本项目代码或工件中核实的 IQL 实现与边界。

## 1. 目标与边界

目标是在不再向环境在线采样的前提下，用原始 DiffSBDD 的完整去噪轨迹和离线质量标签学习一个更偏向高质量分子的条件扩散策略。训练后必须从新 checkpoint 独立生成固定次数分子，再进行 RDKit、结构、对接和性质复筛；训练 loss 或离线 reward 不能直接证明候选分子更好。

当前证据数据集是 `runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/`：100,000 次固定尝试、10,000 个 shard、每条 episode 1,001 个去噪决策、67,694 条 RDKit-valid episode。其条件固定为 8GJL pocket 与 C:1101，不可被描述为跨靶点训练或泛化证据。

本项目中“好/中/坏”是**分子终局质量分层**，而非扩散模型输入标签。每条样本先由 DiffSBDD 生成完整轨迹，再将其终局质量通过 `trajectory_id/sample_id` 绑定回该条轨迹；IQL 对每一个去噪动作进行更新。

```text
8GJL pocket + 原始 DiffSBDD checkpoint
       -> 完整去噪轨迹 (s_t, a_t, s_t+1; t=0...1000)
       -> 终局分子重建、性质/结构/对接事实
       -> High / Medium / Low 分层 + 有限数值 reward sidecar
       -> 冻结 scaffold train/validation/test
       -> BC-all / filtered-BC / AWR / IQL
       -> 新 checkpoint 独立固定次数生成与筛选
```

## 2. DiffSBDD 是什么

DiffSBDD 是口袋条件的 3D 扩散生成器。给定蛋白口袋坐标与原始模型 checkpoint，它从随机噪声逐步去噪，生成配体原子坐标和类别。对每个样本，项目使用如下统一语义：

| 层级 | 核心字段/含义 |
|---|---|
| 扩散状态 | `latents [B, T+2, N, D]`，保存 `z_T ... z_0` 与最终 `xh_final` |
| 行为 | `logps [B, T+1]`、`timesteps [B, T+1]`，记录原始策略的逐步分数与决策时间 |
| 终局结构 | `x`、`categorical`、`nodesxsample`；去 padding 后重建分子 |
| 化学有效性 | `reconstruction_success`、`sanitize_success`、`rdkit_valid`、失败原因 |
| 质量事实 | QED、SA、LogP、TPSA、环/片段/Brenk、对接、PB、ChEMBL 先验等 |

`ddpo/main_diffsbdd_collect_ddp.py` 是当前采集入口。每个 GPU 使用 `seed = local_rank + 1`，即 1-based GPU seed；每个 shard 保存 RNG 状态、清单与 SHA-256，使中断后可按已完成 shard 连续恢复。

### 从扩散到 MDP

离线 IQL 将一个完整去噪过程视为有限时域 MDP：

| 记号 | DiffSBDD 对应物 |
|---|---|
| 状态 `s_t` | 当前 latent、口袋条件、节点数、决策时间 `t` |
| 动作 `a_t` | 原始 DiffSBDD 在该步实际执行的去噪 transition |
| 下一状态 `s_(t+1)` | 该动作后的 latent；最终 decode 为 terminal |
| reward `r_t` | 中间步为 0，终局步为该分子的有限 `reward_total_train` |
| done | 仅最终 decode/终局转移为 1 |

因此一个 High 分子不能只存一条 SMILES 或 SDF 就参与 IQL；必须能回读它的完整 shard、动作和条件。`episode_return` 对每个 transition 都保留终局回报，供 filtered BC/AWR 使用；它不能被中间 transition 的 `reward=0` 替代。

## 3. 你已有的好/中/坏分层如何进入训练

三分类非常适合当前流程，但要同时保存“类别”和“数值 reward”。推荐语义如下：

| 分层 | 建议用途 | 数值回报示例 |
|---|---|---:|
| High（好） | 高质量支持、filtered BC 正向集、各 batch 固定抽样 | `+1.0` |
| Medium（中） | 行为分布过渡区、critic 边界学习 | `0.0` |
| Low（坏） | 保留失败/低质分布，帮助 critic 学会拒绝 | `-1.0` |

`+1/0/-1` 是目前 three-pool 训练配置的离散回报 contract，不是活性真值或最终候选排序。若你已有更细的质量分数，可将其保存为 `reward_total_raw`，但正式训练只应使用版本固定、全有限、可追溯的 `reward_total_train`。不要把缺失对接、重建失败或未评分样本填成 NaN；应记录失败原因并赋予预先声明的有限惩罚。

在当前严格 Brenk 候选规则下，100,000 次固定尝试中有 7,545 个严格通过候选（7.545%），400 个是结构多样的可选子集。建议把 7,545 个作为 High 候选池而非只用 400 个；400 个仅适合 smoke 或多样性对照。High 仍不是实测活性分子，需要后续 docking/PB/相互作用复核。

每个 sidecar 行至少应包含：

```text
sample_id, source_trajectory_format, source_checkpoint_sha256,
source_shard_sha256, reward_schema_version, reward_version,
reconstruction_success, sanitize_success, rdkit_valid,
pb_evaluated, pb_valid, docking_input_success, docking_success,
actual_good, reward_total_train, failure_reason
```

其中 `sample_id` 是轨迹 join 主键；`source_shard_sha256` 与 checkpoint 哈希用于阻止标签错配到另一轮采样。sidecar schema 会拒绝 `NaN/Inf`，且 `actual_good=true` 时要求 RDKit-valid 和 PB-valid 等前置事实成立。

## 4. IQL 的模型组成与数学目标

训练包含四个可学习部分：

| 模块 | 作用 |
|---|---|
| Actor `pi_theta(a|s)` | 原 DiffSBDD 生成网络；计算数据集中动作在当前策略下的 log probability，并做可微更新 |
| Twin Q `Q1,Q2(s,a)` | 两个几何 critic；估计采取给定去噪动作后的回报，取较小者抑制过估计 |
| Value `V(s)` | 估计数据状态的保守价值，不需要对未见动作做 max/采样 |
| Target Q | Q 的慢更新副本，用于稳定 value target/advantage |

实现采用 IQL 的 expectile value regression 和 advantage-weighted behavior cloning：

```text
q_bar(s,a) = min(Q1_target(s,a), Q2_target(s,a))
L_V = E[ |tau - 1[q_bar - V < 0]| * (q_bar - V)^2 ]
y = r + gamma * (1-done) * V(s_next)
L_Q = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)
A(s,a) = q_bar(s,a) - V(s)
w(s,a) = min(exp(beta * A(s,a)), max_weight)
L_actor = -E[w(s,a) * log pi_theta(a|s)]
```

已完成的 last-2-block 训练以及历史 full-unfrozen 尝试均使用 `discount=1.0`、`expectile=0.7`、`target_tau=0.005`、actor warmup 10 steps、`temperature=1.0` 与 `max_weight=10.0`。`max_weight` 是必要的数值保护，但不能替代 gradient finite 检查。

### 为什么不用离线 PPO

采集的 `logps` 是轨迹记录信息，但 ordinary decision 的行为分数存在大量 `-100` 截断，且 final decode 有独立概率分支。因此它们不是安全的旧策略 log probability，不能用于 `exp(new_logprob-old_logprob)` 的离线 PPO ratio。IQL 只用当前 actor 对数据动作重新计算的 log probability 和 critic advantage，BC/AWR 也是安全的基线。

## 5. 训练前必须完成的准备

1. **冻结轨迹来源。** 固定 run 目录、episode index、原始 checkpoint hash、pocket hash 和 shard hash；不混入 DiffGui 或不同 DiffSBDD checkpoint 的轨迹。
2. **完成 sidecar。** 每个 episode 恰有一条可验证的标签行；`reward_total_train` 全有限；保留失败样本在固定尝试分母内。
3. **冻结数据划分。** 按 Murcko scaffold 而非随机分子划分 train/validation/test；每个 split 都要有 High，且报告 High/Medium/Low 数量。
4. **先验数据审计。** 确认 High 的 `actual_good` 与正 scaffold 支持充分；检查标签与 shard 的一一 join、奖励分布及类别泄漏。
5. **先做匹配控制。** Original、BC-all、filtered BC、AWR、IQL 必须使用同一原始 checkpoint、数据版本、pocket、训练预算和最终独立生成预算。
6. **critic 验收。** 在 held-out split 上验证 critic 排序/校准优于仅节点数、RDKit validity 等简单基线，才解释 IQL actor。

## 6. 已完成的三池 IQL 训练与启动方式

已完成训练采用“最后两个 dynamics block 可训练”的 three-pool 配置，而不是最早发生数值错误的 full-unfrozen 尝试：

`/data/disk6/DiffB110/collected_iql_benchmark_100k_260806/iql_worktree/launch/diffsbdd_three_pool_iql_last2blocks_8gpu_every10_10epoch_260820.yaml`

它的设计为：8 GPU DDP；每卡 High/Medium/Low 各 4 条、全局 batch 96；决策时间固定为 `0,10,...,1000` 共 101 个点；每 epoch 101 次 update；actor 仅解冻 `trainable_dynamics_blocks: [3, 4]`。起始 10 epoch 后使用同一 run 目录和 `checkpoints/latest.json` 分段恢复至 100 epoch，完整链记录在 `run_manifest.json`。

已验证的完成状态：epoch 99、global step 10,100、`status=completed`。目录包含 `epoch_0000` 至 `epoch_0099` 的 checkpoint manifest，以及 epoch 99 的 8 个 rank checkpoint 哈希；这证明训练按设定自然结束，但不等同于分子质量已提升。

```bash
ROOT=/data/disk6/DiffB110
WORKTREE=${ROOT}/collected_iql_benchmark_100k_260806/iql_worktree
export PYTHONPATH=${WORKTREE}:${PYTHONPATH:-}

conda run -n dif torchrun --nproc_per_node 8 --master_port 29654 \
  ${WORKTREE}/ddpo/main_diffsbdd_offline_iql_ddp.py \
  --config ${WORKTREE}/launch/diffsbdd_three_pool_iql_last2blocks_8gpu_every10_10epoch_260820.yaml \
  --data-root ${ROOT} \
  --output-dir ${WORKTREE}/analysis/offline_iql_three_pool_last2blocks_8gpu_every10_10epoch_260820 \
  --log-level INFO
```

不要用以上命令覆盖已完成目录。若开始新实验，先复制并版本化 YAML/启动脚本，创建新的输出目录；先运行 `bash -n` 与小规模 preflight，再写 resolved config、日志、metrics、status 和每 epoch checkpoint。

### 历史的全量解冻失败尝试

2026-08-20 的 full-unfrozen 尝试在 `global_step=10` 的首次 actor update 触发 `FloatingPointError: actor gradients contain NaN or Inf`。诊断记录将异常定位到 `module.model.ddpm.gamma.gamma` 有一个非有限元素。这是选择“仅训练 dynamics block 3/4”并先做 smoke/分段续训的工程背景；它不否定后续 last-2-block 训练已经完成。

如果未来要再次尝试全量解冻，应先完成下列独立排障，不能把它与已完成的 last-2-block checkpoint 混为同一个模型：

1. 保留该失败目录和前 10 行 metrics，定位 NaN 来自 actor 的哪一层、哪一条 transition、何种 decision index；不要删除或覆盖证据。
2. 在 actor log-prob、per-sample advantage、weight、loss、每层 gradient 上添加有限性统计与 fail-fast 记录；要保留 `sample_id` 和 `decision_index` 以便回查。
3. 先以 frozen/局部解冻 actor、较小 batch 和同一 three-pool sidecar 做可复现的 1-step、10-step、epoch-0/epoch-1 resume smoke；确认 checkpoint 可恢复。
4. 排除数据与算子问题后，再比较 learning rate、mixed precision、gradient norm、log-prob 归一化和 actor 可训练参数范围。不得通过静默 `nan_to_num` 或丢弃问题样本掩盖错误。
5. 只有连续训练、held-out critic 和独立生成三者均通过，才把全量解冻版本列为该 IQL 支线的可比较候选。

## 8. 训练中应记录什么

每个 epoch 的 checkpoint 至少包含 actor、online/target Q、V、各 optimizer、epoch/global step、resolved config、metrics 和 CPU/CUDA RNG 状态。`metrics/metrics.csv` 应记录：value loss、twin-Q loss、actor loss、advantage mean/std、weight mean/max、effective sample size、Q disagreement、每类样本数量、梯度范数及 non-finite 计数。

每轮还需要在 validation/test split 上输出：reward/类别分层、critic 对 High/Medium/Low 的排序与校准、scaffold-level 指标、与节点数/RDKit-valid baseline 的比较。训练集 loss 下降不是 critic 有效或 actor 改进的判据。

## 9. 训练后怎样使用模型

选择 checkpoint 的顺序应为：先冻结 validation 选择规则，再从通过 critic 验收和数值稳定性检查的 control/IQL checkpoint 中选择。随后对 Original、BC-all、filtered BC、AWR 和 IQL 各自进行独立的固定次数生成；单 GPU 用 `seed=1`，多 GPU 每 shard 用 `seed=gpu_id`（1-based）。不补抽无效分子。

对每个 checkpoint 的生成结果执行相同筛选漏斗：

1. 重建与 RDKit sanitize；报告 `valid / fixed_attempts`。
2. canonical SMILES 去重；报告 unique 数和重复率。
3. Brenk、三元环、片段、异常价态、环病理、元素和性质范围检查。
4. RE-CmeB UniDock 与 ligand efficiency；对接失败单列而非从分母删除。
5. PoseBusters 或等价结构检查、QED/SA/LogP/TPSA/MW/HBA/HBD、ChEMBL 先验复筛。
6. 对 top candidates 做 3D 相互作用人工复核，之后才交给 MD 准备。

输出必须包含固定尝试数、每一步过滤数量、运行日志、resolved config、checkpoint 哈希、SDF/CSV 和失败原因。没有独立生成对照时，只能称为“离线训练信号”，不能称为“分子质量提升”。

## 10. 关键文件索引

| 用途 | 路径 |
|---|---|
| 轨迹采集 | `ddpo/main_diffsbdd_collect_ddp.py` |
| IQL 数学 loss | `collected_iql_benchmark_100k_260806/iql_worktree/ddpo/offline_rl/losses.py` |
| sidecar schema/校验 | `collected_iql_benchmark_100k_260806/iql_worktree/ddpo/offline_rl/reward_sidecar.py` |
| 完成训练的 three-pool 配置 | `collected_iql_benchmark_100k_260806/iql_worktree/launch/diffsbdd_three_pool_iql_last2blocks_8gpu_every10_10epoch_260820.yaml` |
| 完成训练状态/metrics/checkpoint | `collected_iql_benchmark_100k_260806/iql_worktree/analysis/offline_iql_three_pool_last2blocks_8gpu_every10_10epoch_260820/` |
| 历史全量解冻失败状态 | `collected_iql_benchmark_100k_260806/iql_worktree/analysis/offline_iql_three_pool_full_unfrozen_8gpu_101timepoints_260820/status.json` |
| 严格候选池说明 | `work_records/26-8/260819/260819-07 严格Brenk候选正样本与IQL训练规划.md` |
| 项目级路线 | `docs/project_context.md`、`docs/workflow_commands.md` |

## 11. 最小执行清单

```text
[ ] High/Medium/Low 每条均已通过 sample_id 回连完整 trajectory shard
[ ] sidecar 有版本、哈希来源、有限 reward，无 NaN/Inf
[ ] scaffold train/validation/test 冻结，三类在各 split 的数量已审计
[ ] BC-all、filtered BC、AWR 已按相同预算设为对照
[ ] 已完成的 last-2-block checkpoint 已进入独立固定次数生成评估
[ ] held-out critic 优于简单基线
[ ] 每 epoch checkpoint、metrics、log、status 可恢复
[ ] 新 checkpoint 已独立固定次数生成并完成同一筛选漏斗
```

当前 last-2-block 训练已经完成，但只要独立生成和同预算对照尚未完成，适当结论仍是“离线 IQL 训练已完成，分子效果待独立评估”，而不是“已获得优化的 RE-CmeB 候选分子”。
