# DiffSBDD 去噪轨迹离线强化学习（Offline IQL）完整实施方案

> 文档状态：设计基线 v1.0
> 日期：2026-07-30
> 分支：`diffsbdd-xh-offline-iql`
> 独立 worktree：`/home/md10/data/disk6/DiffB110-offline-iql`
> 基线提交：`4031cfe280641f105d6d045ce22da47ab73fb4d5`
> 目标任务：固定 RE-CmeB / 8GJL-C1101 口袋上的 DiffSBDD 纯离线去噪轨迹强化学习
> 推荐主算法：Implicit Q-Learning（IQL）
> 文档性质：实施设计与 go/no-go 规范，不代表算法已完成或效果已经成立

## 0. 一页结论

本项目要解决的问题不是“怎样从最终好坏分子构造偏好对”，而是：

> 给定冻结 DiffSBDD 行为策略生成的完整反向扩散轨迹，以及每条轨迹最终分子的连续奖励和失败信息，能否在不继续向环境采集训练数据的情况下，学习去噪状态—动作价值，并用优势加权的策略提取提高“实际好分子”的尝试级命中率。

该问题可以严格写成有限时域离线 MDP，因此属于真正的离线强化学习。当前数据已经保存 `z_T -> ... -> z_0 -> xh_final` 的完整状态序列，具备实现条件。推荐 IQL，而不是离线 PPO、DQN、原始 SAC 或直接照搬 AliDiff/E2PO。

项目的关键判断分为三层：

| 层级 | 目标 | 当前判断 |
|---|---|---|
| 工程可运行 | shard 能变成 transition，critic/actor 能反传、保存和恢复 | 可行性高 |
| 价值可学习 | Q/V 能在未见轨迹上区分好坏，并满足基本 Bellman 关系 | 有明显风险，需小样本门控 |
| 策略真实提升 | 新模型在独立生成、完整重对接和联合质量评价中超过原模型、BC、AWR | 无理论保证，作为最终实验结论 |

实施顺序固定为：

1. 冻结数据和源码语义；
2. 建立 episode/transition 数据契约；
3. 修正普通反向步、最终解码步和混合 timestep 的策略概率；
4. 完成奖励侧表和数据质量审计；
5. 先做 filtered BC 与 AWR；
6. 证明 geometric critic 可学习；
7. 再进行小规模 IQL actor 更新；
8. 最后才做全量训练和独立生成评价。

任何阶段失败，都应停在对应阶段分析，不允许用“继续堆训练步数”替代诊断。

---

## 1. 研究问题、假设与非目标

### 1.1 核心研究问题

在线 PPO/DDPO 在训练中后期依赖当前策略继续生成能提供有效正向学习信号的分子。当生成分布漂移、实际好分子变少或奖励代理被利用时，当前 rollout 的优势信号会变差，策略容易：

- 找不到足够好的新轨迹；
- 重复利用疏水性、分子大小或结构病理捷径；
- 在局部模式中继续更新；
- 训练 reward 改善但独立完整重对接和联合通过率不改善。

本方案使用一份固定的、覆盖好/中/坏/无效结果的完整轨迹数据集，先离线学习：

\[
Q(s_t,a_t),\qquad V(s_t),\qquad A(s_t,a_t)=Q(s_t,a_t)-V(s_t)
\]

然后提高数据集中高优势去噪动作的条件似然，降低对低优势动作的模仿权重。

### 1.2 可检验假设

本项目包含四个逐级假设：

**H1：数据契约成立。**
收集的 latent、timestep、节点 mask、最终解码状态和 reward 能无歧义重建为离线转移。

**H2：价值信号可学习。**
对于至少一段后期去噪区间，几何 critic 能在 held-out episode 上预测或排序最终连续奖励，而不只是识别节点数或 valid/invalid。

**H3：IQL 的状态条件优势优于纯终局加权。**
IQL 在相同数据和更新预算下优于 filtered BC 和 episode-reward AWR。

**H4：离线提升可以转化为实际生成提升。**
IQL checkpoint 在固定尝试数的独立生成中提高严格联合通过率，且不通过高 LogP、结构病理、重复 scaffold 或无效率换取表面 docking 改善。

只有 H1–H4 依次成立，才能声称“完整去噪轨迹离线 RL 对实际好分子生成有效”。

### 1.3 v1 明确不做的内容

- 不做多口袋泛化；当前固定为 8GJL/C1101。
- 不把节点数 \(N\) 纳入 actor 动作；v1 只优化给定 \(N\) 下的去噪策略。
- 不把 batch diversity 直接作为单条 transition 的 Markov 奖励。
- 不在训练中调用在线 rollout 补数据；独立生成只用于评价，不回灌 v1 训练集。
- 不把 AliDiff/E2PO 代码直接混入 IQL runtime。
- 不直接修改现有 PPO actor 使其同时承担离线 IQL；共享底层概率适配器，训练器保持解耦。
- 不一开始训练全部 1001 步；先做后缀时间窗可学习性验证。
- 不在没有 BC/AWR 对照时宣布 IQL 有效。

---

## 2. 证据基础与现有资产

### 2.1 理论和程序参考

| 参考 | 借用内容 | 本项目差异 |
|---|---|---|
| [DDPO](https://arxiv.org/abs/2305.13301) | 将扩散去噪写成多步 MDP；逐步策略条件概率 | DDPO 是在线 policy gradient，本项目使用静态轨迹做离线 actor-critic |
| [IQL](https://arxiv.org/abs/2110.06169) | expectile V、TD Q、advantage-weighted policy extraction | 原论文处理常规连续控制，本项目动作是可变节点三维 latent |
| [IQL 官方实现](https://github.com/ikostrikov/implicit_q_learning) | twin critic、target update、V/Q/actor 更新次序和基础默认值 | 本项目使用 PyTorch/DDP 和 E(3)-invariant critic |
| [AliDiff](https://arxiv.org/abs/2407.01648) | 最终分子偏好优化的强基线思路 | AliDiff 不学习逐步 Q/V，也不消费已保存的完整 reverse trajectory |
| 现有 DiffSBDD/DDPO 代码 | 采样、节点 mask、口袋条件、逐步高斯概率、最终解码概率 | 需要拆分 final decoder 分支并支持离线混合时间步 |

当前未发现可直接复用的“在分子扩散模型自身去噪 MDP 上运行 IQL”的成熟实现。因此，本方案属于有理论依据的研究性适配，而不是已有仓库的机械移植。

### 2.2 当前正式轨迹数据

正式运行：

```text
/home/md10/data/disk6/DiffB110/runs/diffsbdd_collect/
  diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/
```

正式设置：

| 项目 | 值 |
|---|---:|
| 行为策略 | 原始 DiffSBDD checkpoint |
| 口袋 | 8GJL/C1101 |
| 扩散步数 | 1000 |
| 最终解码动作 | 1 |
| 每 episode 动作数 | 1001 |
| 节点数 | 10–30 |
| 最大节点数 | 30 |
| 总目标 episode | 100,000 |
| shard episode 数 | 10 |
| seed | GPU/rank 对应 1–8 |
| latent 精度 | float32 |
| 预计总存储 | 约 147 GiB |

2026-07-30 审计快照（生成仍在继续）：

| 指标 | 快照值 |
|---|---:|
| 已完成 shard | 1,454 |
| 已完成固定尝试 | 14,540 |
| RDKit-valid | 9,952 |
| validity | 68.446% |
| manifest 累计字节 | 22,951,012,666 bytes |
| 当前运行目录占用 | 约 22 GB |

这些数字只能作为文档编写时的运行中快照，不是 100k 完成结果。

### 2.3 单 shard 现有结构

```text
format
latents
logps
timesteps
nodesxsample
x
categorical
advantages
sample_ids
metadata
reconstruction_rows
rng_state_after
```

正式 shard 的关键 shape：

```text
latents:      [10, 1002, 30, 13]
logps:        [10, 1001]
timesteps:    [10, 1001]
nodesxsample: [10]
x:            [10, 30, 3]
categorical:  [10, 30, 10]
```

语义是：

```text
latents[:, 0]      = z_T
latents[:, 1]      = z_{T-1}
...
latents[:, 1000]   = z_0
latents[:, 1001]   = xh_final（解码采样的连续坐标+连续特征）
x                   = xh_final 的坐标通道
categorical         = z_0 原子特征 argmax 后用于重建的 one-hot
```

每个 SDF 分子含 `sample_id`，可以稳定回接 shard 中对应 episode。无效分子虽然不一定写入 SDF，但仍有 trajectory 和 `reconstruction_rows`，必须在 reward sidecar 中显式记录并给予有限惩罚。

### 2.4 源码冻结前置条件

新分支基线为 `4031cfe`。正式轨迹是在主工作树的精确源码快照下生成的。进入实现前必须：

1. 读取正式 run 内保存的源码快照和 resolved config；
2. 比较收集器、模型适配器、rollout、molecule semantics 的 SHA-256；
3. 将数据生成所需的精确实现以独立逻辑提交迁入本分支，或清楚记录差异；
4. 禁止仅凭“文件名相同”假定训练端与生成端语义一致。

这一步是 H1 的组成部分。

---

## 3. 去噪 MDP 的严格定义

### 3.1 episode 上下文

固定上下文：

\[
c=(\text{pocket},N,\text{sampler schedule})
\]

其中：

- pocket 是固定 8GJL/C1101 口袋；
- 当前 full-atom pocket 约 349 个原子；
- \(N\in[10,30]\) 是 episode 开始前外部采样的 ligand 节点数；
- sampler schedule 固定为数据收集时的 1000 步离散化。

### 3.2 普通反向步

使用正向编号 \(k=0,\ldots,T-1\) 表示决策顺序，或使用扩散时间 \(t=T-1,\ldots,0\)。代码中必须统一保存两者，避免混淆。

对普通反向步：

\[
s_k=(z_t,t,c)
\]

\[
a_k=z_{t-1}
\]

\[
\pi_\theta(a_k|s_k)
=p_\theta(z_{t-1}|z_t,t,c)
\]

\[
s_{k+1}=(a_k,t-1,c)
\]

环境转移是确定的，因为 actor 采样出的 action 本身就是下一状态。

### 3.3 最终解码步

最后一步必须单独定义：

\[
s_T=(z_0,0,c,\text{decode})
\]

\[
a_T=x_{\text{final}}
\]

\[
\pi_\theta(a_T|s_T)
=p_\theta(x_{\text{final}}|z_0,c)
\]

然后环境执行非可微但确定/随机性受控的后处理：

```text
z_0 + xh_final
  -> xh_final 的坐标通道得到坐标 x
  -> z_0 的 atom feature argmax
  -> categorical one-hot
  -> bond reconstruction
  -> RDKit sanitize
  -> scoring
  -> terminal reward
```

当前冻结 DiffSBDD 实现虽然还采样并保存 `xh_final` 的连续 h 通道，但实际
输出 atom type 不消费这些通道。因此 final decode 的有效 action 和 actor
log-prob 只覆盖 xyz；保存的 h 通道作为 provenance 保留。final decode 不能
调用普通 `p(z_s|z_t)` 的 log-prob。

### 3.4 奖励和终止

标准 v1 定义：

\[
r_k=0,\qquad k<T
\]

\[
r_T=R(m,c)
\]

\[
d_T=1
\]

禁止把同一个最终奖励作为即时奖励重复写入 1001 个 transition。那会把 episode return 错误放大 1001 倍。

有限时域下初始设置：

\[
\gamma=1
\]

原因是：

\[
0.99^{1000}\approx 4.3\times10^{-5}
\]

若沿用常规控制任务的 `discount=0.99`，最早期去噪状态几乎收不到终局信号。`gamma<1` 只能作为明确的消融实验。

### 3.5 初始状态与行为策略

\[
z_T\sim\mathcal N(0,I)
\]

数据行为策略 \(\mu\) 是生成 100k episode 的冻结原始 DiffSBDD checkpoint。v1 数据来自单一 checkpoint，优点是行为语义一致，缺点是支持集受限。

IQL 可以在行为支持内偏向更优动作，但不能保证从静态单策略数据中可靠发现完全未出现的去噪决策。

---

## 4. 算法选择与基线

### 4.1 为什么以 IQL 为主

IQL 的关键优点是训练 Q/V 时只评价数据集中的动作，避免在 critic 尚不可靠时查询任意分布外高维 latent 动作。

使用 twin Q：

\[
\bar Q(s,a)=\min(Q_{\bar\phi_1}(s,a),Q_{\bar\phi_2}(s,a))
\]

V 的 expectile loss：

\[
\delta=\bar Q(s,a)-V_\psi(s)
\]

\[
L_V(\psi)
=\mathbb E_D[
|\tau-\mathbb 1(\delta<0)|\delta^2
]
\]

Q 的 TD loss：

\[
y=r+\gamma(1-d)V_\psi(s')
\]

\[
L_Q(\phi_i)
=\mathbb E_D[
(Q_{\phi_i}(s,a)-y)^2
]
\]

策略提取：

\[
A(s,a)=\bar Q(s,a)-V_\psi(s)
\]

\[
w(s,a)=
\min\left(
\exp(\beta A(s,a)),
w_{\max}
\right)
\]

\[
L_\pi(\theta)
=-\mathbb E_D[
w(s,a)\log\pi_\theta(a|s)
]
\]

实现中必须记录未裁剪和已裁剪权重、权重 ESS、优势分布、按 timestep 的权重分布。

### 4.2 强制基线

#### Baseline A：原始 DiffSBDD

不训练，作为所有独立生成指标的相对基线。

#### Baseline B：Filtered BC

仅选高奖励或严格 good episode：

\[
L_{\text{FBC}}
=-\mathbb E_{D_{\text{good}}}
[\log\pi_\theta(a|s)]
\]

回答“只学习好轨迹是否已经足够”。

#### Baseline C：Episode-reward AWR/RWR

不学习 Q/V，直接使用归一化终局奖励：

\[
w_i=\operatorname{clip}
\left(
\exp(\beta_R\tilde R_i),
0,w_{\max}
\right)
\]

\[
L_{\text{AWR}}
=-\mathbb E_D[w_i\log\pi_\theta(a|s)]
\]

回答“逐步状态条件价值是否比简单终局加权更有效”。

#### Baseline D：BC-all

等权模仿全部行为轨迹，用于检查“任何继续训练”本身是否造成漂移。

### 4.3 不采用的主算法

| 算法 | 不作为主线的原因 |
|---|---|
| DQN | 动作是可变节点高维连续 latent |
| 原始 SAC | actor 更新需要查询分布外动作 Q，静态数据下过估计风险高 |
| 离线 PPO | 依赖可靠行为 ratio；现有 log-prob 有截断且 final step 旧消费路径语义错误 |
| CQL | 可作为后续对照，但高维几何动作上的随机/策略动作负样本计算昂贵 |
| AliDiff/E2PO | 是最终分子偏好优化，不学习逐步价值；保留为另一研究路线 |
| 纯回报回归 | 可以作为 critic 预训练/辅助项，但不能替代 IQL 的状态条件价值检验 |

---

## 5. 数据层设计

### 5.1 不重写原始 shard

100k 完整轨迹约 147 GiB。不得为了添加 reward、split 或统计字段重写全部 `.pt`。

采用三层结构：

```text
原始不可变 trajectory shard
  + episode index / split manifest
  + reward sidecar
  -> 训练时 lazy join
```

### 5.2 `DenoisingEpisodeFacts`

计划 dataclass：

```python
@dataclass(frozen=True)
class DenoisingEpisodeFacts:
    sample_id: str
    shard_path: str
    episode_index: int
    rank: int
    shard_index: int
    seed: int
    num_nodes: int
    num_diffusion_steps: int
    rdkit_valid: bool
    reconstruction_status: str
    reward_total_raw: float
    reward_total_train: float
    split: str
```

实际实现还需保留每个 reward component、失败原因、SMILES/scaffold（若存在）和数据版本。

### 5.3 `DenoisingTransitionBatch`

训练 batch 计划字段：

```text
state_latent                 [B,N,D]
action_latent                [B,N,D]
next_state_latent            [B,N,D] 或 terminal placeholder
action_mask                  [B,N,D]
recorded_timestep            [B]
source_diffusion_timestep    [B]
target_diffusion_timestep    [B]
node_mask          [B,N,1]
nodesxsample       [B]
reward             [B]
done               [B]
is_final_decode    [B]
sample_id          list[str]
reward components  可选 CPU metadata
```

由于普通反向步满足 `next_state_latent == action_latent`，仍保留两个语义字段；数据 loader 可共享 tensor storage，但接口不混用。普通步的 source/target timestep 分别为 `s+1` 和 `s`；final decode 为 `0` 和 terminal sentinel `-1`。final action 只有 xyz 通道影响输出，atom type 来自 state `z_0` 的特征通道，因此 final `action_mask` 只开放 xyz。

### 5.4 episode 级切分

同一 episode 的 1001 个 transition 必须全部属于同一个 split。

建议固定：

```text
train 80%
validation 10%
test 10%
```

切分过程：

1. 基于稳定 `sample_id` 哈希；
2. 按 reward stratum、validity 和节点数进行确定性分层校正；
3. 输出不可变 `split_manifest.jsonl`；
4. 保存算法版本、源 manifest 哈希和统计摘要；
5. 不得在训练后根据效果重新划分 test。

test split 用于离线 critic/weight 诊断，不替代新 checkpoint 的独立生成评价。

### 5.5 lazy shard sampler

禁止展开为约 1.001 亿个独立文件或 CSV 行。

推荐读取策略：

1. 每个 DDP rank 获得不重叠 shard 路径；
2. 顺序或块随机加载一个 shard；
3. 从已加载的 10 个 episode 中抽多个 timestep；
4. 一个 shard 尽量产生多个 training batch，摊薄约 15.8 MB 读取成本；
5. 维护有限 LRU shard cache；
6. cache 只保存 CPU tensor；
7. microbatch 时才移动到 GPU；
8. 记录读取带宽、cache hit rate、等待时间和 GPU data stall。

第一版按相同 timestep 组 batch，规避现有模型适配器取 `t[0]` 的问题。后续支持混合 timestep 后再允许完全随机 transition batch。

### 5.6 时间分层采样

均匀随机抽取 1001 步会让早期高噪声状态占据大量训练预算，而这些状态可能很难预测终局质量。

支持配置：

```yaml
time_sampling:
  mode: stratified
  bins:
    - [0, 49]
    - [50, 99]
    - [100, 249]
    - [250, 499]
    - [500, 749]
    - [750, 999]
    - [1000, 1000]  # final decode
```

这里的下标必须明确是“决策顺序 index”还是“扩散 timestep”。配置和日志统一使用：

- `decision_index=0`：最早的 \(z_T\to z_{T-1}\)；
- `decision_index=1000`：最终 \(z_0\to xh_{\text{final}}\)。

### 5.7 数据完整性检查

每个 shard 至少验证：

- format 版本；
- `latents_len == action_len + 1`；
- 普通步数 1000、最终步数 1；
- timestep 顺序；
- padding 全零或满足收集契约；
- `nodesxsample` 范围 10–30；
- tensor finite；
- `sample_id` 全局唯一；
- reconstruction row 与 sample_id 一一对应；
- SDF 中出现的 sample_id 必须能回接；
- 无 SDF 的 episode 仍有明确失败事实；
- reward sidecar 不缺行、不重复；
- split 不跨 episode；
- 源 shard SHA 与索引一致。

---

## 6. 奖励与“实际好分子”定义

### 6.1 训练奖励和最终评价必须分离

定义两套概念：

**训练用连续奖励 \(R_{\text{train}}\)**
为 critic 提供密集的 episode 间排序信息。

**独立评价用严格联合通过 `actual_good`**
回答最终有多少真正可用候选，不能被单项补偿。

不能用训练 reward 自己证明训练有效。

### 6.2 reward sidecar

建议每个 episode 保存：

```text
sample_id
reconstruction_success
sanitize_success
rdkit_valid
pb_valid
docking_input_success
docking_success
docking_raw
ligand_efficiency
qed
sa_raw
logp
tpsa
hba
hbd
fragment_count
carbon_fraction
hetero_atom_count
ring_count
aromatic_ring_count
three_member_ring_count
fused_ring_count
complex_ring_count
brenk_count
chembl_prior / classifier（若启用）
reward_total_raw
reward_total_train
actual_good
failure_reason
reward_version
```

所有失败值必须有限。原始不可用数值可另存 `null`，但提供给 critic 的 `reward_total_train` 不得为 `NaN/Inf`。

### 6.3 v1 奖励原则

- reconstruction/RDKit-invalid：显式最低有限惩罚；
- docking backend failure 与真实差 docking 分开记录；
- docking 使用 ligand efficiency 或尺寸控制，避免只偏向大分子；
- LogP、TPSA、HBA/HBD、卤素、片段和环病理使用连续软惩罚；
- PB-invalid 提供独立惩罚，不与 RDKit-valid 混为一项；
- batch diversity 不进入单 episode Bellman reward；
- 严格 `actual_good` 不建议直接作为唯一二值训练 reward；
- 原始 component、归一化 component 和最终总分全部落盘。

具体权重不能凭直觉写死。Phase 0 必须根据 100k 分布、现有中期证据和已使用 reward 组件给出三个候选版本，再做固定数据消融：

1. `reward_v1_balanced`：docking/LE + 类药性 + 几何病理连续项；
2. `reward_v1_target`：更强调靶点分数，但带明确边界；
3. `reward_v1_quality`：更强调 PB/类药性和结构健康。

### 6.4 `actual_good` 的作用

`actual_good` 是预注册的联合门槛，例如：

```text
sanitize-valid
AND PB-valid
AND 完整重对接成功
AND docking/ligand-efficiency 达标
AND QED/SA/LogP/TPSA 在规定边界
AND 无关键结构病理
AND 单一主片段
```

最终阈值必须在看 IQL 独立生成结果前冻结，并记录版本。训练期快速 score-only 可以参与奖励，但正式 `actual_good` 优先使用独立完整重对接结果。

### 6.5 数据是否足够的审计

奖励完成后必须报告：

- 每个失败层级数量；
- reward 各分位数；
- top 1%、5%、10% 的有效率和 scaffold 数；
- strict actual-good 数量和比例；
- actual-good 在节点数、rank、seed、shard、scaffold 上的覆盖；
- reward 与节点数、LogP、碳比例、环数的相关；
- actor 权重的 effective sample size（ESS）。

操作性门槛：

- 若严格 actual-good 少于约 500 条，仍可使用连续 reward 训练 critic，但“学习实际好轨迹”的结论证据偏弱；
- 若严格 actual-good 少于约 200 条或集中于极少 scaffold，不进入全量 IQL actor 阶段，优先调整奖励/扩大支持；
- actor 权重若长期由极少数 episode 主导，必须降低 \(\beta\)、加强 clipping 或按 scaffold/reward strata 重采样；
- invalid 若占比过大，critic 训练必须分层，否则容易只学 validity。

这些是工程 go/no-go 门槛，不是 IQL 理论定理，应随完整数据审计记录调整理由。

---

## 7. 策略概率适配器

### 7.1 现有两个阻断问题

#### 问题 A：final decode 被当作普通 reverse transition

现有 PPO actor 对 `num_timesteps + 1` 个动作统一调用 `ddpo_log_prob`。真实 shard 核对显示：

```text
stored final logp:             -7.761638
correct final decoder logp:    -7.761608
generic reverse logp:        -100.000000
```

因此必须增加独立 forward mode，例如：

```text
mode="reverse_transition_log_prob"
mode="final_decode_log_prob"
```

#### 问题 B：混合 timestep 使用 `t[0]`

当前 DiffSBDD 包装器在批内 timestep 不相等时仍取第一个标量 timestep。第一阶段使用同 timestep batch；接口修复后增加混合 timestep 等价测试。

### 7.2 已实现接口

```python
class DiffSBDDPolicyAdapter:
    def log_prob(
        self,
        transitions: DenoisingTransitionBatch,
    ) -> PolicyLogProbOutput:
        ...
```

输出：

```python
@dataclass
class PolicyLogProbOutput:
    log_prob: torch.Tensor
    mean: torch.Tensor
    sigma: torch.Tensor
    ordinary_mask: torch.Tensor
    final_decode_mask: torch.Tensor
```

adapter 将 ordinary transition 按 `target_diffusion_timestep` 分组后分别调用
`mode="offline_reverse_log_prob"`，避免 DiffSBDD dynamics 的标量 timestep
接口误用 `t[0]`；final transition 单独调用
`mode="offline_final_decode_log_prob"` 并使用 coordinate-only action mask。

### 7.3 log-prob 聚合语义

现有默认是按有效原子归一化的稳定 pseudo/joint score，并进行 `[-100,0]`
截断；数据层命名为 `recorded_behavior_score`，不把它冒充未截断、归一化
的行为策略 log-prob。IQL 不需要行为策略 ratio，因此旧 `logps` 的截断不
阻止 critic 学习。

但 actor 的加权 NLL 必须预注册一种聚合方式：

- `mean_per_atom`：不同节点数间尺度更稳定，作为 v1 默认候选；
- `full_per_atom`：包含 Normal 常数和 sigma，可作为消融；
- raw full molecule sum：容易随节点数强烈变化，不作为首选。

无论选择哪种，都必须：

- 数据动作当前策略 log-prob 可微；
- padding 不参与；
- constrained action dimension（若有）正确 mask；
- ordinary/final 分开统计；
- 按 timestep 和节点数监控尺度；
- 不使用 stored old logp 计算 PPO ratio。

### 7.4 必须通过的 round-trip 测试

在与生成一致的 GPU/精度上，固定真实 shard 样本：

- 最早普通步；
- 中间普通步；
- 最后普通步；
- final decode；
- 不同节点数；
- batch size 1 和 batch size >1；
- 同 timestep batch；
- 混合 timestep batch。

检查：

- ordinary step 的 stored score 与同聚合/截断定义下的 recomputed score
  在预设容差内；
- final step 分别验证 legacy 全 13 维 score round-trip 和训练使用的
  coordinate-only score，二者不得混称；
- batch 与逐样本计算一致；
- final branch 与普通 branch 不串用；
- gradient 只流向当前 actor 参数；
- padding 修改不改变有效区域 log-prob；
- CPU mismatch 不被误当作 GPU 生成概率错误，正式判据使用同设备测试。

2026-07-30 使用正式 checkpoint、GPU 0 和真实 shard 的 v1 验证结果：

| 检查 | 结果 |
|---|---:|
| ordinary decisions `0/500/999` stored vs recomputed | 逐点误差 `0` |
| legacy final 13D score round-trip | 绝对误差 `1.24e-5` |
| mixed batch vs 逐条计算 | 最大绝对误差 `9.90e-6` |
| padding/final-h 扰动不变性 | 最大绝对误差 `2.67e-5` |
| 收到梯度的参数 tensor | 111 |
| 梯度 finite | 是 |

GPU scatter/reduction 存在约 `1e-5` 级非确定性，因此正式 round-trip 容差设为
`1e-4`，不能要求跨 forward bitwise equal。

---

## 8. 几何 V/Q critic 设计

### 8.1 设计约束

critic 必须满足或近似满足：

- 三维整体平移不变；
- 对旋转/反射的标量价值不变；
- ligand 原子排列不变；
- 可变节点数；
- pocket 条件；
- timestep 条件；
- state 和 action 的相对变化可辨认；
- 输出每 episode 一个标量；
- 不通过 padding、节点编号或 shard 信息作弊。

### 8.2 V 网络

输入图：

```text
pocket nodes + current ligand state nodes
```

节点特征：

- pocket atom/residue feature；
- ligand 连续 atom latent \(h_t\)；
- pocket/ligand role embedding；
- sinusoidal timestep embedding；
- 节点数 embedding（可选，但必须做去除消融）。

坐标：

- pocket 原子坐标；
- ligand 当前 latent 坐标 \(x_t\)。

输出：

1. EGNN/message-passing；
2. ligand 节点 masked mean/sum pooling；
3. pocket pooling；
4. 拼接 timestep/context；
5. MLP scalar head 得到 \(V(s)\)。

### 8.3 Q 网络

首版采用显式双 ligand-copy 图：

```text
pocket nodes
state ligand nodes at x_t
action ligand nodes at x_{t-1}
```

边：

- pocket-pocket 半径边；
- state-state ligand 边；
- action-action ligand 边；
- pocket-state cross edges；
- pocket-action cross edges；
- state atom i 与 action atom i 的 identity edge。

节点 scalar feature：

- \(h_t\)、\(h_{t-1}\)；
- role embedding：pocket/state/action；
- timestep embedding；
- identity correspondence type。

EGNN 只通过距离和标量消息形成图表示，最终输出 E(3)-invariant \(Q(s,a)\)。

不把原始三维位移向量直接拼入普通 MLP，因为这会破坏旋转不变性。

### 8.4 twin Q 与 target

- `Q1`、`Q2` 使用独立参数；
- V 使用独立网络；
- target Q 采用 Polyak update；
- 初始候选 `target_tau=0.005`，来自官方 IQL 常用实现，需按本任务稳定性调整；
- actor 默认读取 `min(target_Q1,target_Q2)-V`；
- 是否使用 target V 作为 Q target 必须在实现文档中固定，不能训练中改变。

### 8.5 图构建性能

每个 episode 约：

- 349 pocket nodes；
- 最多 30 state ligand nodes；
- Q 再增加最多 30 action nodes。

不能对大 batch 使用全连接稠密边。v1：

- pocket-pocket 静态半径边可缓存；
- ligand/pocket cross edge 按 cutoff 构建；
- identity edge 显式添加；
- 记录每类边数量和构图时间；
- 初始 batch 由显存实测决定，不硬编码；
- 若原项目 dense `get_edges` 成为瓶颈，单独实现经测试的 radius graph builder。

### 8.6 防作弊诊断

critic 必须与以下简单 baseline 比较：

- 常数预测；
- 仅节点数；
- 仅 timestep；
- 节点数+timestep；
- 仅 validity 分类；
- 最终分子描述符回归（只作为上限参考，不能用于中间状态 policy）。

如果 geometric critic 只比节点数 baseline 略好，不能进入 actor 更新阶段。

### 8.7 2026-07-30 实现与容量验证状态

当前分支已实现首版解耦几何 critic：

- `critic_adapter.py`：把不可变 DiffSBDD transition 与固定 pocket context
  转成 padded critic 输入；final decode 的 action atom latent 由 action mask
  明确屏蔽；
- `models/graph_builder.py`：按 pocket/pocket、ligand/ligand、
  pocket/ligand 三类 cutoff 构造稀疏有向半径边，并额外加入
  state-action identity edge；
- `models/encoders.py`：消息只依赖 scalar node feature、距离 RBF 和 edge
  type embedding，不将笛卡尔分量输入普通 MLP；
- `models/value_critic.py`：只读取 state+pocket；
- `models/q_critic.py`：读取 state+action+pocket，`TwinQCritic` 的 Q1/Q2
  参数完全独立；
- DiffSBDD adapter 可导出 detached pocket context。

真实 checkpoint/pocket/shard 联调发现一个关键坐标语义：记录的 ligand
latent 已按质心居中，而 PDB pocket 是绝对晶体坐标。若不显式对 pocket
按有效节点质心居中，8 Å pocket-ligand cross edge 会全部消失。当前
`build_geometric_critic_input()` 已统一该坐标系，真实 349-node pocket 的
居中后质心最大绝对值为 \(9.57\times10^{-6}\)。

验证结果：

- 94 个相关回归测试通过、1 个跳过；
- scalar 输出通过旋转、平移、反射、ligand/pocket permutation、
  padding、final-action mask 不变性测试；
- Q1、Q2、V 均有独立有限梯度；
- 10 条真实 episode、QED/invalid finite penalty **容量 smoke** 中，500
  step 后总 MSE 从 1.4312 降至 0.00970（0.68%），Q1/Q2/V Pearson
  分别为 0.9997/0.9974/1.0000，峰值 CUDA allocation 约 591 MiB。

该 smoke 只排除了“网络/数据接口连不上或小样本完全拟合不了”，并不满足
Phase 3 的科学验收。QED proxy 不是正式 reward，训练集拟合也不能替代
held-out reward 排序和简单 baseline 对照。

---

## 9. 软件模块与解耦边界

计划新增结构：

```text
ddpo/offline_rl/
  __init__.py
  protocol.py                 # episode/transition/output dataclass
  dataset.py                  # immutable shard reader + lazy join
  index.py                    # episode index/split manifest
  sampler.py                  # shard-local/time-stratified sampler
  reward_sidecar.py           # reward join/validation，不重复分子重建
  policy_adapter.py           # ordinary/final actor log-prob
  losses.py                   # expectile、Q TD、AWR/IQL actor loss
  trainer.py                  # update order、AMP、DDP、checkpoint
  metrics.py                  # 训练/诊断聚合
  recording.py                # CSV/JSON/log/checkpoint
  models/
    __init__.py
    graph_builder.py
    encoders.py
    value_critic.py
    q_critic.py

ddpo/main_diffsbdd_offline_iql_ddp.py

scripts/offline_rl/
  build_diffsbdd_episode_index.py
  score_diffsbdd_episodes.py
  audit_diffsbdd_offline_dataset.py
  validate_diffsbdd_transition_contract.py
  evaluate_offline_critic.py

launch/
  diffsbdd_offline_iql_smoke.yaml
  diffsbdd_offline_iql_pilot.yaml
  diffsbdd_offline_iql_formal.yaml
  smoke_diffsbdd_offline_iql.sh

test/offline_rl/
  test_protocol.py
  test_dataset.py
  test_split.py
  test_policy_adapter.py
  test_final_decode_logprob.py
  test_mixed_timestep.py
  test_graph_invariance.py
  test_iql_losses.py
  test_trainer_checkpoint.py
```

### 9.1 允许复用

- `DiffSBDDModel` 的采样和概率底层计算；
- 现有 pocket 解析与缓存；
- 统一 molecule semantics；
- `ddpo.rewards.factory` 和现有 reward component；
- DataProto/TensorDict 仅在确实适合的边界；
- 项目现有本地记录规范。

### 9.2 禁止耦合

- 不让 offline dataset 伪装成 PPO replay；
- 不在 PPO actor 内堆 IQL critic 状态；
- 不让 reward scorer 修改原始 trajectory shard；
- 不让 critic 依赖 RDKit molecule 作为中间状态输入；
- 不让评价脚本反向写入训练 split；
- 不让 E2PO preference pair schema 混入 IQL transition schema；
- 不让远程 WandB 成为基本训练证据。

---

## 10. 训练更新流程

### 10.1 单步更新次序

遵循 IQL 参考实现的逻辑顺序：

1. 从 target twin Q 与当前 V 更新 V；
2. 用更新后的/配置冻结的 V 构造 TD target，更新 Q1/Q2；
3. Polyak 更新 target Q；
4. 计算 detached advantage；
5. 更新 actor；
6. 写本地标量、分布统计和 timestep 分层指标。

具体是 actor 先于 Q 还是 Q 先于 actor，必须通过配置/实现固定并记录。首版建议每轮使用同一个 pre-update target Q 快照，避免更新次序产生隐式差异。

### 10.2 critic loss

默认 Huber 或 MSE 需要用 reward scale smoke 决定。至少记录：

- `q1_loss`, `q2_loss`, `v_loss`；
- `q1_mean`, `q2_mean`, `v_mean`；
- `td_target_mean/std/min/max`；
- `bellman_residual`；
- `q_disagreement`；
- expectile 上/下侧样本比例；
- 按 time bin 的上述统计。

### 10.3 actor loss

actor 权重：

\[
w=\exp(\beta A)
\]

必须做：

- `advantage.detach()`；
- exponent 输入 clamp；
- 最终权重 clamp；
- 记录 unclipped/clipped 比例；
- 记录 ESS：

\[
\operatorname{ESS}
=\frac{(\sum_i w_i)^2}{\sum_i w_i^2}
\]

- 按 ordinary/final、time bin、节点数记录 NLL；
- 可选参考策略 KL 只作为稳定器/监控，不回到 PPO ratio。

### 10.4 actor 更新范围

按风险逐级开放：

1. 仅 actor 输出/最后层；
2. 最后若干 dynamics block；
3. 全模型小学习率。

每一级必须与原策略比较：

- parameter delta；
- held-out behavior NLL；
- reference KL；
- 新生成 validity 和 descriptor drift。

不允许第一次 pilot 就全参数高学习率更新。

### 10.5 AMP 和数值精度

- trajectory 仍以 float32 为事实源；
- critic 可在验证后使用 AMP；
- log-prob 的 sigma 很小时优先 float32；
- final decoder round-trip 在 float32 完成；
- AMP 溢出/跳步写入日志；
- 所有 NaN/Inf 触发可定位的 fail-fast，不静默替换 critic loss。

---

## 11. 初始超参数与消融空间

以下只是 pilot 起点，不是正式最优配置：

| 参数 | 初始候选 | 消融 |
|---|---:|---|
| `discount` | 1.0 | 0.999、仅作为消融 |
| `expectile` | 0.8 | 0.7、0.9 |
| `target_tau` | 0.005 | 0.001、0.01 |
| critic LR | 3e-4 | 1e-4、1e-3 |
| value LR | 3e-4 | 1e-4 |
| actor LR | 1e-5 | 3e-6、3e-5 |
| advantage \(\beta\) | 由归一化后 A 分布定 | 低/中/高三档 |
| weight clip | 20 或 50 | 10、100 |
| grad clip | 1.0 | 0.5、5.0 |
| actor update ratio | critic 1–4 次 / actor 1 次 | 仅在 critic 稳定后选择 |
| time window | final 100 步 | 50、250、500、1001 |

官方 IQL 常见默认包含 `expectile=0.8`、`tau=0.005`、网络学习率 `3e-4`；本项目 actor 是预训练大型几何扩散模型，因此 actor LR 从更保守的 `1e-5` 起步。

不得一次展开所有组合。调参顺序：

1. reward scale；
2. critic 架构和时间窗；
3. expectile；
4. advantage temperature/clip；
5. actor LR/解冻范围。

test split 不用于调参。

---

## 12. 分阶段实施与验收门槛

### Phase 0：数据和奖励审计

**任务**

- 冻结正式 run 源码/config/shard 版本；
- 建 episode index；
- 完成 reward sidecar；
- 输出质量、分布、相关性、good strata 和 scaffold 报告；
- 预注册 `actual_good_v1`。

**产物**

```text
analysis/diffsbdd_offline_iql_data_<date>/
  episode_index.jsonl
  split_manifest.jsonl
  reward_sidecar.csv
  dataset_summary.json
  reward_distribution.csv
  global_distribution_comparison.png
  reward_failure_modes.png
```

**通过条件**

- 100% episode 有唯一 reward/failure 行；
- critic 输入 reward 全 finite；
- split 无 episode 泄漏；
- good/medium/bad/invalid 各层数量可报告；
- reward 未被节点数或单一 LogP 几乎完全决定。

### Phase 1：transition 与策略概率契约

**任务**

- 实现 protocol、dataset reader、ordinary/final adapter；
- 修复混合 timestep；
- 完成同设备 round-trip；
- 建 2–10 episode deterministic fixture。

**通过条件**

- 所有 shape/mask/timestep 测试通过；
- final decoder 不走 ordinary branch；
- batch 与逐样本 log-prob 一致；
- CPU/GPU 差异有明确容差和记录；
- actor NLL 有限且梯度非零。

### Phase 2：BC/AWR 基线

**任务**

- BC-all；
- filtered BC；
- episode-reward AWR；
- 仅使用后 100 步；
- 小规模独立生成。

**通过条件**

- checkpoint 可恢复；
- actor 权重 ESS 不崩溃；
- 不出现明显 validity/descriptor 崩坏；
- 得到供 IQL 比较的固定预算结果。

### Phase 3：critic 可学习性

**任务**

- 先用 1,000–5,000 episode；
- final/后 50/后 100/后 250 时间窗；
- V + twin Q；
- held-out 诊断；
- 与节点数/timestep/validity baseline 比较。

**通过条件**

- held-out reward 排序显著优于简单 baseline；
- late-time Q 能区分好/坏 valid 分子，而不仅是 invalid；
- Q1/Q2 不系统性发散；
- Bellman residual 有稳定下降；
- 按 timestep 的值域合理；
- 对旋转/平移/原子置换的 scalar 输出在容差内不变。

失败时先停止 actor 更新，诊断 reward、图架构或 horizon。

### Phase 4：小规模 IQL actor

**任务**

- critic warm-up；
- 仅后 50–100 步更新 actor；
- 最后层/部分 block 解冻；
- 与 BC/AWR 同预算对照；
- 生成快速评价。

**通过条件**

- 好轨迹平均 advantage 高于坏轨迹；
- 权重没有集中到极少 episode；
- reference KL/parameter delta 受控；
- 固定尝试级 validity 不出现不可接受下降；
- 至少在一个预注册联合指标上超过 AWR，才进入扩大阶段。

### Phase 5：全量 pilot

**任务**

- 使用完整 train split；
- 扩展至后 250/500 步；
- 选择 2–3 个 seed；
- 1 GPU smoke 后再 DDP；
- checkpoint 每 epoch 保存。

**通过条件**

- 三个 seed 方向一致或能解释方差；
- 本地日志/metrics/checkpoint 完整；
- 没有数据重复、rank overlap 或 I/O stall 主导；
- IQL 相对 AWR 的收益不只来自更大更新量。

### Phase 6：正式独立评价

先进行每 checkpoint 的快速固定尝试筛查，再对入选 checkpoint 做正式评价。

建议：

```text
快速：每方法/seed 至少 1,000 fixed attempts
正式：根据 actual-good 稀有率扩大到足够估计置信区间的固定尝试数
重复：至少 3 个生成重复
seed：严格遵守 GPU ID 1-based 规范
```

正式比较：

- Original；
- BC-all；
- filtered BC；
- AWR；
- IQL；
- AliDiff/E2PO 仅在另一路线实现成熟后作为外部方法比较。

---

## 13. 评价指标与统计

### 13.1 主指标

唯一主指标：

\[
\text{actual-good hit rate}
=\frac{\#\text{strict actual-good}}
{\#\text{fixed generation attempts}}
\]

分母必须是固定尝试数，不能只使用 SDF 写出数、valid 数或 docking 成功数。

报告 Wilson/Bootstrap 置信区间，并报告绝对差和相对差。

### 13.2 必须报告的漏斗

```text
fixed attempts
-> reconstruction
-> RDKit sanitize-valid
-> PB-valid
-> docking input success
-> full redocking success
-> docking threshold
-> property/structure joint pass
-> unique actual-good
```

### 13.3 次要指标

- QVina score-only；
- UniDock 完整重对接；
- ligand efficiency；
- QED、SA、LogP、TPSA、HBA/HBD；
- carbon fraction、hetero atom count、element diversity；
- ring/aromatic/three-member/fused/complex rings；
- fragment、Brenk、卤素；
- uniqueness；
- Murcko scaffold 数、多样性、新颖性；
- 与 ChEMBL 和原始 DiffSBDD 分布距离；
- top reward decile 的描述符漂移。

### 13.4 离线 critic 指标

- held-out reward MAE/RMSE；
- Spearman/Kendall rank correlation；
- top-vs-bottom reward AUC；
- Bellman residual；
- Q calibration by reward decile；
- Q1/Q2 disagreement；
- advantage by actual-good stratum；
- metrics by decision-time bin；
- actor weight ESS；
- weight clipping rate。

离线 critic 指标只决定是否值得更新 actor，不替代生成评价。

---

## 14. 训练记录、配置和 checkpoint

### 14.1 正式 run 目录

```text
runs/diffsbdd_offline_iql/<run_name>/
  config/
    input_config.yaml
    resolved_config.yaml
    source_snapshot/
  logs/
    train.log
  metrics/
    metrics.csv
    timestep_metrics.csv
    weight_histograms.jsonl
  checkpoints/
    epoch_0001.pt
    ...
  manifests/
    data_manifest.json
    split_manifest.json
    reward_manifest.json
  evaluation/
  completion_report.json
```

### 14.2 日志必须包含

- git commit/status；
- conda 环境；
- CUDA devices/GPU 数；
- 数据目录及 SHA/manifest；
- reward/split 版本；
- actor/critic 参数量；
- time window 和 sampling 策略；
- epoch/global step/elapsed；
- V/Q/actor/component metrics；
- ordinary/final NLL；
- advantage/weight/ESS；
- 每 epoch checkpoint 路径；
- 异常 traceback；
- completion/interruption 状态。

### 14.3 checkpoint

每个完成 epoch 保存：

- actor；
- V；
- Q1/Q2；
- target Q1/Q2；
- optimizers/schedulers/scaler；
- epoch/global step；
- sampler state；
- resolved config；
- data/reward/split manifest hash；
- metrics summary；
- RNG state。

必须实现从 checkpoint 恢复后下一 batch 顺序和权重统计可解释，至少在 deterministic smoke 中验证。

---

## 15. 测试矩阵

### 15.1 单元测试

| 模块 | 测试 |
|---|---|
| protocol | shape、dtype、必需字段、terminal 语义 |
| dataset | shard lazy load、reward join、missing/duplicate |
| split | episode 不泄漏、确定性、strata 统计 |
| sampler | rank 不重叠、time bin 分布、resume |
| reward | finite failure penalty、component 名称 |
| policy adapter | ordinary/final 分支、mask、梯度 |
| graph builder | edge 类型、padding、identity edge |
| V/Q | 输出 shape、finite、E(3)/置换不变 |
| losses | expectile 符号、terminal TD、weight clip/ESS |
| trainer | update 次序、DDP、checkpoint/resume |

### 15.2 集成 smoke

最小 smoke：

```text
2–10 episodes
3–10 ordinary timesteps + final decode
1 GPU
1 epoch
actor 默认冻结
V/Q 至少一次更新
checkpoint + resume
```

然后：

```text
100 episodes
后 50/100 步
actor 最后层更新
生成 20–100 fixed attempts 仅验证流程
```

正式效果结论不能来自 smoke。

### 15.3 代码变更后的标准检查

1. `py_compile` 所有修改 `.py`；
2. focused pytest；
3. 离线 IQL 专用 1 GPU smoke；
4. 若修改共享 DiffSBDD adapter，再运行现有 DiffSBDD shard/PPO 兼容测试；
5. 共享 DDPO 管线受影响时再运行 `launch/smoke.sh`；
6. 检查 CSV/JSON/checkpoint/resume；
7. 所有 shell launcher 最终执行 `bash -n`。

---

## 16. 主要失败模式与处理

| 失败模式 | 表现 | 处理 |
|---|---|---|
| final step 概率错误 | final NLL 饱和到 -100、梯度异常 | 独立 final decoder adapter 和 round-trip |
| 混合 timestep 错算 | batch 与逐样本不一致 | 先 bucket，后向量化修复 |
| critic 只学 validity | invalid/valid AUC 高，valid 内排序差 | valid 内分层训练和单独指标 |
| critic 只学节点数 | Q 与 N 高相关 | 节点数 baseline/平衡采样/去除消融 |
| 1001 步信号传不过去 | early Q 无相关、TD 累积误差 | 后缀课程、gamma=1、可选 MC 辅助 |
| Q 过估计 | Q 急升、Q1/Q2 分歧扩大 | twin min、target、降低 LR、暂停 actor |
| actor 权重塌缩 | ESS 极低、少数轨迹主导 | 降 \(\beta\)、clip、stratified batch |
| 策略离线外推 | NLL/KL 漂移、validity 崩 | 小 LR、局部解冻、reference KL gate |
| reward shortcut | docking 好但 LogP/病理恶化 | 连续 guard + 独立 actual-good |
| 读盘成为瓶颈 | GPU idle、data wait 高 | shard-local 多采样、LRU、worker 调优 |
| 单 checkpoint 支持不足 | IQL 与 AWR 无差或过拟合 | 扩行为数据、混合 checkpoint 需显式 behavior ID |
| strict good 太少 | 高权重样本/scaffold 极少 | 连续 reward、扩大数据、停止全量 actor |
| 多口袋误宣称 | 固定口袋结果被泛化表述 | 文档和实验明确 same-target |

---

## 17. 资源与工作量判断

### 17.1 存储和 I/O

- 原始轨迹约 147 GiB；
- 不复制、不重写；
- index/reward/metrics 预计远小于轨迹主体；
- 训练读取应尽量连续 shard-local；
- 需要监控磁盘吞吐而不是只看 GPU 显存。

### 17.2 GPU

当前 8 GPU 正在生成轨迹，不能占用其做正式训练。可并行完成：

- 数据 schema；
- index/reward sidecar；
- CPU 数据审计；
- 单元测试；
- critic 结构代码；
- 小 fixture 测试。

GPU 空闲后依次进行：

1. probability round-trip；
2. 1 GPU critic smoke；
3. 1 GPU actor pilot；
4. DDP full pilot。

### 17.3 工程工作量

若不计 docking/reward 批处理和 GPU 排队，预计需要约 8–15 个集中开发工作日完成“数据契约—基线—critic—小规模 IQL—可恢复训练”的工程闭环。证明独立生成提升通常还需要多个实验周期，不能把工程完成日期当成科研成功日期。

最耗时部分预计是：

1. reward 数据审计和 actual-good 冻结；
2. geometric Q 输入与构图性能；
3. 1001 步 credit assignment；
4. 同预算 BC/AWR/IQL 正式对照。

---

## 18. Git/worktree 与提交规划

当前独立工作树：

```text
/home/md10/data/disk6/DiffB110-offline-iql
```

分支：

```text
diffsbdd-xh-offline-iql
```

不在 `diffsbdd-xh-e2po` 中实现 IQL，也不移动该 worktree 的未跟踪规划文档。

建议逻辑提交：

```text
260730-4 规划DiffSBDD离线IQL方案
<date>-1 建立离线轨迹数据契约
<date>-2 修复普通步与最终步概率接口
<date>-3 增加离线奖励索引与数据审计
<date>-4 实现几何V和双Q网络
<date>-5 实现IQL损失与训练器
<date>-6 增加BC和AWR对照
<date>-7 增加离线IQL烟测与恢复
<date>-8 完成小规模IQL试验分析
```

每个提交只包含一个逻辑层，不把数据、critic、trainer 和实验结果塞入一个提交。

---

## 19. 最终验收与停止条件

### 19.1 工程完成定义

- 数据索引、奖励、split 可复现；
- ordinary/final/mixed timestep 概率契约通过；
- V/twin Q/target/actor 全部可训练；
- BC/AWR/IQL 共用同一数据和评价接口；
- 1 GPU 与 DDP smoke 通过；
- 每 epoch checkpoint 可恢复；
- 正式日志和指标完整。

### 19.2 科学成功定义

在预注册的固定尝试评价中：

1. IQL 的 actual-good hit rate 超过原始模型；
2. IQL 超过 filtered BC 和 AWR，或在同命中率下显著改善其他联合质量；
3. 提升不是由 validity、唯一率、scaffold 多样性大幅下降换取；
4. 完整重对接支持训练期 score-only 方向；
5. 多个 seed 结果方向一致；
6. descriptor/结构病理没有出现新的明显捷径。

### 19.3 停止或降级条件

出现以下任一情况，应停止全量 IQL actor，保留 baseline 或降级为 AWR：

- final/ordinary 概率契约无法稳定复现；
- valid 分子内部 reward 排序不可学习；
- critic 不优于节点数/validity baseline；
- Q 值持续发散或 Q1/Q2 严重分歧；
- actor 权重 ESS 无法通过温度/clipping 恢复；
- IQL 在同预算下连续多个 seed 不优于 AWR；
- 实际好分子提升完全来自 scaffold 塌缩或性质越界；
- 完整重对接否定训练期改善。

得到“轨迹上的 IQL 不优于 AWR/BC”也是有效科研结论，但必须有完整对照和失败证据。

---

## 20. 立即执行的下一步

Phase 1 的不可变 transition 与策略概率契约，以及首版几何 critic 已完成。
当前下一项任务是补齐 **Phase 0 的 reward sidecar 与审计闭环**：

1. 定义一行一个 `sample_id` 的版本化 reward sidecar schema；
2. 对重复、缺失、非有限训练 reward 和来源不一致 fail-fast；
3. 以 lazy join 方式关联 immutable shard，不回写原始轨迹；
4. 建 episode index 与固定 train/validation/test split manifest；
5. 接入正式 reward scorer，输出 good/medium/bad/invalid 分层和分布审计；
6. 再进行 Phase 2 BC/AWR 与 Phase 3 held-out critic 比较。

在 reward 完整性、split 无泄漏和 held-out baseline 对照完成之前，不启动
全量 IQL actor 训练。
