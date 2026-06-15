# KL 约束实现参考对比报告

日期：2026-06-04

## 结论先行

当前 DiffB110 的 PPO 主体是标准方向：保存 rollout 时的 old logprob，更新时重算 current logprob，使用 `ratio = exp(log_prob_current - log_prob_old)` 做 PPO clipped loss。

但当前额外 KL 约束不是标准 PPO 的 target KL，也不是最常见的 DDPO/RLHF 式 sampled reference logprob penalty，而是一个额外加入 loss 的 analytic Gaussian `KL(ref || current)`。这个设计可以作为“靠近预训练模型”的正则项，但和多数参考实现相比有几个重要差异：

1. 你的 KL 对象是 frozen pretrained reference，不是 PPO old rollout policy。
2. 你的 KL 方向是 `KL(ref || current)`，而常见 policy regularization 更常见 `KL(current || ref)` 或 sampled `logp_current - logp_ref`。
3. 你的 logprob 省略了 `-log(sigma)` 和高斯常数，参考项目 DePPA/RL-Diffusion 的 RL logprob 包含完整高斯 log probability。
4. 你的 logprob 按 `valid_nodes * D` 平均，但 analytic KL mask 后只除以 `valid_nodes`，存在 feature 维度尺度不一致。
5. 你目前没有记录 PPO old-policy 近似 KL、logratio 分布、ratio 分布、target-KL early stop 或 adaptive KL beta，因此“模型更新不动”不能只靠 `KL_Penalty` 判断。

所以，旧 KL sweep 里 `kl_coef * avg_kl` 很小，确实说明“记录下来的 frozen analytic KL 加权值不是主要刹车”；但这不等于 KL 子系统完全可靠。更准确的说法是：当前 KL 约束实现可运行，但不是最标准、诊断不充分，且存在尺度/公式差异，不能用它单独解释 PPO 是否被约束住。

## 你的当前实现

### 代码路径

| 模块 | 位置 | 作用 |
|---|---|---|
| PPO actor | `ddpo/runtime/actor.py` | 计算 PPO ratio、clip loss、可选 KL penalty |
| KL 函数 | `ddpo/kl/gaussian_kl.py` | analytic Gaussian KL |
| frozen reference | `ddpo/runtime/build.py` | `kl_coef > 0` 时 deep copy 当前 DiffSBDD wrapper 并冻结 |
| logprob | `ddpo/integration/diffsbdd_model.py` | 计算每一步 `p(z_s | z_t)` 的 Gaussian logprob |
| rollout | `ddpo/runtime/rollout.py` | 采样时保存 `latents/logps/timesteps`，作为 old policy 轨迹 |

### PPO 主体

`ddpo/runtime/actor.py` 中：

```python
log_prob_current, mu_new, sigma_new = current_model(**kwargs)
dif_logp = log_prob_current - log_prob_old
ratio = torch.exp(dif_logp)
ppo_loss = self.loss(advantages, self.clip_range, ratio)
```

这个是标准 PPO/DDPO 的核心：`log_prob_old` 固定来自 rollout policy，训练时只更新 current policy。

### 额外 KL

当前额外项：

```python
_, mu_ref, sigma_ref = self.ref_model(**kwargs)
kl_penalty = compute_gaussian_kl(mu_ref, sigma_ref, mu_new, sigma_new, mask=node_mask)
loss = ppo_loss + self.kl_coef * kl_penalty
```

`compute_gaussian_kl()` 实现的是：

```text
KL(ref || current)
= log(sigma_current / sigma_ref)
  + (sigma_ref^2 + (mu_ref - mu_current)^2) / (2 sigma_current^2)
  - 0.5
```

这不是 PPO clipped objective 自带的 old-policy KL，而是额外的 pretrained-reference 正则。

### 当前 logprob 尺度

`DiffSBDDModel.compute_log_p_zs_given_zt()`：

```python
log_p = -0.5 * ((zs_detached - mu) ** 2) / (sigma_safe ** 2)
log_prob = log_p_sum / (valid_count + epsilon)
```

它有两个特点：

1. 省略 `-log(sigma)` 和 `-0.5 log(2*pi)`。
2. `valid_count = node_mask.sum() * D`，即按有效原子数和 feature 维度平均。

如果 old/current 的 `sigma` 完全相同，省略常数项对 ratio 影响小；但如果训练中 `sigma` 或时间步尺度有变化，完整 logprob 更稳妥。

## 参考项目对比

### DePPA

DePPA 是最贴近你当前项目的参考，因为它同样是 DiffSBDD 条件扩散 + PPO/RL。

关键实现：

| 代码 | 观察 |
---|---|
| `reference project/DePPA/equivariant_diffusion/conditional_model.py` | `sample_p_zs_given_zt_rl()` 记录 action、mu、sigma、log_prob |
| `reference project/DePPA/lightning_modules.py` | 用 `new_log_prob - old_log_prob` 做 PPO ratio |
| `reference project/DePPA/generate_ligands.py` | 默认 `kl_coeff_pretrain=0.0` |

DePPA 的 logprob：

```python
log_prob_feat = - (action - mu_lig) ** 2 / (2 * sigma_lig ** 2) \
                - torch.log(sigma_lig) \
                - 0.5 * math.log(2 * math.pi)
log_prob_node = log_prob_feat.sum(dim=1)
log_prob_batch = scatter_add(log_prob_node, ligand_mask, dim=0)
log_prob = log_prob_batch / n_atoms
```

DePPA 的 PPO：

```python
logratio = new_log_prob - old_log_prob
ratio = torch.exp(logratio)
unclipped = -advantages * ratio
clipped = -advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
loss_i = torch.mean(torch.maximum(unclipped, clipped))
approx_kl_vals.append(0.5 * torch.mean(logratio ** 2))
clipfrac_vals.append(mean(abs(ratio - 1) > clip_range))
```

可选 pretrained KL：

```python
logratio_ref_theta = pretrained_log_prob - new_log_prob
ratio_ref_theta = torch.exp(logratio_ref_theta)
approx_kl = ratio_ref_theta - 1 - logratio_ref_theta
kl_loss = approx_kl.mean()
total_loss += kl_coeff_pretrain * kl_loss
```

与当前 DiffB110 的差异：

| 项目 | DePPA | DiffB110 当前 |
|---|---|---|
| PPO ratio | 有 | 有 |
| logprob | 完整 Gaussian logprob，含 `-log sigma` 和常数 | 只保留平方项 |
| old-policy approx KL | 明确记录 `0.5 * logratio^2` | 没有 |
| clipfrac | 明确记录 | 有均值，但缺少 per-timestep/ratio 细节 |
| pretrained KL | 可选，默认 0 | `kl_coef > 0` 时 analytic `KL(ref||current)` |
| reward 稳定化 | rank/percentile-to-normal、invalid penalty、connectivity、diversity | 当前已有 reward normalizer，但化学 guard 还不完整 |

DePPA 的启发：不是单靠 KL 调好，而是靠 reward 后处理、invalid penalty、valid/connectivity 过滤、rank normalization、approx KL/clipfrac 诊断共同稳定。

### RL-Diffusion-main

`reference project/RL-Diffusion-main/scripts/RL-guided.py` 更接近标准 DDPO：

```python
old_generative_model = copy.deepcopy(generative_model)
latents, next_latents, old_log_probs = old_generative_model.get_trajectory(...)
new_log_prob = generative_model.get_new_log_prob(...)
ratio = torch.exp(new_log_prob - old_log_prob)
surr1 = ratio * advantages
surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
```

它还有 transition-level Gaussian KL：

```python
kl_penalty = 0.5 * (
    torch.log(new_sigma / old_sigma)
    + (old_sigma**2 + (old_mu - new_mu)**2) / (2 * new_sigma**2)
    - 0.5
).mean()
step_loss = -torch.min(surr1, surr2).mean() + kl_coef * kl_penalty
```

与当前 DiffB110 的差异：

| 项目 | RL-Diffusion | DiffB110 当前 |
|---|---|---|
| KL 参照 | old rollout model 的 mu/sigma | frozen pretrained ref model |
| PPO old model | 每 episode deep copy old model | rollout logprob 固定，ref 另建 |
| logprob | 完整 Gaussian logprob | 省略 `-log sigma` 和常数 |
| KL 默认 | `kl_coef=0.00` | 当前 launch 默认也是 0，旧 sweep 手动开 |
| entropy | 有 entropy 项接口 | 当前没有 |

它支持“transition analytic KL”这个思路，但它约束的是 old policy 与 current policy，而不是固定预训练模型。

### DiffMeta-main

DiffMeta 的 RL 路线不是同一种 PPO-KL。

关键观察：

1. `train_step_gdpo()` / `train_step_olppo()` 采样轨迹和 reward。
2. reward 被标准化、clip 到 `[-5, 5]` 后作为 `advantages`。
3. 训练时对 atom/edge prediction 做 reward-weighted CE/NLL：

```python
loss_X, loss_E = self.ppo_loss(..., reweight=advantages)
loss = loss_X.mean() + lambda * loss_E.mean()
```

DiffMeta 里的 KL 多数是离散扩散训练/验证 KL：

```python
kl_prior(X, E, node_mask)
val_X_kl
val_E_kl
```

这些是 `q(z_T|x)` vs prior、posterior true/pred atom/edge 分布 KL，不是 PPO trust-region KL。

结论：DiffMeta 可以借鉴 reward-weighted denoising、atom/edge 离散扩散评价，但不能直接拿它当“标准 PPO KL”参考。

### RLPF-verl_diffusion

本地 `reference project/RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py` 和当前 `ddpo/runtime/actor.py` 高度相似，而且文件里出现了 `from ddpo.runtime.logging_utils import log_cuda_memory`、`from ddpo.kl import compute_gaussian_kl` 这类本项目命名空间。

因此它不能当作独立第三方参考，只能说明当前 DiffB110 的 actor 很可能来自 RLPF 本地化/改写链路。

### REINVENT / SMILES-RL

这些不是扩散模型，但能说明药物生成 RL 里常见的 prior anchoring 思路。

REINVENT/SMILES-RL 不是 PPO KL，而是：

```text
augmented_likelihood = prior_likelihood + sigma * score
loss = (augmented_likelihood - agent_likelihood)^2
```

含义：让 agent 追求高 reward，但仍然靠 prior likelihood 限制远离化学先验。它对你有概念参考价值，但不能直接替代 diffusion trajectory KL。

## 标准 KL 约束应该怎么理解

“标准”其实有三种常见层次，不能混在一起：

### 1. 标准 PPO old-policy trust region

这是 PPO/DDPO 的主约束：

```text
r_t(theta) = exp(log pi_theta(a_t|s_t) - log pi_old(a_t|s_t))
L_clip = -E[min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t)]
```

常配套诊断：

```text
logratio = logp_new - logp_old
approx_kl ~= 0.5 * mean(logratio^2)
clipfrac = mean(abs(ratio - 1) > clip_range)
```

有些实现还会加：

```text
if approx_kl > target_kl:
    stop PPO inner epochs early
```

这个 KL 约束的对象是 current policy 和 old rollout policy，作用是防止同一批样本被 PPO 多轮复用时更新太猛。

### 2. Frozen reference / pretrained policy regularization

这是另一类约束，目的是防止 RL 后模型离预训练生成分布太远。

常见形式：

```text
reward = task_reward - beta * (logp_current - logp_ref)
```

或：

```text
loss = ppo_loss + beta * KL(pi_current || pi_ref)
```

也可以用 analytic Gaussian KL，但要确保方向、采样分布、归一化、时间步权重一致。

### 3. Diffusion 训练里的 prior/posterior KL

比如 DiffMeta 的 `kl_prior`、`val_X_kl`、`val_E_kl`，它们用于扩散模型 likelihood/validation，不是 RL policy update 的 KL 约束。

## 当前实现和标准做法的主要不一致

| 维度 | 标准 PPO/DDPO 常见做法 | 当前 DiffB110 | 影响 |
|---|---|---|---|
| PPO ratio | `exp(new_logp - old_logp)` | 有 | 主体正确 |
| old-policy approx KL | 记录 `0.5 logratio^2` 或 `old_logp-new_logp` | 没有 | 难判断 PPO 是否真的动了或被 clip 卡住 |
| target KL early stop | 可选常见 | 没有 | 多 PPO epoch 时风险更高 |
| pretrained reference | 可选，通常 sampled logprob penalty 或 `KL(current||ref)` | analytic `KL(ref||current)` | 方向与解释不同 |
| KL 归一化 | 与 logprob 同尺度 | logprob 除 `nodes*D`，KL 除 `nodes` | KL 数值尺度不一致 |
| logprob 公式 | 完整 Gaussian logprob | 省略 `-log sigma` 和常数 | 若 sigma 变化，ratio 可能偏 |
| KL 诊断 | mean/p95/max/per timestep/weighted beta | 只有均值 `KL_Penalty` | 无法定位 spike |
| reward outlier | 需要 clamp/robust normalize | 旧实验 docking 有极端 outlier | 比 KL 更可能导致不稳定或假信号 |

## 对“KL 会不会导致模型不更新”的判断

基于已有旧实验诊断，`kl_coef * avg_kl` 最大约 `0.00263`，远小于 PPO/reward 主项，因此旧 run 里“模型不动”大概率不是 frozen analytic KL 数值本身直接压死。

更可疑的是：

1. reward 稀疏：有效高奖励分子少，advantage 信号弱。
2. reward/outlier：旧 `kl-0p05` epoch 86 出现 docking 极端值，说明 docking reward 没被充分 clamp。
3. 描述子漂移：高碳/低杂原子会让 reward 找到捷径后进入平台。
4. PPO 诊断缺失：没有 old-policy approx KL、ratio 分布、clipfrac per timestep，就无法确认是 ratio 卡住、梯度太小，还是 reward 没信号。
5. logprob 尺度风险：当前 logprob 与 KL 的归一化、公式不完全一致，影响对 KL/ratio 的解释。

## 建议下一步

### P0：先补诊断，不先盲调 kl_coef

需要新增但不改变训练行为的记录：

| 指标 | 含义 |
|---|---|
| `ppo_logratio_mean/std/p95/max` | current 与 old rollout 的 logprob 差 |
| `ppo_approx_kl = 0.5 * logratio^2` | PPO old-policy drift |
| `ratio_mean/std/p95/max` | importance sampling ratio 是否爆或塌 |
| `clipfrac_per_timestep` | 哪些扩散步被 clip |
| `advantage_mean/std/nonzero_ratio` | reward 是否真的产生学习信号 |
| `weighted_ref_kl = kl_coef * kl_penalty` | 真实进入 loss 的 KL 尺度 |

### P0：修正 KL 尺度

如果继续保留 analytic KL，mask 平均建议改为与 logprob 同尺度：

```python
denom = (mask.expand_as(kl_elementwise)).sum()
kl = (kl_elementwise * mask).sum() / (denom + eps)
```

或显式：

```python
kl = (kl_elementwise * mask).sum() / (mask.sum() * D + eps)
```

### P1：补完整 logprob 对照实验

保留旧 logprob 路径不动，增加一个实验开关：

```text
logprob_mode:
  simplified_quadratic
  full_gaussian
```

用同一批 rollout 对比：

| 对比 | 判断 |
|---|---|
| ratio 分布是否变化 | 公式差异是否影响 PPO |
| approx_kl 是否变化 | 是否旧公式低估/高估 policy drift |
| reward 是否更快动 | 是否原 logprob 信号太弱 |

### P1：把 frozen reference KL 改成可解释的两套模式

建议拆成两个可选模式：

| 模式 | 公式 | 用途 |
|---|---|---|
| `old_policy_approx_kl` | `0.5 * (new_logp-old_logp)^2` | PPO 更新步长诊断/early stop |
| `pretrained_logprob_penalty` | `new_logp - ref_logp` 或 Schulman approx KL | 防止偏离预训练生成分布 |
| `analytic_transition_kl` | Gaussian KL between transition distributions | 研究用，需双向/尺度校准 |

不要只叫一个 `kl_coef`，否则很难知道约束的是 old policy、pretrained policy，还是扩散 prior。

### P1：先使用 target KL early stop，而不是大系数 KL loss

推荐先加：

```text
target_kl: 0.01 或 0.02 起步
如果 ppo_approx_kl > 1.5 * target_kl:
    停止当前 PPO inner epoch 后续更新
```

这比直接提高 `kl_coef` 更不容易把梯度压死。

### P1：reward 侧必须同时处理

KL 只能限制“不要走太远”，不能告诉模型“往哪里走”。你的问题里 reward 稀疏和高碳漂移更关键，所以应同时做：

1. docking/classifier reward clamp。
2. invalid/碎片/三元环/bad bond length/异常元素分布 finite penalty。
3. carbon fraction、hetero fraction、aromatic ring、halogen placement guard。
4. top-N 好分子池只收通过 guard 的分子。

## 总判断

当前 DiffB110 的 KL 实现不是完全离谱，它有可解释的 analytic Gaussian reference regularization 形式；但它不是最标准 PPO KL，并且和 DePPA/RL-Diffusion 这类更接近的参考实现相比，缺少完整 logprob、old-policy approx KL、target KL、KL 尺度一致性和 per-timestep 诊断。

如果现在要继续强化学习线，最稳妥的路线不是先调大/调小 `kl_coef`，而是先让 KL/ratio/advantage 变得可诊断：

```text
先补 old-policy approx KL + ratio/clipfrac 详细记录
再修 analytic KL 归一化
再做 full Gaussian logprob ablation
最后再决定 pretrained KL 是 sampled logprob penalty 还是 analytic transition KL
```

这样才能回答“模型为什么不动”：是 reward 没信号、ratio 被 clip、logprob 尺度太小，还是 KL 真在刹车。

## 参考来源

本地代码：

- `ddpo/runtime/actor.py`
- `ddpo/kl/gaussian_kl.py`
- `ddpo/integration/diffsbdd_model.py`
- `reference project/DePPA/lightning_modules.py`
- `reference project/DePPA/equivariant_diffusion/conditional_model.py`
- `reference project/RL-Diffusion-main/scripts/RL-guided.py`
- `reference project/DiffMeta-main/model/diffusion_discrete.py`
- `reference project/REINVENT/train_agent.py`
- `reference project/SMILES-RL/smiles_rl/agent/regularized_mle.py`

外部参考：

- PPO paper: <https://arxiv.org/abs/1707.06347>
- DDPO paper: <https://arxiv.org/abs/2305.13301>
- DDPO PyTorch reference implementation: <https://github.com/kvablack/ddpo-pytorch>
- Schulman approximate KL note used by DePPA comment: <http://joschu.net/blog/kl-approx.html>
