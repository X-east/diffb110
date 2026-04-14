好，我先**不动代码**，把这次重构的**完整分析、目标架构、改动范围、实施顺序、风险点、验证方案**系统讲清楚。

你这次要做的，不是普通修 bug，而是一次**训练主线所有权重构**。  
核心目标可以概括成一句话：

> **把当前“Lightning/DDP 外壳 + EDMActor 真正更新”的双层训练控制，重构成“单卡手写主循环 + EDMActor 唯一优化器所有者”的清晰架构。**

---

# 一、当前问题的本质是什么

你现在之所以想大改，是因为当前工程里其实同时存在两套“谁负责训练”的机制：

## 1. RLPF 这套
以 `RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py` 为核心：

- `EDMActor` 自己创建 optimizer
- 自己做 PPO 风格 loss
- 自己做 `zero_grad -> backward -> grad clip -> step`
- 自己拥有真正的参数更新语义

也就是说：

> **从优化器所有权角度，`EDMActor` 才是训练主体。**

---

## 2. Lightning 这套
以 `ddpo/main_diffsbdd_lightning.py` 和 `ddpo/trainer/lightning_ddpo_module.py` 为核心：

- Lightning `Trainer` 管 epoch / step / checkpoint / logger / 多卡策略
- `LightningDDPOModule.forward()` 里串起：
  - rollout
  - molecule semantics
  - filter
  - reward
  - `actor.update_policy(samples)`

问题就在这里：

> **Lightning 负责外部训练流程，但真正 optimizer.step 的不是 Lightning，而是 `actor.update_policy()`。**

这会带来 5 类结构性问题。

---

# 二、当前双控制结构的具体风险

## 风险 1：优化器所有权不清晰
现在实际是：

- Lightning 认为“我在训练”
- 但 optimizer 真正在 `EDMActor` 里
- `configure_optimizers()` 甚至只是一个 dummy optimizer 兜底

这意味着：
- checkpoint 恢复时容易漏 actor optimizer state
- 训练语义不直观
- 后续排查 loss 异常时很难判断是 Lightning loop 还是 actor 内循环的问题

---

## 风险 2：manual optimization + trainer callback 容易出现假一致
当前 Lightning 路径为了兼容 actor 自己 step，实际上已经进入一种“Lightning 只做框架壳，真实优化在外部对象里”的状态。

这种状态最容易出现：
- callback 触发时机和真实 step 不完全一致
- epoch end 统计和真实 sample/update 次序耦合
- save checkpoint 的时机被框架 hook 影响

也就是说：

> 它看起来像 Lightning 训练，但本质不是典型 Lightning 训练。

---

## 风险 3：单卡与多卡语义并不天然一致
在当前结构里，多卡是 Lightning/DDP 管的，而 `EDMActor` 本身并不是围绕 Lightning 抽象设计出来的。

所以你后续一旦切换：
- 单卡
- 单机多卡
- DDP rank 同步

很容易出现在这些边界上不一致：
- seed
- batch 聚合
- logging
- checkpoint
- molecules.csv merge
- epoch metrics

---

## 风险 4：训练主线不利于 debug
你现在真正的一步训练，散落在多个地方：

- `ddpo/main_diffsbdd_lightning.py`
- `ddpo/trainer/lightning_ddpo_module.py`
- `ddpo/runtime/rollout.py`
- `ddpo/utils/molecule_semantics.py`
- `ddpo/utils/diffsbdd_filter.py`
- `ddpo/rewards/factory.py`
- `RLPF-verl_diffusion/.../edm_actor.py`

这导致任何一个 bug 都会问同一个问题：

> “到底是哪一层在真正控制这一步？”

这是你现在最不该承受的复杂度。

---

## 风险 5：未来 DDP 改造会更痛
如果现在不先把单卡主线做干净，未来做 PyTorch DDP 时会更难，因为你是在：

- 一个已经有 Lightning/DDP 旧语义的工程里
- 再叠加自己想要的 DDP 语义

这会造成“历史包袱上再包一层分布式包袱”。

---

# 三、为什么“先做单卡主线”是正确决策

你的决定是对的，而且从工程角度非常合理。

## 原因 1：先把训练语义唯一化
你这次不是要先追求性能，而是要先追求：

- 训练主循环是谁控制
- optimizer 属于谁
- checkpoint 从哪保存
- batch/epoch 记录在哪发生

一旦这四件事统一了，后面加 DDP 才不会乱。

---

## 原因 2：当前可复用组件已经足够多
现在很多关键组件其实已经是“与 Lightning 无关的独立模块”：

### 可以直接复用的核心模块
- `ddpo/runtime/data_loader.py`
- `ddpo/runtime/rollout.py`
- `ddpo/utils/molecule_semantics.py`
- `ddpo/utils/diffsbdd_filter.py`
- `ddpo/rewards/factory.py`
- `ddpo/runtime/recording/*`
- `RLPF-verl_diffusion/.../edm_actor.py`
- `ddpo/integration/diffsbdd_model.py`

这意味着你不需要推倒重来。

你真正要做的是：

> **把这些独立组件重新放进一个“手写单卡主循环”里。**

---

## 原因 3：单卡能更早暴露真正问题
未来你要解决的问题，很多根本不是 DDP 问题，而是：
- reward 语义
- log prob 是否有限
- PPO ratio 是否稳定
- checkpoint 恢复是否一致
- molecule semantics 是否打通

这些在单卡就能暴露，而且更容易定位。

---

# 四、目标架构应该是什么样

下面是我建议的**目标架构**。

---

## 目标总原则
- 单卡作为唯一 active 主线
- Lightning 主线保留，但降级为 legacy
- 不改 reward 语义
- 不改 diffusion timestep 语义
- 不改 mask / batch 语义
- 不引入 Ray
- 未来 DDP 只包裹单卡主线，不再让框架反向控制 actor

---

## 目标主入口
新增一个明确的主入口，例如：

- `ddpo/main_diffsbdd_single_gpu.py`

这个文件将成为新的**主训练入口**。

---

## 新主入口的职责分层

### 1. 配置与环境层
负责：
- parse args
- load yaml
- set seed
- setup logging
- create run context

### 2. 模型与组件初始化层
负责初始化：
- `DiffSBDDModel`
- `DiffSBDDDataLoader`
- `DiffSBDDRollout`
- `rewarder`
- `DiffSBDDFilter`
- `EDMActor`
- `RunRecorder`

### 3. 单卡训练主循环层
按 epoch 执行：

1. `prompts = dataloader.generate_batch()`
2. `samples = rollout.generate_minibatch(prompts)`
3. `samples = annotate_batch_molecule_semantics(samples)`
4. `samples, filtering_ratio, novelty_penalty_ratio = filters.filter(samples)`
5. `samples = rewarder.calculate_rewards(samples)`
6. `samples.batch["advantages"] = samples.batch["rewards"]`（若当前仍采用 reward=advantage）
7. `metrics = actor.update_policy(samples)`
8. 统计 batch / epoch 指标
9. `run_recorder.on_batch_end(...)`
10. `run_recorder.on_epoch_end(...)`
11. save checkpoint

### 4. 显式 checkpoint 层
不依赖 Lightning callback，改为手写保存：
- model state
- actor optimizer state
- config snapshot
- epoch
- run metadata

---

# 五、单卡主线的数据流应该怎么定义

这是这次改造最关键的东西。

---

## A. 输入层：prompt / batch seed
输入来自 `DiffSBDDDataLoader`：

它输出：
- `nodesxsample`
- `group_index`
- `meta_info["max_n_nodes"]`
- `meta_info["condition"]`

这一层你应该保持不变。

### 原因
`EDMActor.update_policy()` 依赖：
- `meta_info["max_n_nodes"]`
- `meta_info["condition"]`
- `batch["nodesxsample"]`

如果这里改动不谨慎，mask 语义就会出问题。

---

## B. rollout 层：轨迹生成
来自 `DiffSBDDRollout.generate_minibatch(...)`

它负责构造：
- `x`
- `categorical`
- `latents`
- `logps`
- `timesteps`
- `nodesxsample`

这部分也建议保持不变。

### 为什么不能乱动
当前 `EDMActor` 的 loss 是建立在这个轨迹对齐基础上的：

- `latents` 实际是 \(T+2\) 长度栈
- `next_latents` 由 `latents[:, 1:]` 得到
- `logps` 是 \(T+1\)
- `timesteps` 是 \(T+1\)

如果你这里随便改 timestep 顺序或者 stack 方式，PPO 的 old/new log prob 对齐会直接错。

---

## C. molecule semantics 层
来自 `annotate_batch_molecule_semantics(...)`

这层已经很好，应该作为**唯一合法的分子重建入口**。

它统一输出：
- `obabel_reconstruct_success`
- `build_success`
- `sanitize_success`
- `rdkit_valid`
- `smiles`
- `failure_type`
- `rdmols`

这是你整个奖励语义、对接语义、记录语义的稳定边界。

这层不能拆散。

---

## D. filter 层
来自 `DiffSBDDFilter.filter(...)`

建议当前单卡主线继续沿用你现在的配置方式：
- `enable_filtering=False`
- `enable_penalty=False`

也就是：
- 不在 filter 层丢样本
- 不在 filter 层施加 reward 语义
- 只做统计或 dedup 相关准备

这样 reward 的所有权更清楚。

---

## E. reward 层
来自 `build_rewarder(...)` 返回的 rewarder。

这层只做：
- QED / SA / docking 等 reward 计算
- 保持现有字段名与语义

建议保持当前字段：
- `qed_scores`
- `sa_scores`
- `rewards`
- `druglikeness_reward`
- `docking_reward`

不要在这次大改里顺手改字段命名。

---

## F. PPO update 层
唯一主角：`EDMActor.update_policy(samples)`

这是这次架构的中心原则：

> **训练更新的唯一所有者是 `EDMActor`。**

不要再让别的训练框架在这层上方“假装拥有 optimizer”。

---

## G. 记录层
使用 `RunRecorder`：

- `on_run_started()`
- `on_batch_end(...)`
- `on_epoch_end(...)`
- `on_checkpoint_saved(...)`
- `on_run_finished(...)`

建议在单卡里固定：
- `rank=0`
- `world_size=1`

这样现有 recorder 可直接用，不必大改。

---

# 六、要改动的文件清单

下面我分成“新增”“修改”“保留但降级”三类讲。

---

## A. 新增文件

### 1. `ddpo/main_diffsbdd_single_gpu.py`
这是本次重构最核心的新文件。

职责：
- 单卡训练主入口
- 手写 epoch loop
- 手写 checkpoint save
- 调用 recorder

这是整个新架构的主干。

---

### 2. `configs/qed_sa_single_gpu.yaml` 或类似名字
建议新增一个单卡专用配置，比如：

- `configs/qed_sa_single_gpu.yaml`
- 或 `configs/diffsbdd_rlpf_single_gpu.yaml`

这个配置要明确表达：
- 单卡
- 不走 Lightning
- 不走 DDP
- 当前 reward 组合
- 当前 batch 语义

---

### 3. `scripts/run_rl_qed_sa_single_gpu.sh`
新增单卡启动脚本。

职责：
- 激活 conda env
- 设置 `CUDA_VISIBLE_DEVICES=0` 或用户可配置
- 调用新入口 `ddpo/main_diffsbdd_single_gpu.py`

这是后续日常运行的入口。

---

## B. 修改文件

### 1. `ddpo/main_diffsbdd_lightning.py`
不是删除，而是**降级为 legacy**。

建议改动：
- 文件头 docstring 标明：
  - legacy
  - 不再是主路径
  - 当前推荐使用 single-gpu 主入口
- 不建议这次大量重构它的内部逻辑

---

### 2. `ddpo/trainer/lightning_ddpo_module.py`
同样只做降级标注，不建议这次继续往里面加功能。

建议：
- 在文件头或类注释明确标记 legacy
- 告诉未来阅读者：当前 active path 不再依赖该模块

---

### 3. `ddpo/rewards/factory.py`
大概率只需要少量适配，不一定必须大改。

要检查的是：
- 新单卡入口调用方式是否直接兼容
- 是否还有 Lightning 假设
- 是否有多卡依赖

当前看它是相对独立的，可能无需大改。

---

### 4. `ddpo/runtime/recording/writers/metrics_writer.py`
这里要留意一个点：
当前 `SUMMARY_HEADER` 里没有 `checkpoint_path`，但 `update_epoch_checkpoint_path()` 在写这个字段，逻辑略不一致。

如果你单卡入口要写 checkpoint path，这里可能需要顺手修正一下 schema 一致性。

这是一个**记录层小修**，不是架构核心，但值得处理。

---

### 5. `ddpo/runtime/recording/run_recorder.py`
大概率不需要大改，只需要单卡入口正确传：
- `world_size=1`
- `rank=0`

---

## C. 保留但不作为主路径的文件

### 1. `ddpo/main_diffsbdd.py`
这个旧入口可以作为历史参考，但它里面还有 Ray/旧逻辑痕迹，不建议直接复用为主入口。

### 2. `ddpo/main_diffsbdd_lightning.py`
保留。

### 3. `ddpo/trainer/lightning_ddpo_module.py`
保留。

---

# 七、建议的实施顺序

这里给你一个真正可执行的分阶段方案。

---

## Phase 1：建立单卡主骨架
目标：把主控制权从 Lightning 拿出来

做法：
1. 新建 `ddpo/main_diffsbdd_single_gpu.py`
2. 拷贝并精简 `main_diffsbdd_lightning.py` 的初始化部分
3. 去掉所有 Lightning Trainer / callback / TensorBoardLogger 相关内容
4. 保留：
   - config
   - run context
   - logger
   - model
   - dataloader
   - rollout
   - rewarder
   - filter
   - actor
   - recorder

产出：
- 一个能初始化成功但还没完全训练的单卡入口

---

## Phase 2：迁移训练主循环
目标：把 `LightningDDPOModule.forward()` 里的核心顺序搬出来

顺序严格按现有语义：

1. rollout
2. annotate molecule semantics
3. filter
4. reward
5. if missing advantages: set to rewards
6. actor.update_policy
7. stats aggregation

产出：
- 单卡 1 step 跑通

---

## Phase 3：接入 recorder
目标：恢复当前训练输出能力

做法：
- batch 结束时构造 `BatchFacts`
- epoch 结束时构造 `EpochFacts`
- 调用 `RunRecorder`

产出：
- `molecules_rank0.csv`
- `molecules.csv`
- `metrics.json`
- `training_summary.csv`
- `epoch_status.json`

---

## Phase 4：接入 checkpoint
目标：脱离 Lightning callback 直接保存

建议保存内容：
- `model_state`
- `actor_optimizer_state`
- `config`
- `epoch`
- `run_context.run_name`
- `timestamp`

路径：
- `run_context.checkpoints_dir`

命名建议：
- `epoch=0001.ckpt`
- `last.ckpt`

产出：
- 可手动 resume 的基础 checkpoint

---

## Phase 5：配置和启动脚本
目标：让新主线真正可日常使用

新增：
- 单卡 yaml
- 单卡 bash launcher

产出：
- 用户层面可直接运行

---

## Phase 6：smoke test
目标：证明主线闭环可跑

至少测试：
1. 1 epoch 采样是否成功
2. reward 是否成功
3. `actor.update_policy()` 是否返回 finite loss
4. recorder 文件是否生成
5. checkpoint 是否生成
6. 固定 seed 下是否稳定

---

# 八、这次重构的关键风险点

## 风险 1：`EDMActor.update_policy()` 的输入结构不能破
它依赖 `DataProto` 的字段非常具体。

尤其是：
- `meta_info["max_n_nodes"]`
- `batch["latents"]`
- `batch["timesteps"]`
- `batch["logps"]`
- `batch["nodesxsample"]`

所以单卡主入口不能随便把 sample 改成 dict 或普通 tensor 集合。

---

## 风险 2：trajectory 对齐不能改
`rollout.py` 当前已经写了比较清楚的对齐说明。  
这次一定不要“顺手重构成更漂亮的形式”。

凡是牵涉：
- `latents[:, :T+1]`
- `latents[:, 1:]`
- `timesteps`
- `logps`

都要保持一致。

---

## 风险 3：reward 不能隐式改语义
这次是训练控制权改造，不是 reward redesign。  
所以不要：
- 改 QED/SA 的组合意义
- 改 failure penalty 语义
- 改 invalid molecule 处理方式
- 改 docking gate 语义

---

## 风险 4：单卡后 recorder 的 rank 逻辑要简单化但不能乱
单卡时应明确：
- `rank=0`
- `world_size=1`

不要继续保留多卡 merge 的复杂时机判断。

---

## 风险 5：resume 先不要过度设计
如果时间有限，这次先把“保存 checkpoint”打通。  
resume 可以做基础版，但不要为了 resume 把这次重构复杂化。

---

# 九、我建议的目标产物

这次完成后，理想状态应该是：

## 代码层
- 新的单卡训练入口存在
- Lightning 入口变成 legacy
- 配置和脚本明确分离 single-gpu 与 legacy-lightning

## 运行层
你可以直接用类似下面的命令启动：

```bash
bash scripts/run_rl_qed_sa_single_gpu.sh
```

## 输出层
能够在 `OUT/...` 下看到：
- `config.yaml`
- `training_summary.csv`
- `epochs/epoch_0001/metrics.json`
- `epochs/epoch_0001/molecules.csv`
- `checkpoints/epoch=0001.ckpt`
- `checkpoints/last.ckpt`

## 语义层
保持：
- reward 语义不变
- diffusion trajectory 语义不变
- mask 语义不变
- molecule semantics 不变

---

# 十、我建议这次具体改哪些文件

## 必改
- `ddpo/main_diffsbdd_single_gpu.py` 新增
- `configs/...single_gpu....yaml` 新增
- `scripts/...single_gpu....sh` 新增

## 小改
- `ddpo/main_diffsbdd_lightning.py` 标注 legacy
- `ddpo/trainer/lightning_ddpo_module.py` 标注 legacy

## 可能的小修
- `ddpo/runtime/recording/writers/metrics_writer.py`
- 只在必要时修 checkpoint_path / summary schema 这类不一致问题

## 尽量不动
- `ddpo/runtime/rollout.py`
- `ddpo/runtime/data_loader.py`
- `ddpo/utils/molecule_semantics.py`
- `ddpo/utils/diffsbdd_filter.py`
- `RLPF-verl_diffusion/.../edm_actor.py`

因为这些是你当前最核心、最敏感的语义部件。

---

# 十一、我对这次重构的建议边界

## 这次应该做的
- 建单卡主线
- 去掉 Lightning 作为 active path
- 保留现有 reward / rollout / semantics / actor 更新语义
- 保证可运行、可记录、可 checkpoint

## 这次不应该顺手做的
- 重新设计 PPO
- 重新设计 reward system
- 重新设计 actor API
- 重新设计 molecule fields
- 立即上 PyTorch DDP
- 立即删掉所有 legacy 文件

---

# 十二、最终建议

## 我的总体判断
这是一次**非常值得做**的重构，而且应该尽快做。  
因为你现在最大的复杂度不是科学问题，而是**训练控制权混乱**。

## 我建议的实际策略
先做出一个**最小但闭环的单卡主线**：

- 可生成
- 可打 reward
- 可做 PPO update
- 可写 metrics/molecules
- 可保存 checkpoint

然后再考虑未来的 PyTorch DDP。

---

如果你愿意，下一步我可以继续给你输出第二部分：

## “精确到文件级别和函数级别的改造蓝图”
也就是逐文件说明：
- 每个文件准备怎么改
- 每个函数准备新增什么
- 哪些逻辑直接搬
- 哪些逻辑不要碰

这会更适合你在真正开始动代码前做最终确认。