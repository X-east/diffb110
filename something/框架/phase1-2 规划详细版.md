好，下面我给你一版**精确到文件级别、函数级别、调用顺序级别**的改造蓝图。  
我会尽量把“**现状是什么**、**为什么这样改**、**每个文件具体动哪里**、**哪些函数保持不动**、**新入口怎么拼接**、**未来 DDP 怎么接回来**”全部讲透。

---

# 一、先给最终目标一个最准确的定义

你这次不是简单地“去掉 DDP”，而是要把训练架构重心从：

- `Lightning Trainer` 管训练生命周期
- `LightningDDPOModule.forward()` 串业务逻辑
- `EDMActor.update_policy()` 在模块内部偷偷做真正优化

改成：

- **一个手写单卡训练主循环** 直接控制整个训练过程
- `EDMActor` 成为**唯一明确的优化器与 PPO 更新所有者**
- `DiffSBDD` 只负责采样与 log-prob 相关的模型接口
- `reward / filter / molecule semantics / recorder` 作为独立部件插入主循环
- `Lightning` 路径保留但降级为 legacy

一句话：

> **把“训练控制权”从 Lightning 手里拿出来，交还给你自己的主入口。**

---

# 二、现有代码里的真实控制流：逐文件拆解

下面先把当前 active path 按真实运行顺序拆一遍。

---

## 1. 入口文件：`ddpo/main_diffsbdd_lightning.py`

这个文件当前负责：

### 入口职责
- 解析参数：`parse_args()`
- 读取配置：`load_config()`
- 设置 seed
- 创建 `RunContext`
- 初始化 logger
- 初始化：
  - `DiffSBDDModel`
  - `DiffSBDDDataLoader`
  - `DiffSBDDRollout`
  - `rewarder`
  - `DiffSBDDFilter`
  - `EDMActor`
  - `RunRecorder`
- 再把上面所有东西交给 `LightningDDPOModule`
- 然后交给 `pl.Trainer.fit(...)`

当前后半段很关键：

```220:424:ddpo/main_diffsbdd_lightning.py
    lightning_module = LightningDDPOModule(
        model=model,
        rollout=rollout,
        rewarder=rewarder,
        filters=filters,
        actor=actor,
        config=config,
        run_recorder=run_recorder,
    )
    ...
    trainer = pl.Trainer(
        ...
    )
    ...
    trainer.fit(lightning_module, dataloader)
```

### 结论
这个文件当前并不真正“训练”，它只是：
- 初始化组件
- 把控制权交给 Lightning

所以它未来的命运应该是：

## 未来定位
- **保留**
- 但**降级为 legacy 入口**
- 只用于参考，不再作为日常主入口

---

## 2. 训练壳文件：`ddpo/trainer/lightning_ddpo_module.py`

这是当前最关键的中间层。

### 它现在做了什么
`LightningDDPOModule.forward()` 实际上做了完整一步业务训练链：

```112:164:ddpo/trainer/lightning_ddpo_module.py
    def forward(self, batch):
        ...
        samples = self.rollout.generate_minibatch(batch)
        samples = annotate_batch_molecule_semantics(samples)
        samples, filtering_ratio, novelty_penalty_ratio = self.filters.filter(samples)

        try:
            samples = self.rewarder.calculate_rewards(samples)
        ...
        if "advantages" not in samples.batch:
            ...
            samples.batch["advantages"] = samples.batch["rewards"]

        metrics = self.actor.update_policy(samples)
```

也就是说它当前做的是：

1. rollout
2. molecule semantics
3. filter
4. reward
5. advantage fallback
6. actor PPO update

这其实已经是你未来**单卡主循环的原型**了。

---

### 它为什么不应该继续是主路径
因为它所在的类是 `pl.LightningModule`，这会带来一些伪装成“框架能力”的额外复杂度：

#### 复杂点 1：训练真正更新不归 Lightning
```33:53:ddpo/trainer/lightning_ddpo_module.py
        self.actor = actor
        ...
        self.automatic_optimization = False
```

```333:337:ddpo/trainer/lightning_ddpo_module.py
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.001)
```

这里 dummy optimizer 已经说明问题了：

> Lightning 并不是 optimizer 所有者，它只是为了满足框架要求挂了一个假的。

#### 复杂点 2：checkpoint / epoch end / logger 都在绕框架
例如：
- `on_train_epoch_end()`
- `on_save_checkpoint()`
- `on_load_checkpoint()`

这些都在做“框架外真实逻辑”与“框架内生命周期”的对接。

这就是你要拆掉的东西。

---

### 未来定位
- **保留**
- 明确标记为 legacy
- 不再承接 active 训练主线

---

## 3. 采样层：`ddpo/runtime/rollout.py`

这是非常稳定、应该尽量原封不动复用的模块。

### 核心函数
`DiffSBDDRollout.generate_minibatch(...)`

```13:56:ddpo/runtime/rollout.py
    def generate_minibatch(self, prompts: DataProto) -> DataProto:
        ...
        x, h, latents, logps, timesteps, mus, sigmas = self.model.sample(...)
        ...
        latents_stacked = torch.stack(latents, dim=1)
        logps_stacked = torch.stack(logps, dim=1)
        ...
        return DataProto(...)
```

### 它的关键意义
它定义了 `EDMActor.update_policy()` 期望的轨迹结构：

- `latents`
- `logps`
- `timesteps`
- `nodesxsample`
- `group_index`

### 这部分为什么危险
因为 `EDMActor` 默认假定这里的轨迹对齐是正确的。  
如果你未来随便动这部分，很容易破坏：

- old log prob / new log prob 对齐
- `latents[:, :T+1]` 和 `latents[:, 1:]` 的对应关系
- timestep 顺序

### 未来处理方式
- **不重写**
- **直接复用**
- 如果要改，只允许加注释，不改语义

---

## 4. prompt / batch 输入层：`ddpo/runtime/data_loader.py`

这个模块也相对独立。

### 核心函数
- `__init__`
- `generate_batch()`

```36:60:ddpo/runtime/data_loader.py
    def generate_batch(self):
        nodesxsample = torch.randint(...)
        group_index = torch.arange(...).repeat_interleave(...)
        nodesxsample = nodesxsample.repeat_interleave(...)
        ...
        return DataProto(...)
```

### 它提供了什么
为 rollout 提供：
- `nodesxsample`
- `group_index`
- `meta_info["max_n_nodes"]`
- `meta_info["condition"]`

### 未来处理方式
- **继续复用**
- 只在配置层改变 batch_size / sample_group_size / each_prompt_sample
- 不改 batch 语义

---

## 5. 模型适配层：`ddpo/integration/diffsbdd_model.py`

这个文件是 DiffSBDD 与 RLPF actor 之间的最核心桥接层。

### 它的作用
- 包装 DiffSBDD checkpoint
- 提供 actor 期望的模型接口
- 定义 `get_mask(...)`
- 定义 sample/log-prob 所需的桥接行为

### 当前最关键的方法
#### `get_mask(...)`
```212:261:ddpo/integration/diffsbdd_model.py
    def get_mask(
        self, 
        nodesxsample: torch.Tensor, 
        batch_size: int, 
        max_n_nodes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
```

这个方法直接服务于 `EDMActor.calculate_loss()`。

#### 风险
如果你未来改 batch 组织方式而不保持：
- `nodesxsample`
- `batch_size`
- `max_n_nodes`

一致，mask 语义会直接坏。

### 未来处理方式
- **完全保留**
- 不在这次重构中重构模型桥接层
- 只复用

---

## 6. 分子语义层：`ddpo/utils/molecule_semantics.py`

这是当前项目里**科学语义最重要**的模块之一。

### 核心函数
- `annotate_batch_molecule_semantics(...)`
- `compute_sample_molecule_semantics(...)`
- `build_canonical_rdmol_from_sample(...)`

其中最关键的是：

```194:246:ddpo/utils/molecule_semantics.py
def annotate_batch_molecule_semantics(
    data: Any,
    dataset_info: Optional[dict] = None,
    include_legacy_valid_alias: bool = True,
) -> Any:
```

### 它做了什么
对每个 sample：
1. 从 `x/categorical/nodesxsample` 提取真实有效节点
2. OpenBabel 重建结构
3. RDKit sanitize
4. 生成 `rdmols`
5. 生成 `smiles`
6. 生成 `rdkit_valid`
7. 生成 `failure_type`

### 这层的重要性
它是当前你项目中“**最终统一 molecule 对象**”的来源。

后续：
- filter
- reward
- docking
- logging

都依赖它。

### 未来处理方式
- **完全保留**
- 单卡新主线必须直接调用它
- 不允许再出现另一条“自己重建分子”的分支

---

## 7. 过滤层：`ddpo/utils/diffsbdd_filter.py`

当前它已经基本被改造成“基于 canonical semantics 的 filter”。

### 核心函数
`DiffSBDDFilter.filter(...)`

```70:124:ddpo/utils/diffsbdd_filter.py
    def filter(self, data: "DataProto") -> tuple["DataProto", float, float]:
```

### 当前行为
- 如果没 `rdkit_valid` / `smiles`，会先调用 `annotate_batch_molecule_semantics()`
- 根据 `enable_filtering` 决定是否真的去重/删样本
- 根据 `enable_penalty` 决定是否在 filter 层加 novelty penalty

### 你现在的主线配置
实际使用时，你现在偏向：

- `enable_filtering=False`
- `enable_penalty=False`

这意味着：
- filter 不删除样本
- 只做统计和字段整理

### 未来处理方式
- 这次继续保持这个策略
- 不要在重构时把 filter 重新变成 reward 层或样本丢弃层

---

## 8. 奖励工厂：`ddpo/rewards/factory.py`

这是新主线完全可以直接复用的部分。

### 核心函数
`build_rewarder(config, dataset_info, logger)`

```31:110:ddpo/rewards/factory.py
def build_rewarder(config: dict, dataset_info: dict, logger: logging.Logger):
```

### 当前职责
- 根据 config 决定启用哪些 reward component
- 构造：
  - `QEDReward`
  - `SAReward`
  - `DockingReward`
  - legacy `DruglikenessReward`
  - 可选 `ConnectivityReward`
- 最后套一层 `StabilityAwareReward`

### 未来处理方式
- **直接复用**
- 新入口里继续调用
- 只在 config 层决定启用哪些 reward

---

## 9. PPO / optimizer 层：`RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`

这是你当前真正的训练主体。

### 最关键的方法
#### `__init__`
```9:31:RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py
        self.optimizer = optimizer_cls(
            self.model.parameters(),
            lr=self.config["train"]["learning_rate"],
            ...
        )
```

这说明：
- optimizer 就在 actor 里
- 参数更新所有权就在 actor 里

#### `update_policy(...)`
```158:208:RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py
    def update_policy(self, data: DataProto) -> Dict:
```

它负责：
- 从 `DataProto` 中切出训练需要的字段
- 组织 micro-batch
- 调用 `train_batched_samples()`

#### `train_batched_samples(...)`
```31:91:RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py
    def train_batched_samples(self, batched_samples):
```

它真的做：
- zero_grad
- loss.backward
- grad finite 检查
- clip_grad_norm_
- optimizer.step

#### `calculate_loss(...)`
```93:146:RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py
    def calculate_loss(...)
```

它定义了：
- old log prob
- new log prob
- ratio
- clip range
- PPO clipped loss

### 这部分的结论
> 这里就是训练更新语义的唯一中心。

### 未来处理方式
- **完全保留**
- 单卡主线直接调用 `actor.update_policy(samples)`
- 不再让 Lightning 包一层壳

---

## 10. 记录层：`ddpo/runtime/recording/*`

### 核心 facade
`ddpo/runtime/recording/run_recorder.py`

```20:129:ddpo/runtime/recording/run_recorder.py
class RunRecorder:
```

### 它已经很适合单卡
因为它本质上是事件接口：
- `on_run_started()`
- `on_batch_end(...)`
- `on_epoch_end(...)`
- `on_checkpoint_saved(...)`
- `on_run_finished(...)`

### 相关 schema
`ddpo/runtime/recording/schemas.py`

其中关键是：
- `BatchFacts`
- `EpochFacts`
- `CheckpointFacts`
- `RunLifecycleFacts`

### 未来处理方式
- **单卡继续复用**
- 固定：
  - `rank = 0`
  - `world_size = 1`

### 这里的小问题
`metrics_writer.py` 中 `update_epoch_checkpoint_path()` 似乎在更新 CSV 中一个并未出现在 `SUMMARY_HEADER` 里的字段，这属于一个记录层的小不一致，未来可能需要顺手修一下。

---

# 三、未来目标架构：精确到调用关系

下面给你一个明确的**新架构调用图**。

---

## 新入口文件
建议新增：

- `ddpo/main_diffsbdd_single_gpu.py`

---

## 新入口中的模块初始化顺序

### Step 1：基础配置
函数建议：
- `parse_args()`
- `load_config()`

职责：
- 读取 yaml
- 支持 `--config_path`
- 支持 `--save_path`
- 支持 `--seed`
- 支持 `--log_level`
- 可选支持 `--resume`
- 可选支持 `--checkpoint_path`

这里可以几乎照抄 `main_diffsbdd_lightning.py` 的参数接口，保持兼容。

---

### Step 2：环境与输出目录
建议复用：

- `set_seed(...)`
- `create_run_context(...)`
- `setup_logging(...)`

职责：
- 设随机种子
- 生成本次输出目录
- 保存 config snapshot
- 初始化 logger

---

### Step 3：组件初始化
按顺序初始化：

1. `DiffSBDDModel`
2. `dataset_info = dataset_params["crossdock"]`
3. `DiffSBDDDataLoader`
4. `DiffSBDDRollout`
5. `rewarder = build_rewarder(...)`
6. `filters = DiffSBDDFilter(...)`
7. `actor = EDMActor(model, config)`
8. `run_recorder = RunRecorder(...)`

这里顺序基本可以沿用 `main_diffsbdd_lightning.py`。

---

## 新的单卡训练主循环

### 每个 epoch 的固定顺序

#### 1. 生成 prompts
```python
prompts = dataloader.generate_batch()
```

#### 2. rollout 生成轨迹
```python
samples = rollout.generate_minibatch(prompts)
```

#### 3. 注入 canonical molecule semantics
```python
samples = annotate_batch_molecule_semantics(samples)
```

#### 4. 过滤/去重/novelty 统计
```python
samples, filtering_ratio, novelty_penalty_ratio = filters.filter(samples)
```

#### 5. 计算 reward
```python
samples = rewarder.calculate_rewards(samples)
```

#### 6. 补 advantage
```python
if "advantages" not in samples.batch:
    samples.batch["advantages"] = samples.batch["rewards"]
```

#### 7. PPO 更新
```python
metrics = actor.update_policy(samples)
```

#### 8. 计算 batch statistics
包括：
- avg_qed
- avg_sa
- avg_reward
- valid_count
- generated_count

#### 9. 写 batch 记录
构造 `BatchFacts`，调用：
```python
run_recorder.on_batch_end(batch_facts)
```

#### 10. 写 epoch 记录
构造 `EpochFacts`，调用：
```python
run_recorder.on_epoch_end(epoch_facts)
```

#### 11. 保存 checkpoint
自己手写 `save_checkpoint(...)`
然后调用：
```python
run_recorder.on_checkpoint_saved(checkpoint_facts)
```

---

# 四、要新增/拆出的函数蓝图

下面说得更细一点：**单卡入口里建议有哪些函数**。

---

## 文件：`ddpo/main_diffsbdd_single_gpu.py`

建议包含这些函数。

---

## 1. `parse_args()`
### 作用
定义 CLI 参数，保持和当前 Lightning 入口尽可能接近。

### 参数建议
- `--config_path`
- `--save_path`
- `--seed`
- `--log_level`
- `--resume`
- `--checkpoint_path`

### 原因
这样你已有脚本和使用习惯最容易迁移。

---

## 2. `load_config(config_path: str) -> dict`
### 作用
读取 yaml 配置。

### 是否新写
可直接从 Lightning 入口复制。

---

## 3. `build_runtime_components(config, args, logger, run_context)`
### 作用
把组件初始化统一收拢，避免 `main()` 太长。

### 返回建议
返回一个 dict 或 dataclass，包含：
- `model`
- `dataset_info`
- `dataloader`
- `rollout`
- `rewarder`
- `filters`
- `actor`
- `run_recorder`

### 好处
主循环更清晰，未来 DDP 迁移也更容易复用这个构造器。

---

## 4. `compute_batch_stats(samples) -> dict`
### 作用
从 `samples.batch` 里提取：
- `qed_scores`
- `sa_scores`
- `rewards`
- `rdkit_valid`

计算：
- `avg_qed`
- `avg_sa`
- `avg_reward`
- `generated_count`
- `valid_count`
- `valid_ratio`

### 来源
可参考 `LightningDDPOModule._compute_batch_stats(...)`

```64:99:ddpo/trainer/lightning_ddpo_module.py
    def _compute_batch_stats(self, samples) -> Dict[str, float]:
```

### 注意
单卡版本不需要分布式 reduce，直接 local 即 global。

---

## 5. `build_batch_facts(samples, epoch, batch_idx, filtering_ratio, novelty_penalty_ratio, batch_stats) -> BatchFacts`
### 作用
把 sample-level 字段整理成 recorder 需要的 `BatchFacts`。

### 来源
可以参考当前 `LightningDDPOModule.forward()` 中写 `molecule_rows` 的那段逻辑：

```134:197:ddpo/trainer/lightning_ddpo_module.py
            molecule_rows = []
            for mol_id in range(n):
                ...
                molecule_rows.append({...})
            facts = BatchFacts(...)
            self.run_recorder.on_batch_end(facts)
```

### 注意
单卡固定：
- `rank = 0`

---

## 6. `build_epoch_facts(...) -> EpochFacts`
### 作用
把当前 epoch 的统计整理为：
- avg_loss
- reward_mean
- avg_qed
- avg_sa
- generated_total
- rdkit_valid
- rdkit_valid_ratio
- filtering_ratio_mean
- novelty_penalty_ratio_mean
- extra_metrics

### 来源
参考 `LightningDDPOModule.on_train_epoch_end()` 中 `EpochFacts(...)` 的构造方式。

---

## 7. `save_training_checkpoint(...) -> CheckpointFacts`
### 作用
自己保存 checkpoint，并返回 recorder 需要的事实对象。

### 应保存内容
建议包括：
- `model_state`
- `actor_optimizer_state`
- `config`
- `epoch`
- `global_step`（可选）
- `timestamp`
- `run_name`

### 路径建议
- `run_context.checkpoints_dir / f"epoch={epoch:04d}.ckpt"`
- 同步更新 `last.ckpt`

### 参考来源
当前 Lightning 的 checkpoint 中真实重要内容是：

```351:373:ddpo/trainer/lightning_ddpo_module.py
    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_state"] = self.model.state_dict()
        checkpoint["config"] = self.config
        ...
        checkpoint["actor_optimizer_state"] = self.actor.optimizer.state_dict()
```

单卡版本质上就是自己手写这个结构。

---

## 8. `run_single_gpu_training(...)`
### 作用
真正的训练主循环函数。

### 它应该做的事情
- 调用 `run_recorder.on_run_started()`
- 按 epoch 训练
- try/except 捕获中断和失败
- 在结尾调用 `run_recorder.on_run_finished(...)`

---

## 9. `main()`
### 作用
只负责：
- parse args
- load config
- setup env/logger/run_context
- build components
- run training

---

# 五、旧文件的改造蓝图：逐文件说明

下面精确到每个文件应该怎么动。

---

## 文件 1：`ddpo/main_diffsbdd_lightning.py`

## 建议改动
### 改动 A：文件头 docstring 改为 legacy 说明
当前开头写的是 active Lightning 入口：

```1:13:ddpo/main_diffsbdd_lightning.py
"""
Main entry point for DiffSBDD-RLPF DDPO training with PyTorch Lightning.
...
"""
```

建议改成类似：

- This Lightning entrypoint is legacy
- Active workflow has moved to single-gpu mainline
- Kept for reference / backward compatibility only

### 改动 B：`argparse` help 文案也可加 legacy 提示
不是必须，但建议。

### 改动 C：不要在这次重构里继续扩展其逻辑
这个文件应该停止继续演化，避免两条主线同时成长。

---

## 文件 2：`ddpo/trainer/lightning_ddpo_module.py`

## 建议改动
### 改动 A：类 docstring 标明 legacy
当前类说明还是 active path 语气：

```14:18:ddpo/trainer/lightning_ddpo_module.py
class LightningDDPOModule(pl.LightningModule):
    """PyTorch Lightning wrapper for DDPO training loop."""
```

建议改成：
- legacy wrapper
- no longer active primary workflow

### 改动 B：不再新增功能
这次不要继续往这里塞新单卡逻辑。

### 改动 C：保留其作为“旧主循环参考模板”
因为你新单卡入口最重要的逻辑顺序，其实就是从这里抽出来。

---

## 文件 3：`ddpo/runtime/rollout.py`

## 建议改动
### 原则
- 只读
- 不改语义
- 只必要时补注释

### 原因
这是 actor 所依赖的轨迹协议，不应在训练控制权改造时被触碰。

---

## 文件 4：`ddpo/runtime/data_loader.py`

## 建议改动
### 原则
- 不改 batch 语义
- 继续返回 `DataProto`

### 可选小改
如果后续要支持 `num_batches > 1` 的 epoch 内多 batch，可以再考虑调整 `__len__/__next__`，但这次先不要碰。

当前它是一 epoch 一 batch 的语义，这和现有逻辑匹配。

---

## 文件 5：`ddpo/rewards/factory.py`

## 建议改动
### 原则
- 新入口继续直接调用
- 不改 reward 结构

### 可选小改
若要进一步去除 legacy 复杂度，可后续再处理，但不是这次主任务。

---

## 文件 6：`ddpo/runtime/recording/run_recorder.py`

## 建议改动
### 原则
- 直接复用
- 单卡入口设置 `world_size=1`

### 不建议做的事
- 不要为了单卡去重写 recorder
- 不要破坏未来 world_size > 1 的扩展点

---

## 文件 7：`ddpo/runtime/recording/writers/metrics_writer.py`

## 建议改动
### 重点检查
这里存在一个小设计不一致：

`SUMMARY_HEADER` 里没有 `checkpoint_path`：

```22:28:ddpo/runtime/recording/writers/metrics_writer.py
    SUMMARY_HEADER = [
        "epoch", "timestamp", "avg_loss",
        ...
        "failure_counts", "checkpoint_due",
    ]
```

但又有：

```72:79:ddpo/runtime/recording/writers/metrics_writer.py
    def update_epoch_checkpoint_path(self, epoch: int, checkpoint_path: str) -> None:
```

### 建议处理
如果你希望 summary CSV 真记录 checkpoint path：
- 就把 `checkpoint_path` 显式加入 header

否则：
- 删掉这个更新逻辑

### 我的建议
加到 header 里比较合理。  
但这是**记录层一致性修复**，不是核心训练重构。

---

## 文件 8：`RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`

## 建议改动
### 原则
- 这次尽量不改
- 除非 smoke test 暴露单卡路径专属 bug

### 原因
这是 PPO/optimizer 语义核心，随便改最危险。

---

# 六、新增文件的精确蓝图

下面说新增文件应该长什么样。

---

## 文件 A：`ddpo/main_diffsbdd_single_gpu.py`

## 目标
成为新的 active 主入口。

## 代码结构建议
### 顶部
- imports
- path setup
- warning/log suppression
- parse args / load config

### 中部
- component builder
- batch stats helper
- batch facts helper
- epoch facts helper
- checkpoint save helper

### 底部
- `run_single_gpu_training(...)`
- `main()`

---

## 文件 B：`configs/qed_sa_single_gpu.yaml`

## 目标
一个明确表达单卡主线的配置。

## 建议内容
### `name`
例如：
- `diffsbdd-rlpf-qed-sa-single-gpu`

### `model`
保留：
- `checkpoint`
- `time_step`
- `device: "cuda"`

### `pocket`
保留：
- `pdb_path`
- `pdbqt_path`
- `ref_ligand`

### `reward`
例如只启用：
- `[qed, sa]`

### `dataloader`
设置单卡 batch 语义：
- `sample_group_size`
- `each_prompt_sample`
- `epochs`
- `min_nodes`
- `max_nodes`
- `max_n_nodes`

### `train`
保留 actor 所需字段：
- `learning_rate`
- `adam_beta1`
- `adam_beta2`
- `adam_weight_decay`
- `adam_epsilon`
- `clip_range`
- `adv_clip_max`
- `max_grad_norm`
- `train_micro_batch_size`
- `checkpoint_interval`
- `log_interval`
- `seed`

### 去掉 / 忽略
- `ddp`
- `ray`

或者保留但明确不使用。  
我更建议在单卡 config 里**直接不写 `ddp` / `ray`**，减少误导。

---

## 文件 C：`scripts/run_rl_qed_sa_single_gpu.sh`

## 目标
成为新日常入口。

## 行为
- 激活 `dif`
- 设置 `CUDA_VISIBLE_DEVICES`
- 打印 config/path/seed/log_level
- 调用：

```bash
python ddpo/main_diffsbdd_single_gpu.py ...
```

### 和旧脚本的区别
当前测试脚本还调用的是：

```56:60:scripts/run_rl_qed_sa_test.sh
"$PYTHON_BIN" ddpo/main_diffsbdd_lightning.py \
```

新脚本要改为新入口。

---

# 七、未来 PyTorch DDP 应该怎么接回来

这点我也提前说清楚，避免你现在改完以后未来再推倒。

---

## 正确的未来方向
未来不是把 Lightning 再加回来，而是：

> **在“单卡主循环已清晰”的前提下，用 PyTorch DDP 包一层进程级并行。**

---

## 未来 DDP 的正确分层
### 单卡主线负责
- rollout
- semantics
- filter
- reward
- actor.update_policy
- batch stats
- local recorder

### DDP 层负责
- 每 rank 一份进程
- seed = base_seed + rank
- 局部 batch 采样
- epoch end 汇总 reduce
- rank0 merge 记录与 checkpoint 管理

这比当前“Lightning 壳 + actor 真更新”的结构干净得多。

---

# 八、哪些地方绝对不能误改

这部分我单独强调。

---

## 1. `latents/logps/timesteps` 对齐不能改
对应文件：
- `ddpo/runtime/rollout.py`
- `RLPF-verl_diffusion/.../edm_actor.py`

这是一条极其脆弱但当前已经打通的链路。

---

## 2. `nodesxsample` / `max_n_nodes` / `get_mask()` 不能改语义
对应文件：
- `ddpo/runtime/data_loader.py`
- `ddpo/integration/diffsbdd_model.py`
- `edm_actor.py`

---

## 3. molecule semantics 必须保持单一 canonical path
对应文件：
- `ddpo/utils/molecule_semantics.py`

不要出现第二条分子重建路线。

---

## 4. reward 字段名尽量不改
尤其是：
- `rdkit_valid`
- `qed_scores`
- `sa_scores`
- `rewards`
- `smiles`
- `failure_type`

这些字段下游已经有依赖。

---

# 九、最终改造清单：一句话版

## 新增
- `ddpo/main_diffsbdd_single_gpu.py`
- `configs/...single_gpu....yaml`
- `scripts/...single_gpu....sh`

## 轻改
- `ddpo/main_diffsbdd_lightning.py` → 标记 legacy
- `ddpo/trainer/lightning_ddpo_module.py` → 标记 legacy
- `ddpo/runtime/recording/writers/metrics_writer.py` → 视需要修 checkpoint_path schema

## 复用不动
- `ddpo/runtime/data_loader.py`
- `ddpo/runtime/rollout.py`
- `ddpo/integration/diffsbdd_model.py`
- `ddpo/utils/molecule_semantics.py`
- `ddpo/utils/diffsbdd_filter.py`
- `ddpo/rewards/factory.py`
- `RunRecorder` 体系
- `EDMActor`

---

# 十、我建议的执行顺序

1. 新建 `main_diffsbdd_single_gpu.py`
2. 从 `main_diffsbdd_lightning.py` 复制初始化逻辑
3. 从 `LightningDDPOModule.forward()` 复制业务主链顺序
4. 手写 batch stats / BatchFacts / EpochFacts
5. 手写 checkpoint 保存
6. 新建单卡 yaml
7. 新建单卡脚本
8. 给 Lightning 入口加 legacy 标识
9. 跑单卡 smoke test
10. 修记录层小问题

---

如果你愿意，我下一步可以继续给你：

## “单卡入口 `ddpo/main_diffsbdd_single_gpu.py` 的逐段伪代码蓝图”
也就是直接按代码结构给出：
- 文件头怎么写
- 每个 helper 函数签名怎么定
- `main()` 里面每一段放什么

那会更接近真正开始编码前的施工图。