那我继续，给你 **`ddpo/main_diffsbdd_single_gpu.py` 的逐段施工图**，尽量精确到“这一段代码应该负责什么、输入输出是什么、从哪里复用”。

---

# 一、目标文件整体结构

建议这个新文件按下面结构组织：

```python
# 1. imports / path setup / warning suppression
# 2. parse_args()
# 3. load_config()
# 4. helper: _setup_runtime(...)
# 5. helper: _build_components(...)
# 6. helper: _compute_batch_stats(...)
# 7. helper: _build_batch_facts(...)
# 8. helper: _build_epoch_facts(...)
# 9. helper: _save_checkpoint(...)
# 10. run_training(...)
# 11. main()
```

这样做的原因是：
- `main()` 不会过长
- 训练主循环清晰
- checkpoint / recorder / stats 都是显式函数
- 未来做 DDP 时，这些 helper 基本都能复用

---

# 二、文件头应该放什么

---

## 1. 顶部 docstring
建议直接写清楚新入口定位，例如：

```python
"""
Single-GPU active training entrypoint for DiffSBDD + RLPF-verl_diffusion RL.

This entrypoint replaces Lightning/DDP as the active workflow.
It keeps:
- DiffSBDD rollout semantics
- EDMActor PPO update semantics
- canonical molecule semantics
- existing reward semantics
- existing run recording semantics

It intentionally does NOT use:
- PyTorch Lightning Trainer
- Lightning callbacks/checkpoint loop
- DDP orchestration

Future multi-GPU support should wrap this single-GPU logic with PyTorch DDP,
rather than reintroducing Lightning as the active training owner.
"""
```

### 目的
以后任何人一打开这个文件，就知道：
- 这是 active path
- 为什么存在
- 与 Lightning 的关系是什么

---

## 2. import 区
建议分 6 组：

### A. Python 标准库
- `argparse`
- `logging`
- `os`
- `sys`
- `shutil`
- `warnings`
- `datetime`
- `pathlib.Path`

### B. 三方库
- `torch`
- `numpy`
- `yaml`

### C. path setup
和 `main_diffsbdd_lightning.py` 一样，把：
- `PROJECT_ROOT`
- `RLPF-verl_diffusion`
- `DiffSBDD`

加入 `sys.path`

### D. RLPF actor
- `EDMActor`

### E. DiffSBDD constants
- `dataset_params`

### F. 本项目模块
- `DiffSBDDModel`
- `build_rewarder`
- `DiffSBDDFilter`
- `annotate_batch_molecule_semantics`
- runtime:
  - `DiffSBDDDataLoader`
  - `DiffSBDDRollout`
  - `setup_logging`
  - `set_seed`
- recording:
  - `RunRecorder`
  - `create_run_context`
  - `BatchFacts`
  - `EpochFacts`
  - `CheckpointFacts`
  - `RunLifecycleFacts`
  - `RunStatus`

---

## 3. warning suppression / logging suppression
这里建议**复用你当前 Lightning 入口的 suppression 策略**，不要另起一套。

例如：
- RDKit warning suppression
- OpenBabel warning suppression
- transformers warning suppression
- `PYTHONUNBUFFERED=1`

### 原因
这属于输出控制，不改变训练语义。  
从用户体验看，直接沿用最稳。

---

# 三、`parse_args()` 设计

建议直接沿用 Lightning 入口接口，这样你现有脚本几乎不用学习成本。

---

## 建议签名
```python
def parse_args():
    ...
```

## 参数建议
### 必选
- `--config_path`

### 可选
- `--save_path`
- `--seed`
- `--log_level`
- `--resume`
- `--checkpoint_path`

---

## 为什么要保留 `resume`
虽然这次重点不是 resume，但入口保留这个参数是有价值的：
- 接口稳定
- 后续 checkpoint 接上后不需要再改 CLI

---

# 四、`load_config(config_path)` 设计

---

## 建议签名
```python
def load_config(config_path: str) -> dict:
```

## 行为
- `yaml.safe_load`
- 返回 dict

## 备注
这部分可以直接沿用当前 Lightning 入口。

---

# 五、`_setup_runtime(args, config)` 设计

这个 helper 的目标是把**环境层初始化**集中起来。

---

## 建议签名
```python
def _setup_runtime(args, config: dict):
```

## 它负责什么
1. `set_seed(args.seed)`
2. 创建 `RunContext`
3. 初始化 logger
4. 保存 config snapshot
5. 处理 `save_path` override
6. 判定 device
7. 返回运行时公共对象

---

## 建议返回内容
可以返回 tuple，也可以返回 dict。  
我更建议返回 dict，例如：

```python
{
    "config": config,
    "device": device,
    "logger": logger,
    "run_context": run_context,
}
```

---

## 关键细节
### 1. `RunContext` 继续做 single source of truth
你现在已经有：

- `output_dir`
- `logs_dir`
- `epochs_dir`
- `checkpoints_dir`
- `config_snapshot_path`

这些都非常适合新入口继续用。

### 2. config snapshot 继续保留
这非常重要，关系到 reproducibility。

---

# 六、`_build_components(config, device, logger, run_context)` 设计

这个函数是**组件构造器**。

---

## 建议签名
```python
def _build_components(config: dict, device: str, logger, run_context):
```

## 负责初始化
1. `dataset_info`
2. `DiffSBDDModel`
3. `DiffSBDDDataLoader`
4. `DiffSBDDRollout`
5. `rewarder`
6. `filters`
7. `actor`
8. `run_recorder`

---

## 每一项的构造建议

### 1. `dataset_info`
```python
dataset_info = dataset_params["crossdock"]
```

### 2. `model`
直接沿用当前构造方式：

```python
model = DiffSBDDModel(
    checkpoint_path=config["model"]["checkpoint"],
    pocket_pdb_path=config["pocket"]["pdb_path"],
    ref_ligand_path=config["pocket"].get("ref_ligand", "C:CIP"),
    device=device,
    dataset_name="crossdock",
)
```

### 3. `dataloader`
继续用：

```python
dataloader = DiffSBDDDataLoader(
    config=config,
    dataset_info=dataset_info,
    device=device,
    num_batches=config["dataloader"].get("epochs", 100),
)
```

注意：虽然这里叫 `num_batches`，但当前 `__len__`/`__next__` 语义还是一 epoch 一 batch。  
这次先不改。

### 4. `rollout`
```python
rollout = DiffSBDDRollout(model, config)
```

### 5. `rewarder`
```python
rewarder = build_rewarder(config=config, dataset_info=dataset_info, logger=logger)
```

### 6. `filters`
建议继续使用当前“只统计，不删样本”的策略：

```python
filters = DiffSBDDFilter(
    dataset_info=dataset_info,
    enable_filtering=False,
    enable_penalty=False,
)
```

### 7. `actor`
```python
actor = EDMActor(model, config)
actor.max_n_nodes = config["dataloader"]["max_nodes"]
```

注意：这里 `actor.max_n_nodes` 现在是外部补进去的，未来若有时间可以再正规化，但这次先保持现状。

### 8. `run_recorder`
```python
run_recorder = RunRecorder(
    run_context=run_context,
    logger=logger,
    world_size=1,
)
```

这里明确单卡：
- `world_size=1`

---

## 建议返回值
建议返回 dict：

```python
{
    "dataset_info": dataset_info,
    "model": model,
    "dataloader": dataloader,
    "rollout": rollout,
    "rewarder": rewarder,
    "filters": filters,
    "actor": actor,
    "run_recorder": run_recorder,
}
```

---

# 七、`_compute_batch_stats(samples)` 设计

这个函数直接借鉴 `LightningDDPOModule._compute_batch_stats(...)`，但去掉分布式色彩。

---

## 建议签名
```python
def _compute_batch_stats(samples) -> dict:
```

## 输入
- `samples: DataProto`

## 读取字段
- `qed_scores`
- `sa_scores`
- `rewards`
- `rdkit_valid`

## 输出建议
```python
{
    "avg_qed": ...,
    "avg_sa": ...,
    "avg_reward": ...,
    "valid_ratio": ...,
    "generated_molecules": ...,
    "valid_molecules": ...,
}
```

---

## 为什么单独抽函数
因为后面：
- batch facts
- epoch facts
- console logging

都需要这些统计值。

---

# 八、`_build_batch_facts(...)` 设计

这个函数负责把 sample 层数据转成 recorder 需要的 `BatchFacts`。

---

## 建议签名
```python
def _build_batch_facts(
    samples,
    epoch: int,
    batch_idx: int,
    filtering_ratio: float,
    novelty_penalty_ratio: float,
    batch_stats: dict,
) -> BatchFacts:
```

---

## 内部要做什么
### 1. 从 `samples.batch` 里读：
- `smiles`
- `rdkit_valid`
- `qed_scores`
- `sa_scores`
- `rewards`
- `druglikeness_reward`
- `docking_reward`

### 2. 构造 `molecule_rows`
这部分可以几乎照搬当前 `LightningDDPOModule.forward()` 中的逻辑。

### 3. 固定 rank
单卡时明确：
```python
rank = 0
```

### 4. 构造 `BatchFacts`
例如：

```python
facts = BatchFacts(
    epoch=epoch,
    batch_idx=batch_idx,
    rank=0,
    generated_count=n,
    valid_count=...,
    reward_mean=batch_stats["avg_reward"],
    reward_min=...,
    reward_max=...,
    avg_qed=batch_stats["avg_qed"],
    avg_sa=batch_stats["avg_sa"],
    filtering_ratio=filtering_ratio,
    novelty_penalty_ratio=novelty_penalty_ratio,
    molecule_rows=molecule_rows,
)
```

---

## 这里有个小建议
当前 Lightning 版本里：
- `reward_min`
- `reward_max`
很多地方暂时写的是 0.0

单卡版本如果方便，建议真实计算：
- `min(rewards)`
- `max(rewards)`

这样记录更完整。

---

# 九、`_build_epoch_facts(...)` 设计

这个函数负责把当前 epoch 的训练结果转成 `EpochFacts`。

---

## 建议签名
```python
def _build_epoch_facts(
    epoch: int,
    max_epochs: int,
    metrics: dict,
    batch_stats: dict,
    filtering_ratio: float,
    novelty_penalty_ratio: float,
    checkpoint_due: bool,
) -> EpochFacts:
```

---

## 为什么可以简单
因为当前单卡版本下：
- local = global
- 不需要 all_reduce
- 不需要 rank merge

所以 epoch facts 可直接取 batch stats。

---

## 关键字段建议
```python
EpochFacts(
    epoch=epoch,
    timestamp=...,
    avg_loss=float(metrics.get("Loss", 0.0)),
    reward_mean=float(batch_stats.get("avg_reward", 0.0)),
    reward_min=...,
    reward_max=...,
    avg_qed=float(batch_stats.get("avg_qed", 0.0)),
    avg_sa=float(batch_stats.get("avg_sa", 0.0)),
    generated_total=int(batch_stats.get("generated_molecules", 0)),
    rdkit_valid=int(batch_stats.get("valid_molecules", 0)),
    rdkit_valid_ratio=float(batch_stats.get("valid_ratio", 0.0)),
    filtering_ratio_mean=filtering_ratio,
    novelty_penalty_ratio_mean=novelty_penalty_ratio,
    extra_metrics={"max_epochs": max_epochs},
    checkpoint_due=checkpoint_due,
)
```

---

# 十、`_save_checkpoint(...)` 设计

这是新主线中最关键的一个替代 Lightning callback 的函数。

---

## 建议签名
```python
def _save_checkpoint(
    model,
    actor,
    config: dict,
    run_context,
    epoch: int,
    logger,
) -> CheckpointFacts:
```

---

## 建议保存内容
建议保存成一个 dict：

```python
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "actor_optimizer_state": actor.optimizer.state_dict(),
    "config": config,
    "saved_at": timestamp,
}
```

### 可选额外字段
- `run_name`
- `run_id`
- `model_checkpoint_path`
- `seed`

---

## 文件路径建议
### 周期 checkpoint
```python
epoch_path = checkpoints_dir / f"epoch={epoch:04d}.ckpt"
```

### last checkpoint
```python
last_path = checkpoints_dir / "last.ckpt"
```

保存顺序建议：
1. 保存 epoch 文件
2. 再覆盖 `last.ckpt`

---

## 返回 recorder 所需事实
```python
CheckpointFacts(
    epoch=epoch,
    path=str(epoch_path),
    kind="epoch",
    monitor=None,
    monitor_value=None,
    exists=epoch_path.exists(),
    size_bytes=epoch_path.stat().st_size if epoch_path.exists() else 0,
    confirmed=epoch_path.exists(),
    saved_at=timestamp,
)
```

---

## 为什么单独抽出来
因为 checkpoint 是未来 DDP 最容易再用的一层。  
以后 DDP 只要加 rank0 gate 就行。

---

# 十一、`run_training(...)` 主循环设计

这是整个文件最核心的函数。

---

## 建议签名
```python
def run_training(config: dict, args, runtime: dict, components: dict):
```

或者把参数拆开都可以。

---

## 函数内部推荐结构

### 1. 先取出需要的对象
- `logger`
- `run_context`
- `model`
- `dataloader`
- `rollout`
- `rewarder`
- `filters`
- `actor`
- `run_recorder`

---

### 2. run lifecycle 开始
建议先写 manifest：

```python
run_recorder.on_run_started()
```

---

### 3. epoch loop
```python
for epoch in range(1, max_epochs + 1):
```

注意：
- recorder 现在更偏 1-indexed epoch 命名
- 建议新主线也用 1-indexed，保持输出目录一致

---

### 4. 每个 epoch 的实际步骤

#### a. generate batch
```python
prompts = dataloader.generate_batch()
```

#### b. rollout
```python
samples = rollout.generate_minibatch(prompts)
```

#### c. canonical molecule semantics
```python
samples = annotate_batch_molecule_semantics(samples)
```

#### d. filter
```python
samples, filtering_ratio, novelty_penalty_ratio = filters.filter(samples)
```

#### e. reward
```python
samples = rewarder.calculate_rewards(samples)
```

#### f. ensure advantages
```python
if "advantages" not in samples.batch:
    ...
    samples.batch["advantages"] = samples.batch["rewards"]
```

#### g. actor update
```python
metrics = actor.update_policy(samples)
```

#### h. batch stats
```python
batch_stats = _compute_batch_stats(samples)
```

#### i. write batch facts
```python
batch_facts = _build_batch_facts(...)
run_recorder.on_batch_end(batch_facts)
```

#### j. determine checkpoint_due
```python
checkpoint_interval = int(config["train"].get("checkpoint_interval", 1))
checkpoint_due = (epoch % checkpoint_interval == 0)
```

#### k. build epoch facts
```python
epoch_facts = _build_epoch_facts(...)
run_recorder.on_epoch_end(epoch_facts)
```

#### l. save checkpoint if due
```python
if checkpoint_due:
    checkpoint_facts = _save_checkpoint(...)
    run_recorder.on_checkpoint_saved(checkpoint_facts)
```

---

### 5. try/except/finally
建议包住整个训练过程：

#### `KeyboardInterrupt`
- 标记 interrupted
- `run_recorder.on_run_finished(...)`

#### `Exception`
- 标记 failed
- 记录 message
- 再 raise 或直接退出

#### 正常结束
- 标记 completed

---

# 十二、`main()` 设计

---

## 建议顺序
```python
def main():
    args = parse_args()
    config = load_config(args.config_path)

    runtime = _setup_runtime(args, config)
    components = _build_components(...)

    run_training(config, args, runtime, components)
```

### 原则
`main()` 尽量不要自己做业务逻辑。

---

# 十三、旧文件的具体改造建议文案

如果你要把 legacy 标记做得更清楚，建议这样写。

---

## `ddpo/main_diffsbdd_lightning.py`
### 建议在文件头 docstring 中加入：
- This entrypoint is legacy
- Active workflow moved to `ddpo/main_diffsbdd_single_gpu.py`
- Kept for reference and rollback only

---

## `ddpo/trainer/lightning_ddpo_module.py`
### 建议在类 docstring 中加入：
- Legacy Lightning wrapper
- No longer the active training owner
- Preserved for reference and future migration comparisons

---

# 十四、配置文件蓝图

---

## 新文件：`configs/qed_sa_single_gpu.yaml`

建议核心结构是：

```yaml
name: diffsbdd-rlpf-qed-sa-single-gpu

model:
  checkpoint: "./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
  time_step: 500
  device: "cuda"

pocket:
  pdb_path: "./data/8GJJ-C-Cip.pdb"
  pdbqt_path: "./data/8GJJ-C-Cip_charged.pdbqt"
  ref_ligand: "C:1101"

reward:
  enabled: [qed, sa]
  weights:
    qed: 0.5
    sa: 0.5

dataloader:
  sample_group_size: 2
  each_prompt_sample: 8
  epochs: 5
  min_nodes: 10
  max_nodes: 30
  max_n_nodes: 30

train:
  seed: 42
  batch_size: 16
  per_device_batch_size: 16
  train_micro_batch_size: 16
  learning_rate: 1.0e-4
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_weight_decay: 1.0e-4
  adam_epsilon: 1.0e-8
  clip_range: 0.2
  adv_clip_max: 10.0
  max_grad_norm: 1.0
  checkpoint_interval: 1
  log_interval: 1
  disable_tqdm: true

output:
  dir: "./OUT/<run_name>_<YYYYMMDD>_<seq>"
```

---

## 配置上的关键原则
### 建议去掉
- `ddp`
- `ray`

### 为什么
因为单卡主线里它们会制造误导，甚至让后续读配置的人以为还在依赖旧路径。

---

# 十五、启动脚本蓝图

---

## 新文件：`scripts/run_rl_qed_sa_single_gpu.sh`

结构建议和你现在的测试脚本一致，只改三处：

### 1. 默认 GPU
```bash
GPUS="${GPUS:-0}"
```

### 2. config 路径
```bash
CONFIG_PATH="$PROJECT_ROOT/configs/qed_sa_single_gpu.yaml"
```

### 3. Python 入口
```bash
"$PYTHON_BIN" ddpo/main_diffsbdd_single_gpu.py \
  --config_path "$CONFIG_PATH" \
  --seed "$SEED" \
  --log_level "$LOG_LEVEL"
```

---

# 十六、验证蓝图

完成代码后，建议按下面顺序测。

---

## Test 1：纯 import / init
目标：
- 新入口能初始化所有组件
- 不进入 epoch loop 也不炸

---

## Test 2：1 epoch smoke test
目标：
- rollout 成功
- reward 成功
- actor.update_policy 成功
- 无 NaN/Inf

---

## Test 3：输出文件检查
检查：
- `config.yaml`
- `training_summary.csv`
- `epochs/epoch_0001/metrics.json`
- `epochs/epoch_0001/molecules.csv`
- `checkpoints/epoch=0001.ckpt`
- `checkpoints/last.ckpt`

---

## Test 4：checkpoint 内容检查
检查是否包含：
- `model_state`
- `actor_optimizer_state`
- `config`
- `epoch`

---

## Test 5：固定 seed 重跑
目标：
- 排查明显的随机不一致问题

---

# 十七、这张蓝图里最核心的一句话

如果把整个改造压缩成一句最重要的工程准则，就是：

> **新文件只接管“训练控制权”，不要改“采样语义、奖励语义、分子语义、轨迹语义”。**

---

如果你愿意，下一步我可以继续给你最后一层：

## “逐文件实施顺序 + 每个文件先改哪一段，后改哪一段”
也就是非常实操的施工清单，适合你真正开始敲代码前最后确认。