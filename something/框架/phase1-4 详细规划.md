那我再给你最后一层：**逐文件实施顺序 + 每个文件先改哪一段、后改哪一段的施工清单**。  
这层会更偏“真正开始动手前的工单顺序”。

---

# 一、总施工策略

这次不要“到处同时改”，而要按**从主干到外围**的顺序推进。

## 正确顺序
1. **先建新主入口**
2. **再把训练主循环搬进去**
3. **再接 recorder**
4. **再接 checkpoint**
5. **再补 config 和脚本**
6. **最后给 Lightning 打 legacy 标记**
7. **最后做 smoke test 和小修**

这样做的原因是：
- 先把核心训练链路立起来
- 外围兼容性和说明性修改后做
- 一旦 smoke test 失败，更容易定位问题在哪一层

---

# 二、逐文件施工顺序

---

## 第 1 个文件：`ddpo/main_diffsbdd_single_gpu.py`
这是第一优先级，也是最核心的文件。

---

## 这个文件要分 5 步写

### Step 1：先搭文件骨架
先写出：
- imports
- `parse_args()`
- `load_config()`
- `main()`

先别急着把所有 helper 写满。  
目标只是让文件结构立住。

### Step 2：把初始化逻辑搬进来
从 `ddpo/main_diffsbdd_lightning.py` 里搬这些初始化段：
- seed
- config load
- `create_run_context`
- `setup_logging`
- `DiffSBDDModel`
- `DiffSBDDDataLoader`
- `DiffSBDDRollout`
- `build_rewarder`
- `DiffSBDDFilter`
- `EDMActor`
- `RunRecorder`

这一步结束时，应该做到：
- 新入口能初始化所有组件
- 但可以还没开始正式训练

### Step 3：把训练主循环搬进来
主循环顺序严格按现有 `LightningDDPOModule.forward()`：
1. `generate_batch`
2. `generate_minibatch`
3. `annotate_batch_molecule_semantics`
4. `filters.filter`
5. `rewarder.calculate_rewards`
6. `advantages = rewards`
7. `actor.update_policy`

这一步是整个重构中最关键的步骤。

### Step 4：补统计与 recorder 调用
在主循环里补：
- `_compute_batch_stats`
- `_build_batch_facts`
- `_build_epoch_facts`
- `run_recorder.on_batch_end(...)`
- `run_recorder.on_epoch_end(...)`

### Step 5：补 checkpoint 保存
再加：
- `_save_checkpoint`
- `run_recorder.on_checkpoint_saved(...)`

### 这一文件完成后的标准
它应该能做到：
- 单卡完整跑 1 个 epoch
- 有 reward
- 有 PPO update
- 有 molecules / metrics / checkpoint 输出

---

# 三、`ddpo/main_diffsbdd_single_gpu.py` 里具体先后改哪段

再细一点说。

---

## 第一步先写这些函数壳
优先写空壳 / 简版实现：

1. `parse_args()`
2. `load_config()`
3. `_compute_batch_stats()`
4. `_build_batch_facts()`
5. `_build_epoch_facts()`
6. `_save_checkpoint()`
7. `run_training()`
8. `main()`

### 为什么先写壳
因为先把整个文件结构铺开后，你再填逻辑不会丢主线。

---

## 第二步最先填 `main()` 和初始化段
你可以先把下面这条链打通：

```python
args -> config -> runtime -> components -> run_training(...)
```

确保：
- imports 都正确
- 路径都能找到
- 没有明显模块找不到的问题

---

## 第三步填 `run_training()` 的主循环
这个时候只先写核心链，不急着写 recorder：

```python
prompts = dataloader.generate_batch()
samples = rollout.generate_minibatch(prompts)
samples = annotate_batch_molecule_semantics(samples)
samples, filtering_ratio, novelty_penalty_ratio = filters.filter(samples)
samples = rewarder.calculate_rewards(samples)
samples.batch["advantages"] = samples.batch["rewards"]
metrics = actor.update_policy(samples)
```

### 先这样做的原因
如果连这个都不通，后面的 recorder / checkpoint 都没意义。

---

## 第四步补 recorder
在这一步再把 batch/epoch 记录接上。

### 批次层
先构造：
- `molecule_rows`
- `BatchFacts`

然后调用：
- `run_recorder.on_batch_end(...)`

### epoch 层
再构造：
- `EpochFacts`

然后调用：
- `run_recorder.on_epoch_end(...)`

---

## 第五步补 checkpoint
这一步最后再接。

因为 checkpoint 保存依赖：
- 主循环已经通
- recorder 结构已经明确
- epoch 编号和输出路径已经稳定

---

# 四、第 2 个文件：`configs/qed_sa_single_gpu.yaml`
这是第二优先级。

---

## 先做什么
从现有：
- `configs/ddp_gpu_qed_sa_test.yaml`

复制出一份新文件。

---

## 然后按下面顺序修改

### Step 1：改名字
把 `name` 改成单卡语义清晰的名字，比如：
- `diffsbdd-rlpf-qed-sa-single-gpu`

### Step 2：去掉多卡配置
删掉或不写：
- `ddp`
- `ray`

### Step 3：明确单卡 batch 语义
确保：
- `sample_group_size`
- `each_prompt_sample`
- `train_micro_batch_size`
- `batch_size`
- `per_device_batch_size`

这些值在单卡场景下是自洽的。

### Step 4：reward 继续只保留 `qed` / `sa`
避免引入你已经删掉的 `connectivity`。

### Step 5：`train.seed` 建议显式保留
因为未来单卡 smoke test 很依赖 reproducibility。

---

## 这个文件完成后的标准
它应该明确表达：

> 这是一个不依赖 Lightning/DDP 的单卡 RL 训练配置。

---

# 五、第 3 个文件：`scripts/run_rl_qed_sa_single_gpu.sh`
这是第三优先级。

---

## 先做什么
直接复制：
- `scripts/run_rl_qed_sa_test.sh`

---

## 然后按顺序改

### Step 1：改脚本头注释
明确这是：
- single GPU launcher
- active mainline launcher

### Step 2：改默认 GPU
```bash
GPUS="${GPUS:-0}"
```

### Step 3：改 config 路径
指向新的单卡 yaml。

### Step 4：改 Python 入口
从：
- `ddpo/main_diffsbdd_lightning.py`

改为：
- `ddpo/main_diffsbdd_single_gpu.py`

### Step 5：保留环境激活逻辑
包括：
- `conda activate dif`
- `BABEL_LIBDIR`
- `PYTHONUNBUFFERED=1`

这些不要乱改。

---

## 这个文件完成后的标准
用户可以直接运行：

```bash
bash scripts/run_rl_qed_sa_single_gpu.sh
```

启动单卡主线。

---

# 六、第 4 个文件：`ddpo/main_diffsbdd_lightning.py`
这是第四优先级，属于“降级说明性修改”。

---

## 改动顺序
### Step 1：改 docstring
把 active 叙述改成 legacy 叙述。

### Step 2：必要时在 `main()` 启动时打印 warning
可以加一条 logger warning，例如：
- This Lightning entrypoint is legacy
- Prefer `main_diffsbdd_single_gpu.py`

### Step 3：不要继续改内部逻辑
这一文件在这次重构里应当“冻结”。

---

# 七、第 5 个文件：`ddpo/trainer/lightning_ddpo_module.py`
这也是降级说明性修改。

---

## 改动顺序
### Step 1：改类 docstring
说明它是：
- legacy wrapper
- preserved for reference

### Step 2：必要时补注释
标明：
- active training ownership moved to single-GPU mainline

### Step 3：不改 forward 业务逻辑
这次它只是“旧流程模板”，不是新主线。

---

# 八、第 6 个文件：`ddpo/runtime/recording/writers/metrics_writer.py`
这是一个可选小修文件，优先级在 smoke test 之后。

---

## 为什么它重要
当前这里可能存在一个 schema 小不一致：

- `SUMMARY_HEADER` 中没有 `checkpoint_path`
- 但 `update_epoch_checkpoint_path()` 似乎想往 CSV 里写 `checkpoint_path`

---

## 改动顺序
### Step 1：先不要碰
等单卡主线跑通后看是否真的影响输出。

### Step 2：如果输出里需要 checkpoint path
那就：
- 把 `checkpoint_path` 加到 `SUMMARY_HEADER`
- 在 `write_epoch_summary()` 中初始化该字段

### Step 3：如果不需要
那就考虑删掉 `update_epoch_checkpoint_path()` 或暂时不调用它

---

# 九、哪些文件这次明确不要改

这点也很重要。

---

## 1. `ddpo/runtime/rollout.py`
理由：
- trajectory 协议敏感
- actor 强依赖
- 当前已经跑通过

## 2. `ddpo/runtime/data_loader.py`
理由：
- batch 语义已经稳定
- 这次改主线，不改 batch 协议

## 3. `ddpo/integration/diffsbdd_model.py`
理由：
- mask / sample_p_zs_given_zt / model bridge 核心敏感

## 4. `ddpo/utils/molecule_semantics.py`
理由：
- canonical semantics 核心，不要顺手重构

## 5. `ddpo/utils/diffsbdd_filter.py`
理由：
- 当前已适配 canonical semantics
- 新主线直接复用即可

## 6. `RLPF-verl_diffusion/.../edm_actor.py`
理由：
- optimizer / PPO 核心
- 本次只承认它的所有权，不重构它

---

# 十、真正开始改代码时的最优执行节奏

我给你一个最稳的“改-测-再改”节奏。

---

## Round 1：只搭新入口，不跑训练
做完：
- 新文件创建
- imports 正确
- 能启动到初始化完成

### 目标
先验证：
- 路径
- import
- config
- logger
- model init

---

## Round 2：只接核心训练链
接入：
- rollout
- semantics
- filter
- reward
- actor.update_policy

### 目标
先看：
- 1 step 能不能跑完
- 有无 shape / mask / log-prob 问题

---

## Round 3：接 recorder
接入：
- batch facts
- epoch facts

### 目标
先看：
- `molecules_rank0.csv`
- `molecules.csv`
- `metrics.json`
- `training_summary.csv`

---

## Round 4：接 checkpoint
接入：
- epoch ckpt
- last.ckpt
- checkpoint tracker

### 目标
检查：
- 文件存在
- 内容正确
- actor optimizer state 存进去了

---

## Round 5：补脚本和 config
最后再补启动脚本与配置，使它真正好用。

---

# 十一、完成后你应该能得到什么

重构完成后的项目状态应该是：

---

## Active path
- `ddpo/main_diffsbdd_single_gpu.py`
- `configs/qed_sa_single_gpu.yaml`
- `scripts/run_rl_qed_sa_single_gpu.sh`

---

## Legacy path
- `ddpo/main_diffsbdd_lightning.py`
- `ddpo/trainer/lightning_ddpo_module.py`

---

## 运行体验
你以后日常调试应该是：

```bash
bash scripts/run_rl_qed_sa_single_gpu.sh
```

而不是再走 Lightning/DDP。

---

## 输出体验
每次训练都会在 `OUT/...` 下产出：
- config snapshot
- training_summary
- epoch metrics
- molecules.csv
- checkpoints

---

# 十二、最后给你一句施工总原则

如果你开始动手时只记一件事，就记这个：

> **先把“训练控制权”从 Lightning 脱出来，再考虑任何别的优化。**

也就是说：
- 不要一边拆 Lightning，一边顺手改 reward
- 不要一边建单卡入口，一边重构 actor
- 不要一边写新入口，一边重新设计 recorder

先把**单卡主循环**建立为唯一主线，剩下的都能后续渐进做。

---

如果你下一步要我继续，我就不再讲方案了，而是会直接进入**真正实现**：
- 新建单卡入口
- 新建 config
- 新建 launcher
- 标记 Lightning legacy
- 然后做 smoke test。