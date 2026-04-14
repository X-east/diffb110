下面给你一版完整、详细、结构化的修复总结。我会把：

1. 项目当前主线架构

2. 训练执行链路

3. 记录/日志/输出链路

4. 奖励系统架构

5. 这次 bug 的根因

6. 我实际做了什么修复

7. 修复后验证到了什么

8. 还没完全收尾的点

9. 建议下一步

全部串起来讲清楚。

---

# 一、项目当前主线架构

这个仓库现在的主线不是老的 Ray 路线，而是：

- DiffSBDD：负责分子扩散生成本体

- ddpo/runtime/*：把 DiffSBDD 产物包装成 DDPO/RLPF 训练流能吃的结构

- RLPF-verl_diffusion：提供一部分 PPO/DDPO 优化内核，例如 EDMActor

- PyTorch Lightning：负责多卡 DDP、logger、checkpoint、epoch 生命周期

- ddpo/rewards/*：奖励构造与组件化组合

- ddpo/runtime/recording/*：这次重点相关的运行记录主线

ddpo/runtime/__init__.py 已经把这条“Lightning mainline runtime”明确组织出来了：

__init__.pyLines 1-38

这说明目前主线已经被拆成几个清晰层次：

---

## 1. 入口层：ddpo/main_diffsbdd_lightning.py

这个文件是当前主训练入口。它负责：

- 解析命令行

- 读 YAML 配置

- 设置 seed

- 创建 RunContext

- 初始化 logging

- 初始化 model / dataloader / rollout / reward / filter / actor

- 构造 Lightning module

- 构造 Trainer 和 callbacks

- 启动 trainer.fit(...)

你最近跑的脚本 run_rl_qed_sa_test.sh 最终就是走这里。

---

## 2. Lightning 编排层：LightningDDPOModule

这是 PyTorch Lightning 的封装壳。它自己不发明新的 RL 语义，而是负责：

- 在 training_step 中做 rollout、filter、reward、policy update

- 把每个 batch 的统计信息记录下来

- 在 epoch end 汇总全局统计

- 把这些统计交给 recording 主线落盘

它的职责本质上是：

> “用 Lightning 驱动训练生命周期，但真正的 DDPO 更新语义仍由现有 actor / reward / rollout 保持。”

---

## 3. 运行记录层：ddpo/runtime/recording/*

这是这次修复最相关的部分。它把训练结果落到统一输出目录，例如：

- config.yaml

- training_summary.csv

- run_manifest.json

- epochs/epoch_0001/metrics.json

- epochs/epoch_0001/molecules.csv

- checkpoints/checkpoint_manifest.csv

这层的结构大致是：

### RunContext

负责描述一次 run 的目录结构和关键路径。

run_context.pyLines 11-24

注意：它是 frozen=True 的 dataclass。这就是这次 bug 的根因核心。

### RunRecorder

负责消费结构化事实对象并写文件：

- BatchFacts

- EpochFacts

- CheckpointFacts

- RunLifecycleFacts

### RecordingCheckpointCallback

负责在 Lightning epoch 结束后，把 ModelCheckpoint 当前状态转成 CheckpointFacts，再交给 RunRecorder。

callbacks.pyLines 13-40

---

# 二、训练执行链路

你的主线训练现在可以按下面的执行顺序理解：

## 第 1 步：入口读取配置并创建 run

main_diffsbdd_lightning.py 调用 create_run_context(...)，建立本次训练的输出目录和文件路径。

## 第 2 步：初始化训练部件

入口负责创建：

- DiffSBDDModel

- DiffSBDDDataLoader

- DiffSBDDRollout

- rewarder

- filters

- EDMActor

- LightningDDPOModule

## 第 3 步：Lightning 驱动训练

trainer.fit(lightning_module, dataloader)

Lightning 负责：

- 多进程 DDP 启动

- rank 分发

- epoch 循环

- callback 生命周期

- logger 与 checkpoint 触发

## 第 4 步：每个 batch 内部的实际训练逻辑

在 LightningDDPOModule.forward() 里：

1. rollout 采样

2. 分子语义标注

3. filter

4. reward 计算

5. advantage 计算

6. actor 更新策略

7. 返回 loss + metrics + samples

## 第 5 步：每个 epoch 的结构化落盘

在 on_train_epoch_end() 里，module 汇总本 epoch 各 rank 的统计，并生成 EpochFacts 给 recorder。

## 第 6 步：checkpoint 记录

RecordingCheckpointCallback 读取 Lightning checkpoint callback 当前状态，生成 CheckpointFacts。

---

# 三、记录与输出架构

这次 run 的输出结构是清晰的，而且已经在真实训练中落出来了。当前你看到的目录里已经有：

- config.yaml

- run_manifest.json

- training_summary.csv

- lightning_logs/version_0/hparams.yaml

- epochs/epoch_0001/metrics.json

- epochs/epoch_0001/epoch_status.json

- epochs/epoch_0001/molecules_rank0.csv

- epochs/epoch_0001/molecules_rank1.csv

- epochs/epoch_0001/molecules.csv

- checkpoints/checkpoint_manifest.csv

RunRecorder 这层里，MetricsWriter 负责 training_summary.csv 和 epoch_status.json：

run_recorder.pyLines 17-76

这意味着当前记录主线已经具备：

- epoch 汇总

- per-rank molecule 明细

- merged molecule csv

- checkpoint manifest

- run manifest

---

# 四、奖励系统架构

奖励部分也已经被组织成清晰架构，不再是入口文件里硬编码拼接。

## 1. 奖励工厂：ddpo/rewards/factory.py

这个模块是主线奖励组装点。它从 config 中读取：

- enabled reward components

- weights

- composite 配置

- docking backend 配置

然后组装成一个 CompositeReward，最后再包进 StabilityAwareReward。

factory.pyLines 117-161

## 2. 组合奖励：CompositeReward

它负责多个 reward component 的线性组合：

composite.pyLines 32-67

它的语义是：

- 每个组件各自算出 reward

- 按配置权重组合

- 所有组件都失败时用 fallback_reward

- 每个组件 reward 仍保存在 batch 里，便于日志/检查

这一点符合你的项目规则：reward component 必须可见、可配置、可追踪。

---

# 五、这次 bug 的根因

现在说这次真正修的是什么。

## 现象

一开始训练在多卡 DDP 启动后，没有进入正常 epoch，而是报：

- FrozenInstanceError: cannot assign to field 'run_id'

- 随后 Lightning DDP 抛出：

- DeadlockDetectedException

日志堆栈显示问题出现在：

- trainer._log_hyperparams()

- TensorBoardLogger.save()

- save_hparams_to_yaml(...)

- apply_to_collection(...)

## 根因链条

这个错误不是 reward、不是 diffusion、不是 PPO，而是：

### 1. RunContext 是冻结 dataclass

run_context.pyLines 11-24

### 2. LightningDDPOModule 调用了 save_hyperparameters(...)

Lightning 会把构造参数中的对象都当成 hparams 递归处理，准备写进 hparams.yaml。

### 3. 但当时没有排除 run_context

也就是说，一个冻结 dataclass 的运行时对象被塞进了 Lightning logger 的超参数保存流程。

### 4. Lightning 的递归处理会尝试对对象做集合转换/赋值

对于普通对象问题不大，但对 frozen=True dataclass，就会触发：

- FrozenInstanceError

### 5. DDP 下一个 rank 报错后，其他 rank 还在 barrier / 生命周期等待

于是最终表面上看到的是：

- DeadlockDetectedException

所以这次 bug 的本体是：

> 运行时记录对象 RunContext 错误进入了 Lightning 超参数持久化路径。

---

# 六、我实际做了什么修复

我做的是最小修复，没有改训练语义，只改了 Lightning hparams 边界。

## 1. 在 LightningDDPOModule 中排除运行态对象

现在 save_hyperparameters(...) 显式忽略了：

- model

- rollout

- rewarder

- filters

- actor

- run_context

- run_recorder

lightning_ddpo_module.pyLines 42-51

### 这个修复的意义

它明确了一个重要边界：

- 可序列化、可复现实验配置 才属于 hparams

- 运行时状态对象 不应该进入 logger hparams

这非常符合科学代码与可复现要求。

---

## 2. 增加了回归测试

我加了专门的测试，验证：

- run_context 不会进入 module.hparams

- run_recorder 不会进入 module.hparams

- TensorBoardLogger.save() 能正常生成 hparams.yaml

test_recording_mainline.pyLines 24-63

---

## 3. 修正测试隔离问题

在测试中我还处理了 DIFFSBDD_RUN_OUTPUT_DIR 环境变量串扰，避免不同测试复用同一个输出目录。

这不是主 bug 本体，但会导致：

- training_summary.csv 多写一行

- 测试不稳定

- 回归覆盖失真

所以这个也顺手修掉了。

---

# 七、修复后已经验证到什么程度

这个部分很重要：不是只靠“代码看起来对”，而是已经有了实际运行证据。

## 1. 训练已经真实进入多卡 DDP 并跑完了至少 1 个 epoch

你最新的日志明确显示：

- DDP 两卡启动成功

- rank 0 / rank 1 都起来了

- Epoch 1 正常完成

- 统计正常输出

也就是说，之前那个“刚启动就炸”的问题已经消失。

## 2. hparams.yaml 已经成功写出

这点非常关键，因为之前就是在这里崩溃的。

当前输出目录中已经存在：

- lightning_logs/version_0/hparams.yaml

这说明：

- Lightning logger 的 hparams 保存流程已经通过

- 冻结 dataclass 不再进入这个流程

## 3. recording 主线已正常落出 epoch 结果

已经落出了：

- training_summary.csv

- run_manifest.json

- epochs/epoch_0001/metrics.json

- epochs/epoch_0001/molecules.csv

这说明 run recorder 主线是活的。

## 4. 我本地按项目正确 PYTHONPATH 跑了测试

test/test_recording_mainline.py 跑通，结果是 3 passed。

---

# 八、这次修复没有改动的语义边界

这部分我明确说一下，避免误解。

## 没改 diffusion 语义

没有改：

- x0

- xt

- timestep 顺序

- denoising / rollout 语义

## 没改 mask / batch 语义

没有改：

- padded / valid 节点处理

- batch flattening

- nodesxsample 相关语义

## 没改 PPO / DDPO 语义

没有改：

- reward

- advantage

- ratio / clipped ratio

- entropy bonus

- actor 更新逻辑

## 没改 reward meaning

没有改：

- druglikeness 计算语义

- docking 计算语义

- composite reward 加权语义

所以这是一次很“干净”的修复：

只修复 logger / hparams 的运行态对象边界，不触碰科学语义。

---

# 九、当前还没有完全完成的部分

虽然主 bug 已修，但整体还有两个尾巴。

## 1. checkpoint 落盘/确认链路还没完全闭环

当前你这次 run 的输出里：

- training_summary.csv 已写入 epoch 1

- checkpoint_manifest.csv 已写入一条记录

- 但 checkpoints/ 目录下没有实际 .ckpt

- 记录里 path 为空、confirmed=0

终端也打印了：

- [CHECKPOINT] Epoch 1: not confirmed

这说明：

- recorder 的 checkpoint 记录链路已经工作

- 但 Lightning 的 last_model_path 在那个时点没有对应到实际文件

- 所以 checkpoint 的“记录”与“文件真实落盘”还没有完全对上

这不是这次主崩溃 bug，但它是目前 recording/输出主线的一个剩余问题。

---

## 2. 测试命令需要标准化入口

你直接跑：

会因为 verl_diffusion 不在默认 PYTHONPATH 上而报错。

原因是：

- ddpo.runtime 间接依赖 RLPF-verl_diffusion

- 主入口在运行时手动加了 sys.path

- 裸 pytest 没有这个路径设置

所以当前测试应这样跑：

这说明测试环境入口还不够统一。

---

# 十、整体状态总结

我给你一个最简洁但完整的结论：

## 已修好的

### bug6 主问题

已修复。

具体是：

- 冻结的 RunContext 不再进入 Lightning hparams 保存路径

- FrozenInstanceError 不再发生

- DDP 不再因此 deadlock

- 训练已能真实进入并完成 epoch 1

- hparams.yaml 已成功写出

- 回归测试已补并通过

---

## 当前主线架构状态

### 已成型

现在主线可以清楚理解为：

- main_diffsbdd_lightning.py：训练入口

- LightningDDPOModule：训练生命周期编排壳

- reward factory + composite reward：奖励配置驱动组装

- run_context + run_recorder + checkpoint callback：结构化训练记录

- DiffSBDD + runtime adapter + EDMActor：保留现有生成/RL 核心语义

这套架构已经具备“主线可运行、可记录、可测试”的雏形。

---

## 还没完全完成的

1. checkpoint 文件真实落盘与 recorder 确认还没闭环

2. 测试入口还缺少统一环境封装

3. 这次训练是被 ^Z 挂起，不是完整结束

---

# 十一、建议下一步

如果你让我继续做，最值得优先收尾的是：

## 下一步 1：查 checkpoint 为什么没真正写出来

目标：

- 让 checkpoint_manifest.csv 的 path/exists/confirmed 正确

- 让 training_summary.csv 的 checkpoint_path 正确更新

- 确保 epoch 到点后真的有 .ckpt

## 下一步 2：统一 pytest 入口

目标：

- 做一个测试脚本或 conftest/路径注入

- 避免每次手工补 PYTHONPATH

---

如果你愿意，我下一条可以继续给你写一版：

“bug6 修复后的完整技术变更说明 + 未完成事项清单（适合存档/汇报版）”

会更像正式文档风格。