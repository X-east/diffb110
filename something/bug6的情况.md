下面给你的是一个彻底收束训练记录 / log 记录 / checkpoint 保存 / 分子落盘 / epoch 汇总的完整处理方案。

我把它按“现状诊断 → 设计目标 → 主线架构 → 模块职责 → 生命周期 → 目录协议 → 迁移顺序 → 验证标准”完整展开。

我先给一句总纲：

# 总纲结论

你这个问题的本质，不是“某个 logger 没配好”，也不是“checkpoint callback 有点小毛病”。

而是当前项目里，训练外围记录系统没有被当成一个独立的主线子系统来设计。

它现在是：

- 一部分挂在 Lightning hook 上

- 一部分挂在自定义 logger 上

- 一部分挂在 MoleculeRecorder

- 一部分设计在 BatchRecorder 里但没接主线

- 一部分约定写在文档里但代码没落实

- 一部分由 DDP 各 rank 各自生成目录，破坏了“同一次训练只有一个 run 上下文”的前提

所以要彻底解决，必须把这部分从“到处顺手写一点输出”提升为：

## 一个独立的、统一入口、统一协议、统一 run 上下文、与训练核心解耦的记录子系统

---

# 一、先明确这部分要解决什么问题

你说得很对，这部分不牵扯程序运转核心，所以最适合彻底收拾干净。

也正因为它不碰 diffusion / reward / PPO / mask 核心语义，所以可以大胆做结构整顿。

这个外围系统，实际上要覆盖 6 类记录能力：

1. run 级元数据

- 这次训练是谁、何时启动、用什么配置、输出到哪里

1. 控制台输出

- 给人看的，高信噪比

1. 文件日志

- 给排查问题看的，完整、可追溯

1. 结构化指标

- 给后分析、表格、画图、对比实验看的

1. 分子级明细

- 给化学分析、失败样本检查、奖励溯源看的

1. checkpoint 管理

- 给恢复训练、选最佳模型、记录保存状态看的

当前最大的问题是：这 6 类东西没有被分层，也没有被统一入口调度。

---

# 二、现状里最核心的结构性问题

---

## 1. 没有“唯一 run 上下文”

现在每个进程自己算 output_dir，导致 DDP 下同一次训练出现多个 run 目录。

这个问题不只是目录名错，而是会连锁破坏：

- log 落点

- molecule CSV 落点

- checkpoint 落点

- merge 逻辑

- resume 路径

- 训练后分析口径

所以第一原则必须是：

## 一次训练 = 一个唯一 run_id = 一个唯一 output_dir = 一个唯一 artifacts root

这个东西必须在训练开始前确定，而且所有 rank 共享，不能各自现算。

---

## 2. 记录逻辑散落在多个对象里，职责重叠

现在至少有：

- setup_logging

- Lightning 自带 TensorBoardLogger

- MoleculeRecorder

- BatchRecorder

- on_train_epoch_end 里的终端摘要

- ModelCheckpoint

- finally 里的训练结束打印

问题是这些东西的职责没有被严格划分。

比如：

- MoleculeRecorder 既在做 per-rank 明细，又在做 epoch 全局汇总

- BatchRecorder 设计了 batch 级 CSV，但主线没用

- epoch summary 一部分在终端打印，一部分写 epoch_summary_global.csv

- 项目文档要求 training_summary.csv 和 epochs/epoch_xxxx/...，但代码主线没实现

- checkpoint 状态既在 Lightning callback 里，又在 module hook 里自己检查

结果就是：

功能不是没有，而是互相踩边界。

---

## 3. 记录体系和文档契约脱节

项目文档已经定义了比较清晰的输出结构：

- config.yaml

- training_summary.csv

- checkpoints/

- epochs/epoch_0001/metrics.json

- epochs/epoch_0001/molecules.csv

- logs/training_*.log

- logs/molecules_*.csv

- logs/batch_stats_*.csv

- lightning_logs/version_0/...

但实际代码主线没有完整收束到这个契约上。

所以当前最危险的不是“看起来有点乱”，而是：

## 文档、代码、运行结果，三者没有统一协议

只要协议不统一，后面任何分析脚本都要靠猜。

---

## 4. 事件时序没有抽象出来

外围记录系统最怕“时序不清”。

现在典型例子就是：

- epoch 刚结束

- 你在 on_train_epoch_end 里检查 checkpoint

- 但 Lightning 的 checkpoint callback 可能尚未完成路径更新或落盘确认

这说明当前代码把“记录点”理解成了“随便找个 hook 打印一下”。

但真正稳健的做法应该是：

## 每种产物必须绑定到明确的生命周期事件

比如：

- run started

- batch finished

- epoch metrics reduced

- epoch artifacts merged

- checkpoint saved

- run ended

- run failed

- run interrupted

没有事件抽象，记录系统永远会充满“看起来差不多”的错误时机。

---

# 三、目标架构：把外围记录系统收束成一个独立主线

我建议把整个外围系统设计为：

## Run Recording Mainline

一个围绕“训练 run 生命周期”的独立子系统

它和训练核心的关系是：

- 训练核心负责产生事实

- 样本

- reward

- batch metrics

- epoch metrics

- checkpoint 事件

- 记录主线负责消费这些事实并持久化

- 不参与训练决策

- 不修改 reward 语义

- 不改 diffusion / mask / PPO 逻辑

- 只做标准化记录和产物组织

也就是说：

## 训练核心 = 事实生产者

## 记录主线 = 事实归档者

这就是你要的“解耦，统一调用”。

---

# 四、设计原则

下面这 10 条，是我认为必须坚持的原则。

---

## 原则 1：唯一 run 上下文

训练一启动，立刻构造一个 RunContext，包括：

- run_id

- run_name

- start_time

- output_dir

- logs_dir

- epochs_dir

- checkpoints_dir

- lightning_logs_dir

- config_snapshot_path

- summary_csv_path

并且：

- 只生成一次

- 所有 rank 共享

- 所有记录器都从这里取路径

- 不允许任何模块自己再拼 output_dir

---

## 原则 2：统一事件总线，不允许随地写文件

以后不要让：

- LightningDDPOModule

- MoleculeRecorder

- BatchRecorder

- main_diffsbdd_lightning.py

- 各种 callback

各自决定“我现在写个 CSV / print 一句 / 检查一下路径”。

应该改为：

## 所有外围记录都走统一调用入口

例如概念上：

- record_run_started(...)

- record_batch_end(...)

- record_epoch_end(...)

- record_checkpoint_saved(...)

- record_run_end(...)

- record_run_failed(...)

这样训练代码只负责上报事件，不负责决定怎么落盘。

---

## 原则 3：人读输出、机器读输出分离

必须明确区分：

### A. 人读输出

- 终端简报

- training_*.log

### B. 机器读输出

- training_summary.csv

- epochs/epoch_xxxx/metrics.json

- epochs/epoch_xxxx/molecules.csv

- checkpoint manifest

人读输出追求简洁，机器读输出追求稳定 schema。

现在最大问题之一是：终端摘要、CSV、log 混成一锅。

---

## 原则 4：全局聚合和本地明细分层

DDP 下必须明确两种东西：

### 本地 rank 事实

- 当前 rank 生成了哪些分子

- 当前 rank 的 batch 明细

### 全局事实

- 本 epoch 全局生成数

- 本 epoch 全局有效率

- 本 epoch 全局 reward mean

- 本 epoch 全局 checkpoint 状态

不要让一个文件同时承担“本地中间数据”和“全局最终结论”。

---

## 原则 5：checkpoint 只记录事实，不猜测时机

不要再在 on_train_epoch_end 里推断：

- “应该保存了吧”

- “last_model_path 应该有了吧”

应该改成：

## 只有在 checkpoint callback 真正确认保存后，才发出 checkpoint saved 事件

否则只记录：

- epoch 结束

- checkpoint pending / not attempted / skipped by interval

不要把“尚未确认”伪装成“保存失败”。

---

## 原则 6：目录结构是协议，不是实现细节

目录结构必须成为“主线契约”。

所有外部分析、resume、人工检查都依赖它稳定。

所以以后不能出现：

- 某些 run 只有 molecules/

- 某些 run 只有 lightning_logs/

- 某些 run 没 logs/

- 某些 run 才有 epoch_summary_global.csv

必须统一。

---

## 原则 7：结构化产物优先于自由文本

日志可以变，文案可以变，但结构化文件必须稳定。

所以后分析和审计应该优先依赖：

- CSV

- JSON

- manifest

而不是 grep 训练日志。

---

## 原则 8：记录系统不能反向污染训练语义

因为这是外围系统，所以必须守住边界：

- 不能因为记录需要改变 reward 计算路径

- 不能因为记录需要改变 batch 语义

- 不能因为 checkpoint 记录而改变优化逻辑

- 不能因为分子落盘而改变 molecule canonical path

记录系统只能读取训练事实，不能反向影响训练行为。

---

## 原则 9：失败样本也要记录，但明确标注失败类型

这是科学代码最重要的一点。

分子明细里不能只写成功样本。

必须显式保留：

- build_failed

- sanitize_failed

- docking_input_failed

- docking_backend_failed

- success

否则后面分析 reward 或有效率时，永远会被“静默跳过”污染。

---

## 原则 10：先统一主线，再谈扩展

不要先做什么“支持好多后端 logger”“支持很多可选 schema”。

先把主线收束成唯一协议，再考虑扩展。

---

# 五、我建议的最终主线结构

---

## 1. 顶层：RunContext

这是整个记录系统的根对象。

职责：

- 生成并冻结本次训练唯一标识

- 管理所有标准路径

- 保存 config snapshot

- 暴露只读路径给所有 recorder / writer / callback

它应该是唯一真源，不允许各模块自行拼路径。

### 必须包含

- run_id

- run_name

- started_at

- output_dir

- logs_dir

- epochs_dir

- checkpoints_dir

- lightning_logs_dir

- training_summary_csv

- config_snapshot_path

- run_manifest_path

---

## 2. 第二层：RunRecorder

这是统一调用入口。

训练代码不直接写文件，而是统一调用它：

- on_run_start

- on_batch_end

- on_epoch_end

- on_checkpoint_saved

- on_run_end

- on_run_error

- on_run_interrupted

RunRecorder 内部再调具体 writer。

这就是“统一调用”。

---

## 3. 第三层：按职责拆 writer，不按历史文件名拆

我建议拆成 5 个 writer：

### A. ConsoleReporter

职责：

- 终端高信噪比输出

- 只做人读摘要

- 不写 CSV/JSON

### B. TrainingLogger

职责：

- 写 logs/training_*.log

- 完整记录生命周期事件

- 记录异常、栈、警告、路径审计信息

### C. MetricsWriter

职责：

- 写 training_summary.csv

- 写 epochs/epoch_xxxx/metrics.json

### D. MoleculeWriter

职责：

- 写 rank 本地明细

- 合并全局分子 CSV

- 写 epoch 级 molecules.csv

- 必须支持 failure_type / rdkit_valid / reward components

### E. CheckpointTracker

职责：

- 跟踪 checkpoint callback 发出的真实保存事件

- 维护 checkpoint manifest

- 给终端输出“已确认保存”

- 绝不提前猜测

这样拆分后，每个组件职责单一，后面维护不容易互相污染。

---

# 六、目录结构应该如何彻底统一

我建议严格收束为下面这个主线结构：

---

## 为什么这样设计

### training_summary.csv

- 一行一个 epoch

- 是整个 run 的主 summary

- 后分析第一入口

### epochs/epoch_xxxx/metrics.json

- 存本 epoch 详细指标

- 适合保留更丰富字段

- 比 CSV 更利于扩展

### epochs/epoch_xxxx/molecules.csv

- 本 epoch 全局分子明细

- 后分析按 epoch 查最方便

### epochs/epoch_xxxx/molecules_rank*.csv

- 保留原始 rank 级事实

- 便于审计 merge 是否正确

### checkpoints/checkpoint_manifest.csv

- 明确记录每个 checkpoint 的 epoch、路径、大小、是否存在、保存时间

- 以后 resume / 选最佳 / 审计都靠它

---

# 七、每种文件的唯一职责

这是彻底解耦的关键。

---

## 1. config.yaml

职责：

- 本次运行配置快照

- 不更新、不追加

---

## 2. run_manifest.json

职责：

- 记录 run 级元信息

- 如：

- run_id

- host

- world_size

- seed

- start_time

- end_time

- status

- command

- git/repo信息（如可得）

- output_dir

它是训练 run 的“身份证”。

---

## 3. logs/training_*.log

职责：

- 完整文本日志

- 记录生命周期、告警、异常、路径信息

- 用于 debug

不承担结构化数据源职责。

---

## 4. training_summary.csv

职责：

- epoch 主汇总

- 一行一个 epoch

- schema 稳定

我建议字段至少包括：

- epoch

- timestamp

- avg_loss

- ppo_loss

- clip_frac

- generated_total

- rdkit_valid

- rdkit_valid_ratio

- reward_mean

- reward_min

- reward_max

- qed_mean

- sa_mean

- dock_mean

- druglikeness_mean

- build_failed

- sanitize_failed

- docking_input_failed

- docking_backend_failed

- checkpoint_saved

- checkpoint_path

---

## 5. epochs/epoch_xxxx/metrics.json

职责：

- 存更细粒度 epoch 数据

- 可包含：

- global metrics

- per-rank contribution

- reward component means

- failure counts

- advantage stats

- timing stats

- checkpoint state

这个文件可以比 CSV 更丰富。

---

## 6. epochs/epoch_xxxx/molecules.csv

职责：

- 当前 epoch 全局分子明细

- 是后分析主入口

- 不必去 molecules_all.csv 里再筛 epoch

字段建议至少包括：

- epoch

- rank

- batch

- molecule_id

- sample_id

- smiles

- canonical_smiles

- rdkit_valid

- failure_type

- qed

- sa_raw

- sa_norm

- druglikeness_reward

- dock_score_raw

- dock_score_norm

- docking_reward

- reward_total

---

## 7. epochs/epoch_xxxx/molecules_rank*.csv

职责：

- 原始局部事实

- 便于审计分布式合并

- 如果全局 merge 出问题，可以追踪来源

---

## 8. checkpoints/checkpoint_manifest.csv

职责：

- 真实 checkpoint 台账

字段建议：

- epoch

- step

- kind (epoch, last, best)

- path

- exists

- size_bytes

- monitor

- monitor_value

- saved_at

- global_rank

- confirmed

这个文件能彻底终结“到底保存没保存”的混乱。

---

# 八、统一调用方式应该长什么样

训练主线以后应该只做“上报”，而不是直接写文件。

---

## 训练入口阶段

在 main_diffsbdd_lightning.py：

### 只做

- 创建 RunContext

- 初始化 RunRecorder

- 把它注入 Lightning module / callback / trainer 周边

- 把 output_dir 固化给所有进程

### 不再做

- 到处自己拼路径

- 自己额外打印“Training completed”

- 自己直接推断 checkpoint 状态

---

## LightningDDPOModule 阶段

### 只负责上报事件

例如概念上：

- batch 完成 -> 上报 batch summary 和分子批次事实

- epoch 聚合完成 -> 上报 epoch metrics

- 训练结束 -> 上报 run end

### 不再直接负责

- 自己决定写哪个 CSV

- 自己 merge 文件

- 自己在 epoch end 检查 checkpoint path

- 自己拼 summary 文案

---

## checkpoint callback 阶段

### 只负责

- 当 checkpoint 真正保存成功时，发出 checkpoint_saved 事件

### 不负责

- 训练摘要

- 分子汇总

- run 结束打印

---

# 九、DDP 下的正确数据流

这是你现在最需要收束清楚的一块。

---

## 阶段 1：每个 rank 写本地临时事实

每个 rank 在自己共享 run 目录下，写：

- epochs/epoch_0001/molecules_rank0.csv

- epochs/epoch_0001/molecules_rank1.csv

注意：

- 这时候所有 rank 都在同一个 run 目录

- 只是文件名区分 rank

---

## 阶段 2：epoch 末做全局数值 reduce

数值型指标通过分布式 reduce 聚合，得到：

- generated_total

- valid_total

- reward_mean

- avg_qed

- avg_sa

- failure counts

- reward component means

这些是全局数值真相。

---

## 阶段 3：rank0 读取本 epoch 各 rank 文件，做全局 molecules 合并

rank0 只在所有 rank 文件都 flush 且 barrier 后执行合并：

- 输出 epochs/epoch_0001/molecules.csv

这个文件和 epoch metrics 配套。

---

## 阶段 4：rank0 写 epoch metrics 和 training_summary.csv

- epochs/epoch_0001/metrics.json

- training_summary.csv 追加一行

---

## 阶段 5：checkpoint callback 真正保存时，再写 checkpoint manifest

注意顺序：

- epoch ended ≠ checkpoint saved

- checkpoint 是单独事件

这样以后终端就能明确区分：

- epoch 已完成

- checkpoint 已确认保存 / 本 epoch 不触发保存 / 保存失败

---

# 十、checkpoint 应该如何彻底收束

现在 checkpoint 的核心问题不是保存逻辑本身，而是状态表达混乱。

我建议把 checkpoint 状态分成 4 类，不能混写。

---

## 1. not_due

本 epoch 按 interval 不该保存

例如 checkpoint_interval=5，epoch 1 就不该保存

---

## 2. pending

epoch 已结束，但 checkpoint callback 尚未完成确认

---

## 3. saved

已确认存在文件

- path

- size

- time

- type (last / best / epoch)

---

## 4. failed

尝试保存但失败

- exception

- path

- time

---

## 为什么这样重要

因为你现在把：

- pending

误报成了

- failed-like

终端看到 last_model_path is empty，自然以为坏了。

而正确表达应该是：

- Epoch 1 completed

- Checkpoint: pending confirmation

或者

- Checkpoint: not_due

或者

- Checkpoint: saved -> .../last.ckpt

---

# 十一、MoleculeRecorder 和 BatchRecorder 怎么处理

这个地方一定要定主线，不能继续双轨。

---

## 我的建议：保留思想，重构归并，不保留并行主线

### MoleculeRecorder

保留它负责“分子级明细”的部分，但不要再让它兼任：

- epoch summary

- run summary

- 全局 checkpoint 口径

### BatchRecorder

如果你确实需要 batch 级统计，就让它只做：

- logs/batch_stats_<ts>.csv

但不要让它再写另一套“分子主明细”。

---

## 最合理的收束方案

### A. MoleculeWriter

统一负责所有 molecule-level 数据

- per-rank raw

- per-epoch merged

- failure_type

- reward components

### B. BatchStatsWriter

只负责 batch 级统计

- logs/batch_stats_*.csv

### C. MetricsWriter

只负责 epoch / run summary

这样就解开了：

- 分子明细

- batch 统计

- epoch 汇总

三者的职责纠缠。

---

# 十二、控制台输出应该怎么定义

终端输出绝不能继续“谁想到哪句就打印哪句”。

我建议终端只保留 6 类消息：

1. run start

2. epoch progress

3. epoch summary

4. checkpoint status

5. run end

6. run fail/interrupted

而且每类都固定模板。

例如：

### run start

- run_id

- output_dir

- world_size

- batch size

- max epochs

### epoch summary

- generated_total

- rdkit_valid_ratio

- reward_mean

- qed_mean

- sa_mean

- failure counts

### checkpoint status

- saved

- not_due

- pending

- failed

### run end

- final checkpoint

- best checkpoint

- summary path

- molecules path

这样控制台就会非常干净。

---

# 十三、日志系统应该如何收口

日志体系建议只保留三层：

---

## 1. Console

- 面向人

- 高信噪比

- 只打印摘要

---

## 2. training_*.log

- 面向 debug

- 完整文本日志

- 所有生命周期事件

- 完整异常和 stack

---

## 3. 结构化文件

- CSV / JSON / manifest

- 面向程序分析和审计

---

## 不要做的事

- 不要让 CSV 充当日志

- 不要让训练日志成为主 summary 数据源

- 不要让 TensorBoard 成为唯一实验记录真源

TensorBoard 是辅助，不是审计主源。

---

# 十四、与 Lightning 的解耦边界怎么定

你说“统一调用，也要解耦”，这个点非常关键。

我的建议是：

## Lightning 只负责触发生命周期事件

## 记录主线负责把事件写成产物

也就是：

### Lightning module / callback 应该知道

- 什么时候一个 batch 完了

- 什么时候一个 epoch 真结束了

- 什么时候 checkpoint 真保存了

### 但 Lightning module / callback 不应该知道

- 文件夹结构怎么设计

- CSV schema 是什么

- JSON 字段怎么组织

- 终端文案怎么统一

- training_summary.csv 如何追加

这部分必须下沉到独立 recorder 子系统。

---

# 十五、推荐的模块边界

我建议最终代码结构大概变成：

---

## 各模块职责

### run_context.py

- 生成唯一 run_id

- 固化 output_dir

- 暴露标准路径

### run_recorder.py

- 统一入口

- 按事件分发给各 writer

### console_reporter.py

- 终端模板输出

### training_logger.py

- 文本日志

### metrics_writer.py

- training_summary.csv

- epochs/epoch_xxxx/metrics.json

### molecule_writer.py

- rank 原始分子文件

- epoch 全局分子文件

- merge

### checkpoint_tracker.py

- checkpoint manifest

- 状态确认

### schemas.py

- 统一字段定义

- 避免各处手写 CSV header

---

# 十六、迁移策略：怎么在不碰训练核心的前提下彻底替换

这部分一定要分阶段，不要一次全推翻。

---

## Phase 1：先收束 run 上下文

目标：

- 一次训练只产生一个 output_dir

- 所有 rank 共用

这是第一优先级。

不做这一步，后面全部白搭。

---

## Phase 2：引入 RunContext + RunRecorder

目标：

- 所有外围记录从统一入口走

- 训练核心不再直接写文件

这一步仍然不碰 reward / diffusion / PPO 核心。

---

## Phase 3：统一 summary 主线

目标：

- 落地 training_summary.csv

- 落地 epochs/epoch_xxxx/metrics.json

- 明确终端摘要来源与结构化 summary 一致

---

## Phase 4：统一 molecule 主线

目标：

- rank 原始文件

- epoch 全局合并文件

- 统一字段 schema

- failure_type 显式记录

---

## Phase 5：统一 checkpoint 主线

目标：

- checkpoint manifest

- 正确 lifecycle

- 明确 saved/not_due/pending/failed

---

## Phase 6：移除旧分散逻辑

目标：

- 删除或降级：

- 旧的 MoleculeRecorder 汇总逻辑

- 未接主线的 BatchRecorder 分子输出

- scattered console prints

- 模块内随手写 summary

这一步完成后，外围系统就真正收束了。

---

# 十七、必须避免的坏模式

为了“彻底解决”，下面这些模式要明确禁掉。

---

## 坏模式 1：模块内部自己拼 output 路径

必须禁止。

---

## 坏模式 2：在多个地方各写一份 epoch summary

必须禁止。

---

## 坏模式 3：把 checkpoint 状态从路径字段反推

必须禁止。

要由 checkpoint 事件驱动，不要靠猜。

---

## 坏模式 4：把全局 CSV merge 建立在“也许所有 rank 文件都在同目录”的隐含前提上

必须禁止。

要先保证统一 run context，再 merge。

---

## 坏模式 5：人读输出和机器读输出共用一套逻辑

必须禁止。

一套简报，一套结构化 schema。

---

## 坏模式 6：finally 无条件打印 completed

必须禁止。

要区分：

- completed

- interrupted

- failed

---

# 十八、这个外围系统最终应该达到的效果

彻底收束后，你应该得到这样的体验：

---

## 训练开始时

- 终端只打印一次统一 run 信息

- 明确唯一 output_dir

- 所有 rank 写入同一 run 目录

---

## epoch 结束时

- 终端显示全局 summary

- training_summary.csv 追加一行

- epochs/epoch_xxxx/metrics.json 写好

- epochs/epoch_xxxx/molecules.csv 写好

- checkpoint 若该保存则后续收到明确 saved 事件

---

## 训练中断时

- run_manifest.json 标记 status=interrupted

- 日志明确结束原因

- 已完成 epoch 产物完整可读

- checkpoint 状态清晰

---

## 训练失败时

- run_manifest.json 标记 status=failed

- 日志保留 stack

- 已写产物不被误标“completed”

---

## 训练结束后

你只需进入一个 run 目录，就能找到：

- 配置

- 完整日志

- 主 summary

- 每个 epoch 的详细 metrics

- 每个 epoch 的分子明细

- checkpoint 台账

- TensorBoard 日志

这才叫“主线化”。

---

# 十九、建议的验收标准

如果要说“彻底解决”，我建议按下面标准验收。

---

## A. 单次 2 卡训练验收

必须满足：

- 只产生一个 run 目录

- rank0/rank1 文件都在同一 run 下

- training_summary.csv 正常追加

- epochs/epoch_0001/molecules.csv 含全局 2 卡样本

- checkpoint 状态表达清晰

---

## B. 中断恢复验收

- 手动中断后，run status = interrupted

- 已完成 epoch 记录完整

- checkpoint manifest 状态真实

- resume 路径明确

---

## C. 异常失败验收

- 训练异常退出时不打印 completed

- run_manifest 标记 failed

- training log 有 stack

- 已有结构化文件不损坏

---

## D. 文档一致性验收

- 实际输出结构和文档完全一致

- 不再需要“这版代码其实看另一个目录”

---

## E. 数据一致性验收

以下口径必须一致：

- 终端 epoch summary

- training_summary.csv

- epochs/epoch_xxxx/metrics.json

- epochs/epoch_xxxx/molecules.csv 统计得到的 generated/valid count

如果不一致，就说明主线仍未收束成功。

---

# 二十、风险评估

这次整顿虽然属于外围系统，但也有两个风险点要提前说清：

---

## 风险 1：旧分析脚本可能依赖旧路径

比如它们可能还读：

- molecules_all.csv

- epoch_summary_global.csv

- 某些旧 logs/*.csv

所以迁移时最好有一个过渡期：

- 新主线落地

- 旧文件暂时兼容输出一版

- 后续再完全移除

---

## 风险 2：Lightning 生命周期理解不清会导致新一轮时序问题

特别是 checkpoint。

所以一定要把“epoch end”和“checkpoint saved”拆成两个事件，不要混。

---

# 二十一、我给你的最终方案摘要

如果压缩成一句设计要求，就是：

## 用“唯一 RunContext + 统一 RunRecorder + 分职责 Writer + 明确生命周期事件 + 稳定目录协议”

## 把训练外围记录系统收束成一个独立主线，彻底替代当前散落在入口、module、recorder、callback 各处的分散输出逻辑。

---

# 二十二、我建议的实际落地顺序

如果你下一步要我直接动手，我建议严格按这个顺序改：

1. 先修 output_dir 全局统一

2. 引入 RunContext

3. 把 training_summary.csv 和 epochs/epoch_xxxx/metrics.json 立起来

4. 重构 molecule 记录为唯一主线

5. checkpoint 改成事件确认式记录

6. 删除旧散落打印和旧汇总路径

7. 最后更新文档，和代码完全对齐

---

如果你愿意，下一步我可以直接继续做两件事里的一个：

### 方案 A：我先给你出一份“精确到模块/类/函数”的重构蓝图

会列出：

- 新增哪些文件

- 现有哪些类保留/废弃

- 每个 hook 改成什么调用

### 方案 B：我直接开始改代码

按照上面的顺序先做第一阶段：

- 全局唯一 output_dir

- 统一 run context

- 统一 summary 主线

如果你让我选，我建议：直接走 B，先把主线立起来。