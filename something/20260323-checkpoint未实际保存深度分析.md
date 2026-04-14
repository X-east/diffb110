# 2026-03-23 checkpoint未实际保存深度分析

## 1. 问题概述

用户观察到训练输出中出现类似：

> `[CHECKPOINT] Epoch 2: Saved last.ckpt (11.6MB)`

但实际检查输出目录后，并没有发现新的 checkpoint 文件，因此怀疑日志与实际保存行为不一致。

本报告的目标是：

1. 确认这条信息是否来自真正的 Lightning checkpoint 回调
2. 确认 checkpoint 理论保存路径与实际检查路径是否一致
3. 确认是否存在 DDP 多进程导致的输出目录错位
4. 解释为什么会出现“显示已保存，但实际上没有新 checkpoint”的现象

---

## 2. 结论先行

**结论：这条 `[CHECKPOINT] Epoch 2: Saved last.ckpt (...)` 不能被视为“本 epoch 真的成功写出了新的 checkpoint”。**

从代码与产物证据综合判断，至少存在以下三个问题叠加：

### 结论 A：epoch 结束时的“Saved last.ckpt”是**自定义打印**，不是 Lightning 官方保存成功确认
项目里这条消息来自 `LightningDDPOModule.on_train_epoch_end()` 中的手写目录扫描逻辑，而不是 `ModelCheckpoint` 回调返回的真实保存结果。

### 结论 B：这段自定义检查读取的目录来源不可靠，**可能与真正 checkpoint 回调写入目录错位**
它读取的是 `self.trainer.default_root_dir/checkpoints`，但 checkpoint 回调显式配置的是 `output_dir/checkpoints`。两者虽然有时可能恰好一致，但代码上并没有把它们强绑定为同一个确定路径。

### 结论 C：本次 2026-03-23 的 2 卡测试存在明显的**输出目录分裂/错位现象**
同一轮训练附近出现了两个时间相邻的输出目录：

- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193553/`
- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193600/`

其中：
- `193553` 目录里有 `lightning_logs/version_0/`
- `193600` 目录里有训练日志，但训练结束时 `Best checkpoint:` 和 `Last checkpoint:` 都是空字符串
- 两个目录里都**没有** `checkpoints/`

这说明至少有一部分进程/阶段写到了不同输出目录，或者主进程与子进程对 `output_dir` 的解析不一致。

因此，用户看到的“Saved last.ckpt”大概率是：

> **一个基于目录扫描的乐观/误导性状态打印，而不是当前这次 Epoch 2 真实完成写盘的证据。**

---

## 3. 证据拆解

## 3.1 训练入口里，checkpoint 回调的目标目录是 `output_dir/checkpoints`
训练入口 `ddpo/main_diffsbdd_lightning.py` 中，checkpoint 回调是这样创建的：

- `dirpath=os.path.join(output_dir, "checkpoints")`
- `save_last=True`
- `every_n_epochs=1`

这表示**理论上**每个 epoch 都应该在 `output_dir/checkpoints/` 下写 checkpoint，并维护 `last.ckpt`。

但这里的关键前提是：

1. 当前进程最终真正使用的 `output_dir` 是一致的
2. Lightning checkpoint 回调确实被正常触发
3. 没有在中途异常退出、目录漂移、进程重入等情况下被绕开

---

## 3.2 “Saved last.ckpt”并不是来自 checkpoint 回调，而是 epoch end 里自己扫描目录打印
`ddpo/trainer/lightning_ddpo_module.py` 在 `on_train_epoch_end()` 中做了如下逻辑：

1. 拼接 `checkpoint_dir = os.path.join(self.trainer.default_root_dir or "./outputs", "checkpoints")`
2. 如果目录存在，就列出其中所有 `.ckpt`
3. 排序后取最后一个文件
4. 打印：`[CHECKPOINT] Epoch {current_epoch}: Saved {latest_ckpt} ({ckpt_size:.1f}MB)`

这有两个重要含义：

### 含义 1：这不是“刚刚保存成功”的事件消息
它只是**epoch 结束时去看某个目录里是否存在 `.ckpt` 文件**。

只要那个目录里原本就有旧的 `last.ckpt`，它就可能继续打印“Saved last.ckpt”。

也就是说，这条消息的实际语义更接近：

> “我在某个目录里看到了一个叫 `last.ckpt` 的文件”

而不是：

> “当前这个 epoch 刚刚成功保存了一个新的 checkpoint”

### 含义 2：它扫描的是 `trainer.default_root_dir`，不是 checkpoint callback 的 `dirpath`
这会造成一个结构性风险：

- 真正保存路径：`output_dir/checkpoints`
- 自检路径：`trainer.default_root_dir/checkpoints`

如果二者不完全一致，就会出现：

- 回调写到 A
- 自检读 B
- 打印出的“已保存”与真实写入状态不一致

这正是当前问题最核心的嫌疑点之一。

---

## 3.3 训练结束日志直接显示：`Best checkpoint` 和 `Last checkpoint` 都是空
对于 `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193600/training_20260323_193600.log`，日志末尾是：

- `Training completed`
- `Best checkpoint:`
- `Last checkpoint:`

后两个字段为空字符串。

这意味着：

### 解释 A：Lightning 的 `ModelCheckpoint` 回调没有记录到任何有效 checkpoint 路径
如果 checkpoint 回调正常保存过，通常 `checkpoint_callback.last_model_path` 不应该为空。

### 解释 B：至少从主训练入口记录的回调对象状态看，本次训练结束时并没有拿到“最后一个 checkpoint 的真实路径”
这与“Epoch 2 显示 Saved last.ckpt”形成直接冲突。

因此，**日志内部已经自相矛盾**：

- 一边说 epoch 时“Saved last.ckpt”
- 另一边在训练结束时说 `Last checkpoint:` 为空

当两者冲突时，更可信的应该是 checkpoint 回调对象最终保存的 `last_model_path`，因为它更接近回调真实状态，而不是手工目录扫描字符串。

---

## 3.4 实际产物检查：目标输出目录里没有 `checkpoints/`
本次重点核查的两个目录：

### 目录 1：`OUT/diffsbdd-ddpo-qed-sa-test_20260323_193600/`
实际看到：
- `config.yaml`

没有看到：
- `checkpoints/`
- `lightning_logs/`

### 目录 2：`OUT/diffsbdd-ddpo-qed-sa-test_20260323_193553/`
实际看到：
- `config.yaml`
- `lightning_logs/version_0/...`

没有看到：
- `checkpoints/`

这说明：

1. 本次运行并未在显式声明的输出目录下稳定地产生完整产物结构
2. 至少 `training log`、`lightning_logs`、`checkpoints` 三类产物没有稳定落在同一个目录
3. “检查用户看到的目录没有新 checkpoint”这件事，并不是误看路径，而是**确实没有看到应该存在的 `checkpoints/last.ckpt`**

---

## 3.5 存在两个相邻时间戳输出目录，强烈提示 DDP/多进程阶段重复求值 `output_dir`
训练入口中，`output_dir` 的模板变量替换包含：

- `<run_name>`
- `<YYYYMMDD>`
- `<seq>`，其中 `<seq>` 实际被 `datetime.now().strftime("%H%M%S")` 替换

这意味着只要 `main()` 在不同时间被再次执行，就会得到不同输出目录。

当前实际观测到：

- `193553`
- `193600`

两个目录只差 7 秒，且名字完全相同，只是秒级时间戳不同。

这非常像以下情形之一：

### 可能机制 1：DDP 子进程重新进入了顶层初始化逻辑
如果多进程启动模式下，某个子进程再次执行到 `main()` 顶层，并重新计算 `datetime.now()`，那么会得到新的 `output_dir`。

### 可能机制 2：主进程/launcher 与工作进程分别生成了不同输出目录
比如：
- 主进程创建并写日志到 `193600`
- 某个实际执行 Lightning logger 的进程在 `193553`

### 可能机制 3：同一轮调试/重启留下了混杂目录
这也是可能的，但结合以下事实，单靠“残留旧目录”不足以解释全部问题：

- `193553` 有 `lightning_logs`
- `193600` 有完整训练结束日志
- `193600` 的 `Last checkpoint:` 为空

这更像是**一次训练流程内部不同阶段写到了不同 run 目录**，而不是简单的旧目录残留。

---

## 3.6 为什么会出现“Epoch 2 Saved last.ckpt (11.6MB)”这种很像真的信息？
根本原因是：

### 原因 1：消息文案设计过强，语义超过了它实际能证明的范围
代码不是在 checkpoint 回调成功后打印：

- “callback just saved file X”

而是在 epoch end 扫描目录后打印：

- “Saved {latest_ckpt} ({size})”

这会让人误以为是强因果关系，但实际上只是弱观测。

### 原因 2：它按文件名排序取“最后一个 `.ckpt`”
如果目录里原本已有：
- `last.ckpt`
- 某个旧 epoch ckpt

那么它完全可能每个 epoch 都继续打印 `last.ckpt`，即使这个文件并未在本 epoch 刷新。

### 原因 3：它没有校验 mtime、inode、hash 或 callback 返回路径
它既没有检查：

- 文件修改时间是否晚于本 epoch 开始时间
- 文件是否是新创建
- 文件大小是否发生变化
- 该路径是否等于 `checkpoint_callback.last_model_path`

所以这个打印从设计上就不能证明“新 checkpoint 已保存”。

---

## 4. 根因分层分析

## 4.1 第一层根因：**日志语义失真**
这是最直接的根因。

当前 `[CHECKPOINT] Epoch X: Saved last.ckpt (...)` 的实现方式，会把“目录里看到了一个 ckpt 文件”表达成“本 epoch 保存成功”。

这是一次**状态观察被误包装成事件确认**。

---

## 4.2 第二层根因：**checkpoint 自检目录与真实保存目录没有统一使用同一来源**
实际写入目录来自：

- `ModelCheckpoint(dirpath=os.path.join(output_dir, "checkpoints"))`

自检目录来自：

- `self.trainer.default_root_dir/checkpoints`

只要 `default_root_dir != output_dir`，就可能：

- 回调写到了 A
- 自检看的是 B

进而出现误报、漏报或看见旧文件。

---

## 4.3 第三层根因：**DDP/多进程下 output_dir 生成方式不稳**
`output_dir` 使用秒级时间戳，并在 Python 代码里运行时动态生成。

这种方式在单进程下问题不大，但在 DDP/多进程场景下存在风险：

- 如果多个进程都运行到这段逻辑，就会得到不同秒值
- 从而形成多个相邻输出目录
- 最终导致日志、logger、checkpoint 回调、自检逻辑落到不同目录

本次 `193553` 与 `193600` 的并存，就是这个问题的非常强的外部信号。

---

## 4.4 第四层根因：**训练结束后 checkpoint 回调对象没有任何保存路径，说明真正的保存链路大概率没完成**
训练末尾：

- `Best checkpoint:` 空
- `Last checkpoint:` 空

这通常意味着：

1. 回调根本没执行到有效保存
2. 或回调对象状态没有被正确维护
3. 或训练流程提前退出/异常中断到使 checkpoint 状态未更新
4. 或进程间状态与主日志进程不一致

无论是哪一种，都比“epoch end 扫描目录看到一个文件名”更接近真实保存链路。

---

## 5. 本次问题最可能的真实场景复原

综合代码与产物，我认为最可能的实际过程是：

### 场景复原（高概率）
1. 训练以 DDP 2 卡方式启动
2. 运行过程中，至少两个进程/阶段分别计算了 `output_dir`
3. 于是生成了两个相近的 run 目录：`193553` 与 `193600`
4. 某些产物（如 TensorBoard 事件文件）落在 `193553`
5. 主日志落在 `193600`
6. checkpoint 回调没有在主日志对应目录下形成有效产物
7. epoch end 的自定义检查又使用了另一个目录来源去扫描 `.ckpt`
8. 最终打印出“Saved last.ckpt”，但这并不能证明本次 epoch 在当前 run 目录下真的写出了新 checkpoint
9. 训练结束时回调对象的 `last_model_path` 为空，进一步坐实“没有可靠的新 checkpoint 保存结果”

---

## 6. 为什么用户的观察是对的

用户说：

> “事实上并没有保存新的 checkpoint”

从现有证据看，这个判断是**成立的**，至少对当前检查的 run 目录而言是成立的。

因为：

1. 目标输出目录下没有 `checkpoints/last.ckpt`
2. 训练结束日志里的 `Last checkpoint:` 为空
3. `Best checkpoint:` 也为空
4. 存在输出目录错位，导致 epoch 内打印不可信

所以，用户看到的现象不是错觉，而是代码当前 checkpoint 状态报告机制本身有误导性。

---

## 7. 风险评估

这个问题不是简单的“打印不准确”，它会进一步影响：

### 7.1 实验可复现性
如果训练者以为 epoch 2 已保存 `last.ckpt`，但实际上没有，那么：
- 中断恢复会失败
- checkpoint 路径记录会失真
- 实验追踪不可靠

### 7.2 结果归档
输出目录分裂意味着：
- `config.yaml`
- `training log`
- `lightning_logs`
- `checkpoints`

可能不在同一 run 目录，破坏一次实验的归档完整性。

### 7.3 后续分析和断点续训
如果用户依据误报去 resume：
- 可能找不到 checkpoint
- 可能误用旧 checkpoint
- 可能把一次训练和另一次训练的目录混在一起

---

## 8. 文档层冲突（需要记录）
根据规则，修改训练/路径前应先读：

- `docs/environment_info.md`
- `docs/project_paths.md`
- `docs/workflow_commands.md`
- `docs/project_context.md`

但仓库实际情况是：

- 存在 `docs/environment_info.md`
- 存在 `docs/workflow_commands.md`
- 实际文件名是 `docs/project_path.md`，**不是** `docs/project_paths.md`
- `docs/project_context.md` **不存在**

这是一个文档与规则不一致的问题，虽然不直接导致 checkpoint bug，但会影响排查时对“规范路径来源”的判断，应单独记录。

---

## 9. 建议的后续验证（不改代码前提下）

本次报告不改代码，但建议下一步按以下顺序验证：

### 验证 1：在一次全新 2 卡测试中固定输出目录
不要用 `<seq>` 动态模板，直接给定唯一 `--save_path`，观察：
- 是否仍出现多 run 目录
- `checkpoints/last.ckpt` 是否真实出现
- 训练结束时 `checkpoint_callback.last_model_path` 是否仍为空

### 验证 2：对 epoch 2 前后检查 checkpoint 目录时间戳
核对：
- `last.ckpt` 是否新建
- `mtime` 是否真的更新
- 文件大小是否变化

### 验证 3：区分“打印看到的路径”和“回调实际路径”
需要明确打印：
- `checkpoint_callback.dirpath`
- `checkpoint_callback.last_model_path`
- `trainer.default_root_dir`

如果三者不同，就能直接证明当前误报链条。

### 验证 4：检查 DDP 进程是否重复运行顶层 output_dir 生成逻辑
重点观察：
- 是否多个进程分别打印 `Output directory:`
- 是否每个进程生成不同秒级目录

---

## 10. 最终判断

### 最终判断一句话版
**`[CHECKPOINT] Epoch 2: Saved last.ckpt (11.6MB)` 这条信息不等于“epoch 2 确实成功写出了新的 checkpoint”。当前更接近“自定义目录扫描误报 + DDP 输出目录错位 + 真正 checkpoint 回调未给出有效 last path”的组合问题。**

### 最核心的三条判断

1. **这条日志不是 Lightning 官方保存成功事件，而是自定义扫描目录后的打印。**
2. **当前训练运行存在输出目录错位迹象，导致日志目录、lightning_logs 目录、checkpoint 目录不一致。**
3. **训练结束时 `Last checkpoint:` 为空，是“实际上没有可靠新 checkpoint”的强证据。**

---

## 11. 本报告使用的关键证据文件

- `ddpo/main_diffsbdd_lightning.py`
- `ddpo/trainer/lightning_ddpo_module.py`
- `configs/ddp_gpu_qed_sa_test.yaml`
- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193553/training_20260323_193553.log`
- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193600/training_20260323_193600.log`
- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193553/`
- `OUT/diffsbdd-ddpo-qed-sa-test_20260323_193600/`

---

## 12. 给用户的直接回答

如果只回答“为什么日志说保存了，但实际上没有新 checkpoint”，最准确的说法是：

> 不是 Lightning 真正告诉你“已经保存成功”，而是模块在 epoch 结束时自己去扫某个 `checkpoints` 目录，看到有 `.ckpt` 文件就打印了 `Saved last.ckpt`。这段自检逻辑读取的目录和真正回调保存目录并不严格绑定，再叠加本次 DDP 运行里输出目录出现了 `193553`/`193600` 的分裂，最终造成了“日志显示已保存，但当前 run 目录下并没有新的 checkpoint 文件”。训练结束时 `Last checkpoint:` 为空，也进一步说明这次并没有形成一个可靠的新 checkpoint 保存结果。
