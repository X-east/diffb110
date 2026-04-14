## 一、Bug 5 的本质到底是什么

Bug 5 表面上看是“多 backend、多时代遗留堆叠”的问题，但深入看，真正的问题不是 backend 数量本身，而是：

1. DockingReward 同时承担了太多职责

- backend 选择

- ligand 输入准备

- backend 执行

- score 解析

- reward 归一化

- failure 写回

1. 多个 backend 的差异被直接塞进 reward 层

- autodock_gpu

- unidock

- smina

1. legacy/backend 选择逻辑没有在结构上降权

- 注释里说主线是 autodock_gpu

- 但代码结构上仍然容易让人误以为三者平权

1. failure 语义不够显式

- 很多地方用 0.0 间接表示失败

- reward 层需要反推 backend 是否失败

- 输入失败和后端失败边界不够清楚

所以 Bug 5 的真正问题可以概括为一句话：

> docking 这条链路缺少清晰分层，reward 层被 backend/IO/legacy 复杂度污染。

---

# 二、这次修复的总原则

这次修复不是“大重构主线”，而是围绕主线收束做结构解耦。

核心原则始终没变：

1. 不破坏主线收束

- 主入口仍然是 ddpo/main_diffsbdd_lightning.py

- 不让 legacy Ray 路线回流主线

1. 不改 canonical molecule semantics

- 继续保持 canonical rdmol 为 downstream 单一分子对象

- 继续保持 rdkit_valid 是 docking gate

1. 不改 reward 数学语义

- docking reward 仍然是 score_raw / normalization_factor

- failure 仍然给 failure_penalty

1. 只拆结构，不改主线实验含义

- backend 选择从 reward 层搬出去

- 输入准备从 reward 层拆出去

- backend 返回改成结构化，但兼容原有浮点语义

---

# 三、修复后的主线结构变化

---

## 1. factory.py 现在成为唯一 reward/backend 装配点

这是第一性修复。

现在 ddpo/rewards/factory.py 不只是“顺手构造一下 reward”，而是变成了唯一的 docking backend 策略中心。

这里最关键的变化是显式 registry：

factory.pyLines 14-69

这段代码的意义非常大：

- backend 选择被集中化

- support level 被显式建模

- execution mode 被显式建模

- 构造逻辑和策略逻辑分开了

以前是 if/elif 逻辑分支，虽然也能跑，但“结构地位”不够清晰。

现在是：

- autodock_gpu：主线 supported

- unidock：legacy

- smina：debug

这就把“主线是谁”从注释变成了架构事实。

---

## 2. DockingReward 不再负责 backend 选择

这是第二个核心修复。

现在 DockingReward 构造函数已经不再接收 backend="autodock_gpu" 然后自己做分支，而是直接接收注入好的 backend 实例：

docking.pyLines 49-80

这意味着：

- reward 层只知道“一个 backend 实现”

- reward 层不再决定 backend

- backend 选择权已经完全收束到 factory

这一步是真正实现“解耦而不打散主线”的关键。

---

## 3. ligand preparation 被独立成层

第三个核心修复是把输入准备抽到 docking_preparation.py。

现在这个模块只负责：

- canonical 分子构建

- SDF 导出

- batch 输入准备

- 单样本输入准备

- 清理临时文件

核心结构如下：

docking_preparation.pyLines 20-38

以及：

docking_preparation.pyLines 43-64

这一步的价值在于：

- reward 层不再背输入文件细节

- backend 层不再自己构建 canonical molecule

- 输入失败有了独立边界

也就是说，输入准备失败和backend 执行失败已经被结构性分开。

---

# 四、backend 层现在是怎么被收敛的

---

## 1. backend 接口从“裸分数”升级为结构化结果

这个变化是第三阶段最关键的一步。

在 ddpo/rewards/docking_backends/base.py 中新增了：

base.pyLines 9-22

以及新的抽象接口：

base.pyLines 29-56

同时保留了兼容接口：

- dock_single() -> float

- dock_batch() -> list[float]

这意味着：

- 新主线消费结构化结果

- 旧调用方式仍然不至于一下子崩掉

这是非常稳妥的解耦策略。

---

## 2. AutoDockGPUBackend 被清晰定义为主线 supported backend

现在它的地位已经写死为：

autodock_gpu.pyLines 50-54

并且它现在返回结构化结果，比如：

- ligand 转换失败

- 非零退出

- score parse 失败

- timeout

- backend exception

例如：

autodock_gpu.pyLines 160-170

这比之前“失败就返回 0.0”清楚得多。

---

## 3. UniDockBackend 被降级为 legacy，但保留为显式可选项

现在它的结构地位是：

unidock.pyLines 17-21

这意味着：

- 没有删功能

- 但它已经不是“主线等价候选”

这正符合你一开始要求的：

> 可选，但不要揉在一起，要解耦，而且主线保持清晰。

---

## 4. SminaBackend 被进一步降为 debug-oriented backend

现在它的结构定位是：

smina.pyLines 9-13

这比“legacy CPU backend”更进一步：

- 它不再被暗示成正常生产路径

- 它更像调试/应急路径

这个降级是正确的，因为当前项目方向已经很明确是 GPU docking 主线。

---

# 五、DockingReward 现在真正只剩 reward 语义了

经过前几阶段后，DockingReward 的职责已经明显收缩。

---

## 1. 它的核心职责现在是什么

### A. 调 preparation 层拿输入

batch 路径：

docking.pyLines 132-140

### B. 调 backend 结果接口

docking.pyLines 150-154

### C. 把结构化结果映射成 reward / failure

docking.pyLines 123-131

### D. 写回 batch

docking.pyLines 256-262

这说明 reward 层已经不再承担 backend 策略、文件组织、分子构建等不属于它的职责。

---

## 2. 主线语义有没有被改坏？

目前看，没有。

保留下来的关键主线语义包括：

1. rdkit_valid == 0 仍然是 docking gate

2. canonical rdmol -> SDF 路线没变

3. 成功时仍然是 score_raw / normalization_factor

4. 失败时仍然是 failure_penalty

5. failure_type 仍然只在原值是 success 时覆盖

所以这次修复是结构性修复，不是实验语义修复。

---

# 六、factory 层现在已经变成清晰的策略中心

第四阶段以后，factory 的职责非常明确。

---

## 1. backend policy 已经显式化

_build_docking_backend(...) 不再只是“构造一个对象”，而是在执行 policy：

factory.pyLines 90-118

这段代码的意义在于：

- invalid backend 会明确报错

- 非主线 backend 会明确 warning

- support level 成为运行时可见事实

---

## 2. 这使主线认知负担显著下降

现在读代码时，开发者不用再问：

- “到底哪个 backend 是当前推荐？”

- “是不是三种都平权？”

- “主线到底默认走什么？”

因为 factory 已经把这个问题结构性回答了。

---

# 七、测试层现在已经补上了哪些“结构保护网”

第五阶段的重要性在于：

它不是继续改逻辑，而是把前面四阶段的边界锁住。

---

## 1. factory policy regression

新增测试：

- test/test_docking_factory_policy.py

它锁定：

- 只有 autodock_gpu 是 supported

- _build_docking_backend(...) 确实按 registry 走

- legacy/debug backend 地位不会回退

- invalid backend 报错不会模糊

这保证 future edits 不会偷偷把 policy 再写散。

---

## 2. preparation layer regression

新增测试：

- test/test_docking_preparation.py

它锁定：

- batch 准备的成功/失败分流

- single 准备失败时临时文件清理

- single 准备成功时 cleanup 可用

这保证输入准备层边界稳定。

---

## 3. DockingReward 行为 regression

新增测试：

- test/test_docking_reward_behavior.py

它锁定：

- reward 归一化写回

- backend failure 写回

- rdkit_valid gate 行为

- preparation cleanup 行为

这基本把 reward 层最关键的主线语义保护住了。

---

# 八、这次修复完成后的最终收益

---

## 收益 1：主线更清楚

现在 docking 这条链路已经分成了：

1. factory.py：唯一策略/装配点

2. docking_preparation.py：输入准备层

3. docking.py：reward 语义层

4. docking_backends/*：backend 执行层

这是一个非常清楚的分层。

---

## 收益 2：legacy backend 不再污染主线

以前虽然文档上说 autodock_gpu 是主线，但代码结构上并没有“强表达”。

现在已经有了：

- supported

- legacy

- debug

以及 MAINLINE_DOCKING_BACKENDS == {"autodock_gpu"}。

所以主线真正收束了。

---

## 收益 3：failure 边界更清楚

现在已经能显式区分：

- 输入准备失败

- backend 执行失败

- backend 具体失败原因

这对于训练日志、调试、统计都是非常有价值的。

---

## 收益 4：后续继续演进时风险降低

因为已经有了回归测试，未来你再做：

- 第六阶段

- 更深层 logging

- legacy backend 进一步软封存

风险都会小很多。

---

# 九、当前仍然保留的边界与未完成项

虽然 Bug 5 的主问题已经基本解决，但还有一些“不是这次必须做、但未来可继续做”的点。

---

## 1. failure_type 还没有做统一枚举层

当前 backend failure 已经能写成：

- docking_backend:timeout

- docking_backend:score_parse_failed

这是好的，但还没有提升成一个统一的、项目级的枚举/常量层。

未来如果继续做第六阶段，可以统一一下 failure 命名约定。

---

## 2. 目前测试以回归/单测为主，还没做真实轻量 integration

现在的测试主要锁结构边界，非常对路。

但如果你后面要继续推进稳定性，可以加：

- 一个只验证 factory + reward + preparation 串起来的轻量 integration test

- 不必跑真实 GPU docking，但可以做假 backend 的端到端 smoke test

---

## 3. 文档里的 tests_root 与实际目录存在小冲突

文档写的是 tests/，实际当前仓库存在的是 test/。

本次为了最小改动，我把新增测试写到了实际存在的 test/ 下。

这个不是核心 bug，但后面可以顺手统一。

---

# 十、最终结论

如果把 Bug 5 的目标定义为：

> 在不破坏原有主线收束的前提下，把 docking 相关的多 backend / 多时代堆叠从 reward 主线中解耦出来，并让主线默认路径保持清晰、单一、可审计。

那么这次修复已经基本达成目标。

## 现在的最终状态可以概括为：

- 主线入口没变

- canonical molecule semantics 没变

- reward 数学语义没变

- backend 选择被收束进 factory

- 输入准备被独立成层

- backend 返回结构化结果

- legacy/debug backend 被结构性降权

- 回归测试已经覆盖策略层、准备层、reward 层

所以从工程角度看，Bug 5 已经从“多 backend 多时代堆叠型 bug”变成了：

> 一个有清晰 policy、有清晰 preparation、有清晰 reward layer、有清晰 backend layer 的可维护结构。