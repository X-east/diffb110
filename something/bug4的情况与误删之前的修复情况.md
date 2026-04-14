# bug4 工程化总结

---

## 1. 背景

当前项目的目标架构是：

- DiffSBDD：生成模型主体

- ddpo/runtime/*：主线兼容/适配层

- RLPF/DDPO：有效训练语义

- PyTorch Lightning：主线 orchestrator

- Ray runtime：历史实现路径，不应主导当前主线

主线入口是：

- ddpo/main_diffsbdd_lightning.py

因此，对 bug4 的处理原则是：

1. 不破坏 Lightning 主线训练语义

2. 不改变 DDPO/PPO 核心优化语义

3. 不改变 trajectory / timestep / mask / reward 主语义

4. 优先让外围层服从主线 canonical 语义

5. 只做最小、局部、可逆修复

---

## 2. 问题定义

bug4 不是单点运行错误，而是一个边界与语义收口问题。

主要表现为三类：

### P1. 主线 protocol 边界不清

主线 runtime 和奖励相关模块直接依赖：

- verl_diffusion.protocol.DataProto

- verl_diffusion.protocol.TensorDict

导致主线代码与旧协议实现耦合过深。

### P2. 存在第二套 molecule validity / stability 判定路径

主线已经建立 canonical molecule semantics：

- annotate_batch_molecule_semantics(...)

- canonical legality field = rdkit_valid

但 StabilityAwareReward 原实现仍然单独重建 molecule 并重新 sanitize，且未服从 canonical pipeline。

### P3. legality 字段命名双轨混用

内部 canonical 已趋向：

- rdkit_valid

但外围仍广泛使用：

- valid

- total_valid

- valid_ratio

导致旧命名继续在导出层、展示层、兼容层中扩散。

---

## 3. 根因分析

---

### 根因 1：主线演进过程中“入口迁移了，但边界没有完全迁移”

虽然训练入口已经迁到 Lightning，但部分运行时协议、兼容字段、日志语义仍保留旧体系习惯。

结果：

- 主线看起来是新架构

- 实际上关键协议/命名边界仍依赖旧实现

---

### 根因 2：canonical molecule semantics 已建立，但外围组件未完全服从

主线已经明确规定：

- OpenBabel reconstruction → RDKit Mol

- rdkit_valid 为合法性 gate

- 不允许不同组件各自重建 molecule 形成分叉语义

但历史包装层仍保留“自己判定 validity/stability”的惯性实现。

结果：

- reward / logging / validity 口径有潜在分裂

---

### 根因 3：兼容字段没有被明确降级

valid 本应只是过渡兼容字段，但未被明确标记为“legacy only”。

结果：

- 新代码容易继续沿用旧字段

- canonical 字段 rdkit_valid 无法完全成为唯一主字段

---

## 4. 修复目标

本次 bug4 修复目标不是大重构，而是：

### G1. 收回主线协议入口归属

让主线内部优先通过 ddpo.runtime 访问 protocol，而不是到处直接引用 verl_diffusion.protocol

### G2. 消除 validity/stability 的分叉语义

让外围包装层直接依赖主线 canonical legality 结果

### G3. 将 rdkit_valid 明确为主字段

同时保留必要的 legacy fallback，避免破坏现有外部兼容面

### G4. 把 CSV / summary / 展示层逐步转向 canonical naming

但不一次性删除所有旧命名

---

# 5. 已完成修复项

---

## Fix-1：新增 protocol adapter，收口主线协议入口

### 改动

新增：

- ddpo/runtime/protocol.py

由它 re-export 当前协议实现：

- DataProto

- TensorDict

并把主线相关模块的 import 收口到：

- ddpo.runtime.protocol

### 目的

- 保持当前 RL 所需数据格式不变

- 将 protocol 的“归属边界”收回到主线代码里

- 为后续可能的协议替换保留单一入口

### 影响

- 不改变任何 batch/trajectory/reward 语义

- 仅改变 import 依赖方向

### 风险

- 低

---

## Fix-2：主线周边 protocol type hint / import 同步收口

### 涉及区域

- ddpo/runtime/*

- ddpo/rewards/* 中主线相关 type hint

- ddpo/utils/diffsbdd_filter.py type hint

### 目的

- 避免主线/主线周边继续显式暴露旧协议路径

- 让 ddpo 内部协议入口更加统一

### 风险

- 极低

- 不改运行行为

---

## Fix-3：重写 StabilityAwareReward 的语义来源

### 原问题

旧实现会：

- 自己 build molecule

- 自己 sanitize

- 形成主线之外的第二套 validity/stability 判定链

### 修复方式

现在 StabilityAwareReward 改为：

- 直接读取主线 canonical field：rdkit_valid

- 将其映射为兼容字段 stability

- 若 rdkit_valid 缺失则显式报错

### 结果

- stability 不再是竞争性语义字段

- 变为 canonical legality 的兼容暴露层

### 风险

- 低

- 语义更一致，科学正确性更高

---

## Fix-4：明确 rdkit_valid 是 canonical internal legality field

### 改动位置

- ddpo/utils/molecule_semantics.py

### 修复方式

通过接口文档/注释明确：

- rdkit_valid = 主字段

- valid = legacy alias only

### 目的

- 防止后续代码继续把 valid 当主字段使用

- 明确主线 canonical legality 口径

### 风险

- 无行为风险

- 仅提升边界清晰度

---

## Fix-5：filter 层保留兼容 alias，但显式标记为 legacy

### 改动位置

- ddpo/utils/diffsbdd_filter.py

### 修复方式

保留：

- filtered_data.batch['valid'] = filtered_rdkit_valid

但增加明确注释：

- 仅为 downstream compatibility 保留

- 内部逻辑以 rdkit_valid 为准

### 目的

- 不破坏兼容性

- 同时阻止旧字段继续冒充主字段

### 风险

- 低

---

## Fix-6：MoleculeRecorder 的 CSV 导出改为 canonical-first

### 原问题

CSV 导出主字段仍是：

- valid

### 修复方式

明细和汇总 CSV 主列改为：

- rdkit_valid

- valid_legacy

同时保持汇总统计优先读取：

- rdkit_valid

- fallback 到 valid_legacy

### 目的

- 导出层不再继续放大旧字段 valid

- 为后续分析/读取逻辑提供 canonical 列名

### 风险

- 低到中

- 外部脚本若硬编码旧列名，需同步适配

- 但保留 valid_legacy 已显著降低迁移风险

---

## Fix-7：get_summary() 增量返回 canonical summary 键

### 原问题

MoleculeRecorder.get_summary() 只返回：

- total_valid

- valid_ratio

### 修复方式

增量补上：

- rdkit_valid

- rdkit_valid_ratio

保留旧键不删。

### 目的

- 让新读取方自然使用 canonical naming

- 老读取方继续可用

### 风险

- 极低

---

## Fix-8：训练结束展示优先读取 canonical summary 键

### 改动位置

- ddpo/trainer/lightning_ddpo_module.py

- 仅 on_train_end 展示层

### 修复方式

展示读取顺序改为：

1. 优先 rdkit_valid

2. 优先 rdkit_valid_ratio

3. 若缺失再 fallback 到旧键

### 目的

- 主线展示面不再主动强化旧命名

- 同时不影响当前运行

### 风险

- 极低

- 不涉及训练计算逻辑

---

# 6. 影响范围

---

## 已影响模块

### runtime / compatibility

- ddpo/runtime/protocol.py

- ddpo/runtime/reward_wrappers.py

- ddpo/runtime/data_loader.py

- ddpo/runtime/rollout.py

### reward / type boundary

- ddpo/rewards/druglikeness.py

- ddpo/rewards/composite.py

- ddpo/rewards/docking.py

- ddpo/rewards/factory.py

### canonical legality / filter / recorder

- ddpo/utils/molecule_semantics.py

- ddpo/utils/diffsbdd_filter.py

- ddpo/utils/molecule_recorder.py

### display layer

- ddpo/trainer/lightning_ddpo_module.py

---

## 未改动主线核心区域

以下核心逻辑没有被语义性修改：

- Lightning 训练主流程

- actor update/policy optimization

- PPO/DDPO advantage / ratio / clip 语义

- rollout sampling 主逻辑

- timestep / mask / batch 语义

- reward component 加权主逻辑

- docking 主逻辑

- legacy Ray 入口

这点非常关键，说明 bug4 修复遵守了“收边界，不动主链”的原则。

---

# 7. 当前收益

---

## 收益 1：protocol 归属更清楚

主线内部不再到处显式依赖旧 protocol 路径，降低后续架构调整成本。

## 收益 2：合法性口径更统一

rdkit_valid 更接近真正 single source of truth，stability 不再是第二套独立判定系统。

## 收益 3：CSV / summary / 展示层开始向 canonical naming 迁移

后续新代码更容易沿正确字段写。

## 收益 4：兼容性保留较好

没有强制移除 valid / total_valid / valid_ratio，因此迁移风险可控。

---

# 8. 风险评估

---

## 已控制风险

### R1. 主线训练语义被误改

已避免。

没有修改核心训练步骤、优化语义、trajectory/mask/timestep 语义。

### R2. 立即破坏旧读取方

已部分避免。

保留了：

- valid

- valid_legacy

- total_valid

- valid_ratio

### R3. reward/docking 含义被静默改动

已避免。

此次没有改 reward 权重、奖励组合逻辑、docking 主路径。

---

## 仍存在的残余风险

### R4. 外部脚本可能硬编码旧 CSV 列名

如果某些外部分析工具依赖：

- valid

则需要迁移到：

- rdkit_valid

- 或 fallback 到 valid_legacy

### R5. metric 名称仍保留旧风格

例如：

- valid_molecules

- valid_ratio

虽然 source 已统一到 rdkit_valid，但命名上仍可能造成认知混淆。

### R6. legacy runtime 岛仍存在

例如：

- ddpo/main_diffsbdd.py

- verl_diffusion 旧 trainer/runtime

这些仍是独立兼容区，尚未清理。

---

# 9. 当前状态判断

---

## 结论

bug4 已从“结构性扩散问题”进入“可控兼容过渡态”。

### 可以明确认为已经解决的部分

- protocol 入口收口

- stability 分叉语义消除

- canonical legality 命名收口开始生效

- recorder/summary/展示侧开始 canonical-first

### 仍然保留但可接受的部分

- legacy alias 仍保留

- 部分 metric 名仍是旧风格

- legacy runtime 路径未动

---

# 10. 当前推荐状态描述

可以用下面这句话描述当前状态：

> bug4 的核心问题已得到显著缓解：主线协议入口、molecule legality、summary/csv/展示命名已经开始围绕 ddpo 主线和 rdkit_valid canonical 语义收口；当前仍保留少量 legacy alias 和 metric 命名作为兼容层，但已不再主导主线行为。

---

# 11. 后续建议

---

## 优先级 P1：metric naming 双写兼容

对主线周边统计结果增量补充：

- rdkit_valid_molecules

- rdkit_valid_ratio

同时保留：

- valid_molecules

- valid_ratio

### 目的

- 不破坏现有日志/监控

- 逐步推动 canonical naming 成为展示层默认值

---

## 优先级 P2：梳理外部分析脚本/文档

重点检查是否还有外部代码硬编码读取：

- valid

- total_valid

- valid_ratio

并改成：

- 优先 rdkit_valid*

- fallback 到 legacy 键

---

## 优先级 P3：长期再考虑 legacy runtime 清理

仅在需要时处理：

- ddpo/main_diffsbdd.py

- verl_diffusion 旧 runtime 直接依赖面

这个不建议现在做，以免扩大风险。

---

# 12. 最终工程结论

## 问题性质

- 不是单点报错

- 是主线边界、canonical semantics、compatibility naming 的系统性收口问题

## 修复策略

- 不动主链

- 让外围依赖主线

- canonical-first

- legacy fallback

- 小步可逆

## 当前成效

- 已完成 bug4 的高收益低风险修复

- 主线边界显著更清晰

- 剩余问题主要属于兼容层收尾，而非核心结构失控