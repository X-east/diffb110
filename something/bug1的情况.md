
## 一、Bug1 是什么

根据之前分析文档，bug1 的核心问题是：

2026-03-26-bug详细分析.mdLines 1347-1349

也就是说，这个 bug 不是单一一行代码错误，而是一个主链语义不统一问题：

- 有的地方在用 valid

- 有的地方在用 rdkit_valid

- 有的地方自己重新 build molecule

- 有的地方又走另一套 reward / docking 构建逻辑

- filter、reward、trainer、recorder 对“分子是否有效”的判断来源不完全一致

这会导致一个很严重的问题：

> 同一个 sample，在不同模块里可能被不同标准判断为“有效/无效”，

> 从而让 validity、reward、统计、日志、docking failure 都不再是同一个语义系统。

---

# 二、Bug1 的根因

根因可以拆成 4 类。

## 1. molecule build 没有单一真相来源

修复前，系统并不是所有地方都严格通过同一个 canonical 入口来构建 downstream molecule。

这意味着：

- validity 判断和 reward 计算可能不是同一个分子对象

- docking 前的 molecule build 和 druglikeness 用的语义不完全一致

---

## 2. “合法性字段”没有完全收口

系统里同时存在：

- valid

- rdkit_valid

这两个名字虽然很多时候表达的是同一个意思，但在代码组织上会制造歧义：

- 谁才是内部主字段？

- 谁只是兼容别名？

- 新代码应该读哪个？

如果这件事不收口，以后代码还会继续发散。

---

## 3. failure semantics 不完整

尤其是 docking 路径里，之前很多地方只是：

- 返回 False

- 或者给一个 failure penalty

但没有把失败写成明确的 failure_type，导致后续分析时难区分：

- 是 molecule 本身无效

- 还是 SDF 写出失败

- 还是 docking backend 失败

---

## 4. trainer / recorder / filter / reward 没有完全围绕同一字段闭环

即使主路径已经逐渐统一，如果：

- trainer 还能 fallback 到 valid

- recorder 还能 fallback 到 valid

- filter 还写回两套字段

- reward 仍然依赖旧语义

那整个系统还是不算“彻底统一”。

---

# 三、这次修复的目标

这次 bug1 修复的目标非常明确：

## 目标 1

建立唯一 canonical molecule semantics 入口

## 目标 2

把 rdkit_valid 收口为内部主合法性字段

## 目标 3

让 filter / reward / trainer / recorder / docking

都消费同一套 molecule semantics

## 目标 4

让 docking failure 不再只是 penalty，而是可解释的 failure type

---

# 四、这次修复过程中具体做了什么

下面按文件讲。

---

## 1. 在 ddpo/utils/molecule_semantics.py 里明确 canonical molecule semantics

这个文件现在成为整个系统的唯一标准分子语义入口。

### 1.1 统一了 canonical build 路径

核心入口是：

molecule_semantics.pyLines 99-121

它现在明确负责：

- 从 sample 中提取 geometry

- 用 OpenBabel 路径重建结构

- 得到 downstream 使用的 RDKit Mol

- 用 RDKit sanitize 做最终 legality gate

- 返回标准化 failure type

这一步的意义是：

> 后面所有模块，不该再各自重新发明一套 molecule build。

---

### 1.2 明确了 OpenBabel 和 RDKit 的分工

当前语义是：

- OpenBabel：用于 structure reconstruction

- RDKit：用于 downstream canonical Mol 和合法性判断

对应代码：

molecule_semantics.pyLines 51-67

以及：

molecule_semantics.pyLines 70-82

所以最后系统认的有效性是：

- OpenBabel 成功重建

- 且 RDKit sanitize 通过

---

### 1.3 明确了 sanitize_success 和 rdkit_valid 当前关系

我没有在这次修复里强行做复杂拆层，而是先把真实语义写清楚：

molecule_semantics.pyLines 29-37

这一步是为了避免后面的人误解：

- 以为 sanitize_success 和 rdkit_valid 是两套不同 gate

- 实际上当前实现里它们是等价的

---

### 1.4 保留 valid 兼容别名，但不再把它当内部真相

批量注入函数现在是：

molecule_semantics.pyLines 193-235

这里做了两件事：

1. 内部主字段固定为 rdkit_valid

2. valid 只作为 legacy alias 可选保留

这一步非常关键，因为它意味着：

> 从现在开始，系统内部的“真相字段”是 rdkit_valid，

> valid 只是兼容层，不再是设计中心。

---

## 2. 在 ddpo/trainer/lightning_ddpo_module.py 中把训练主链收口到 rdkit_valid

训练主链现在的流程是：

lightning_ddpo_module.pyLines 123-139

这说明训练主线已经是：

1. 先采样

2. 统一注入 canonical molecule semantics

3. filter 消费它

4. reward 消费它

这一步解决了“主链顺序和语义来源不一致”的问题。

---

### 2.1 统计逻辑只认 rdkit_valid

批量统计部分现在改成了：

lightning_ddpo_module.pyLines 59-67

修复前这里还可能 fallback 到 valid。

修复后主链内部已经明确只认 rdkit_valid。

这一步的意义是：

> 训练日志、epoch 统计、valid ratio 的定义都被收口到了一个字段上。

---

## 3. 在 ddpo/utils/molecule_recorder.py 中把记录逻辑收口到 rdkit_valid

分子记录器现在也不再 fallback 到旧字段，而是直接认 rdkit_valid：

molecule_recorder.pyLines 126-131

这一步很重要，因为以前即使训练主链收口了，如果 recorder 还依赖旧字段，最终落盘数据仍然会继续传播旧语义。

修完以后：

- CSV 记录

- 分子明细

- epoch 汇总

都以 rdkit_valid 为唯一内部来源。

---

## 4. 在 ddpo/utils/diffsbdd_filter.py 中确认 filter 只消费 canonical semantics

Filter 现在已经明确声明：

diffsbdd_filter.pyLines 1-8

以及：

diffsbdd_filter.pyLines 49-57

这说明 filter 不再自己重建 molecule，而是消费统一语义。

这一步修复的是：

- dedup / novelty 逻辑不再有独立 molecule build 分支

- filter 的有效性判定不再漂移

---

## 5. 在 ddpo/rewards/druglikeness.py 中统一到 canonical molecule build

Druglikeness reward 也明确接入 canonical build：

druglikeness.pyLines 114-128

这一步保证了：

- druglikeness 使用的 molecule

- validity gate 使用的 molecule

- trainer / filter 看到的 molecule semantics

是同一套。

也就是说，druglikeness 不再是另一条平行宇宙。

---

## 6. 在 ddpo/rewards/docking.py 中补全 docking failure semantics

这是这次修复里最关键的收尾点。

### 6.1 把 _build_mol_to_sdf() 从 bool 升级为 (success, failure_type)

现在：

docking.pyLines 527-550

以前它只是 True/False，现在它能明确告诉后续：

- 失败是不是 input failure

- 是否来自 canonical build failure

- 是否来自 SDF 导出失败

这一步直接修掉了 docking 路径里“失败不可解释”的问题。

---

### 6.2 AutoDock-GPU 路径现在会显式记录 input/backend failure

AutoDock 路径里：

docking.pyLines 561-601

backend 失败时：

docking.pyLines 603-617

以及 score 为失败结果时：

docking.pyLines 619-624

这意味着现在 docking 不再只是“失败就给 penalty”，而是：

- 前处理失败 → FAILURE_DOCKING_INPUT

- backend 执行失败 → FAILURE_DOCKING_BACKEND

---

### 6.3 UniDock 路径同样补齐了 failure semantics

UniDock 路径也做了同样的修复：

docking.pyLines 639-701

这一步保证不同 docking backend 的 failure semantics 口径一致。

---

### 6.4 smina 路径也被收成 (reward, failure_type)

单分子 smina 直接路径现在也升级了：

docking.pyLines 703-745

以及 docking backend 没有解析到 affinity 时：

docking.pyLines 759-763

这里也统一了：

- input failure

- backend failure

---

### 6.5 最终把 docking failure 写回 failure_type

最后在 calculate_rewards() 里把 docking 失败写回 batch：

docking.pyLines 814-819

这一步很关键，因为它让 failure 不再停留在局部函数返回值，而是进入全局 batch semantics。

也就是说：

> 现在训练、记录、分析都能看到 docking 的明确失败原因。

---

# 五、这次修复为什么这样设计

这次修复遵循的是最小、局部、可回退原则，没有去推倒重做。

## 原则 1：不改 reward 语义本身

这次不是去改 reward 数值定义，不是去改 PPO / DDPO 逻辑，也不是去改 diffusion timestep 语义。

只改：

- molecule build 语义来源

- validity 真相字段

- failure semantics 可解释性

这样风险最小。

---

## 原则 2：不强行一次性删光兼容层

比如 valid 没有立刻完全删除，而是缩成 legacy alias：

molecule_semantics.pyLines 227-235

这样做的好处是：

- 旧代码如果还有少量边界依赖，不会立刻全炸

- 但主链已经明确转移到 rdkit_valid

---

## 原则 3：把真相集中到 molecule_semantics.py

不是在每个模块都各写一遍，而是集中在一个 canonical 入口里。

这符合科学代码里最重要的要求：

- 语义清晰

- 可解释

- 可审计

- 不容易漂移

---

# 六、修完以后，系统语义变成什么样了

现在整个主链可以概括成下面这个流程：

也就是：

- molecule build：统一

- validity：统一

- reward 输入：统一

- failure 标注：统一

---

# 七、这次修复解决了哪些具体风险

## 1. 解决了 validity 和 reward 可能不是同一语义的问题

现在 reward 使用的 molecule build 和 validity gate 使用同一套 canonical path。

---

## 2. 解决了 trainer / recorder 统计口径可能漂移的问题

现在 trainer 统计和 recorder 落盘都直接认 rdkit_valid。

---

## 3. 解决了 docking failure 不可解释的问题

现在 docking 失败可以明确区分：

- openbabel_reconstruct_failed

- sanitize_failed

- docking_input_failed

- docking_backend_failed

---

## 4. 解决了 filter 再维护一套独立 molecule reconstruction 逻辑的问题

filter 现在只消费 canonical semantics，不再自己重建。

---

# 八、这次修复后，Bug1 可以怎么定性

我的结论是：

## Bug1 现在可以定性为：

已完成主线修复，并完成最后收尾。

更准确地说：

- canonical molecule build：已统一

- rdkit_valid：已成为内部主合法性字段

- filter / reward / trainer / recorder / docking：已接入同一语义系统

- docking failure semantics：已打通

---

# 九、还剩哪些不是 Bug1 主问题、但未来可继续优化的点

这些现在已经不是 bug1 必修项了，但可以作为后续 clean-up：

1. 是否彻底删除 valid 兼容别名

2. 是否删除/归档完全不用的 legacy 文件

3. 是否把 sanitize_success 和 rdkit_valid 做成真正分层字段

4. 是否补完整 smoke tests

这些更像后续增强，不影响本次 bug1 完成判定。

---

# 十、最终总结版

如果你要一个最简洁的归档表述，可以直接用下面这段：

## Bug1 修复总结

bug1 的本质是 molecule build / validity / reward 语义不统一。

本次修复以 ddpo/utils/molecule_semantics.py 为 canonical 入口，统一了 sample 到 downstream RDKit molecule 的构建路径，明确了 OpenBabel 负责结构重建、RDKit 负责合法性判定的职责边界；将 rdkit_valid 收口为内部主合法性字段；让 filter、druglikeness reward、docking reward、trainer 和 recorder 全部消费同一套分子语义；同时把 docking 路径中的 failure 从“仅 penalty”升级为显式 failure_type，区分 input failure 与 backend failure。至此，Bug1 的主线问题和最后收尾均已完成。