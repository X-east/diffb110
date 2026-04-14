# Bug Cluster 5：多 backend / 多时代堆叠型 bug 深度分析

我这次不是只看你提到的 ddpo/rewards/docking.py，而是把它放回整个当前主线里一起看了：

- ddpo/rewards/docking.py

- ddpo/rewards/docking_backends/*

- ddpo/rewards/factory.py

- ddpo/rewards/composite.py

- ddpo/runtime/reward_wrappers.py

- ddpo/utils/molecule_semantics.py

- ddpo/main_diffsbdd_lightning.py

- ddpo/main_diffsbdd.py

- configs/ddp_gpu_druglikeness_only.yaml

- configs/ddp_gpu_qed_sa_test.yaml

- scripts/run_rl_qed_sa_8gpu_128.sh

- docs/project_context.md

我先给一句最核心的结论：

> Bug 5 的本质不是“docking 代码有点乱”，而是“项目已经进入主线切换期，但 docking 相关代码停留在多时代共存状态”。

也就是说，这不是一个普通实现层 bug，而是一个架构收敛失败导致的堆叠型 bug。

---

# 一、先说最终判断

## 最终判断 1

当前仓库已经有明确的“目标真相”：

- 主训练入口：ddpo/main_diffsbdd_lightning.py

- 主训练编排：Lightning DDP

- 主 docking backend：autodock_gpu

- 主 molecule 语义：OpenBabel reconstruct -> canonical RDKit Mol -> downstream reward/docking

- 不再让 Ray 影响主线

这些在 docs/project_context.md 里已经讲得非常清楚。

---

## 最终判断 2

但 DockingReward 相关代码仍保留了多个时代叠层：

1. backend 叠层

- autodock_gpu

- unidock

- smina

1. 执行模型叠层

- GPU 单分子循环 subprocess

- GPU batch 模式

- CPU 顺序模式

1. 历史架构叠层

- Lightning mainline

- legacy Ray-runtime 入口仍在仓库活跃层

1. I/O 管线叠层

- canonical RDKit -> SDF

- SDF -> PDBQT

- DLG / SDF / stdout 等不同解析通路

---

## 最终判断 3

所以 Bug 5 的危害不在“现在马上报错”，而在：

- 当前没启用 docking 时，它潜伏

- 一旦未来启用 docking reward，它会把多个旧时代分支一起唤醒

- 最终你面对的是：

- reward 有数

- 日志也有

- 训练也能跑

- 但很难确定数值来自哪条语义路径

这在科研代码里非常危险。

---

# 二、Bug 5 的“结构图”到底是什么

我把这个 cluster 拆成 6 层。

---

## Layer 1：项目“声明的主线”已经明确

从 docs/project_context.md 来看，项目当前方向已经不是模糊状态，而是非常明确：

- Lightning 是主线 orchestrator

- Ray 是历史 runtime，不是主线要求

- autodock_gpu 是主 docking backend

- unidock 是 legacy/broken

- 要避免新的 Ray orchestration

- molecule semantics 要统一成 canonical rdmol

所以从“项目规范层”看，主线已经定了。

---

## Layer 2：Reward factory 已经部分收缩，但没收完

ddpo/rewards/factory.py 现在的逻辑是：

- reward 组件从 config 推断 enabled

- 如果启用 druglikeness，就实例化 DruglikenessReward

- 如果启用 docking，才实例化 DockingReward

- docking backend 从 config 读 backend

这一步其实已经比过去好很多了，因为它说明：

- 主线不再默认强依赖 docking

- 只有显式启用 docking 才进入 docking reward 路径

所以 factory 层已经在做“主线瘦身”。

但问题是：

factory 只把 docking 入口缩窄了，没有把 docking 内部的多时代叠层拆开。

也就是说：

- 入口收敛了

- 内核没收敛

---

## Layer 3：DockingReward 仍然是“混合控制器”

这是 Bug 5 的核心。

现在的 ddpo/rewards/docking.py 虽然已经比老代码干净很多，甚至开头明确写了：

- AutoDock-GPU 主线

- UniDock legacy

- Smina legacy

- 不让 Lightning mainline 在 import 时依赖 Ray

这个方向是对的。

但它仍然同时承担了太多角色：

1. reward 语义包装

2. backend 选择器

3. 临时文件组织器

4. ligand 导出流程控制器

5. docking backend 调度器

6. backend 失败结果归一器

7. score normalization 执行器

8. failure_type 回填器

也就是说，它表面上叫 DockingReward，实际上已经是：

> Reward + Backend Router + IO Pipeline + Failure Mapping + Score Adapter

这就是典型的巨石类（god object）倾向。

---

## Layer 4：backend 是“可选”，但不是“同构”

从代码看，目前三种 backend 并不是同一种执行模型的不同实现，而是三种完全不同的系统：

### 1. autodock_gpu

特点：

- canonical rdmol -> SDF

- backend 内部再把 SDF 转成 PDBQT

- 每个 ligand 单独调用 AutoDock-GPU subprocess

- 解析 DLG 或 stdout

- GPU 设备通过 --devnum 指定，且是 1-indexed

### 2. unidock

特点：

- canonical rdmol -> SDF

- 一次性 --gpu_batch

- 输出目录回扫结果文件再解析

- 用 CUDA_VISIBLE_DEVICES 指 GPU

- 输出解析路径与 AutoDock-GPU 完全不同

### 3. smina

特点：

- canonical rdmol -> SDF

- CPU 顺序调用

- 直接 os.popen(...)

- 用 stdout 文本正则抽分数

- 没有 GPU 语义，也没有 batch 语义

所以这里真正的问题不是“有 3 个 backend”，而是：

> 这 3 个 backend 并不共享同一个失败模型、设备模型、I/O 模型、解析模型。

这意味着把它们塞进同一个 DockingReward 里，不是在做“多态”，而是在做“多系统拼盘”。

---

## Layer 5：legacy 时代虽然退出主线，但还没退出仓库活跃区

我专门看了 ddpo/main_diffsbdd.py。

这个文件开头仍然写得很明确：

- 它是 legacy / historical runtime path

- 保留 Ray-based trainer lifecycle

- 不是 preferred entrypoint

同时它还在 import ray。

所以现在的真实状态是：

- Ray 已经不再是主线 docking reward 的内部执行依赖

- 但 Ray 时代没有从仓库“认知层”完全退出

这会带来一个非常现实的问题：

### 认知污染

一个后来维护代码的人会很难立刻判断：

- 哪条是生产路径

- 哪条是遗留参考路径

- 哪些后端还真的该维护

- 哪些只是历史包袱

换句话说，Bug 5 不只是运行期 bug，还是代码认知 bug。

---

## Layer 6：配置层也在“保留结构”，进一步增强了混淆

我看了两个配置：

### configs/ddp_gpu_druglikeness_only.yaml

虽然 reward.enabled 只有 druglikeness，但 reward.docking 整块字段仍然完整保留，包括：

- backend: "autodock_gpu"

- fld_file

- box center/size

- nrun

- heuristics

### configs/ddp_gpu_qed_sa_test.yaml

同样，虽然注释写的是：

- 当前不会启用 DockingReward

- 这里只是保留字段结构

但 docking 配置也在。

这本身不是错，问题是它向系统释放了一个信号：

> docking 虽然没启用，但它像是“随时可以切上”的第一等功能。

然而实际上现在并不是“任意 backend 都是第一等支持”。

所以这里形成了一个典型错配：

- 配置层显得很宽

- 实际主线支持面很窄

这正是堆叠型 bug 容易长期存活的土壤。
# 三、Bug 5 的根因，不止一句“历史路径没及时收缩”

你原文里说根因是：

> 历史路径没有及时收缩

这句话是对的，但还不够深。

我把根因拆成 5 个层次。

---

## 根因 1：主线切换完成了一半

项目已经从：

- Ray runtime 主导

- 多旧路径并存

切到了：

- Lightning 主线

- canonical molecule semantics

- docking 按需启用

但是只完成了“一半”：

- 入口收了

- 内核没完全收

- backend 角色分层没完成

所以代码表现为“新旧秩序叠在一起”。

---

## 根因 2：把“backend 可选性”误写成了“单类多分支”

可选 backend 本来没问题。

问题在于过去通常会直接这么写：

- if backend == autodock_gpu

- elif backend == unidock

- else smina

一开始很省事，但一旦 backend 之间不是简单差几个参数，而是差整套执行模型，就会出问题。

也就是说：

> 这里不是 backend 太多，而是 backend 差异太大，却还被塞在同一控制流里。

---

## 根因 3：reward 层承担了太多系统职责

理论上 reward 层应该只定义：

- 成功时如何算 reward

- 失败时给什么 penalty

- 哪些字段写回 batch

但当前 DockingReward 实际承担了：

- ligand build/export

- tempdir 组织

- backend 初始化与缓存

- subprocess 调度

- score parse

- failure mapping

这使得任何 backend 细节变化都会直接污染 reward 语义层。

---

## 根因 4：legacy backend 没被清晰降级

从项目文档看：

- autodock_gpu 是 primary

- unidock 是 legacy / broken

- smina 是 legacy CPU

但在代码结构里，三者仍然共处一个 backend 体系，而且 __init__.py 还一起 export：

- AutoDockGPUBackend

- UniDockBackend

- SminaBackend

这在结构上给人的感觉仍是“平级正式支持”。

所以真正的问题不是代码里还有 legacy，而是：

> legacy 没有被结构性地降权。

---

## 根因 5：设备语义天然不一致，却缺少顶层策略统一

这是很容易被忽略、但未来最可能炸的点。

### AutoDock-GPU

- 用 --devnum

- 1-indexed

- 明确不应依赖 CUDA_VISIBLE_DEVICES 重映射

### UniDock

- 用 CUDA_VISIBLE_DEVICES

- 0-indexed 语义

### 训练脚本

- 统一设置 CUDA_VISIBLE_DEVICES

虽然 AutoDockGPUBackend 内部会删除 CUDA_VISIBLE_DEVICES，算是做了防御，但这只是 backend 内部自救，不是架构级统一策略。

所以你现在是：

- 训练 orchestration 一套设备语义

- docking backend 又各自一套设备语义

这就是典型的跨时代设备模型堆叠。

---

# 四、为什么这个 bug 在当前不爆，但未来会成倍放大

因为当前大部分配置都还是：

- 只启用 druglikeness

- docking 只是“结构保留”

所以现在系统没有真正把 Bug 5 的大部分风险路径跑起来。

但一旦启用 docking，系统就会在一个 batch 内同时穿过这些边界：

1. canonical sample semantics

2. docking input export

3. backend device assignment

4. subprocess / batch execution

5. backend-specific score parsing

6. failure_type 回填

7. composite reward 汇总

8. Lightning 日志与 DDP 汇总

这意味着 Bug 5 的问题不是线性增加，而是乘法式增加。

因为你不是多一个功能，而是多一整层系统边界。

---

# 五、Bug 5 具体会导致哪些问题

我按“你未来最容易踩到的真实坑”列。

---

## 问题 1：你以为是在切 backend，实际上是在切实验语义

表面看是：

- backend: autodock_gpu

- backend: unidock

- backend: smina

但实际上切换 backend 的同时，你还切换了：

- 输入文件格式要求

- 设备映射策略

- 批处理模型

- 失败返回模式

- 输出解析规则

- 性能瓶颈位置

所以它不是简单的策略切换，而是实验定义切换。

---

## 问题 2：失败类型不够可审计

现在虽然已经有：

- docking_input_failed

- docking_backend_failed

这比以前好很多。

但从整体看，还缺几个重要维度：

- 每个 backend 的解析失败计数

- ligand export 失败计数

- 后端调用失败计数

- score parse 失败计数

- backend timeout 计数

没有这些时，未来 reward 变差你很难判断：

- 是 molecule quality 真变差

- 还是 docking 管线在 silently fail

---

## 问题 3：配置空间宽于验证空间

代码和 config 允许很多 backend 选择，但真正被项目上下文确认“可信”的只有 autodock_gpu。

这会导致：

- 用户以为“都能用”

- 实际只有一个被主线验证过

- 其余路径一旦启动，就是边跑边排雷

---

## 问题 4：主线维护者认知负担过大

未来你自己一个月后回看，最难受的不是报错，而是：

- 哪个 backend 是真实 supported？

- 哪个 backend 只是保留兼容？

- 哪个输出字段可以拿来做科研结论？

- 哪个路径只是 debug fallback？

这会极大拖慢维护效率。

---

## 问题 5：DockingReward 会重新长成巨石类

如果不治理，未来你每做一次改动都会往里继续塞：

- 更复杂的 tempdir 规则

- 更多 backend 选项

- 更多参数兼容

- 更多分数 parse fallback

- 更多日志打印特例

最终 DockingReward 会变成全项目最难维护的文件之一。
# 六、当前其实已经“做对了”的部分

不是全盘负面，也要客观看。

---

## 已经做对的点 1：canonical molecule semantics 已经建立

ddpo/utils/molecule_semantics.py 已经把主语义明确下来：

- sample -> geometry

- OpenBabel reconstruct

- RDKit sanitize

- rdkit_valid

- failure_type

- canonical rdmol

- downstream export

这是非常关键的基础。

也就是说，Bug 5 现在不是“完全无序”，而是在一个相对正确的新语义上，包着多时代 backend 壳子。

---

## 已经做对的点 2：druglikeness / docking 正在靠同一 canonical path 收敛

DruglikenessReward 现在也已经调用 canonical molecule build 了。

DockingReward 里的 _build_mol_to_sdf 也走 build_canonical_rdmol_from_sample(...)。

这说明最危险的“分子重建语义分叉”正在被修正。

所以当前 Bug 5 的主战场，已经从“化学语义错乱”转移到了：

> backend / runtime / IO / lifecycle 的结构耦合问题。

---

## 已经做对的点 3：factory 已经把 docking 变成按需启用

这避免了你每次训练都被 docking 依赖拖住。

这是收敛的必要前提。

---

## 已经做对的点 4：Ray 已经退出 docking 主文件

你文档里提到 ray remote，但就当前 ddpo/rewards/docking.py 而言，Ray 已经不在主实现里了。

这说明工程方向是对的，只是没收尾。

---

# 七、我对 Bug 5 的定性分级

我给它的定性是：

## 不是 P0 的原因

因为当前默认训练路径还没真正开启 docking，所以它不是“当前立刻阻断主线训练”的 bug。

## 但它绝不是低优先级

因为它是一个未来会指数级放大的结构 bug。

所以我会把它定成：

### P1.5 / P2 之间的“战略型 bug”

- 短期不一定爆

- 但只要进入 docking reward 主线，马上会成为主故障源之一

---

# 八、正确的解决方向：不是砍掉“可选”，而是把“可选”真正解耦

你前面问得很对：

“是不是应该做成可以随便选，但是不要揉在一起，而是解耦开？”

我的答案仍然是：

> 对，应该保留可选性，但必须把 backend 选择权和 backend 实现细节彻底解耦。

下面我给出具体可执行方案。

---

# 九、解决方案：分三阶段做，最稳

---

## 第一阶段：收“语义边界”，不改 reward 数学定义

目标：让当前行为更可解释，但不改实验含义。

### 1. 把 DockingReward 缩回“reward 语义层”

它只应该负责：

- 调用一个 backend interface

- 接收 docking 结果

- 做 normalization

- 写回：

- docking_reward

- dock_score_raw

- dock_success

- failure_type

它不应该继续直接承担：

- 多 backend 细节分发

- 每个 backend 的 temp file 管线细节

- parser 分支细节

---

### 2. backend 选择只放在 factory / builder 层

现在 factory 已经迈出一步了，但还不够。

应进一步收紧成：

- build_rewarder(...) 里负责根据 config 选 docking backend

- DockingReward 构造时直接接收 backend instance，而不是再接收 backend name 然后自己判断

也就是说从：

- DockingReward(backend="autodock_gpu", ...)

转向：

- DockingReward(backend_impl=AutoDockGPUBackend(...), ...)

这样 DockingReward 就不再知道三条分支。

---

### 3. 明确“supported / legacy / experimental”分层

建议在结构上明确三档：

#### supported

- AutoDockGPUBackend

#### legacy / disabled by default

- UniDockBackend

- SminaBackend

这样不是删功能，而是把支持等级写进结构。

否则你代码层还在暗示：

“三个 backend 都是正式平权”。

---

## 第二阶段：拆 backend 内部职责

目标：把“多 backend”从“多控制流”变成“多实现类”。

### 1. 抽出 ligand preparation/export 层

目前 docking 里最容易纠缠的是：

- rdmol -> SDF

- SDF -> PDBQT

- tempdir / tmpfile

- 输出清理

建议独立成一个 preparation/export 层。

因为这些逻辑既不属于 reward，也不完全属于 backend 打分语义。

这样以后 backend 类只关心：

- 我需要什么输入文件

- 我怎么调 backend

- 我怎么解析结果

---

### 2. backend 类只做 3 件事

每个 backend 只负责：

1. 接收已经准备好的 ligand inputs

2. 调后端命令

3. 返回结构化结果

返回值建议统一成结构化对象，而不是只给一个 float 列表。

比如概念上应该统一成：

- score_raw

- success

- failure_reason

- backend_name

这样 reward 层就不需要自己推断失败语义。

---

### 3. 去掉 backend 实现里的“隐性策略”

例如：

- AutoDock-GPU 自己偷偷删 CUDA_VISIBLE_DEVICES

- UniDock 自己改 env

- smina 自己用 os.popen

这些可以继续保留实现，但要提升为显式日志可见策略。

至少每次启用 docking 时日志里要能看到：

- effective backend

- GPU selection policy

- ligand input format

- receptor / map file

- batch mode or sequential mode

---

## 第三阶段：让 config 和文档成为唯一真相

目标：避免未来再次回到“看代码猜主线”。

### 1. config 中显式标注 backend support level

不一定写进 yaml 字段，但至少在加载时对非主线 backend 给 warning。

比如：

- autodock_gpu: supported

- unidock: legacy / unverified

- smina: debug / cpu fallback only

---

### 2. 对非主线 backend 做 hard warning

如果用户选了：

- unidock

- smina

系统应该在训练启动时明确打印警告，而不是静默切换。

这样未来实验日志里能看出：

“这不是正式主线 backend”。

---

### 3. 文档与代码统一

当前文档已经写得很清楚了，下一步要让代码结构也体现这种清晰度。

即：

- 文档说 autodock_gpu 是主线

- 代码结构里也应该让它显得是主线

- legacy backend 不应再看起来像正式平权实现
# 十、如果现在就要开始治理，最应该先做哪几步

我按“最值得先做”的顺序排。

---

## Top 1：把 DockingReward 从 backend 选择逻辑里解放出来

这是最大收益点。

收益：

- reward 语义独立

- backend 复杂性不再污染 reward 层

- 后续测试粒度清晰

---

## Top 2：把 autodock_gpu 定为唯一 supported 主线路径

不是删 unidock/smina，而是先结构性降级。

收益：

- 减少认知噪声

- 减少未来主线回归成本

- 与项目上下文一致

---

## Top 3：给 docking 返回结构化结果，而不是只返回 float reward

收益：

- failure 统计更清楚

- reward 分析更可审计

- 后续 CSV / summary 能看懂到底失败在哪

---

## Top 4：把设备映射策略提升到显式日志

尤其是 AutoDock-GPU vs UniDock 的设备编号差异。

收益：

- 避免最隐蔽的 GPU 编号 bug

- 方便多卡排障

---

## Top 5：补 4 类最小测试

最少要有：

1. canonical rdmol -> docking input export smoke test

2. AutoDockGPUBackend 单 ligand smoke test

3. DockingReward 一批样本的 failure mapping test

4. fixed config 下 backend selection test

---

# 十一、如果不治理，会发生什么

这个我说得直接一点。

---

## 后果 1：启用 docking 后，问题不是“会不会报错”，而是“结果能不能解释”

这是最坏的科研风险。

---

## 后果 2：每次改 backend 都像拆炸弹

因为 reward、IO、device、parser 绑在一起。

---

## 后果 3：legacy 路径会不断复活

你今天不处理，明天加一个小功能时又会顺手往旧分支里补代码。

---

## 后果 4：主线会被“表面可选性”拖垮

表面看起来很灵活，实际上会让主线越来越不稳定。
# 十二、我给 Bug 5 的最终定义

我建议你把 Bug 5 定义为：

> “docking 奖励模块在主线收敛过程中，仍保留多 backend、多执行模型、多时代设备/IO 语义于同一控制体内，导致主线可选性与实现耦合混杂，形成未来启用 docking 时的高阶结构性风险。”

这比“多 backend 堆叠型 bug”更完整。

---

# 十三、我的最终建议，简洁版

## 诊断

- 主线方向已经明确

- docking 入口已部分收敛

- 但 docking 内核仍是多时代叠层

- 现在最大的风险是“backend 可选性”和“实现耦合”没有拆开

## 最小修复方向

- 保留 backend 可选

- 但把 backend 选择移出 DockingReward

- 把 DockingReward 缩成 reward 语义包装器

- 把 ligand export / backend invocation / parse 拆出去

- 让 autodock_gpu 成为唯一 supported 主线路径

- 让 unidock/smina 明确降级为 legacy

## 风险

- 不应在治理过程中改变 reward 数学定义

- 不应改变 canonical molecule semantics

- 不应引入新的 Ray 依赖

- 不应模糊 failure_type 语义

## 下一步最该测

- 单分子 AutoDock-GPU smoke test

- 一批样本 docking input/export smoke test

- backend selection 的 config-to-runtime 一致性

- failure_type 统计是否与实际失败阶段一致

---

如果你要，我下一步可以继续直接给你一份：

1. Bug 5 的“按文件整改方案”

精确到 docking.py / factory.py / docking_backends/* / config / tests 分别怎么改；

或者

1. Bug 5 的“重构蓝图”

我直接把目标架构画成模块级设计，告诉你每个类该保留什么职责。