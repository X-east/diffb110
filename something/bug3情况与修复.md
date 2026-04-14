对，基于你现在补充的这个前提，bug3 应该这样拆：

- RLPF actor 里的 RL 训练语义本身是完整的

- Lightning 只是分布式/多卡运行时外壳

- 所以 bug3 的核心已经不再是“RL 不完整”

- 而是两件事：

1. 集成层语义表达不够清楚

2. reward 如何正确进入 DiffSBDD 的策略更新链，需要被讲清楚

下面我按这两个层面详细拆。

---

# 一、先给最终判断：bug3 的正确定性

## 不是这个

- 不是“actor 真更新，所以架构错了”

- 不是“RL 语义天然不完整”

## 而是这个

bug3 本质上是：

> 完整的 RLPF RL 内核，被接进了 Lightning 多卡壳之后，外层代码表意仍像标准 Lightning trainer，导致训练控制权、loss 语义、reward 回传路径不够显式。

所以 bug3 更准确的中文标题，我建议改成：

## 建议标题

Bug Cluster 3：RLPF 训练内核与 Lightning 分布式外壳的语义边界不清

或者更短一点：

Bug Cluster 3：RL 训练控制权表达不清

---

# 二、现在这个系统真实是怎么工作的

你现在的主线，其实已经可以很清楚地抽象成 5 层：

## 第 1 层：DiffSBDD 负责“策略模型”

它不是在做监督学习标签拟合，而是在 RL 里充当当前策略。

更准确地说：

- DiffSBDD 生成一条扩散反演轨迹

- 这条轨迹上每一步都有旧策略下的 log-prob

- 后面 actor 再用当前参数重新评估这些 step 的 log-prob

- 然后用 PPO/DDPO 风格 loss 更新参数

所以在 RL 语义里，DiffSBDD 扮演的是：

> stochastic diffusion policy

---

## 第 2 层：rollout 负责“采样轨迹”

DiffSBDDRollout.generate_minibatch() 会先从 DiffSBDD 采样出一批分子，同时保留 RL 训练需要的轨迹信息：

- x

- categorical

- latents

- logps

- timesteps

- nodesxsample

这一层非常关键，因为这里决定了：

> reward 不是直接喂给 x0 的回归标签，

> 而是绑定到整条采样轨迹上。

换句话说，reward 对应的是“这次采样整条轨迹最终产出的样本质量”。

---

## 第 3 层：rewarder 负责“对最终样本打分”

然后外层会对 rollout 生成的样本做：

1. 分子语义标注

2. filter 统计/筛选

3. reward 计算

你现在的 reward 链大致是：

- build_rewarder(...)

- CompositeReward

- DruglikenessReward / DockingReward

- StabilityAwareReward

最终 reward 被写回 samples.batch['rewards']。

这个动作非常重要，因为它定义了：

> reward 的宿主不是模型参数，也不是 diffusion 每一步的 state，

> 而是当前 batch 的采样结果容器 DataProto.batch。

---

## 第 4 层：actor 负责“把 reward 转成策略梯度更新”

这一步才是 reward 真正“回到模型”的地方。

你的 actor 不会做这种事：

- 直接把 reward 当成 differentiable scalar 对 x 求导

- 直接把 reward 反传过 RDKit / docking / QED / SA

它做的是 RL 的标准路径：

1. 取 rollout 时记录的 old_log_prob

2. 用当前 DiffSBDD 参数重新计算 current_log_prob

3. 计算 ratio = exp(new - old)

4. 用 advantage/reward 加权 PPO clip objective

5. backward

6. optimizer.step()

所以 reward 回到 DiffSBDD 的方式不是：

> reward 直接作为扩散模型输出的监督标签

而是：

> **reward 作为策略梯度权重，通过 log-prob ratio loss，间接推动 DiffSBDD 参数更新**

这点是整个 bug3 分析里最核心的一句。

---

## 第 5 层：Lightning 只负责“多卡外壳”

你已经明确说了，你用 Lightning 的目的是：

- 多卡 GPU 并行

- DDP 生命周期

- callback / logger / checkpoint 外壳

那它的真实角色就是：

> distributed runtime shell

而不是 RL trainer 本体。

---

# 三、现在的问题到底拆成哪些点

在这个正确前提下，bug3 现在应该拆成 4 个具体问题。

---

## 问题 1：外层代码“看起来像”标准 Lightning trainer，但实际上不是

这是最大的问题，而且是表意问题，不是算法问题。

现在外层模块会让人形成错误心智：

- training_step() 看起来是主训练步

- configure_optimizers() 看起来定义了优化器

- train_loss 看起来是 Lightning 真正优化的 loss

但你的真实语义是：

- training_step() 更像 orchestration

- 真 optimizer 在 actor 里

- 真 backward / step 在 actor 里

- train_loss 更像 actor 回报出来的统计指标

### 所以这个问题的风险是

后续任何人只要按标准 Lightning 心智改：

- optimizer

- scheduler

- grad clip

- accumulate grad

- AMP

- resume

都可能改错层。

### 这不是训练错误

但这是维护风险非常高的语义误导。

---

## 问题 2：configure_optimizers() 是误导性最强的一处

即使你故意不用 Lightning optimizer，这段仍然会误导人。

因为只要别人看到这里有个 optimizer，就会天然假设：

- 这是训练主优化器

- 改这个 lr 会生效

- scheduler 会挂在这个 optimizer 上

但实际上不是。

### 所以这里的核心问题不是“有 bug”

而是：

> 这个接口保留了 Lightning 标准 trainer 的外观，但不承载真实训练语义。

---

## 问题 3：advantages = rewards 这个 fallback 需要重新定性

基于你说“RLPF RL 内核完整”，那这句不能再被简单当成“PPO 不完整”的铁证。

更准确地说，它是：

> 集成层 fallback / 兼容桥接点

也就是说，这里暴露的问题不是“actor 不会算 PPO”，而是：

- 外层给 actor 的字段约定还不够硬

- advantages 字段在集成接口上没有被完全固定

- 所以留了一个 rewards -> advantages 的回退路径

### 这个回退路径的问题在于

它会模糊这几个概念：

- reward

- advantage

- normalized advantage

- policy weight

所以这里仍然值得列为 bug3 子问题，但要精准表述为：

> 不是 RL 内核不完整，而是外层输入契约仍保留过渡态 fallback。

---

## 问题 4：checkpoint / resume 的语义所有权还不够清楚

既然 actor 是真 trainer，那么从训练状态语义上看，真正需要完整保存/恢复的不只是：

- model state

还应该包括至少：

- actor optimizer state

- 可能的 step / epoch 对齐信息

- 任何 RL trainer 自己内部依赖的状态

如果当前 checkpoint 更多是 Lightning 壳在存，而 actor trainer state 没被显式纳入统一语义，那就会造成一个非常典型的问题：

> 模型参数 resume 了，但 trainer 状态是否完整恢复，不够透明。

这个问题不是“肯定坏了”，但它一定是 bug3 的一部分，因为它直接关联：

- 恢复训练的一致性

- 优化器动量

- 实验可复现性

- 中断续训理解成本

---

# 四、你的 reward 到底该怎么“传回” DiffSBDD 扩散模型

这个问题必须讲得非常准确。

## 先说错误理解

reward 不应该这样传回：

- 直接对 x 反传

- 直接对 RDKit/QED/SA/docking score 反传

- 把 reward 当成 diffusion 每一步的监督标签

- 把 reward 塞进 DiffSBDD 的 forward 当额外回归目标

因为这些 reward 大多是：

- 非可微

- 后处理得到

- 只定义在最终样本层面

所以 reward 不能用“监督学习标签”的方式回传。

---

## 正确理解

reward 应该通过 policy gradient / DDPO 路径回传。

也就是：

### 第一步：DiffSBDD 先采样

模型先采样得到一批分子和对应轨迹：

- latents

- timesteps

- old_logps

这里的 old_logps 是“采样这条轨迹时旧策略给出的概率信息”。

---

### 第二步：对最终样本打 reward

然后 rewarder 在最终样本层面给每个 sample 一个标量 reward：

- rewards[i]

例如：

- 药物相似性奖励

- docking 奖励

- composite reward

这些 reward 放回 samples.batch['rewards']。

---

### 第三步：reward 变成 actor update 的权重

外层把 reward/advantage 传进 actor：

- 最理想：显式传 advantages

- 目前过渡：缺失时回落到 rewards

然后 actor 对轨迹中每个 step 重新算新策略 log-prob。

---

### 第四步：用 PPO/DDPO loss 更新 DiffSBDD 参数

actor 会构造类似：

L=−E[min⁡(rtA,clip⁡(rt,1−ϵ,1+ϵ)A)]L=−E[min(rt​A,clip(rt​,1−ϵ,1+ϵ)A)]

其中：

- AA = sample-level advantage 或 reward

- rt=exp⁡(log⁡pnew−log⁡pold)rt​=exp(logpnew​−logpold​)

注意这里的关键点：

## reward 不直接作用于 diffusion 坐标

而是作用于：

> “这条扩散去噪轨迹在 PPO 目标里应该被鼓励还是压低”

也就是说，reward 对 DiffSBDD 的影响路径是：

reward→advantage→PPO/DDPO loss→backward→DiffSBDD parametersreward→advantage→PPO/DDPO loss→backward→DiffSBDD parameters

这才是正确答案。

---

# 五、你现在 reward 回传链的真实语义图

可以概括成这一条：

## 当前正确主线

### 1. rollout 采样

DiffSBDD 生成：

- x

- categorical

- latents

- logps

- timesteps

### 2. reward 计算

rewarder 根据最终分子性质，写入：

- rewards

- 组件 reward

- 其它统计字段

### 3. 外层准备 actor 输入

外层把：

- latents

- timesteps

- logps

- nodesxsample

- advantages 或 rewards

交给 actor

### 4. actor 重新评估策略

actor 用当前 DiffSBDD 参数重新算：

- log_prob_current

### 5. PPO/DDPO 更新

用：

- old_log_prob

- new_log_prob

- advantage/reward

构造 clipped objective

### 6. 参数更新落到 DiffSBDD

由于 actor 持有的是 DiffSBDD 模型参数，optimizer.step() 后，更新真正落在 DiffSBDD 上

---

# 六、从 RL 语义上说，你的 reward 最该怎么传

如果我按“最干净的语义”来给建议，不是让你大改架构，而是建议你固定这条契约：

## 推荐契约

### 1. rewarder 只负责输出 reward，不负责伪造 advantage

rewarder 输出：

- rewards

- 各组件分数

- failure_type

- valid flags

不要让 rewarder 混入 PPO 内部语义。

---

### 2. actor update 入口显式要求一个训练权重字段

理想上应明确其中之一：

- advantages

- 或者 policy_weights

而不是依赖模糊 fallback。

---

### 3. 如果当前没有 baseline，就明确声明

如果你的 RLPF 路径当前本来就是 sample-level reward weighted update，那就明确写清：

- 当前训练使用 reward-as-advantage

- 这是该 RL 变体的既定设计

- 不是误写，不是缺字段偶然退化

这个“声明”很重要，因为它决定 bug3 的性质。

---

### 4. reward 应保持 sample 级，不要伪造 timestep 级 reward

你当前 actor loss 用的是 sample-level advantages，并在各 diffusion timestep 上共享使用。

这在 DDPO 风格里是能成立的。

不要为了“看起来更细”而随便把最终 reward 硬拆到每个 timestep，除非你有严格定义。

否则会破坏轨迹语义。

---

# 七、bug3 现在可以正式拆成哪些子项

我建议在文档里拆成下面 4 个子 bug。

---

## Bug 3.1：训练控制权在 actor，但外层表意仍像 Lightning trainer

### 现象

外层模块命名和接口让人误以为 Lightning 是主 trainer。

### 实际

Lightning 只是多卡壳；actor 才是真 trainer。

### 风险

后续维护者按 Lightning 心智误改 optimizer/loss/scheduler/resume。

---

## Bug 3.2：reward 回传路径没有被明确文档化

### 现象

容易误以为 reward 要直接传回扩散模型输出。

### 实际

reward 应通过 PPO/DDPO 的 log-prob ratio loss 间接作用于 DiffSBDD 参数。

### 风险

后续容易引入错误的“直接 reward 监督”改法。

---

## Bug 3.3：advantages 输入契约仍保留 fallback 语义

### 现象

外层在缺失 advantages 时会直接回退到 rewards。

### 实际

这更像 integration-layer fallback，而不是 RL 内核定义本身。

### 风险

模糊 reward 与 advantage 的边界，增加误读风险。

---

## Bug 3.4：checkpoint/resume 的 trainer-state 归属不够显式

### 现象

当前更容易看清 model state，较难一眼看清 actor trainer state 是否被完整纳入。

### 风险

恢复训练时语义不透明，影响复现与调试。

---

# 八、如果只回答你一句：“reward 该怎么传回 DiffSBDD？”

最准确的一句话是：

> reward 不应该直接反传到 DiffSBDD 生成的坐标/原子类型上，而应该先挂到 sample/trajectory 上，作为 actor 计算 PPO/DDPO policy loss 的权重，再通过 old/new log-prob ratio 的反向传播间接更新 DiffSBDD 参数。

再压缩一点就是：

> reward 回到 DiffSBDD 的正确路径是：reward → advantage/policy weight → actor PPO loss → backward → DiffSBDD parameters。

---

# 九、基于你当前架构，我的建议结论

## 你现在最该坚持的架构事实

- actor 是真 trainer

- Lightning 是多卡分布式壳

- reward 通过 PPO/DDPO 更新路径间接传回 DiffSBDD

- 不是直接监督 DiffSBDD 输出

## 你现在最该修的不是 RL 内核

而是：

1. 把这个分层事实写清楚

2. 把 reward/advantage 输入契约写清楚

3. 把 checkpoint/resume 的 owner 写清楚


# 修复情况
## 总结结论（先说结果）

bug3 主体修复是成功的，而且你现在的训练语义已经清晰很多：

- 真正的优化器和反向传播在 RLPF 的 EDMActor 内执行；

- Lightning 层主要承担 DDP 外壳、日志和 checkpoint 生命周期；

- reward 通过 advantage 和 PPO 比率进入扩散模型参数更新链路（不是直接写回 diffusion state）。

但我也看到几个仍然存在的工程问题（不是 bug3 核心，但会影响训练观测与可恢复性），后面我会列。

---

## 我怎么判断 bug3 已修复（核心证据）

### 1) 训练 owner 语义已经显式化

你在 lightning_ddpo_module 的模块说明和 training_step 注释中，已经明确写了：

- 参数更新在 self.actor.update_policy(...) 内发生；

- Lightning 返回的 loss 是“汇报/日志用途”。

这正是 bug3 里“语义边界不清”的关键修复点。

### 2) training_step 返回 loss，训练循环不再中断

你之前辅助脚本 something/fix_training_step.py 也说明了历史问题是 training_step 返回 None。

现在代码已改为返回 loss，并且从你终端实跑看，5 个 epoch 完整跑完，说明这个断点已经修好。

### 3) configure_optimizers 已改成“兼容优化器 + manual optimization”

当前是：

- self.automatic_optimization = False

- 返回一个 dummy SGD 供 Lightning 初始化流程通过

这和你的架构目标一致：Lightning 不拥有真实优化语义。

### 4) reward → advantage → PPO loss → DiffSBDD 参数 的链路是通的

我核对了完整链路：

- rollout 产出 latents/logps/timesteps

- reward 写入 samples.batch['rewards']

- Lightning 层在缺失优势时计算 batch-normalized advantage

- actor 用 sample_p_zs_given_zt(..., prev_sample=...) 重新评估当前策略下 log-prob

- PPO ratio + clipped objective 做 backward，AdamW(self.model.parameters()) 执行 step

所以你关心的“reward 是否传回 DiffSBDD”答案是：是，且方式正确（策略梯度语义）。

### 5) checkpoint 恢复语义比之前完整

on_save_checkpoint/on_load_checkpoint 现在带了 actor_optimizer_state，恢复时也会 load。

这对 RL 连续训练非常关键，属于 bug3 语义修复的加分项。

---

## 这次修复后的运行表现（你最新实跑）

从你终端记录看（scripts/run_rl_qed_sa_test.sh）：

- 2 卡训练完成 5 epoch；

- epoch 级 loss / reward 正常输出（无中途崩溃）；

- RDKit Validity 统计在控制台有波动（例如第4轮 87.5%），符合真实训练波动；

- 没再出现“training_step 返回非法值导致训练失败”的典型症状。

这说明 bug3 的“训练循环语义不闭合”问题已经实质解除。

---

## 我认为还需要你注意的 4 个剩余问题（非 bug3 主体，但很重要）

### A) checkpoint 仍未真正落盘确认

日志每轮都提示 last_model_path is empty，且该 run 输出目录里没有 checkpoints/。

这会直接影响断点续训可靠性。

（这和 bug3 的“语义明确”是两回事，是你现在新的工程阻塞点）

### B) 多卡 rank 输出目录可能发生“秒级分叉”

我看到同一轮有两个目录时间戳（如 ...102242 和 ...102249）同时出现，且 rank1 文件落在另一个目录。

这会导致：

- rank0/rank1 产物分裂；

- merge 逻辑异常；

- checkpoint 和 molecules 汇总不一致。

根因通常是每个进程各自 datetime.now() 生成了不同 output_dir。

### C) epoch_summary_global.csv 的 generated_total/valid 为 0

虽然控制台显示每轮 16/16 等正常计数，但 epoch_summary_global.csv 里是 0。

这说明 recorder merge 统计链路仍有 bug（多半和 B 的目录分叉有关）。

### D) 训练结束打印 epoch 数有 off-by-one

max_epochs=5，结束时打印 Total Epochs: 6。

这是显示层小 bug，但会误导训练审计。

---

## 对 bug3 修复质量的最终评级（我给你的客观评价）

- 科学语义正确性：✅ 通过（reward 回传与 PPO/DDPO 语义正确）

- 训练控制权表达：✅ 明显改善（actor owner 已清楚）

- 可维护性：✅ 比之前好很多

- 工程闭环（checkpoint/输出一致性）：⚠️ 仍需修（与 bug3 并列优先级）