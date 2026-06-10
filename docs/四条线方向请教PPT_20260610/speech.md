# DiffB110 四条探索线方向请教 speaker notes

## Slide 1: 封面

这次汇报不是最终成果展示，而是想请老师帮我判断课题应该收敛到哪条主线。我目前围绕 3D 分子扩散模型强化学习做了四条探索线：DiffSBDD 强化学习、DiffSBDD 内部小改、DiffSBDD x/h/e 大改，以及 DiffGui 强化学习。核心问题是：底座质量、reward hacking 和方法贡献之间，应该怎么取舍。

---

注意点：
- 重点：这是方向请教，不是包装结果。
- 节奏：开场直接说明希望老师帮忙判断主线。

## Slide 2: 两个底座：DiffSBDD vs DiffGui

这里先解释两个底座的差别。DiffSBDD 主要生成坐标和原子类型，最后靠规则工具重建键，所以速度快、工程成熟，但成键质量弱。DiffGui 则原生建模 atom、bond/halfedge 和坐标，更接近联合生成，所以成功样本合法性更强。但这里必须谨慎，DiffGui 的 100% sanitize 是成功写入并进入评估的 SDF 口径，不代表原始生成没有失败。

---

注意点：
- 重点：不要把 DiffGui 说成完全无失败。
- 补充：左右分子图由用户后续手动加入。

## Slide 3: 共同背景：底座质量决定 RL 难度

这页是共同背景。同一靶点横评里，DiffSBDD 的 sanitize 只有约 66.6%，DiffGui 成功写入样本为 100%。但 DiffGui 仍有 raw reconstruction failure，而且它也会生成复杂环和高碳分子。这说明问题不是单纯哪个模型合法性更好，而是 RL 会在什么样的底座上学习。如果底座本身坏分子太多，reward 信号会大量消耗在惩罚坏分子上。

---

注意点：
- 画面引导：先看 sanitize 图，再看右侧口径说明。
- 重点：评估口径必须讲清楚。

## Slide 4: C 线：DiffSBDD RL 跑通，但 reward hacking 明显

C 线是工程最成熟的一条。DDPO/PPO、多 GPU、reward factory、描述子监控这些都跑通了。但旧训练暴露出明显 reward hacking：模型向高碳、低极性、少杂原子方向漂移。classifier 信号稀疏，docking 又偏好某些疏水复杂结构，所以 PPO 很容易学到不理想的捷径。

---

注意点：
- 重点：C 线价值是工具链和诊断，不一定是最佳主线。

## Slide 5: A 线：DiffSBDD 小改低风险，但改善有限

A 线是低风险修补方案，比如 atom prior、geometry refine、local mask 和 bond head。它的优点是容易接入现有 DiffSBDD 和 RL 流程，但它没有改变扩散变量，最终成键仍然靠规则重建。因此它更适合做 ablation 或辅助增强，而不是承担完整论文主线。

---

注意点：
- 重点：低风险不等于适合主线。

## Slide 6: B 线：x/h/e 大改触及根因，但不是原创机制

B 线结构上最接近 DiffSBDD 的成键根因，因为它尝试把 bond/halfedge 作为 e 通道加入扩散。但这个方向不能写成原创机制，因为 x/h/e 或 atom-bond-coordinate 联合建模已经被 DiffGui 等模型实现过。更准确地说，这是把已有联合建模思想适配到 DiffSBDD 上，工程风险很高，也不确定最终能否超过直接使用 DiffGui。

---

注意点：
- 重点：不要说“创新性最高”。
- 补充：这里的价值是适配、验证和对照。

## Slide 7: D 线：DiffGui RL 已跑通，进入分布稳定化测试

D 线是我目前最倾向的主线。DiffGui 底座更强，RL rollout、logprob、PPO actor 和 reward-only 子进程已经接通。最新 ring-quality 测试中，ring 和 carbon 有一点改善信号，但 docking 没有同步改善。这说明路线有希望，但不能说已经成功。下一步真正要研究的是多目标 reward 和分布稳定化。

---

注意点：
- 重点：ring/carbon 有早期信号，docking 未同步改善。
- 节奏：这里要诚实，不要夸大。

## Slide 8: 四条线风险-收益判断

这页是我目前的主线判断。C 线成熟，适合作为工具链和对照；A 线低风险，适合作为 ablation；B 线触及根因但属于高风险适配；D 线底座最好、工程已跑通、成本可控，所以最适合作为当前主推候选。

---

注意点：
- 重点：主推 D，不是否定其它线，而是给它们重新定位。

## Slide 9: 想请教老师的关键问题

这里我想请老师帮忙判断几个关键问题：DiffGui RL 是否足够作为主线？reward hacking 和 distribution-stabilized RL 能否构成方法贡献？B 线是否值得继续投入，还是收缩成适配模块？以及如果面向计算机论文，审稿人更看重生成质量、docking、分布稳定化，还是模型结构新颖性？

---

注意点：
- 重点：问题要聚焦方向选择。

## Slide 10: 两种可收敛论文 story

最后我把可能 story 收敛成两个。Story A 是 DiffGui 上的 distribution-stabilized RL，这是我当前更倾向的方向。Story B 是 DiffSBDD 的 bond-aware adaptation，风险更高，更适合作为支线或后续扩展。我希望老师判断的是：Story A 的方法贡献是否足够，以及 Story B 是否还值得继续押注。

---

注意点：
- 重点：给出个人倾向，但保留老师判断空间。
