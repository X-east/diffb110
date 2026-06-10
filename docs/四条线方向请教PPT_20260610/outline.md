# 四条线方向请教 PPT 大纲草案

> 状态：outline draft，仅用于确认页数、内容和图件映射；尚未生成 slide images / deck_spec / PPTX。  
> 目标听众：药学/交叉方向老师。  
> 目标：请老师判断哪条路线最有希望改善 RE-CmeB 靶点适配分子的生成质量，以及下一步如何做更贴近药学筛选的验证。

---

## Slide 1: 封面

- 标题：`DiffB110：RE-CmeB 靶点适配分子生成的四条探索线`
- 副标题：`生成质量、靶点适配与课题路线选择`
- 关键信息：RE-CmeB / DiffSBDD / DiffGui / DDPO-PPO / x-h-e diffusion
- 视觉想法：四条路线从同一个药学目标分叉，最后汇聚到“优先推进哪条生成路线”
- Layout role：cover
- Required images：无

---

## Slide 2: 两个底座：DiffSBDD vs DiffGui

- DiffSBDD：生成 ligand 坐标 `x` 和原子类型 `h`，最后依赖 OpenBabel / RDKit 从点云重建键；速度快，RL 工程成熟，但成键/合法性弱
- DiffGui：原生建模 atom、bond/halfedge 与坐标的联合关系，更接近 `x/h/e` 联合生成；成功写入样本的可重建合法性更强
- 关键口径：DiffGui 横评的 100% sanitize 是成功写入并进入评估的 SDF 口径；日志显示另有失败重建样本被丢弃
- 当前判断：DiffGui 的 200 步成本已可控，瓶颈主要在 PPO 更新，不是生成本身
- 我的当前倾向：优先把 DiffGui 作为下一阶段主底座，同时保留 DiffSBDD RL 作为工程对照
- Layout role：model introduction / visual comparison
- Required images：
  - **占位图，用户后续手动加入**：左侧 DiffSBDD 生成分子样貌，右侧 DiffGui 生成分子样貌；strict placeholder，PPT 生成时保留空白图框和标题，不自动补图

---

## Slide 3: 共同背景：底座质量决定 RL 难度

- 同一靶点 8GJL_C1101 六模型横评
- DiffSBDD 评估 SDF sanitize 只有 66.6%，但速度和 docking 潜力不差
- DiffGui 评估 SDF sanitize 为 100%，但日志显示为得到 500 个成功 SDF 另有 55 个 failed / reconstruction error 样本
- DiffGui 和 DiffSBDD 都会生成复杂环/稠环分子；DiffGui 在这组数据里 ring mean=4.68、芳香分子率=70.2%、carbon fraction=0.820
- 这解释了为什么 RL 在 DiffSBDD 上容易陷入大量坏分子惩罚
- Layout role：data evidence
- Required images：
  - 六模型 sanitize 对比；strict input asset；保留坐标轴、图例和数值趋势；图注必须标明“成功写入/进入评估 SDF 口径”

    ![six model sanitize](../../analysis/8gjl_c1101_benchmark_20260603_500mol_500steps/evaluation/six_models/sanitize_rate.png)

---

## Slide 4: C 线：DiffSBDD 强化学习跑通，但出现 reward hacking

- DDPO/PPO 工程已经比较完整：多 GPU、reward factory、descriptor logging、checkpoint
- 旧训练暴露出高碳化、TPSA 下降、杂原子减少
- classifier reward 太稀疏，docking 与 classifier 目标冲突
- 下一步候选：KL + ChemistryGuard + GoodMoleculePool + dense classifier shaping
- Layout role：case study / failure diagnosis
- Required images：
  - RL 描述子漂移；strict input asset；突出 carbon/TPSA/hetero drift

    ![rl descriptor drift](../../analysis/rl_high_carbon_sparse_reward_20260604/rl_descriptor_trends.png)

---

## Slide 5: A 线：内部小改低风险，但改善有限

- 目标：不改变 DiffSBDD 扩散变量，只加 atom prior / geometry refine / local mask / bond head
- 优点：工程风险低，容易接到现有 RL 主线
- 问题：不引入显式 bond diffusion，最终成键仍依赖规则重建
- 结论：适合作为 ablation 或低风险增强，但不一定能显著提高靶点适配分子质量
- Layout role：comparison / ablation
- Required images：
  - 小改 ablation sanitize；strict input asset；用于说明轻改改善有限

    ![small change sanitize](../../analysis/refinement_ablation_smoke_atomdiag_20260603/sanitize_success_by_method.png)

---

## Slide 6: B 线：x/h/e 联合扩散最根本，但风险最高

- 目标：把 bond / halfedge 作为 e 通道加入扩散过程
- 已完成：halfedge 数据链、edge denoiser、xhe joint skeleton、DiffGui 对照分析
- 关键认识：不是“最后预测键”，而是每一步联合去噪时 edge 要参与坐标更新
- 难点：loss scale、edge schedule、联合采样、与 DDPO logprob 对接
- 当前不确定性：这条线复杂度最高，且我不确定改完后是否一定有效；更适合作为高风险探索或解释性支线
- Layout role：technical architecture / risk
- Required images：
  - xhe 长测 overview；strict input asset；用于说明训练和评估仍不稳定

    ![xhe longtest overview](../../analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/longtest_overview.png)

---

## Slide 7: D 线：DiffGui RL 已跑通，但收益和成本还不清楚

- DiffGui 成功写入样本质量更强，x/h/e 联合扩散天然具备 bond 先验
- 但它也有 raw reconstruction failure 和复杂环/高碳分布偏差，RL 不能只看 sanitize
- 已解耦接入 DiffGui rollout/logprob/PPO actor/reward-only 子进程
- 200 步 8GPU 每 epoch 约 30 分钟，其中 rollout 约 3 分钟；成本可控，主要瓶颈是 PPO 更新
- 初版 docking+classifier 未见明显 reward 上升；当前已转向 ring-quality 趋势测试：last3 / lr=3e-5 / 24 epochs / docking+LE+ring_complexity+chemistry_guard
- 当前早期结果：ring/carbon 有轻微改善信号，但 docking 未同步改善；更适合作为“分布稳定化 RL”主线候选继续验证
- Layout role：experiment status
- Required images：
  - DiffGui RL training summary；strict input asset；用于说明初版收益不明显

    ![diffgui rl summary](../../OUT/diffgui-8gpu-chembl21-dock_20260609_224313/metrics/training_summary.png)

---

## Slide 8: 四条线的质量-可控性判断

- C 线：最成熟，适合作为工程对照；风险是 reward hacking 影响药学可用性
- A 线：最低风险，适合辅助；风险是改进幅度不够
- B 线：结构改造最根本，但主要是把已有 x/h/e 联合建模思想嫁接到 DiffSBDD；风险是工程太复杂且收益不确定
- D 线：底座最好、成本可控、当前最倾向；风险是 docking 与分布稳定化未同步改善
- Layout role：decision matrix
- Required images：无，建议生成四象限矩阵：`短期可控性` vs `对分子质量改善的确定性`，并用强调色标出 DiffGui；B 线标为“高风险适配/重构”，避免误标为原创机制

---

## Slide 9: 想请教老师的关键问题

- 1. 哪条线最可能真正改善 RE-CmeB 靶点适配分子质量？
- 2. 应先修底座，还是继续做 RL 靶点适配优化？
- 3. reward hacking 是否会让分子“分数好看但药学可用性更差”？
- 4. x/h/e 大改应继续做完整模型，还是收缩成 bond/reconstruction 模块？
- 5. 在 DiffGui 成本可控的情况下，是否应把 DiffGui 作为主生成底座推进？
- 6. 下一步验证应优先看 docking、LE、化学合理性、分布稳定性、scaffold 多样性，还是人工可检查候选分子？
- Layout role：Q&A / consultation
- Required images：无

---

## Slide 10: 两种可收敛课题路线

- Story A：`DiffGui 上的 RE-CmeB 靶点适配分子生成优化`
  - 主线：D 线
  - 目标：用更强 x/h/e 底座和 RL 生成更适配靶点、化学更合理的候选分子
  - C 线提供 reward hacking 诊断和 guard/pool/KL 工具链
- Story B：`显式 bond/edge 先验修复 DiffSBDD 生成质量`
  - 主线：B 线
  - 贡献：参考已有 x/h/e joint diffusion 思路，对 DiffSBDD 做 bond-aware 适配/重构；重点是验证是否能修复 DiffSBDD 成键短板，而不是声称联合扩散机制本身原创
  - RL 作为 RE-CmeB 靶点适配应用验证
- 当前个人倾向：Story A；需要老师判断是否更有利于产出可用候选分子，以及 Story B 是否值得继续押注
- Layout role：closing / recommendation
- Required images：无，建议生成双路径收敛图
