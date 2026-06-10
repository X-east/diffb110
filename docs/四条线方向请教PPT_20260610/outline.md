# 四条线方向请教 PPT 大纲草案

> 状态：outline draft，仅用于确认页数、内容和图件映射；尚未生成 slide images / deck_spec / PPTX。  
> 目标听众：计算机方向教授。  
> 目标：请教授判断课题应收敛到哪条主线，以及下一步如何设计更像论文的技术路线。

---

## Slide 1: 封面

- 标题：`DiffB110：3D 分子扩散模型强化学习的四条探索线`
- 副标题：`底座质量、reward hacking 与课题主线选择`
- 关键信息：RE-CmeB / DiffSBDD / DiffGui / DDPO-PPO / x-h-e diffusion
- 视觉想法：四条路线从同一个目标分叉，最后汇聚到“论文主线选择”
- Layout role：cover
- Required images：无

---

## Slide 2: 两个底座：DiffSBDD vs DiffGui

- DiffSBDD：生成 ligand 坐标 `x` 和原子类型 `h`，最后依赖 OpenBabel / RDKit 从点云重建键；速度快，RL 工程成熟，但化学合法性弱
- DiffGui：原生建模 atom、bond/halfedge 与坐标的联合关系，更接近 `x/h/e` 联合生成；质量明显更强
- 当前判断：DiffGui 的 200 步成本已可控，瓶颈主要在 PPO 更新，不是生成本身
- 我的当前倾向：优先把 DiffGui 作为下一阶段主底座，同时保留 DiffSBDD RL 作为工程对照
- Layout role：model introduction / visual comparison
- Required images：
  - **占位图，用户后续手动加入**：左侧 DiffSBDD 生成分子样貌，右侧 DiffGui 生成分子样貌；strict placeholder，PPT 生成时保留空白图框和标题，不自动补图

---

## Slide 3: 共同背景：底座质量决定 RL 难度

- 同一靶点 8GJL_C1101 六模型横评
- DiffSBDD sanitize 只有 66.6%，但速度和 docking 潜力不差
- DiffGui / DePPA 说明更强结构先验可以显著改善合法性
- 这解释了为什么 RL 在 DiffSBDD 上容易陷入大量坏分子惩罚
- Layout role：data evidence
- Required images：
  - 六模型 sanitize 对比；strict input asset；保留坐标轴、图例和数值趋势

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
- 结论：适合作为 ablation 或低风险增强，但是否足够成为论文主线不确定
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

- DiffGui 底座质量更强，x/h/e 联合扩散天然具备 bond 先验
- 已解耦接入 DiffGui rollout/logprob/PPO actor/reward-only 子进程
- 200 步 8GPU 每 epoch 约 30 分钟，其中 rollout 约 3 分钟；成本可控，主要瓶颈是 PPO 更新
- 初版 docking+classifier 未见明显 reward 上升；当前启动 pure docking / lr=1e-5 / last2 诊断
- 当前倾向：如果 pure docking 能推出趋势，就优先把 DiffGui RL 作为主线继续做多目标稳定化
- Layout role：experiment status
- Required images：
  - DiffGui RL training summary；strict input asset；用于说明初版收益不明显

    ![diffgui rl summary](../../OUT/diffgui-8gpu-chembl21-dock_20260609_224313/metrics/training_summary.png)

---

## Slide 8: 四条线的风险-收益判断

- C 线：最成熟，适合作为工程主线；风险是 reward hacking
- A 线：最低风险，适合辅助；风险是改进幅度不够
- B 线：创新性最高；风险是工程太复杂且收益不确定
- D 线：底座最好、成本可控、当前最倾向；风险是 PPO 收益仍需验证
- Layout role：decision matrix
- Required images：无，建议生成四象限矩阵：`短期可控性` vs `方法创新性`，并用强调色标出 DiffGui

---

## Slide 9: 想请教老师的 6 个问题

- 1. 哪条线最像可发表的计算机方法？
- 2. 应先修底座，还是继续做 RL 稳定化？
- 3. reward hacking / sparse reward 本身能否构成论文贡献？
- 4. x/h/e 大改应继续做完整模型，还是收缩成 bond/reconstruction 模块？
- 5. 在 DiffGui 成本可控的情况下，是否应把 DiffGui 作为主底座推进？
- 6. 审稿人更看重生成质量、docking、reward 防护，还是方法新颖性？
- Layout role：Q&A / consultation
- Required images：无

---

## Slide 10: 两种可收敛论文 story

- Story A：`DiffGui 上的 reward-guided 3D 分子生成`
  - 主线：D 线
  - 贡献：把更强 x/h/e 底座接入 RL，研究 docking 优化与分布稳定化
  - C 线提供 reward hacking 诊断和 guard/pool/KL 工具链
- Story B：`显式 bond/edge 先验改善 DiffSBDD`
  - 主线：B 线
  - 贡献：x/h/e joint diffusion 或轻量 bond-aware 模块
  - RL 作为应用验证
- 当前个人倾向：Story A；需要老师判断是否足够有方法贡献，以及 Story B 是否值得继续押注
- Layout role：closing / recommendation
- Required images：无，建议生成双路径收敛图
