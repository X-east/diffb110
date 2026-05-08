# DiffB110: DiffSBDD-DDPO 分子生成与优化

## 项目目标

基于 DiffSBDD（E(n)-等变3D分子扩散模型）与 DDPO/PPO 强化学习，生成对 RE-CmeB 蛋白质靶点具有高结合亲和力的类药分子。

## 当前核心能力

### 已完成

- **DDP 多卡训练** — `torchrun` + `main_diffsbdd_ddp.py`，8卡生产已验证
- **UniDock 对接奖励** — 当前 8GJL 配置主后端，AutoDock-GPU/Smina 备选
- **QED + SA 类药性奖励** — 已完成
- **奖励 z-score 归一化** — RewardNormalizer，全局统计
- **KL 散度约束** — 已接入 EDMActor.calculate_loss()，kl_coef 可配置（当前默认 0.0）
- **代码大规模解耦重构** — Docking 模块 4 层分离、分子语义统一、加氢+Gasteiger 电荷
- **配置驱动工厂模式** — runtime/build.py 从 YAML 构建全部组件
- **参考模型深拷贝** — ref_model for KL 计算，独立参数无共享
- **训练记录系统** — recording/ 模块完整

### 未完成验证项

- KL 散度约束未实际测试（kl_coef=0.0）
- 加氢 + Gasteiger 电荷对 docking 效果未在真实训练中验证
- compute_gaussian_kl() 缺少 NaN/Inf 防护
- 缺少 KL 分离日志（只有混合 loss，无法单独监控 KL 值）
- 奖励漂移（reward drift）尚未治理

## 技术路线

| 组件 | 技术 | 说明 |
|------|------|------|
| 生成底座 | DiffSBDD | E(n)-等变扩散，口袋条件生成 |
| RL 框架 | DDPO/PPO | EDMActor 策略更新，clip + KL |
| 对接后端 | UniDock（主）/ AutoDock-GPU / Smina | GPU 直连，无 Ray |
| 类药性 | QED + SA | 从统一 rdmol 计算 |
| 归一化 | z-score | RewardNormalizer，全局统计 |
| KL 约束 | N(μ_ref, σ_ref) vs N(μ_new, σ_new) | deepcopy ref_model，kl_coef 可配 |

## 下一阶段研究方向

1. **训练正确性验证** — 完整流程跑通 + KL 启用验证
2. **奖励漂移治理** — 正常分子参考流形约束（materials/ 中的方案）
3. **奖励函数更新设计** — 3D 几何惩罚 → descriptor Mahalanobis → 判别器奖励
4. **高维特征约束** — 多视图分子特征（descriptor + fingerprint + 3D）
