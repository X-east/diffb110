# TODO 清单

**最后更新**: 2026-05-08

## 已归档（原 TODO 1-5）

原 5 项 TODO 已全部解决或不再相关：

| 原 TODO | 当前状态 |
|----------|----------|
| 1. CLI 参数不匹配 (`--config` vs `--config_path`) | 已解决：当前主入口 `main_diffsbdd_ddp.py` 使用 `--config_path` |
| 2. Ray 脚本标记为 legacy | 已明确弃用 Ray，所有主流程使用 torchrun |
| 3. 对接后端文档混淆 | 已通过 docking_config.py / docking_registry.py 解决，4 层解耦 |
| 4. Uni-Dock 导入冲突 | 当前使用 conda 安装的 `unidock` 包，不再使用本地源 |
| 5. PyTorch 版本兼容 | 当前 `torch==2.0.1+cu118`，有警告但不阻断执行 |

---

## 当前 TODO

### 优先级 P0：训练正确性

- [ ] **KL NaN/Inf 防护** — `compute_gaussian_kl()` 缺少 NaN/Inf 保护，极端 σ 可导致训练爆炸
  - 文件：`ddpo/kl/gaussian_kl.py`
  - 修复：在 return 前加 `torch.clamp(kl_elementwise, min=0.0, max=100.0)` + `torch.where(isfinite, ...)`

- [ ] **KL 分离日志** — `calculate_loss()` 只返回混合 loss，无法单独监控 KL 值
  - 文件：`RLPF-verl_diffusion/verl_diffusion/worker/actor/edm_actor.py`
  - 修复：增加 `kl_penalty` 字段到返回 info

- [ ] **KL 启用测试** — `kl_coef > 0` 下的训练行为验证
  - 需要先完成上面两项修复

### 优先级 P1：奖励系统验证

- [ ] **加氢+Gasteiger 对 docking 效果评估** — `prepare_mol_for_docking()` 在真实训练中对对接分数的影响
- [ ] **完整训练流程验证** — 8卡 DDP 端到端跑通，指标追踪

### 优先级 P2：奖励漂移治理

- [ ] **3D 几何正常性惩罚** — 异常键长、clash、UFF 能量（不需要参考集，最易实现）
- [ ] **descriptor Mahalanobis 惩罚** — 需要离线构建参考集统计量（见 `materials/260507-3`）
- [ ] **可选判别器正常性奖励** — 见 `materials/260507-4` / `260507-6`

### 优先级 P3：代码清理

- [ ] `molecule_semantics.py` 第 156 行死代码清理（碎片分支不可达，非本次改动引入）
- [ ] `ddpo/main_diffsbdd.py` Ray 逻辑标记或移除
- [ ] `something/old-scripts/` 归档整理
