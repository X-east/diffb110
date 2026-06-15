# DiffSBDD-xhe 大改线 最终版总结

> 记录 x/h/e 三通道联合扩散的所有改动和最终结果。

---

## 一、做了什么改动

在 DiffSBDD 原有坐标 `x` + 原子类型 `h` 双通道扩散之外，新增键类型 `e` 通道：

| 改动项 | 内容 |
|--------|------|
| **数据** | 从 CrossDocked 提取 10 万分子 halfedge 数据（SDF→RDKit bond→halfedge index/type→oracle） |
| **模型** | 在 EGNN 中新增 edge embedding，键类型参与每层 message passing |
| **扩散** | 新增 edge categorical diffusion（5 类：no_bond/single/double/triple/aromatic），支持 uniform/absorb 两种 prior |
| **loss** | 新增 edge KL-divergence loss（`edge_loss_scale=100`），与原 x/h loss 联合训练 |
| **采样** | 采样链同时更新 x/h/e，用 predicted edge 替代 OpenBabel 规则成键 |
| **训练** | 支持 warm-start 原 DiffSBDD checkpoint 和 scratch 两种模式 |

### 训练配置

```yaml
diffusion_mode: xhe
edge_prior: absorb          # uniform 或 absorb
beta_schedule: segment      # 键扩散噪声调度
edge_loss_mode: diffusion_kl
edge_loss_scale: 100        # 键 loss 权重
joint_nf: 32 / hidden_nf: 128 / n_layers: 5
8 GPU, batch=2, accumulate=4 → effective batch=16
```

---

## 二、实验结果

### 2.1 Standalone edge denoiser（独立键去噪模型）

**配置**：10k 分子, 6 GPU, 3 epoch

| 指标 | 值 |
|------|----|
| 评估 real-bond F1 | **≈0.85** |
| reconstruct t100 sanitize | **≈0.78** |
| prior t100 sanitize | **0**（随机噪声无法重建有效键） |

### 2.2 x/h/e joint diffusion（联合扩散）

#### warm-start 版（从 DiffSBDD 原版 checkpoint 启动）

| 指标 | epoch 24-38 |
|------|------------|
| real_bond_acc | **0.69** |
| error_t_lig | 35-40 |
| val loss | 26-108 |

**24 epoch 即稳定**，预训练 x/h 表征对键类型学习至关重要。

#### scratch 版（从零训练）

| 阶段 | epoch | real_bond_acc | error_t_lig |
|------|------|:---:|:---:|
| 前期 | 16-50 | 0.25 | 63.9 |
| 中期 | 51-100 | 0.29 | 60.3 |
| 后期 | 101-134 | 0.31 | 57.4 |

134 epoch 后 real_bond_acc 仅 0.46，**仍在缓慢上升但远未收敛**。

### 2.3 warm-start vs scratch 对比

| 指标 | scratch (134 ep) | warm-start (38 ep) | warm-start 优势 |
|------|:---:|:---:|:---:|
| 最佳 real_bond_acc | 0.46 | **0.69** | +50% |
| 最佳 error_t_lig | 47.1 | **34.8** | -26% |
| 稳定性 | 仍在震荡 | 24 ep 已稳定 | — |

---

## 三、最终判断

1. **技术可行**：x/h/e 联合扩散在 DiffSBDD 框架上可以训练，warm-start 下 real_bond_acc 可达 0.69
2. **scratch 太慢**：从头训练 134 epoch 仅到 0.46，要达到 warm-start 水平可能需要 300+ epoch
3. **与 DiffGui 的能力重复**：DiffGui 本身已是成熟的 x/h/e 联合扩散模型，在 DiffSBDD 上补键扩散本质上是在重复已有工作
4. **采样链未完成验证**：predicted-edge reconstruction 替代 OpenBabel 成键的端到端生成测试尚未跑通
5. **创新点不足**：x/h/e 联合建模已被 DiffGui 等实现，适配到 DiffSBDD 框架缺乏独立创新

### 当前状态

- warm-start 训练已完成（38 epoch）
- scratch 长测训练到 134 epoch 后暂停
- 联合采样和 predicted-edge reconstruction 的端到端验证未完成
- **建议收缩为参考/记录，不作为主线推进**

---

## 四、关键文件索引

| 内容 | 路径 |
|------|------|
| scratch 长测分析 | `analysis/diffsbdd_xhe_diffgui_scratch_20260606_longtest/` |
| xh/xhe 对比 | `analysis/formal_xh_xhe_comparison_20260605_full_8gpu/` |
| standalone edge 10k | `analysis/edge_diffusion_ddp_20260604_10k_6gpu_test3/` |
| xhe 监督训练 | `analysis/diffsbdd_xhe_supervised_20260604_20k_6gpu/` |
| edge denoiser ablation | `analysis/edge_denoiser_ablation_summary_20260603/` |
| predicted reconstruction | `analysis/diffsbdd_xhe_predicted_reconstruction_20260605_val/` |
| warm-start 配置 | `launch/diffsbdd_xhe_diffgui_scratch_8gpu.yaml` |
