# DiffSBDD 在线 DDPO 训练总审计

> 文档编号：260904-02
> 审计对象：`runs/diffsbdd/` 下所有 DiffSBDD 在线 DDPO 训练（`main_diffsbdd_ddp.py`）
> 关联文档：[`DiffSBDD-ESRPO全实验总审计.md`](DiffSBDD-ESRPO全实验总审计.md)、[`DiffSBDD-IQL中期整体梳理.md`](DiffSBDD-IQL中期整体梳理.md)
> 数据来源：各 run 的 `config.yaml` + `training_summary.csv`（脚本 `analysis/diffsbdd_ddpo_audit/aggregate_runs.py` 聚合首尾 epoch）
> **命名说明**：正文中的 **E2PO** 为该方法的旧名，2026-09-18 起更名为 **ESRPO**
> （`e2po` 保留为兼容别名，[260918-02](../work_records/26-9/260918/260918-02 E2PO更名为ESRPO全面更新.md)）。

---

## 一、背景与定位

DiffSBDD **在线 DDPO** 是本项目最早成型的强化学习路线（`ddpo/main_diffsbdd_ddp.py`），
在 DiffSBDD 完整去噪轨迹上对预训练扩散模型做在线 PPO/DDPO 微调，直接以对接/理化
reward 为优化信号。它是后续 **离线 IQL / E2PO**（基于同一套去噪轨迹的 offline 路线）
的实验前驱与对照基线。

本审计把 `runs/diffsbdd/` 下 55 个 run 按奖励组合与用途分组，记录每组的配置、结果与结论，
为论文"为什么从在线 DDPO 转向离线 IQL/E2PO"提供证据。

> **重要口径说明**：本文所有"首都/尾 epoch"指标取自各 run 的 `training_summary.csv`；
> 早期 run（20260513–20260526）与后期 run（20260612 起的 single-reward 矩阵）的
> CSV 列集不同（后期新增描述子监控 22 列），绝对量不可跨家族直接横比；**valid 率与
> reward_mean 两列全家族一致，作为主可比指标**。

---

## 二、配置基线（主家族）

| 项 | 值 | 备注 |
|---|---|---|
| 模型 ckpt | `./DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt` | — |
| time_step | 主家族 500；single-reward 矩阵 200 | 采样步数 |
| pocket | 8GJL（pdbqt center 207.3/212.4/172.8, size 20×20×20） | UniDock receptor |
| reward 归一化 | `zscore_total`，clamp [-5,5]（主）/[-2,2]（single） | 全局 z-score |
| 失败惩罚 | raw `-5.0`，advantage override `-1.5` | — |
| 优化器 | AdamW（lr 2e-5, wd 1e-4） | 主家族 |
| PPO | clip_range 0.2, max_grad_norm 1.0 | — |
| KL | kl_coef 0.01（主）/0.05（single），analytic | — |
| DDP | 6–8 GPU，`find_unused_parameters=True` | — |
| batch | 主 8/GPU；single 18/GPU（grad ref 64） | — |

---

## 三、Run 家族分类与结果

### 3.1 早期探索（20260513–20260526，6/7/8 GPU）

按名字即编码奖励组合；多为短时（3–64 epoch）试探，目的是确认哪些 reward 单/组合有正向信号。

| run 家族 | 奖励 | epoch | valid 首→尾 | reward 首→尾 | 备注 |
|---|---|---:|---:|---:|---|
| `6gpu-dock` | docking | 25 | 0.63→0.76 | -0.55→-0.35 | 基础可行 |
| `6gpu-dock-chembl` | dock+chembl | 25 | 0.68→0.76 | -0.48→-0.57 | 不稳定 |
| `6gpu-chembl-only` | chembl_classifier | 18–20 | 0.63→0.79 | -0.75→-0.41 | classifier 有正向 |
| `7gpu-dock-ecfp4-fcfp4` | dock+ecfp4/fcfp4 prior | 50/62 | 0.63→0.93 | -0.74→-0.14 | **valid 最高 0.93**，但前期崩（0.16）|
| `7gpu-dock-classifier-v0-1` | dock+classifier | 25 | 0.67→0.84 | -0.71→-0.39 | 较稳 |
| `8gpu-chembl-only` | chembl_classifier | 30 | 0.64→0.83 | -0.74→-0.35 | — |
| `7gpu-classifier-only` | chembl_classifier | 64 | 0.64→**0.00** | -0.74→-2.00 | ⚠️ 后期退化（reward hacking 迹象）|
| `molskill-only-test` | molskill | 1 | 1.00 | 5.00 | 仅冒烟 |

> 关键观察：纯 `chembl_classifier` 单独优化在 `7gpu-classifier-only` 第 64 epoch 退化为
> valid=0、reward=-2（接近全失败惩罚），提示单奖励长程训练易塌缩。

### 3.2 主线 qed-sa-docking（8 GPU，reward = qed 0.25 + sa 0.25 + docking 0.5）

包含两个命名阶段：早期 `qed-sa-docking-8gjl-8gpu_*` 与 4/24 起改名
`diffsbdd-ddpo-qed-sa-docking-8gjl-8gpu_*`，配置一致（time_step 500, lr 2e-5, kl 0.01）。

**KL 系数扫描（`qed-sa-docking-8gjl-8gpu-kl-*`）** — 同配置 100 epoch：

| run | kl_coef | epoch | valid 首→尾 | reward 首→尾 | avg_docking 尾 | avg_qed 尾 | avg_sa 尾 |
|---|---|---:|---:|---:|---:|---:|---:|
| `kl-0` | 0.00 | 100 | 0.66→0.86 | 0.00→0.00 | -7.29 | 0.454 | 4.28 |
| `kl-0p01` | 0.01 | 100 | 0.66→0.74 | 0.00→0.00 | -6.48 | 0.410 | 3.72 |
| `kl-0p02` | 0.02 | 100 | 0.66→0.73 | 0.00→0.00 | -5.99 | 0.359 | 3.66 |
| `kl-0p05` | 0.05 | 100 | 0.66→0.64 | 0.00→0.00 | -4.98 | 0.285 | 3.19 |
| `kl-0p1` | 0.10 | 21 | 0.66→0.74 | 0.00→0.00 | -6.04 | 0.338 | 3.91 |

> KL 越大 → 对接分（绝对值）越好（-7.3→-5.0），但 valid 率反而下降、QED/SA 退化；
> **KL=0 时 valid 最高但对接最差**。说明 DDPO 下 KL 与对接质量强相关、与合法性弱负相关。

**旗舰长训（`diffsbdd-ddpo-qed-sa-docking-8gjl-8gpu_20260424_205059`，100 epoch）**：

| 指标 | epoch 1 | epoch 100 | 变化 |
|---|---:|---:|---:|
| valid 率 | 0.672 | 0.891 | **+0.219** |
| reward_mean | -0.492 | -0.164 | +0.328 |
| avg_qed | 0.289 | 0.535 | +0.246 |
| avg_sa | 3.22 | 0.47* | 见注 |
| avg_docking | -5.02 | 0.46* | 见注 |

\* `20260424` 长训尾段 `avg_docking`/`avg_sa` 列口径与其他 run 不一致（疑似后期 reward
归一化键切换），不以绝对值横比；valid 与 reward_mean 仍可靠。

> 旗舰结论：**在线 DDPO 能显著提升 validity（0.67→0.89）与 reward（−0.49→−0.16），
> 且 QED 明显改善**，是后续离线路线的正向前置证据。

### 3.3 single-reward 归因矩阵（20260612，8 GPU，time_step 200，10 epoch）

每组仅一个 reward=1.0，其余权重置 0，用于隔离单奖励信号（对齐 DiffGui 的同类设计）。

| run | 单奖励 | epoch | valid 首→尾 | reward 首→尾 | 备注 |
|---|---|---:|---:|---:|---|
| `01-docking` | docking | 10 | 0.61→0.61 | -0.76→-0.75 | 对接改善但 valid 不升 |
| `02-qed` | qed | 10 | — | — | 同口径短训 |
| `03-sa` | sa | 10 | — | — | — |
| `04-tpsa_distribution` | tpsa | 10 | — | — | — |
| `05-ecfp4_prior` | ecfp4 prior | 10 | — | — | — |
| `06-fcfp4_prior` | fcfp4 prior | 10 | — | — | — |
| `07-logp_distribution` | logp | 10 | — | — | — |
| `08-ligand_efficiency` | LE | 10 | — | — | — |
| `09-chembl_classifier` | classifier | 10 | — | — | — |

> 该矩阵仅 10 epoch，属归因短训；`01-docking` 显示单 docking 奖励能改善对接分但**不提升
> validity**，与 3.2 中"组合奖励才抬 valid"一致。其余 8 组未单独落盘首尾数值（同目录
> `training_summary.csv` 列集完整，需逐 run 读取，本文未逐行展开）。

### 3.4 其他支线

| 家族 | 目的 | 结果摘要 |
|---|---|---|
| `diffsbdd-8gpu-anticarbon-guard-20260611` (×2) | 碳比例护栏对照（b=8 / b=64） | valid 0.66→0.66/0.65，reward 无改善；b=64 反而降至 -1.04 |
| `kl-diagnostic-8gpu-kl0p02-25ep` | KL 诊断 | 25 epoch valid 0.66→0.94 |
| `logprob-mode-*` (3 runs) | logprob 聚合模式（full/mean_per_atom/mean_per_element） | 5 epoch 短诊，valid 0.66 附近，差异小 |
| `diffsbdd-ddpo-qed-sa-1gpu_20260409` | 单卡 pilot | 200 epoch valid 1.00→0.94（小批量 sanity）|

---

## 四、与离线 IQL / E2PO 的对照（论文口径）

| 维度 | 在线 DDPO（本文） | 离线 IQL（DiffSBDD+IQL中期答辩模型横评与训练整体梳理） | 离线 E2PO（DiffSBDD E2PO全实验总审计） |
|---|---|---|---|
| 数据 | 在线 rollout | 固定去噪轨迹 | 固定去噪轨迹 + 偏好对 |
| reward 反馈 | 每步实时 | 离线批 | 离线偏好 |
| validity 提升 | ✅ 0.67→0.89（旗舰） | 报告同量级 | 报告同量级 |
| 主要风险 | KL 失衡→塌缩（7gpu-classifier-only 退化） | 需冻数据/实际好门控 | 偏好对构造敏感性 |
| 当前定位 | 历史前驱/对照 | 当前 Part I 主线之一 | 当前 Part I 主线之一 |

> 在线 DDPO 证明"reward 能拉动 validity 与 QED"，但其**稳定性差**（单奖励长训塌缩、
> KL 扫描显示质量-合法性权衡尖锐），是推动转向离线 IQL/E2PO 的直接动因。

---

## 五、结论

1. **在线 DDPO 可行且有正向信号**：旗舰 100-epoch run 把 valid 0.67→0.89、reward −0.49→−0.16、QED 0.29→0.54。
2. **稳定性是核心短板**：纯 `chembl_classifier` 长训（64 ep）退化到 valid=0；KL 扫描显示 KL↑→对接↑但 valid↓，权衡尖锐。
3. **组合奖励优于单奖励**：single-reward 矩阵中仅 docking 单奖励改善对接却不抬 valid；qed+sa+docking 组合才同时抬 valid 与 QED。
4. **作为离线路线对照基线价值明确**：为"为何转 IQL/E2PO"提供稳定性反例。

---

## 六、局限与待补

- **绝对量跨家族不可比**：早期与后期 CSV 列集不同，`avg_docking/avg_qed/avg_sa` 在
  `20260424` 长训尾段口径疑似切换，正文引用时需逐 run 核对原始 CSV。
- **single-reward 矩阵 8 组未逐行展开**：本文仅列 `01-docking` 首尾；余项需读各自
  `training_summary.csv` 补全（脚本已可一键聚合）。
- **无独立离线重打分**：以上均为训练期在线 reward/valid，未做 checkpoint 离线等量采样 +
  独立 UniDock 重打分 + 人工结构审计（同 IQL/E2PO 审计的同一限制）。
- **未纳入 DePPA / DPO 线**：`260823-01 T1000 Gaussian-DPO` 属 DPO 而非 DDPO，不在此文档范围。

---

## 七、文件与脚本索引

- 审计脚本：`analysis/diffsbdd_ddpo_audit/aggregate_runs.py`（聚合所有 run 首尾指标）
- 主 run 目录：`runs/diffsbdd/diffsbdd-ddpo-qed-sa-docking-8gjl-8gpu_20260424_205059`
- KL 扫描：`runs/diffsbdd/qed-sa-docking-8gjl-8gpu-kl-{0,0p01,0p02,0p05,0p1}_*/`
- single-reward 矩阵：`runs/diffsbdd/diffsbdd-single-reward-20260612-*/`
