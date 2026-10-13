# DiffSBDD ESRPO-ChEMBL 版总审计（跨源 ChEMBL 偏好对）

> **定位：ESRPO 的 ChEMBL 版本。**
> 主线 ESRPO（`DiffSBDD-ESRPO全实验总审计.md`）的 winner/loser 都来自模型自身端点；
> 本版本把 **winner 换成 ChEMBL 真实分子**（R1–R5 全清 + 对接好），目标从“不违规”升级为“像 ChEMBL”。
> 最后更新：2026-09-29　状态：**进行中**（地基已验证，reward/配对仍需修正）
> 关联：`work_records/26-9/260928/`（ChEMBL 参照集与对接管线）、`docs/project_context.md`
> 代码：`analysis/crosssource_pairs_260929/`、`data/crosssource/chembl_endpoint_shards_260929/`

---

## 〇、流程总览（一页版）

**一句话**：把 **ChEMBL 真实分子当 winner**、**DiffSBDD 自身端点当 loser**，构建跨源偏好对，
让目标从“不违规”升级为“像 ChEMBL”。训练代码零改动——DPO 不需要轨迹，
`endpoint_data._load_endpoint` 只读 `sample_ids / x / categorical / nodesxsample` 四个字段。

**端到端七步**

```
① ChEMBL 179 万小分子（参照集）
      │  RDKit ETKDG + MMFF 3D
      ▼
② UniDock 对接 RE-CmeB（8GJL 口袋，vina/fast，pose 已在口袋系）
      │  快照 60/90 片 → 成功 1,155,027，分数中位 −9.16
      ▼
③ chembl_to_endpoint_shards.py → 305,000 winners / 305 shards
      │  （映射到 DiffSBDD 词表 K=10，payload N_MAX=30，与现有 store 兼容）
      ▼
④ DiffSBDD 20 万自身端点 → R1–R5 规则 + docking sidecar → loser 池
      │  （winner 取 viol=0 且 dock≤−9.0；loser 取 viol≥1 且 dock≥−6.5）
      ▼
⑤ build_crosssource_pairs.py → pairs_size 3,814 对
      │  （HA 严格相等，winner 全局唯一）
      ▼
⑥ ESRPO 训练（x-only、β=1.0、lr 5e-5、单卡、1 epoch）
      │
      ▼
⑦ checkpoint 扫描 → 生成 250 attempts → 对接（LF + 无 H）→ 四件套评估
```

**当前最佳点（size-800）与目标线**

| 指标                         | Original | **size-800** |                                  ChEMBL 参照 |
| ---------------------------- | -------: | -----------------: | -------------------------------------------: |
| 对接中位（clean & HA10–30） |   −8.97 |   **−9.50** |                                       −9.16 |
| R1–R5 全清率                |    50.4% |              41.7% | **90.60%**（HA10–30）／87.61%（全量） |

**状态**：地基已验证（HA 匹配消除简单化、对接全面超越 Original）。
**待办 5 项见 §六**；当前最主要的未解决问题是 **loser 定义漏洞导致的对接 hacking**（§4.3 发现 6）。

---

## 一、动机

ESRPO 主线（structure-matched，x-only）的偏好对全部来自 DiffSBDD 自身端点，
reward 是 `-结构违规数`。这条路线的结构性缺陷是：

1. **只有负向清单**：R1–R5 只惩罚“坏环”，而“无环/退化分子”自动满分 →
   模型可以用“生成更简单分子”来刷结构分（reward hacking）；
2. **没有“像 ChEMBL”的信号**：真实类药物化学空间（ChEMBL）的信息没有进入训练。

本工作线把 ChEMBL 真实分子（179 万小分子参照集）作为**好分子**，
从 DiffSBDD 生成端点中挑**坏分子**，构建跨源偏好对：

```
winner = ChEMBL 分子（R1–R5 全清 + 对接好，dock pose 来自 RE-CmeB 口袋）
loser  = DiffSBDD 端点（R1–R5 违规 + 对接差）
reward = -(结构违规数) - dock_weight × docking_raw        # dock_weight = 0.5
```

**关键技术结论**：DPO 不需要轨迹。`ddpo/global_dpo/endpoint_data._load_endpoint`
只从 shard 读取 `sample_ids / x / categorical / nodesxsample` 四个字段
（latents/logps 仅 DDPO 用），因此任何分子的 3D 构象只要能序列化成同样的
payload 就能作为 winner 进入现有 ESRPO trainer，**训练代码零改动**。

---

## 二、数据管道

### 2.1 ChEMBL 侧：179 万分子对接 RE-CmeB（winner 池来源）

| 项                                      | 值                                                                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| 输入                                    | `data/chembl_smallmol_reference/reference_smallmol.csv.gz`（1,790,085 条）                                                                |
| 3D 构象                                 | RDKit ETKDG + MMFF 优化（每分子独立 SDF）                                                                                                   |
| 对接                                    | UniDock 1.1.3，vina scoring / fast / num_modes=1 / seed=42，box 与五模型横评一致（受体`data/8gjl.pdbqt`，center 207.293/212.430/172.771） |
| 分区                                    | 90 片 × 20,000 分子；8 卡并行，~8 mol/s/卡                                                                                                 |
| **完成快照（2026-09-29 10:05）**  | **60/90 片**，1,173,527 分子入结果，成功 **1,155,027（98.42%）**                                                                |
| 分数分布（有效负分 1,141,012 = 97.23%） | **中位 −9.16**，mean −9.12，P5 −11.33，P95 −6.85，min −18.55                                                                     |
| 异常值                                  | 14,015 个`score ≥ 0`（1.2%，UniDock 偶发不收敛），分析时按 `score < 0` 过滤                                                            |

**流水线要点**（细节见 [260928-04](../../work_records/26-9/260928/260928-04 ChEMBL全量对接RE-CmeB流水线与UniDock调查.md)）：

- UniDock 整片随机 SIGSEGV（rc=-11）→ **降级为 500 子批**逐个重试，坏分子只污染自己的子批；`.retry.claim` 允许失败片重做一次；
- **代价（重要）**：**走 fallback 的 shard 只写分数、不保留 pose 文件**（`outputs/` 为空），
  导致约 24%（随机抽样）的 docking 成功分子没有 pose（顺序遍历时前段 60%）。
  这些分子无法作为 winner（没有口袋系 3D 坐标），转换时直接跳过；
  pose 可用性由“毒分子落在哪”决定，与化学性质近似无关，**不引入选择性偏差**；
- md10 任务会抢占 CPU，使 3D 生成从 ~75 mol/s 降到 ~30 mol/s（总时长由 3D 供给决定）。

### 2.2 ChEMBL pose → DiffSBDD endpoint shard 格式

脚本：`analysis/crosssource_pairs_260929/chembl_to_endpoint_shards.py`

1. 读 winner 的 unidock pose SDF（`docking/shard_*/outputs/{chembl_id}_out.sdf`，**坐标已在口袋系**）；
2. 只保留重原子，元素映射到 DiffSBDD 词表 `['C','N','O','S','B','Br','Cl','P','I','F']`（K=10，与参照集构建白名单一致）；
3. 序列化为与现有 store 完全兼容的 payload：

```python
{"format": "chembl_endpoint_v1",
 "sample_ids": [...],            # chembl_id
 "x": [B, 30, 3],                # N_MAX=30，多余位置补 0
 "categorical": [B, 30, 10],     # one-hot
 "nodesxsample": [B]}
```

4. 输出 `data/crosssource/chembl_endpoint_shards_260929/shard_XXXX.pt` + `winner_index.csv`
   （sample_id / shard / index / n_nodes / ring_count / docking_raw）。

**已转换规模**：305,000 winners / 305 shards（全量 winner 池 728,367，含 fallback 无 pose 者约 17 万，未跑完）。

### 2.3 DiffSBDD loser 侧

- 端点来源：`runs/diffsbdd_collect/diffsbdd-original-t1000-8gpu-100k-trajectory_20260730_194643/shards/rank_*/shard_*.pt`（20 万端点，2 万 shards）；
- 规则打分：`scripts/offline_rl/structure_pathology_rules.py`（R1–R5，与主线口径一致）；
  - 快速路径：旧 ESRPO manifest 的 `reward_loser`（dock_weight=0 时恰为 `-违规数`）可直接换算，免扫 store；
- docking：`analysis/offline_iql_labels_alidiff_unidock_affinity_8gpu_full_260824/reward_sidecar.csv`（含 `docking_raw`、`ring_count`）。

---

## 三、配对方法（`build_crosssource_pairs.py`）

### 3.1 reward 与极端筛选

```
reward = -structural_violation_count - 0.5 × docking_raw
```

| 侧                | 筛选条件                                                           |
| ----------------- | ------------------------------------------------------------------ |
| winner（ChEMBL）  | `viol == 0`（转换时已保证）且 `docking_raw ≤ -9.0`            |
| loser（DiffSBDD） | `viol ≥ 1` 且 `docking_raw ≥ -6.5`（对接失败记 0，自动满足） |

配对约束：`reward_gap ≥ min_gap`；**winner 全局唯一**（`allow_endpoint_reuse=false` 要求同一 endpoint 不得跨 pair 复用）。

### 3.2 匹配维度演化（四轮实验）

| 版本                             | 匹配键               |            对数 |                         HA 差 | 结果                                                   |
| -------------------------------- | -------------------- | --------------: | ----------------------------: | ------------------------------------------------------ |
| `pairs_fast`                   | (HA)                 |           3,890 |                             0 | 基线                                                   |
| `pairs_ring`                   | **(HA, 环数)** |           3,487 |                             0 | 环数差强制 0；**未能阻止简单化**                 |
| `pairs_random`                 | 无匹配               |          14,648 |               **+11.4** | ❌**否决**：尺寸成为主信号，模型会学“变大分子” |
| **`pairs_size`（当前）** | **仅 HA**      | **3,814** | **0.00（max\|Δ\|=0）** | ✅ 尺寸封死，环数差 +0.48 温和传递                     |

**关键决策记录**：完全随机配对会让 winner（ChEMBL）平均比 loser 大 11.4 个重原子，
尺寸因此成为可套利的自由信号，必须强制 HA 相等。HA 匹配后，ChEMBL 与 DiffSBDD 端点
的环数差为 **+0.48**（winner 中位 3 环 vs loser 2 环），双向分布
（+1: 1326 对，0: 1202 对，−1: 519 对）——“复杂度”作为温和信号传递，而非单边压制。

---

## 四、训练与实验结论

### 4.1 训练配置

- 入口 `ddpo/main_diffsbdd_global_dpo.py` + `launch/diffsbdd_esrpo_crosssource_*_smoke.yaml`
- ESRPO / x-only（v_weight=0）、β=1.0、timestep_grid=500、batch_pairs=2、
  freeze encoder/decoder + egnn 前 3 blocks（trainable 40.6%）
- lr **5e-5**（主线的 5 倍）、1 epoch、单卡 GPU 5、`--pair-count 3400`
- **检查点每 200 updates 一存**（支撑“早停点扫描”）

### 4.2 三版生成对比（250 attempts/点，统一评估口径）

|                              | Original | ring-400 |    ring-final | **size-400** | **size-800** | **size-final** |
| ---------------------------- | -------: | -------: | ------------: | -----------------: | -----------------: | -------------------: |
| 宽松 sanitize                |    67.5% |    81.6% |         82.8% |    **89.6%** |    **91.2%** |                88.4% |
| **0 环率**             |       2% |      29% | **46%** |                 4% |       **1%** |                   2% |
| ≥2 环                       |      87% |      31% |           30% |                87% |                87% |                  82% |
| 芳香环 ≥1                   |      57% |      38% |           24% |                44% |                39% |                  39% |
| 单片段率                     |      91% |      35% |           28% |                73% |                67% |                  67% |
| QED 中位                     |     0.62 |     0.39 |          0.26 |     **0.58** |               0.55 |                 0.50 |
| R1–R5 clean（of sanitized） |    50.4% |    83.8% |         80.2% |              39.3% |              41.7% |                49.8% |

### 4.3 关键发现

**发现 1：负向规则奖励必然导向退化（ring 版验证）**
R1–R5 全是环系规则 → **无环分子自动合规**。ring 匹配只在配对时强制同环数，
无法阻止推理时大面积生成 0 环分子：0 环率 29%（step400）→ **46%（final）**，
QED 0.39 → 0.26，单片段率掉到 28%（碎片化）。ring 版 84% 的 R1–R5 合规率
**大部分是退化分子刷出来的泡沫**（绝对合理产率：ring-400 68% vs Original 34%）。

**发现 2：HA 匹配彻底消除简单化（size 版验证）**
0 环率回到 1–4%（= Original 水平），≥2 环 87% 保持，QED 回升到 0.50–0.58，
sanitize 达 88–91%。**尺寸封死后，模型只能学化学质量，不能学“变小变简单”**。

**发现 3：对接全面超越 Original（size 版）**

| 口径（clean & HA10–30，LF + 无 H，同 unidock 命令） |             中位 |
| ---------------------------------------------------- | ---------------: |
| Original                                             |           −8.97 |
| size-400                                             |           −9.32 |
| **size-800（最佳点）**                         | **−9.50** |
| size-final                                           |           −9.32 |
| ChEMBL 参照（dock pose）                             |           −9.16 |

**优于 Original 0.53 kcal/mol，优于 ChEMBL 参照 0.34，且有效性 100%**（ring 版仅 48%）。

**发现 4：训练健康度反转**
ring 版“越训越崩”（step400 最佳）；size 版 **“越训越好”**
（R1R5: 39.3% → 41.7% → 49.8%，同时 0 环稳定在 2%、QED 缓降至 0.50）
→ size 匹配的信号是健康的，可以放心延长训练。

**发现 5：目标线校准（评价标准）**

| 参照                                         |    R1–R5 全清率 |
| -------------------------------------------- | ---------------: |
| ChEMBL**HA10–30**（与生成同尺寸口径） | **90.60%** |
| ChEMBL 全量（无窗口）                        |           87.61% |
| Original 生成                                |            50.4% |
| size-final                                   |            49.8% |

**“R1R5 > Original 50.4%”不是成功判据**——真正的目标是 **≈90%**（对齐真实类药物空间）。
注意目标是**匹配而非最大化**：ChEMBL 本身有 ~9.4% 的“违规”分子（大环天然产物等），
生成分布若显著高于 90% 反而说明比真实空间更保守、多样性受损。

**发现 6（未解决）：loser 定义漏洞 → 新的对接 hacking**
loser 筛选含 `docking_raw ≥ -6.5`，导致 **“R1–R5 违规但对接很好”的分子被排除在 loser 之外**。
模型因此学到“病态环系没关系，只要对接好”，并用怪结构（稠密三元环笼、`-PH2`、`SF5`、
烯醇等）自由刷对接分。2D 图证据：`analysis/crosssource_pairs_260929/size800_top16.png`
（top 分子多为 −12 ~ −15 的笼状三元环）。这解释了 size 版 R1–R5 只有 41.7%。

### 4.4 口径陷阱（务必遵守）

1. **采样制度不可混比**：五模型横评 / E2PO（现称 ESRPO）step4000 的生成由
   `run_diffsbdd_shard.py` 产生（**未传 resamplings，默认 0**），合理产率仅 28.7%；
   主线生成脚本用 `resamplings=10`（产率 80%+）。**两种制度的“合理分子对接中位”天然不可比**
   （幸存者偏差不同）。历史值 −8.46 属前者；今天所有对照统一用 resamplings=10。
2. **对接流程必须对齐**：LF（最大片段）+ **不 AddHs** + `vina/fast/num_modes=1/seed=42`。
   我方流程复现历史值成功（Original clean −8.98 ≈ 历史 −8.92），可作校准锚点。
3. **winner 坐标必须在口袋系**（用 dock pose，不能用口袋外的初始构象）。

---

## 五、产物与复现

```
# 1. ChEMBL 已对接分子的 per-molecule 规则（~100 s）
python analysis/crosssource_pairs_260929/compute_chembl_docked_rules.py --workers 18
# 2. pose -> endpoint shard
python analysis/crosssource_pairs_260929/chembl_to_endpoint_shards.py --batch-size 1000
# 3. 配对（当前推荐：仅 HA 匹配）
python analysis/crosssource_pairs_260929/build_crosssource_pairs.py \
  --loser-manifest-glob 'analysis/diffsbdd_structure_pairs_matched_260831/*pairs.csv' \
  --match-key size --min-gap 0.5 \
  --output-dir analysis/crosssource_pairs_260929/pairs_size
# 4. 训练（单卡）
CUDA_VISIBLE_DEVICES=5 python ddpo/main_diffsbdd_global_dpo.py \
  --config launch/diffsbdd_esrpo_crosssource_size_smoke.yaml \
  --allow-small-pairs --epochs 1 --learning-rate 5e-5 --pair-count 3400
# 5. 检查点生成扫描
bash analysis/crosssource_pairs_260929/scan_size_ckpts.sh
# 6. 对接评估（LF + 无 H）
python analysis/crosssource_pairs_260929/evaluate_gen_docking.py \
  --sdf OUT/crosssource_size_scan/step000800/molecules.sdf \
  --work-dir analysis/crosssource_pairs_260929/size_scan_dock/step000800
```

| 类别          | 路径                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| 脚本          | `analysis/crosssource_pairs_260929/*.py`、`*_ckpts.sh`                                           |
| winner shards | `data/crosssource/chembl_endpoint_shards_260929/`（305 shards + winner_index.csv）                 |
| 配对清单      | `analysis/crosssource_pairs_260929/{pairs_fast,pairs_ring,pairs_size,pairs_random}/`               |
| 训练产出      | `OUT/diffsbdd_esrpo_crosssource_{dockw05,ring,size}_smoke/`                                        |
| 生成扫描      | `OUT/crosssource_{ring,size}_scan/`                                                                |
| 对接评估      | `analysis/crosssource_pairs_260929/{gen_docking_gpu5,e2po_ref_dock,orig_ref_dock,size_scan_dock}/` |
| 2D 图         | `analysis/crosssource_pairs_260929/size800_{top16,random16}.png` 等                                |
| 配置          | `launch/diffsbdd_esrpo_crosssource_{dockw05,ring,size}_smoke.yaml`                                 |

---

## 六、待办（按优先级）

1. **修正 loser 定义**：改为 `viol ≥ 1`（**不限制 docking**）——病理分子即使对接好也必须是 loser；
2. **加入 ChEMBL 分布打分（正向项）**：多维性质（环数、芳香环数、HBD/HBA、TPSA、cLogP、
   Fsp3、杂原子数、QED）按在 ChEMBL 参照分布中的分位打分，作为 `+ w2 × dist_score`
   （w2 应大于 dock_weight），把目标从“不违规”换成“像 ChEMBL”，并覆盖异常基团；
3. **延长 size 训练**（信号健康，3–5 epoch），用 R1R5 距 90% 的差距 + 护栏指标选停留点；
4. 补完 ChEMBL 剩余 30 片对接（并考虑让 fallback 路径也保存 pose）；
5. 评估环节固定四件套：sanitize / R1–R5（对齐 90% 目标线）/ 形态护栏（0 环、QED、单片段）/ 对接（clean & HA10–30 口径）。
