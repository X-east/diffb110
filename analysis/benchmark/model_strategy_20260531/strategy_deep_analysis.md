# DiffB110 下一步路线深度分析 2026-05-31

本报告只基于已有代码、已有参考项目、已有生成输出和本轮统一 RDKit 统计；没有修改 DiffB110 源码，也没有重新启动 GPU 生成。

配套数据：
- `model_generation_quality_summary.csv`：统一性质/几何/组成统计。
- `generation_timing_summary.csv`：统一速度统计。
- `diffsbdd_rl_training_key_epochs.csv`：DiffSBDD dock+classifier RL 关键 epoch。
- `molecule_level_metrics.csv`：分子级指标。
- `validity_by_model.png`、`quality_indicators_by_model.png`、`diffgui_steps_speed_curve.png`。

## 1. 总判断

最重要的结论：不要把当前问题理解为“reward 还不够强”。真正的问题是基础生成分布和化学结构先验太弱，RL 正在坏先验上做局部搜索，所以会稳定地找到 docking/classifier 的漏洞。继续只调 docking、classifier 权重或 PPO 超参，收益会越来越小。

推荐主线是“双轨制”：

1. 短期继续保留 DiffSBDD 作为高速 RL 实验底座，但必须补上结构先验：原子类型/尺寸 prior、成键 posterior、键长/角度/clash guidance、显式坏拓扑惩罚、hard negative classifier。
2. 中期把 DePPA 作为最优先的新 base / RL 原型，因为它和 DiffSBDD 同属连续坐标扩散，速度实测最好，且已有 denoising policy optimization 代码路径。
3. DiffGui 不建议立刻整体替换成 RL 主模型；它更适合作为 teacher、高质量候选生成器、成键/离散边扩散设计参考。直接接入现有 DDPO 可行，但工程量明显大，因为它同时有连续坐标、离散 atom、离散 halfedge/bond，以及 bond predictor guidance。
4. BInD、PoLiGenX/PILOT 更适合作为中长期模块库：借鉴 bond/inter edge diffusion、POVME atom prior、geometry guidance、latent conditioning、property guidance。当前不建议直接替换主线。

## 2. 实测速度和质量

### 2.1 速度

| 模型 | 规模 | steps | GPU | 墙钟时间 | 单卡等效 s/mol | 并行墙钟 s/mol | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| DiffGui | 500 | 100 | 6 | 1594.86 s | 18.56 | 3.19 | 6 卡约 1129 mol/h |
| DiffGui | 500 | 200 | 6 | 3422.50 s | 36.49 | 6.85 | 6 卡约 526 mol/h |
| DiffGui | 500 | 1000 | 6 | 13960.05 s | 160.51 | 27.92 | 6 卡约 129 mol/h |
| DePPA | 10 | 500 | 1 | 23.67 s | 2.37 | - | 本轮最快 3D pocket 候选 |
| BInD | 50 SDF | 1000 | 1 | 634.41 s | 12.69/落盘 | - | RDKit valid 27/50 |
| TargetDiff | 10 samples | 500 | 1 | 192.88 s | 19.29/sample | - | 只有 1 个 SDF |
| DiffMeta | 10 | 500 | 1 | 32.98 s | 3.30/sample | - | 2D graph，不是 3D SBDD |

DiffGui 的 batch 并行后没有单分子 212 s 那么夸张，但对 RL 仍然重：1000 steps 用 6 卡生成 500 个需要约 3.88 小时。100 steps/200 steps 的速度明显可用，尤其 200 steps 是一个更现实的质量-速度折中。

### 2.2 DiffGui 100/200/1000 质量趋势

| 指标 | 100 steps | 200 steps | 1000 steps | 解释 |
|---|---:|---:|---:|---|
| RDKit valid | 500/500 | 500/500 | 500/500 | 聚合 SDF 都可读 |
| unique rate | 100.0% | 99.8% | 98.8% | 1000 steps 反而重复略多 |
| QED mean | 0.583 | 0.587 | 0.592 | 差异不大 |
| SA score mean | 4.19 | 3.93 | 3.56 | 1000 steps 合成难度最好 |
| carbon fraction | 0.782 | 0.776 | 0.763 | 1000 steps 更接近 ChEMBL 均值 0.732 |
| hetero atoms | 5.25 | 5.43 | 5.78 | 1000 steps 更好 |
| has aromatic ring | 59.6% | 68.8% | 81.0% | steps 增加明显改善芳香性 |
| has aromatic heterocycle | 40.6% | 45.8% | 52.8% | 仍低于 ChEMBL 约 67%，但远好于 DiffSBDD |
| three-member ring | 14.6% | 10.4% | 2.4% | 1000 steps 大幅减少坏小环 |
| bad bond-length frac | 0.07% | 0.08% | 0.03% | 显式 bond/bond guidance 很有效 |

结论：100/200 steps 对质量有影响，但不是“不能用”。1000 steps 最好，主要好在芳香环、三元环、SA、杂原子更合理；200 steps 已经比当前 DiffSBDD RL 末轮强很多，速度比 1000 steps 快约 4.4 倍。若用 DiffGui 做筛选/teacher，建议：

- 快速大批量：200 steps。
- 高质量小批量 teacher / benchmark：1000 steps。
- 只做粗召回：100 steps，但要对三元环和芳香缺失做过滤。

### 2.3 DiffSBDD base / RL 的核心问题

统一统计里，DiffSBDD base filtered 5829 的表面指标并非全坏：valid 5829/5829、QED 0.546、TPSA 77.4、杂原子 7.20。但结构拓扑明显偏离药物分布：

| 指标 | ChEMBL 参考 | DiffSBDD base | DiffSBDD RL epoch25 | DiffGui 200 |
|---|---:|---:|---:|---:|
| carbon fraction | 0.732 | 0.748 | 0.794 | 0.776 |
| hetero atoms | 6.36 | 7.20 | 4.56 | 5.43 |
| has aromatic ring | 95.69% | 59.89% | 7.52% | 68.8% |
| aromatic heterocycle | 66.97% | 22.34% | 0.80% | 45.8% |
| three-member ring | 低 | 36.42% | 48.54% | 10.4% |
| fused ring | - | 76.58% | 57.92% | 96.2% |
| co-only high-C | 约很低 | 2.30% | 14.18% | 0.20% |
| no non-O hetero | 4.10% | 4.96% | 23.83% | 2.60% |

RL 末轮有效率从 66.96% 到 83.82%，docking 从 -7.66 到 -8.62，看起来在优化；但分布方向变坏：aromatic ring mean 只有 0.08，has aromatic ring 仅 7.5%，three-member ring 48.5%，carbon fraction 0.794，no non-O hetero 23.8%。这说明 docking + 当前 classifier 的组合仍在鼓励非芳香、脂肪族、多环/小环、偏疏水或低非氧杂原子的结构。

这不是 PPO 不工作，而是 PPO 工作得太“认真”：它在当前模型可达空间里找到了 reward 漏洞。

## 3. 为什么 DiffSBDD + 现有 RL 难训

DiffSBDD 的短板不是单点：

1. 没有显式 bond diffusion。最终键主要靠距离/规则重建，模型本身并不知道“这个 atom pair 应该是 single/double/aromatic/no bond”。
2. 原子类型和空间坐标耦合弱。坐标扩散能生成形状，但 atom type 分布、hetero signature、芳香性不能被强约束。
3. 键长和角度没有进入生成 policy。后处理能 sanitize，但无法把“合理键长/角度”变成采样过程中的先验。
4. reward 是终局分子级信号。docking/classifier 只在最后评分，PPO 很难学到“哪一步生成了坏三元环/坏成键/无芳香环”。
5. classifier 当前仍有 blind spot。之前诊断显示它能惩罚高碳/低杂原子，但对复杂环和三元环漏；RL 就会钻这个漏洞。

所以继续使用 DiffSBDD 可以，但必须把“化学结构先验”前移到模型采样或重建，而不能全压给最后的 reward。

## 4. 参考项目对 DiffB110 最值得借鉴什么

### 4.1 DiffGui

DiffGui 最值得借鉴的是：连续坐标 + 离散 atom + 离散 bond/halfedge 同时扩散，并且训练里有 `bond_len_loss`，采样时还有独立 bond predictor guidance。源码上：

- `configs/train/train.yml`：`categorical_space: discrete`、`diff_atom`、`diff_bond`、`bond_len_loss: True`、1000 timesteps。
- `models/model.py`：对 node type 和 halfedge type 走 categorical transition；采样中对 node/halfedge 做 `log_sample_categorical`；bond predictor guidance 对 position 加梯度修正。
- `models/bond_predictor.py`：独立预测 0/1/2/3/aromatic bond type，真实键权重大于 no-bond。
- `scripts/sample.py`：生成后用 predicted edges 重建，过滤 fragment/B，再做 Vina score-only 排序。

这解释了为什么 DiffGui 的键长非常好：本轮 bad bond-length frac 约 0.03%-0.08%，比 DiffSBDD RL 末轮 4.12% 好很多。

但 DiffGui 直接接现有 DDPO 不轻：

- action 同时包含连续 `x` 和离散 atom/bond；PPO logprob 必须包含 Normal 坐标项、categorical atom 项、categorical halfedge 项。
- DiffGui 采样里会“直到成功为止”并丢弃失败分子；RL 不能这么做，否则会有 selection bias，失败也要进入 batch 并拿有限惩罚。
- bond predictor guidance 是对坐标的确定性梯度修正。若它属于 policy，要定义 logprob；若它属于 environment transform，PPO 梯度和真实采样分布不完全一致。
- 每步完整 EGNN + halfedge categorical + bond predictor guidance，训练吞吐会显著慢于 DiffSBDD。
- 源码中 `classifier_free()` 创建了 `batch_lab_zero`，但 uncond forward 仍传 `batch_lab`，这会让 unconditional 分支不是真正 unconditional；若要用它做 property-guided RL 或条件采样，应先修这个点。

建议定位：DiffGui 先做 teacher / benchmark / 高质量候选池 / 成键模块参考，不要一开始就作为 PPO 主模型。

### 4.2 DePPA

DePPA 是当前最值得优先推进的新 base：

- 实测 10/10 valid，500 steps 外层 23.67 s，约 2.37 s/mol。
- `crossdock_fullatom_cond.yml` 是 500-step full-atom pocket conditioning，EGNN 6 层，结构比 DiffGui 轻。
- `conditional_model.py` 已有 `sample_p_zs_given_zt_rl()` 和 `rl_given_pocket()`，记录 reverse denoising log_prob。
- `lightning_modules.py` 已经实现 reward、advantage、PPO clipped objective、pretrained KL。
- 它仍主要靠 OpenBabel/EDM distance thresholds 成键，但 pipeline 已有 sanitize、largest fragment、UFF relax、3-3 ring filter。

这和现有 DiffSBDD DDPO 的连续扩散路径最接近。若要换底座，DePPA 的改造量最小，速度也最现实。

### 4.3 BInD

BInD 直接做主模型暂不理想：本轮 50 个 SDF 只有 27 个 RDKit valid，外层 634 s。但它的设计非常值得拆模块：

- 同时扩散 ligand atom `h`、coordinate `x`、ligand bond edge `e`、protein-ligand interaction edge `i`。
- loss 权重里 atom/bond/interaction 都是 100，位置是 1，说明它把离散结构当成核心任务。
- 采样有 POVME atom prior、repaint、bond distance guidance、bond angle guidance、inter distance guidance、steric guidance。

建议借鉴：把 BInD 的 guidance 思路移植到 DiffSBDD/DePPA，而不是整体迁移 BInD。

### 4.4 PoLiGenX / PILOT

本地有完整源码，但没有找到可用生成 checkpoint。脚本和 notebook 都指向 `best_valid.ckpt`，实际搜索未发现 PoLiGenX/PILOT 的本地权重。有源码不等于有权重；目前不能形成速度/质量基准。

方法上很有价值：

- `atoms_continuous: false`、`bonds_continuous: false`，是离散 atom/bond diffusion。
- `ligand_pocket_interaction: true`。
- `use_latent_encoder: true`，latent_dim 128。
- `joint_property_prediction: true`，默认训练 SA/property head。
- `timesteps: 500`，`prior_n_atoms: targetdiff`，EQGAT 12 层。

建议定位：中长期参考 latent-conditioned pocket diffusion、property guidance、PoseBusters 几何评估。拿到权重前不要作为工程主线。

### 4.5 TargetDiff / DiffMeta

TargetDiff 在本口袋上 10 samples 只写出 1 个 SDF，而且是多片段/重建差，不建议直接替换主模型。可以参考 pocket-conditioned atom/coord diffusion 架构，但必须另配 bond predictor/重建约束。

DiffMeta 2D graph diffusion 速度尚可，8/10 valid，QED 高，但不是 3D pocket-conditioned。适合作为 2D scaffold/property teacher 或 classifier/negative mining 数据源，不适合作为 DiffB110 主生成器。

## 5. 下一步技术路线

### 路线 A：继续 DiffSBDD，但补结构先验

这是最快可落地、也最稳的短期路线。

优先级 1：成键从规则推断升级成 bond posterior。

- 训练一个 `P(bond_type | atom_i, atom_j, x_i, x_j, pocket context)` 的 bond predictor。
- 数据来自 CrossDocked/PDBBind/ChEMBL 3D conformers，类别为 no/single/double/triple/aromatic。
- 先不改 DiffSBDD 扩散主干，只在重建时用 bond posterior + 距离 prior + valence constraints 做最大权匹配/约束解码。
- 失败分子不丢弃，记录 failure type，并给 finite penalty。

优先级 2：采样阶段加入轻量几何 guidance。

- bond distance guidance：已预测成键的 pair 距离拉回合理区间。
- bond angle guidance：只对高置信局部三元结构/两跳结构启用，避免太慢。
- steric clash guidance：排斥 ligand 内部 clash 和 pocket clash。
- 每 k 步启用一次，或只在后 20%-30% denoising steps 启用，控制速度。

优先级 3：修 reward，不再让 docking 单独主导。

建议把 reward 分成三层：

1. hard finite penalties：invalid、fragment、多片段、bad bond length、internal clash、three-member ring、高 carbon、低非氧杂原子、无芳香环且复杂饱和环。
2. soft distribution rewards：atom type/hetero signature/aromatic ring/size/logP/TPSA 与 ChEMBL strict pool 或目标口袋已知活性分布对齐。
3. expensive rewards：docking、classifier，只对通过结构门控的分子给高权重。

当前 classifier reward 应从饱和的 `[-1, 1]` 地板改成 logit/temperature 版；并补 hard negatives：three-member ring、ring_complex、no_aromatic+fused、co-only high-C、top-reward bad。

优先级 4：原子类型/尺寸 prior。

- ligand atom count 不要只用原 DiffSBDD 先验，改成 pocket-conditioned + ChEMBL strict 分布。
- 对 N/S/halogen/aromatic heterocycle 做分层采样或轻量 prior correction。
- 对训练 batch 做分布监控：has_N、has_halogen、aromatic_heterocycle、carbon_fraction、three_member_ring，不能只看 QED/docking。

这条路线的好处是速度保留，和当前 DDPO 记录/多 GPU/奖励系统兼容；坏处是上限仍受 DiffSBDD 主干限制。

### 路线 B：DePPA 作为新 base

这是我认为最值得马上开一个分支验证的路线。

改造步骤：

1. 先把 DePPA 的 generator 包成和 `DiffSBDDModel.sample()` 类似的 adapter，输出 rdmol/sdf/smiles/trajectory/logprob。
2. 接入现有 `CompositeReward`，先不改 reward。
3. 用 DePPA 自带 `rl_given_pocket()` / `sample_p_zs_given_zt_rl()` 思路打通 logprob 与 PPO update。
4. 对比相同 reward 下 DiffSBDD vs DePPA 的：valid、aromatic、three-member、bond length、docking、classifier、秒/mol。
5. 若 DePPA 分布明显更好，再把 DiffSBDD 的训练资源逐步迁移过去。

风险：

- DePPA 仍没有显式 bond diffusion，长期也需要 bond predictor/geometry gate。
- 需要把 Lightning 风格训练逻辑和现有 DDP/DDPO 记录系统对齐。
- 需要确认在 RE-CmeB 口袋上批量 500/1000 个时速度仍稳定，本轮只有 10 个。

但整体风险低于直接 DiffGui-PPO。

### 路线 C：DiffGui 作为 teacher 或中长期 RL

短期使用方式：

- 用 200 steps 生成大批候选，1000 steps 生成高质量 teacher 小批量。
- 用现有 docking/classifier/几何指标重排，得到高质量 pseudo-positive。
- 蒸馏给 DiffSBDD/DePPA：atom type 分布、bond predictor、几何 guidance、teacher-selected 口袋姿态。

中长期若要 DiffGui-RL：

1. 修 `classifier_free()` 的 unconditional label。
2. 改采样 API：不能只返回最终 mol，要返回每步 pos/node/halfedge action、old logprob、time、mask。
3. 实现 `evaluate_logprob()`：对旧轨迹重算当前 policy 下 continuous + categorical logprob。
4. 定义 bond guidance 在 PPO 中的地位：禁用、固定 environment transform、或单独建 differentiable policy。
5. 失败分子进入 batch，不能 resample 到成功后才算。
6. 优先只对 100/200 steps 做 RL，1000 steps 作为 teacher/reference。

可行，但不是“把现有 DiffSBDD DDPO 换个 model 调用”那么简单。

## 6. 推荐执行顺序

第一阶段，1-2 周：

1. 固化统一评估：每次生成都输出本报告这些指标，尤其 aromatic/hetero/three-member/bond length/clash。
2. 在 DiffSBDD 主线上加 hard filter/finite penalty 设计方案，不先重训大模型。
3. 训练或接入一个 bond predictor，先只用于重建/过滤。
4. 用 DiffGui 200 steps 生成一个 teacher pool，作为结构质量上限和 hard negative/positive 数据源。

第二阶段，2-4 周：

1. 开 DePPA adapter 分支，跑同一 RE-CmeB 口袋的 500/1000 分子基准。
2. 接现有 classifier+docking reward 做最小 PPO 或 reward-guided fine-tuning。
3. 比较 DePPA-RL 与 DiffSBDD-RL 的分布漂移；如果 DePPA 明显更稳，迁移主线。

第三阶段，4 周以上：

1. 把 bond diffusion / edge diffusion 正式并入主模型，优先参考 DiffGui 和 BInD。
2. 做 DiffGui/PoLiGenX 风格的离散 atom/bond + 连续坐标 joint policy。
3. 考虑 latent-conditioned pocket generator，用 PoLiGenX/PILOT 思路做 property/shape control。

## 7. 具体决策

如果目标是“最快看到更合理分子”：继续 DiffSBDD，但马上补 bond predictor + hard penalties + geometry gate，同时用 DiffGui 200/1000 steps 做 teacher。

如果目标是“换掉落后的基础模型”：优先 DePPA，不是 DiffGui。DePPA 更快、更接近现有连续扩散 RL；DiffGui 更强但接 PPO 更复杂。

如果目标是“最高质量分子候选”：DiffGui 1000 steps 可作为小批量高质量 generator；DiffGui 200 steps 可作为中等吞吐候选池。

如果目标是“长期方法先进性”：最终方向应该是类似 DiffGui/PoLiGenX/BInD 的 joint atom-bond-coordinate diffusion，而不是继续只靠坐标扩散 + 后处理规则成键。

