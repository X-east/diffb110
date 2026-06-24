# DiffB110: DiffGui / DiffSBDD 分子生成与强化学习优化

> 以 DiffGui 强化学习为当前主线，保留 DiffSBDD 强化学习与模型改造作为对照和备用路线的口袋特异性 3D 分子生成、奖励建模、DDPO/PPO 优化和分子质量诊断工作区。

**最后更新**: 2026-06-18

## 项目定位

DiffB110 现在不是单一模型仓库，而是一个完整研究工作区。当前主线已经转向 DiffGui 强化学习，重点是在同一靶点/同一任务上验证 reward 优化是否真实、分子是否仍然合法且结构合理。工作区包含：

- DiffGui 参考模型与解耦强化学习流程；
- DiffSBDD 生成底座、强化学习对照和模型内部改造；
- DDPO/PPO runtime、reward system、训练记录和 replay/good-pool 机制；
- ChEMBL MLP / D-MPNN 分类器与分子先验；
- UniDock docking、QED/SA/LE、环结构、化学 guard、多样性和分布漂移监控；
- `analysis/` 与 `work_records/` 中沉淀的实验图表、失败分析和路线决策。

当前研究重点已经从“单纯提高 docking reward”转为：

1. 保持生成分子合法性和多样性；
2. 抑制高碳、低极性、复杂坏环、reward hacking 等漂移；
3. 用明确的结构 facts、classifier/D-MPNN 和 docking 共同筛选候选物；
4. 对 DiffSBDD 与 DiffGui 的能力边界分别测试，而不是混成一条路线。

## 当前主线

| 主线 | 主要目录 | 当前定位 |
|---|---|---|
| DiffGui RL | `ddpo/diffgui/`、`reference project/DiffGui/`、`launch/run_diffgui_*.sh`、`OUT/diffgui-*` | 当前正式强化学习主线，入口为 `ddpo/main_diffgui_docking_ddp.py` |
| DiffSBDD RL | `ddpo/`、`DiffSBDD/`、`launch/`、`OUT/diffsbdd-*` | 保留的 DDPO/PPO 对照线，入口为 `ddpo/main_diffsbdd_ddp.py` |
| ChEMBL / D-MPNN reward | `chembl_classifier/`、`ddpo/rewards/chembl_*` | MLP 与 D-MPNN 分类器，用于 reward、筛选和 top 分子复核 |
| DiffSBDD 小改 | `DiffSBDD/equivariant_diffusion/`、`analysis/diffsbdd_small_mod/` | atom prior、geometry/bond loss、局部质量修正 |
| DiffSBDD x/h/e 大改 | `DiffSBDD/edge_diffusion/`、`analysis/diffsbdd_xhe/` | halfedge/edge diffusion 验证路线，仍属于高风险研究线 |
| 双 R / 对称侧链 | `ddpo/constraints/`、`analysis/dual_sidechain_*`、`work_records/26-6/260617*` | 已完成可行性复核；不作为当前主方法平台 |

## 项目结构

```text
DiffB110/
├── ddpo/                    # DDPO/PPO runtime、reward、DiffSBDD/DiffGui 集成
│   ├── main_diffgui_docking_ddp.py # DiffGui 当前主线 DDP 强化学习入口
│   ├── main_diffsbdd_ddp.py        # DiffSBDD 保留/对照 DDP 强化学习入口
│   ├── rewards/
│   ├── runtime/
│   ├── diffgui/runtime/
│   └── integration/
├── DiffSBDD/                # DiffSBDD 生成模型、inpaint、x/h/e、edge diffusion 探索
├── chembl_classifier/       # ChEMBL MLP 与 D-MPNN 分类器训练、评估和模型文件
├── launch/                  # 当前训练、smoke、矩阵实验的主要配置和启动脚本
├── data/                    # 靶点、配体、ChEMBL、CrossDocked、RND-EPI 等输入数据
├── OUT/                     # 正式训练、生成、smoke 的主要运行输出目录
├── analysis/                # 分析报告、CSV/JSON/SDF、图表和路线复盘
├── work_records/            # 任务级工作记录，按日期归档
├── work_days/               # 日报、周报、汇报材料
├── docs/                    # 当前项目说明、路径和工作流命令
├── materials/               # 历史设计笔记 archive
├── scripts/                 # 分子处理、docking、绘图和专项分析脚本
├── test/                    # 单元/集成测试
├── tools/                   # analysis 整理等辅助工具
└── reference project/       # DiffGui、DePPA、molskill、现代 SBDD/RL 参考项目
```

说明：

- 正式运行输出使用 `OUT/`。
- `analysis/` 是分析证据库，不作为正式训练输出根目录。
- `output/` 不再作为活跃项目目录；历史 PDF 应归入 `work_days/` 或相关记录目录。

## 主要入口

### DiffGui PPO/DDPO

DiffGui 模型路径必须使用专门的 `diffgui` 环境。不要把 DiffGui-only 依赖装进 `dif`。

```bash
bash launch/run_diffgui_docking_train_ddp_smoke.sh
bash launch/run_diffgui_deppa_last3_lr1e5_b18x8_t200_100ep_20260623.sh
bash launch/run_diffgui_ring_control_matrix_20260615.sh
```

核心 Python 入口：

```bash
torchrun --nproc_per_node=<N> ddpo/main_diffgui_docking_ddp.py --config <launch/config.yaml>
python ddpo/main_diffgui_reward_only.py --config <launch/config.yaml>
```

统计口径：

- `docs/DiffGui强化学习统计清单.md`

### DiffSBDD DDPO/PPO

```bash
bash launch/smoke.sh

CONFIG_PATH=launch/8gpu_dock.yaml bash launch/run.sh
CONFIG_PATH=launch/7gpu_dock_classifier_v0_1.yaml bash launch/run.sh
```

核心 Python 入口：

```bash
torchrun --nproc_per_node=<N> ddpo/main_diffsbdd_ddp.py --config <launch/config.yaml>
```

### DiffSBDD 生成

```bash
python DiffSBDD/generate_ligands.py <CHECKPOINT> \
  --pdbfile <PDB_FILE> \
  --outfile <OUTPUT_FILE> \
  --n_samples <N>
```

### ChEMBL / D-MPNN 分类器

```bash
conda run -n dif python chembl_classifier/train.py
conda run -n dif python chembl_classifier/evaluate.py
conda run -n dif python chembl_classifier/dmpnn/train.py --config chembl_classifier/config_dmpnn_v0_1_scaffold_strongpoly.yaml
```

## 主要证据目录

| 目录 | 用途 |
|---|---|
| `OUT/` | 每次正式训练/生成/smoke 的原始运行证据 |
| `analysis/` | 每次分析、图表、SDF、CSV、报告 |
| `work_records/` | 每次任务的动机、命令、结果和风险 |
| `work_days/` | 面向日报/周报/月报的汇总材料 |
| `launch/archive/logs/` | 批量 launcher 的历史 master log |

判断训练状态时优先看：

1. `OUT/<run>/logs/rank_00/train.log`
2. `OUT/<run>/training_summary.csv`
3. `OUT/<run>/summary.json`
4. `OUT/<run>/epochs/epoch_*/molecules.csv`
5. `OUT/<run>/checkpoints/`
6. `nvidia-smi` 进程和显存证据

## 当前重要结论

- DiffGui RL 是当前主训练平台，评估重点是同一靶点内的 reward 优化、坏环/复杂环控制、成药性、多样性和 top-N 复筛。
- DiffSBDD RL 仍保留为速度更快的对照和备用路线，但不再是当前主线。
- DiffSBDD x/h/e 大改有理论价值，但工程风险高，目前作为参考记录和方法储备。
- MLP classifier 不能单独承担药物样约束；D-MPNN、explicit ring facts、docking 和人工/规则审计需要配合。
- 好分子 replay 只能短窗口、小比例，并需要 KL/clip gate；正式混入 PPO 前必须先做 retained pool 审计。

## 文档入口

| 文件 | 内容 |
|---|---|
| `docs/project_context.md` | 当前项目方向、分子语义、reward 体系 |
| `docs/project_paths.md` | 当前目录、输出、记录和参考项目路径 |
| `docs/workflow_commands.md` | 训练、生成、DiffGui、分类器、测试命令 |
| `docs/DiffGui强化学习统计清单.md` | 当前 DiffGui RL 主线的统计与阶段总结口径 |
| `work_records/26-6/260618/260618-05 项目文件夹架构全景梳理.md` | 按文件夹架构的全项目介绍 |
