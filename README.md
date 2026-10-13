# DiffB110: RE-CmeB 口袋条件化 3D 分子生成

Git上传范围与提交检查见[版本管理规范](docs/GitHub上传与版本管理规范.md)；
外部源码获取见[参考依赖说明](docs/参考项目获取与复现.md)。

> ⚠️ **本文档的“当前路线”表述停留在 2026-08-17（离线 IQL 口径），已过时。**
> **最新管线状态与入口请看 [`docs/管线总览.md`](docs/管线总览.md)；文档索引见 [`docs/README.md`](docs/README.md)。**
> （当前主线：DiffSBDD 保留线 ESRPO；其 ChEMBL 版本见 `docs/overview/DiffSBDD-ESRPO-ChEMBL版总审计.md`。）

> 当前项目分为两部分：第一部分已确定，围绕“扩散模型横评形成基线证据 -> 完整去噪轨迹离线学习 -> 独立生成和多指标筛选 -> 最优候选进入分子动力学模拟”推进；第二部分待定。

**最后更新**: 2026-09-30（头部横幅与文档入口；正文结论仍为 2026-08-17 口径）

## 项目定位

DiffB110 当前第一部分是一个面向 RE-CmeB / CmeB 靶点的完整计算发现流程。横评仍用于保留各生成模型的靶点适配、几何合法性和对接表现证据，但不再决定当前强化学习底座。当前正式强化学习路线是：以 DiffSBDD 固定行为策略采集的完整去噪轨迹构造离线 MDP，在通过数据、actual-good 支持、held-out critic 和 actor 门槛后，比较 BC、filtered BC、AWR 与 trajectory IQL；再对新 checkpoint 独立生成的分子进行多指标筛选和人工结构审查，选择 MD 候选。

第一部分的主流程：

```text
扩散模型横评
  -> 固化横评证据（不以单项第一决定 RL 底座）
  -> DiffSBDD 完整去噪轨迹采集与 reward sidecar
  -> BC / filtered BC / AWR / trajectory IQL 离线比较
  -> IQL 后模型独立生成
  -> RDKit / docking / LE / QED-SA / ChEMBL-like / 结构病理筛选
  -> 最优候选分子
  -> 分子动力学模拟
```

工作区包含：

- DiffSBDD 生成底座、完整去噪轨迹采集与离线 IQL 研究线；
- DiffGui 参考模型、历史 PPO/DDPO 实验与横评基线；
- DDPO/PPO runtime、离线 BC/AWR/IQL 原型、reward system 与训练记录；
- ChEMBL MLP / D-MPNN 分类器与分子先验；
- UniDock docking、QED/SA/LE、环结构、化学 guard、多样性和分布漂移监控；
- `analysis/` 与 `work_records/` 中沉淀的实验图表、失败分析和路线决策。

当前第一部分的研究重点已经从“单纯提高 docking reward”转为：

1. 保持生成分子合法性和多样性；
2. 抑制高碳、低极性、复杂坏环、reward hacking 等漂移；
3. 用明确的结构 facts、classifier/D-MPNN 和 docking 共同筛选候选物；
4. 将筛选后的 top 候选分子交给分子动力学模拟，而不是只停留在训练 reward 或 docking 分数。

## 当前项目部分

| 部分 / 阶段 | 主要目录 | 当前定位 |
|---|---|---|
| 第一部分-横评证据 | `analysis/benchmark/`、`work_records/26-7/260730/` | 固化多模型起点与筛选标准；不把横评单项领先直接等同于 RL 底座选择 |
| 第一部分-DiffSBDD 离线 IQL | `ddpo/main_diffsbdd_collect_ddp.py`、`work_records/26-7/260731/`、独立 IQL worktree | 当前正式研究路线：完整轨迹、reward sidecar、冻结 split、BC/AWR/IQL 对照与 go/no-go |
| 第一部分-IQL 后生成与筛选 | `analysis/`、`scripts/docking/`、`chembl_classifier/`、`ddpo/utils/molecular_descriptors.py` | 仅用新 checkpoint 的独立生成分子进行 RDKit、PB、docking、LE、QED/SA、ChEMBL-like 和结构病理筛选 |
| 第一部分-MD 候选 | 后续 `analysis/md_simulation/` 或独立 MD 记录目录 | 从筛选结果中选择最优分子进入分子动力学模拟；MD 协议待正式确定 |
| 第二部分 | 待定 | 暂不写成确定路线 |
| 历史/保留对照 | `ddpo/diffgui/`、`ddpo/main_diffgui_docking_ddp.py`、`ddpo/main_diffsbdd_ddp.py` | DiffGui PPO/DDPO 与在线 DiffSBDD DDPO 均保留为横评、历史复现或对照，不作为当前正式训练路线 |

## 项目结构

```text
DiffB110/
├── ddpo/                    # DDPO/PPO runtime、reward、DiffSBDD/DiffGui 集成
│   ├── main_diffsbdd_collect_ddp.py # DiffSBDD 完整去噪轨迹采集入口
│   ├── main_diffsbdd_ddp.py        # 历史/对照在线 DDPO 入口
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

### DiffSBDD 离线 IQL（当前正式研究路线）

离线 IQL 使用 DiffSBDD 轨迹；当前主仓库尚无配置化正式 IQL 启动入口。开始 actor 训练前必须完成 100k 轨迹/sidecar、actual-good 与 scaffold 支持审计、冻结 split、held-out critic go/no-go；随后以相同预算比较 BC-all、filtered BC、AWR、IQL。

```bash
conda run -n dif python ddpo/main_diffsbdd_collect_ddp.py --help
```

配置化 IQL actor 入口尚未完成；在既定 gates 通过前，不应以历史 DiffGui 或在线 DDPO 命令替代。

统计口径：

- `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md`

### 历史在线 PPO/DDPO 对照

```bash
bash launch/smoke.sh

CONFIG_PATH=launch/8gpu_dock.yaml bash launch/run.sh
CONFIG_PATH=launch/7gpu_dock_classifier_v0_1.yaml bash launch/run.sh
```

核心 Python 入口：

```bash
torchrun --nproc_per_node=<N> ddpo/main_diffsbdd_ddp.py --config <launch/config.yaml>  # 历史/对照
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

- 第一部分已经确定：横评作为起点证据后，使用 DiffSBDD 完整去噪轨迹开展离线 IQL，并仅以 IQL/对照 checkpoint 的独立生成分子筛选 MD 候选。
- 当前主训练路线是 DiffSBDD 离线 IQL；其结论需由 actual-good 支持、held-out critic、独立生成、完整重对接和结构质量共同决定，不能由离线训练 loss 或 reward 单独决定。
- DiffGui PPO/DDPO 仍保留为重要历史结果、横评基线和可能的对照，但不再是当前主线。
- DiffSBDD x/h/e 大改有理论价值，但工程风险高，目前作为参考记录和方法储备。
- MLP classifier 不能单独承担药物样约束；D-MPNN、explicit ring facts、docking 和人工/规则审计需要配合。
- 好分子 replay 只能短窗口、小比例，并需要 KL/clip gate；正式混入 PPO 前必须先做 retained pool 审计。

## 文档入口

| 文件 | 内容 |
|---|---|
| **`docs/管线总览.md`** | **所有管线状态、入口、结论速查（唯一总览入口，先看这个）** |
| `docs/README.md` | docs 全量文档索引（定位 / 时效 / 维护规则） |
| `docs/overview/*.md` | 管线级实验总审计（ESRPO / ESRPO-ChEMBL / 在线 DDPO / Global DPO / IQL / DiffGui RL） |
| `docs/project_context.md` | 项目方向、分子语义、reward 体系 |
| `docs/project_paths.md` | 当前目录、输出、记录和参考项目路径（部分待更新） |
| `docs/workflow_commands.md` | 训练、生成、DiffGui、分类器、测试命令 |
| `docs/论文/大论文三章规划.md` | 论文骨架与证据索引 |
| `work_records/26-7/260731/260731-01 DiffSBDD去噪轨迹离线IQL支线全过程记录.md` | 离线 IQL 路线、门槛和比较设计（对照线） |
| `work_records/26-6/260618/260618-05 项目文件夹架构全景梳理.md` | 按文件夹架构的全项目介绍 |
