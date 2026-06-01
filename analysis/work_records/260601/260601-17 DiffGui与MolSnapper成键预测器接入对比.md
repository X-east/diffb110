# 260601-17 DiffGui 与 MolSnapper 成键预测器接入对比

## 原因

上一轮已经把 DiffSBDD 生成点云和 reconstruction/成键评估解耦。本轮目标是把 DiffGui、MolSnapper 的 pretrained bond predictor 真正接进统一 benchmark，在同一批 DiffSBDD atom cloud 上和 OpenBabel、EDM 距离规则做对比。

这一步要回答的问题是：

1. 直接借用成熟模型的 bond predictor，能不能明显修复 DiffSBDD 的成键问题？
2. 外部 predictor 是应该直接作为主成键模块，还是只作为结构参考/教师模型？
3. 如果后续训练本地 bond predictor，当前解耦框架是否已经足够支撑统一消融？

## 实现方式

### 为什么用 batch worker

`dif` 环境可以跑 DiffSBDD，但没有 `torch_geometric`，不能直接 import DiffGui/MolSnapper 的模型。已有独立环境：

- `diffgui`
- `molsnapper_modern_ref`

所以本轮采用 batch adapter：

1. `dif` 环境生成同一批 DiffSBDD atom cloud。
2. 把点云、原子类型、DiffGui 需要的 pocket protein feature 序列化到 `pkl`。
3. 通过 `CONDA_EXE` 调用对应独立环境，一次性运行 external worker。
4. worker 输出 predicted halfedge `bond_index/bond_type`。
5. 回到 `dif` 环境构建 RDKit mol，并走统一 metrics/reporting。

这种方式避免污染 `dif` 环境，也保证 DiffGui/MolSnapper 各用自己的依赖环境。

### 新增/修改代码

- `DiffSBDD/analysis/reconstruction_benchmark/methods.py`
  - 新增 `ExternalBondPredictorBatchMethod`。
  - 新增 method aliases：
    - `diffgui`, `diffgui_bond`, `diffgui_pretrained`
    - `molsnapper`, `molsnapper_bond`, `molsnapper_pretrained`
  - 新增 predicted halfedge 到 RDKit mol 的构建逻辑。
  - external worker 使用 `CONDA_EXE`，避免 `condabin/conda` 不可执行导致的 `PermissionError`。
- `DiffSBDD/analysis/reconstruction_benchmark/external_bond_predictor_worker.py`
  - 在外部 conda env 中加载 pretrained checkpoint。
  - 支持 DiffGui protein-ligand bond predictor。
  - 支持 MolSnapper ligand-only bond predictor。
  - 输出每个 sample 的 predicted halfedge。
- `DiffSBDD/analysis/generated_reconstruction_ablation.py`
  - 采样时为每个 DiffSBDD sample 附带 `context`。
  - 为 DiffGui 构造 28 维 protein atom feature：
    - 6 维元素 one-hot：H/C/N/O/S/Se
    - 21 维氨基酸类型
    - 1 维 backbone flag
  - 新增 CLI：
    - `--external-bond-device`
    - `--diffgui-env`
    - `--molsnapper-env`

## 重要适配细节

### DiffGui

DiffGui checkpoint：

- `reference project/DiffGui/ckpt/bond_trained.pt`
- iteration: 600000
- ligand node types: 16
- protein node types: 28
- edge types: 5

DiffSBDD 生成结果只有元素类型，没有芳香原子标签。因此本轮将 ligand atom type 映射到 DiffGui 的 non-aromatic 类型：

| DiffSBDD atom | DiffGui node type |
|---|---:|
| C | 2 |
| N | 4 |
| O | 6 |
| F | 8 |
| P | 9 |
| S | 11 |
| Cl | 13 |
| Br | 14 |
| I | 15 |
| B | 1 |

这会削弱 DiffGui 对芳香 bond 的预测能力，所以本轮 DiffGui 结果中 aromatic ring count 为 0 不能简单解释为 DiffGui 不会芳香，而是 DiffSBDD 缺少 aromatic atom feature，adapter 只能先做 non-aromatic 零样本映射。

### MolSnapper

MolSnapper checkpoint：

- `reference project/modern_sbdd_20260531/MolSnapper/ckpt/bondpred.pt`
- iteration: 300000
- node types: 8
- edge types: 5
- atomic numbers: C/N/O/F/P/S/Cl + mask

MolSnapper 不吃 protein context，属于 ligand-only predictor。它不支持 B/Br/I；如果 DiffSBDD 生成这些元素，adapter 会记录 `UnsupportedAtoms`。本轮 n=100 没有因 unsupported atom 成为主要错误，主要问题仍然是 valence/kekulize。

## 测试命令

语法检查：

```bash
conda run -n dif python -m py_compile DiffSBDD/analysis/generated_reconstruction_ablation.py DiffSBDD/analysis/reconstruction_benchmark/__init__.py DiffSBDD/analysis/reconstruction_benchmark/methods.py DiffSBDD/analysis/reconstruction_benchmark/metrics.py DiffSBDD/analysis/reconstruction_benchmark/reporting.py DiffSBDD/analysis/reconstruction_benchmark/external_bond_predictor_worker.py
```

结果：通过。

n=4 全方法 smoke：

```bash
conda run -n dif python DiffSBDD/analysis/generated_reconstruction_ablation.py --checkpoint DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile data/8GJL.pdb --ref-ligand C:1101 --outdir analysis/generated_reconstruction_benchmark_20260601_external_smoke_n4 --n-samples 4 --batch-size 4 --timesteps 10 --methods openbabel edm diffgui molsnapper --seed 20260601
```

结果：完成，两个 external worker 均成功返回预测。

n=100 统一对比：

```bash
conda run -n dif python DiffSBDD/analysis/generated_reconstruction_ablation.py --checkpoint DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt --pdbfile data/8GJL.pdb --ref-ligand C:1101 --outdir analysis/generated_reconstruction_benchmark_20260601_external_n100 --n-samples 100 --batch-size 20 --timesteps 10 --methods openbabel edm diffgui molsnapper --seed 20260601
```

结果：完成，墙钟时间约 271 秒。

## n=100 结果

| method | sanitize | bad bond length | fragments | largest fragment | aromatic rings | QED | SA | TPSA | sec/mol |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| openbabel | 0.700 | 0.173 | 1.170 | 0.982 | 0.260 | 0.405 | 0.398 | 35.639 | 0.0025 |
| edm | 0.170 | 0.027 | 3.610 | 0.696 | 0.010 | 0.079 | 0.102 | 8.710 | 0.0019 |
| diffgui_bond | 0.700 | 0.216 | 1.010 | 0.998 | 0.000 | 0.430 | 0.373 | 33.565 | 0.7060 |
| molsnapper_bond | 0.310 | 0.204 | 1.000 | 1.000 | 0.350 | 0.198 | 0.191 | 15.609 | 0.0690 |

错误分布：

| method | 主要错误 |
|---|---|
| openbabel | 70 成功，30 AtomValenceException |
| edm | 17 成功，83 AtomValenceException |
| diffgui_bond | 70 成功，30 AtomValenceException |
| molsnapper_bond | 31 成功，36 AtomValenceException，33 KekulizeException |

## 解释

### DiffGui adapter

DiffGui 的 sanitize success 和 OpenBabel 持平，都是 0.70；碎片指标更好，`fragments=1.010`、`largest_fragment=0.998`，说明它确实学会了更完整的 ligand graph。

但是：

- bad bond length rate 从 OpenBabel 的 0.173 升到 0.216。
- aromatic ring count 为 0。
- CPU external env 下速度约 0.706 秒/分子，比 OpenBabel 慢约 282 倍。

这说明直接零样本接入 pretrained DiffGui bond predictor 不能直接作为最终成键模块。它有结构完整性优势，但因为 DiffSBDD 输入缺 aromatic atom feature，并且坐标/元素分布与 DiffGui 训练域不完全一致，键长和芳香性并没有自然变好。

### MolSnapper adapter

MolSnapper 的连接性最完整，`fragments=1.000`、`largest_fragment=1.000`，但 sanitize 只有 0.31。它比 EDM 好，但远低于 OpenBabel/DiffGui。

主要原因：

- MolSnapper 是 ligand-only，不看 protein pocket。
- 原子类型集合比 DiffSBDD 窄。
- 它更像完整 molecular graph prior，不适合直接零样本修复 DiffSBDD 的 pocket-conditioned noisy atom cloud。

### OpenBabel / EDM 对照

OpenBabel 仍然是最强 baseline：快、sanitize 高、药物指标不差。但它的 bad bond length rate 高，且 OpenBabel warning 很多，说明它确实会硬连键。

EDM 距离规则 bad bond length 最低，但 sanitize 和碎片化太差，只能作为下界和 sanity check。

## 当前结论

直接把外部 pretrained predictor 接到 DiffSBDD 上，不能立刻替代 OpenBabel。

但这一步非常有价值，因为它证明了：

1. 解耦 benchmark 已经能支持 external predictor、OpenBabel、EDM 的同批点云统一对比。
2. DiffGui-style halfedge predictor 是值得借鉴的方向，但应该本地重训或蒸馏，而不是零样本硬套。
3. MolSnapper 的 ligand-only prior 可以作为辅助教师或对照，不适合作为当前主成键模块。
4. 后续真正应做的是 CrossDocked oracle halfedge 数据链 + 本地 bond predictor，而不是直接把 external checkpoint 当成最终答案。

## 对后续课题路线的意义

短期最稳路线：

1. 保留 OpenBabel 作为 baseline/兜底。
2. 用本轮 adapter 作为参考，实现本地 `DiffSBDDHalfedgeBondPredictor`。
3. 训练数据使用 CrossDocked oracle halfedge。
4. 输入使用 DiffSBDD 当前能稳定提供的内容：
   - ligand atom type
   - ligand coordinates
   - all-pairs halfedge index
   - 可选 pocket atom feature
5. 指标继续用本 benchmark：
   - validity/sanitize
   - bad bond length
   - fragments
   - aromatic rings
   - three-member rings
   - QED/SA/LogP/TPSA
   - sec/mol

论文上可以写成：

- 一章：DiffSBDD 生成点云的 reconstruction/bond-aware 改造。
- 一章：RL 训练时加入化学几何护栏。

外部模型的作用不是直接替代底座，而是提供成熟数据链和 halfedge predictor 设计范式。

## 输出文件

- `analysis/generated_reconstruction_benchmark_20260601_external_smoke_n4/generated_reconstruction_benchmark_report.md`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/generated_reconstruction_benchmark_report.md`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/generated_reconstruction_summary.csv`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/generated_reconstruction_per_molecule.csv`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/openbabel_sanitized.sdf`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/edm_sanitized.sdf`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/diffgui_bond_sanitized.sdf`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/molsnapper_bond_sanitized.sdf`
- `analysis/generated_reconstruction_benchmark_20260601_external_n100/*_by_method.png`

## 剩余风险

- DiffGui adapter 当前把 ligand atom 映射到 non-aromatic 类型，不能代表 DiffGui 在原生输入上的完整能力。
- External predictor 都在 CPU 环境跑，速度不代表 GPU 部署速度。
- n=100 已经比 smoke 稳，但还不是最终统计；后续本地 predictor 成型后应跑 n=500 或 n=1000。
- 当前 predicted bond reconstruction 只做 RDKit 基础建图，没有引入 DiffGui/MolSnapper 原项目完整 fix_aromatic/fix_valence 后处理；后续可作为独立方法继续加入 benchmark。
