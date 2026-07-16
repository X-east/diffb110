# DiffGui / DiffSBDD Top50 对接与相互作用图交付包

## 核心文件

- `01_PPT/DiffGui_DiffSBDD_rep1_PBvalid_top50_original_editable_with_top1_interactions.pptx`：最终 6 页可编辑 PPT。前 4 页为两模型 Top50 原始二维结构，第 5、6 页为 PAβN 参考叠合与各模型 Top1 相互作用图。
- `02_PyMOL/8GJL_PABN_DiffGui_DiffSBDD_top1_master.pse`：总 PyMOL 会话，包含 8GJL 蛋白、PAβN、DiffGui mol001、DiffSBDD mol001 四个独立对象。
- `03_Schrodinger/8GJL_DiffGui_DiffSBDD_PBvalid_Top50_poseviewer.mae`：未压缩 Schrödinger Maestro 文件，共 101 个 entry；第 1 个为 8GJL/RE-CmeB 蛋白，随后为 DiffGui Top50 和 DiffSBDD Top50 的真实对接 pose。
- 同目录 `.maegz` 为内容相同的压缩 Maestro 版本。

## 文件夹说明

| 文件夹 | 内容 |
|---|---|
| `01_PPT` | 最终 PPT、原始四页 PPT、用户提供的参考样式 PPT |
| `02_PyMOL` | 两张 3D 对接图、分页 PSE、总 PSE、可复现 PML |
| `03_Schrodinger` | 101-entry MAEGZ、PAβN 与两个 Top1 的 2D interaction 图 |
| `04_Structures` | 受体 PDB/PDBQT、Top1 pose/complex、两模型 Top50 SDF、二维结构图 |
| `05_QA_logs` | 最终 PDF、两张新增页预览、日志、CSV/JSON 映射和摘要 |
| `06_Scripts` | 资产生成脚本和 PPT 追加脚本 |

## 分子与颜色

| 对象 | 来源 | Docking score | PyMOL 颜色 |
|---|---|---:|---|
| PAβN | `ligand_000008_out.sdf`，真实对接到 8GJL | -11.161 | 青色 |
| DiffGui mol001 | `diffgui_000762_out.sdf`，PB-valid Top1 | -15.466 | 黄色 |
| DiffSBDD mol001 | `diffsbdd_000207_out.sdf`，PB-valid Top1 | -13.850 | 黄色；总 PSE 中为洋红色以便与 DiffGui 区分 |

## 验证

- 最终 PPT：6 页；第 5、6 页均为 4 个独立图片对象和 5 个可编辑文本框。
- 字体：所有文本 run 均为 Latin/数字 Arial、East Asian Microsoft YaHei。
- PyMOL：未隐藏配体附近 cartoon，仅通过旋转、俯仰和 zoom 优化视角。
- Schrödinger 2D interaction：使用保留 SDF 键级的 receptor-ligand complex，DiffGui、DiffSBDD 和 PAβN 三张图拓扑各不相同且与源结构一致。
- MAEGZ：解压后检测到 101 个 `f_m_ct` entry。

## 打开方式

```bash
pymol 02_PyMOL/8GJL_PABN_DiffGui_DiffSBDD_top1_master.pse
```

`03_Schrodinger/8GJL_DiffGui_DiffSBDD_PBvalid_Top50_poseviewer.mae` 或对应 `.maegz` 均可直接使用 Maestro 打开。
