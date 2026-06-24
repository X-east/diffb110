"""Create a decision report for the honokiol dual-R generation strategy."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for report artifacts.")
    parser.add_argument(
        "--failure-summary",
        type=Path,
        default=Path(
            "analysis/dual_sidechain_honokiol_9pc_20260617/"
            "real_diffsbdd_inpaint_smoke/failure_reason_analysis/summary.json"
        ),
        help="JSON summary from the fixed-core inpaint failure analysis.",
    )
    parser.add_argument(
        "--postprocess-summary",
        type=Path,
        default=Path(
            "analysis/dual_sidechain_honokiol_9pc_20260617/"
            "real_diffsbdd_inpaint_smoke/postprocess_attachment/postprocess_attachment_summary.csv"
        ),
        help="CSV summary from the attachment-aware post-processing trial.",
    )
    return parser.parse_args()


def main() -> int:
    """Generate strategy matrix artifacts and a Markdown report."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failure_summary = _read_json(args.failure_summary)
    postprocess_summary = _read_csv(args.postprocess_summary)
    strategy_rows = _strategy_rows()

    _write_csv(args.out_dir / "strategy_matrix.csv", strategy_rows)
    _plot_strategy_matrix(args.out_dir / "strategy_matrix.png", strategy_rows)
    _plot_current_failure(args.out_dir / "current_failure_rates.png", failure_summary, postprocess_summary)
    _write_report(
        args.out_dir / "report.md",
        failure_summary=failure_summary,
        postprocess_summary=postprocess_summary,
        strategy_rows=strategy_rows,
        failure_summary_path=args.failure_summary,
        postprocess_summary_path=args.postprocess_summary,
    )
    print(f"wrote {args.out_dir}")
    return 0


def _read_json(path: Path) -> Any:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into rows."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to CSV."""
    if not rows:
        raise ValueError("rows must not be empty")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _strategy_rows() -> list[dict[str, Any]]:
    """Return the ranked solution candidates."""
    return [
        {
            "route": "R-group enumeration + dual placement + docking",
            "short_name": "Enumeration",
            "goal": "先得到化学一致且能对接口袋的双R候选",
            "expected_success": 4,
            "engineering_effort": 2,
            "scientific_novelty": 2,
            "risk": 2,
            "time_to_first_result": 1,
            "recommendation": "第一优先级：作为能否存在好候选的阳性基线",
        },
        {
            "route": "Attachment-aware guided inpaint",
            "short_name": "Guided inpaint",
            "goal": "在采样中约束两个锚点附近必须生成root并禁连非目标核心",
            "expected_success": 3,
            "engineering_effort": 3,
            "scientific_novelty": 3,
            "risk": 3,
            "time_to_first_result": 2,
            "recommendation": "第二优先级：比后处理更诚实，可复用现有audit",
        },
        {
            "route": "Explicit edge / shared graph diffusion",
            "short_name": "x/h/e edge",
            "goal": "显式生成或约束R1/R2的 atom type、bond type、attachment bond",
            "expected_success": 4,
            "engineering_effort": 5,
            "scientific_novelty": 5,
            "risk": 4,
            "time_to_first_result": 4,
            "recommendation": "正式方法路线：需要数据/训练/消融，不适合马上救结果",
        },
        {
            "route": "Raw fixed-core inpaint + post-hoc bond perception",
            "short_name": "Raw inpaint",
            "goal": "继续依赖坐标生成和OpenBabel/EDM后验猜键",
            "expected_success": 1,
            "engineering_effort": 1,
            "scientific_novelty": 1,
            "risk": 5,
            "time_to_first_result": 1,
            "recommendation": "停止作为主线；仅保留为负对照",
        },
        {
            "route": "Post-hoc snap attachment",
            "short_name": "Snap",
            "goal": "生成后平移组件、强行加C0/C11 attachment bond",
            "expected_success": 2,
            "engineering_effort": 2,
            "scientific_novelty": 1,
            "risk": 5,
            "time_to_first_result": 1,
            "recommendation": "只作为调试/上界测试，不能作为结果方法",
        },
    ]


def _plot_strategy_matrix(path: Path, rows: list[dict[str, Any]]) -> None:
    """Plot a compact decision matrix."""
    metrics = [
        ("expected_success", "Success"),
        ("engineering_effort", "Effort"),
        ("scientific_novelty", "Novelty"),
        ("risk", "Risk"),
        ("time_to_first_result", "Time"),
    ]
    labels = [str(row["short_name"]) for row in rows]
    values = [[float(row[key]) for key, _ in metrics] for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    image = ax.imshow(values, vmin=1, vmax=5, cmap="viridis")
    ax.set_xticks(range(len(metrics)), [name for _, name in metrics])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("Dual-R solution strategy matrix (1=low, 5=high)")
    for row_idx, row in enumerate(values):
        for col_idx, value in enumerate(row):
            color = "white" if value < 2.6 else "black"
            ax.text(col_idx, row_idx, f"{value:.0f}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_current_failure(
    path: Path,
    failure_summary: list[dict[str, Any]],
    postprocess_summary: list[dict[str, str]],
) -> None:
    """Plot the observed attachment success rates from current experiments."""
    labels = []
    left_rates = []
    right_rates = []
    dual_rates = []
    for row in failure_summary:
        total = max(int(row["total"]), 1)
        labels.append(str(row["method"]))
        left_rates.append(float(row["left_bond_presence"]) / total)
        right_rates.append(float(row["right_bond_presence"]) / total)
        dual_rates.append(0.0)

    strict_rows = [row for row in postprocess_summary if row["setting"].endswith("_strict")]
    for row in strict_rows:
        labels.append(row["setting"])
        left_rates.append(0.0)
        right_rates.append(0.0)
        dual_rates.append(float(row["dual_r_success_rate_after_postprocess"]))

    x_positions = list(range(len(labels)))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar([x - width for x in x_positions], left_rates, width=width, label="C0 attached")
    ax.bar(x_positions, right_rates, width=width, label="C11 attached")
    ax.bar([x + width for x in x_positions], dual_rates, width=width, label="dual-R success")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Rate")
    ax.set_xticks(x_positions, labels, rotation=20, ha="right")
    ax.set_title("Current fixed-core inpaint attachment failure")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(
    path: Path,
    *,
    failure_summary: list[dict[str, Any]],
    postprocess_summary: list[dict[str, str]],
    strategy_rows: list[dict[str, Any]],
    failure_summary_path: Path,
    postprocess_summary_path: Path,
) -> None:
    """Write the Markdown strategy report."""
    failure_table = _failure_table(failure_summary)
    postprocess_table = _postprocess_table(postprocess_summary)
    strategy_table = _strategy_table(strategy_rows)
    failure_summary_link = os.path.relpath(failure_summary_path.resolve(), start=path.parent.resolve())
    postprocess_summary_link = os.path.relpath(postprocess_summary_path.resolve(), start=path.parent.resolve())
    text = f"""# Honokiol 双 R 侧链生成失败后的解决路线

## 结论先行

当前结果差，不是因为 reward、分类器或后处理还不够强，而是任务表述错位：

```text
现在做的是 fixed-core inpaint + 坐标生成 + 后验猜键
实际需要的是 fixed scaffold + 两个指定 attachment site + 两条可自由构象但化学一致的 R branch
```

所以继续调 `add_n_nodes`、OpenBabel/EDM 重建阈值或 post-hoc snap，不会稳定解决问题。下一步应该分成两条线：

1. **结果线**：先做 `R-group enumeration + 双位点独立构象/对接/审计`，快速回答“是否存在两边都能适配的同一 R 基团”。
2. **方法线**：再做 `attachment-aware / explicit-edge` 生成，把 anchor、attachment bond、禁连核心、R1/R2 一致性放到生成机制里，而不是生成后猜。

## 当前失败证据

固定核已经保住了，但两侧目标 attachment 没有形成。尤其是右侧 `C11`，OpenBabel raw generation 中一次都没形成目标键。

{failure_table}

![Current fixed-core inpaint attachment failure](current_failure_rates.png)

后处理也说明了同一件事：`strict` 基本失败；`snap` 能制造少数形式成功，但那不是可信生成结果。

{postprocess_table}

原始数据：

- [failure summary]({failure_summary_link})
- [postprocess summary]({postprocess_summary_link})

## 网上方法对应关系

这件事在分子扩散里属于典型的 **atom-bond inconsistency / anchor-aware generation** 问题。

| 方法/方向 | 相关点 | 对本项目的启发 |
|---|---|---|
| DiffLinker 类 linker diffusion | 固定片段和 anchor 条件下生成连接部分 | 说明“固定片段 + anchor”应进入生成条件，而不是生成后才判断 |
| Pocket2Mol 类自回归 SBDD | 在蛋白口袋上下文中逐步加原子/键 | 更适合“从指定锚点长出来”的语义，attachment 是生成过程的一部分 |
| DecompDiff 类 scaffold/arms 分解 | 把分子拆成 scaffold、arms，并显式考虑 bond diffusion | 与“固定 honokiol 核 + 两条 arms”最接近 |
| MolDiff / MiDi 类 atom-bond 联合生成 | 直接指出坐标扩散后验猜键会造成 atom-bond inconsistency | 证明当前 raw inpaint 失败不是偶然，而是表示层缺陷 |
| RDKit constrained embedding / reaction enumeration | 可控替换 R 基团并生成构象 | 最适合先建立阳性基线和候选库 |

参考来源：

- DiffLinker: https://arxiv.org/abs/2210.05274
- Pocket2Mol: https://arxiv.org/abs/2205.07249
- DecompDiff: https://arxiv.org/abs/2403.07902
- MolDiff: https://arxiv.org/abs/2305.07508
- RDKit `ConstrainedEmbed` / virtual library enumeration: https://www.rdkit.org/docs/source/rdkit.Chem.AllChem.html
- RDKit reaction API: https://www.rdkit.org/docs/source/rdkit.Chem.rdChemReactions.html

## 本地代码可复用基础

| 本地模块 | 已有能力 | 能否用于下一步 |
|---|---|---|
| `ddpo.constraints.dual_sidechain` | phenolic OH site 识别、双 R 审计、graph signature、失败分类 | 可以直接保留 |
| `DiffSBDD/inpaint.py` | fixed-core 坐标/类型恢复，固定核化学图保留 | 可作为负对照和后续 guided inpaint 入口 |
| `DiffSBDD/edge_utils.py` | RDKit bond -> halfedge label，bond type 0/1/2/3/4 | 可用于显式边约束 |
| `DiffSBDD/reconstruction/builder.py` | `predicted_edge` / `oracle_edge` 从 halfedge 重建 RDKit | 是正式方法的关键接口 |
| `DiffSBDD/edge_diffusion` | decoupled x/h/e sampler/model/test | 有基础，但仍需训练数据、条件输入和采样约束 |
| `reference project/DiffGui/models/bond_predictor.py` | protein-conditioned ligand halfedge predictor | 可借鉴蛋白上下文进入边预测 |
| `reference project/modern_sbdd_20260531/MolSnapper/models/bond_predictor.py` | ligand node-edge bond predictor | 可借鉴显式 bond predictor 结构 |

## 策略矩阵

分数含义：1 低，5 高。`Risk` 和 `Effort` 越高越难；`Success/Novelty` 越高越有价值。

{strategy_table}

![Dual-R solution strategy matrix](strategy_matrix.png)

## 推荐执行顺序

### 第 1 步：停止把 raw inpaint 当主线

保留 fixed-core inpaint 作为负对照即可。它已经证明固定核能保住，但没有证明模型能从两个 phenolic replacement sites 长出有效双 R。

需要停止的是：

```text
生成随机新增原子 -> OpenBabel/EDM 猜键 -> 期待两边自动成为相同 R
```

这个流程没有 attachment identity、branch identity、bond order identity，也没有禁止新原子乱接母核。

### 第 2 步：先做 R 基团枚举/双位点放置/对接基线

这是最务实的一步。流程是：

```text
R library
  -> 同一个 R graph 分别接到 C0 和 C11
  -> 两边独立 constrained embedding / conformer search
  -> 固定 honokiol core 或弱约束 core
  -> MMFF/ETKDG 初筛
  -> 蛋白 clash + docking/rescore
  -> dual_sidechain audit
```

这条线的意义不是“替代扩散模型”，而是回答一个关键问题：在 4dxd_9pc 口袋里，**同一个 R 基团是否有可能同时适配两边**。如果枚举路线都找不到合理候选，继续训练 diffusion 也很可能只是扩大失败空间。

建议先用小库：

| R 类型 | 原因 |
|---|---|
| `C`, `CC`, `CO`, `CN`, `OC`, `C(O)C` | 验证 attachment 和构象流程 |
| 小芳环/杂芳环单取代 | 验证你想要的侧链复杂度上限 |
| ChEMBL/片段库中 1-anchor R group | 后续扩大化学空间 |

输出指标：

| 指标 | 判断标准 |
|---|---|
| dual audit success | 必须 100%，因为同一 R 是由构建保证的 |
| sanitize success | >90% |
| 两侧 clash rate | 低于 raw generation |
| docking score / ligand efficiency | 用于排序，不作为唯一标准 |
| R size/ring complexity | 控制不要回到“环太复杂”问题 |

### 第 3 步：做 attachment-aware guided inpaint 原型

如果还想保留 DiffSBDD 生成味道，下一步不是 snap 后处理，而是把 snap 的几何目标前移到采样过程中：

```text
C0-R1_root 距离接近 1.45-1.55 A
C11-R2_root 距离接近 1.45-1.55 A
生成原子禁止接到非 C0/C11 的固定核心原子
R1/R2 root 类型同步或有限候选同步
R1/R2 branch size 同步
失败分子有限惩罚或直接丢弃
```

这一步仍可允许失败，因为你的目标不是每个样本都成功，而是生成池里出现可信候选。

### 第 4 步：正式方法转向 explicit edge / shared graph

如果目标是论文式机制，“两边构象自由但化学结构一致”最终应落到显式图上：

```text
R1.x / R2.x        分别感知左右蛋白环境，自由构象
R1 atom labels     与 R2 atom labels 同步或共享 latent
R1 halfedge labels 与 R2 halfedge labels 同步或共享 latent
C0-R1 attachment   显式 single/aromatic-compatible bond
C11-R2 attachment  显式 single/aromatic-compatible bond
```

本地已经有 halfedge 表示和 `predicted_edge` 重建，因此不是从零开始。但要注意：现有 `edge_diffusion` 更像 decoupled 实验支线，还不是可以直接替换主 DiffSBDD generation 的成熟入口。

## 具体模块清单

| 模块 | 作用 | 是否新建 |
|---|---|---|
| `ddpo/constraints/dual_sidechain/` | 保留双 R 审计和失败分类 | 已有 |
| `scripts/build_dual_r_enumeration_baseline.py` | 从 R 库构建双取代 honokiol 候选 | 建议新建 |
| `scripts/evaluate_dual_r_enumeration_baseline.py` | clash、descriptor、audit、docking manifest | 建议新建 |
| `ddpo/constraints/dual_attachment.py` | attachment root 距离、禁连核心、branch role 规则 | 建议新建 |
| `DiffSBDD/edge_diffusion/constraints.py` | halfedge 级 must-link / cannot-link / pair-equality mask | 正式方法时新建 |
| `DiffSBDD/reconstruction/builder.py` | 从 constrained halfedge 重建 RDKit | 已有，可扩展 |

## 我建议现在马上做什么

最合理的下一步是先实现 **R-group enumeration baseline**，不要急着把强化学习或新 diffusion 机制套上去。

理由：

1. 它能用很低成本证明“同一个 R，两边不同构象”在这个蛋白环境中是否可行。
2. 它能产出真实可看的 SDF，而不是 snap 造出来的形式成功。
3. 它会给后续 diffusion 方法提供正样本、R size 上限、attachment bond 规则和失败模式。
4. 如果枚举候选都很差，就应该重新审视这个双 R 设计本身，而不是继续训练模型。

建议第一个小实验：

```text
输入：fixed_core_no_oh.sdf + 10-30 个小 R groups
输出：dual-R substituted SDF + audit CSV + clash/docking/descriptor report
筛选：sanitize_valid && dual_audit_success && no severe clash && ring_count controlled
```

然后再决定：

```text
有可行候选 -> 用这些候选反推生成约束/训练数据
没有可行候选 -> 缩小 R 空间或改设计假设
```
"""
    path.write_text(text, encoding="utf-8")


def _failure_table(rows: list[dict[str, Any]]) -> str:
    """Return Markdown table for failure summary."""
    lines = [
        "| method | total | C0 attached | C11 attached | dual success | left mean dist | right mean dist |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        total = int(row["total"])
        left = int(row["left_bond_presence"])
        right = int(row["right_bond_presence"])
        left_mean = float(row["left_min_distance_min_mean_max"][1])
        right_mean = float(row["right_min_distance_min_mean_max"][1])
        lines.append(
            f"| {row['method']} | {total} | {left}/{total} | {right}/{total} | 0/{total} | "
            f"{left_mean:.3f} A | {right_mean:.3f} A |"
        )
    return "\n".join(lines)


def _postprocess_table(rows: list[dict[str, str]]) -> str:
    """Return Markdown table for post-processing summary."""
    lines = [
        "| setting | input | postprocess ok | dual-R success | interpretation |",
        "|---|---:|---:|---:|---|",
    ]
    notes = {
        "openbabel_strict": "真实几何下两侧 attachment 均不成立",
        "edm_strict": "仅 1 个可加 attachment，但 R graph 不一致",
        "openbabel_snap": "强行移动组件也不能得到一致双 R",
        "edm_snap": "有形式成功，但属于人工修复上界",
    }
    for row in rows:
        setting = row["setting"]
        lines.append(
            f"| {setting} | {row['input_total']} | {row['postprocess_success']} | "
            f"{row['dual_r_success']} | {notes.get(setting, '')} |"
        )
    return "\n".join(lines)


def _strategy_table(rows: list[dict[str, Any]]) -> str:
    """Return Markdown table for strategy rows."""
    lines = [
        "| route | success | effort | novelty | risk | recommendation |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['short_name']} | {row['expected_success']} | {row['engineering_effort']} | "
            f"{row['scientific_novelty']} | {row['risk']} | {row['recommendation']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
