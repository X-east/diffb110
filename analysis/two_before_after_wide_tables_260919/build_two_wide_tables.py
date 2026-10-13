#!/usr/bin/env python
"""两张横向宽表：DiffGui 强化学习前后 / DiffSBDD ESRPO 训练前后。

行 = 条件（前 / 后），列 = 指标，便于直接放进幻灯片横条。

数据来源（全部从既有产物读取，不手抄）
--------------------------------------
DiffGui（每 epoch 64 次生成尝试）：
  - runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs/epoch_0001|0089/molecules.csv
      → 重原子数 / 三元环 / 螺环 / 桥环 / 可旋转键 / R1–R4 各子规则：RDKit + structure_pathology_rules 现算
  - analysis/diffgui_rl/deppa_full_20260626_training_analysis/official_epoch_summary_selected.csv
      → 对接均值 / valid 比例 / 碳比例 / 杂原子数 / LogP / TPSA / 环数 / 芳香环
  - analysis/diffgui_rl/deppa_full_20260626_training_analysis/component_epoch_summary.csv
      → QED / SA

DiffSBDD ESRPO（固定 1000 次生成尝试）：
  - analysis/e2po_mainline_reasonable_eval/reasonable_molecules_metrics.csv
      → 行 Original 与 step_004000

三套分母（表内逐列标注）：
  DiffGui 侧：可用分子（ep1 n=57 / ep89 n=61）
  DiffSBDD 侧：生成尝试 / relaxed 分子 / 合理分子（R1–R5 全清）

输出：table1_diffgui_rl_wide.{png,svg} / table2_diffsbd_esrpo_wide.{png,svg}
      wide_tables.md / wide_tables.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd_md

REPO = Path("/home/md666/data/disk6/DiffB110")
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts" / "offline_rl"))

from structure_pathology_rules import compute_structure_pathology  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DIFFGUI_RUN = REPO / "runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809/epochs"
DIFFGUI_ANALYSIS = REPO / "analysis/diffgui_rl/deppa_full_20260626_training_analysis"
ESRPO_METRICS = REPO / "analysis/e2po_mainline_reasonable_eval/reasonable_molecules_metrics.csv"

CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(CJK_FONT).exists():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=CJK_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

STYLE = {
    "good": ("#e8f6ef", "#1e8449"),
    "bad": ("#fdecea", "#c0392b"),
    "warn": ("#fef5e7", "#b9770e"),
    "neutral": ("#ffffff", "#1b1b1b"),
}


# --------------------------------------------------------------------------- #
# data assembly
# --------------------------------------------------------------------------- #
def diffgui_molecule_stats(epoch: int) -> dict:
    """Recompute per-molecule facts from the run's molecules.csv."""
    frame = pd.read_csv(DIFFGUI_RUN / f"epoch_{epoch:04d}" / "molecules.csv", low_memory=False)
    rows = []
    for _, record in frame.iterrows():
        smiles = record.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
            continue
        if pd.isna(record.get("docking_raw")):
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        facts = compute_structure_pathology(mol).as_row()
        rings = mol.GetRingInfo().AtomRings()
        rows.append(
            {
                "hac": mol.GetNumHeavyAtoms(),
                "three": int(any(len(r) == 3 for r in rings)),
                "spiro": int(rd_md.CalcNumSpiroAtoms(mol) > 0),
                "bridgehead": int(rd_md.CalcNumBridgeheadAtoms(mol) > 0),
                "rotb": int(rd_md.CalcNumRotatableBonds(mol)),
                "r1": int(facts["r1_large_ring_violation"]),
                "r2": int(facts["r2_ring_system_violation"]),
                "r4any": int(facts["r4_any_three_ring_violation"]),
                "r5": int(facts["r5_carbon_fraction_violation"]),
                "r14": int(facts["is_structurally_pathological"]),
            }
        )
    data = pd.DataFrame(rows)

    def pct(column: str) -> float:
        return float(data[column].mean() * 100)

    return {
        "n": len(data),
        "hac": float(data["hac"].mean()),
        "three_pct": pct("three"),
        "spiro_pct": pct("spiro"),
        "bridgehead_pct": pct("bridgehead"),
        "rotb": float(data["rotb"].mean()),
        "r14_pct": pct("r14"),
        "r1_pct": pct("r1"),
        "r2_pct": pct("r2"),
        "r4any_pct": pct("r4any"),
        "r5_pct": pct("r5"),
    }


TABLE1_COLUMNS = [
    "可用分子数",
    "valid /\ndocking\nsuccess %",
    "对接均值\n(kcal/mol)",
    "QED\n均值",
    "SA\n均值",
    "重原子数",
    "杂原子数",
    "碳比例",
    "LogP",
    "TPSA",
    "含三元环\n%",
    "螺环\n%",
    "桥环\n%",
    "环数",
    "芳香环",
    "可旋转键",
    "R1–R4\n%",
    "R1 大环\n%",
    "R2 环系规模\n%",
    "R4 三元环任一\n%",
    "R5 碳比例\n%",
]


def build_table1() -> tuple[list[str], list[str], list[list[str]], dict]:
    official = pd.read_csv(DIFFGUI_ANALYSIS / "official_epoch_summary_selected.csv").set_index("epoch")
    component = pd.read_csv(DIFFGUI_ANALYSIS / "component_epoch_summary.csv")
    component = component[component["source"] == "official_merged"].set_index("epoch")

    conditions = ["DiffGui ep1\n（RL 前）", "DiffGui ep89\n（RL 后）"]
    body: list[list[str]] = []
    for epoch in (1, 89):
        stats = diffgui_molecule_stats(epoch)
        row = official.loc[epoch]
        comp = component.loc[epoch]
        body.append(
            [
                f"{stats['n']}",
                f"{row['molecule_rdkit_valid_ratio'] * 100:.2f}",
                f"{row['docking_raw_mean']:.2f}",
                f"{comp['qed_raw_mean']:.3f}",
                f"{comp['sa_raw_mean']:.2f}",
                f"{stats['hac']:.1f}",
                f"{row['hetero_atom_mean']:.2f}",
                f"{row['carbon_fraction']:.3f}",
                f"{row['logp_mean']:.2f}",
                f"{row['tpsa_mean']:.1f}",
                f"{stats['three_pct']:.1f}",
                f"{stats['spiro_pct']:.1f}",
                f"{stats['bridgehead_pct']:.1f}",
                f"{row['ring_count_mean']:.2f}",
                f"{row['aromatic_ring_mean']:.2f}",
                f"{stats['rotb']:.2f}",
                f"{stats['r14_pct']:.1f}",
                f"{stats['r4b_pct']:.1f}",
                f"{stats['r4c_pct']:.1f}",
                f"{stats['r5_pct']:.1f}",
            ]
        )

    highlights = {
        (0, "valid /\ndocking\nsuccess %"): "good",
        (0, "对接均值\n(kcal/mol)"): "good",
        (0, "QED\n均值"): "good",
        (1, "含三元环\n%"): "bad",
        (1, "螺环\n%"): "bad",
        (1, "桥环\n%"): "bad",
        (1, "R4b\n%"): "bad",
        (1, "R4c\n%"): "bad",
        (1, "SA\n均值"): "warn",
        (1, "芳香环"): "warn",
        (1, "可旋转键"): "warn",
    }
    return conditions, TABLE1_COLUMNS, body, highlights


TABLE2_COLUMNS = [
    "生成尝试",
    "严格 sanitize\n%（尝试）",
    "relaxed\n分子数",
    "合理分子数",
    "合理分子产率\n%（尝试）",
    "占 relaxed\n%",
    "R1–R4\n%（relaxed）",
    "R1–R5\n%（relaxed）",
    "碎片分子\n%（尝试）",
    "对接中位\n（合理）",
    "QED 中位\n（合理）",
    "SA 中位\n（合理）",
    "重原子数\n（合理）",
    "分子量\n（合理）",
    "LogP\n（合理）",
    "TPSA\n（合理）",
    "HBD\n（合理）",
    "环数\n（合理）",
    "芳香环\n（合理）",
    "碳比例\n（合理）",
]

TABLE2_KEYS = [
    ("attempts", lambda v: f"{int(v)}"),
    ("strict_sanitize_pct_of_attempts", lambda v: f"{v:.1f}"),
    ("relaxed_molecules", lambda v: f"{int(v)}"),
    ("reasonable_count", lambda v: f"{int(v)}"),
    ("reasonable_yield_per_attempt_pct", lambda v: f"{v:.1f}"),
    ("reasonable_pct_of_relaxed", lambda v: f"{v:.1f}"),
    ("r1r4_pathology_rate_of_relaxed_pct", lambda v: f"{v:.2f}"),
    ("r1r5_pathology_rate_of_relaxed_pct", lambda v: f"{v:.2f}"),
    ("fragment_pct_of_attempts", lambda v: f"{v:.1f}"),
    ("dock_median_reasonable", lambda v: f"{v:.2f}"),
    ("qed_median_reasonable", lambda v: f"{v:.3f}"),
    ("sa_score_median_reasonable", lambda v: f"{v:.3f}"),
    ("heavy_atom_count_mean_reasonable", lambda v: f"{v:.1f}"),
    ("mw_mean_reasonable", lambda v: f"{v:.1f}"),
    ("logp_mean_reasonable", lambda v: f"{v:.2f}"),
    ("tpsa_mean_reasonable", lambda v: f"{v:.1f}"),
    ("hbd_mean_reasonable", lambda v: f"{v:.2f}"),
    ("ring_count_mean_reasonable", lambda v: f"{v:.3f}"),
    ("aromatic_ring_count_mean_reasonable", lambda v: f"{v:.3f}"),
    ("carbon_fraction_mean_reasonable", lambda v: f"{v:.3f}"),
]


def build_table2() -> tuple[list[str], list[str], list[list[str]], dict]:
    frame = pd.read_csv(ESRPO_METRICS).set_index("label")
    conditions = ["Original\n（训练前）", "step 4000\n（推荐停留点）"]
    body = [
        [fmt(frame.loc[label, key]) for key, fmt in TABLE2_KEYS]
        for label in ("Original", "step_004000")
    ]
    highlights = {
        (1, "严格 sanitize\n%（尝试）"): "good",
        (1, "合理分子产率\n%（尝试）"): "good",
        (1, "占 relaxed\n%"): "good",
        (1, "R1–R4\n%（relaxed）"): "good",
        (1, "R1–R5\n%（relaxed）"): "good",
        (1, "环数\n（合理）"): "good",
        (1, "QED 中位\n（合理）"): "warn",
        (1, "分子量\n（合理）"): "warn",
        (1, "TPSA\n（合理）"): "warn",
        (1, "碎片分子\n%（尝试）"): "bad",
    }
    return conditions, TABLE2_COLUMNS, body, highlights


# --------------------------------------------------------------------------- #
# rendering (inch-based coordinate system)
# --------------------------------------------------------------------------- #
def render(conditions, columns, body, highlights, title, note, path_stem) -> None:
    label_w, col_w = 2.70, 1.52
    header_h, row_h = 1.15, 1.05
    width = label_w + col_w * len(columns)
    height = header_h + row_h * len(body)

    fig, ax = plt.subplots(figsize=(width + 0.3, height + 1.30), dpi=300)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")

    def cell(x, y, w, h, text, *, face, color, size, weight, edge="#d4d4d4"):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0",
            linewidth=0.9, edgecolor=edge, facecolor=face))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, fontweight=weight, color=color)

    # header
    top = height
    for j, label in enumerate(columns):
        cell(label_w + j * col_w, top - header_h, col_w, header_h, label,
             face="#eef2f7", color="#1b1b1b", size=12.5, weight="bold", edge="#b8c4d0")
    cell(0, top - header_h, label_w, header_h, "条件",
         face="#eef2f7", color="#1b1b1b", size=13.5, weight="bold", edge="#b8c4d0")

    # rows
    for i, (name, row) in enumerate(zip(conditions, body)):
        y = top - header_h - (i + 1) * row_h
        cell(0, y, label_w, row_h, name, face="#f7f9fb", color="#1b1b1b",
             size=13.5, weight="bold")
        for j, value in enumerate(row):
            face, color = STYLE[highlights.get((i, columns[j]), "neutral")]
            cell(label_w + j * col_w, y, col_w, row_h, value,
                 face=face, color=color, size=15.0, weight="bold")

    ax.text(0, height + 0.55, title, ha="left", va="bottom", fontsize=19,
            fontweight="bold", color="#111111")
    ax.text(0, -0.18, note, ha="left", va="top", fontsize=12, color="#666666")

    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path_stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t1 = build_table1()
    t2 = build_table2()

    render(*t1, title="表 1  DiffGui 强化学习前后（ep1 → ep89）",
           note="绿色 = 改善，红色 = 恶化，橙 = 需注意的次优方向；分母均为该 epoch 可用分子（ep1 n=57 / ep89 n=61）",
           path_stem=OUT_DIR / "table1_diffgui_rl_wide")

    render(*t2, title="表 2  DiffSBDD ESRPO 训练前后（Original → step 4000）",
           note="绿色 = 改善，红色 = 恶化/代价，橙 = 需注意的次优方向；"
                "口径逐列标注：尝试 / relaxed 分子 / 合理分子（R1–R5 全清）",
           path_stem=OUT_DIR / "table2_diffsbd_esrpo_wide")

    lines = ["# 两张横向宽表", ""]
    for title, tables in (
        ("表 1  DiffGui 强化学习前后（ep1 → ep89）", t1),
        ("表 2  DiffSBDD ESRPO 训练前后（Original → step 4000）", t2),
    ):
        conditions, columns, body, _ = tables
        lines += [f"## {title}", "",
                  "| 条件 | " + " | ".join(c.replace("\n", " ") for c in columns) + " |",
                  "| --- | " + " | ".join("---:" for _ in columns) + " |"]
        for name, row in zip(conditions, body):
            lines.append("| " + name.replace("\n", "") + " | " + " | ".join(row) + " |")
        lines.append("")
    (OUT_DIR / "wide_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (OUT_DIR / "wide_tables.csv").open("w", encoding="utf-8-sig") as handle:
        for tag, tables in (("diffgui_rl", t1), ("diffsbdd_esrpo", t2)):
            conditions, columns, body, _ = tables
            handle.write(f"# {tag}\n")
            handle.write("condition," + ",".join(c.replace("\n", " ") for c in columns) + "\n")
            for name, row in zip(conditions, body):
                handle.write(name.replace("\n", " ") + "," + ",".join(row) + "\n")
            handle.write("\n")
    print("wrote wide tables to", OUT_DIR)


if __name__ == "__main__":
    main()
