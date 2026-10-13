#!/usr/bin/env python
"""DiffGui RL ep1 / ep89 与 ChEMBL-36 类药参照的横向宽表。

行（竖排条件）：
    DiffGui ep1   —— 强化学习起点
    DiffGui ep89  —— 强化学习终点（该 run 实际完成 89 个完整 epoch）
    ChEMBL36 参照 —— 类药化学空间的外部参照（"标准样子"）

列（横向铺开）：
    n、对接均值、含三元环%、螺环%、桥环%、环数均值、芳香环均值、可旋转键、
    重原子数、R1%、R2%、R3%、R4%、R5%、R1–R4%、R1–R5%

R4 口径：三元环子规则（R4a 含杂原子 / R4b 过度连接 / R4c 螺环或稠合）的**并集**，
即 ``r4_any_three_ring_violation``，不等于三者相加。R4a/R4b/R4c 的分项仍写入
``wide_table.json``，便于需要拆开时查阅。

口径（两侧使用同一个实现，不混用）：
  - ChEMBL：``data/chembl/chembl_36_molecules.csv.gz``
      = chembl_structure_pipeline 标准化 + mol_wt 150–500 + heavy_atoms 10–30，
      再做 largest-fragment 归约（本文件 fragment_reduced = 0，等价 no-op）。
  - DiffGui：``runs/diffgui-deppa-full-.../epochs/epoch_XXXX/molecules.csv``，
      取 canonical_smiles 非空且 docking_raw 非空的分子。
  - 环系统计与 R1–R5 判定统一由 RDKit + ``scripts/offline_rl/structure_pathology_rules.py``
    计算；ChEMBL 侧没有对接，故对接列为空。

输出：wide_table.csv / wide_table.md / wide_table.png / wide_table.json / wide_table.log
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors as rd_md

REPO = Path("/home/md666/data/disk6/DiffB110")
sys.path.insert(0, str(REPO / "scripts" / "offline_rl"))

from structure_pathology_rules import compute_structure_pathology  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = Path(__file__).resolve().parent
CHEMBL_CSV = REPO / "data/chembl" / "chembl_36_molecules.csv.gz"
DIFFGUI_RUN = REPO / "runs" / "diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809" / "epochs"
WORKERS = 18
DIFFGUI_EPOCHS = (1, 89)

LOGGER = logging.getLogger("wide-table")

RULE_FIELDS = (
    "r1_large_ring_violation",
    "r2_ring_system_violation",
    "r3_four_ring_violation",
    "r4a_three_ring_hetero_violation",
    "r4b_three_ring_over_attached_violation",
    "r4c_three_ring_spiro_or_fused_violation",
    "r5_carbon_fraction_violation",
)


# --------------------------------------------------------------------------- #
# metric computation (one implementation for both sides)
# --------------------------------------------------------------------------- #
def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    """Largest connected component, mirroring scripts/offline_rl labelling."""
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumAtoms())


def metrics_from_smiles(smiles: str, *, reduce_fragment: bool) -> dict | None:
    """Compute every table metric for one SMILES, or None when unparsable."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if reduce_fragment:
        mol = _largest_fragment(mol)

    ring_atom_rings = mol.GetRingInfo().AtomRings()
    facts = compute_structure_pathology(mol)
    row = facts.as_row()
    out = {
        "hac": mol.GetNumHeavyAtoms(),
        "three_ring": int(any(len(r) == 3 for r in ring_atom_rings)),
        "spiro": int(rd_md.CalcNumSpiroAtoms(mol) > 0),
        "bridgehead": int(rd_md.CalcNumBridgeheadAtoms(mol) > 0),
        "ring_count": mol.GetRingInfo().NumRings(),
        "aromatic_ring_count": rd_md.CalcNumAromaticRings(mol),
        "rotatable_bonds": rd_md.CalcNumRotatableBonds(mol),
        "r4_any": int(row["r4_any_three_ring_violation"]),
        "r14": int(row["is_structurally_pathological"]),
        "r15": int(row["is_pathological"]),
    }
    for field in RULE_FIELDS:
        out[field] = int(row[field])
    return out


def _worker(smiles: str) -> dict | None:
    return metrics_from_smiles(smiles, reduce_fragment=True)


def collect_chembl() -> tuple[list[dict], dict]:
    LOGGER.info("reading %s", CHEMBL_CSV.relative_to(REPO))
    frame = pd.read_csv(CHEMBL_CSV, usecols=["canonical_smiles"])
    smiles_list = frame["canonical_smiles"].astype(str).tolist()
    LOGGER.info("ChEMBL rows=%d; evaluating with %d workers", len(smiles_list), WORKERS)
    with Pool(processes=WORKERS) as pool:
        results = pool.map(_worker, smiles_list, chunksize=2048)
    scored = [r for r in results if r is not None]
    meta = {
        "source": str(CHEMBL_CSV.relative_to(REPO)),
        "rows": len(smiles_list),
        "parse_failures": len(smiles_list) - len(scored),
        "n": len(scored),
    }
    LOGGER.info("ChEMBL scored=%d parse_failures=%d", len(scored), meta["parse_failures"])
    return scored, meta


def collect_diffgui(epoch: int) -> tuple[list[dict], dict]:
    path = DIFFGUI_RUN / f"epoch_{epoch:04d}" / "molecules.csv"
    frame = pd.read_csv(path, low_memory=False)
    rows: list[dict] = []
    n_no_smiles = 0
    n_no_dock = 0
    for _, r in frame.iterrows():
        smiles = r.get("canonical_smiles")
        if not isinstance(smiles, str) or not smiles.strip() or smiles.lower() == "nan":
            n_no_smiles += 1
            continue
        dock = r.get("docking_raw")
        if pd.isna(dock):
            n_no_dock += 1
            continue
        metrics = metrics_from_smiles(smiles, reduce_fragment=False)
        if metrics is None:
            n_no_smiles += 1
            continue
        metrics["docking_raw"] = float(dock)
        rows.append(metrics)
    meta = {
        "source": str(path.relative_to(REPO)),
        "rows": int(len(frame)),
        "n": len(rows),
        "dropped_no_smiles": n_no_smiles,
        "dropped_no_docking": n_no_dock,
    }
    LOGGER.info(
        "DiffGui ep%d rows=%d used=%d (no_smiles=%d no_docking=%d)",
        epoch, meta["rows"], meta["n"], n_no_smiles, n_no_dock,
    )
    return rows, meta


# --------------------------------------------------------------------------- #
# aggregation + table assembly
# --------------------------------------------------------------------------- #
def _pct(rows: list[dict], key: str) -> float:
    return 100.0 * sum(r[key] for r in rows) / len(rows)


def summarize(rows: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(sum(r[key] for r in rows) / len(rows))

    out = {
        "n": len(rows),
        "docking_raw": mean("docking_raw") if "docking_raw" in rows[0] else None,
        "three_ring_pct": _pct(rows, "three_ring"),
        "spiro_pct": _pct(rows, "spiro"),
        "bridgehead_pct": _pct(rows, "bridgehead"),
        "ring_count": mean("ring_count"),
        "aromatic_ring_count": mean("aromatic_ring_count"),
        "rotatable_bonds": mean("rotatable_bonds"),
        "hac": mean("hac"),
        "r1_pct": _pct(rows, "r1_large_ring_violation"),
        "r2_pct": _pct(rows, "r2_ring_system_violation"),
        "r3_pct": _pct(rows, "r3_four_ring_violation"),
        "r4_any_pct": _pct(rows, "r4_any"),
        "r4a_pct": _pct(rows, "r4a_three_ring_hetero_violation"),
        "r4b_pct": _pct(rows, "r4b_three_ring_over_attached_violation"),
        "r4c_pct": _pct(rows, "r4c_three_ring_spiro_or_fused_violation"),
        "r5_pct": _pct(rows, "r5_carbon_fraction_violation"),
        "r14_pct": _pct(rows, "r14"),
        "r15_pct": _pct(rows, "r15"),
    }
    return out


COLUMNS = [
    ("n", "n", "int"),
    ("docking_raw", "对接均值\n(kcal/mol)", "f2"),
    ("three_ring_pct", "含三元环\n%", "p1"),
    ("spiro_pct", "螺环\n%", "p1"),
    ("bridgehead_pct", "桥环\n%", "p1"),
    ("ring_count", "环数\n均值", "f2"),
    ("aromatic_ring_count", "芳香环\n均值", "f2"),
    ("rotatable_bonds", "可旋转键\n均值", "f2"),
    ("hac", "重原子数\n均值", "f2"),
    ("r1_pct", "R1\n大环", "p1"),
    ("r2_pct", "R2\n环系规模", "p1"),
    ("r3_pct", "R3\n四元环", "p1"),
    ("r4_any_pct", "R4\n三元环任一", "p1"),
    ("r5_pct", "R5\n碳比例", "p1"),
    ("r14_pct", "R1–R4\n命中", "p1"),
    ("r15_pct", "R1–R5\n命中", "p1"),
]

# cells highlighted for the ep89 row (the worsening / diagnostic columns)
HIGHLIGHT_EP89 = {"three_ring_pct", "r4_any_pct"}


def _fmt(value: float | None, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "p1":
        return f"{value:.1f}%"
    return f"{value:.2f}"


def render_png(rows: list[tuple[str, dict]], path: Path) -> None:
    cjk = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(cjk).exists():
        font_manager.fontManager.addfont(cjk)
        family = font_manager.FontProperties(fname=cjk).get_name()
    else:
        family = "DejaVu Sans"
    plt.rcParams["font.family"] = family
    plt.rcParams["axes.unicode_minus"] = False

    header = [label for _, label, _ in COLUMNS]
    body = [[_fmt(metrics[key], kind) for key, _, kind in COLUMNS] for _, metrics in rows]
    row_labels = [name for name, _ in rows]

    fig, ax = plt.subplots(figsize=(20.0, 3.1), dpi=300)
    ax.axis("off")
    table = ax.table(
        cellText=body,
        rowLabels=row_labels,
        colLabels=header,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.5)
    table.scale(1.0, 2.4)

    n_cols = len(header)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d4d4d4")
        cell.set_linewidth(0.8)
        if r == 0:
            cell.set_facecolor("#eef2f7")
            cell.set_text_props(weight="bold", fontsize=11)
        elif c == -1:
            name = row_labels[r - 1]
            cell.set_facecolor("#e8f6ef" if "ChEMBL" in name else "#fdf3e3")
            cell.set_text_props(weight="bold", fontsize=11.5)
        else:
            key = COLUMNS[c][0]
            if "ChEMBL" in row_labels[r - 1]:
                cell.set_facecolor("#f3faf6")
            elif "ep89" in row_labels[r - 1] and key in HIGHLIGHT_EP89:
                cell.set_facecolor("#fdecea")
                cell.set_text_props(weight="bold", color="#c0392b")
            else:
                cell.set_facecolor("white")
    # let the row-label column breathe (header row has no -1 cell)
    for r in range(1, len(row_labels) + 1):
        label_cell = table._cells.get((r, -1))
        if label_cell is not None:
            label_cell.set_width(0.16 / max(1, n_cols) * 3.0)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT_DIR / "wide_table.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    chembl_rows, chembl_meta = collect_chembl()
    diffgui: dict[int, dict] = {}
    metas: dict[str, dict] = {"chembl36_reference": chembl_meta}
    for epoch in DIFFGUI_EPOCHS:
        rows, meta = collect_diffgui(epoch)
        diffgui[epoch] = summarize(rows)
        metas[f"diffgui_ep{epoch}"] = meta

    chembl_summary = summarize(chembl_rows)
    conditions = [
        (f"DiffGui ep1（RL 起点）", diffgui[1]),
        (f"DiffGui ep89（RL 终点）", diffgui[89]),
        (f"ChEMBL36 参照（n={chembl_summary['n']:,}）", chembl_summary),
    ]

    # ---- CSV ---------------------------------------------------------------
    with (OUT_DIR / "wide_table.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition"] + [key for key, _, _ in COLUMNS])
        for name, metrics in conditions:
            writer.writerow([name] + [_fmt(metrics[key], kind) for key, _, kind in COLUMNS])

    # ---- Markdown ----------------------------------------------------------
    lines = [
        "| 条件 | " + " | ".join(label.replace("\n", "") for _, label, _ in COLUMNS) + " |",
        "| --- | " + " | ".join("---" for _ in COLUMNS) + " |",
    ]
    for name, metrics in conditions:
        lines.append(
            "| " + name + " | " + " | ".join(_fmt(metrics[key], kind) for key, _, kind in COLUMNS) + " |"
        )
    (OUT_DIR / "wide_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- PNG ---------------------------------------------------------------
    render_png(conditions, OUT_DIR / "wide_table.png")

    # ---- JSON --------------------------------------------------------------
    payload = {
        "convention": {
            "chembl": "chembl_structure_pipeline standardised, mol_wt 150-500, "
                      "heavy_atoms 10-30, largest-fragment reduction (no-op for this file)",
            "diffgui": "molecules.csv rows with non-empty canonical_smiles and non-null docking_raw",
            "ring_and_pathology": "RDKit + scripts/offline_rl/structure_pathology_rules.py "
                                  "(identical implementation on both sides)",
            "definitions": {
                "three_ring_pct": "molecule has >=1 ring of size 3",
                "spiro_pct": "CalcNumSpiroAtoms > 0",
                "bridgehead_pct": "CalcNumBridgeheadAtoms > 0",
                "ring_count": "RingInfo().NumRings()",
                "aromatic_ring_count": "CalcNumAromaticRings",
                "rotatable_bonds": "CalcNumRotatableBonds (default)",
                "r4_any_pct": "union of R4a/R4b/R4c (r4_any_three_ring_violation); "
                              "not the sum of the three sub-rules",
                "docking_raw": "UniDock score, not available for ChEMBL",
            },
        },
        "sources": metas,
        "rows": {name: metrics for name, metrics in conditions},
    }
    (OUT_DIR / "wide_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for name, metrics in conditions:
        LOGGER.info(
            "%-34s n=%d dock=%s 3ring=%.1f%% spiro=%.1f%% bridge=%.1f%% R1-R4=%.1f%% R1-R5=%.1f%%",
            name, metrics["n"],
            "—" if metrics["docking_raw"] is None else f"{metrics['docking_raw']:.2f}",
            metrics["three_ring_pct"], metrics["spiro_pct"], metrics["bridgehead_pct"],
            metrics["r14_pct"], metrics["r15_pct"],
        )
    LOGGER.info("wrote wide_table.{csv,md,png,json} to %s", OUT_DIR)


if __name__ == "__main__":
    main()
