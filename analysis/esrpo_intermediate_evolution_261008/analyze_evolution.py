"""ESRPO 中间生成轨迹（step_*）的分子质量演化分析。

数据源：OUT/diffsbdd_e2po_mainline_intermediate_gen/step_*/
        （每个 step 含 molecules.sdf 与 docking_relaxed/docking_results.csv）

指标口径与 `analysis/diffgui_ep1ep89_vs_chembl_260918/build_wide_table.py` 保持一致：
RDKit + scripts/offline_rl/structure_pathology_rules.py；largest-fragment 归约；
R4 为 R4a/R4b/R4c 并集。

用法：conda run -n dif python analyze_evolution.py
"""

from __future__ import annotations

import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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
GEN_DIR = REPO / "OUT" / "diffsbdd_e2po_mainline_intermediate_gen"
WORKERS = min(18, os.cpu_count() or 1)

# step 目录 -> 名义更新步数（step_final 取自 metrics.csv 的 train_updates=18116）
STEPS: list[tuple[str, int]] = [
    ("step_002000", 2000),
    ("step_004000", 4000),
    ("step_006000", 6000),
    ("step_008000", 8000),
    ("step_010000", 10000),
    ("step_012000", 12000),
    ("step_014000", 14000),
    ("step_016000", 16000),
    ("step_018000", 18000),
    ("step_final", 18116),
]

# ChEMBL36 参照（来自 analysis/chembl_deviation_analysis 与 260918 宽表，同口径）
CHEMBL_REF = {
    "three_ring_pct": 4.5,
    "spiro_pct": 2.6,
    "bridgehead_pct": 2.7,
    "ring_count": 3.06,
    "aromatic_ring_count": 2.24,
    "rotatable_bonds": 4.42,
    "hac": 23.87,
    "r4_any_pct": 1.32,
    "r14_pct": 10.4,
}

LOGGER = logging.getLogger("esrpo-evolution")


def _largest_fragment(mol: Chem.Mol) -> Chem.Mol:
    try:
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    except Exception:
        return mol
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def molecule_metrics(smiles: str) -> dict | None:
    """单个分子的结构与病理指标。"""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    mol = _largest_fragment(mol)
    facts = compute_structure_pathology(mol)
    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()
    return {
        "three_ring_pct": float(any(len(r) == 3 for r in atom_rings)),
        "spiro_pct": float(rd_md.CalcNumSpiroAtoms(mol) > 0),
        "bridgehead_pct": float(rd_md.CalcNumBridgeheadAtoms(mol) > 0),
        "ring_count": float(ring_info.NumRings()),
        "aromatic_ring_count": float(rd_md.CalcNumAromaticRings(mol)),
        "rotatable_bonds": float(rd_md.CalcNumRotatableBonds(mol)),
        "hac": float(mol.GetNumHeavyAtoms()),
        "r1_pct": float(facts.r1_large_ring_violation),
        "r2_pct": float(facts.r2_ring_system_violation),
        "r3_pct": float(facts.r3_four_ring_violation),
        "r4_any_pct": float(facts.r4_any_three_ring_violation),
        "r5_pct": float(facts.r5_carbon_fraction_violation),
    }


def _worker(args):
    return molecule_metrics(args)


def analyse_step(label: str, step: int) -> dict | None:
    path = GEN_DIR / label / "docking_relaxed" / "docking_results.csv"
    if not path.exists():
        LOGGER.warning("%s 缺少 docking_results.csv，跳过", label)
        return None

    frame = pd.read_csv(path)
    frame["success"] = frame["success"].astype(str).str.lower() == "true"

    smiles = frame["input_smiles"].fillna("").astype(str)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        feats = list(pool.map(_worker, smiles, chunksize=32))
    ok = [f for f in feats if f is not None]

    n_rows = len(frame)
    n_smiles = len(ok)
    n_dock = int(frame["success"].sum())

    # UniDock 偶发崩溃会写出正分值（甚至 1e6 量级），属无效对接，
    # 必须剔除，否则单条垃圾值即可把整步均值抬到三位数以上。
    scores = frame.loc[frame["success"], "score"].astype(float)
    valid = scores[scores <= 0.0]
    n_invalid = int((scores > 0.0).sum())

    row: dict = {
        "step_label": label,
        "updates": step,
        "n_ligands": n_rows,
        "n_valid_smiles": n_smiles,
        "n_docking_success": n_dock,
        "n_docking_valid": int(len(valid)),
        "n_docking_invalid": n_invalid,
        "docking_success_pct": 100.0 * len(valid) / n_rows if n_rows else float("nan"),
    }

    row["docking_mean"] = valid.mean() if len(valid) else float("nan")
    row["docking_median"] = valid.median() if len(valid) else float("nan")
    row["docking_best"] = valid.min() if len(valid) else float("nan")
    row["docking_p25"] = valid.quantile(0.25) if len(valid) else float("nan")
    row["docking_p75"] = valid.quantile(0.75) if len(valid) else float("nan")

    feat_frame = pd.DataFrame(ok)
    if not feat_frame.empty:
        for col in feat_frame.columns:
            row[col] = 100.0 * feat_frame[col].mean() if col.endswith("_pct") else feat_frame[col].mean()

    LOGGER.info(
        "%-12s updates=%-6d valid=%3d invalid=%2d/%3d median=%.2f R4=%.1f%% 3ring=%.1f%% spiro=%.1f%% rings=%.2f",
        label, step, len(valid), n_invalid, n_rows,
        row.get("docking_median", float("nan")),
        row.get("r4_any_pct", float("nan")),
        row.get("three_ring_pct", float("nan")),
        row.get("spiro_pct", float("nan")),
        row.get("ring_count", float("nan")),
    )
    return row


# --------------------------------------------------------------------------- #
# 绘图
# --------------------------------------------------------------------------- #
def setup_font():
    for candidate in ("Arial", "Liberation Sans", "DejaVu Sans"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=candidate), fallback_to_default=False)
            plt.rcParams["font.family"] = candidate
            return
        except Exception:
            continue


def plot_evolution(table: pd.DataFrame):
    setup_font()
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 1.2

    specs = [
        ("docking_median", "Docking median (kcal/mol)", "#E74C3C", None, "a_docking_evolution"),
        ("r4_any_pct", "R4 three-ring pathology (%)", "#8E44AD", CHEMBL_REF["r4_any_pct"], "b_r4_evolution"),
        ("three_ring_pct", "Contains 3-membered ring (%)", "#F39C12", CHEMBL_REF["three_ring_pct"], "c_threering_evolution"),
        ("spiro_pct", "Spiro ring (%)", "#3498DB", CHEMBL_REF["spiro_pct"], "d_spiro_evolution"),
        ("ring_count", "Ring count (mean)", "#1ABC9C", CHEMBL_REF["ring_count"], "e_ringcount_evolution"),
        ("r14_pct", None, None, None, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(19, 9.5))
    axes = axes.ravel()
    for ax, (key, ylabel, color, ref, _) in zip(axes, specs):
        if key == "r14_pct":
            ax.axis("off")
            continue
        ax.plot(table["updates"], table[key], marker="o", color=color, linewidth=2.2, markersize=7)
        if key == "docking_median":
            ax.fill_between(
                table["updates"], table["docking_p25"], table["docking_p75"],
                color=color, alpha=0.15, label="P25–P75",
            )
            ax.legend(fontsize=11)
        if ref is not None:
            ax.axhline(ref, color="#2ECC71", linestyle="--", linewidth=1.8, alpha=0.9)
            ax.text(
                table["updates"].iloc[0], ref,
                f"ChEMBL ref {ref}",
                color="#27AE60", fontsize=11, va="bottom", ha="left", fontweight="bold",
            )
        ax.set_xlabel("Training updates", fontsize=13)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)

    mono = [s for s in specs if s[2] is None and s[0] != "r14_pct"]
    for ax, (key, ylabel, color, _, _) in zip(axes, mono):
        ax.set_title(ylabel, fontsize=13, fontweight="bold")

    fig.suptitle("ESRPO training: generated molecule quality vs. update steps",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "esrpo_evolution_grid.png", dpi=300)
    plt.close(fig)

    # 单张合并图：对接 vs R4 双轴
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.plot(table["updates"], table["docking_median"], marker="o", color="#E74C3C",
             linewidth=2.4, markersize=8, label="对接中位数")
    ax1.set_xlabel("Training updates", fontsize=14)
    ax1.set_ylabel("Docking median (kcal/mol)", fontsize=14, color="#E74C3C")
    ax1.tick_params(labelsize=12)
    ax2 = ax1.twinx()
    ax2.plot(table["updates"], table["r4_any_pct"], marker="s", color="#8E44AD",
             linewidth=2.4, markersize=8, label="R4 three-ring pathology")
    ax2.set_ylabel("R4 three-ring pathology (%)", fontsize=14, color="#8E44AD")
    ax2.tick_params(labelsize=12)
    ax2.axhline(CHEMBL_REF["r4_any_pct"], color="#2ECC71", linestyle="--", linewidth=1.8)
    ax1.set_title("ESRPO: R4 pathology drops while docking median slightly degrades",
                  fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "esrpo_docking_vs_pathology.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(OUT_DIR / "evolution.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    rows = [r for r in (analyse_step(label, step) for label, step in STEPS) if r]
    table = pd.DataFrame(rows).sort_values("updates").reset_index(drop=True)

    table.to_csv(OUT_DIR / "evolution_metrics.csv", index=False, encoding="utf-8-sig")
    header = "| " + " | ".join(table.columns) + " |"
    sep = "| " + " | ".join("---" for _ in table.columns) + " |"
    body = [
        "| " + " | ".join(
            f"{v:.3f}" if isinstance(v, float) else str(v) for v in record
        ) + " |"
        for record in table.itertuples(index=False)
    ]
    (OUT_DIR / "evolution_metrics.md").write_text(
        "\n".join([header, sep, *body]) + "\n", encoding="utf-8"
    )
    plot_evolution(table)
    LOGGER.info("wrote evolution_metrics.{csv,md} + 2 png to %s", OUT_DIR)


if __name__ == "__main__":
    main()
