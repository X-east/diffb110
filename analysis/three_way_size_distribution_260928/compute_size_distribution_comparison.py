#!/usr/bin/env python
"""三方分子大小（重原子数 HA）分布调查，含 ChEMBL「仅小分子」口径。

调查对象
--------
ChEMBL 侧（以官方 SQLite 的 molecule_dictionary.molecule_type 判定）：
  1. ChEMBL 标准化全集（未截断，全部类型）
  2. ChEMBL 标准化全集（未截断，**仅 Small molecule**）
  3. ChEMBL 参照子集（项目窗口 HA10-30，全部类型）      <- 现状参照集
  4. ChEMBL 参照子集（项目窗口 HA10-30，**仅 Small molecule**）  <- 候选参照集

DiffSBDD 侧：
  5. 训练集配体 HA 边缘分布（checkpoint node_histogram 行边缘）
  6. 8GJL 条件先验（node_histogram[:, 349]，即 sample_conditional(n2=349) 用的那一行）

实测侧：
  7. DiffSBDD Original 生成实测
  8. 采集 200k 端点

口径说明
--------
- 总体 5/6 来自检查点内的高斯平滑联合直方图（smooth_sigma=1.0），是**概率质量**；
  分位由 CDF 反解得到，不作为分子级样本。
- 总体 8 的 HA 等于采集时分配的 nodesxsample（训练侧尺寸先验），不是重建后原子数。
- ChEMBL 侧 max HA = 681（寡核苷酸），网格开到 700 以避免截断堆积。

输出
----
size_distribution_pmf.csv / size_distribution_summary.csv / window_coverage.csv
molecule_type_breakdown.csv     各 ChEMBL 口径的 molecule_type 构成
summary.json
size_distribution_comparison.png / .svg
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import font_manager

PROJECT_ROOT = Path("/home/md666/data/disk6/DiffB110")
OUT_DIR = Path(__file__).resolve().parent

CKPT = PROJECT_ROOT / "DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt"
CHEMBL_DB = PROJECT_ROOT / "data/chembl/chembl_36/chembl_36_sqlite/chembl_36.db"
CHEMBL_ALL = PROJECT_ROOT / "data/chembl_standardized_all/chembl_36_standardized_all_properties.csv.gz"
CHEMBL_REF = PROJECT_ROOT / "data/chembl/chembl_36_properties.csv.gz"
GEN_SDF = PROJECT_ROOT / (
    "analysis/diffsbdd_global_e2po_structure_matched_x_only_generation_1000_260901"
    "/molecules_relaxed_largest.sdf"
)
EPISODE_INDEX = PROJECT_ROOT / "analysis/offline_iql_full_index_rebuilt_260824_v7/episode_index.jsonl"

POCKET_SIZE = 349  # 8GJL, ref_ligand C:1101, dist_cutoff=8.0 实测
HA_MAX = 700  # ChEMBL 全集实测最大 HA = 681
PLOT_XLIM = 70  # 展示范围
SMALL_MOLECULE = "Small molecule"

# (key, 图例, 颜色, kind)  kind: molecule=分子级实测 / smooth=平滑直方图
POPULATIONS = [
    ("chembl_all", "ChEMBL 全集·全部类型（未截断）", "#BDC3C7", "molecule"),
    ("chembl_small_all", "ChEMBL 全集·仅小分子（未截断）", "#5DADE2", "molecule"),
    ("chembl_ref", "ChEMBL 参照·全部类型 HA10-30（现状）", "#2C3E50", "molecule"),
    ("chembl_small_ref", "ChEMBL 参照·仅小分子 HA10-30（候选）", "#1F618D", "molecule"),
    ("ckpt_marginal", "DiffSBDD 训练集配体（边缘）", "#E74C3C", "smooth"),
    ("ckpt_cond_349", f"DiffSBDD 8GJL 条件先验 (n2={POCKET_SIZE})", "#E67E22", "smooth"),
    ("gen_original", "DiffSBDD Original 生成实测", "#F39C12", "molecule"),
    ("collect_200k", "你的采集 200k 端点", "#8E44AD", "molecule"),
]

WINDOWS = {"[10, 30]": (10, 30), "[12, 35]": (12, 35)}


def setup_fonts() -> None:
    """Install a CJK font so Chinese labels render correctly."""
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=candidate).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def pmf_from_counts(values: np.ndarray, hi: int = HA_MAX) -> np.ndarray:
    """Return a normalised PMF over integer heavy-atom counts 0..hi."""
    counts = np.bincount(np.asarray(values, dtype=int).clip(0, hi), minlength=hi + 1)
    return counts / counts.sum()


def pmf_from_smoothed_histogram(mass: np.ndarray, hi: int = HA_MAX) -> np.ndarray:
    """Return a normalised PMF from a smoothed histogram slice, zero-padded to hi+1."""
    mass = np.asarray(mass, dtype=float)[: hi + 1]
    if mass.size < hi + 1:
        mass = np.concatenate([mass, np.zeros(hi + 1 - mass.size, dtype=float)])
    total = mass.sum()
    if total <= 0:
        raise ValueError("empty histogram slice")
    return mass / total


def load_molecule_types() -> pd.DataFrame:
    """Return ``chembl_id -> molecule_type`` from the official ChEMBL SQLite."""
    query = """
        SELECT md.chembl_id AS chembl_id, md.molecule_type AS molecule_type
        FROM molecule_dictionary AS md
        JOIN compound_structures AS cs ON md.molregno = cs.molregno
        WHERE cs.canonical_smiles IS NOT NULL
    """
    connection = sqlite3.connect(f"file:{CHEMBL_DB}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(query, connection)
    finally:
        connection.close()
    frame["molecule_type"] = frame["molecule_type"].fillna("None")
    return frame


def load_chembl(path: Path) -> pd.DataFrame:
    """Read ``chembl_id`` + ``heavy_atoms`` from a gzipped ChEMBL property table."""
    return pd.read_csv(path, usecols=["chembl_id", "heavy_atoms"], low_memory=False)


def load_generated_ha() -> np.ndarray:
    """Heavy-atom counts of the Mainline Original generation batch."""
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    supplier = Chem.SDMolSupplier(str(GEN_SDF), sanitize=False, removeHs=True)
    return np.array([mol.GetNumAtoms() for mol in supplier if mol is not None], dtype=int)


def load_collected_nodes() -> np.ndarray:
    """Per-endpoint node counts of the frozen 200k collection index."""
    values: list[int] = []
    with EPISODE_INDEX.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            node_count = json.loads(line).get("num_nodes")
            if node_count:
                values.append(int(node_count))
    return np.asarray(values, dtype=int)


def quantiles_from_pmf(pmf: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    """Invert a PMF's CDF to integer quantiles."""
    cdf = np.cumsum(pmf)
    return {f"P{int(q * 100):02d}": float(np.searchsorted(cdf, q)) for q in qs}


def main() -> None:
    """Compute and plot the eight-population heavy-atom distribution comparison."""
    setup_fonts()
    sns.set_theme(style="whitegrid", context="talk")
    setup_fonts()  # sns.set_theme 会重置 font.family，必须在其之后再装一次
    plt.rcParams["axes.edgecolor"] = "#d4d4d4"
    plt.rcParams["axes.linewidth"] = 1.2

    types = load_molecule_types()
    chembl_all = load_chembl(CHEMBL_ALL).merge(types, on="chembl_id", how="left")
    chembl_ref = load_chembl(CHEMBL_REF).merge(types, on="chembl_id", how="left")
    chembl_all["molecule_type"] = chembl_all["molecule_type"].fillna("(unmatched)")
    chembl_ref["molecule_type"] = chembl_ref["molecule_type"].fillna("(unmatched)")

    breakdown_rows: list[dict[str, object]] = []
    for name, frame in (
        ("chembl_all_未截断", chembl_all),
        ("chembl_ref_HA10-30", chembl_ref),
    ):
        counts = frame["molecule_type"].value_counts()
        for molecule_type, count in counts.items():
            breakdown_rows.append(
                {
                    "table": name,
                    "molecule_type": molecule_type,
                    "n": int(count),
                    "share_pct": round(100.0 * count / len(frame), 4),
                    "ha_median": float(frame.loc[frame.molecule_type == molecule_type, "heavy_atoms"].median()),
                    "ha_max": int(frame.loc[frame.molecule_type == molecule_type, "heavy_atoms"].max()),
                }
            )
    pd.DataFrame(breakdown_rows).to_csv(OUT_DIR / "molecule_type_breakdown.csv", index=False)

    small_all = chembl_all.loc[chembl_all.molecule_type == SMALL_MOLECULE, "heavy_atoms"].to_numpy(int)
    small_ref = chembl_ref.loc[chembl_ref.molecule_type == SMALL_MOLECULE, "heavy_atoms"].to_numpy(int)

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    joint = np.asarray(payload["hyper_parameters"]["node_histogram"], dtype=float)
    marginal = joint.sum(axis=1)
    conditional = joint[:, min(POCKET_SIZE, joint.shape[1] - 1)]

    molecule_values: dict[str, np.ndarray] = {
        "chembl_all": chembl_all["heavy_atoms"].to_numpy(int),
        "chembl_small_all": small_all,
        "chembl_ref": chembl_ref["heavy_atoms"].to_numpy(int),
        "chembl_small_ref": small_ref,
        "gen_original": load_generated_ha(),
        "collect_200k": load_collected_nodes(),
    }
    pmfs = {key: pmf_from_counts(values) for key, values in molecule_values.items()}
    pmfs["ckpt_marginal"] = pmf_from_smoothed_histogram(marginal)
    pmfs["ckpt_cond_349"] = pmf_from_smoothed_histogram(conditional)
    sizes: dict[str, int | None] = {key: int(values.size) for key, values in molecule_values.items()}
    sizes["ckpt_marginal"] = None
    sizes["ckpt_cond_349"] = None

    grid = np.arange(HA_MAX + 1)
    pmf_frame = pd.DataFrame({"heavy_atoms": grid})
    for key, _, _, _ in POPULATIONS:
        pmf_frame[key] = pmfs[key]
    pmf_frame.to_csv(OUT_DIR / "size_distribution_pmf.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    qs = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
    for key, label, _, kind in POPULATIONS:
        pmf = pmfs[key]
        support = np.flatnonzero(pmf > 1e-12)
        summary_rows.append(
            {
                "population": label,
                "key": key,
                "kind": kind,
                "n": sizes[key],
                "mean": round(float((grid * pmf).sum()), 3),
                "min": int(support.min()),
                "max": int(support.max()),
                **quantiles_from_pmf(pmf, qs),
            }
        )
        row: dict[str, object] = {"population": label, "key": key}
        for window_label, (lo, hi) in WINDOWS.items():
            row[f"share_{window_label}"] = round(float(pmf[lo : hi + 1].sum()), 4)
        row["share_below_10"] = round(float(pmf[:10].sum()), 4)
        row["share_above_30"] = round(float(pmf[31:].sum()), 4)
        coverage_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    coverage_frame = pd.DataFrame(coverage_rows)
    summary_frame.to_csv(OUT_DIR / "size_distribution_summary.csv", index=False)
    coverage_frame.to_csv(OUT_DIR / "window_coverage.csv", index=False)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": str(CKPT),
                "chembl_sqlite": str(CHEMBL_DB),
                "pocket_size": POCKET_SIZE,
                "chembl_all": str(CHEMBL_ALL),
                "chembl_ref": str(CHEMBL_REF),
                "generated_sdf": str(GEN_SDF),
                "episode_index": str(EPISODE_INDEX),
                "note_smooth": "ckpt_marginal / ckpt_cond_349 来自高斯平滑联合直方图 (sigma=1.0)，为概率质量",
                "note_chembl_max_ha": "ChEMBL 全集最大 HA = 681（Oligonucleotide）",
                "summary": summary_rows,
                "coverage": coverage_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ plot
    figure, axes = plt.subplots(1, 2, figsize=(20, 8), constrained_layout=True)

    ax = axes[0]
    for key, label, color, kind in POPULATIONS:
        linestyle = "--" if kind == "smooth" or key.endswith("_all") else "-"
        ax.plot(
            grid,
            pmfs[key],
            linewidth=2.6,
            linestyle=linestyle,
            color=color,
            label=label,
            zorder=3,
        )
    pmf_top = max(float(pmfs[key].max()) for key, *_ in POPULATIONS)
    for index, (lo, hi, name) in enumerate(((10, 30, "[10, 30]"), (12, 35, "[12, 35]"))):
        for bound in (lo, hi):
            ax.axvline(bound, color="#E74C3C", linestyle=":", linewidth=1.5, alpha=0.7, zorder=2)
        ax.text(
            lo,
            pmf_top * (1.42 - 0.13 * index),
            f"{name} 边界",
            color="#E74C3C",
            fontsize=11.5,
            fontweight="bold",
            va="center",
            ha="left",
        )
    ax.set_xlim(0, PLOT_XLIM)
    ax.set_ylim(0, pmf_top * 1.58)
    ax.set_xlabel("重原子数 HA (heavy atom count)", fontsize=15)
    ax.set_ylabel("概率质量 PMF", fontsize=15)
    ax.set_title("A. 各总体的重原子数分布", fontsize=17, fontweight="bold")
    ax.tick_params(labelsize=12.5)
    ax.legend(fontsize=10.5, frameon=True, loc="upper right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax = axes[1]
    labels = [label for _, label, _, _ in POPULATIONS]
    y = np.arange(len(labels))
    height = 0.36
    for offset, (window_label, _) in enumerate(WINDOWS.items()):
        values = coverage_frame[f"share_{window_label}"].to_numpy()
        bars = ax.barh(
            y + (offset - 0.5) * height,
            values * 100,
            height=height,
            label=f"落在 {window_label} 内",
            color=["#3498DB", "#1ABC9C"][offset],
            alpha=0.9,
        )
        for bar, value in zip(bars, values):
            ax.text(
                value * 100 + 1.5,
                bar.get_y() + bar.get_height() / 2,
                f"{value * 100:.1f}%",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
    ax.axvline(100, color="#E74C3C", linestyle="--", linewidth=1.8, alpha=0.8, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xlabel("分子落在窗口内的比例 (%)", fontsize=15)
    ax.set_title("B. 各总体落在候选窗口内的比例", fontsize=17, fontweight="bold")
    ax.tick_params(labelsize=12.5)
    ax.legend(fontsize=11.5, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    figure.suptitle(
        "ChEMBL（全部类型 vs 仅小分子）/ DiffSBDD / 采集 200k 的重原子数分布对比（2026-09-28）",
        fontsize=18,
        fontweight="bold",
    )
    figure.savefig(OUT_DIR / "size_distribution_comparison.png", dpi=300, facecolor="white")
    figure.savefig(OUT_DIR / "size_distribution_comparison.svg", facecolor="white")
    plt.close(figure)

    print(summary_frame.to_string(index=False))
    print()
    print(coverage_frame.to_string(index=False))
    print()
    print(pd.DataFrame(breakdown_rows).to_string(index=False))
    print(f"\nwrote {OUT_DIR}")


if __name__ == "__main__":
    main()
