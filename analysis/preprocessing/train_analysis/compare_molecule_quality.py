"""
从 SMILES 计算额外的物化属性，对比两个运行的分子合理性。

用法:
    python analysis/train_analysis/compare_molecule_quality.py \\
        OUT/kl-0p01_xxx OUT/kl-0_xxx --label-a "KL=0.01" --label-b "KL=0"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 12, "axes.titlesize": 14,
    "axes.labelsize": 12, "legend.fontsize": 10, "lines.linewidth": 1.5,
})
COLOR_A = "#1f77b4"
COLOR_B = "#d62728"


# ── RDKit 描述符计算 ──────────────────────────────────────────────────────


def compute_descriptors(smiles: str) -> dict[str, float]:
    """从 SMILES 计算一组物化性质。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"MW": np.nan, "LogP": np.nan, "HBA": np.nan, "HBD": np.nan,
                "RotB": np.nan, "RingCount": np.nan, "AromaticRings": np.nan,
                "FractionCSP3": np.nan, "TPSA": np.nan, "HeavyAtomCount": np.nan}
    return {
        "MW": Descriptors.ExactMolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotB": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
        "AromaticRings": Lipinski.NumAromaticRings(mol),
        "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms(),
    }


def get_descriptor_labels() -> list[tuple[str, str, tuple | None]]:
    """返回 (列名, 标签, xlim) 列表。"""
    return [
        ("MW", "Molecular Weight", (100, 600)),
        ("LogP", "LogP", (-2, 8)),
        ("HBA", "H-Bond Acceptors", (0, 12)),
        ("HBD", "H-Bond Donors", (0, 6)),
        ("RotB", "Rotatable Bonds", (0, 20)),
        ("RingCount", "Ring Count", (0, 8)),
        ("AromaticRings", "Aromatic Rings", (0, 6)),
        ("FractionCSP3", "Fraction C sp³", (0, 1)),
        ("TPSA", "TPSA (Å²)", (0, 200)),
        ("HeavyAtomCount", "Heavy Atom Count", (10, 50)),
    ]


def lipinski_violations(row: dict) -> int:
    """计算 Lipinski Rule-of-5 违反数。"""
    violations = 0
    if row.get("MW", 0) > 500:
        violations += 1
    if row.get("LogP", 0) > 5:
        violations += 1
    if row.get("HBA", 0) > 10:
        violations += 1
    if row.get("HBD", 0) > 5:
        violations += 1
    return violations


# ── 加载与计算 ─────────────────────────────────────────────────────────────


def load_and_enrich(run_dir: str, epochs: list[int], sample_ratio: float = 1.0) -> pd.DataFrame:
    """加载多个 epoch 的分子并计算 RDKit 描述符。"""
    all_rows: list[dict] = []
    for ep in epochs:
        path = os.path.join(run_dir, "epochs", f"epoch_{ep:04d}", "molecules.csv")
        if not os.path.isfile(path):
            continue
        df = pd.read_csv(path)
        valid = df[df["rdkit_valid"] == 1].copy()
        if sample_ratio < 1.0:
            valid = valid.sample(frac=sample_ratio, random_state=42)
        for _, row in valid.iterrows():
            desc = compute_descriptors(row["smiles"])
            desc["epoch"] = ep
            desc["smiles"] = row["smiles"]
            desc["qed_raw"] = row["qed_raw"]
            desc["sa_raw"] = row["sa_raw"]
            desc["docking_raw"] = row["docking_raw"]
            desc["reward_total"] = row["reward_total"]
            all_rows.append(desc)
    result = pd.DataFrame(all_rows)
    if not result.empty:
        result["LipinskiViolations"] = result.apply(lipinski_violations, axis=1)
    return result


def get_available_epochs(run_dir: str) -> list[int]:
    epochs_dir = os.path.join(run_dir, "epochs")
    if not os.path.isdir(epochs_dir):
        return []
    return sorted(int(m.group(1)) for d in os.listdir(epochs_dir)
                  if (m := re.match(r"epoch_(\d+)", d)))


# ── 绘图 ──────────────────────────────────────────────────────────────────


def plot_descriptor_distributions(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """物化属性分布对比: 并排 KDE 图。"""
    descs = get_descriptor_labels()
    n = len(descs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten()

    for idx, (col, title, xlim) in enumerate(descs):
        ax = axes_flat[idx]
        if col in df_a.columns and not df_a[col].dropna().empty:
            sns.kdeplot(df_a[col].dropna(), ax=ax, label=label_a, color=COLOR_A, fill=True, alpha=0.3)
        if col in df_b.columns and not df_b[col].dropna().empty:
            sns.kdeplot(df_b[col].dropna(), ax=ax, label=label_b, color=COLOR_B, fill=True, alpha=0.3)
        ax.set_xlabel(title)
        ax.set_ylabel("Density")
        ax.set_title(f"Distribution of {title}")
        if xlim:
            ax.set_xlim(*xlim)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # 隐藏多余子图
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    path = os.path.join(save_dir, "04_descriptor_distributions.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_lipinski_comparison(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """Lipinski 违反数对比 + QED vs SA 合理性散点。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Lipinski 违反数直方图 ──
    ax = axes[0]
    max_va = max(df_a["LipinskiViolations"].max(), df_b["LipinskiViolations"].max(), 4)
    bins = np.arange(-0.5, max_va + 1.5, 1)
    ax.hist(df_a["LipinskiViolations"], bins=bins, alpha=0.6, color=COLOR_A, label=label_a)
    ax.hist(df_b["LipinskiViolations"], bins=bins, alpha=0.6, color=COLOR_B, label=label_b)
    ax.set_xlabel("Lipinski Violations")
    ax.set_ylabel("Count")
    ax.set_title("Lipinski Rule-of-5 Violations")
    ax.legend()
    ax.grid(alpha=0.3)

    # ── QED vs SA 散点 ──
    ax = axes[1]
    ax.scatter(df_a["qed_raw"], df_a["sa_raw"], s=10, alpha=0.3, color=COLOR_A, label=label_a, edgecolors="none")
    ax.scatter(df_b["qed_raw"], df_b["sa_raw"], s=10, alpha=0.3, color=COLOR_B, label=label_b, marker="s", edgecolors="none")
    # 合理区域 (QED>0.5, SA<5)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5)
    ax.fill_betweenx([1, 5], 0.5, 1, alpha=0.08, color="green", label="Reasonable zone")
    ax.set_xlabel("QED")
    ax.set_ylabel("SA (lower = easier)")
    ax.set_title("QED vs SA (reasonable zone: QED>0.5, SA<5)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # ── 分子量 vs LogP 散图 ──
    ax = axes[2]
    ax.scatter(df_a["MW"], df_a["LogP"], s=10, alpha=0.3, color=COLOR_A, label=label_a, edgecolors="none")
    ax.scatter(df_b["MW"], df_b["LogP"], s=10, alpha=0.3, color=COLOR_B, label=label_b, marker="s", edgecolors="none")
    # Lipinski 边界
    ax.axvline(x=500, color="gray", linestyle="--", alpha=0.5, label="MW=500")
    ax.axhline(y=5, color="gray", linestyle="--", alpha=0.5, label="LogP=5")
    ax.set_xlabel("Molecular Weight")
    ax.set_ylabel("LogP")
    ax.set_title("MW vs LogP (Lipinski boundaries)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "05_lipinski_analysis.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sa_vs_docking_reasonableness(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """合成难度 vs docking 结合力散点，看 KL 约束是否抑制了极端分子。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── SA vs Docking ──
    ax = axes[0]
    ax.scatter(df_a["sa_raw"], df_a["docking_raw"], s=10, alpha=0.3, color=COLOR_A, label=label_a, edgecolors="none")
    ax.scatter(df_b["sa_raw"], df_b["docking_raw"], s=10, alpha=0.3, color=COLOR_B, label=label_b, marker="s", edgecolors="none")
    ax.set_xlabel("SA (higher = harder)")
    ax.set_ylabel("Docking (more negative = better)")
    ax.set_title("SA vs Docking — are extreme molecules reasonable?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.invert_yaxis()

    # ── SA 分布（越靠近1越好） ──
    ax = axes[1]
    if "sa_raw" in df_a.columns:
        sns.kdeplot(df_a["sa_raw"].dropna(), ax=ax, label=label_a, color=COLOR_A, fill=True, alpha=0.3)
    if "sa_raw" in df_b.columns:
        sns.kdeplot(df_b["sa_raw"].dropna(), ax=ax, label=label_b, color=COLOR_B, fill=True, alpha=0.3)
    # 标记 SA 可接受区域
    ax.axvline(x=3, color="green", linestyle="--", alpha=0.5, label="Easy (SA<3)")
    ax.axvline(x=6, color="orange", linestyle="--", alpha=0.5, label="Hard (SA>6)")
    ax.axvline(x=8, color="red", linestyle="--", alpha=0.5, label="Extreme (SA>8)")
    ax.set_xlabel("SA Score")
    ax.set_ylabel("Density")
    ax.set_title("SA Distribution — Synthetic Accessibility")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "06_sa_docking_reasonableness.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_extreme_molecule_ratio(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    label_a: str, label_b: str, save_dir: str,
) -> str:
    """极端分子占比对比（SA>6，MW>500, LogP>5, 违反>=2等）。"""
    thresholds = [
        ("SA > 6 (hard to synthesize)", df_a["sa_raw"] > 6, df_b["sa_raw"] > 6),
        ("SA > 8 (very hard)", df_a["sa_raw"] > 8, df_b["sa_raw"] > 8),
        ("MW > 500", df_a["MW"] > 500, df_b["MW"] > 500),
        ("LogP > 5", df_a["LogP"] > 5, df_b["LogP"] > 5),
        ("Lipinski >= 2 violations", df_a["LipinskiViolations"] >= 2, df_b["LipinskiViolations"] >= 2),
        ("Ring Count > 6", df_a["RingCount"] > 6, df_b["RingCount"] > 6),
    ]

    labels = []
    ratios_a, ratios_b = [], []
    for label, mask_a, mask_b in thresholds:
        labels.append(label)
        r_a = mask_a.sum() / len(df_a) * 100
        r_b = mask_b.sum() / len(df_b) * 100
        ratios_a.append(r_a)
        ratios_b.append(r_b)

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.5)))
    y = np.arange(len(labels))
    h = 0.35
    ax.barh(y - h / 2, ratios_a, h, label=label_a, color=COLOR_A, alpha=0.8)
    ax.barh(y + h / 2, ratios_b, h, label=label_b, color=COLOR_B, alpha=0.8)
    for i, (ra, rb) in enumerate(zip(ratios_a, ratios_b)):
        ax.text(ra + 0.3, i - h / 2, f"{ra:.1f}%", va="center", fontsize=8, color=COLOR_A)
        ax.text(rb + 0.3, i + h / 2, f"{rb:.1f}%", va="center", fontsize=8, color=COLOR_B)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ratio of All Valid Molecules (%)")
    ax.set_title("Extreme / Undesirable Molecule Ratio")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(save_dir, "07_extreme_ratios.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_report(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    label_a: str, label_b: str,
    plot_paths: list[str], save_dir: str,
) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append(f"  分子合理性对比报告 — {label_a} vs {label_b}")
    lines.append("=" * 72)
    lines.append("")
    lines.append(f"  {label_a}: {len(df_a)} valid 分子")
    lines.append(f"  {label_b}: {len(df_b)} valid 分子")
    lines.append("")

    # ── 核心描述符均值对比 ──
    lines.append("-" * 72)
    lines.append("  1. 物化性质均值对比")
    lines.append("-" * 72)
    lines.append("")
    descs = get_descriptor_labels()
    lines.append(f"  {'性质':<22} {label_a:<18} {label_b:<18} {'std_a':<12} {'std_b':<12}")
    lines.append("  " + "-" * 82)
    for col, label, xlim in descs:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        m_a = df_a[col].mean()
        s_a = df_a[col].std()
        m_b = df_b[col].mean()
        s_b = df_b[col].std()
        diff = m_a - m_b
        wider = "A" if s_a > s_b else "B"
        lines.append(f"  {label:<22} {m_a:>8.2f} ± {s_a:.2f}  {m_b:>8.2f} ± {s_b:.2f}  "
                     f"({diff:+.2f})  wider={wider}")

    lines.append("")

    # ── Lipinski ──
    lines.append("-" * 72)
    lines.append("  2. Lipinski Rule-of-5")
    lines.append("-" * 72)
    lines.append("")
    for label, df in [(label_a, df_a), (label_b, df_b)]:
        v_counts = df["LipinskiViolations"].value_counts().sort_index()
        total = len(df)
        pct_0 = (v_counts.get(0, 0) / total * 100)
        pct_1 = (v_counts.get(1, 0) / total * 100)
        pct_ge2 = ((v_counts.get(2, 0) + v_counts.get(3, 0) + v_counts.get(4, 0)) / total * 100)
        lines.append(f"  {label}:  0 violations: {pct_0:.1f}%  1 violation: {pct_1:.1f}%  >=2 violations: {pct_ge2:.1f}%")

    lines.append("")

    # ── 极端分子比例 ──
    lines.append("-" * 72)
    lines.append("  3. 极端/不合理分子比例")
    lines.append("-" * 72)
    lines.append("")
    thresholds = [
        ("SA > 6 (难以合成)", df_a["sa_raw"] > 6, df_b["sa_raw"] > 6),
        ("SA > 8 (极难合成)", df_a["sa_raw"] > 8, df_b["sa_raw"] > 8),
        ("MW > 500", df_a["MW"] > 500, df_b["MW"] > 500),
        ("LogP > 5 (脂溶性过高)", df_a["LogP"] > 5, df_b["LogP"] > 5),
        ("Lipinski >= 2 违反", df_a["LipinskiViolations"] >= 2, df_b["LipinskiViolations"] >= 2),
        ("RingCount > 6 (环过多)", df_a["RingCount"] > 6, df_b["RingCount"] > 6),
        ("RotB > 15 (过于柔性)", df_a["RotB"] > 15, df_b["RotB"] > 15),
    ]
    for label, mask_a, mask_b in thresholds:
        r_a = mask_a.sum() / len(df_a) * 100
        r_b = mask_b.sum() / len(df_b) * 100
        winner = label_a if r_a < r_b else label_b
        lines.append(f"  {label:<30} {label_a}: {r_a:.1f}%  {label_b}: {r_b:.1f}%  ← {winner} 更合理")

    lines.append("")

    # ── 图表 ──
    lines.append("-" * 72)
    lines.append("  4. 生成图表")
    lines.append("-" * 72)
    for p in plot_paths:
        if p:
            lines.append(f"  📊 {os.path.basename(p)}")

    lines.append("")
    lines.append("=" * 72)

    report = "\n".join(lines)
    report_path = os.path.join(save_dir, "molecule_quality_report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    return report_path


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="分子合理性对比分析")
    parser.add_argument("run_a", type=str, help="运行 A (如 KL>0)")
    parser.add_argument("run_b", type=str, help="运行 B (如 KL=0)")
    parser.add_argument("--label-a", type=str, default=None)
    parser.add_argument("--label-b", type=str, default=None)
    parser.add_argument("--sample-ratio", type=float, default=0.5,
                        help="分子采样比例 (默认0.5，加速)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="最多加载的 epoch 数 (默认全部)")
    args = parser.parse_args()

    run_a = os.path.abspath(args.run_a)
    run_b = os.path.abspath(args.run_b)
    label_a = args.label_a or os.path.basename(run_a.rstrip("/"))
    label_b = args.label_b or os.path.basename(run_b.rstrip("/"))

    save_dir = os.path.join(run_a, f"compare_vs_{label_b.replace('/', '_')}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n📁 加载分子数据...")
    epochs_a = get_available_epochs(run_a)
    epochs_b = get_available_epochs(run_b)
    if args.max_epochs:
        epochs_a = epochs_a[:args.max_epochs]
        epochs_b = epochs_b[:args.max_epochs]

    print(f"  {label_a}: {len(epochs_a)} epochs")
    print(f"  {label_b}: {len(epochs_b)} epochs")

    print(f"  Computing RDKit descriptors (sampling {args.sample_ratio*100:.0f}%)...")
    df_a = load_and_enrich(run_a, epochs_a, sample_ratio=args.sample_ratio)
    df_b = load_and_enrich(run_b, epochs_b, sample_ratio=args.sample_ratio)
    print(f"  {label_a}: {len(df_a)} valid molecules")
    print(f"  {label_b}: {len(df_b)} valid molecules")

    plot_paths: list[str] = []

    print("  绘图: 物化属性分布对比...")
    plot_paths.append(plot_descriptor_distributions(df_a, df_b, label_a, label_b, save_dir))

    print("  绘图: Lipinski 分析...")
    plot_paths.append(plot_lipinski_comparison(df_a, df_b, label_a, label_b, save_dir))

    print("  绘图: SA vs Docking 合理性...")
    plot_paths.append(plot_sa_vs_docking_reasonableness(df_a, df_b, label_a, label_b, save_dir))

    print("  绘图: 极端分子比例对比...")
    plot_paths.append(plot_extreme_molecule_ratio(df_a, df_b, label_a, label_b, save_dir))

    print("  生成: 合理性分析报告...")
    report_path = generate_report(df_a, df_b, label_a, label_b, plot_paths, save_dir)

    print(f"\n✅ 合理性分析完成!\n")
    print(f"  报告: {report_path}")
    for p in plot_paths:
        if p:
            print(f"  图表: {p}")

    print()
    try:
        with open(report_path) as f:
            print(f.read())
    except Exception:
        pass


if __name__ == "__main__":
    main()
