#!/usr/bin/env python3
"""Analyze DiffGui next6 epoch property distributions."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "OUT"
ANALYSIS_DIR = Path(__file__).resolve().parent
TABLE_DIR = ANALYSIS_DIR / "tables"
FIGURE_DIR = ANALYSIS_DIR / "figures"

RUNS = [
    ("01", "dock_aromatic_pathology", "diffgui-next6-20260618-01-dock_aromatic_pathology_20260618_211501"),
    ("02", "le_aromatic_pathology", "diffgui-next6-20260618-02-le_aromatic_pathology_20260619_020218"),
    ("03", "dock_le_aromatic_pathology", "diffgui-next6-20260618-03-dock_le_aromatic_pathology_20260619_064544"),
    ("04", "ring_last3_pathology_guard", "diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410"),
    ("05", "chembl_aromatic_pathology", "diffgui-next6-20260618-05-chembl_aromatic_pathology_20260619_160546"),
    ("06", "dock_chembl_aromatic_pathology", "diffgui-next6-20260618-06-dock_chembl_aromatic_pathology_20260619_204502"),
]

CONTINUE_RUN = "diffgui-next6-20260618-04-ring_last3_pathology_guard-continue-to50_20260621_225705"

PROPERTIES = [
    ("mol_weight", "Mol. weight"),
    ("logp", "LogP"),
    ("tpsa", "TPSA"),
    ("hba", "HBA"),
    ("hbd", "HBD"),
    ("ring_count", "Ring count"),
    ("aromatic_ring_count", "Aromatic rings"),
    ("rotatable_bonds", "Rotatable bonds"),
]


@dataclass(frozen=True)
class SourceSpec:
    """One SDF source to analyze."""

    run_idx: str
    run_slug: str
    condition: str
    epoch: int
    sdf_path: Path
    source_type: str


def _max_fused_ring_system_size(mol: Chem.Mol) -> int:
    atom_rings = [tuple(ring) for ring in mol.GetRingInfo().AtomRings()]
    if not atom_rings:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency = [set() for _ in ring_sets]
    for left_idx, left in enumerate(ring_sets):
        for right_idx in range(left_idx + 1, len(ring_sets)):
            if len(left & ring_sets[right_idx]) >= 2:
                adjacency[left_idx].add(right_idx)
                adjacency[right_idx].add(left_idx)
    seen: set[int] = set()
    largest = 1
    for start in range(len(ring_sets)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in adjacency[node]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        largest = max(largest, size)
    return largest


def _benzene_like_ring_count(mol: Chem.Mol) -> int:
    count = 0
    for ring in mol.GetRingInfo().AtomRings():
        atoms = [mol.GetAtomWithIdx(atom_idx) for atom_idx in ring]
        if len(ring) == 6 and all(atom.GetIsAromatic() for atom in atoms) and all(atom.GetAtomicNum() == 6 for atom in atoms):
            count += 1
    return count


def _mol_rows(source: SourceSpec) -> list[dict[str, object]]:
    supplier = Chem.SDMolSupplier(str(source.sdf_path), sanitize=True, removeHs=False)
    rows: list[dict[str, object]] = []
    total_records = 0
    for mol_idx, mol in enumerate(supplier, start=1):
        total_records += 1
        if mol is None:
            continue
        ring_info = mol.GetRingInfo()
        atom_rings = list(ring_info.AtomRings())
        heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        carbon_count = sum(1 for atom in heavy_atoms if atom.GetAtomicNum() == 6)
        hetero_count = len(heavy_atoms) - carbon_count
        rows.append(
            {
                "run_idx": source.run_idx,
                "run_slug": source.run_slug,
                "condition": source.condition,
                "epoch": source.epoch,
                "source_type": source.source_type,
                "sdf_path": str(source.sdf_path.relative_to(ROOT)),
                "mol_index": mol_idx,
                "mol_weight": Descriptors.MolWt(mol),
                "logp": Crippen.MolLogP(mol),
                "tpsa": rdMolDescriptors.CalcTPSA(mol),
                "hba": Lipinski.NumHAcceptors(mol),
                "hbd": Lipinski.NumHDonors(mol),
                "ring_count": rdMolDescriptors.CalcNumRings(mol),
                "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
                "benzene_like_ring_count": _benzene_like_ring_count(mol),
                "three_member_ring_count": sum(1 for ring in atom_rings if len(ring) == 3),
                "max_fused_ring_system_size": _max_fused_ring_system_size(mol),
                "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
                "heavy_atom_count": len(heavy_atoms),
                "carbon_fraction": carbon_count / len(heavy_atoms) if heavy_atoms else np.nan,
                "hetero_atom_count": hetero_count,
            }
        )
    if total_records == 0:
        raise ValueError(f"No SDF records found: {source.sdf_path}")
    return rows


def collect_properties() -> pd.DataFrame:
    """Collect molecule-level descriptors for six runs and the 50-epoch continuation."""
    sources: list[SourceSpec] = []
    for idx, slug, dirname in RUNS:
        run_dir = OUT / dirname
        sources.append(
            SourceSpec(idx, slug, "E01", 1, run_dir / "epochs/epoch_0001/molecules.sdf", "next6_20ep")
        )
        sources.append(
            SourceSpec(idx, slug, "E20", 20, run_dir / "epochs/epoch_0020/molecules.sdf", "next6_20ep")
        )
    sources.append(
        SourceSpec(
            "04",
            "ring_last3_pathology_guard",
            "E50",
            50,
            OUT / CONTINUE_RUN / "epochs/epoch_0050/molecules.sdf",
            "continue_to50",
        )
    )
    all_rows: list[dict[str, object]] = []
    for source in sources:
        if not source.sdf_path.exists():
            raise FileNotFoundError(source.sdf_path)
        all_rows.extend(_mol_rows(source))
    return pd.DataFrame(all_rows)


def summarize_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize properties by run and condition."""
    metric_cols = [name for name, _ in PROPERTIES] + [
        "benzene_like_ring_count",
        "three_member_ring_count",
        "max_fused_ring_system_size",
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atom_count",
    ]
    grouped = df.groupby(["run_idx", "run_slug", "condition", "epoch"], sort=False)
    rows: list[dict[str, object]] = []
    for key, group in grouped:
        run_idx, run_slug, condition, epoch = key
        row: dict[str, object] = {
            "run_idx": run_idx,
            "run_slug": run_slug,
            "condition": condition,
            "epoch": int(epoch),
            "records": int(len(group)),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce").dropna()
            row[f"{col}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{col}_median"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_training() -> pd.DataFrame:
    """Read concise epoch-level metrics for the 20->50 continuation."""
    metrics_path = OUT / CONTINUE_RUN / "training_summary.csv"
    df = pd.read_csv(metrics_path)
    wanted = [
        "epoch",
        "reward_total_mean",
        "reconstruction_success_ratio",
        "molecule_rdkit_valid_ratio",
        "docking_success_ratio",
        "docking_raw_mean",
        "logp_mean",
        "tpsa_mean",
        "aromatic_ring_mean",
        "ring_count_mean",
        "three_member_ring_mean",
        "duplicate_ratio",
    ]
    return df[[col for col in wanted if col in df.columns]].copy()


def plot_violin_grid(df: pd.DataFrame, *, title: str, output: Path) -> None:
    """Plot property violin grid with mean and median annotations."""
    conditions = list(dict.fromkeys(df["condition"].tolist()))
    positions = np.arange(1, len(conditions) + 1)
    colors = {"E01": "#6aaed6", "E20": "#f18f3b", "E50": "#4daf4a"}

    fig, axes = plt.subplots(2, 4, figsize=(23, 11))
    axes = axes.ravel()
    for ax, (metric, label) in zip(axes, PROPERTIES):
        data = [
            pd.to_numeric(df.loc[df["condition"] == condition, metric], errors="coerce").dropna().to_numpy()
            for condition in conditions
        ]
        parts = ax.violinplot(data, positions=positions, widths=0.78, showextrema=False, showmeans=False, showmedians=False)
        for body, condition in zip(parts["bodies"], conditions):
            body.set_facecolor(colors.get(condition, "#999999"))
            body.set_edgecolor("#333333")
            body.set_alpha(0.65)
        for pos, values, condition in zip(positions, data, conditions):
            if len(values) == 0:
                continue
            mean_val = float(np.mean(values))
            median_val = float(np.median(values))
            ax.scatter(pos - 0.06, mean_val, marker="o", s=58, c="#111111", zorder=3)
            ax.scatter(pos + 0.06, median_val, marker="D", s=52, c="#ffffff", edgecolors="#111111", zorder=3)
            y_top = float(np.nanmax(values))
            y_bottom = float(np.nanmin(values))
            y_pad = (y_top - y_bottom) * 0.05 if y_top > y_bottom else 0.5
            ax.text(
                pos,
                y_top + y_pad,
                f"mean {mean_val:.2f}\nmed {median_val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(label, fontsize=15, fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels(conditions, fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, fontsize=19, fontweight="bold")
    fig.text(0.5, 0.02, "Black circle = mean; white diamond = median", ha="center", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_training_metrics(training_df: pd.DataFrame, output: Path) -> None:
    """Plot key 20->50 continuation metrics."""
    metrics = [
        ("reward_total_mean", "Reward mean"),
        ("docking_raw_mean", "Docking raw mean"),
        ("molecule_rdkit_valid_ratio", "RDKit valid ratio"),
        ("reconstruction_success_ratio", "Reconstruction ratio"),
        ("logp_mean", "LogP mean"),
        ("ring_count_mean", "Ring count mean"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()
    for ax, (col, label) in zip(axes, metrics):
        if col not in training_df.columns:
            ax.axis("off")
            continue
        ax.plot(training_df["epoch"], training_df[col], marker="o", linewidth=2.2, markersize=4.5)
        first = training_df.iloc[0][col]
        last = training_df.iloc[-1][col]
        ax.set_title(f"{label}: {first:.3g} -> {last:.3g}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=12)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
    fig.suptitle("DiffGui next6-04 continuation: epoch 21-50", fontsize=18, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_report(summary: pd.DataFrame, training_df: pd.DataFrame) -> None:
    """Write a concise Markdown report."""
    e50 = training_df.iloc[-1]
    e21 = training_df.iloc[0]
    run04 = summary[(summary["run_idx"] == "04") & (summary["condition"].isin(["E20", "E50"]))]
    display_cols = [
        "run_idx",
        "condition",
        "records",
        "mol_weight_mean",
        "mol_weight_median",
        "logp_mean",
        "logp_median",
        "tpsa_mean",
        "tpsa_median",
        "ring_count_mean",
        "ring_count_median",
        "aromatic_ring_count_mean",
        "aromatic_ring_count_median",
    ]
    run04_table = _markdown_table(run04[[col for col in display_cols if col in run04.columns]])
    lines = [
        "# DiffGui next6 20/50 epoch property analysis",
        "",
        "## 50 轮续训情况",
        "",
        f"- 续训目录：`OUT/{CONTINUE_RUN}`",
        "- 口径：从原始 04 的 epoch 20 checkpoint 续训，输出 epoch 21-50；训练日志里 `epochs=30, epoch_offset=20, final_epoch=50`。",
        f"- epoch 50：reward mean `{e50.get('reward_total_mean', np.nan):.4f}`，docking raw mean `{e50.get('docking_raw_mean', np.nan):.3f}`，RDKit valid ratio `{e50.get('molecule_rdkit_valid_ratio', np.nan):.3f}`，reconstruction ratio `{e50.get('reconstruction_success_ratio', np.nan):.3f}`。",
        f"- epoch 21 到 50：reward mean `{e21.get('reward_total_mean', np.nan):.4f}` -> `{e50.get('reward_total_mean', np.nan):.4f}`；LogP mean `{e21.get('logp_mean', np.nan):.2f}` -> `{e50.get('logp_mean', np.nan):.2f}`；ring count mean `{e21.get('ring_count_mean', np.nan):.2f}` -> `{e50.get('ring_count_mean', np.nan):.2f}`。",
        "",
        "![50 epoch training metrics](figures/continue50_training_metrics.png)",
        "",
        "## 六个 20 轮训练前后性质分布",
        "",
        "每个 run 取 `epoch_0001` 和 `epoch_0020` 的 valid RDKit 分子，计算分子量、LogP、TPSA、HBA、HBD、总环数、芳香环数、可旋转键数。图中黑点为平均值，白色菱形为中位数。",
        "",
        "![six-run property violin](figures/next6_epoch1_vs20_property_violin_grid.png)",
        "",
        "## 04 续训 20 到 50 对比",
        "",
        "![run04 epoch20 to epoch50](figures/run04_epoch20_vs50_property_violin_grid.png)",
        "",
        "## 表格",
        "",
        "- `tables/property_summary.csv`：每个 run/epoch 的 mean 和 median。",
        "- `tables/properties_long.csv`：逐分子性质表。",
        "- `tables/continue50_training_metrics.csv`：epoch 21-50 训练指标。",
        "",
        "### 04 epoch 20 vs 50 摘要",
        "",
        run04_table,
        "",
    ]
    (ANALYSIS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a Markdown table without optional dependencies."""
    if df.empty:
        return "_No rows._"
    formatted = df.copy()
    for col in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[col]):
            formatted[col] = formatted[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
    headers = [str(col) for col in formatted.columns]
    rows = [[str(value) for value in row] for row in formatted.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    properties = collect_properties()
    summary = summarize_properties(properties)
    training = summarize_training()

    properties.to_csv(TABLE_DIR / "properties_long.csv", index=False)
    summary.to_csv(TABLE_DIR / "property_summary.csv", index=False)
    training.to_csv(TABLE_DIR / "continue50_training_metrics.csv", index=False)

    plot_violin_grid(
        properties[properties["condition"].isin(["E01", "E20"])],
        title="DiffGui next6: epoch 1 vs epoch 20 property distributions",
        output=FIGURE_DIR / "next6_epoch1_vs20_property_violin_grid.png",
    )
    plot_violin_grid(
        properties[(properties["run_idx"] == "04") & (properties["condition"].isin(["E20", "E50"]))],
        title="DiffGui next6-04: epoch 20 vs epoch 50 property distributions",
        output=FIGURE_DIR / "run04_epoch20_vs50_property_violin_grid.png",
    )
    plot_training_metrics(training, FIGURE_DIR / "continue50_training_metrics.png")
    write_report(summary, training)
    print(f"[done] molecules={len(properties)} summary_rows={len(summary)} output={ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
