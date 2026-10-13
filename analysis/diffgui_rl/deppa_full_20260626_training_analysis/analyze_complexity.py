"""Analyze molecular size and complexity drift across DiffGui DePPA training epochs."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.edgecolor"] = "#d4d4d4"
plt.rcParams["axes.linewidth"] = 1.2

RUN_DIR = Path("/home/md666/data/disk6/DiffB110/runs/diffgui-deppa-full-lr1e5-wd1e4-b8x8-t200-100ep_20260626_165809")
OUT_DIR = Path(__file__).resolve().parent
SUMMARY_HEADER = [
    "epoch",
    "n_total",
    "n_valid",
    "num_atoms_mean",
    "num_atoms_median",
    "num_atoms_q10",
    "num_atoms_q90",
    "heavy_atoms_mean",
    "heavy_atoms_median",
    "heavy_atoms_q10",
    "heavy_atoms_q90",
    "mw_mean",
    "mw_median",
    "logp_mean",
    "logp_median",
    "tpsa_mean",
    "tpsa_median",
    "qed_mean",
    "qed_median",
    "sa_raw_mean",
    "sa_raw_median",
    "rotatable_bonds_mean",
    "rotatable_bonds_median",
    "num_rings_mean",
    "num_rings_median",
    "aromatic_rings_mean",
    "aromatic_rings_median",
    "hetero_atoms_mean",
    "hetero_atoms_median",
    "carbon_fraction_mean",
    "carbon_fraction_median",
    "has_rings_ratio",
    "has_aromatic_ring_ratio",
    "has_hetero_atom_ratio",
]


def _compute_mol_descriptors(mol: Chem.Mol) -> dict[str, float | int]:
    """Return a dictionary of size/complexity descriptors for a valid RDKit mol."""
    heavy_atoms = mol.GetNumHeavyAtoms()
    atom_counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        atom_counts[atom.GetSymbol()] = atom_counts.get(atom.GetSymbol(), 0) + 1
    carbon_count = atom_counts.get("C", 0)
    return {
        "heavy_atoms": heavy_atoms,
        "mw": float(Descriptors.MolWt(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "qed": float(QED.qed(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "num_rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_rings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "hetero_atoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "carbon_fraction": float(carbon_count / heavy_atoms) if heavy_atoms > 0 else 0.0,
    }


def _quantile(values: pd.Series, q: float) -> float:
    """Return quantile or NaN if no data."""
    if values.empty:
        return math.nan
    return float(values.quantile(q))


def _load_epoch_molecules(epoch_dir: Path) -> pd.DataFrame:
    """Load the merged molecules.csv for one epoch."""
    csv_path = epoch_dir / "molecules.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    df["epoch"] = int(epoch_dir.name.split("_")[-1])
    return df


def _per_molecule_descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptors for valid molecules in a merged epoch dataframe."""
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        base = dict(row)
        if int(base.get("rdkit_valid", 0)) != 1:
            base["descriptor_success"] = 0
            rows.append(base)
            continue
        smiles = str(base.get("canonical_smiles", ""))
        if not smiles:
            base["descriptor_success"] = 0
            rows.append(base)
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            base["descriptor_success"] = 0
            rows.append(base)
            continue
        desc = _compute_mol_descriptors(mol)
        base.update(desc)
        base["descriptor_success"] = 1
        rows.append(base)
    return pd.DataFrame(rows)


def _aggregate_epoch(df: pd.DataFrame) -> dict[str, Any]:
    """Return per-epoch summary statistics for valid molecules."""
    valid_df = df[df["rdkit_valid"] == 1]
    desc_df = valid_df[valid_df["descriptor_success"] == 1]
    out: dict[str, Any] = {"epoch": int(df["epoch"].iloc[0]), "n_total": len(df), "n_valid": len(valid_df)}

    for col in [
        "num_atoms",
        "heavy_atoms",
        "mw",
        "logp",
        "tpsa",
        "qed",
        "sa_raw",
        "rotatable_bonds",
        "num_rings",
        "aromatic_rings",
        "hetero_atoms",
        "carbon_fraction",
    ]:
        if col not in desc_df.columns or desc_df[col].isna().all():
            out[f"{col}_mean"] = math.nan
            out[f"{col}_median"] = math.nan
            out[f"{col}_q10"] = math.nan
            out[f"{col}_q90"] = math.nan
            continue
        s = desc_df[col].dropna()
        out[f"{col}_mean"] = float(s.mean())
        out[f"{col}_median"] = float(s.median())
        out[f"{col}_q10"] = _quantile(s, 0.10)
        out[f"{col}_q90"] = _quantile(s, 0.90)

    out["has_rings_ratio"] = float((desc_df["num_rings"] > 0).mean()) if not desc_df.empty else math.nan
    out["has_aromatic_ring_ratio"] = float((desc_df["aromatic_rings"] > 0).mean()) if not desc_df.empty else math.nan
    out["has_hetero_atom_ratio"] = float((desc_df["hetero_atoms"] > 0).mean()) if not desc_df.empty else math.nan
    return out


def _spine_style(ax: plt.Axes) -> None:
    """Apply project-standard spine styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _plot_time_series(summary: pd.DataFrame, out_dir: Path) -> None:
    """Plot mean descriptors over epochs."""
    metrics = [
        ("num_atoms", "Number of heavy atoms"),
        ("heavy_atoms", "Heavy atoms"),
        ("mw", "Molecular weight (Da)"),
        ("logp", "LogP"),
        ("tpsa", "TPSA (Å²)"),
        ("qed", "QED"),
        ("sa_raw", "SA score"),
        ("rotatable_bonds", "Rotatable bonds"),
        ("num_rings", "Rings"),
        ("aromatic_rings", "Aromatic rings"),
        ("hetero_atoms", "Heteroatoms"),
        ("carbon_fraction", "Carbon fraction"),
    ]

    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows), dpi=300)
    axes = axes.flatten()
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        mean_col = f"{col}_mean"
        ax.plot(summary["epoch"], summary[mean_col], color="#3498DB", linewidth=1.5)
        ax.fill_between(
            summary["epoch"],
            summary[f"{col}_q10"],
            summary[f"{col}_q90"],
            color="#3498DB",
            alpha=0.15,
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(label)
        _spine_style(ax)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "complexity_time_series.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_window_comparison(per_mol: pd.DataFrame, out_dir: Path) -> None:
    """Violin plots comparing early (1-10) vs late (80-89) epochs."""
    per_mol = per_mol[per_mol["rdkit_valid"] == 1].copy()
    per_mol["window"] = per_mol["epoch"].apply(
        lambda e: "early (1-10)" if e <= 10 else ("late (80-89)" if 80 <= e <= 89 else "other")
    )
    plot_df = per_mol[per_mol["window"].isin(["early (1-10)", "late (80-89)"])]
    metrics = [
        ("num_atoms", "Heavy atoms"),
        ("mw", "Molecular weight (Da)"),
        ("rotatable_bonds", "Rotatable bonds"),
        ("num_rings", "Rings"),
        ("aromatic_rings", "Aromatic rings"),
        ("hetero_atoms", "Heteroatoms"),
        ("carbon_fraction", "Carbon fraction"),
        ("logp", "LogP"),
        ("tpsa", "TPSA (Å²)"),
        ("qed", "QED"),
        ("sa_raw", "SA score"),
    ]

    n_cols = 3
    n_rows = math.ceil(len(metrics) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4.5 * n_rows), dpi=300)
    axes = axes.flatten()
    palette = {"early (1-10)": "#E74C3C", "late (80-89)": "#3498DB"}
    for idx, (col, label) in enumerate(metrics):
        ax = axes[idx]
        sns.violinplot(
            data=plot_df,
            x="window",
            y=col,
            palette=palette,
            inner="quartile",
            ax=ax,
            order=["early (1-10)", "late (80-89)"],
        )
        sns.stripplot(
            data=plot_df,
            x="window",
            y=col,
            color="black",
            alpha=0.10,
            size=2,
            ax=ax,
            order=["early (1-10)", "late (80-89)"],
        )
        ax.set_xlabel("")
        ax.set_ylabel(label)
        ax.set_title(label)
        _spine_style(ax)
    for idx in range(len(metrics), len(axes)):
        axes[idx].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "complexity_window_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(per_mol: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: docking score vs size/complexity descriptors."""
    df = per_mol[per_mol["rdkit_valid"] == 1].copy()
    cols = ["num_atoms", "heavy_atoms", "rotatable_bonds", "num_rings", "aromatic_rings", "hetero_atoms"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=300)
    axes = axes.flatten()
    for idx, col in enumerate(cols):
        ax = axes[idx]
        ax.scatter(df[col], df["docking_raw"], alpha=0.3, s=10, color="#3498DB")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Docking raw score")
        ax.set_title(col.replace("_", " ").title())
        _spine_style(ax)
    fig.tight_layout()
    fig.savefig(out_dir / "complexity_vs_docking.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Entry point: load data, compute descriptors, save tables and plots."""
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_dir = RUN_DIR / "epochs"

    epoch_dirs = sorted(epochs_dir.glob("epoch_*"))
    logger.info("Found %d epoch directories", len(epoch_dirs))

    per_mol_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    for epoch_dir in epoch_dirs:
        df = _load_epoch_molecules(epoch_dir)
        if df.empty:
            logger.warning("No molecules.csv for %s", epoch_dir.name)
            continue
        desc_df = _per_molecule_descriptors(df)
        per_mol_rows.append(desc_df)
        summary_rows.append(_aggregate_epoch(desc_df))

    per_mol = pd.concat(per_mol_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    per_mol_path = out_dir / "per_molecule_complexity.csv"
    summary_path = out_dir / "per_epoch_complexity_summary.csv"
    per_mol.to_csv(per_mol_path, index=False)
    summary.to_csv(summary_path, index=False)
    logger.info("Wrote %s and %s", per_mol_path.name, summary_path.name)

    _plot_time_series(summary, out_dir)
    _plot_window_comparison(per_mol, out_dir)
    _plot_correlation(per_mol, out_dir)
    logger.info("Plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
