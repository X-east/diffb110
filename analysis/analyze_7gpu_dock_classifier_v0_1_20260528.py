#!/usr/bin/env python3
"""Detailed analysis for the 2026-05-28 docking + classifier DDPO run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "OUT" / "7gpu-dock-classifier-v0-1_20260528_000841"
)
SASCORER_PATH = PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score" / "sascorer.py"

PLOT_DPI = 160
VALID_COLOR = "#2A9D8F"
REWARD_COLOR = "#E76F51"
DOCK_COLOR = "#457B9D"
CLASSIFIER_COLOR = "#8E6C8A"
FAIL_COLOR = "#D62828"
NEUTRAL_COLOR = "#6C757D"
ACCENT_COLOR = "#F4A261"


def _load_sa_scorer():
    if not SASCORER_PATH.exists():
        raise FileNotFoundError(f"SA scorer not found: {SASCORER_PATH}")
    spec = importlib.util.spec_from_file_location("diffsbdd_sascorer", SASCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SA scorer from: {SASCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculateScore


calculate_sa_score = _load_sa_scorer()


def _parse_failure_counts(value: object) -> dict[str, int]:
    if pd.isna(value):
        return {}
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    text = str(value).strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = json.loads(text.replace("'", '"'))
        except json.JSONDecodeError:
            return {}
    return {str(k): int(v) for k, v in data.items()}


def _safe_float(value: object) -> float:
    try:
        if pd.isna(value):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _lipinski_pass(mol: Chem.Mol) -> int:
    logp = Crippen.MolLogP(mol)
    return int(
        Descriptors.ExactMolWt(mol) < 500
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
        and -2 <= logp <= 5
        and rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    )


def _descriptor_record(row: pd.Series) -> dict[str, object]:
    smiles = row.get("smiles")
    base = {
        "epoch": int(row["epoch"]),
        "batch_idx": int(row["batch_idx"]),
        "rank": int(row["rank"]),
        "molecule_id": int(row["molecule_id"]),
        "smiles": smiles if isinstance(smiles, str) else "",
        "reward_total": _safe_float(row.get("reward_total")),
        "reward_total_raw": _safe_float(row.get("reward_total_raw")),
        "reward_total_norm": _safe_float(row.get("reward_total_norm")),
        "docking_raw": _safe_float(row.get("docking_raw")),
        "docking_reward": _safe_float(row.get("docking_reward")),
        "chembl_classifier_reward": _safe_float(row.get("chembl_classifier_reward")),
        "ligand_efficiency_raw": _safe_float(row.get("ligand_efficiency_raw")),
        "rdkit_valid": int(row.get("rdkit_valid", 0) == 1),
    }
    if not isinstance(smiles, str) or not smiles.strip() or base["rdkit_valid"] != 1:
        base["descriptor_valid"] = 0
        return base

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        base["descriptor_valid"] = 0
        return base

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        base["descriptor_valid"] = 0
        return base

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    heavy = mol.GetNumHeavyAtoms()
    carbon = sum(symbol == "C" for symbol in atoms)
    hetero = sum(symbol != "C" and symbol != "H" for symbol in atoms)
    halogen = sum(symbol in {"F", "Cl", "Br", "I"} for symbol in atoms)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    base.update(
        {
            "descriptor_valid": 1,
            "mol_wt": Descriptors.MolWt(mol),
            "exact_mol_wt": Descriptors.ExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "heavy_atom_count": int(heavy),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
            "saturated_ring_count": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
            "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
            "formal_charge": int(Chem.GetFormalCharge(mol)),
            "qed": QED.qed(mol),
            "sa_score": calculate_sa_score(mol),
            "lipinski_pass": _lipinski_pass(mol),
            "carbon_atoms": int(carbon),
            "hetero_atoms": int(hetero),
            "n_atoms": int(len(atoms)),
            "n_n": int(sum(symbol == "N" for symbol in atoms)),
            "n_o": int(sum(symbol == "O" for symbol in atoms)),
            "n_s": int(sum(symbol == "S" for symbol in atoms)),
            "n_halogen": int(halogen),
            "carbon_fraction": carbon / heavy if heavy else np.nan,
            "hetero_fraction": hetero / heavy if heavy else np.nan,
            "murcko_scaffold": scaffold,
            "has_scaffold": int(bool(scaffold)),
        }
    )
    return base


def _read_summary(run_dir: Path) -> pd.DataFrame:
    summary = pd.read_csv(run_dir / "training_summary.csv")
    failure_df = (
        summary["failure_counts"]
        .apply(_parse_failure_counts)
        .apply(pd.Series)
        .fillna(0)
        .astype(int)
    )
    for col in failure_df.columns:
        summary[f"failure_{col}"] = failure_df[col]
    failure_cols = [
        c
        for c in summary.columns
        if c.startswith("failure_") and pd.api.types.is_numeric_dtype(summary[c])
    ]
    summary["failed_total"] = summary[failure_cols].sum(axis=1) - summary.get(
        "failure_success", 0
    )
    summary["failed_ratio"] = summary["failed_total"] / summary["generated_total"]
    summary["docking_backend_failure_ratio"] = summary.get(
        "failure_FAILURE_DOCKING_BACKEND", 0
    ) / summary["generated_total"]
    return summary


def _load_molecules(run_dir: Path) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((run_dir / "epochs").glob("epoch_*/molecules.csv")):
        frames.append(pd.read_csv(csv_path))
    if not frames:
        raise FileNotFoundError(f"No epoch molecule CSV files found under {run_dir}")
    molecules = pd.concat(frames, ignore_index=True)
    records = [_descriptor_record(row) for _, row in molecules.iterrows()]
    props = pd.DataFrame(records)
    numeric_cols = [c for c in props.columns if c not in {"smiles", "murcko_scaffold"}]
    props[numeric_cols] = props[numeric_cols].apply(pd.to_numeric, errors="coerce")
    return props


def _numeric_summary(
    df: pd.DataFrame, columns: Iterable[str], group_col: str | None = None
) -> pd.DataFrame:
    present = [c for c in columns if c in df.columns]
    if group_col is None:
        out = df[present].agg(["count", "mean", "median", "std", "min", "max"]).T
        out["q25"] = df[present].quantile(0.25)
        out["q75"] = df[present].quantile(0.75)
        return out.reset_index(names="metric")
    grouped = df.groupby(group_col)[present]
    parts = [
        grouped.mean().add_suffix("_mean"),
        grouped.median().add_suffix("_median"),
        grouped.std().add_suffix("_std"),
        grouped.quantile(0.25).add_suffix("_q25"),
        grouped.quantile(0.75).add_suffix("_q75"),
    ]
    return pd.concat(parts, axis=1).reset_index()


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _finish_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_training_overview(summary: pd.DataFrame, out_dir: Path) -> None:
    epochs = summary["epoch"]
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Training Overview: Docking + ChEMBL Classifier", fontsize=18, weight="bold")

    ax = axes[0, 0]
    ax.plot(epochs, summary["avg_loss"], marker="o", color=NEUTRAL_COLOR)
    ax.set_title("PPO Loss")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    ax.plot(epochs, summary["rdkit_valid_ratio"] * 100, marker="o", color=VALID_COLOR)
    ax.set_title("RDKit Valid Ratio")
    ax.set_ylabel("%")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    ax.plot(epochs, summary["reward_mean"], marker="o", color=REWARD_COLOR, label="normalized/clamped")
    ax.plot(epochs, summary["avg_reward_raw"], marker="s", color=ACCENT_COLOR, label="raw", alpha=0.8)
    ax.axhline(0, color="#999999", lw=1, ls="--")
    ax.set_title("Reward Mean")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(epochs, summary["avg_docking"], marker="o", color=DOCK_COLOR)
    ax.set_title("Docking Score (lower is better)")
    ax.set_ylabel("kcal/mol")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(epochs, summary["avg_chembl_classifier_reward"], marker="o", color=CLASSIFIER_COLOR)
    ax.set_title("ChEMBL Classifier Reward")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax = axes[1, 2]
    ax.stackplot(
        epochs,
        summary.get("failure_success", 0),
        summary.get("failure_fragmented_molecule", 0),
        summary.get("failure_sanitize_failed", 0),
        summary.get("failure_FAILURE_DOCKING_BACKEND", 0),
        labels=["success", "fragmented", "sanitize_failed", "docking_backend"],
        colors=[VALID_COLOR, ACCENT_COLOR, FAIL_COLOR, NEUTRAL_COLOR],
        alpha=0.78,
    )
    ax.set_title("Outcome Breakdown")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Molecules")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)

    ax = axes[2, 0]
    ax.plot(epochs, summary["carbon_fraction"] * 100, marker="o", color=FAIL_COLOR, label="carbon %")
    ax2 = ax.twinx()
    ax2.plot(epochs, summary["hetero_atom_mean"], marker="s", color=DOCK_COLOR, label="hetero atoms")
    ax.set_title("Composition")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Carbon %")
    ax2.set_ylabel("Hetero atom count")
    ax.grid(alpha=0.25)

    ax = axes[2, 1]
    ax.plot(epochs, summary["descriptor_mmd_vs_baseline"], marker="o", color=CLASSIFIER_COLOR)
    ax.set_title("Descriptor MMD vs Baseline")
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.25)

    ax = axes[2, 2]
    ax.plot(epochs, summary["time_generate"], label="generate", marker="o")
    ax.plot(epochs, summary["time_reward_docking"], label="docking", marker="s")
    ax.plot(epochs, summary["time_train"], label="train", marker="^")
    ax.set_title("Timing")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    _finish_plot(fig, out_dir / "01_training_overview.png")


def _plot_property_trends(epoch_props: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.suptitle("Molecular Property Trends from Valid SMILES", fontsize=18, weight="bold")
    epoch = epoch_props["epoch"]
    specs = [
        ("logp_mean", "logp_q25", "logp_q75", "LogP"),
        ("tpsa_mean", "tpsa_q25", "tpsa_q75", "TPSA"),
        ("mol_wt_mean", "mol_wt_q25", "mol_wt_q75", "Molecular Weight"),
        ("heavy_atom_count_mean", "heavy_atom_count_q25", "heavy_atom_count_q75", "Heavy Atoms"),
        ("hba_mean", "hba_q25", "hba_q75", "HBA"),
        ("hbd_mean", "hbd_q25", "hbd_q75", "HBD"),
        ("ring_count_mean", "ring_count_q25", "ring_count_q75", "Rings"),
        ("aromatic_ring_count_mean", "aromatic_ring_count_q25", "aromatic_ring_count_q75", "Aromatic Rings"),
        ("sa_score_mean", "sa_score_q25", "sa_score_q75", "SA Score"),
    ]
    for ax, (mean_col, q25_col, q75_col, title) in zip(axes.ravel(), specs):
        ax.fill_between(epoch, epoch_props[q25_col], epoch_props[q75_col], alpha=0.22, color=DOCK_COLOR)
        ax.plot(epoch, epoch_props[mean_col], marker="o", color=DOCK_COLOR)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    _finish_plot(fig, out_dir / "02_property_trends.png")


def _plot_final_distributions(final_valid: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    fig.suptitle("Final Epoch Property Distributions", fontsize=18, weight="bold")
    cols = [
        ("reward_total", "Reward"),
        ("docking_raw", "Docking raw"),
        ("chembl_classifier_reward", "Classifier reward"),
        ("mol_wt", "MolWt"),
        ("logp", "LogP"),
        ("tpsa", "TPSA"),
        ("hba", "HBA"),
        ("hbd", "HBD"),
        ("rot_bonds", "Rot bonds"),
        ("ring_count", "Rings"),
        ("qed", "QED"),
        ("sa_score", "SA score"),
    ]
    for ax, (col, title) in zip(axes.ravel(), cols):
        values = final_valid[col].dropna()
        ax.hist(values, bins=28, color=DOCK_COLOR, alpha=0.82)
        ax.axvline(values.median(), color=REWARD_COLOR, ls="--", lw=1.5, label="median")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    _finish_plot(fig, out_dir / "03_final_epoch_distributions.png")


def _plot_correlations(final_valid: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = [
        "reward_total",
        "reward_total_raw",
        "docking_raw",
        "docking_reward",
        "chembl_classifier_reward",
        "ligand_efficiency_raw",
        "mol_wt",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rot_bonds",
        "ring_count",
        "aromatic_ring_count",
        "qed",
        "sa_score",
        "carbon_fraction",
        "hetero_atoms",
    ]
    present = [c for c in cols if c in final_valid.columns]
    corr = final_valid[present].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(present)), present, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(present)), present, fontsize=8)
    ax.set_title("Final Epoch Spearman Correlation")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _finish_plot(fig, out_dir / "04_reward_property_correlations.png")
    return corr.reset_index(names="metric")


def _plot_top_vs_all(final_valid: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    final = final_valid.copy()
    reward_cut = final["reward_total"].quantile(0.9)
    docking_cut = final["docking_raw"].quantile(0.1)
    final["group"] = "all_valid"
    final.loc[final["reward_total"] >= reward_cut, "group"] = "top10_reward"
    final.loc[final["docking_raw"] <= docking_cut, "group"] = "top10_docking"
    props = ["mol_wt", "logp", "tpsa", "hba", "hbd", "rot_bonds", "ring_count", "qed", "sa_score"]
    rows = []
    for group, group_df in final.groupby("group"):
        for prop in props:
            rows.append(
                {
                    "group": group,
                    "metric": prop,
                    "count": len(group_df[prop].dropna()),
                    "mean": group_df[prop].mean(),
                    "median": group_df[prop].median(),
                    "q25": group_df[prop].quantile(0.25),
                    "q75": group_df[prop].quantile(0.75),
                }
            )
    table = pd.DataFrame(rows)

    pivot = table.pivot(index="metric", columns="group", values="median")
    fig, ax = plt.subplots(figsize=(13, 7))
    pivot.plot(kind="bar", ax=ax, color=[NEUTRAL_COLOR, DOCK_COLOR, REWARD_COLOR])
    ax.set_title("Final Epoch Median Properties: All vs Top Molecules")
    ax.set_ylabel("Median")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="")
    _finish_plot(fig, out_dir / "05_top_vs_all_properties.png")
    return table


def _plot_rank_and_scaffold(props: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    final_epoch = int(props["epoch"].max())
    final = props[props["epoch"] == final_epoch].copy()
    rank = (
        final.groupby("rank")
        .agg(
            total=("molecule_id", "count"),
            valid=("descriptor_valid", "sum"),
            reward_median=("reward_total", "median"),
            docking_median=("docking_raw", "median"),
            classifier_median=("chembl_classifier_reward", "median"),
        )
        .reset_index()
    )
    rank["valid_ratio"] = rank["valid"] / rank["total"]

    valid = final[final["descriptor_valid"] == 1]
    scaffold_counts = valid["murcko_scaffold"].fillna("").replace("", "(no scaffold)").value_counts().head(15)
    element_cols = ["carbon_atoms", "hetero_atoms", "n_n", "n_o", "n_s", "n_halogen"]
    element_means = valid[element_cols].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    ax = axes[0]
    ax.bar(rank["rank"].astype(str), rank["valid_ratio"] * 100, color=VALID_COLOR)
    ax.set_title("Final Epoch Valid Ratio by Rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("%")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    scaffold_counts.sort_values().plot(kind="barh", ax=ax, color=DOCK_COLOR)
    ax.set_title("Top Murcko Scaffolds")
    ax.set_xlabel("Count")

    ax = axes[2]
    element_means.plot(kind="bar", ax=ax, color=ACCENT_COLOR)
    ax.set_title("Mean Element Counts")
    ax.grid(axis="y", alpha=0.25)

    _finish_plot(fig, out_dir / "06_rank_scaffold_elements.png")
    return rank


def _plot_failure_and_timing(summary: pd.DataFrame, out_dir: Path) -> None:
    epochs = summary["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    ax = axes[0]
    failure_cols = [
        c
        for c in [
            "failure_fragmented_molecule",
            "failure_sanitize_failed",
            "failure_FAILURE_DOCKING_BACKEND",
        ]
        if c in summary.columns
    ]
    for col in failure_cols:
        ax.plot(epochs, summary[col], marker="o", label=col.replace("failure_", ""))
    ax.set_title("Failure Modes")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1]
    timing_cols = [
        "time_generate",
        "time_reward_docking",
        "time_reward_chembl_classifier",
        "time_reward_normalize",
        "time_train",
    ]
    for col in timing_cols:
        if col in summary.columns:
            ax.plot(epochs, summary[col], marker="o", label=col)
    ax.set_title("Runtime Components")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Seconds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _finish_plot(fig, out_dir / "07_failures_and_timing.png")


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.{digits}f}"


def _markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    small = df.head(max_rows).copy()
    if small.empty:
        return "_empty_"

    def format_cell(value: object) -> str:
        if isinstance(value, float):
            if math.isnan(value):
                return "nan"
            return f"{value:.4f}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    headers = [str(c) for c in small.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(format_cell(row[c]) for c in small.columns) + " |")
    return "\n".join(lines)


def _write_report(
    run_dir: Path,
    out_dir: Path,
    summary: pd.DataFrame,
    props: pd.DataFrame,
    final_summary: pd.DataFrame,
    top_reward: pd.DataFrame,
    top_docking: pd.DataFrame,
    rank_summary: pd.DataFrame,
) -> None:
    final_epoch = int(summary["epoch"].max())
    first = summary.iloc[0]
    last = summary.iloc[-1]
    best_docking = summary.loc[summary["avg_docking"].idxmin()]
    best_valid = summary.loc[summary["rdkit_valid_ratio"].idxmax()]
    best_reward = summary.loc[summary["reward_mean"].idxmax()]
    valid_props = props[props["descriptor_valid"] == 1]
    final_valid = valid_props[valid_props["epoch"] == final_epoch]
    unique_smiles = int(final_valid["smiles"].nunique())
    unique_scaffolds = int(final_valid["murcko_scaffold"].replace("", np.nan).nunique())

    metric_rows = pd.DataFrame(
        [
            ["epochs", len(summary)],
            ["generated_total", int(summary["generated_total"].sum())],
            ["rdkit_valid_total", int(summary["rdkit_valid"].sum())],
            ["final_valid_ratio", last["rdkit_valid_ratio"]],
            ["best_valid_ratio_epoch", int(best_valid["epoch"])],
            ["best_valid_ratio", best_valid["rdkit_valid_ratio"]],
            ["final_avg_docking", last["avg_docking"]],
            ["best_avg_docking_epoch", int(best_docking["epoch"])],
            ["best_avg_docking", best_docking["avg_docking"]],
            ["final_classifier_reward", last["avg_chembl_classifier_reward"]],
            ["final_reward_mean", last["reward_mean"]],
            ["best_reward_epoch", int(best_reward["epoch"])],
            ["best_reward_mean", best_reward["reward_mean"]],
            ["final_descriptor_mmd", last["descriptor_mmd_vs_baseline"]],
            ["final_unique_smiles", unique_smiles],
            ["final_unique_scaffolds", unique_scaffolds],
        ],
        columns=["metric", "value"],
    )
    metric_rows.to_csv(out_dir / "run_key_numbers.csv", index=False)

    prop_focus = final_summary[
        final_summary["metric"].isin(
            [
                "mol_wt",
                "logp",
                "tpsa",
                "hba",
                "hbd",
                "rot_bonds",
                "heavy_atom_count",
                "ring_count",
                "aromatic_ring_count",
                "qed",
                "sa_score",
                "carbon_fraction",
                "hetero_atoms",
                "lipinski_pass",
            ]
        )
    ]

    lines = [
        "# 7gpu-dock-classifier-v0-1 训练详细分析",
        "",
        f"- Run: `{run_dir}`",
        "- 配置: docking=0.5, chembl_classifier=0.5, per-component normalization enabled, total reward z-score + clamp [-2, 2]",
        f"- 状态: completed, epoch 1-{final_epoch}",
        f"- 总生成: {int(summary['generated_total'].sum())}；RDKit valid: {int(summary['rdkit_valid'].sum())} ({summary['rdkit_valid'].sum() / summary['generated_total'].sum():.2%})",
        f"- 末轮: valid={int(last['rdkit_valid'])}/{int(last['generated_total'])} ({last['rdkit_valid_ratio']:.2%}), avg_docking={last['avg_docking']:.4f}, classifier_reward={last['avg_chembl_classifier_reward']:.4f}, reward_mean={last['reward_mean']:.4f}",
        f"- 末轮多样性: unique_smiles={unique_smiles}, unique_murcko_scaffolds={unique_scaffolds}",
        "",
        "## 图表索引",
        "",
        "![training overview](01_training_overview.png)",
        "",
        "![property trends](02_property_trends.png)",
        "",
        "![final distributions](03_final_epoch_distributions.png)",
        "",
        "![correlations](04_reward_property_correlations.png)",
        "",
        "![top vs all](05_top_vs_all_properties.png)",
        "",
        "![rank scaffold elements](06_rank_scaffold_elements.png)",
        "",
        "![failures and timing](07_failures_and_timing.png)",
        "",
        "## 关键数值",
        "",
        _markdown_table(metric_rows, max_rows=50),
        "",
        "## 末轮性质汇总",
        "",
        _markdown_table(prop_focus, max_rows=50),
        "",
        "## 末轮按 reward 排名前 20",
        "",
        _markdown_table(
            top_reward[
                [
                    "rank",
                    "molecule_id",
                    "reward_total",
                    "reward_total_raw",
                    "docking_raw",
                    "chembl_classifier_reward",
                    "mol_wt",
                    "logp",
                    "tpsa",
                    "qed",
                    "sa_score",
                    "smiles",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## 末轮按 docking 排名前 20",
        "",
        _markdown_table(
            top_docking[
                [
                    "rank",
                    "molecule_id",
                    "docking_raw",
                    "docking_reward",
                    "reward_total",
                    "chembl_classifier_reward",
                    "mol_wt",
                    "logp",
                    "tpsa",
                    "qed",
                    "sa_score",
                    "smiles",
                ]
            ],
            max_rows=20,
        ),
        "",
        "## 末轮 rank 平衡",
        "",
        _markdown_table(rank_summary, max_rows=20),
        "",
        "## 文件说明",
        "",
        "- `epoch_summary_key_metrics.csv`: 训练级指标、失败模式与计时。",
        "- `all_valid_molecule_properties.csv`: 所有 epoch 的有效分子性质与奖励记录。",
        "- `epoch_property_summary.csv`: 每个 epoch 的性质均值/中位数/分位数。",
        "- `final_epoch_property_summary.csv`: 末轮性质描述统计。",
        "- `top_molecules_by_reward.csv` / `top_molecules_by_docking.csv`: 末轮候选分子排名表。",
        "- `final_epoch_correlations.csv`: 末轮 reward/docking/classifier 与性质的 Spearman 相关。",
    ]
    (out_dir / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(run_dir: Path) -> Path:
    out_dir = run_dir / "analysis_detailed_properties"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_summary(run_dir)
    props = _load_molecules(run_dir)
    valid_props = props[props["descriptor_valid"] == 1].copy()
    final_epoch = int(summary["epoch"].max())
    final_valid = valid_props[valid_props["epoch"] == final_epoch].copy()

    property_cols = [
        "mol_wt",
        "exact_mol_wt",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rot_bonds",
        "heavy_atom_count",
        "ring_count",
        "aromatic_ring_count",
        "aliphatic_ring_count",
        "saturated_ring_count",
        "fraction_csp3",
        "formal_charge",
        "qed",
        "sa_score",
        "lipinski_pass",
        "carbon_atoms",
        "hetero_atoms",
        "n_atoms",
        "n_n",
        "n_o",
        "n_s",
        "n_halogen",
        "carbon_fraction",
        "hetero_fraction",
        "reward_total",
        "reward_total_raw",
        "docking_raw",
        "docking_reward",
        "chembl_classifier_reward",
        "ligand_efficiency_raw",
    ]

    epoch_props = _numeric_summary(valid_props, property_cols, group_col="epoch")
    final_summary = _numeric_summary(final_valid, property_cols)
    top_reward = final_valid.sort_values(
        ["reward_total", "docking_raw"], ascending=[False, True]
    ).head(50)
    top_docking = final_valid.sort_values(
        ["docking_raw", "reward_total"], ascending=[True, False]
    ).head(50)

    scaffold_summary = (
        final_valid["murcko_scaffold"]
        .fillna("")
        .replace("", "(no scaffold)")
        .value_counts()
        .rename_axis("murcko_scaffold")
        .reset_index(name="count")
    )
    failure_summary = summary[
        [
            c
            for c in summary.columns
            if c
            in {
                "epoch",
                "generated_total",
                "rdkit_valid",
                "rdkit_valid_ratio",
                "failed_total",
                "failed_ratio",
                "docking_backend_failure_ratio",
            }
            or c.startswith("failure_")
        ]
    ]

    _save_table(summary, out_dir / "epoch_summary_key_metrics.csv")
    _save_table(valid_props, out_dir / "all_valid_molecule_properties.csv")
    _save_table(epoch_props, out_dir / "epoch_property_summary.csv")
    _save_table(final_summary, out_dir / "final_epoch_property_summary.csv")
    _save_table(top_reward, out_dir / "top_molecules_by_reward.csv")
    _save_table(top_docking, out_dir / "top_molecules_by_docking.csv")
    _save_table(scaffold_summary, out_dir / "final_epoch_scaffold_counts.csv")
    _save_table(failure_summary, out_dir / "failure_breakdown_by_epoch.csv")

    _plot_training_overview(summary, out_dir)
    _plot_property_trends(epoch_props, out_dir)
    _plot_final_distributions(final_valid, out_dir)
    corr = _plot_correlations(final_valid, out_dir)
    _save_table(corr, out_dir / "final_epoch_correlations.csv")
    top_vs_all = _plot_top_vs_all(final_valid, out_dir)
    _save_table(top_vs_all, out_dir / "final_epoch_top_vs_all_property_summary.csv")
    rank_summary = _plot_rank_and_scaffold(props, out_dir)
    _save_table(rank_summary, out_dir / "final_epoch_rank_summary.csv")
    _plot_failure_and_timing(summary, out_dir)

    _write_report(
        run_dir=run_dir,
        out_dir=out_dir,
        summary=summary,
        props=props,
        final_summary=final_summary,
        top_reward=top_reward,
        top_docking=top_docking,
        rank_summary=rank_summary,
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not (run_dir / "training_summary.csv").exists():
        raise FileNotFoundError(f"training_summary.csv not found under {run_dir}")

    out_dir = run_analysis(run_dir)
    print(f"Saved detailed analysis to: {out_dir}")


if __name__ == "__main__":
    main()
