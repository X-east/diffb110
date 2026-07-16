#!/usr/bin/env python3
"""Analyze the completed 100-epoch DiffGui PB-fast DePPA training run."""

from __future__ import annotations

import json
import logging
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Lipinski, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
RUN_DIR = ROOT / "runs/diffgui-deppa-pbfast-cpu4-lr1e5-wd1e4-b8x7-t200-100ep_20260715_2018"
CHEMBL_PATH = ROOT / "data/chembl/chembl_36_properties.csv.gz"
WORK_RECORD_DIR = ROOT / "work_records/26-7/260716"
WORK_RECORD_PATH = WORK_RECORD_DIR / "260716-1 DiffGui PB-fast 100epoch训练与分子性质分析.md"
MAX_WORKERS = 18

LOGGER = logging.getLogger("pbfast_100ep_analysis")

WINDOWS = [
    ("early_01_10", 1, 10),
    ("middle_46_55", 46, 55),
    ("late_91_100", 91, 100),
]

SUMMARY_METRICS = [
    "reward_for_advantage_mean",
    "reward_total_scored_mean",
    "docking_raw_mean",
    "docking_reward_mean",
    "ligand_efficiency_raw_mean",
    "reconstruction_success_ratio",
    "posebusters_pb_valid_rate_attempts",
    "posebusters_pb_valid_rate_sanitize",
    "docking_success_ratio_total",
    "duplicate_ratio",
    "unique_canonical_smiles",
    "unique_scaffolds",
    "descriptor_mmd_vs_baseline",
    "ecfp4_mean_pairwise_similarity",
    "fcfp4_mean_pairwise_similarity",
    "PPO_Loss",
    "PPO_ApproxKL",
    "ClipFrac",
    "RatioMean",
    "GradNorm",
    "total_seconds",
    "rollout_seconds",
    "posebusters_seconds",
    "reward_seconds",
    "ppo_seconds",
    "logp_mean",
    "tpsa_mean",
    "hba_mean",
    "hbd_mean",
    "carbon_fraction",
    "hetero_atom_mean",
    "ring_count_mean",
    "aromatic_ring_mean",
    "three_member_ring_total",
    "valence_violation_total",
    "fragment_multi_ratio",
]

MOLECULE_NUMERIC_COLUMNS = [
    "epoch",
    "rank",
    "local_mol_index",
    "reconstruction_success",
    "rdkit_valid",
    "sanitize_valid",
    "pb_valid",
    "reward_eligible",
    "reward_found",
    "docking_success",
    "reward_total",
    "reward_for_advantage",
    "deppa_reward",
    "qed_raw",
    "sa_raw",
    "docking_raw",
    "docking_reward",
    "ligand_efficiency_raw",
    "deppa_qed_normal",
    "deppa_sa_normal",
    "deppa_vina_score_normal",
    "deppa_distance_normal",
]

PROPERTY_COLUMNS = [
    "heavy_atoms_calc",
    "mol_logp_calc",
    "tpsa_calc",
    "hba_calc",
    "hbd_calc",
    "rotatable_bonds_calc",
    "carbon_fraction_calc",
    "hetero_atom_count_calc",
    "nitrogen_count_calc",
    "oxygen_count_calc",
    "sulfur_count_calc",
    "halogen_count_calc",
    "polar_hetero_count_calc",
    "element_diversity_calc",
    "fraction_csp3_calc",
    "ring_count_calc",
    "aromatic_ring_count_calc",
    "three_member_ring_count_calc",
    "fused_ring_pair_count_calc",
    "spiro_atom_count_calc",
    "bridgehead_atom_count_calc",
    "fragment_count_calc",
    "complex_ring_flag_calc",
]


def _setup_logging() -> None:
    """Configure durable analysis logging."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = logging.FileHandler(OUT_DIR / "analysis.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    LOGGER.addHandler(stream_handler)


def _to_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Convert selected dataframe columns to numeric values."""
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _window_label(epoch: int) -> str:
    """Return the named comparison window for an epoch."""
    for label, start, end in WINDOWS:
        if start <= epoch <= end:
            return label
    return "other"


def _load_summary() -> pd.DataFrame:
    """Load the authoritative 100-row training summary."""
    path = RUN_DIR / "training_summary.csv"
    summary = pd.read_csv(path)
    summary = _to_numeric(summary, ["epoch", *SUMMARY_METRICS])
    if len(summary) != 100 or summary["epoch"].astype(int).tolist() != list(range(1, 101)):
        raise RuntimeError("training_summary.csv is not a complete contiguous 1-100 epoch run")
    if not summary["status"].eq("ok").all():
        raise RuntimeError("training_summary.csv contains a non-ok epoch")
    summary["window"] = summary["epoch"].astype(int).map(_window_label)
    LOGGER.info("loaded training summary rows=%d", len(summary))
    return summary


def _load_molecules() -> pd.DataFrame:
    """Load all 56 policy records from every completed epoch."""
    frames: list[pd.DataFrame] = []
    for epoch in range(1, 101):
        path = RUN_DIR / "epochs" / f"epoch_{epoch:04d}" / "molecules.csv"
        frame = pd.read_csv(path)
        if len(frame) != 56:
            raise RuntimeError(f"expected 56 molecule rows at epoch {epoch}, found {len(frame)}")
        frames.append(frame)
    molecules = pd.concat(frames, ignore_index=True, sort=False)
    molecules = _to_numeric(molecules, MOLECULE_NUMERIC_COLUMNS)
    molecules["window"] = molecules["epoch"].astype(int).map(_window_label)
    LOGGER.info("loaded molecule rows=%d", len(molecules))
    return molecules


def _descriptor_from_smiles(smiles: object) -> dict[str, float]:
    """Calculate monitoring descriptors from one canonical SMILES."""
    empty = {column: math.nan for column in PROPERTY_COLUMNS}
    if not isinstance(smiles, str) or not smiles:
        return empty
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return empty
    atoms = list(mol.GetAtoms())
    heavy_atoms = mol.GetNumHeavyAtoms()
    carbon_atoms = sum(atom.GetAtomicNum() == 6 for atom in atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in {1, 6} for atom in atoms)
    nitrogen_atoms = sum(atom.GetAtomicNum() == 7 for atom in atoms)
    oxygen_atoms = sum(atom.GetAtomicNum() == 8 for atom in atoms)
    sulfur_atoms = sum(atom.GetAtomicNum() == 16 for atom in atoms)
    halogen_atoms = sum(atom.GetAtomicNum() in {9, 17, 35, 53} for atom in atoms)
    element_diversity = len({atom.GetAtomicNum() for atom in atoms if atom.GetAtomicNum() != 1})
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    fused_pairs = sum(len(left & right) >= 2 for i, left in enumerate(rings) for right in rings[i + 1 :])
    three_member_rings = sum(len(ring) == 3 for ring in rings)
    spiro_atoms = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridgehead_atoms = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return {
        "heavy_atoms_calc": float(heavy_atoms),
        "mol_logp_calc": float(Crippen.MolLogP(mol)),
        "tpsa_calc": float(rdMolDescriptors.CalcTPSA(mol)),
        "hba_calc": float(rdMolDescriptors.CalcNumHBA(mol)),
        "hbd_calc": float(rdMolDescriptors.CalcNumHBD(mol)),
        "rotatable_bonds_calc": float(Lipinski.NumRotatableBonds(mol)),
        "carbon_fraction_calc": float(carbon_atoms / heavy_atoms) if heavy_atoms else math.nan,
        "hetero_atom_count_calc": float(hetero_atoms),
        "nitrogen_count_calc": float(nitrogen_atoms),
        "oxygen_count_calc": float(oxygen_atoms),
        "sulfur_count_calc": float(sulfur_atoms),
        "halogen_count_calc": float(halogen_atoms),
        "polar_hetero_count_calc": float(nitrogen_atoms + oxygen_atoms + sulfur_atoms),
        "element_diversity_calc": float(element_diversity),
        "fraction_csp3_calc": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "ring_count_calc": float(len(rings)),
        "aromatic_ring_count_calc": float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "three_member_ring_count_calc": float(three_member_rings),
        "fused_ring_pair_count_calc": float(fused_pairs),
        "spiro_atom_count_calc": float(spiro_atoms),
        "bridgehead_atom_count_calc": float(bridgehead_atoms),
        "fragment_count_calc": float(len(Chem.GetMolFrags(mol))),
        "complex_ring_flag_calc": float(bool(fused_pairs or spiro_atoms or bridgehead_atoms)),
    }


def _add_descriptors(molecules: pd.DataFrame) -> pd.DataFrame:
    """Add RDKit descriptors for every RDKit-valid molecule using 18 workers."""
    output = molecules.copy()
    smiles_values = output["canonical_smiles"].where(output["rdkit_valid"].eq(1), None).tolist()
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        descriptors = list(executor.map(_descriptor_from_smiles, smiles_values, chunksize=64))
    descriptor_df = pd.DataFrame(descriptors)
    for column in PROPERTY_COLUMNS:
        output[column] = descriptor_df[column]
    LOGGER.info(
        "computed descriptors parsed=%d invalid_or_missing=%d",
        int(output["mol_logp_calc"].notna().sum()),
        int(output["mol_logp_calc"].isna().sum()),
    )
    return output


def _epoch_molecule_summary(molecules: pd.DataFrame) -> pd.DataFrame:
    """Aggregate molecule-level reward components and descriptors by epoch."""
    parsed = molecules[molecules["rdkit_valid"].eq(1) & molecules["mol_logp_calc"].notna()].copy()
    aggregations: dict[str, tuple[str, str]] = {
        "qed_raw_mean_calc": ("qed_raw", "mean"),
        "sa_raw_mean_calc": ("sa_raw", "mean"),
        "element_diversity_mean": ("element_diversity_calc", "mean"),
        "nitrogen_count_calc": ("nitrogen_count_calc", "mean"),
        "oxygen_count_calc": ("oxygen_count_calc", "mean"),
        "sulfur_count_calc": ("sulfur_count_calc", "mean"),
        "halogen_count_calc": ("halogen_count_calc", "mean"),
        "polar_hetero_count_calc": ("polar_hetero_count_calc", "mean"),
        "fraction_csp3_mean": ("fraction_csp3_calc", "mean"),
        "fused_ring_pair_mean": ("fused_ring_pair_count_calc", "mean"),
        "spiro_atom_mean": ("spiro_atom_count_calc", "mean"),
        "bridgehead_atom_mean": ("bridgehead_atom_count_calc", "mean"),
        "complex_ring_ratio": ("complex_ring_flag_calc", "mean"),
        "three_member_ring_ratio": ("three_member_ring_count_calc", lambda x: float((x > 0).mean())),
    }
    for component in [
        "deppa_qed_normal",
        "deppa_sa_normal",
        "deppa_vina_score_normal",
        "deppa_distance_normal",
    ]:
        if component in parsed.columns:
            aggregations[f"{component}_mean"] = (component, "mean")
    summary = parsed.groupby("epoch", as_index=False).agg(**aggregations)
    return summary


def _window_summary(summary: pd.DataFrame, molecules: pd.DataFrame) -> pd.DataFrame:
    """Build early, middle, and late 10-epoch summaries."""
    rows: list[dict[str, Any]] = []
    parsed = molecules[molecules["rdkit_valid"].eq(1) & molecules["mol_logp_calc"].notna()].copy()
    for label, start, end in WINDOWS:
        epoch_part = summary[summary["epoch"].between(start, end)]
        molecule_part = parsed[parsed["epoch"].between(start, end)]
        row: dict[str, Any] = {
            "window": label,
            "start_epoch": start,
            "end_epoch": end,
            "n_epochs": len(epoch_part),
            "n_policy_molecules": int(len(molecules[molecules["epoch"].between(start, end)])),
            "n_rdkit_valid_molecules": int(len(molecule_part)),
        }
        for metric in SUMMARY_METRICS:
            if metric in epoch_part.columns:
                row[metric] = float(epoch_part[metric].mean())
        for metric in PROPERTY_COLUMNS:
            row[metric] = float(molecule_part[metric].mean())
        for metric in ["qed_raw", "sa_raw", "docking_raw", "reward_total"]:
            if metric in molecule_part.columns:
                row[f"molecule_{metric}_mean"] = float(molecule_part[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _trend_summary(summary: pd.DataFrame, window_summary: pd.DataFrame) -> pd.DataFrame:
    """Quantify early-to-late changes and whole-run linear slopes."""
    early = window_summary.iloc[0]
    late = window_summary.iloc[-1]
    metrics = [
        "reward_for_advantage_mean",
        "reward_total_scored_mean",
        "docking_raw_mean",
        "reconstruction_success_ratio",
        "posebusters_pb_valid_rate_attempts",
        "docking_success_ratio_total",
        "PPO_ApproxKL",
        "ClipFrac",
        "RatioMean",
        "descriptor_mmd_vs_baseline",
        "mol_logp_calc",
        "tpsa_calc",
        "carbon_fraction_calc",
        "hetero_atom_count_calc",
        "nitrogen_count_calc",
        "oxygen_count_calc",
        "halogen_count_calc",
        "polar_hetero_count_calc",
        "ring_count_calc",
        "aromatic_ring_count_calc",
        "three_member_ring_count_calc",
        "fused_ring_pair_count_calc",
        "element_diversity_calc",
    ]
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        if metric not in early.index or metric not in late.index:
            continue
        early_value = float(early[metric])
        late_value = float(late[metric])
        source_metric = metric if metric in summary.columns else None
        slope = math.nan
        if source_metric:
            valid = summary[["epoch", source_metric]].dropna()
            if len(valid) > 1:
                slope = float(np.polyfit(valid["epoch"], valid[source_metric], 1)[0])
        rows.append(
            {
                "metric": metric,
                "early_mean": early_value,
                "late_mean": late_value,
                "absolute_change": late_value - early_value,
                "relative_change_percent": (
                    (late_value - early_value) / abs(early_value) * 100.0
                    if early_value != 0 and math.isfinite(early_value)
                    else math.nan
                ),
                "slope_per_epoch": slope,
            }
        )
    return pd.DataFrame(rows)


def _load_chembl_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load matching ChEMBL descriptors and return summary plus a fixed sample."""
    columns = [
        "heavy_atoms",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rotatable_bonds",
        "num_rings",
        "num_aromatic_rings",
        "fraction_csp3",
        "num_spiro_atoms",
        "num_bridgehead_atoms",
        "num_heteroatoms",
        "carbon_fraction",
    ]
    chembl = pd.read_csv(CHEMBL_PATH, usecols=columns)
    chembl = chembl.replace([np.inf, -np.inf], np.nan).dropna()
    summary_rows = []
    for column in columns:
        series = chembl[column]
        summary_rows.append(
            {
                "metric": column,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "q05": float(series.quantile(0.05)),
                "q25": float(series.quantile(0.25)),
                "median": float(series.median()),
                "q75": float(series.quantile(0.75)),
                "q95": float(series.quantile(0.95)),
            }
        )
    sample = chembl.sample(n=min(5000, len(chembl)), random_state=1)
    LOGGER.info("loaded ChEMBL rows=%d sample=%d", len(chembl), len(sample))
    return pd.DataFrame(summary_rows), sample


def _top_reward_decile(molecules: pd.DataFrame) -> pd.DataFrame:
    """Return late-window PB-valid molecules in the top raw-reward decile."""
    late = molecules[
        molecules["epoch"].between(91, 100)
        & molecules["pb_valid"].eq(1)
        & molecules["reward_found"].eq(1)
        & molecules["mol_logp_calc"].notna()
    ].copy()
    threshold = float(late["reward_total"].quantile(0.90))
    top = late[late["reward_total"] >= threshold].sort_values("reward_total", ascending=False)
    top.attrs["threshold"] = threshold
    return top


def _rolling(ax: plt.Axes, x: pd.Series, y: pd.Series, label: str, color: str) -> None:
    """Plot raw values and a 7-epoch centered rolling mean."""
    values = pd.to_numeric(y, errors="coerce")
    ax.plot(x, values, color=color, alpha=0.25, lw=0.8)
    ax.plot(x, values.rolling(7, center=True, min_periods=1).mean(), color=color, lw=2.0, label=label)


def _save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    """Save one figure as PNG and SVG."""
    png = OUT_DIR / f"{stem}.png"
    svg = OUT_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def _plot_training_reward(summary: pd.DataFrame, epoch_molecules: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = summary["epoch"]
    _rolling(axes[0, 0], x, summary["reward_for_advantage_mean"], "PPO reward incl. penalties", "#1f77b4")
    _rolling(axes[0, 0], x, summary["reward_total_scored_mean"], "PB-valid raw reward", "#ff7f0e")
    axes[0, 0].set_ylabel("Reward")
    axes[0, 0].legend()

    _rolling(axes[0, 1], x, summary["docking_raw_mean"], "Docking raw (lower better)", "#2ca02c")
    axes[0, 1].set_ylabel("UniDock score")
    axes[0, 1].legend()

    merged = summary[["epoch"]].merge(epoch_molecules, on="epoch", how="left")
    _rolling(axes[1, 0], x, merged["qed_raw_mean_calc"], "QED", "#9467bd")
    ax_sa = axes[1, 0].twinx()
    _rolling(ax_sa, x, merged["sa_raw_mean_calc"], "SA raw", "#8c564b")
    axes[1, 0].set_ylabel("QED")
    ax_sa.set_ylabel("SA raw")
    axes[1, 0].legend(loc="upper left")
    ax_sa.legend(loc="upper right")

    for column, label, color in [
        ("deppa_qed_normal_mean", "QED norm", "#1f77b4"),
        ("deppa_sa_normal_mean", "SA norm", "#ff7f0e"),
        ("deppa_vina_score_normal_mean", "Vina norm", "#2ca02c"),
        ("deppa_distance_normal_mean", "Diversity norm", "#d62728"),
    ]:
        if column in merged.columns:
            _rolling(axes[1, 1], x, merged[column], label, color)
    axes[1, 1].set_ylabel("Normalized component")
    axes[1, 1].legend(ncol=2)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("Training reward and docking trends")
    fig.tight_layout()
    return _save_figure(fig, "01_training_reward_docking")


def _plot_validity_diversity(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = summary["epoch"]
    for column, label, color in [
        ("reconstruction_success_ratio", "Reconstruction", "#1f77b4"),
        ("posebusters_pb_valid_rate_attempts", "PB-valid / attempts", "#ff7f0e"),
        ("docking_success_ratio_total", "Docking / attempts", "#2ca02c"),
    ]:
        _rolling(axes[0, 0], x, summary[column], label, color)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_ylabel("Rate")
    axes[0, 0].legend()

    for column, label, color in [
        ("reconstruction_success", "Reconstructed", "#1f77b4"),
        ("posebusters_pb_valid", "PB-valid", "#ff7f0e"),
        ("docking_success", "Docking success", "#2ca02c"),
    ]:
        _rolling(axes[0, 1], x, summary[column], label, color)
    axes[0, 1].set_ylabel("Count / 56")
    axes[0, 1].legend()

    for column, label, color in [
        ("duplicate_ratio", "Duplicate ratio", "#d62728"),
        ("ecfp4_mean_pairwise_similarity", "ECFP4 similarity", "#9467bd"),
        ("fcfp4_mean_pairwise_similarity", "FCFP4 similarity", "#8c564b"),
    ]:
        _rolling(axes[1, 0], x, summary[column], label, color)
    axes[1, 0].set_ylabel("Diversity diagnostic")
    axes[1, 0].legend()

    _rolling(axes[1, 1], x, summary["descriptor_mmd_vs_baseline"], "Descriptor MMD vs ChEMBL", "#e377c2")
    ax_scaffold = axes[1, 1].twinx()
    _rolling(ax_scaffold, x, summary["unique_scaffolds"], "Unique scaffolds", "#17becf")
    axes[1, 1].set_ylabel("MMD")
    ax_scaffold.set_ylabel("Scaffold count")
    axes[1, 1].legend(loc="upper left")
    ax_scaffold.legend(loc="upper right")

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("Validity, success, and diversity")
    fig.tight_layout()
    return _save_figure(fig, "02_validity_diversity")


def _plot_ppo_runtime(summary: pd.DataFrame) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    x = summary["epoch"]
    _rolling(axes[0, 0], x, summary["PPO_ApproxKL"], "ApproxKL", "#d62728")
    axes[0, 0].set_yscale("symlog", linthresh=1.0)
    axes[0, 0].set_ylabel("ApproxKL (symlog)")
    axes[0, 0].legend()

    _rolling(axes[0, 1], x, summary["ClipFrac"], "ClipFrac", "#1f77b4")
    _rolling(axes[0, 1], x, summary["RatioMean"], "RatioMean", "#ff7f0e")
    axes[0, 1].axhline(1.0, color="#888888", lw=0.8)
    axes[0, 1].set_ylabel("PPO ratio")
    axes[0, 1].legend()

    _rolling(axes[1, 0], x, summary["PPO_Loss"], "PPO loss", "#2ca02c")
    ax_grad = axes[1, 0].twinx()
    _rolling(ax_grad, x, summary["GradNorm"], "GradNorm", "#9467bd")
    axes[1, 0].set_ylabel("Loss")
    ax_grad.set_ylabel("GradNorm")
    axes[1, 0].legend(loc="upper left")
    ax_grad.legend(loc="upper right")

    for column, label, color in [
        ("rollout_seconds", "Rollout", "#1f77b4"),
        ("posebusters_seconds", "PB-fast", "#ff7f0e"),
        ("reward_seconds", "Reward", "#2ca02c"),
        ("ppo_seconds", "PPO", "#d62728"),
    ]:
        _rolling(axes[1, 1], x, summary[column], label, color)
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].legend(ncol=2)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("PPO drift and runtime")
    fig.tight_layout()
    return _save_figure(fig, "03_ppo_stability_runtime")


def _plot_property_trends(summary: pd.DataFrame, epoch_molecules: pd.DataFrame) -> tuple[Path, Path]:
    merged = summary.merge(epoch_molecules, on="epoch", how="left")
    panels = [
        ("logp_mean", "LogP", "#1f77b4"),
        ("tpsa_mean", "TPSA", "#ff7f0e"),
        ("carbon_fraction", "Carbon fraction", "#2ca02c"),
        ("hetero_atom_mean", "Hetero atoms", "#d62728"),
        ("element_diversity_mean", "Element diversity", "#9467bd"),
        ("fraction_csp3_mean", "Fraction Csp3", "#8c564b"),
        ("ring_count_mean", "Ring count", "#e377c2"),
        ("aromatic_ring_mean", "Aromatic rings", "#7f7f7f"),
        ("complex_ring_ratio", "Complex-ring fraction", "#17becf"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    for ax, (column, label, color) in zip(axes.flat, panels):
        _rolling(ax, merged["epoch"], merged[column], label, color)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("Molecular property drift (RDKit-valid molecules)")
    fig.tight_layout()
    return _save_figure(fig, "04_molecular_property_trends")


def _boxplot_groups(
    axes: np.ndarray,
    groups: list[tuple[str, pd.DataFrame]],
    metrics: list[tuple[str, str]],
) -> None:
    """Draw consistent group boxplots on a grid."""
    colors = ["#4c78a8", "#f2cf5b", "#e45756", "#72b7b2"]
    for ax, (column, label) in zip(axes.flat, metrics):
        values = [group[column].dropna().to_numpy() for _, group in groups]
        box = ax.boxplot(values, labels=[name for name, _ in groups], showfliers=False, patch_artist=True)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)


def _plot_property_distributions(molecules: pd.DataFrame) -> tuple[Path, Path]:
    parsed = molecules[molecules["rdkit_valid"].eq(1) & molecules["mol_logp_calc"].notna()]
    groups = [(label.replace("_", " "), parsed[parsed["epoch"].between(start, end)]) for label, start, end in WINDOWS]
    metrics = [
        ("mol_logp_calc", "LogP"),
        ("tpsa_calc", "TPSA"),
        ("hba_calc", "HBA"),
        ("hbd_calc", "HBD"),
        ("carbon_fraction_calc", "Carbon fraction"),
        ("hetero_atom_count_calc", "Hetero atoms"),
        ("ring_count_calc", "Ring count"),
        ("aromatic_ring_count_calc", "Aromatic rings"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    _boxplot_groups(axes, groups, metrics)
    fig.suptitle("Early, middle, and late molecular property distributions")
    fig.tight_layout()
    return _save_figure(fig, "05_property_distributions_early_middle_late")


def _plot_chembl_top_decile(
    molecules: pd.DataFrame,
    top_decile: pd.DataFrame,
    chembl_sample: pd.DataFrame,
) -> tuple[Path, Path]:
    parsed = molecules[molecules["rdkit_valid"].eq(1) & molecules["mol_logp_calc"].notna()]
    early = parsed[parsed["epoch"].between(1, 10)]
    late = parsed[parsed["epoch"].between(91, 100)]
    chembl = chembl_sample.rename(
        columns={
            "mol_log_p": "mol_logp_calc",
            "tpsa": "tpsa_calc",
            "num_h_acceptors": "hba_calc",
            "num_rings": "ring_count_calc",
            "num_aromatic_rings": "aromatic_ring_count_calc",
            "carbon_fraction": "carbon_fraction_calc",
        }
    )
    groups = [
        ("Early", early),
        ("Late", late),
        ("Late top 10%", top_decile),
        ("ChEMBL", chembl),
    ]
    metrics = [
        ("mol_logp_calc", "LogP"),
        ("tpsa_calc", "TPSA"),
        ("hba_calc", "HBA"),
        ("ring_count_calc", "Ring count"),
        ("aromatic_ring_count_calc", "Aromatic rings"),
        ("carbon_fraction_calc", "Carbon fraction"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    _boxplot_groups(axes, groups, metrics)
    fig.suptitle("Generated molecules vs late top-reward decile and ChEMBL")
    fig.tight_layout()
    return _save_figure(fig, "06_chembl_topdecile_comparison")


def _plot_structural_risks(summary: pd.DataFrame, epoch_molecules: pd.DataFrame) -> tuple[Path, Path]:
    merged = summary.merge(epoch_molecules, on="epoch", how="left")
    x = merged["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    _rolling(axes[0, 0], x, merged["three_member_ring_ratio"], "3-member-ring fraction", "#d62728")
    _rolling(axes[0, 0], x, merged["complex_ring_ratio"], "Complex-ring fraction", "#9467bd")
    axes[0, 0].set_ylabel("Fraction")
    axes[0, 0].legend()

    for column, label, color in [
        ("fused_ring_pair_mean", "Fused ring pairs", "#1f77b4"),
        ("spiro_atom_mean", "Spiro atoms", "#ff7f0e"),
        ("bridgehead_atom_mean", "Bridgehead atoms", "#2ca02c"),
    ]:
        _rolling(axes[0, 1], x, merged[column], label, color)
    axes[0, 1].set_ylabel("Mean count")
    axes[0, 1].legend()

    _rolling(axes[1, 0], x, merged["fragment_multi_ratio"], "Multi-fragment ratio", "#8c564b")
    ax_valence = axes[1, 0].twinx()
    _rolling(ax_valence, x, merged["valence_violation_total"], "Valence violations", "#e377c2")
    axes[1, 0].set_ylabel("Fragment ratio")
    ax_valence.set_ylabel("Violation count")
    axes[1, 0].legend(loc="upper left")
    ax_valence.legend(loc="upper right")

    for column, label, color in [
        ("fraction_logp_high", "High LogP", "#d62728"),
        ("fraction_tpsa_low", "Low TPSA", "#ff7f0e"),
        ("fraction_hba_zero_or_low", "Low HBA", "#1f77b4"),
    ]:
        _rolling(axes[1, 1], x, merged[column], label, color)
    axes[1, 1].set_ylabel("Risk fraction")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("Structural and descriptor risk signals")
    fig.tight_layout()
    return _save_figure(fig, "07_structural_risk_trends")


def _plot_element_composition(summary: pd.DataFrame, epoch_molecules: pd.DataFrame) -> tuple[Path, Path]:
    merged = summary.merge(epoch_molecules, on="epoch", how="left")
    x = merged["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for column, label, color in [
        ("nitrogen_count_calc", "Nitrogen", "#1f77b4"),
        ("oxygen_count_calc", "Oxygen", "#d62728"),
        ("sulfur_count_calc", "Sulfur", "#ff7f0e"),
        ("halogen_count_calc", "Halogens", "#2ca02c"),
    ]:
        _rolling(axes[0, 0], x, merged[column], label, color)
    axes[0, 0].set_ylabel("Mean atom count")
    axes[0, 0].legend(ncol=2)

    _rolling(axes[0, 1], x, merged["hetero_atom_mean"], "All hetero atoms", "#9467bd")
    _rolling(axes[0, 1], x, merged["polar_hetero_count_calc"], "N + O + S", "#17becf")
    axes[0, 1].set_ylabel("Mean atom count")
    axes[0, 1].legend()

    _rolling(axes[1, 0], x, merged["tpsa_mean"], "TPSA", "#ff7f0e")
    ax_hba = axes[1, 0].twinx()
    _rolling(ax_hba, x, merged["hba_mean"], "HBA", "#1f77b4")
    axes[1, 0].set_ylabel("TPSA")
    ax_hba.set_ylabel("HBA")
    axes[1, 0].legend(loc="upper left")
    ax_hba.legend(loc="upper right")

    _rolling(axes[1, 1], x, merged["carbon_fraction"], "Carbon fraction", "#7f7f7f")
    _rolling(axes[1, 1], x, merged["fraction_csp3_mean"], "Fraction Csp3", "#8c564b")
    axes[1, 1].set_ylabel("Fraction")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_xlabel("Epoch")
    fig.suptitle("Element composition and polarity drift")
    fig.tight_layout()
    return _save_figure(fig, "09_element_composition_trends")


def _plot_window_summary(window_summary: pd.DataFrame) -> tuple[Path, Path]:
    metrics = [
        ("reward_for_advantage_mean", "PPO reward"),
        ("reward_total_scored_mean", "PB-valid raw reward"),
        ("docking_raw_mean", "Docking raw"),
        ("posebusters_pb_valid_rate_attempts", "PB-valid rate"),
        ("PPO_ApproxKL", "ApproxKL"),
        ("mol_logp_calc", "LogP"),
        ("tpsa_calc", "TPSA"),
        ("ring_count_calc", "Ring count"),
    ]
    labels = ["Early", "Middle", "Late"]
    colors = ["#4c78a8", "#f2cf5b", "#e45756"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (column, title) in zip(axes.flat, metrics):
        values = window_summary[column].to_numpy(dtype=float)
        ax.bar(labels, values, color=colors)
        if column == "PPO_ApproxKL":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Early-middle-late window comparison")
    fig.tight_layout()
    return _save_figure(fig, "08_window_summary")


def _copy_figures_to_work_record(figures: list[Path]) -> None:
    """Copy PNG figures next to the work record for self-contained rendering."""
    WORK_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    for figure in figures:
        if figure.suffix.lower() == ".png":
            shutil.copy2(figure, WORK_RECORD_DIR / figure.name)


def main() -> None:
    """Run the full completed-run analysis."""
    _setup_logging()
    summary = _load_summary()
    molecules = _add_descriptors(_load_molecules())
    epoch_molecules = _epoch_molecule_summary(molecules)
    window_summary = _window_summary(summary, molecules)
    trend_summary = _trend_summary(summary, window_summary)
    chembl_summary, chembl_sample = _load_chembl_reference()
    top_decile = _top_reward_decile(molecules)

    selected_summary_columns = [
        "epoch",
        "status",
        *[column for column in SUMMARY_METRICS if column in summary.columns],
        "checkpoint",
    ]
    summary[selected_summary_columns].to_csv(OUT_DIR / "epoch_metrics_selected.csv", index=False)
    molecule_columns = [
        "epoch",
        "rank",
        "local_mol_index",
        "global_mol_id",
        "canonical_smiles",
        "murcko_scaffold",
        "reconstruction_success",
        "rdkit_valid",
        "pb_valid",
        "reward_found",
        "docking_success",
        "reward_total",
        "reward_for_advantage",
        "deppa_reward",
        "qed_raw",
        "sa_raw",
        "docking_raw",
        "docking_reward",
        "ligand_efficiency_raw",
        *PROPERTY_COLUMNS,
    ]
    molecules[[column for column in molecule_columns if column in molecules.columns]].to_csv(
        OUT_DIR / "molecule_properties_all.csv", index=False
    )
    epoch_molecules.to_csv(OUT_DIR / "molecule_epoch_summary.csv", index=False)
    window_summary.to_csv(OUT_DIR / "window_summary.csv", index=False)
    trend_summary.to_csv(OUT_DIR / "early_late_change_summary.csv", index=False)
    chembl_summary.to_csv(OUT_DIR / "chembl_reference_summary.csv", index=False)
    top_decile.to_csv(OUT_DIR / "late_top_reward_decile.csv", index=False)

    figure_pairs = [
        _plot_training_reward(summary, epoch_molecules),
        _plot_validity_diversity(summary),
        _plot_ppo_runtime(summary),
        _plot_property_trends(summary, epoch_molecules),
        _plot_property_distributions(molecules),
        _plot_chembl_top_decile(molecules, top_decile, chembl_sample),
        _plot_structural_risks(summary, epoch_molecules),
        _plot_element_composition(summary, epoch_molecules),
        _plot_window_summary(window_summary),
    ]
    figures = [path for pair in figure_pairs for path in pair]
    _copy_figures_to_work_record(figures)

    checkpoint_manifest = pd.read_csv(RUN_DIR / "checkpoints/checkpoint_manifest.csv")
    analysis_summary = {
        "run_dir": str(RUN_DIR),
        "run_status": json.loads((RUN_DIR / "run_manifest.json").read_text(encoding="utf-8"))["status"],
        "epochs": int(len(summary)),
        "policy_molecule_records": int(len(molecules)),
        "rdkit_valid_records": int(molecules["rdkit_valid"].eq(1).sum()),
        "pb_valid_records": int(molecules["pb_valid"].eq(1).sum()),
        "docking_success_records": int(molecules["docking_success"].eq(1).sum()),
        "checkpoints": int(len(checkpoint_manifest)),
        "top_reward_decile_threshold": float(top_decile.attrs["threshold"]),
        "late_top_reward_decile_count": int(len(top_decile)),
        "output_dir": str(OUT_DIR),
        "figures": [str(path) for path in figures],
        "tables": [
            "epoch_metrics_selected.csv",
            "molecule_properties_all.csv",
            "molecule_epoch_summary.csv",
            "window_summary.csv",
            "early_late_change_summary.csv",
            "chembl_reference_summary.csv",
            "late_top_reward_decile.csv",
        ],
    }
    (OUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(analysis_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("analysis complete output=%s", OUT_DIR)


if __name__ == "__main__":
    main()
