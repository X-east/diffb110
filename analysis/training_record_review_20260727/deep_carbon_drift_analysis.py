"""Recompute molecule-level carbon and polarity drift across major RL runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Draw, Lipinski, QED, rdMolDescriptors


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent

DEPA_ROOT = ROOT / "OUT/deppa-8gjl-official-4gpu-20260720/output/8GJL/rollouts"
PBFAST_CSV = ROOT / "analysis/diffgui_rl/pbfast_100ep_20260715_training_analysis/molecule_properties_all.csv"
T200_CSV = ROOT / "analysis/diffgui_rl/deppa_full_20260626_training_analysis/per_molecule_complexity.csv"
DS_RUN = ROOT / "runs/diffsbdd/7gpu-dock-classifier-v0-1_20260529_133609"
DG20_RUN = ROOT / "runs/diffgui/diffgui-next6-20260618-04-ring_last3_pathology_guard_20260619_112410"
DG50_RUN = ROOT / "runs/diffgui/diffgui-next6-20260618-04-ring_last3_pathology_guard-continue-to50_20260621_225705"
CHEMBL_CSV = ROOT / "data/chembl/chembl_36_properties.csv.gz"

HALOGENS = {9, 17, 35, 53}
POLAR_HETERO = {7, 8, 16}
METHOD_ORDER = [
    "DiffSBDD dock+classifier",
    "DiffGui Next6-04",
    "DiffGui DePPA t200",
    "DiffGui PB-fast",
    "DePPA original",
]


def _configure_plotting() -> None:
    """Apply a compact publication-oriented plotting theme."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _fused_ring_pair_count(mol: Chem.Mol) -> int:
    """Count ring pairs sharing at least two atoms."""
    rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    return sum(len(rings[i] & rings[j]) >= 2 for i in range(len(rings)) for j in range(i + 1, len(rings)))


def _molecule_metrics(mol: Chem.Mol) -> dict[str, float | int | str]:
    """Calculate one consistent descriptor panel for a sanitized RDKit molecule."""
    heavy_atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    atomic_numbers = [atom.GetAtomicNum() for atom in heavy_atoms]
    heavy_count = len(atomic_numbers)
    carbon_count = sum(number == 6 for number in atomic_numbers)
    hetero_count = sum(number != 6 for number in atomic_numbers)
    nitrogen_count = sum(number == 7 for number in atomic_numbers)
    oxygen_count = sum(number == 8 for number in atomic_numbers)
    sulfur_count = sum(number == 16 for number in atomic_numbers)
    polar_hetero_count = sum(number in POLAR_HETERO for number in atomic_numbers)
    halogen_count = sum(number in HALOGENS for number in atomic_numbers)
    ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
    fragments = Chem.GetMolFrags(mol)
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "heavy_atom_count": heavy_count,
        "carbon_count": carbon_count,
        "hetero_atom_count": hetero_count,
        "nitrogen_count": nitrogen_count,
        "oxygen_count": oxygen_count,
        "sulfur_count": sulfur_count,
        "polar_hetero_count": polar_hetero_count,
        "halogen_count": halogen_count,
        "element_diversity": len(set(atomic_numbers)),
        "all_carbon": int(heavy_count > 0 and hetero_count == 0),
        "no_nos": int(polar_hetero_count == 0),
        "halogen_only_hetero": int(hetero_count > 0 and hetero_count == halogen_count),
        "carbon_fraction": carbon_count / heavy_count if heavy_count else np.nan,
        "mol_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "qed_calc": QED.qed(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "ring_count": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_ring_count": rdMolDescriptors.CalcNumAromaticRings(mol),
        "three_member_ring_count": sum(size == 3 for size in ring_sizes),
        "large_ring_count": sum(size >= 7 for size in ring_sizes),
        "fused_ring_pair_count": _fused_ring_pair_count(mol),
        "spiro_atom_count": rdMolDescriptors.CalcNumSpiroAtoms(mol),
        "bridgehead_atom_count": rdMolDescriptors.CalcNumBridgeheadAtoms(mol),
        "fragment_count": len(fragments),
        "formal_charge_abs": abs(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
    }


def _read_sdf_csv_pair(
    sdf_path: Path,
    csv_path: Path,
    method: str,
    epoch: int,
    cohort: str,
) -> pd.DataFrame:
    """Read aligned SDF and per-molecule CSV records without dropping row positions."""
    source = pd.read_csv(csv_path)
    supplied = list(Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True))
    aligned_source = source
    if len(supplied) != len(source):
        for flag_column in ("rdkit_valid", "reconstruction_success", "reward_found"):
            if flag_column not in source:
                continue
            candidate = source[source[flag_column] == 1].copy()
            if len(supplied) == len(candidate):
                aligned_source = candidate
                break
    if len(supplied) != len(aligned_source):
        raise ValueError(
            f"SDF/CSV row mismatch after validity alignment: {sdf_path} ({len(supplied)}) "
            f"vs {csv_path} ({len(source)} total, {len(aligned_source)} aligned)"
        )
    records: list[dict[str, Any]] = []
    for row_index, (mol, source_row) in enumerate(zip(supplied, aligned_source.to_dict("records"))):
        if mol is None:
            continue
        record: dict[str, Any] = {
            "method": method,
            "epoch": epoch,
            "cohort": cohort,
            "source_file": str(sdf_path.relative_to(ROOT)),
            "source_row": row_index,
        }
        record.update(_molecule_metrics(mol))
        for column in (
            "reward_total",
            "reward_for_advantage",
            "deppa_reward",
            "qed",
            "qed_raw",
            "sa",
            "sa_raw",
            "vina_score",
            "docking_raw",
            "rdkit_valid",
            "pb_valid",
            "docking_success",
            "connectivity",
        ):
            if column in source_row:
                record[column] = source_row[column]
        records.append(record)
    return pd.DataFrame.from_records(records)


def _load_deppa_original() -> pd.DataFrame:
    """Load all original DePPA rollout SDFs and aligned score rows."""
    frames: list[pd.DataFrame] = []
    for epoch_dir in sorted(DEPA_ROOT.glob("epoch_*")):
        epoch = int(epoch_dir.name.split("_")[1])
        cohort = "early" if epoch <= 10 else "late" if epoch >= 91 else "middle"
        for sdf_path in sorted(epoch_dir.glob("rank_*.sdf")):
            frames.append(
                _read_sdf_csv_pair(
                    sdf_path,
                    sdf_path.with_suffix(".csv"),
                    "DePPA original",
                    epoch,
                    cohort,
                )
            )
    result = pd.concat(frames, ignore_index=True)
    LOGGER.info("Loaded original DePPA molecules: n=%d", len(result))
    return result


def _load_endpoint_pair(run: Path, epochs: list[tuple[int, str]], method: str) -> pd.DataFrame:
    """Load selected early/late epoch endpoints from a project run."""
    frames = []
    for epoch, cohort in epochs:
        epoch_dir = run / "epochs" / f"epoch_{epoch:04d}"
        frames.append(_read_sdf_csv_pair(epoch_dir / "molecules.sdf", epoch_dir / "molecules.csv", method, epoch, cohort))
    return pd.concat(frames, ignore_index=True)


def _load_diffgui_t200() -> pd.DataFrame:
    """Recalculate descriptors from the formal DiffGui t200 molecule table."""
    source = pd.read_csv(T200_CSV)
    source = source[(source["epoch"] <= 89) & (source["rdkit_valid"] == 1) & source["canonical_smiles"].notna()].copy()
    records = []
    for row in source.to_dict("records"):
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            continue
        record: dict[str, Any] = {
            "method": "DiffGui DePPA t200",
            "epoch": int(row["epoch"]),
            "cohort": "early" if row["epoch"] <= 10 else "late" if row["epoch"] >= 80 else "middle",
            "source_file": str(T200_CSV.relative_to(ROOT)),
            "source_row": row.get("global_mol_id", ""),
        }
        record.update(_molecule_metrics(mol))
        for column in ("reward_total", "reward_for_advantage", "deppa_reward", "qed_raw", "sa_raw", "docking_raw"):
            record[column] = row.get(column, np.nan)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    LOGGER.info("Loaded DiffGui t200 molecules: n=%d", len(result))
    return result


def _load_pbfast() -> pd.DataFrame:
    """Recalculate descriptors from the complete PB-fast molecule table."""
    source = pd.read_csv(PBFAST_CSV)
    source = source[(source["rdkit_valid"] == 1) & source["canonical_smiles"].notna()].copy()
    records = []
    for row in source.to_dict("records"):
        mol = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if mol is None:
            continue
        epoch = int(row["epoch"])
        record: dict[str, Any] = {
            "method": "DiffGui PB-fast",
            "epoch": epoch,
            "cohort": "early" if epoch <= 10 else "late" if epoch >= 91 else "middle",
            "source_file": str(PBFAST_CSV.relative_to(ROOT)),
            "source_row": row.get("global_mol_id", ""),
        }
        record.update(_molecule_metrics(mol))
        for column in (
            "reward_total",
            "reward_for_advantage",
            "deppa_reward",
            "qed_raw",
            "sa_raw",
            "docking_raw",
            "rdkit_valid",
            "pb_valid",
            "docking_success",
            "reward_found",
        ):
            record[column] = row.get(column, np.nan)
        records.append(record)
    result = pd.DataFrame.from_records(records)
    LOGGER.info("Loaded PB-fast molecules: n=%d", len(result))
    return result


def _summarize_cohort(group: pd.DataFrame) -> pd.Series:
    """Summarize carbon dominance, polarity, topology, and available target metrics."""
    summary: dict[str, float] = {
        "n": len(group),
        "all_carbon_rate": group["all_carbon"].mean(),
        "no_nos_rate": group["no_nos"].mean(),
        "halogen_only_hetero_rate": group["halogen_only_hetero"].mean(),
        "hetero_le1_rate": (group["hetero_atom_count"] <= 1).mean(),
        "carbon_ge_0_8_rate": (group["carbon_fraction"] >= 0.8).mean(),
        "carbon_ge_0_9_rate": (group["carbon_fraction"] >= 0.9).mean(),
        "carbon_fraction_mean": group["carbon_fraction"].mean(),
        "hetero_atom_mean": group["hetero_atom_count"].mean(),
        "polar_hetero_mean": group["polar_hetero_count"].mean(),
        "halogen_mean": group["halogen_count"].mean(),
        "nitrogen_mean": group["nitrogen_count"].mean(),
        "oxygen_mean": group["oxygen_count"].mean(),
        "sulfur_mean": group["sulfur_count"].mean(),
        "logp_mean": group["logp"].mean(),
        "logp_gt5_rate": (group["logp"] > 5).mean(),
        "tpsa_mean": group["tpsa"].mean(),
        "tpsa_lt20_rate": (group["tpsa"] < 20).mean(),
        "hba_mean": group["hba"].mean(),
        "hba_zero_rate": (group["hba"] == 0).mean(),
        "hbd_mean": group["hbd"].mean(),
        "qed_mean": group["qed_calc"].mean(),
        "ring_count_mean": group["ring_count"].mean(),
        "aromatic_ring_mean": group["aromatic_ring_count"].mean(),
        "three_member_ring_rate": (group["three_member_ring_count"] > 0).mean(),
        "large_ring_rate": (group["large_ring_count"] > 0).mean(),
        "fused_ring_pair_mean": group["fused_ring_pair_count"].mean(),
        "multi_fragment_rate": (group["fragment_count"] > 1).mean(),
    }
    for column in ("reward_total", "deppa_reward", "vina_score", "docking_raw", "pb_valid", "docking_success"):
        summary[f"{column}_mean"] = pd.to_numeric(group.get(column), errors="coerce").mean() if column in group else np.nan
    return pd.Series(summary)


def _chembl_summary() -> pd.DataFrame:
    """Calculate exact comparable ChEMBL 36 statistics from 1.55M property rows."""
    columns = [
        "heavy_atoms",
        "num_heteroatoms",
        "num_carbon_atoms",
        "carbon_fraction",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rings",
        "num_aromatic_rings",
    ]
    source = pd.read_csv(CHEMBL_CSV, usecols=columns)
    row = {
        "method": "ChEMBL 36",
        "cohort": "reference",
        "n": len(source),
        "all_carbon_rate": (source["num_heteroatoms"] == 0).mean(),
        "no_nos_rate": np.nan,
        "halogen_only_hetero_rate": np.nan,
        "hetero_le1_rate": (source["num_heteroatoms"] <= 1).mean(),
        "carbon_ge_0_8_rate": (source["carbon_fraction"] >= 0.8).mean(),
        "carbon_ge_0_9_rate": (source["carbon_fraction"] >= 0.9).mean(),
        "carbon_fraction_mean": source["carbon_fraction"].mean(),
        "hetero_atom_mean": source["num_heteroatoms"].mean(),
        "polar_hetero_mean": np.nan,
        "halogen_mean": np.nan,
        "nitrogen_mean": np.nan,
        "oxygen_mean": np.nan,
        "sulfur_mean": np.nan,
        "logp_mean": source["mol_log_p"].mean(),
        "logp_gt5_rate": (source["mol_log_p"] > 5).mean(),
        "tpsa_mean": source["tpsa"].mean(),
        "tpsa_lt20_rate": (source["tpsa"] < 20).mean(),
        "hba_mean": source["num_h_acceptors"].mean(),
        "hba_zero_rate": (source["num_h_acceptors"] == 0).mean(),
        "hbd_mean": source["num_h_donors"].mean(),
        "qed_mean": np.nan,
        "ring_count_mean": source["num_rings"].mean(),
        "aromatic_ring_mean": source["num_aromatic_rings"].mean(),
        "three_member_ring_rate": np.nan,
        "large_ring_rate": np.nan,
        "fused_ring_pair_mean": np.nan,
        "multi_fragment_rate": np.nan,
        "reward_total_mean": np.nan,
        "deppa_reward_mean": np.nan,
        "vina_score_mean": np.nan,
        "docking_raw_mean": np.nan,
        "pb_valid_mean": np.nan,
        "docking_success_mean": np.nan,
    }
    return pd.DataFrame([row])


def _reward_stratified(data: pd.DataFrame) -> pd.DataFrame:
    """Compare late-window top target/reward deciles with all late valid molecules."""
    configurations = {
        "DiffSBDD dock+classifier": ("reward_total", True),
        "DiffGui Next6-04": ("reward_total", True),
        "DiffGui DePPA t200": ("reward_total", True),
        "DiffGui PB-fast": ("reward_total", True),
        "DePPA original": ("vina_score", True),
    }
    rows = []
    for method, (score_column, larger_is_better) in configurations.items():
        frame = data[(data["method"] == method) & (data["cohort"] == "late")].copy()
        frame[score_column] = pd.to_numeric(frame[score_column], errors="coerce")
        scored = frame[frame[score_column].notna()].copy()
        if scored.empty:
            continue
        threshold = scored[score_column].quantile(0.9 if larger_is_better else 0.1)
        top = scored[scored[score_column] >= threshold] if larger_is_better else scored[scored[score_column] <= threshold]
        for stratum, subset in (("all_scored", scored), ("top_10pct", top)):
            summary = _summarize_cohort(subset).to_dict()
            summary.update({"method": method, "stratum": stratum, "score_column": score_column, "threshold": threshold})
            rows.append(summary)
    return pd.DataFrame(rows)


def _correlations(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate molecule- and epoch-level proxy/property correlations."""
    rows = []
    settings = {
        "DePPA original": ("vina_score", 1.0),
        "DiffGui DePPA t200": ("docking_raw", -1.0),
        "DiffGui PB-fast": ("docking_raw", -1.0),
    }
    properties = ["all_carbon", "no_nos", "carbon_fraction", "hetero_atom_count", "polar_hetero_count", "halogen_count", "logp", "tpsa", "qed_calc"]
    for method, (score_column, direction) in settings.items():
        frame = data[data["method"] == method].copy()
        frame["target_strength"] = pd.to_numeric(frame[score_column], errors="coerce") * direction
        for level in ("molecule", "epoch"):
            test = frame if level == "molecule" else frame.groupby("epoch", as_index=False).mean(numeric_only=True)
            for prop in properties:
                pair = test[["target_strength", prop]].dropna()
                rows.append(
                    {
                        "method": method,
                        "level": level,
                        "score_column": score_column,
                        "property": prop,
                        "n": len(pair),
                        "pearson_r": pair.corr(method="pearson").iloc[0, 1] if len(pair) > 2 else np.nan,
                        "spearman_r": pair.corr(method="spearman").iloc[0, 1] if len(pair) > 2 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _plot_carbon_drift(summary: pd.DataFrame) -> None:
    """Plot early/late carbon dominance and polarity changes with ChEMBL reference."""
    metrics = [
        ("all_carbon_rate", "All-carbon\nmolecules (%)", 100),
        ("no_nos_rate", "No N/O/S\nmolecules (%)", 100),
        ("carbon_ge_0_9_rate", "Carbon fraction\n>= 0.90 (%)", 100),
        ("hetero_le1_rate", "Hetero atoms\n<= 1 (%)", 100),
        ("tpsa_lt20_rate", "TPSA < 20\nA^2 (%)", 100),
        ("hba_zero_rate", "HBA = 0\n(%)", 100),
        ("logp_gt5_rate", "LogP > 5\n(%)", 100),
        ("polar_hetero_mean", "Mean N+O+S\ncount", 1),
    ]
    colors = {"early": "#5B8DB8", "late": "#D96459"}
    fig, axes = plt.subplots(2, 4, figsize=(11.7, 5.8))
    x = np.arange(len(METHOD_ORDER))
    for ax, (metric, title, scale) in zip(axes.flat, metrics):
        for offset, cohort in ((-0.16, "early"), (0.16, "late")):
            values = []
            for method in METHOD_ORDER:
                row = summary[(summary["method"] == method) & (summary["cohort"] == cohort)]
                values.append(row.iloc[0][metric] * scale if not row.empty else np.nan)
            ax.bar(x + offset, values, width=0.30, color=colors[cohort], label=cohort.capitalize())
        chembl = summary[(summary["method"] == "ChEMBL 36") & (summary["cohort"] == "reference")]
        if not chembl.empty and pd.notna(chembl.iloc[0][metric]):
            ax.axhline(chembl.iloc[0][metric] * scale, color="#333333", linestyle="--", linewidth=1, label="ChEMBL 36")
        ax.set_title(title, fontsize=9)
        ax.set_xticks(x, ["DS\nD+C", "DG\nNext6", "DG\nt200", "DG\nPB", "DePPA"], fontsize=7)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(loc="upper left", fontsize=7)
    fig.suptitle("RL repeatedly enriches carbon-dominant and polarity-poor molecules", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_figure(fig, "carbon_polarity_early_late_panel")


def _plot_deppa_trends(data: pd.DataFrame) -> None:
    """Plot original DePPA epoch-level target and carbon/polarity trajectories."""
    frame = data[data["method"] == "DePPA original"].copy()
    epoch = frame.groupby("epoch").agg(
        n=("canonical_smiles", "size"),
        all_carbon_rate=("all_carbon", "mean"),
        no_nos_rate=("no_nos", "mean"),
        carbon_fraction=("carbon_fraction", "mean"),
        hetero_atoms=("hetero_atom_count", "mean"),
        polar_hetero=("polar_hetero_count", "mean"),
        halogens=("halogen_count", "mean"),
        logp=("logp", "mean"),
        tpsa=("tpsa", "mean"),
        qed=("qed_calc", "mean"),
        vina_score=("vina_score", "mean"),
    ).reset_index()
    epoch.to_csv(OUTPUT_DIR / "deppa_original_epoch_element_trends.csv", index=False)
    smooth = epoch.set_index("epoch").rolling(10, min_periods=3, center=True).mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8), sharex=True)
    panels = [
        ("vina_score", "QVina reward", "#2A9D8F"),
        ("all_carbon_rate", "All-carbon rate", "#C44536"),
        ("no_nos_rate", "No N/O/S rate", "#D97706"),
        ("carbon_fraction", "Mean carbon fraction", "#5B5F97"),
        ("polar_hetero", "Mean N+O+S count", "#4C78A8"),
        ("logp", "Mean LogP", "#8C564B"),
    ]
    for ax, (metric, label, color) in zip(axes.flat, panels):
        ax.scatter(epoch["epoch"], epoch[metric], s=8, color=color, alpha=0.28, linewidths=0)
        ax.plot(smooth["epoch"], smooth[metric], color=color, linewidth=1.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.18)
    for ax in axes[1]:
        ax.set_xlabel("Training epoch")
    fig.suptitle("Original DePPA: QVina optimization tracks carbonization", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_figure(fig, "deppa_original_carbonization_trends")


def _plot_reward_shortcuts(stratified: pd.DataFrame) -> None:
    """Plot late-window enrichment in top target/reward deciles."""
    metrics = [
        ("all_carbon_rate", "All-carbon", 100),
        ("no_nos_rate", "No N/O/S", 100),
        ("carbon_ge_0_9_rate", "C fraction >= 0.90", 100),
        ("tpsa_lt20_rate", "TPSA < 20", 100),
        ("hba_zero_rate", "HBA = 0", 100),
        ("logp_gt5_rate", "LogP > 5", 100),
    ]
    methods = [method for method in METHOD_ORDER if method in set(stratified["method"])]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.8))
    x = np.arange(len(methods))
    for ax, (metric, title, scale) in zip(axes.flat, metrics):
        all_values = []
        top_values = []
        for method in methods:
            subset = stratified[stratified["method"] == method].set_index("stratum")
            all_values.append(subset.loc["all_scored", metric] * scale)
            top_values.append(subset.loc["top_10pct", metric] * scale)
        ax.bar(x - 0.17, all_values, 0.34, label="Late scored", color="#9AA6B2")
        ax.bar(x + 0.17, top_values, 0.34, label="Top 10%", color="#D96459")
        ax.set_title(f"{title} (%)")
        ax.set_xticks(x, ["DS\nD+C", "DG\nNext6", "DG\nt200", "DG\nPB", "DePPA"][: len(methods)], fontsize=7)
        ax.grid(axis="y", alpha=0.18)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("High target/reward molecules retain or amplify carbon shortcuts", fontsize=12, y=1.01)
    fig.tight_layout()
    _save_figure(fig, "late_top_reward_carbon_enrichment")


def _plot_deppa_all_carbon_examples(data: pd.DataFrame) -> None:
    """Show the highest-QVina late-stage all-carbon DePPA molecules."""
    examples = data[
        (data["method"] == "DePPA original")
        & (data["cohort"] == "late")
        & (data["all_carbon"] == 1)
        & data["vina_score"].notna()
    ].nlargest(12, "vina_score").copy()
    columns = [
        "epoch",
        "canonical_smiles",
        "vina_score",
        "qed_calc",
        "logp",
        "tpsa",
        "heavy_atom_count",
        "ring_count",
        "three_member_ring_count",
        "large_ring_count",
        "fused_ring_pair_count",
    ]
    examples[columns].to_csv(OUTPUT_DIR / "deppa_late_top_all_carbon_examples.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(11.4, 7.6))
    for ax, row in zip(axes.flat, examples.to_dict("records")):
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        image = Draw.MolToImage(mol, size=(420, 300), kekulize=False)
        ax.imshow(image)
        ax.set_title(
            f"ep {int(row['epoch'])} | QVina {row['vina_score']:.2f} | LogP {row['logp']:.2f}\n"
            f"rings {int(row['ring_count'])} | 3-ring {int(row['three_member_ring_count'])} | TPSA {row['tpsa']:.0f}",
            fontsize=7,
        )
        ax.axis("off")
    fig.suptitle("Original DePPA: highest-QVina all-carbon molecules in epochs 91-100", fontsize=12, y=0.995)
    fig.tight_layout()
    _save_figure(fig, "deppa_late_top_all_carbon_structures")


def _save_figure(fig: plt.Figure, stem: str) -> None:
    """Export editable vector and high-resolution preview formats."""
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the complete carbon-drift audit and export source data and figures."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    _configure_plotting()
    frames = [
        _load_endpoint_pair(DS_RUN, [(1, "early"), (25, "late")], "DiffSBDD dock+classifier"),
        _load_endpoint_pair(DG20_RUN, [(20, "early")], "DiffGui Next6-04"),
        _load_endpoint_pair(DG50_RUN, [(50, "late")], "DiffGui Next6-04"),
        _load_diffgui_t200(),
        _load_pbfast(),
        _load_deppa_original(),
    ]
    molecules = pd.concat(frames, ignore_index=True, sort=False)
    molecules.to_csv(OUTPUT_DIR / "carbon_drift_molecule_level.csv.gz", index=False, compression="gzip")

    selected = molecules[molecules["cohort"].isin(["early", "late"])].copy()
    summary = selected.groupby(["method", "cohort"], sort=False).apply(_summarize_cohort, include_groups=False).reset_index()
    summary = pd.concat([summary, _chembl_summary()], ignore_index=True, sort=False)
    summary.to_csv(OUTPUT_DIR / "carbon_drift_cohort_summary.csv", index=False)

    stratified = _reward_stratified(molecules)
    stratified.to_csv(OUTPUT_DIR / "late_reward_stratified_summary.csv", index=False)
    correlations = _correlations(molecules)
    correlations.to_csv(OUTPUT_DIR / "target_property_correlations.csv", index=False)

    _plot_carbon_drift(summary)
    _plot_deppa_trends(molecules)
    _plot_reward_shortcuts(stratified)
    _plot_deppa_all_carbon_examples(molecules)

    manifest = {
        "figure_contract": {
            "core_conclusion": "RL repeatedly enriches carbon-dominant, polarity-poor chemotypes; target-score gains do not imply improved applicability.",
            "evidence_chain": [
                "Five directly recalculated RL trajectories compared early versus late",
                "Exact ChEMBL 36 aggregate reference over 1,550,090 molecules",
                "Original DePPA 100-epoch element and property trajectories",
                "Late-window top target/reward decile enrichment",
            ],
            "archetype": "quantitative grid",
            "backend": "Python/matplotlib",
            "exports": ["PNG", "SVG", "PDF", "source CSV"],
        },
        "molecule_rows": int(len(molecules)),
        "source_paths": [str(path.relative_to(ROOT)) for path in [PBFAST_CSV, T200_CSV, CHEMBL_CSV]],
        "limitations": [
            "Different methods use different reward and docking backends; target scores are not pooled on one numerical scale.",
            "DiffSBDD and Next6 comparisons use selected endpoint epochs; long runs use ten-epoch windows.",
            "ChEMBL no-N/O/S and halogen-only rates are not reported because the property table lacks element-specific counts.",
        ],
    }
    (OUTPUT_DIR / "deep_review_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Analysis complete: molecule_rows=%d", len(molecules))


if __name__ == "__main__":
    main()
