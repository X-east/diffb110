#!/usr/bin/env python3
"""Evaluate ChEMBL classifier output distribution on base DiffSBDD samples."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDF = (
    PROJECT_ROOT
    / "OUT"
    / "260407_generate_ligands_10000"
    / "260407generated_10000_filtered.sdf"
)
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "chembl_classifier"
    / "models"
    / "chembl_classifier_selected_v0_1.pt"
)
DEFAULT_OUT_DIR = (
    PROJECT_ROOT
    / "OUT"
    / "260407_generate_ligands_10000"
    / "classifier_selected_v0_1_base_distribution"
)
SASCORER_PATH = PROJECT_ROOT / "DiffSBDD" / "analysis" / "SA_Score" / "sascorer.py"
MOLSKILL_ROOT = PROJECT_ROOT / "reference project" / "molskill"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(MOLSKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(MOLSKILL_ROOT))

from chembl_classifier.model import ChemblClassifier  # noqa: E402
from molskill.data.featurizers import get_featurizer  # noqa: E402


PLOT_DPI = 160
COLOR_MAIN = "#457B9D"
COLOR_ACCENT = "#E76F51"
COLOR_GOOD = "#2A9D8F"
COLOR_WARN = "#F4A261"
COLOR_BAD = "#D62828"


def _load_sa_scorer():
    if not SASCORER_PATH.exists():
        raise FileNotFoundError(f"SA scorer not found: {SASCORER_PATH}")
    spec = importlib.util.spec_from_file_location("diffsbdd_sascorer", SASCORER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load SA scorer from {SASCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.calculateScore


calculate_sa_score = _load_sa_scorer()


def _map_classifier_reward(
    probs: np.ndarray,
    good_threshold: float = 0.5,
    bad_threshold: float = 0.5,
    cap_high: float = 0.8,
    cap_low: float = 0.2,
    max_reward: float = 1.0,
    penalty: float = -1.0,
) -> np.ndarray:
    mapped = np.zeros_like(probs, dtype=np.float64)
    high_mask = probs >= good_threshold
    low_mask = probs < bad_threshold
    scale_h = max(cap_high - good_threshold, 1e-8)
    scale_l = max(bad_threshold - cap_low, 1e-8)
    mapped[high_mask] = max_reward * np.clip(
        (probs[high_mask] - good_threshold) / scale_h, 0.0, 1.0
    )
    mapped[low_mask] = penalty * np.clip(
        (bad_threshold - probs[low_mask]) / scale_l, 0.0, 1.0
    )
    return mapped


def _lipinski_pass(mol: Chem.Mol) -> int:
    logp = Crippen.MolLogP(mol)
    return int(
        Descriptors.ExactMolWt(mol) < 500
        and Lipinski.NumHDonors(mol) <= 5
        and Lipinski.NumHAcceptors(mol) <= 10
        and -2 <= logp <= 5
        and rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    )


def _atom_counts(mol: Chem.Mol) -> dict[str, int | float]:
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    heavy = mol.GetNumHeavyAtoms()
    carbon = sum(symbol == "C" for symbol in atoms)
    hetero = sum(symbol != "C" and symbol != "H" for symbol in atoms)
    return {
        "carbon_atoms": int(carbon),
        "hetero_atoms": int(hetero),
        "n_n": int(sum(symbol == "N" for symbol in atoms)),
        "n_o": int(sum(symbol == "O" for symbol in atoms)),
        "n_s": int(sum(symbol == "S" for symbol in atoms)),
        "n_halogen": int(sum(symbol in {"F", "Cl", "Br", "I"} for symbol in atoms)),
        "carbon_fraction": carbon / heavy if heavy else np.nan,
        "hetero_fraction": hetero / heavy if heavy else np.nan,
    }


def _descriptor_record(mol: Chem.Mol, mol_id: int) -> dict[str, object]:
    smiles = Chem.MolToSmiles(mol, canonical=True)
    record: dict[str, object] = {
        "molecule_id": mol_id,
        "smiles": smiles,
        "mol_wt": Descriptors.MolWt(mol),
        "exact_mol_wt": Descriptors.ExactMolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hba": int(Lipinski.NumHAcceptors(mol)),
        "hbd": int(Lipinski.NumHDonors(mol)),
        "rot_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "aliphatic_ring_count": int(rdMolDescriptors.CalcNumAliphaticRings(mol)),
        "saturated_ring_count": int(rdMolDescriptors.CalcNumSaturatedRings(mol)),
        "spiro_atoms": int(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
        "bridgehead_atoms": int(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
        "fraction_csp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "qed": QED.qed(mol),
        "sa_score": calculate_sa_score(mol),
        "lipinski_pass": _lipinski_pass(mol),
    }
    record.update(_atom_counts(mol))
    return record


def _load_valid_molecules(sdf_path: Path) -> tuple[list[Chem.Mol], pd.DataFrame]:
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    mols: list[Chem.Mol] = []
    records: list[dict[str, object]] = []
    failed = 0

    for mol_id, mol in enumerate(supplier, start=1):
        if mol is None:
            failed += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            records.append(_descriptor_record(mol, mol_id))
            mols.append(mol)
        except Exception:
            failed += 1

    df = pd.DataFrame(records)
    df.attrs["failed_molecule_count"] = failed
    df.attrs["input_record_count"] = len(supplier)
    return mols, df


def _predict_probabilities(
    mols: list[Chem.Mol],
    model_path: Path,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    featurizer = get_featurizer("morgan_count_rdkit_2d")
    features = []
    kept_indices = []
    for idx, mol in enumerate(mols):
        try:
            feat = featurizer.get_feat(mol)
        except Exception:
            feat = None
        if feat is None:
            continue
        features.append(feat)
        kept_indices.append(idx)

    if not features:
        raise RuntimeError("No molecules could be featurized")

    model = ChemblClassifier.from_checkpoint(str(model_path), device=device)
    X = np.stack(features, axis=0).astype(np.float32)
    probs = []
    for start in range(0, len(X), batch_size):
        batch = torch.from_numpy(X[start : start + batch_size]).to(device)
        prob = model.predict_proba(batch).detach().cpu().numpy().reshape(-1)
        probs.append(prob)
    return np.asarray(kept_indices, dtype=int), np.concatenate(probs)


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flag_prob_lt_0_01"] = out["classifier_prob"] < 0.01
    out["flag_prob_lt_0_05"] = out["classifier_prob"] < 0.05
    out["flag_prob_lt_0_10"] = out["classifier_prob"] < 0.10
    out["flag_prob_lt_0_20"] = out["classifier_prob"] < 0.20
    out["flag_prob_ge_0_50"] = out["classifier_prob"] >= 0.50
    out["flag_prob_ge_0_80"] = out["classifier_prob"] >= 0.80
    out["flag_reward_floor"] = out["classifier_reward_mapped"] <= -0.999
    out["flag_reward_positive"] = out["classifier_reward_mapped"] > 0
    out["flag_reward_gt_0_5"] = out["classifier_reward_mapped"] > 0.5
    out["flag_complex_sa_gt_5"] = out["sa_score"] > 5
    out["flag_complex_sa_gt_6"] = out["sa_score"] > 6
    out["flag_ring_ge_4"] = out["ring_count"] >= 4
    out["flag_ring_ge_6"] = out["ring_count"] >= 6
    out["flag_rot_gt_10"] = out["rot_bonds"] > 10
    out["flag_mw_gt_500"] = out["mol_wt"] > 500
    out["flag_logp_gt_4"] = out["logp"] > 4
    out["flag_logp_gt_5"] = out["logp"] > 5
    out["flag_carbon_gt_0_8"] = out["carbon_fraction"] > 0.8
    out["flag_bridge_or_spiro"] = (out["bridgehead_atoms"] > 0) | (out["spiro_atoms"] > 0)
    out["flag_lipinski_fail"] = out["lipinski_pass"] == 0
    return out


def _summary_table(df: pd.DataFrame, failed: int, input_count: int) -> pd.DataFrame:
    probs = df["classifier_prob"]
    rewards = df["classifier_reward_mapped"]
    rows = [
        ("input_records", input_count),
        ("valid_scored", len(df)),
        ("sanitize_or_featurize_failed", failed),
        ("prob_mean", probs.mean()),
        ("prob_median", probs.median()),
        ("prob_q01", probs.quantile(0.01)),
        ("prob_q05", probs.quantile(0.05)),
        ("prob_q10", probs.quantile(0.10)),
        ("prob_q25", probs.quantile(0.25)),
        ("prob_q75", probs.quantile(0.75)),
        ("prob_q90", probs.quantile(0.90)),
        ("prob_q95", probs.quantile(0.95)),
        ("prob_q99", probs.quantile(0.99)),
        ("prob_lt_0_01_ratio", (probs < 0.01).mean()),
        ("prob_lt_0_05_ratio", (probs < 0.05).mean()),
        ("prob_lt_0_10_ratio", (probs < 0.10).mean()),
        ("prob_lt_0_20_ratio", (probs < 0.20).mean()),
        ("prob_ge_0_50_ratio", (probs >= 0.50).mean()),
        ("prob_ge_0_80_ratio", (probs >= 0.80).mean()),
        ("mapped_reward_mean", rewards.mean()),
        ("mapped_reward_median", rewards.median()),
        ("mapped_reward_floor_ratio", (rewards <= -0.999).mean()),
        ("mapped_reward_positive_ratio", (rewards > 0).mean()),
        ("mapped_reward_gt_0_5_ratio", (rewards > 0.5).mean()),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _bin_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50, 0.80, 1.0]
    labels = ["<0.01", "0.01-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.50", "0.50-0.80", ">=0.80"]
    out = df.copy()
    out["prob_bin"] = pd.cut(
        out["classifier_prob"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=False,
    )
    out.loc[out["classifier_prob"] >= 0.8, "prob_bin"] = ">=0.80"
    prop_cols = [
        "classifier_prob",
        "classifier_reward_mapped",
        "mol_wt",
        "logp",
        "tpsa",
        "qed",
        "sa_score",
        "ring_count",
        "rot_bonds",
        "heavy_atom_count",
        "carbon_fraction",
        "hetero_atoms",
        "spiro_atoms",
        "bridgehead_atoms",
    ]
    grouped = out.groupby("prob_bin", observed=False)
    prop_summary = grouped[prop_cols].agg(["count", "mean", "median"]).reset_index()
    prop_summary.columns = [
        "_".join([str(x) for x in col if str(x)])
        for col in prop_summary.columns.to_flat_index()
    ]
    flag_cols = [c for c in out.columns if c.startswith("flag_")]
    flag_summary = grouped[flag_cols].mean().reset_index()
    return prop_summary, flag_summary


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    small = df.head(max_rows).copy()
    if small.empty:
        return "_empty_"

    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}" if np.isfinite(value) else "nan"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(c) for c in small.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in small.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in small.columns) + " |")
    return "\n".join(lines)


def _finish_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_probability_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    probs = df["classifier_prob"]
    rewards = df["classifier_reward_mapped"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Classifier Output on Base DiffSBDD Generated Molecules", fontsize=16, weight="bold")

    ax = axes[0, 0]
    ax.hist(probs, bins=60, color=COLOR_MAIN, alpha=0.85)
    for x, label in [(0.2, "cap_low=0.2"), (0.5, "threshold=0.5"), (0.8, "cap_high=0.8")]:
        ax.axvline(x, color=COLOR_ACCENT, ls="--", lw=1.2)
        ax.text(x, ax.get_ylim()[1] * 0.9, label, rotation=90, va="top", fontsize=8)
    ax.set_xlabel("Classifier probability")
    ax.set_ylabel("Count")
    ax.set_title("Probability Histogram")
    ax.grid(alpha=0.2)

    ax = axes[0, 1]
    sorted_probs = np.sort(probs)
    y = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
    ax.plot(sorted_probs, y, color=COLOR_MAIN, lw=2)
    for x in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
        ax.axvline(x, color="#999999", ls="--", lw=0.8)
    ax.set_xlabel("Classifier probability")
    ax.set_ylabel("CDF")
    ax.set_title("Cumulative Distribution")
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    ax.hist(rewards, bins=50, color=COLOR_ACCENT, alpha=0.85)
    ax.axvline(-1, color=COLOR_BAD, ls="--", lw=1.2)
    ax.axvline(0, color="#333333", ls="--", lw=1.2)
    ax.axvline(1, color=COLOR_GOOD, ls="--", lw=1.2)
    ax.set_xlabel("Mapped classifier reward")
    ax.set_ylabel("Count")
    ax.set_title("Current Reward Mapping")
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    flags = {
        "p<0.01": (probs < 0.01).mean(),
        "p<0.05": (probs < 0.05).mean(),
        "p<0.10": (probs < 0.10).mean(),
        "p<0.20": (probs < 0.20).mean(),
        "p>=0.50": (probs >= 0.50).mean(),
        "p>=0.80": (probs >= 0.80).mean(),
    }
    ax.bar(flags.keys(), [v * 100 for v in flags.values()], color=[COLOR_BAD] * 4 + [COLOR_GOOD] * 2)
    ax.set_ylabel("Molecules (%)")
    ax.set_title("Threshold Fractions")
    ax.grid(axis="y", alpha=0.2)

    _finish_plot(fig, out_dir / "01_classifier_probability_distribution.png")


def _plot_property_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Classifier Probability vs Molecular Properties", fontsize=16, weight="bold")
    specs = [
        ("logp", "LogP"),
        ("sa_score", "SA Score"),
        ("ring_count", "Ring Count"),
        ("mol_wt", "Molecular Weight"),
        ("qed", "QED"),
        ("carbon_fraction", "Carbon Fraction"),
    ]
    for ax, (col, title) in zip(axes.ravel(), specs):
        ax.scatter(df[col], df["classifier_prob"], s=8, alpha=0.35, color=COLOR_MAIN)
        ax.set_xlabel(title)
        ax.set_ylabel("Classifier probability")
        ax.set_title(title)
        ax.grid(alpha=0.2)
    _finish_plot(fig, out_dir / "02_probability_vs_properties.png")


def _plot_bins(prop_summary: pd.DataFrame, flag_summary: pd.DataFrame, out_dir: Path) -> None:
    labels = prop_summary["prob_bin"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Property and Complexity by Classifier Probability Bin", fontsize=16, weight="bold")

    ax = axes[0, 0]
    ax.bar(labels, prop_summary["classifier_prob_count"], color=COLOR_MAIN)
    ax.set_title("Molecule Count by Probability Bin")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    for col, label in [
        ("logp_mean", "LogP"),
        ("sa_score_mean", "SA"),
        ("ring_count_mean", "Rings"),
    ]:
        full_col = col if col in prop_summary.columns else col.replace("_mean", "_mean")
        ax.plot(labels, prop_summary[full_col], marker="o", label=label)
    ax.set_title("Mean Properties")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 0]
    flag_cols = [
        "flag_complex_sa_gt_5",
        "flag_ring_ge_4",
        "flag_logp_gt_4",
        "flag_mw_gt_500",
        "flag_lipinski_fail",
    ]
    bottom = np.zeros(len(flag_summary))
    for col in flag_cols:
        values = flag_summary[col].to_numpy() * 100
        ax.bar(labels, values, bottom=bottom, label=col.replace("flag_", ""))
        bottom += values * 0
    ax.set_title("Complexity Flag Rate by Bin")
    ax.set_ylabel("% in bin")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)

    ax = axes[1, 1]
    for col, label in [
        ("mol_wt_mean", "MW"),
        ("heavy_atom_count_mean", "Heavy atoms"),
        ("rot_bonds_mean", "Rot bonds"),
    ]:
        ax.plot(labels, prop_summary[col], marker="o", label=label)
    ax.set_title("Size and Flexibility")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    ax.grid(alpha=0.2)

    _finish_plot(fig, out_dir / "03_probability_bin_properties.png")


def _plot_reasonable_region(df: pd.DataFrame, out_dir: Path) -> None:
    reasonable = (
        (df["mol_wt"].between(180, 450))
        & (df["logp"].between(0.5, 3.8))
        & (df["sa_score"] <= 5.0)
        & (df["ring_count"] <= 4)
        & (df["rot_bonds"] <= 8)
        & (df["heavy_atom_count"].between(12, 32))
        & (df["carbon_fraction"] <= 0.78)
        & (df["hetero_atoms"] >= 2)
        & (df["tpsa"].between(35, 130))
        & (df["formal_charge"].between(-1, 1))
    )
    out = df.copy()
    out["reasonable_region"] = reasonable
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Candidate Desired Region vs Classifier Probability", fontsize=16, weight="bold")

    ax = axes[0]
    ax.hist(
        [out.loc[reasonable, "classifier_prob"], out.loc[~reasonable, "classifier_prob"]],
        bins=40,
        label=["reasonable_region", "outside_region"],
        color=[COLOR_GOOD, COLOR_BAD],
        alpha=0.75,
    )
    ax.set_title("Probability Split by Reasonable Region")
    ax.set_xlabel("Classifier probability")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1]
    rates = {
        "reasonable_region": reasonable.mean(),
        "reasonable & p>=0.5": (reasonable & (out["classifier_prob"] >= 0.5)).mean(),
        "outside & p>=0.5": ((~reasonable) & (out["classifier_prob"] >= 0.5)).mean(),
        "outside & p<0.2": ((~reasonable) & (out["classifier_prob"] < 0.2)).mean(),
    }
    ax.bar(rates.keys(), [v * 100 for v in rates.values()], color=[COLOR_GOOD, COLOR_GOOD, COLOR_WARN, COLOR_BAD])
    ax.set_ylabel("All molecules (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Region / Probability Cross Fractions")
    ax.grid(axis="y", alpha=0.2)

    _finish_plot(fig, out_dir / "04_reasonable_region_probability.png")
    region_summary = (
        out.groupby("reasonable_region")
        .agg(
            n=("smiles", "size"),
            prob_mean=("classifier_prob", "mean"),
            prob_median=("classifier_prob", "median"),
            prob_ge_0_5=("classifier_prob", lambda s: (s >= 0.5).mean()),
            prob_lt_0_2=("classifier_prob", lambda s: (s < 0.2).mean()),
            reward_floor=("classifier_reward_mapped", lambda s: (s <= -0.999).mean()),
            logp_mean=("logp", "mean"),
            sa_mean=("sa_score", "mean"),
            ring_mean=("ring_count", "mean"),
            mw_mean=("mol_wt", "mean"),
        )
        .reset_index()
    )
    region_summary.to_csv(out_dir / "reasonable_region_classifier_summary.csv", index=False)


def _write_report(
    sdf_path: Path,
    model_path: Path,
    out_dir: Path,
    summary: pd.DataFrame,
    prop_summary: pd.DataFrame,
    flag_summary: pd.DataFrame,
) -> None:
    metrics = summary.set_index("metric")["value"]
    verdict = "不可直接作为稠密 RL reward"
    if metrics["prob_lt_0_20_ratio"] < 0.5 and metrics["prob_ge_0_50_ratio"] > 0.2:
        verdict = "可作为候选 dense reward，但仍需校准"

    lines = [
        "# 基础 DiffSBDD 生成分子的 classifier 输出分布",
        "",
        f"- SDF: `{sdf_path}`",
        f"- Model: `{model_path}`",
        f"- 结论: **{verdict}**",
        "",
        "## 关键结论",
        "",
        f"- scored valid molecules: {int(metrics['valid_scored'])}",
        f"- probability median: {metrics['prob_median']:.4f}",
        f"- probability mean: {metrics['prob_mean']:.4f}",
        f"- p < 0.20: {metrics['prob_lt_0_20_ratio']:.2%}",
        f"- p >= 0.50: {metrics['prob_ge_0_50_ratio']:.2%}",
        f"- p >= 0.80: {metrics['prob_ge_0_80_ratio']:.2%}",
        f"- mapped reward floor (-1): {metrics['mapped_reward_floor_ratio']:.2%}",
        f"- mapped reward positive: {metrics['mapped_reward_positive_ratio']:.2%}",
        "",
        "## 图表",
        "",
        "![probability distribution](01_classifier_probability_distribution.png)",
        "",
        "![probability vs properties](02_probability_vs_properties.png)",
        "",
        "![bin properties](03_probability_bin_properties.png)",
        "",
        "![reasonable region](04_reasonable_region_probability.png)",
        "",
        "## Summary",
        "",
        _markdown_table(summary, max_rows=80),
        "",
        "## Probability Bin Property Summary",
        "",
        _markdown_table(prop_summary, max_rows=20),
        "",
        "## Probability Bin Flag Summary",
        "",
        _markdown_table(flag_summary, max_rows=20),
        "",
        "## 文件",
        "",
        "- `base_diffsbdd_classifier_scores.csv`: 每个有效分子的 classifier probability/reward 与性质。",
        "- `classifier_probability_summary.csv`: probability、reward 映射和阈值比例。",
        "- `property_by_probability_bin.csv`: classifier probability 分箱后的性质统计。",
        "- `complexity_flags_by_probability_bin.csv`: 分箱后的复杂度/疏水/规则失败比例。",
        "- `top_classifier_molecules.csv` / `bottom_classifier_molecules.csv`: classifier 高低分分子。",
    ]
    (out_dir / "classifier_base_distribution_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_analysis(sdf_path: Path, model_path: Path, out_dir: Path, device: str, batch_size: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    mols, df = _load_valid_molecules(sdf_path)
    failed = int(df.attrs.get("failed_molecule_count", 0))
    input_count = int(df.attrs.get("input_record_count", len(mols)))

    kept_indices, probs = _predict_probabilities(mols, model_path, device, batch_size)
    df = df.iloc[kept_indices].reset_index(drop=True)
    df["classifier_prob"] = probs
    df["classifier_reward_mapped"] = _map_classifier_reward(probs)
    df = _add_flags(df)

    summary = _summary_table(df, failed + (len(mols) - len(kept_indices)), input_count)
    prop_summary, flag_summary = _bin_tables(df)
    top = df.sort_values("classifier_prob", ascending=False).head(100)
    bottom = df.sort_values("classifier_prob", ascending=True).head(100)

    df.to_csv(out_dir / "base_diffsbdd_classifier_scores.csv", index=False)
    summary.to_csv(out_dir / "classifier_probability_summary.csv", index=False)
    prop_summary.to_csv(out_dir / "property_by_probability_bin.csv", index=False)
    flag_summary.to_csv(out_dir / "complexity_flags_by_probability_bin.csv", index=False)
    top.to_csv(out_dir / "top_classifier_molecules.csv", index=False)
    bottom.to_csv(out_dir / "bottom_classifier_molecules.csv", index=False)

    _plot_probability_distribution(df, out_dir)
    _plot_property_relationships(df, out_dir)
    _plot_bins(prop_summary, flag_summary, out_dir)
    _plot_reasonable_region(df, out_dir)
    _write_report(sdf_path, model_path, out_dir, summary, prop_summary, flag_summary)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if not args.sdf.exists():
        raise FileNotFoundError(f"SDF not found: {args.sdf}")
    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    out_dir = run_analysis(
        sdf_path=args.sdf.resolve(),
        model_path=args.model.resolve(),
        out_dir=args.out_dir.resolve(),
        device=args.device,
        batch_size=args.batch_size,
    )
    print(f"Saved classifier distribution analysis to: {out_dir}")


if __name__ == "__main__":
    main()
