"""Analyze curated DiffGui/DiffSBDD target SDF batches."""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors


RDLogger.DisableLog("rdApp.*")

SCRIPT_DIR = Path(__file__).resolve().parent
CURATED_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
FIG_DIR = SCRIPT_DIR / "figures"
TABLE_DIR = SCRIPT_DIR / "tables"
DOCKING_DIR = SCRIPT_DIR / "docking"
MANIFEST = CURATED_DIR / "manifest.csv"
CHEMBL_PROPERTIES = REPO_ROOT / "data/chembl/chembl_36_properties.csv.gz"

SA_SCORE_DIR = REPO_ROOT / "DiffSBDD/analysis/SA_Score"
sys.path.insert(0, str(SA_SCORE_DIR))
import sascorer  # noqa: E402


ELEMENT_KEYS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]
MODEL_ORDER = ["DiffSBDD", "DiffGui"]
TARGET_ORDER = ["8GJL/C1101", "5XZR/8J6", "9JH3/R:401"]
MODEL_PALETTE = {"DiffSBDD": "#D55E00", "DiffGui": "#0072B2"}

DOCKING_SOURCES = {
    ("8GJL/C1101", "DiffGui"): REPO_ROOT
    / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/docking/unidock_fast_8gjl_c1101/DiffGui/docking_results.csv",
    ("8GJL/C1101", "DiffSBDD"): REPO_ROOT
    / "analysis/benchmark/8gjl_c1101_benchmark_20260603_500mol_500steps/docking/unidock_fast_8gjl_c1101/DiffSBDD/docking_results.csv",
    ("5XZR/8J6", "DiffGui"): REPO_ROOT
    / "analysis/benchmark/5xzr_8j6_benchmark_20260611/docking/unidock_fast_5xzr_8j6/DiffGui/docking_results.csv",
    ("5XZR/8J6", "DiffSBDD"): REPO_ROOT
    / "analysis/benchmark/5xzr_8j6_benchmark_20260611/docking/unidock_fast_5xzr_8j6/DiffSBDD/docking_results.csv",
    ("9JH3/R:401", "DiffGui"): DOCKING_DIR
    / "9JH3_R401/DiffGui/docking_results.csv",
    ("9JH3/R:401", "DiffSBDD"): DOCKING_DIR
    / "9JH3_R401/DiffSBDD/docking_results.csv",
}


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else math.nan


def _safe_median(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.median()) if len(clean) else math.nan


def _safe_quantile(values: pd.Series, q: float) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.quantile(q)) if len(clean) else math.nan


def _sanitize_copy(mol: Chem.Mol) -> Chem.Mol | None:
    copy = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(copy)
    except Exception:
        return None
    return copy


def _bad_bond_fraction(mol: Chem.Mol) -> tuple[int, int, float, float]:
    if mol.GetNumConformers() == 0 or mol.GetNumBonds() == 0:
        return 0, 0, 0.0, 0.0
    conf = mol.GetConformer()
    table = Chem.GetPeriodicTable()
    bad = 0
    checked = 0
    errors: list[float] = []
    for bond in mol.GetBonds():
        atom_a = bond.GetBeginAtom()
        atom_b = bond.GetEndAtom()
        if atom_a.GetAtomicNum() == 1 or atom_b.GetAtomicNum() == 1:
            continue
        pos_a = conf.GetAtomPosition(atom_a.GetIdx())
        pos_b = conf.GetAtomPosition(atom_b.GetIdx())
        dist = pos_a.Distance(pos_b)
        ideal = table.GetRcovalent(atom_a.GetAtomicNum()) + table.GetRcovalent(
            atom_b.GetAtomicNum()
        )
        if ideal <= 0:
            continue
        ratio = dist / ideal
        checked += 1
        errors.append(abs(dist - ideal))
        if ratio < 0.70 or ratio > 1.45:
            bad += 1
    return bad, checked, bad / checked if checked else 0.0, mean(errors) if errors else 0.0


def _count_fragments(mol: Chem.Mol) -> tuple[int, float]:
    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    if not fragments:
        return 0, 0.0
    heavy = max(1, mol.GetNumHeavyAtoms())
    largest = max(
        sum(1 for idx in frag if mol.GetAtomWithIdx(idx).GetAtomicNum() != 1)
        for frag in fragments
    )
    return len(fragments), largest / heavy


def _atom_counts_and_fractions(mol: Chem.Mol) -> tuple[dict[str, int], dict[str, float]]:
    counts: Counter[str] = Counter()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        counts[atom.GetSymbol()] += 1
    heavy = sum(counts.values())
    fractions = {
        f"{key.lower()}_fraction": counts.get(key, 0) / heavy if heavy else 0.0
        for key in ELEMENT_KEYS
    }
    known = sum(counts.get(key, 0) for key in ELEMENT_KEYS)
    fractions["other_fraction"] = (heavy - known) / heavy if heavy else 0.0
    return dict(counts), fractions


def _ring_features(mol: Chem.Mol | None) -> dict[str, float]:
    if mol is None:
        return {
            "ring_count": math.nan,
            "aromatic_ring_count": math.nan,
            "benzene_like_ring_count": math.nan,
            "three_member_ring_count": math.nan,
            "fused_ring_pair_count": math.nan,
            "max_fused_ring_system_size": math.nan,
            "macrocycle_count": math.nan,
            "max_ring_size": math.nan,
            "spiro_atoms": math.nan,
            "bridgehead_atoms": math.nan,
            "ring_complexity": math.nan,
            "has_aromatic_ring": False,
            "has_benzene_like_ring": False,
            "has_three_member_ring": False,
            "has_fused_ring": False,
            "has_macrocycle": False,
            "has_spiro": False,
            "has_bridgehead": False,
            "high_ring_count": False,
            "complex_ring_flag": False,
        }
    atom_rings = [set(ring) for ring in mol.GetRingInfo().AtomRings()]
    ring_sizes = [len(ring) for ring in atom_rings]
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
    benzene_like = _benzene_like_ring_count(mol, atom_rings)
    three = sum(1 for size in ring_sizes if size == 3)
    macro = sum(1 for size in ring_sizes if size >= 8)
    spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    bridge = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    fused_pairs = _fused_ring_pair_count(atom_rings)
    max_fused_system = _max_fused_ring_system_size(atom_rings)
    high_ring_count = ring_count >= 6
    complex_ring = (
        fused_pairs > 0
        or spiro > 0
        or bridge > 0
        or macro > 0
        or high_ring_count
    )
    return {
        "ring_count": ring_count,
        "aromatic_ring_count": aromatic,
        "benzene_like_ring_count": benzene_like,
        "three_member_ring_count": three,
        "fused_ring_pair_count": fused_pairs,
        "max_fused_ring_system_size": max_fused_system,
        "macrocycle_count": macro,
        "max_ring_size": max(ring_sizes) if ring_sizes else 0,
        "spiro_atoms": spiro,
        "bridgehead_atoms": bridge,
        "ring_complexity": ring_count + aromatic + three + macro + spiro + bridge + fused_pairs,
        "has_aromatic_ring": aromatic > 0,
        "has_benzene_like_ring": benzene_like > 0,
        "has_three_member_ring": three > 0,
        "has_fused_ring": fused_pairs > 0,
        "has_macrocycle": macro > 0,
        "has_spiro": spiro > 0,
        "has_bridgehead": bridge > 0,
        "high_ring_count": high_ring_count,
        "complex_ring_flag": complex_ring,
    }


def _benzene_like_ring_count(mol: Chem.Mol, atom_rings: list[set[int]]) -> int:
    count = 0
    for ring in atom_rings:
        if len(ring) != 6:
            continue
        atoms = [mol.GetAtomWithIdx(idx) for idx in ring]
        if all(atom.GetIsAromatic() and atom.GetAtomicNum() == 6 for atom in atoms):
            count += 1
    return count


def _fused_ring_pair_count(atom_rings: list[set[int]]) -> int:
    fused_pairs = 0
    for i, ring_a in enumerate(atom_rings):
        for ring_b in atom_rings[i + 1 :]:
            if len(ring_a & ring_b) >= 2:
                fused_pairs += 1
    return fused_pairs


def _max_fused_ring_system_size(atom_rings: list[set[int]]) -> int:
    if not atom_rings:
        return 0
    adjacency = {idx: set() for idx in range(len(atom_rings))}
    for i, ring_a in enumerate(atom_rings):
        for j in range(i + 1, len(atom_rings)):
            if len(ring_a & atom_rings[j]) >= 2:
                adjacency[i].add(j)
                adjacency[j].add(i)
    seen: set[int] = set()
    max_size = 1
    for start in adjacency:
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
        max_size = max(max_size, size)
    return max_size


def _blank_metric_row(
    manifest_row: dict[str, str],
    record_index: int,
    read_error: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "target": manifest_row["target"],
        "source_pdb": manifest_row["source_pdb"],
        "reference_ligand": manifest_row["reference_ligand"],
        "model": manifest_row["model"],
        "timesteps": int(manifest_row["timesteps"]),
        "record_index": record_index,
        "sdf_file": manifest_row["curated_file"],
        "rdkit_readable": False,
        "sanitize_success": False,
        "read_error": read_error,
        "smiles": "",
    }
    for key in [
        "heavy_atoms",
        "num_atoms",
        "num_bonds",
        "bad_bond_lengths",
        "checked_bond_lengths",
        "bad_bond_length_fraction",
        "mean_bond_length_abs_error",
        "num_fragments",
        "largest_fragment_atom_frac",
        "hetero_atoms",
        "carbon_fraction",
        "qed",
        "sa_score",
        "logp",
        "tpsa",
        "hba",
        "hbd",
        "rotatable_bonds",
        "mw",
        *[f"{element.lower()}_fraction" for element in ELEMENT_KEYS],
        "other_fraction",
    ]:
        row[key] = math.nan
    row.update(_ring_features(None))
    return row


def _mol_metric_row(
    manifest_row: dict[str, str],
    record_index: int,
    mol: Chem.Mol,
) -> dict[str, object]:
    sanitized = _sanitize_copy(mol)
    metric_mol = sanitized if sanitized is not None else mol
    smiles = ""
    if sanitized is not None:
        smiles = Chem.MolToSmiles(Chem.RemoveHs(sanitized), isomericSmiles=True)
    bad, checked, bad_fraction, mean_error = _bad_bond_fraction(metric_mol)
    fragments, largest_fragment = _count_fragments(metric_mol)
    atom_counts, fractions = _atom_counts_and_fractions(metric_mol)
    heavy = metric_mol.GetNumHeavyAtoms()
    carbon = atom_counts.get("C", 0)

    row: dict[str, object] = {
        "target": manifest_row["target"],
        "source_pdb": manifest_row["source_pdb"],
        "reference_ligand": manifest_row["reference_ligand"],
        "model": manifest_row["model"],
        "timesteps": int(manifest_row["timesteps"]),
        "record_index": record_index,
        "sdf_file": manifest_row["curated_file"],
        "rdkit_readable": True,
        "sanitize_success": sanitized is not None,
        "read_error": "",
        "smiles": smiles,
        "heavy_atoms": heavy,
        "num_atoms": metric_mol.GetNumAtoms(),
        "num_bonds": metric_mol.GetNumBonds(),
        "bad_bond_lengths": bad,
        "checked_bond_lengths": checked,
        "bad_bond_length_fraction": bad_fraction,
        "mean_bond_length_abs_error": mean_error,
        "num_fragments": fragments,
        "largest_fragment_atom_frac": largest_fragment,
        "hetero_atoms": heavy - carbon,
        "carbon_fraction": carbon / heavy if heavy else 0.0,
        "qed": QED.qed(sanitized) if sanitized is not None else math.nan,
        "sa_score": sascorer.calculateScore(sanitized) if sanitized is not None else math.nan,
        "logp": Crippen.MolLogP(sanitized) if sanitized is not None else math.nan,
        "tpsa": rdMolDescriptors.CalcTPSA(sanitized) if sanitized is not None else math.nan,
        "hba": Lipinski.NumHAcceptors(sanitized) if sanitized is not None else math.nan,
        "hbd": Lipinski.NumHDonors(sanitized) if sanitized is not None else math.nan,
        "rotatable_bonds": Lipinski.NumRotatableBonds(sanitized)
        if sanitized is not None
        else math.nan,
        "mw": Descriptors.MolWt(sanitized) if sanitized is not None else math.nan,
    }
    row.update(fractions)
    row.update(_ring_features(sanitized))
    return row


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_per_molecule_metrics(manifest_rows: list[dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for manifest_row in manifest_rows:
        sdf_path = CURATED_DIR / manifest_row["curated_file"]
        supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
        for record_index, mol in enumerate(supplier):
            if mol is None:
                rows.append(_blank_metric_row(manifest_row, record_index, "rdkit_none"))
            else:
                rows.append(_mol_metric_row(manifest_row, record_index, mol))
    return pd.DataFrame(rows)


def load_docking_scores() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for (target, model), path in DOCKING_SOURCES.items():
        if not path.exists():
            frames.append(
                pd.DataFrame(
                    {
                        "target": [target],
                        "model": [model],
                        "record_index": [-1],
                        "docking_success": [False],
                        "docking_score": [math.nan],
                        "docking_failure_reason": [f"missing_docking_csv:{path}"],
                        "docking_csv": [str(path)],
                    }
                )
            )
            continue
        df = pd.read_csv(path)
        df = df.rename(
            columns={
                "benchmark_index": "record_index",
                "success": "docking_success",
                "score": "docking_score",
                "failure_reason": "docking_failure_reason",
            }
        )
        df["target"] = target
        df["model"] = model
        df["record_index"] = pd.to_numeric(df["record_index"], errors="coerce").astype("Int64")
        df["docking_score"] = pd.to_numeric(df["docking_score"], errors="coerce")
        df["docking_success"] = df["docking_success"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )
        df["docking_csv"] = str(path)
        keep = [
            "target",
            "model",
            "record_index",
            "docking_success",
            "docking_score",
            "docking_failure_reason",
            "docking_csv",
        ]
        frames.append(df[keep])
    return pd.concat(frames, ignore_index=True)


def merge_metrics_and_docking(metrics_df: pd.DataFrame, docking_df: pd.DataFrame) -> pd.DataFrame:
    docking_df = docking_df[docking_df["record_index"] >= 0].copy()
    merged = metrics_df.merge(
        docking_df,
        on=["target", "model", "record_index"],
        how="left",
    )
    merged["docking_success"] = merged["docking_success"].fillna(False).astype(bool)
    merged["docking_failure_reason"] = merged["docking_failure_reason"].fillna("missing")
    merged["positive_docking_score"] = merged["docking_success"] & (
        pd.to_numeric(merged["docking_score"], errors="coerce") > 0
    )
    merged["favorable_docking_score"] = merged["docking_success"] & (
        pd.to_numeric(merged["docking_score"], errors="coerce") < 0
    )
    return merged


def summarize_batches(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (target, model), sub in df.groupby(["target", "model"], sort=False):
        valid = sub[sub["sanitize_success"]].copy()
        docked = sub[sub["docking_success"]].copy()
        favorable = sub[sub["favorable_docking_score"]].copy()
        smiles = valid["smiles"].dropna()
        smiles = smiles[smiles != ""]
        row: dict[str, object] = {
            "target": target,
            "model": model,
            "records": len(sub),
            "rdkit_readable": int(sub["rdkit_readable"].sum()),
            "sanitize_valid": int(sub["sanitize_success"].sum()),
            "sanitize_rate": float(sub["sanitize_success"].mean()),
            "unique_smiles": int(smiles.nunique()),
            "unique_rate_valid": float(smiles.nunique() / len(smiles)) if len(smiles) else math.nan,
            "docking_success": int(sub["docking_success"].sum()),
            "docking_success_rate": float(sub["docking_success"].mean()),
            "positive_docking_scores": int(sub["positive_docking_score"].sum()),
            "positive_docking_score_rate": float(sub["positive_docking_score"].mean()),
            "dock_score_mean": _safe_mean(docked["docking_score"]),
            "dock_score_median": _safe_median(docked["docking_score"]),
            "dock_score_best_min": _safe_quantile(docked["docking_score"], 0.0),
            "dock_score_p10": _safe_quantile(docked["docking_score"], 0.10),
            "dock_score_p90": _safe_quantile(docked["docking_score"], 0.90),
            "favorable_dock_score_median": _safe_median(favorable["docking_score"]),
            "qed_median": _safe_median(valid["qed"]),
            "sa_score_median": _safe_median(valid["sa_score"]),
            "logp_median": _safe_median(valid["logp"]),
            "tpsa_median": _safe_median(valid["tpsa"]),
            "mw_median": _safe_median(valid["mw"]),
            "heavy_atoms_median": _safe_median(valid["heavy_atoms"]),
            "ring_count_median": _safe_median(valid["ring_count"]),
            "aromatic_ring_median": _safe_median(valid["aromatic_ring_count"]),
            "benzene_like_ring_mean": _safe_mean(valid["benzene_like_ring_count"]),
            "benzene_like_ring_median": _safe_median(valid["benzene_like_ring_count"]),
            "ring_complexity_median": _safe_median(valid["ring_complexity"]),
            "fused_ring_pair_median": _safe_median(valid["fused_ring_pair_count"]),
            "max_fused_ring_system_median": _safe_median(valid["max_fused_ring_system_size"]),
            "carbon_fraction_median": _safe_median(valid["carbon_fraction"]),
            "hetero_atoms_median": _safe_median(valid["hetero_atoms"]),
            "multi_fragment_rate": float((sub["num_fragments"] > 1).mean()),
            "has_benzene_like_ring_rate": float(valid["has_benzene_like_ring"].mean())
            if len(valid)
            else math.nan,
            "three_member_ring_rate": float(sub["has_three_member_ring"].mean()),
            "fused_ring_rate": float(sub["has_fused_ring"].mean()),
            "spiro_rate": float(sub["has_spiro"].mean()),
            "bridgehead_rate": float(sub["has_bridgehead"].mean()),
            "macrocycle_rate": float(sub["has_macrocycle"].mean()),
            "high_ring_count_rate": float(sub["high_ring_count"].mean()),
            "complex_ring_rate": float(sub["complex_ring_flag"].mean()),
            "bad_bond_any_rate": float((sub["bad_bond_length_fraction"] > 0).mean()),
        }
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["target"] = pd.Categorical(summary["target"], TARGET_ORDER, ordered=True)
    summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["target", "model"]).reset_index(drop=True)


def write_tables(
    per_molecule: pd.DataFrame,
    batch_summary: pd.DataFrame,
    benzene_comparison: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    per_molecule.to_csv(TABLE_DIR / "per_molecule_metrics_with_docking.csv", index=False)
    batch_summary.to_csv(TABLE_DIR / "batch_summary.csv", index=False)
    compact_cols = [
        "target",
        "model",
        "records",
        "sanitize_rate",
        "unique_rate_valid",
        "docking_success_rate",
        "dock_score_median",
        "dock_score_best_min",
        "qed_median",
        "sa_score_median",
        "logp_median",
        "tpsa_median",
        "ring_count_median",
        "aromatic_ring_median",
        "benzene_like_ring_mean",
        "has_benzene_like_ring_rate",
        "three_member_ring_rate",
        "complex_ring_rate",
        "multi_fragment_rate",
    ]
    batch_summary[compact_cols].to_csv(TABLE_DIR / "compact_summary.csv", index=False)
    ring_cols = [
        "target",
        "model",
        "records",
        "ring_count_median",
        "aromatic_ring_median",
        "benzene_like_ring_mean",
        "benzene_like_ring_median",
        "has_benzene_like_ring_rate",
        "ring_complexity_median",
        "fused_ring_pair_median",
        "max_fused_ring_system_median",
        "three_member_ring_rate",
        "fused_ring_rate",
        "spiro_rate",
        "bridgehead_rate",
        "macrocycle_rate",
        "high_ring_count_rate",
        "complex_ring_rate",
    ]
    batch_summary[ring_cols].to_csv(TABLE_DIR / "ring_risk_summary.csv", index=False)
    benzene_comparison.to_csv(TABLE_DIR / "benzene_ring_comparison.csv", index=False)


def strict_property_pool_mask(props: pd.DataFrame) -> pd.Series:
    property_core = (
        props["heavy_atoms"].between(10, 30)
        & props["mol_wt"].between(150, 500)
        & props["mol_log_p"].between(-1, 6)
        & props["tpsa"].between(20, 140)
        & props["num_h_acceptors"].between(1, 10)
        & props["num_h_donors"].between(0, 5)
        & props["num_rotatable_bonds"].le(10)
        & props["num_heteroatoms"].ge(1)
        & props["carbon_fraction"].between(0.35, 0.90)
    )
    return (
        property_core
        & props["lipinski_violations"].eq(0)
        & props["carbon_fraction"].le(0.88)
        & props["num_heteroatoms"].ge(2)
        & props["num_aromatic_rings"].le(4)
        & props["num_rings"].le(6)
        & props["mol_log_p"].between(0, 5.5)
        & props["tpsa"].between(20, 120)
    )


def _summarize_chembl_benzene_baseline(
    props: pd.DataFrame,
    label: str,
    subset: str,
) -> dict[str, object]:
    benzene_like = (
        props["num_aromatic_rings"] - props["num_aromatic_heterocycles"]
    ).clip(lower=0)
    return {
        "source": "ChEMBL36 properties",
        "target": "ChEMBL36",
        "model": label,
        "subset": subset,
        "records": int(len(props)),
        "has_benzene_like_ring_rate": float((benzene_like > 0).mean()),
        "benzene_like_ring_mean": float(benzene_like.mean()),
        "benzene_like_ring_median": float(benzene_like.median()),
        "benzene_like_ring_p95": float(benzene_like.quantile(0.95)),
        "benzene_definition": "num_aromatic_rings - num_aromatic_heterocycles, clipped at 0",
    }


def compute_chembl_benzene_baselines() -> pd.DataFrame:
    chembl_cols = [
        "heavy_atoms",
        "mol_wt",
        "mol_log_p",
        "tpsa",
        "num_h_acceptors",
        "num_h_donors",
        "num_rotatable_bonds",
        "num_heteroatoms",
        "carbon_fraction",
        "num_rings",
        "num_aromatic_rings",
        "num_aromatic_heterocycles",
        "lipinski_violations",
    ]
    chembl = pd.read_csv(CHEMBL_PROPERTIES, usecols=chembl_cols, compression="gzip")
    strict = chembl.loc[strict_property_pool_mask(chembl)].copy()
    return pd.DataFrame(
        [
            _summarize_chembl_benzene_baseline(chembl, "ChEMBL36 full", "full"),
            _summarize_chembl_benzene_baseline(strict, "ChEMBL36 strict", "strict_property_pool"),
        ]
    )


def build_benzene_comparison(batch_summary: pd.DataFrame) -> pd.DataFrame:
    generated = batch_summary[
        [
            "target",
            "model",
            "sanitize_valid",
            "has_benzene_like_ring_rate",
            "benzene_like_ring_mean",
            "benzene_like_ring_median",
        ]
    ].copy()
    generated = generated.rename(columns={"sanitize_valid": "records"})
    generated["source"] = "Generated benchmark"
    generated["subset"] = "six_target_batches"
    generated["benzene_like_ring_p95"] = math.nan
    generated["benzene_definition"] = "exact 6-member all-carbon aromatic ring from RDKit atom rings"
    generated = generated[
        [
            "source",
            "target",
            "model",
            "subset",
            "records",
            "has_benzene_like_ring_rate",
            "benzene_like_ring_mean",
            "benzene_like_ring_median",
            "benzene_like_ring_p95",
            "benzene_definition",
        ]
    ]
    baselines = compute_chembl_benzene_baselines()
    return pd.concat([generated, baselines], ignore_index=True)


def _prepare_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target"] = pd.Categorical(out["target"], TARGET_ORDER, ordered=True)
    out["model"] = pd.Categorical(out["model"], MODEL_ORDER, ordered=True)
    out["batch"] = out["target"].astype(str) + " / " + out["model"].astype(str)
    return out


def _annotate_violin_medians(ax: plt.Axes, data: pd.DataFrame, value_col: str) -> None:
    offsets = {"DiffSBDD": -0.20, "DiffGui": 0.20}
    for target_idx, target in enumerate(TARGET_ORDER):
        for model in MODEL_ORDER:
            values = pd.to_numeric(
                data[(data["target"] == target) & (data["model"] == model)][value_col],
                errors="coerce",
            ).dropna()
            if values.empty:
                continue
            median_value = float(values.median())
            ax.text(
                target_idx + offsets[model],
                median_value,
                f"{median_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#202020",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.72,
                },
            )


def _plot_rate_bar_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    rate_col: str,
    title: str,
) -> None:
    plot_df = summary.copy()
    plot_df["target"] = pd.Categorical(plot_df["target"], TARGET_ORDER, ordered=True)
    plot_df["model"] = pd.Categorical(plot_df["model"], MODEL_ORDER, ordered=True)
    plot_df["rate_pct"] = plot_df[rate_col] * 100.0
    sns.barplot(
        data=plot_df,
        x="target",
        y="rate_pct",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        ax=ax,
    )
    ax.set_title(title, fontsize=17)
    ax.set_xlabel("")
    ax.set_ylabel("% molecules", fontsize=14)
    ax.set_ylim(0, max(8, float(plot_df["rate_pct"].max()) * 1.25))
    ax.grid(axis="y", alpha=0.2)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=10)
    if ax.legend_:
        ax.legend_.remove()


def plot_docking_violin(df: pd.DataFrame) -> None:
    plot_df = _prepare_plot_df(df[df["favorable_docking_score"]].copy())
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    sns.violinplot(
        data=plot_df,
        x="target",
        y="docking_score",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        inner="quartile",
        cut=0,
        linewidth=0.8,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df.sample(min(len(plot_df), 900), random_state=20260622),
        x="target",
        y="docking_score",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        dodge=True,
        palette=MODEL_PALETTE,
        size=1.5,
        alpha=0.25,
        ax=ax,
        legend=False,
    )
    _annotate_violin_medians(ax, plot_df, "docking_score")
    ax.set_title("UniDock score distribution by target and model")
    ax.set_xlabel("")
    ax.set_ylabel("UniDock vina score, kcal/mol (lower is better)")
    ax.grid(axis="y", alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="", frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "docking_score_violin.png", dpi=220)
    fig.savefig(FIG_DIR / "docking_score_violin.svg")
    plt.close(fig)


def plot_property_violin_grid(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    valid = _prepare_plot_df(df[df["sanitize_success"]].copy())
    props = [
        ("qed", "QED"),
        ("sa_score", "SA score"),
        ("logp", "LogP"),
        ("tpsa", "TPSA"),
        ("mw", "Molecular weight"),
        ("heavy_atoms", "Heavy atoms"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(18, 14.8))
    for ax, (col, title) in zip(axes[:2].flat, props):
        sns.violinplot(
            data=valid,
            x="target",
            y=col,
            hue="model",
            order=TARGET_ORDER,
            hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE,
            inner="quartile",
            cut=0,
            linewidth=0.7,
            ax=ax,
        )
        _annotate_violin_medians(ax, valid, col)
        ax.set_title(title, fontsize=17)
        ax.set_xlabel("")
        ax.set_ylabel(col, fontsize=14)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", labelsize=13)
        ax.tick_params(axis="y", labelsize=13)
        ax.legend_.remove()
    _plot_rate_bar_panel(axes[2, 0], summary, "three_member_ring_rate", "Three-member ring rate")
    _plot_rate_bar_panel(axes[2, 1], summary, "complex_ring_rate", "Complex ring rate")
    axes[2, 2].axis("off")
    axes[2, 2].text(
        0.02,
        0.92,
        "Complex ring definition",
        fontsize=18,
        fontweight="bold",
        ha="left",
        va="top",
    )
    axes[2, 2].text(
        0.02,
        0.76,
        "A molecule is flagged if it has any of:\n"
        "- fused ring pair\n"
        "- spiro atom\n"
        "- bridgehead atom\n"
        "- macrocycle (ring size >= 8)\n"
        "- high ring count (ring_count >= 6)\n\n"
        "Numbers on violins are medians.",
        fontsize=14,
        ha="left",
        va="top",
        linespacing=1.45,
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
        fontsize=15,
    )
    fig.suptitle("Molecular property distributions with ring-risk rates", y=0.995, fontsize=22)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.6, w_pad=2.0)
    fig.savefig(FIG_DIR / "property_violin_grid.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "property_violin_grid.svg", bbox_inches="tight")
    plt.close(fig)


def plot_structure_violin_grid(df: pd.DataFrame) -> None:
    valid = _prepare_plot_df(df[df["sanitize_success"]].copy())
    props = [
        ("ring_count", "Ring count"),
        ("aromatic_ring_count", "Aromatic rings"),
        ("ring_complexity", "Ring complexity"),
        ("carbon_fraction", "Carbon fraction"),
        ("hetero_atoms", "Hetero atoms"),
        ("num_fragments", "Fragments"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.8))
    for ax, (col, title) in zip(axes.flat, props):
        sns.violinplot(
            data=valid,
            x="target",
            y=col,
            hue="model",
            order=TARGET_ORDER,
            hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE,
            inner="quartile",
            cut=0,
            linewidth=0.7,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.grid(axis="y", alpha=0.2)
        ax.legend_.remove()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.suptitle("Structural distributions", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG_DIR / "structure_violin_grid.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "structure_violin_grid.svg", bbox_inches="tight")
    plt.close(fig)


def plot_quality_heatmap(summary: pd.DataFrame) -> None:
    heat_cols = [
        "sanitize_rate",
        "unique_rate_valid",
        "docking_success_rate",
        "positive_docking_score_rate",
        "multi_fragment_rate",
        "three_member_ring_rate",
        "complex_ring_rate",
        "bad_bond_any_rate",
    ]
    labels = {
        "sanitize_rate": "sanitize",
        "unique_rate_valid": "unique",
        "docking_success_rate": "dock ok",
        "positive_docking_score_rate": "dock > 0",
        "multi_fragment_rate": "multi-frag",
        "three_member_ring_rate": "3-ring",
        "complex_ring_rate": "complex ring",
        "bad_bond_any_rate": "bad bond",
    }
    mat = summary.copy()
    mat["batch"] = mat["target"].astype(str) + "\n" + mat["model"].astype(str)
    mat = mat.set_index("batch")[heat_cols].rename(columns=labels)
    fig, ax = plt.subplots(figsize=(9, 5.4))
    sns.heatmap(
        mat,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "rate"},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Quality and structural warning rates")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "quality_rate_heatmap.png", dpi=220)
    fig.savefig(FIG_DIR / "quality_rate_heatmap.svg")
    plt.close(fig)


def plot_benzene_comparison(benzene_comparison: pd.DataFrame) -> None:
    plot_df = benzene_comparison.copy()
    plot_df["display_label"] = np.where(
        plot_df["source"].eq("Generated benchmark"),
        plot_df["target"].astype(str) + " / " + plot_df["model"].astype(str),
        plot_df["model"].astype(str),
    )
    plot_df["color"] = plot_df["model"].map(MODEL_PALETTE).fillna("#6B7280")
    plot_df["benzene_rate_pct"] = plot_df["has_benzene_like_ring_rate"] * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), sharey=True)
    y = np.arange(len(plot_df))
    panels = [
        ("benzene_rate_pct", "Has benzene-like ring", "% molecules", 100.0, "%.1f"),
        ("benzene_like_ring_mean", "Mean benzene-like rings", "rings / molecule", None, "%.2f"),
    ]
    for ax, (col, title, xlabel, xlim, fmt) in zip(axes, panels):
        values = pd.to_numeric(plot_df[col], errors="coerce").fillna(0.0)
        ax.barh(y, values, color=plot_df["color"], alpha=0.9)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.grid(axis="x", alpha=0.22)
        ax.tick_params(axis="x", labelsize=13)
        if xlim is not None:
            ax.set_xlim(0, xlim)
            label_pad = 1.0
        else:
            upper = max(0.2, float(values.max()) * 1.22)
            ax.set_xlim(0, upper)
            label_pad = upper * 0.015
        for yi, value in zip(y, values):
            ax.text(
                value + label_pad,
                yi,
                fmt % value,
                va="center",
                ha="left",
                fontsize=12,
                fontweight="bold",
            )
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(plot_df["display_label"], fontsize=12)
    axes[0].invert_yaxis()
    fig.suptitle("Benzene-like ring comparison vs ChEMBL36", fontsize=22, y=0.99)
    fig.text(
        0.5,
        0.015,
        "Generated: exact 6-member all-carbon aromatic ring. "
        "ChEMBL36: approximated as num_aromatic_rings - num_aromatic_heterocycles.",
        ha="center",
        fontsize=12,
        color="#404040",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), w_pad=2.4)
    fig.savefig(FIG_DIR / "benzene_ring_chembl_comparison.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "benzene_ring_chembl_comparison.svg", bbox_inches="tight")
    plt.close(fig)


def plot_docking_success_bar(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    plot_df["target"] = pd.Categorical(plot_df["target"], TARGET_ORDER, ordered=True)
    plot_df["model"] = pd.Categorical(plot_df["model"], MODEL_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=plot_df,
        x="target",
        y="docking_success_rate",
        hue="model",
        order=TARGET_ORDER,
        hue_order=MODEL_ORDER,
        palette=MODEL_PALETTE,
        ax=ax,
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("Docking success rate")
    ax.set_title("Docking success by target and model")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=2, fontsize=8)
    ax.legend(title="", frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "docking_success_rate_bar.png", dpi=220)
    fig.savefig(FIG_DIR / "docking_success_rate_bar.svg")
    plt.close(fig)


def plot_docking_vs_qed(df: pd.DataFrame) -> None:
    plot_df = _prepare_plot_df(df[df["favorable_docking_score"] & df["sanitize_success"]].copy())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), sharey=True)
    for ax, target in zip(axes, TARGET_ORDER):
        sub = plot_df[plot_df["target"] == target]
        sns.scatterplot(
            data=sub,
            x="qed",
            y="docking_score",
            hue="model",
            hue_order=MODEL_ORDER,
            palette=MODEL_PALETTE,
            s=16,
            alpha=0.45,
            linewidth=0,
            ax=ax,
        )
        ax.set_title(target)
        ax.set_xlabel("QED")
        ax.grid(alpha=0.2)
        if ax is not axes[0]:
            ax.set_ylabel("")
        else:
            ax.set_ylabel("UniDock score")
        ax.legend_.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Docking score vs QED", y=1.03, fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "docking_vs_qed_scatter.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "docking_vs_qed_scatter.svg", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    per_molecule: pd.DataFrame,
    summary: pd.DataFrame,
    benzene_comparison: pd.DataFrame,
) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    plot_docking_violin(per_molecule)
    plot_property_violin_grid(per_molecule, summary)
    plot_structure_violin_grid(per_molecule)
    plot_quality_heatmap(summary)
    plot_benzene_comparison(benzene_comparison)
    plot_docking_success_bar(summary)
    plot_docking_vs_qed(per_molecule)


def _format_value(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, sep]
    for _, row in df[columns].iterrows():
        lines.append("| " + " | ".join(_format_value(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame, benzene_comparison: pd.DataFrame) -> None:
    report = SCRIPT_DIR / "report.md"
    compact = summary.copy()
    percent_cols = [
        "sanitize_rate",
        "unique_rate_valid",
        "docking_success_rate",
        "positive_docking_score_rate",
        "multi_fragment_rate",
        "has_benzene_like_ring_rate",
        "three_member_ring_rate",
        "complex_ring_rate",
        "bad_bond_any_rate",
    ]
    for col in percent_cols:
        compact[col] = compact[col] * 100
    table_cols = [
        "target",
        "model",
        "records",
        "sanitize_rate",
        "docking_success_rate",
        "dock_score_median",
        "dock_score_best_min",
        "qed_median",
        "sa_score_median",
        "logp_median",
        "tpsa_median",
        "ring_count_median",
        "benzene_like_ring_mean",
        "has_benzene_like_ring_rate",
        "three_member_ring_rate",
        "complex_ring_rate",
    ]
    benzene_view = benzene_comparison.copy()
    benzene_view["group"] = np.where(
        benzene_view["source"].eq("Generated benchmark"),
        benzene_view["target"].astype(str) + " / " + benzene_view["model"].astype(str),
        benzene_view["model"].astype(str),
    )
    benzene_view["has_benzene_like_ring_pct"] = (
        benzene_view["has_benzene_like_ring_rate"] * 100.0
    )
    benzene_cols = [
        "group",
        "records",
        "has_benzene_like_ring_pct",
        "benzene_like_ring_mean",
        "benzene_like_ring_median",
    ]
    best_rows = []
    for target in TARGET_ORDER:
        sub = summary[summary["target"].astype(str) == target]
        if sub.empty:
            continue
        best = sub.sort_values("dock_score_median").iloc[0]
        best_rows.append(
            f"- {target}: median docking 更低的是 `{best['model']}` ({best['dock_score_median']:.3f})."
        )
    text = f"""# DiffGui / DiffSBDD 三靶点分子统计

本报告比较 `../` 目录下整理出的 6 批 500-molecule 产物。

对接分数为 UniDock vina score，数值越低代表预测结合越强。对接小提琴图只画成功且有利的分数（`score < 0`），避免少数正分离群值把主分布压扁；完整分数仍保留在 CSV 表里。

复杂环率定义：分子只要出现 fused ring pair、spiro atom、bridgehead atom、macrocycle（ring size >= 8），或 `ring_count >= 6`，即计为 complex ring。

苯环判定：生成分子用 RDKit atom rings 精确计数 6 元全碳芳香环；ChEMBL36 大数据集基线复用现有 properties 表，用 `num_aromatic_rings - num_aromatic_heterocycles` 近似 benzene-like rings。

## 汇总表

表中 rate 均为百分比。

{_markdown_table(compact, table_cols)}

## 苯环与 ChEMBL36 基线

`has_benzene_like_ring_pct` 是含至少一个苯环/苯并环的分子比例；`benzene_like_ring_mean` 是平均每个分子的苯环数。

{_markdown_table(benzene_view, benzene_cols)}

## 快速结论

{chr(10).join(best_rows)}
- 5XZR/8J6 复用已有 benchmark docking 结果，其中 DiffGui 和 DiffSBDD 都有一个 UniDock batch 失败，所以两者 docking success rate 都是 74.4%。
- 9JH3/R:401 是本次新补跑 docking，receptor 为 `9JH3_pocket8A.pdbqt`，box 为以参考配体中心为中心的 20 A 立方盒。
- 9JH3 DiffGui 聚合 SDF 原先因标题空行被去掉导致 RDKit 不能读取，本次已从原始 500 个单分子 SDF 重新规范化聚合。

## 图

![Docking score violin](figures/docking_score_violin.png)

![Property violin grid](figures/property_violin_grid.png)

![Structure violin grid](figures/structure_violin_grid.png)

![Quality heatmap](figures/quality_rate_heatmap.png)

![Benzene ring comparison](figures/benzene_ring_chembl_comparison.png)

![Docking success bar](figures/docking_success_rate_bar.png)

![Docking vs QED](figures/docking_vs_qed_scatter.png)

## 输出文件

- `tables/per_molecule_metrics_with_docking.csv`
- `tables/batch_summary.csv`
- `tables/compact_summary.csv`
- `tables/ring_risk_summary.csv`
- `tables/benzene_ring_comparison.csv`
- `figures/*.png` and `figures/*.svg`
- `docking/9JH3_R401/`：本次新生成的 9JH3 UniDock 分数和日志
"""
    report.write_text(text, encoding="utf-8")


def main() -> None:
    manifest_rows = read_manifest()
    metrics = compute_per_molecule_metrics(manifest_rows)
    docking = load_docking_scores()
    merged = merge_metrics_and_docking(metrics, docking)
    summary = summarize_batches(merged)
    benzene_comparison = build_benzene_comparison(summary)
    write_tables(merged, summary, benzene_comparison)
    make_plots(merged, summary, benzene_comparison)
    write_report(summary, benzene_comparison)
    print(f"[analysis] molecules={len(merged)} batches={len(summary)}")
    print(f"[analysis] wrote {TABLE_DIR}")
    print(f"[analysis] wrote {FIG_DIR}")
    print(f"[analysis] wrote {SCRIPT_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
