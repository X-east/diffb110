#!/usr/bin/env python3
"""
Evaluate DiffSBDD generated molecules with GuacaMol-style metrics.

Metrics:
    - Validity: fraction of molecules that RDKit can sanitize
    - Uniqueness: fraction of unique SMILES among valid molecules  
    - Novelty: fraction of unique SMILES not in training set (requires --ref_smiles)
    - Diversity: 1 - mean pairwise Tanimoto similarity (Morgan fp, sampled if >1000)
    - Properties: LogP, QED, SA, MW, TPSA, HBD, HBA, RotBonds, Scaffold
    - KL Divergence: property distribution vs reference (requires --ref_properties)

Usage:
    # Basic evaluation (no reference needed)
    python evaluate_with_guacamol.py --sdf generated.sdf --outdir ./eval

    # With reference SMILES for novelty
    python evaluate_with_guacamol.py --sdf generated.sdf --outdir ./eval \
        --ref_smiles train_smiles.txt

    # With reference properties for KL divergence
    python evaluate_with_guacamol.py --sdf generated.sdf --outdir ./eval \
        --ref_smiles train_smiles.txt \
        --ref_properties train_properties.csv
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

# GuacaMol
try:
    from guacamol.utils.chemistry import canonicalize_list
    HAS_GUACAMOL = True
except ImportError:
    HAS_GUACAMOL = False

# SA Score
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
SASCORER_PATH = DIFFSBDD_ROOT / "analysis" / "SA_Score" / "sascorer.py"
sys.path.insert(0, str(DIFFSBDD_ROOT / "analysis"))
if SASCORER_PATH.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("sascorer", SASCORER_PATH)
    sascorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sascorer)
    calculateScore = sascorer.calculateScore
else:
    calculateScore = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def sdf_to_mols(sdf_path):
    """Read SDF and return list of RDKit mols (None for failed)."""
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)
    return [mol for mol in supplier]


def try_sanitize(mol):
    """Try to sanitize a mol. Return sanitized mol or None."""
    if mol is None:
        return None
    try:
        mol_copy = Chem.Mol(mol)
        Chem.SanitizeMol(mol_copy)
        return mol_copy
    except Exception:
        return None


def mol_to_canonical_smiles(mol):
    """Convert mol to canonical SMILES (no isomerism, no Hs)."""
    if mol is None:
        return None
    try:
        mol = Chem.RemoveHs(mol)
        Chem.RemoveStereochemistry(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def compute_validity(mols):
    """Return (valid_mols, valid_smiles, validity_ratio)."""
    valid_mols = []
    valid_smiles = []
    for mol in mols:
        san = try_sanitize(mol)
        if san is not None:
            valid_mols.append(san)
            smi = mol_to_canonical_smiles(san)
            if smi is not None:
                valid_smiles.append(smi)
    validity = len(valid_mols) / len(mols) if mols else 0.0
    return valid_mols, valid_smiles, validity


def compute_uniqueness(smiles_list):
    """Return (unique_smiles, uniqueness_ratio)."""
    if not smiles_list:
        return [], 0.0
    unique = list(set(smiles_list))
    uniqueness = len(unique) / len(smiles_list)
    return unique, uniqueness


def compute_novelty(unique_gen_smiles, ref_smiles_set):
    """Return (novel_smiles, novelty_ratio). None if no reference."""
    if ref_smiles_set is None:
        return None, None
    if not unique_gen_smiles:
        return [], 0.0
    novel = [s for s in unique_gen_smiles if s not in ref_smiles_set]
    novelty = len(novel) / len(unique_gen_smiles)
    return novel, novelty


def compute_diversity(smiles_list, sample_size=1000, seed=42):
    """
    Compute internal diversity = 1 - mean pairwise Tanimoto similarity.
    Uses Morgan fingerprints (radius=2, nBits=2048).
    If len > sample_size, randomly sample.
    """
    if len(smiles_list) < 2:
        return 0.0, 0

    rng = random.Random(seed)
    if len(smiles_list) > sample_size:
        selected = rng.sample(smiles_list, sample_size)
    else:
        selected = smiles_list

    fps = []
    for smi in selected:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))

    if len(fps) < 2:
        return 0.0, len(fps)

    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            similarities.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))

    mean_sim = np.mean(similarities) if similarities else 0.0
    diversity = 1.0 - mean_sim
    return diversity, len(fps)


def compute_properties(smiles_list):
    """Compute molecular properties for a list of SMILES."""
    records = []
    for smi in tqdm(smiles_list, desc="Computing properties"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        except Exception:
            scaffold = ""

        sa = calculateScore(mol) if calculateScore else None

        records.append({
            "smiles": smi,
            "MolWt": round(Descriptors.MolWt(mol), 4),
            "ExactMolWt": round(Descriptors.ExactMolWt(mol), 4),
            "LogP": round(Crippen.MolLogP(mol), 4),
            "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
            "HBA": int(Lipinski.NumHAcceptors(mol)),
            "HBD": int(Lipinski.NumHDonors(mol)),
            "RotBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
            "NumRings": int(rdMolDescriptors.CalcNumRings(mol)),
            "NumAromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            "QED": round(QED.qed(mol), 4),
            "SA_Score": round(sa, 2) if sa is not None else None,
            "Murcko_Scaffold": scaffold,
        })
    return pd.DataFrame(records)


def kl_divergence(p, q, eps=1e-10):
    """KL divergence between two discrete distributions."""
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return np.sum(p * np.log(p / q))


def compute_kl_for_property(gen_values, ref_values, bins=50):
    """Compute KL divergence for a continuous property using histograms."""
    gen_values = np.array(gen_values, dtype=float)
    ref_values = np.array(ref_values, dtype=float)
    min_val = min(gen_values.min(), ref_values.min())
    max_val = max(gen_values.max(), ref_values.max())
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    gen_hist, _ = np.histogram(gen_values, bins=bin_edges)
    ref_hist, _ = np.histogram(ref_values, bins=bin_edges)

    return kl_divergence(gen_hist, ref_hist)


def load_reference_smiles(path):
    """Load reference SMILES from a text file (one per line)."""
    with open(path, 'r') as f:
        smiles = [line.strip() for line in f if line.strip()]
    if HAS_GUACAMOL:
        smiles = canonicalize_list(smiles, include_stereocenters=False)
    else:
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
                  for s in smiles if Chem.MolFromSmiles(s)]
    return set(smiles)


def load_reference_properties(path):
    """Load reference properties from CSV or compute from SDF."""
    path = Path(path)
    if path.suffix == '.csv':
        return pd.read_csv(path)
    elif path.suffix == '.sdf':
        mols = sdf_to_mols(path)
        valid_mols, valid_smiles, _ = compute_validity(mols)
        return compute_properties(valid_smiles)
    else:
        raise ValueError(f"Unsupported reference file format: {path.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DiffSBDD generated molecules with GuacaMol-style metrics"
    )
    parser.add_argument("--sdf", type=Path, required=True, help="Generated SDF file")
    parser.add_argument("--outdir", type=Path, default=Path("./eval_results"),
                        help="Output directory")
    parser.add_argument("--ref_smiles", type=Path, default=None,
                        help="Reference SMILES file (one per line) for novelty")
    parser.add_argument("--ref_properties", type=Path, default=None,
                        help="Reference CSV/SDF for KL divergence")
    parser.add_argument("--sample_size", type=int, default=1000,
                        help="Sample size for diversity calculation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load
    logger.info(f"Loading generated molecules from {args.sdf}")
    mols = sdf_to_mols(args.sdf)
    logger.info(f"Total molecules in SDF: {len(mols)}")

    # Step 2: Validity
    valid_mols, valid_smiles, validity = compute_validity(mols)
    logger.info(f"Validity: {validity * 100:.2f}% ({len(valid_mols)}/{len(mols)})")

    # Step 3: Uniqueness
    unique_smiles, uniqueness = compute_uniqueness(valid_smiles)
    logger.info(f"Uniqueness: {uniqueness * 100:.2f}% ({len(unique_smiles)}/{len(valid_smiles)})")

    # Step 4: Novelty
    ref_smiles_set = None
    if args.ref_smiles:
        logger.info(f"Loading reference SMILES from {args.ref_smiles}")
        ref_smiles_set = load_reference_smiles(args.ref_smiles)
        logger.info(f"Reference SMILES loaded: {len(ref_smiles_set)}")
    novel_smiles, novelty = compute_novelty(unique_smiles, ref_smiles_set)
    if novelty is not None:
        logger.info(f"Novelty: {novelty * 100:.2f}% ({len(novel_smiles)}/{len(unique_smiles)})")

    # Step 5: Diversity
    diversity, div_n = compute_diversity(unique_smiles, sample_size=args.sample_size, seed=args.seed)
    logger.info(f"Diversity (sampled n={div_n}): {diversity:.4f}")

    # Step 6: Properties
    logger.info("Computing molecular properties...")
    gen_df = compute_properties(unique_smiles)
    gen_csv = args.outdir / f"{args.sdf.stem}_properties.csv"
    gen_df.to_csv(gen_csv, index=False)
    logger.info(f"Properties saved to {gen_csv}")

    prop_summary = {}
    for col in ["MolWt", "LogP", "TPSA", "QED", "SA_Score", "RotBonds"]:
        if col in gen_df.columns and len(gen_df[col].dropna()) > 0:
            prop_summary[col] = {
                "mean": float(gen_df[col].mean()),
                "std": float(gen_df[col].std()),
                "median": float(gen_df[col].median()),
            }

    # Step 7: Scaffold
    scaffold_counts = Counter(gen_df["Murcko_Scaffold"].dropna())
    scaffold_summary = {
        "total_unique_scaffolds": len(scaffold_counts),
        "scaffold_uniqueness": round(len(scaffold_counts) / len(gen_df), 4) if len(gen_df) > 0 else 0.0,
        "top_10_scaffolds": scaffold_counts.most_common(10),
    }

    # Step 8: KL Divergence
    kl_results = {}
    if args.ref_properties:
        logger.info(f"Loading reference properties from {args.ref_properties}")
        ref_df = load_reference_properties(args.ref_properties)
        ref_csv = args.outdir / f"{args.sdf.stem}_ref_properties.csv"
        ref_df.to_csv(ref_csv, index=False)

        for col in ["MolWt", "LogP", "TPSA", "QED", "SA_Score", "RotBonds"]:
            if col in gen_df.columns and col in ref_df.columns:
                g = gen_df[col].dropna().values
                r = ref_df[col].dropna().values
                if len(g) > 0 and len(r) > 0:
                    kl = compute_kl_for_property(g, r)
                    kl_results[col] = float(kl)
                    logger.info(f"KL Divergence ({col}): {kl:.4f}")

    # Final report
    report = {
        "input_sdf": str(args.sdf),
        "total_molecules": len(mols),
        "valid_molecules": len(valid_mols),
        "validity": round(validity, 4),
        "unique_molecules": len(unique_smiles),
        "uniqueness": round(uniqueness, 4),
        "novel_molecules": len(novel_smiles) if novel_smiles is not None else None,
        "novelty": round(novelty, 4) if novelty is not None else None,
        "diversity": round(diversity, 4),
        "diversity_sample_size": div_n,
        "property_summary": prop_summary,
        "scaffold_summary": scaffold_summary,
        "kl_divergence": kl_results,
    }

    report_path = args.outdir / f"{args.sdf.stem}_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 60)
    logger.info(f"Validity:   {validity * 100:.2f}%")
    logger.info(f"Uniqueness: {uniqueness * 100:.2f}%")
    if novelty is not None:
        logger.info(f"Novelty:    {novelty * 100:.2f}%")
    else:
        logger.info(f"Novelty:    N/A (provide --ref_smiles)")
    logger.info(f"Diversity:  {diversity:.4f}")
    for col, v in prop_summary.items():
        logger.info(f"{col}: {v['mean']:.3f} ± {v['std']:.3f}")
    for col, v in kl_results.items():
        logger.info(f"KL({col}): {v:.4f}")
    logger.info(f"Unique scaffolds: {scaffold_summary['total_unique_scaffolds']}")
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
