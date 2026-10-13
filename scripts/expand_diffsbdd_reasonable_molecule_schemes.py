#!/usr/bin/env python3
"""Compute extra structure/drug-likeness facts and count strict candidate schemes."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

SCHEMES = {
    "structure_strict": "normality_pre_chembl + PAINS=0 + NIH=0 + ZINC=0",
    "drug_strict": "QED>=0.5 + SA<=5 + LogP 0-4 + TPSA 30-120 + MW 150-500",
    "drug_very_strict": "QED>=0.6 + SA<=4.5 + LogP 0-4 + TPSA 40-100 + MW 200-500",
    "3d_reasonable": "structure strict + rotatable<=8 + Fsp3>=0.25 + aromatic rings<=4 + stereo<=4",
    "balanced_reasonable": "drug strict + 3d_reasonable",
    "normality_chmbl_nearest_002": "balanced_reasonable + nearest ChEMBL UMAP<=0.02",
    "normality_chmbl_nearest_003": "balanced_reasonable + nearest ChEMBL UMAP<=0.03",
    "normality_chmbl_k20_010": "balanced_reasonable + ChEMBL k20 UMAP<=0.10",
    "strict_chmbl_nearest_002": "drug very strict + structure strict + nearest ChEMBL UMAP<=0.02",
    "strict_chmbl_k20_010": "drug very strict + 3d_reasonable + ChEMBL k20 UMAP<=0.10",
    "scaffold_friendly": "normality + QED>=0.5 + SA<=5 + rotatable<=10 + Fsp3>=0.20 + aromatic rings<=5",
    "compact_druglike": "normality + MW 180-450 + heavy atoms 12-30 + LogP 0-4 + TPSA 30-110 + QED>=0.5",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--umap-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=28)
    return parser.parse_args()


def catalog_counts(molecule: Chem.Mol) -> tuple[int, int, int]:
    """Return PAINS, NIH, and ZINC alert counts."""
    counts = []
    for catalog_name in ("PAINS", "NIH", "ZINC"):
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, catalog_name))
        counts.append(len(FilterCatalog.FilterCatalog(params).GetMatches(molecule)))
    return tuple(counts)


def descriptor_task(task: tuple[str, str]) -> dict[str, object]:
    trajectory_id, smiles = task
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(trajectory_id)
    pains, nih, zinc = catalog_counts(mol)
    heavy = max(1, mol.GetNumHeavyAtoms())
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    return {
        "trajectory_id": trajectory_id,
        "pains_count": pains,
        "nih_count": nih,
        "zinc_count": zinc,
        "rotatable_bond_count": int(Lipinski.NumRotatableBonds(mol)),
        "total_ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        "aromatic_ring_count": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "aromatic_atom_fraction": float(aromatic_atoms / heavy),
        "stereocenter_count": int(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False))),
        "hetero_atom_count": int(sum(atom.GetAtomicNum() not in (1, 6) for atom in mol.GetAtoms())),
    }


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    facts = pd.read_csv(args.facts, low_memory=False)
    metrics = pd.read_csv(args.umap_metrics, low_memory=False)
    frame = facts.merge(metrics, on=["trajectory_id", "canonical_smiles", "umap1", "umap2"], validate="one_to_one")
    if len(frame) != 67_694:
        raise ValueError(f"expected 67694 valid rows, got {len(frame)}")
    tasks = list(zip(frame.trajectory_id.astype(str), frame.canonical_smiles.astype(str)))
    with mp.get_context("spawn").Pool(args.workers) as pool:
        extra = pd.DataFrame(pool.map(descriptor_task, tasks, chunksize=128))
    frame = frame.merge(extra, on="trajectory_id", validate="one_to_one")
    frame["structure_strict"] = frame.normality_pre_chembl & frame.pains_count.eq(0) & frame.nih_count.eq(0) & frame.zinc_count.eq(0)
    frame["drug_strict"] = frame.qed.ge(.5) & frame.sa_raw.le(5) & frame.logp.between(0,4) & frame.tpsa.between(30,120) & frame.mol_wt.between(150,500)
    frame["drug_very_strict"] = frame.qed.ge(.6) & frame.sa_raw.le(4.5) & frame.logp.between(0,4) & frame.tpsa.between(40,100) & frame.mol_wt.between(200,500)
    frame["3d_reasonable"] = frame.structure_strict & frame.rotatable_bond_count.le(8) & frame.fraction_csp3.ge(.25) & frame.aromatic_ring_count.le(4) & frame.stereocenter_count.le(4)
    frame["balanced_reasonable"] = frame.drug_strict & frame["3d_reasonable"]
    frame["normality_chmbl_nearest_002"] = frame.balanced_reasonable & frame.umap_nearest_chembl_distance.le(.02)
    frame["normality_chmbl_nearest_003"] = frame.balanced_reasonable & frame.umap_nearest_chembl_distance.le(.03)
    frame["normality_chmbl_k20_010"] = frame.balanced_reasonable & frame.umap_chembl_k20_distance.le(.10)
    frame["strict_chmbl_nearest_002"] = frame.drug_very_strict & frame.structure_strict & frame.umap_nearest_chembl_distance.le(.02)
    frame["strict_chmbl_k20_010"] = frame.drug_very_strict & frame["3d_reasonable"] & frame.umap_chembl_k20_distance.le(.10)
    frame["scaffold_friendly"] = frame.normality_pre_chembl & frame.qed.ge(.5) & frame.sa_raw.le(5) & frame.rotatable_bond_count.le(10) & frame.fraction_csp3.ge(.20) & frame.aromatic_ring_count.le(5)
    frame["compact_druglike"] = frame.normality_pre_chembl & frame.qed.ge(.5) & frame.mol_wt.between(180,450) & frame.heavy_atom_count.between(12,30) & frame.logp.between(0,4) & frame.tpsa.between(30,110)
    rows = []
    for scheme_id, description in SCHEMES.items():
        mask = frame[scheme_id].astype(bool)
        rows.append({"scheme_id": scheme_id, "description": description, "pass_count": int(mask.sum()), "rate_valid": float(mask.mean()), "rate_attempts": float(mask.sum()/100000)})
    frame.to_csv(args.output_dir / "expanded_reasonableness_facts.csv.gz", index=False, compression="gzip")
    pd.DataFrame(rows).to_csv(args.output_dir / "expanded_scheme_counts.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps({"status": "complete", "input_valid": len(frame), "workers": args.workers, "schemes": rows, "note": "No UMAP fit or transform was performed; existing coordinates were used only as optional screening metrics."}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
