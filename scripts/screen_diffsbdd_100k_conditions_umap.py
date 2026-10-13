#!/usr/bin/env python3
"""Screen frozen DiffSBDD trajectories with available actual-good chemistry gates.

The formal actual-good rule also requires docking, ligand efficiency, and
PoseBusters validity. Those three full-dataset labels do not yet exist, so this
script deliberately reports a chemistry-only partial gate rather than calling
it a formal positive label.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffb110_screen_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, FilterCatalog, QED, rdMolDescriptors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DiffSBDD"))
from analysis.SA_Score.sascorer import calculateScore as calculate_sa_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ATTEMPTS = 100_000
CHEMBL_COLOR = "#A8ADB3"
ALL_GENERATED_COLOR = "#D95F59"
PASSED_COLOR = "#16803C"
ALLOWED_ORGANIC_ELEMENTS = frozenset({"C", "N", "O", "S", "P", "F", "Cl", "Br", "I"})


def parse_args() -> argparse.Namespace:
    """Parse input tables and output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-manifest", type=Path, required=True)
    parser.add_argument("--projected-coordinates", type=Path, required=True)
    parser.add_argument("--reference-coordinates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    return parser.parse_args()


def _largest_fused_system(atom_rings: list[tuple[int, ...]]) -> int:
    """Return the number of rings in the largest fused ring system."""
    if not atom_rings:
        return 0
    ring_sets = [set(ring) for ring in atom_rings]
    adjacency = [set() for _ in ring_sets]
    for left_index, left in enumerate(ring_sets):
        for right_index in range(left_index + 1, len(ring_sets)):
            if len(left & ring_sets[right_index]) >= 2:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)
    largest = 1
    seen: set[int] = set()
    for start in range(len(ring_sets)):
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            index = stack.pop()
            size += 1
            for neighbor in adjacency[index]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        largest = max(largest, size)
    return largest


def _descriptor_row(task: tuple[str, str]) -> dict[str, Any]:
    """Calculate deterministic chemistry facts for one canonical SMILES string."""
    trajectory_id, smiles = task
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"cannot parse valid canonical SMILES for {trajectory_id}")
    atom_rings = [tuple(ring) for ring in molecule.GetRingInfo().AtomRings()]
    ring_sets = [set(ring) for ring in atom_rings]
    fused_members: set[int] = set()
    for left_index, left in enumerate(ring_sets):
        for right_index in range(left_index + 1, len(ring_sets)):
            if len(left & ring_sets[right_index]) >= 2:
                fused_members.update((left_index, right_index))
    catalog_params = FilterCatalog.FilterCatalogParams()
    catalog_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    brenk_count = len(FilterCatalog.FilterCatalog(catalog_params).GetMatches(molecule))
    max_fused_system = _largest_fused_system(atom_rings)
    complex_ring_count = int(max_fused_system >= 3)
    complex_ring_count += int(rdMolDescriptors.CalcNumSpiroAtoms(molecule) > 0)
    complex_ring_count += int(rdMolDescriptors.CalcNumBridgeheadAtoms(molecule) > 2)
    hba = int(rdMolDescriptors.CalcNumHBA(molecule))
    hbd = int(rdMolDescriptors.CalcNumHBD(molecule))
    mol_wt = float(Descriptors.MolWt(molecule))
    logp = float(Descriptors.MolLogP(molecule))
    lipinski_violations = int(mol_wt > 500.0) + int(logp > 5.0) + int(hba > 10) + int(hbd > 5)
    heavy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    heavy_atom_count = len(heavy_atoms)
    carbon_fraction = sum(atom.GetSymbol() == "C" for atom in heavy_atoms) / heavy_atom_count
    return {
        "trajectory_id": trajectory_id,
        "qed": float(QED.qed(molecule)),
        "sa_raw": float(calculate_sa_score(molecule)),
        "logp": logp,
        "tpsa": float(rdMolDescriptors.CalcTPSA(molecule)),
        "mol_wt": mol_wt,
        "heavy_atom_count": heavy_atom_count,
        "hba": hba,
        "hbd": hbd,
        "formal_charge": int(Chem.GetFormalCharge(molecule)),
        "fragment_count": len(Chem.GetMolFrags(molecule, sanitizeFrags=False)),
        "carbon_fraction": carbon_fraction,
        "allowed_organic_elements_only": all(atom.GetSymbol() in ALLOWED_ORGANIC_ELEMENTS for atom in heavy_atoms),
        "lipinski_violation_count": lipinski_violations,
        "three_member_ring_count": sum(1 for ring in atom_rings if len(ring) == 3),
        "fused_ring_count": len(fused_members),
        "max_fused_ring_system_size": max_fused_system,
        "complex_ring_count": complex_ring_count,
        "brenk_count": brenk_count,
    }


def calculate_descriptors(valid: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Calculate descriptor facts in parallel while retaining trajectory identity."""
    tasks = list(zip(valid["trajectory_id"].astype(str), valid["canonical_smiles"].astype(str)))
    context = mp.get_context("spawn")
    with context.Pool(processes=workers) as pool:
        rows = pool.map(_descriptor_row, tasks, chunksize=128)
    output = pd.DataFrame(rows)
    if len(output) != len(valid) or output["trajectory_id"].duplicated().any():
        raise ValueError("descriptor output lost or duplicated trajectory IDs")
    return output


def add_pass_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Apply the pre-existing actual-good thresholds that need no docking backend."""
    filters = {
        "qed_ge_0_4": "QED >= 0.4",
        "sa_le_6": "SA <= 6",
        "logp_in_neg1_to_5": "-1 <= LogP <= 5",
        "tpsa_in_20_to_140": "20 <= TPSA <= 140",
        "three_member_rings_eq_0": "3-member rings = 0",
        "fused_rings_le_4": "Fused rings <= 4",
        "complex_rings_le_1": "Complex rings <= 1",
        "brenk_eq_0": "Brenk alerts = 0",
    }
    output = frame.copy()
    output["qed_ge_0_4"] = output["qed"].ge(0.4)
    output["sa_le_6"] = output["sa_raw"].le(6.0)
    output["logp_in_neg1_to_5"] = output["logp"].between(-1.0, 5.0, inclusive="both")
    output["tpsa_in_20_to_140"] = output["tpsa"].between(20.0, 140.0, inclusive="both")
    output["three_member_rings_eq_0"] = output["three_member_ring_count"].eq(0)
    output["fused_rings_le_4"] = output["fused_ring_count"].le(4)
    output["complex_rings_le_1"] = output["complex_ring_count"].le(1)
    output["brenk_eq_0"] = output["brenk_count"].eq(0)
    chemistry_filter_ids = list(filters)
    output["chemistry_only_actual_good"] = output[chemistry_filter_ids].all(axis=1)
    output["single_fragment"] = output["fragment_count"].eq(1)
    output["allowed_organic_elements"] = output["allowed_organic_elements_only"]
    output["mol_wt_in_120_to_600"] = output["mol_wt"].between(120.0, 600.0, inclusive="both")
    output["heavy_atoms_in_10_to_35"] = output["heavy_atom_count"].between(10, 35, inclusive="both")
    output["hba_le_10"] = output["hba"].le(10)
    output["hbd_le_5"] = output["hbd"].le(5)
    output["formal_charge_abs_le_1"] = output["formal_charge"].abs().le(1)
    output["lipinski_violations_le_1"] = output["lipinski_violation_count"].le(1)
    output["carbon_fraction_le_0_85"] = output["carbon_fraction"].le(0.85)
    filters["chemistry_only_actual_good"] = "All original chemistry gates"
    filters.update(
        {
            "single_fragment": "Single molecular fragment",
            "allowed_organic_elements": "Allowed organic elements only",
            "mol_wt_in_120_to_600": "120 <= MW <= 600",
            "heavy_atoms_in_10_to_35": "10 <= heavy atoms <= 35",
            "hba_le_10": "HBA <= 10",
            "hbd_le_5": "HBD <= 5",
            "formal_charge_abs_le_1": "|Formal charge| <= 1",
            "lipinski_violations_le_1": "Lipinski violations <= 1",
            "carbon_fraction_le_0_85": "Carbon fraction <= 0.85",
        }
    )
    normality_filter_ids = list(filters)
    output["normality_pre_chembl"] = output[normality_filter_ids].all(axis=1)
    filters["normality_pre_chembl"] = "All normality gates before ChEMBL similarity"
    return output, filters


def count_filters(frame: pd.DataFrame, filters: dict[str, str]) -> pd.DataFrame:
    """Report each pass count against both legal and fixed-attempt denominators."""
    rows: list[dict[str, Any]] = [
        {
            "filter_id": "rdkit_valid",
            "condition": "RDKit valid",
            "availability": "computed",
            "pass_count": len(frame),
            "rate_of_rdkit_valid": 1.0,
            "rate_of_100k_attempts": len(frame) / ATTEMPTS,
        }
    ]
    for filter_id, description in filters.items():
        count = int(frame[filter_id].sum())
        rows.append(
            {
                "filter_id": filter_id,
                "condition": description,
                "availability": "computed",
                "pass_count": count,
                "rate_of_rdkit_valid": count / len(frame),
                "rate_of_100k_attempts": count / ATTEMPTS,
            }
        )
    for filter_id, description in (
        ("posebusters_pb_valid", "PoseBusters valid on docked pose"),
        ("docking_le_neg8", "Docking score <= -8.0"),
        ("ligand_efficiency_ge_0_25", "Ligand efficiency >= 0.25"),
        ("formal_actual_good", "All formal actual-good gates"),
    ):
        rows.append(
            {
                "filter_id": filter_id,
                "condition": description,
                "availability": "not_computed_no_full_label_sidecar",
                "pass_count": np.nan,
                "rate_of_rdkit_valid": np.nan,
                "rate_of_100k_attempts": np.nan,
            }
        )
    return pd.DataFrame(rows)


def configure_plotting() -> None:
    """Configure readable panel typography."""
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "svg.fonttype": "none",
        }
    )


def plot_filter_panels(chembl: pd.DataFrame, frame: pd.DataFrame, filters: dict[str, str], output_path: Path) -> None:
    """Plot every available gate in the same fixed UMAP coordinate system."""
    n_columns = 3
    n_rows = math.ceil(len(filters) / n_columns)
    figure, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4.25 * n_rows + 1.2), dpi=200, sharex=True, sharey=True)
    for axis, (filter_id, description) in zip(axes.flat, filters.items()):
        selected = frame[frame[filter_id]]
        axis.scatter(chembl["umap1"], chembl["umap2"], s=0.35, c=CHEMBL_COLOR, alpha=0.11, linewidths=0)
        axis.scatter(frame["umap1"], frame["umap2"], s=0.45, c=ALL_GENERATED_COLOR, alpha=0.06, linewidths=0)
        axis.scatter(selected["umap1"], selected["umap2"], s=1.2, c=PASSED_COLOR, alpha=0.32, linewidths=0)
        axis.set_title(f"{description}\npass: {len(selected):,} / 100,000 attempts")
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
    for axis in axes.flat[len(filters):]:
        axis.set_visible(False)
    figure.suptitle("DiffSBDD 100k: distributions after individual chemistry filters", fontsize=18)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> int:
    """Compute all available filters, tables, and fixed-reference UMAP panels."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_manifest = pd.read_csv(args.all_manifest, low_memory=False)
    projected = pd.read_csv(args.projected_coordinates, low_memory=False)
    reference = pd.read_csv(args.reference_coordinates, low_memory=False)
    valid = all_manifest[all_manifest["rdkit_valid"].eq(1)].copy()
    if len(all_manifest) != ATTEMPTS or len(valid) != 67_694:
        raise ValueError("unexpected frozen 100k manifest counts")
    frame = valid.merge(
        projected[["trajectory_id", "umap1", "umap2"]],
        on="trajectory_id",
        how="inner",
        validate="one_to_one",
    )
    if len(frame) != len(valid):
        raise ValueError("not every valid trajectory has a fixed-reference UMAP coordinate")
    descriptors = calculate_descriptors(frame, args.workers)
    frame = frame.merge(descriptors, on="trajectory_id", how="inner", validate="one_to_one")
    frame, filters = add_pass_columns(frame)
    counts = count_filters(frame, filters)
    chembl = reference[reference["model"].eq("ChEMBL36")].copy()
    if len(chembl) != 100_000:
        raise ValueError("fixed reference must contain ChEMBL100k")
    configure_plotting()
    plot_filter_panels(chembl, frame, filters, args.output_dir / "individual_chemistry_filter_umap_panels.png")
    frame.to_csv(args.output_dir / "valid_trajectory_screening_facts.csv.gz", index=False, compression="gzip")
    counts.to_csv(args.output_dir / "filter_counts.csv", index=False)
    summary = {
        "status": "completed",
        "attempts": ATTEMPTS,
        "rdkit_valid": int(len(frame)),
        "rdkit_invalid": ATTEMPTS - int(len(frame)),
        "coordinate_system": "fixed old five-model ECFP4 UMAP reference",
        "screening_scope": "available chemistry gates from frozen actual-good policy",
        "formal_actual_good": "not computed: full docking, ligand-efficiency, and docked-pose PoseBusters sidecars are absent",
        "workers": args.workers,
        "computed_filters": filters,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
