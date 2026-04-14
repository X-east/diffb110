#!/usr/bin/env python3
"""
Analyze an SDF file with RDKit and export selected molecular properties to CSV.

Edit the configuration block below directly, then run:
    python analyze_valid_sdf_properties.py

Optional CLI overrides:
    python analyze_valid_sdf_properties.py --sdf /path/to/file.sdf
    python analyze_valid_sdf_properties.py --sdf /path/to/file.sdf --max_successful 500
    python analyze_valid_sdf_properties.py --sdf /path/to/file.sdf --max_successful -1
    python analyze_valid_sdf_properties.py --sdf /path/to/file.sdf --out /path/to/file.csv
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ===================== user controls =====================
SDF_FILE = Path("/home/md666/data/disk6/DiffB110/OUT/260407_generate_ligands_10000/260407generated_10000_filtered.sdf")
OUTPUT_CSV = None  # Set to a Path to force a custom output name; None uses auto naming
MAX_SUCCESSFUL_MOLECULES = None  # Set to None to calculate all molecules
# =========================================================

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIFFSBDD_ROOT = PROJECT_ROOT / "DiffSBDD"
SASCORER_PATH = DIFFSBDD_ROOT / "analysis" / "SA_Score" / "sascorer.py"
sys.path.insert(0, str(PROJECT_ROOT))

if not SASCORER_PATH.exists():
    raise FileNotFoundError(f"SA scorer not found: {SASCORER_PATH}")

spec = importlib.util.spec_from_file_location("diffsbdd_sascorer", SASCORER_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Unable to load SA scorer module from: {SASCORER_PATH}")

sascorer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sascorer_module)
calculateScore = sascorer_module.calculateScore

OUTPUT_COLUMNS = [
    "MolWt",
    "ExactMolWt",
    "LogP",
    "TPSA",
    "HBA",
    "HBD",
    "RotBonds",
    "HeavyAtomCount",
    "NumRings",
    "NumAromaticRings",
    "FractionCSP3",
    "FormalCharge",
    "QED",
    "SA_Score",
    "Lipinski_Pass",
    "Murcko_Scaffold",
]


def compute_lipinski_pass(mol: Chem.Mol) -> bool:
    logp = Crippen.MolLogP(mol)
    return all(
        [
            Descriptors.ExactMolWt(mol) < 500,
            Lipinski.NumHDonors(mol) <= 5,
            Lipinski.NumHAcceptors(mol) <= 10,
            -2 <= logp <= 5,
            rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10,
        ]
    )


def calculate_record(mol: Chem.Mol) -> dict:
    Chem.SanitizeMol(mol)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

    return {
        "MolWt": round(Descriptors.MolWt(mol), 4),
        "ExactMolWt": round(Descriptors.ExactMolWt(mol), 4),
        "LogP": round(Crippen.MolLogP(mol), 4),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 4),
        "HBA": int(Lipinski.NumHAcceptors(mol)),
        "HBD": int(Lipinski.NumHDonors(mol)),
        "RotBonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
        "NumRings": int(rdMolDescriptors.CalcNumRings(mol)),
        "NumAromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FractionCSP3": round(rdMolDescriptors.CalcFractionCSP3(mol), 4),
        "FormalCharge": int(Chem.GetFormalCharge(mol)),
        "QED": round(QED.qed(mol), 4),
        "SA_Score": round(calculateScore(mol), 4),
        "Lipinski_Pass": int(compute_lipinski_pass(mol)),
        "Murcko_Scaffold": scaffold,
    }


def analyze_sdf(sdf_path: Path, max_successful: int | None) -> pd.DataFrame:
    supplier = Chem.SDMolSupplier(str(sdf_path), sanitize=False, removeHs=False)

    records = []
    failed_ids = []

    for mol_id, mol in enumerate(supplier, start=1):
        if mol is None:
            failed_ids.append(mol_id)
            continue

        try:
            records.append(calculate_record(mol))
            if max_successful is not None and len(records) >= max_successful:
                break
        except Exception:
            failed_ids.append(mol_id)

    df = pd.DataFrame(records, columns=OUTPUT_COLUMNS)

    target_msg = max_successful if max_successful is not None else "All"
    print(f"Loaded molecules: {len(supplier)}")
    print(f"Successfully analyzed and kept: {len(df)}")
    print(f"Target successful molecules: {target_msg}")
    print(f"Failed molecules encountered before cutoff: {len(failed_ids)}")

    if max_successful is not None and len(df) < max_successful:
        print(f"Warning: only {len(df)} successful molecules were found.")

    return df


def build_output_suffix(max_successful: int | None) -> str:
    if max_successful is None:
        return "all"
    return f"top{max_successful}"


def default_output_path(sdf_path: Path, max_successful: int | None) -> Path:
    suffix = build_output_suffix(max_successful)
    return sdf_path.with_name(f"{sdf_path.stem}_properties_{suffix}.csv")


def resolve_paths() -> tuple[Path, Path, int | None]:
    parser = argparse.ArgumentParser(
        description="Analyze SDF molecular properties with RDKit"
    )
    parser.add_argument("--sdf", type=Path, default=None, help="Input SDF file path")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV file path (optional; default name follows input file and molecule count)",
    )
    parser.add_argument(
        "--max_successful",
        type=int,
        default=None,
        help="Keep the first N successfully analyzed molecules (set to -1 for all)",
    )
    args = parser.parse_args()

    sdf_path = args.sdf if args.sdf is not None else SDF_FILE

    max_successful = (
        args.max_successful
        if args.max_successful is not None
        else MAX_SUCCESSFUL_MOLECULES
    )
    if max_successful == -1:
        max_successful = None

    if sdf_path is None:
        raise ValueError("Please set SDF_FILE at the top of the script or pass --sdf")

    if max_successful is not None and max_successful <= 0:
        raise ValueError("max_successful must be > 0 (or -1 for all)")

    if args.out is not None:
        out_path = args.out
    elif OUTPUT_CSV is not None:
        out_path = Path(OUTPUT_CSV)
    else:
        out_path = default_output_path(Path(sdf_path), max_successful)

    return Path(sdf_path), Path(out_path), max_successful


def main() -> None:
    sdf_path, out_path, max_successful = resolve_paths()

    if not sdf_path.exists():
        raise FileNotFoundError(f"SDF file not found: {sdf_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Input SDF       : {sdf_path}")
    print(f"Output CSV      : {out_path}")
    print(f"Max successful  : {max_successful}")
    print("=" * 70)

    df = analyze_sdf(sdf_path, max_successful)
    df.to_csv(out_path, index=False)

    print(f"Saved CSV: {out_path}")


if __name__ == "__main__":
    main()

