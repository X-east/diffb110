"""Convert a 2D MOL file to a 3D SDF file using RDKit.

Generates a 3D conformer via ETKDG embedding followed by MMFF94 optimization.

Usage:
    conda run -n dif python scripts/molecule/mol_to_sdf.py \\
        --input data/Honokiol.mol \\
        --output Honokiol_3d.sdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def mol_to_sdf(input_path: Path, output_path: Path, seed: int = 42) -> dict:
    """Convert a 2D MOL file to a 3D SDF.

    Args:
        input_path:  Path to the input .mol file.
        output_path: Path for the output .sdf file.
        seed:        Random seed for conformer embedding.

    Returns:
        dict with ``num_atoms``, ``num_bonds``, ``status``.
    """
    mol = Chem.MolFromMolFile(str(input_path), removeHs=False)
    if mol is None:
        raise ValueError(f"RDKit could not read {input_path}")

    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, randomSeed=seed, useRandomCoords=False)
    if result < 0:
        raise RuntimeError(f"EmbedMolecule failed for {input_path} (return code {result})")

    mmff_result = AllChem.MMFFOptimizeMolecule(mol)
    if mmff_result < 0:
        raise RuntimeError(f"MMFFOptimizeMolecule failed for {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Chem.MolToMolFile(mol, str(output_path))

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "num_atoms": mol.GetNumAtoms(),
        "num_bonds": mol.GetNumBonds(),
        "mmff_energy": str(mmff_result),
        "seed": seed,
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert 2D .mol to 3D .sdf via RDKit ETKDG + MMFF94."
    )
    p.add_argument("--input", type=Path, required=True, help="Input .mol file")
    p.add_argument("--output", type=Path, required=True, help="Output .sdf file")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = p.parse_args()

    info = mol_to_sdf(args.input, args.output, args.seed)
    print(
        f"✓ {info['num_atoms']} atoms, {info['num_bonds']} bonds → "
        f"{info['output']}"
    )


if __name__ == "__main__":
    main()
