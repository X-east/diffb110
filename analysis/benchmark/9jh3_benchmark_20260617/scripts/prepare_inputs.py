"""Prepare 9JH3/A1D5R inputs for DiffSBDD + DiffGui generation."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from rdkit import Chem

STANDARD_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _parse_atom_line(line: str) -> dict | None:
    record = line[:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None
    try:
        serial = int(line[6:11])
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        resseq = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    element = line[76:78].strip()
    if not element:
        element = "".join(ch for ch in atom_name if ch.isalpha())[:2].title()
    return {
        "line": line.rstrip("\n"),
        "serial": serial,
        "record": record,
        "atom_name": atom_name,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "element": element.strip().title(),
        "xyz": (x, y, z),
        "residue_key": (chain, resseq, resname),
    }


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def prepare(source_pdb: Path, out_dir: Path, lig_chain: str, lig_resseq: int, cutoff: float = 8.0):
    atoms = []
    for line in source_pdb.read_text(errors="ignore").splitlines():
        a = _parse_atom_line(line)
        if a:
            atoms.append(a)

    # Ligand: HETATM with matching chain and resseq, non-H
    ligand_atoms = [
        a for a in atoms
        if a["record"] == "HETATM" and a["chain"] == lig_chain
        and a["resseq"] == lig_resseq and a["element"].upper() != "H"
    ]
    if not ligand_atoms:
        raise ValueError(f"No ligand atoms for chain={lig_chain} resseq={lig_resseq}")

    lig_coords = [a["xyz"] for a in ligand_atoms]

    # Protein: ATOM with standard resnames, non-H
    protein_atoms = [
        a for a in atoms
        if a["record"] == "ATOM" and a["resname"] in STANDARD_RESNAMES
        and a["element"].upper() != "H"
    ]

    # Group protein by residue
    residue_map = {}
    for a in protein_atoms:
        residue_map.setdefault(a["residue_key"], []).append(a)

    # Select pocket residues within cutoff of ligand
    selected_keys = []
    for key, res_atoms in residue_map.items():
        min_d = min(_dist(a["xyz"], lc) for a in res_atoms for lc in lig_coords)
        if min_d <= cutoff:
            selected_keys.append(key)
    selected_keys.sort()
    selected_set = set(selected_keys)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Write protein-only PDB
    (out_dir / "9JH3_protein_only.pdb").write_text(
        "\n".join(a["line"] for a in protein_atoms) + "\nEND\n"
    )

    # Write pocket PDB
    (out_dir / "9JH3_pocket8A.pdb").write_text(
        "\n".join(a["line"] for a in protein_atoms if a["residue_key"] in selected_set) + "\nEND\n"
    )

    # Write residue list
    residue_ids = [f"{c}:{r}" for c, r, _rn in selected_keys]
    (out_dir / "9JH3_pocket_residues_8A.txt").write_text(" ".join(residue_ids) + "\n")

    # Write ligand PDB
    (out_dir / "9JH3_ligand.pdb").write_text(
        "\n".join(a["line"] for a in ligand_atoms) + "\nEND\n"
    )

    # Convert ligand to SDF via OpenBabel
    lig_pdb = out_dir / "9JH3_ligand.pdb"
    lig_sdf = out_dir / "9JH3_ligand.sdf"
    result = subprocess.run(
        ["obabel", "-ipdb", str(lig_pdb), "-osdf", "-O", str(lig_sdf)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"obabel failed: {result.stderr}")
    mol = next((m for m in Chem.SDMolSupplier(str(lig_sdf), sanitize=False) if m is not None), None)
    if mol is None:
        raise RuntimeError(f"RDKit could not read {lig_sdf}")

    summary = {
        "source_pdb": str(source_pdb),
        "ligand": f"{lig_chain}:{lig_resseq}",
        "cutoff_A": cutoff,
        "protein_heavy_atoms": len(protein_atoms),
        "ligand_heavy_atoms": len(ligand_atoms),
        "pocket_residue_count": len(selected_keys),
        "pocket_atom_count": sum(1 for a in protein_atoms if a["residue_key"] in selected_set),
        "pocket_pdb": str(out_dir / "9JH3_pocket8A.pdb"),
        "ligand_sdf": str(lig_sdf),
        "ligand_atoms": mol.GetNumAtoms(),
        "ligand_bonds": mol.GetNumBonds(),
        "residue_list": residue_ids,
    }
    (out_dir / "input_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-pdb", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--ligand-chain", default="R")
    p.add_argument("--ligand-resseq", type=int, default=401)
    p.add_argument("--cutoff", type=float, default=8.0)
    args = p.parse_args()
    summary = prepare(args.source_pdb.resolve(), args.out_dir.resolve(),
                      args.ligand_chain, args.ligand_resseq, args.cutoff)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
