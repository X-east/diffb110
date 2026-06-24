"""Prepare receptor PDB from a protein-ligand complex PDB for UniDock docking.

Extracts:
  - protein_only.pdb  (ATOM lines only, standard residues, no H)
  - ligand_center.json (geometric center of the reference ligand)

Usage:
    conda run -n dif python scripts/docking/prepare_receptor.py \\
        --source-pdb data/4dxd_9pc.pdb \\
        --out-dir analysis/honokiol_9pc_docking_20260617 \\
        --ligand-chain A --ligand-resseq 402

The output ligand_center.json can be fed to dock_sdf_unidock.py via:
    --center-json analysis/.../ligand_center.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Standard protein residue names (exclude HETATM cofactors/waters)
STANDARD_RESNAMES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
    "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
}


def _parse_atom_line(line: str) -> dict | None:
    record = line[:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None
    try:
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
        atom_name = line[12:16].strip()
        element = "".join(ch for ch in atom_name if ch.isalpha())[:2].title()
    return {
        "line": line.rstrip("\n"),
        "record": record,
        "resname": resname,
        "chain": chain,
        "resseq": resseq,
        "element": element.strip().title(),
        "xyz": (x, y, z),
    }


def prepare_receptor(
    source_pdb: Path,
    out_dir: Path,
    ligand_chain: str,
    ligand_resseq: int,
) -> dict:
    """Extract protein-only PDB and ligand geometric center.

    Args:
        source_pdb: Path to the protein-ligand complex PDB file.
        out_dir: Output directory.
        ligand_chain: Chain ID of the reference ligand (e.g. "A").
        ligand_resseq: Residue sequence number of the reference ligand (e.g. 402).

    Returns:
        dict with keys: protein_pdb, center_x, center_y, center_z,
                        protein_heavy_atoms, ligand_heavy_atoms, etc.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    atoms: list[dict] = []
    for line in source_pdb.read_text(errors="ignore").splitlines():
        a = _parse_atom_line(line)
        if a:
            atoms.append(a)

    # --- Ligand: HETATM with matching chain/resseq, non-H ---
    ligand_atoms = [
        a for a in atoms
        if a["record"] == "HETATM"
        and a["chain"] == ligand_chain
        and a["resseq"] == ligand_resseq
        and a["element"].upper() != "H"
    ]
    if not ligand_atoms:
        raise ValueError(
            f"No ligand heavy atoms found for chain={ligand_chain} resseq={ligand_resseq} "
            f"in {source_pdb}"
        )

    lig_coords = [a["xyz"] for a in ligand_atoms]
    center_x = sum(c[0] for c in lig_coords) / len(lig_coords)
    center_y = sum(c[1] for c in lig_coords) / len(lig_coords)
    center_z = sum(c[2] for c in lig_coords) / len(lig_coords)

    # --- Protein: ATOM with standard resnames, non-H ---
    protein_atoms = [
        a for a in atoms
        if a["record"] == "ATOM"
        and a["resname"] in STANDARD_RESNAMES
        and a["element"].upper() != "H"
    ]

    # Write protein-only PDB
    protein_pdb = out_dir / "protein_only.pdb"
    protein_pdb.write_text(
        "\n".join(a["line"] for a in protein_atoms) + "\nEND\n"
    )

    # Write ligand center JSON
    center_info = {
        "source_pdb": str(source_pdb.resolve()),
        "ligand": f"{ligand_chain}:{ligand_resseq}",
        "center_x": round(center_x, 3),
        "center_y": round(center_y, 3),
        "center_z": round(center_z, 3),
        "protein_heavy_atoms": len(protein_atoms),
        "ligand_heavy_atoms": len(ligand_atoms),
        "protein_pdb": str(protein_pdb.resolve()),
    }
    center_json = out_dir / "ligand_center.json"
    center_json.write_text(json.dumps(center_info, indent=2, sort_keys=True) + "\n")

    return center_info


def main() -> None:
    p = argparse.ArgumentParser(
        description="Extract protein-only PDB and ligand center from a complex PDB."
    )
    p.add_argument("--source-pdb", type=Path, required=True,
                   help="Protein-ligand complex PDB file")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory")
    p.add_argument("--ligand-chain", default="A",
                   help="Chain ID of the reference ligand (default: A)")
    p.add_argument("--ligand-resseq", type=int, required=True,
                   help="Residue sequence number of the reference ligand")
    args = p.parse_args()

    info = prepare_receptor(
        source_pdb=args.source_pdb.resolve(),
        out_dir=args.out_dir.resolve(),
        ligand_chain=args.ligand_chain,
        ligand_resseq=args.ligand_resseq,
    )
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
