"""Prepare shared 8GJL/C1101 inputs for the model benchmark."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem


STANDARD_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
}


@dataclass(frozen=True)
class PdbAtom:
    """Minimal PDB atom record needed for pocket extraction."""

    line: str
    serial: int
    record: str
    atom_name: str
    resname: str
    chain: str
    resseq: int
    icode: str
    element: str
    xyz: tuple[float, float, float]

    @property
    def residue_key(self) -> tuple[str, int, str, str]:
        """Return a stable residue key."""

        return (self.chain, self.resseq, self.icode, self.resname)


def _parse_atom_line(line: str) -> PdbAtom | None:
    """Parse an ATOM/HETATM line without requiring Biopython."""

    record = line[:6].strip()
    if record not in {"ATOM", "HETATM"}:
        return None
    try:
        serial = int(line[6:11])
        atom_name = line[12:16].strip()
        resname = line[17:20].strip()
        chain = line[21].strip() or "_"
        resseq = int(line[22:26])
        icode = line[26].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None
    element = line[76:78].strip()
    if not element:
        element = "".join(ch for ch in atom_name if ch.isalpha())[:2].title()
    return PdbAtom(
        line=line.rstrip("\n"),
        serial=serial,
        record=record,
        atom_name=atom_name,
        resname=resname,
        chain=chain,
        resseq=resseq,
        icode=icode,
        element=element.strip().title(),
        xyz=(x, y, z),
    )


def _dist(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _read_atoms(path: Path) -> list[PdbAtom]:
    atoms: list[PdbAtom] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        atom = _parse_atom_line(line)
        if atom is not None:
            atoms.append(atom)
    return atoms


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_ligand_pdb(source_pdb: Path, ligand_atoms: list[PdbAtom], out_pdb: Path) -> None:
    ligand_serials = {atom.serial for atom in ligand_atoms}
    lines: list[str] = [atom.line for atom in ligand_atoms]
    for line in source_pdb.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("CONECT"):
            continue
        serial_tokens = [line[i : i + 5].strip() for i in range(6, len(line), 5)]
        serials = [int(tok) for tok in serial_tokens if tok.isdigit()]
        if serials and serials[0] in ligand_serials:
            kept = [s for s in serials if s in ligand_serials]
            if len(kept) > 1:
                lines.append("CONECT" + "".join(f"{s:5d}" for s in kept))
    lines.append("END")
    _write_text(out_pdb, "\n".join(lines) + "\n")


def _convert_ligand_to_sdf(ligand_pdb: Path, out_sdf: Path) -> dict[str, object]:
    out_sdf.parent.mkdir(parents=True, exist_ok=True)
    command = ["obabel", "-ipdb", str(ligand_pdb), "-osdf", "-O", str(out_sdf)]
    result = subprocess.run(command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            "OpenBabel failed while converting C1101 PDB to SDF:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
    supplier = Chem.SDMolSupplier(str(out_sdf), sanitize=False, removeHs=False)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        raise RuntimeError(f"RDKit could not read generated ligand SDF: {out_sdf}")
    return {
        "ligand_sdf_atoms": mol.GetNumAtoms(),
        "ligand_sdf_bonds": mol.GetNumBonds(),
    }


def prepare_inputs(source_pdb: Path, out_dir: Path, ligand_chain: str, ligand_resseq: int, cutoff: float) -> dict[str, object]:
    """Create protein, pocket, ligand, and DiffSBDD-style residue-list inputs."""

    atoms = _read_atoms(source_pdb)
    ligand_atoms = [
        atom
        for atom in atoms
        if atom.record == "HETATM"
        and atom.chain == ligand_chain
        and atom.resseq == ligand_resseq
        and atom.element.upper() != "H"
    ]
    if not ligand_atoms:
        raise ValueError(f"No ligand atoms found for {ligand_chain}:{ligand_resseq} in {source_pdb}")
    ligand_coords = [atom.xyz for atom in ligand_atoms]

    protein_atoms = [
        atom
        for atom in atoms
        if atom.record == "ATOM"
        and atom.resname in STANDARD_RESNAMES
        and atom.element.upper() != "H"
    ]
    residue_to_atoms: dict[tuple[str, int, str, str], list[PdbAtom]] = {}
    for atom in protein_atoms:
        residue_to_atoms.setdefault(atom.residue_key, []).append(atom)

    selected_keys: list[tuple[str, int, str, str]] = []
    for key, res_atoms in residue_to_atoms.items():
        min_dist = min(_dist(atom.xyz, lig_xyz) for atom in res_atoms for lig_xyz in ligand_coords)
        if min_dist <= cutoff:
            selected_keys.append(key)
    selected_keys = sorted(selected_keys, key=lambda item: (item[0], item[1], item[2], item[3]))
    selected_key_set = set(selected_keys)

    out_dir.mkdir(parents=True, exist_ok=True)
    protein_only_pdb = out_dir / "8GJL_C1101_protein_only.pdb"
    pocket_pdb = out_dir / "8GJL_C1101_pocket8A.pdb"
    ligand_pdb = out_dir / "8GJL_C1101_ligand.pdb"
    ligand_sdf = out_dir / "8GJL_C1101_ligand.sdf"
    residue_txt = out_dir / "8GJL_C1101_pocket_residues_8A.txt"
    deppa_dir = out_dir / "deppa_test_pocket"

    _write_text(protein_only_pdb, "\n".join(atom.line for atom in protein_atoms) + "\nEND\n")
    _write_text(pocket_pdb, "\n".join(atom.line for atom in protein_atoms if atom.residue_key in selected_key_set) + "\nEND\n")
    residue_ids = [f"{chain}:{resseq}" for chain, resseq, _icode, _resname in selected_keys]
    _write_text(residue_txt, " ".join(residue_ids) + "\n")
    _make_ligand_pdb(source_pdb, ligand_atoms, ligand_pdb)
    ligand_summary = _convert_ligand_to_sdf(ligand_pdb, ligand_sdf)

    deppa_dir.mkdir(parents=True, exist_ok=True)
    _write_text(deppa_dir / "8GJL.pdb", protein_only_pdb.read_text(encoding="utf-8"))
    _write_text(deppa_dir / "8GJL_1.txt", " ".join(residue_ids) + "\n")
    _write_text(deppa_dir / "8GJL_1.sdf", ligand_sdf.read_text(encoding="utf-8"))

    summary: dict[str, object] = {
        "source_pdb": str(source_pdb),
        "ligand": f"{ligand_chain}:{ligand_resseq}",
        "cutoff": cutoff,
        "protein_heavy_atoms": len(protein_atoms),
        "ligand_heavy_atoms": len(ligand_atoms),
        "pocket_residue_count": len(selected_keys),
        "pocket_atom_count": sum(1 for atom in protein_atoms if atom.residue_key in selected_key_set),
        "pocket_residues": residue_ids,
        "protein_only_pdb": str(protein_only_pdb),
        "pocket_pdb": str(pocket_pdb),
        "ligand_pdb": str(ligand_pdb),
        "ligand_sdf": str(ligand_sdf),
        "residue_txt": str(residue_txt),
        "deppa_test_dir": str(deppa_dir),
        **ligand_summary,
    }
    _write_text(out_dir / "input_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pdb", type=Path, default=Path("data/8GJL.pdb"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ligand-chain", default="C")
    parser.add_argument("--ligand-resseq", type=int, default=1101)
    parser.add_argument("--cutoff", type=float, default=8.0)
    args = parser.parse_args()
    summary = prepare_inputs(
        source_pdb=args.source_pdb.resolve(),
        out_dir=args.out_dir.resolve(),
        ligand_chain=args.ligand_chain,
        ligand_resseq=args.ligand_resseq,
        cutoff=args.cutoff,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
