"""Extract complete protein residues within a cutoff of an SDF ligand."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from rdkit import Chem


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """Return Euclidean distance between two 3D points."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def main() -> None:
    """Write all standard protein residues close to the reference ligand."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--protein", required=True, type=Path)
    parser.add_argument("--ligand", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=float, default=10.0)
    args = parser.parse_args()

    mol = next((m for m in Chem.SDMolSupplier(str(args.ligand), removeHs=False) if m), None)
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Cannot read a 3D ligand from {args.ligand}")
    ligand_positions = [tuple(map(float, p)) for p in mol.GetConformer().GetPositions()]

    residue_lines: dict[tuple[str, str, str], list[str]] = {}
    residue_positions: dict[tuple[str, str, str], list[tuple[float, float, float]]] = {}
    for line in args.protein.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.startswith("ATOM"):
            continue
        try:
            key = (line[21:22], line[22:26], line[26:27])
            position = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:
            continue
        residue_lines.setdefault(key, []).append(line)
        residue_positions.setdefault(key, []).append(position)

    selected = {
        key
        for key, positions in residue_positions.items()
        if min(_distance(position, ligand) for position in positions for ligand in ligand_positions)
        <= args.cutoff
    }
    output_lines = [line for key in sorted(selected) for line in residue_lines[key]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + "\nEND\n", encoding="utf-8")
    print(f"residues={len(selected)} atoms={len(output_lines)} output={args.output}")


if __name__ == "__main__":
    main()
