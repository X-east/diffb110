from __future__ import annotations

import argparse
from pathlib import Path

import torch
from rdkit import Chem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-pt", required=True)
    parser.add_argument("--out-sdf", required=True)
    args = parser.parse_args()

    records = torch.load(args.result_pt, map_location="cpu")
    out_sdf = Path(args.out_sdf)
    out_sdf.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with Chem.SDWriter(str(out_sdf)) as writer:
        for idx, record in enumerate(records):
            mol = record.get("mol")
            if mol is None:
                continue
            mol.SetProp("_Name", f"decomp_{idx:04d}")
            smiles = record.get("smiles", "")
            if smiles:
                mol.SetProp("smiles", smiles)
            ligand_filename = record.get("ligand_filename", "")
            if ligand_filename:
                mol.SetProp("ligand_filename", ligand_filename)
            writer.write(mol)
            written += 1

    print(f"written={written} out_sdf={out_sdf}")


if __name__ == "__main__":
    main()
