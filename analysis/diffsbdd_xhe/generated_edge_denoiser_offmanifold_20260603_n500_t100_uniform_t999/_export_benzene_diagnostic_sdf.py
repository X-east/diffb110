from __future__ import annotations

import csv
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem


def main() -> None:
    outdir = Path(__file__).resolve().parent
    input_csv = outdir / "aromatic_benzene_examples.csv"
    output_sdf = outdir / "aromatic_benzene_diagnostic_examples.sdf"

    with input_csv.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    writer = Chem.SDWriter(str(output_sdf))
    written = 0
    skipped = 0

    for row in rows:
        mol = Chem.MolFromSmiles(row["smiles"])
        if mol is None:
            skipped += 1
            continue
        AllChem.Compute2DCoords(mol)
        title = f"{row['tag']}|{row['method']}|sample_{row['sample_idx']}"
        mol.SetProp("_Name", title)
        for key, value in row.items():
            mol.SetProp(key, value)
        writer.write(mol)
        written += 1

    writer.close()
    print(f"wrote={written} skipped={skipped} output={output_sdf}")


if __name__ == "__main__":
    main()
