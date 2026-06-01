from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

from rdkit import Chem


def _ligand_center(sdf_path: Path) -> list[float]:
    supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
    mol = supplier[0]
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Could not read ligand coordinates from {sdf_path}")
    conf = mol.GetConformer()
    return [
        sum(conf.GetAtomPosition(i)[axis] for i in range(mol.GetNumAtoms()))
        / mol.GetNumAtoms()
        for axis in range(3)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()

    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    processed_dir = run_dir / "processed"
    keys_dir = run_dir / "keys"
    processed_dir.mkdir(parents=True, exist_ok=True)
    keys_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo / "data"))
    from processor_no_ligand import PocketDataProcessor

    protein = repo / "example" / "2f0z_protein.pdb"
    ligand = repo / "example" / "2f0z_ligand.sdf"
    center = _ligand_center(ligand)

    processor = PocketDataProcessor(data_dir=str(repo / "example"), save_dir=str(processed_dir))
    data = processor._processor(str(protein), center)
    if data is None:
        raise RuntimeError("DeepICL pocket-only processor returned None")

    key = "2f0z_protein"
    with (processed_dir / key).open("wb") as handle:
        pickle.dump(data, handle)
    with (keys_dir / "test_keys.pkl").open("wb") as handle:
        pickle.dump([key], handle)

    print(f"processed={processed_dir / key}")
    print(f"keys={keys_dir / 'test_keys.pkl'}")
    print(f"center={center}")
    print(f"pocket_atoms={data[3].shape[0]}")


if __name__ == "__main__":
    main()
