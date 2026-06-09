"""Create a tiny DiffSBDD-xhe train/val subset for joint sampler smoke tests."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np


SRC_DIR = Path("data/crossdocked_raw/processed_edge_noH_full")
OUT_DIR = Path("analysis/diffsbdd_xhe_joint_20260605_smoke/data")


def write_subset(src_path: Path, dst_path: Path, max_molecules: int) -> None:
    """Write the first molecules from one processed DiffSBDD npz file."""
    with np.load(src_path, allow_pickle=True) as data:
        n_molecules = min(int(max_molecules), len(data["names"]))
        ligand_mask = data["lig_mask"] < n_molecules
        pocket_mask = data["pocket_mask"] < n_molecules
        payload = {
            "names": data["names"][:n_molecules],
            "lig_coords": data["lig_coords"][ligand_mask],
            "lig_one_hot": _drop_empty_others_column(data["lig_one_hot"][ligand_mask]),
            "lig_mask": data["lig_mask"][ligand_mask],
            "pocket_coords": data["pocket_coords"][pocket_mask],
            "pocket_one_hot": _drop_empty_others_column(data["pocket_one_hot"][pocket_mask]),
            "pocket_mask": data["pocket_mask"][pocket_mask],
        }
        if "receptors" in data.files:
            payload["receptors"] = data["receptors"][:n_molecules]
        if "lig_bond_index" in data.files and "lig_bond_mask" in data.files:
            bond_mask = data["lig_bond_mask"] < n_molecules
            payload["lig_bond_index"] = data["lig_bond_index"][:, bond_mask]
            payload["lig_bond_type"] = data["lig_bond_type"][bond_mask]
            payload["lig_bond_mask"] = data["lig_bond_mask"][bond_mask]
            payload["num_lig_bonds"] = data["num_lig_bonds"][:n_molecules]
        np.savez(dst_path, **payload)
        print(
            dst_path,
            "molecules=", n_molecules,
            "lig_atoms=", int(ligand_mask.sum()),
            "pocket_nodes=", int(pocket_mask.sum()),
            "directed_bonds=", int(payload.get("lig_bond_type", np.array([])).shape[0]),
        )


def _drop_empty_others_column(one_hot: np.ndarray) -> np.ndarray:
    """Drop the CrossDocked 'others' column when it is empty."""
    if one_hot.shape[1] == 11 and float(one_hot[:, 10].sum()) == 0.0:
        return one_hot[:, :10]
    return one_hot


def main() -> None:
    """Create the tiny smoke dataset."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_subset(SRC_DIR / "train.npz", OUT_DIR / "train.npz", 32)
    write_subset(SRC_DIR / "val.npz", OUT_DIR / "val.npz", 16)
    shutil.copy2(SRC_DIR / "size_distribution.npy", OUT_DIR / "size_distribution.npy")


if __name__ == "__main__":
    main()
